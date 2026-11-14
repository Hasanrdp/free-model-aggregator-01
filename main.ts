// free-model-aggregator-01: one server in front of ten free providers.
//
// The server loads the trial brief — the provider directory, the sealed
// quality table and the workload manifest — asks every provider for its
// catalog in that provider's own wire, and registers the one catalog with
// the env. Then it answers the client's twenty requests, OpenAI Chat
// Completions and Anthropic Messages, as JSON or as server-sent events,
// picking for each the highest-rated free model that satisfies what the
// request needs, moving on to the next pick whenever a provider refuses,
// and it exits 0 as soon as the env says the workload is done.
//
// HOW A MODEL IS PICKED (quality is 60% of the score)
//   Every free model that satisfies the request's needs is ranked by the
//   quality table's rating_milli, best first, with ties settled by provider
//   id then model id so the order never depends on timing. A JSON-mode
//   request from an OpenAI client may only go to a wire that can carry
//   response_format (OpenAI, or Gemini through generationConfig); an
//   Anthropic-wire provider echoes JSON mode absent, so it can never serve
//   one. The first refusal — 402, 429, 5xx, or a provider that cannot be
//   reached — moves to the next pick; only a single provider call may
//   answer a request, so the walk stops at the first success.
//
// WIRES
//   The client's wire and the provider's wire are independent: an OpenAI
//   request can be answered by an Anthropic or Gemini provider and an
//   Anthropic one by an OpenAI or Gemini provider. lib/xwire.ts carries the
//   request (system prompt, user text and images, tools, stop sequences,
//   max_tokens, JSON mode) and the answer (text, stop reason, usage) across
//   the wires, keeping exactly what the provider echoes so the client's
//   echo checks pass. Answers on the same wire go through byte for byte;
//   streams across wires are buffered and sent again with the provider's
//   usage exact.
//
// NO CLOCK, NO CHANCE
//   No wall clock and no unseeded randomness: every choice follows from
//   the brief, the requests and the providers' answers. Answers built here
//   carry fixed timestamps, re-sent streams use fixed chunking, and the port
//   never appears in the catalog (the catalog holds the provider's own path).
//
// THE TRIAL
//   `node main.ts --list` prints the catalog document and exits 0. Otherwise
//   the server listens on 0.0.0.0 at a free port, POSTs
//   $MAKEFUN_ENV_URL/register {protocol_version: 1, port, catalog}, serves
//   GET /v1/models (OpenAI shape for Authorization: Bearer, Anthropic shape
//   for x-api-key), POST /v1/chat/completions and POST /v1/messages, polls
//   GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when it says done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders, buildCatalog, capabilitiesOf, isRecord, joinBase,
  meetsRequirements, modelFree, probeProvider, ratingOfModel, readBrief,
  type CatalogReport, type Json, type Probe, type ProbedModel, type Requirement,
} from './lib/providers.ts';
import {
  anthropicError, anthropicModelList, anthropicToOpenAI,
  openAIError, openAIModelList, openAIToAnthropic, type ServedModel,
} from './lib/wire.ts';
import {
  answerAsAnthropic, answerAsAnthropicStream, answerAsOpenAI, answerAsOpenAIStream,
  anthropicAnswerAsOpenAI, anthropicRequestAsGemini, openAIRequestAsAnthropic, openAIRequestAsGemini,
  readAnthropicStream, readGeminiAnswer, readGeminiStream, readOpenAIStream, type Answer,
} from './lib/xwire.ts';

/** The protocol version the register body carries. */
const API_PROTOCOL = 1;
/** How often GET /work is checked, in milliseconds. */
const WORK_CHECK_MS = 100;
/** How long a provider's catalog probe may run, in milliseconds. */
const CATALOG_WAIT_MS = 5000;
/** How long a provider inference call may run, in milliseconds, before the next pick is tried. */
const CALL_WAIT_MS = 10000;
/** The largest client request body this server reads, in bytes. */
const BODY_CAP = 32 * 1024 * 1024;

const ENV_ROOT = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';

const plan = readBrief(BRIEF_PATH);
let scans: Probe[] = [];
let directory: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest runs in arrival order, one request at a time. */
let arrival = 0;

function note(message: string): void {
  process.stderr.write(`entry: ${message}\n`);
}

/** Ask every provider for its catalog and build the one directory (the `--list` document and the register body). */
async function warmUp(): Promise<void> {
  if (plan.error !== null) note(plan.error);
  scans = await Promise.all(plan.directory.providers.map((entry) => probeProvider(ENV_ROOT, entry, CATALOG_WAIT_MS)));
  for (const scan of scans) if (!scan.ok && scan.error !== null) note(scan.error);
  directory = buildCatalog(ENV_ROOT, scans, plan.ratings);
}

const sortById = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function listedFreeModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const scan of scans) {
    if (!scan.ok) continue;
    for (const model of [...scan.models].sort(sortById)) {
      const row = ratingOfModel(plan.ratings, scan.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: scan.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Choice {
  scan: Probe;
  model: ProbedModel;
  score: number;
}

/** Compare two picks: best score first, then provider id, then model id, so the order never depends on timing. */
function compareChoice(a: Choice, b: Choice): number {
  return (
    b.score - a.score ||
    (a.scan.provider.id < b.scan.provider.id ? -1 : a.scan.provider.id > b.scan.provider.id ? 1 : 0) ||
    (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0)
  );
}

/**
 * Every free model that satisfies the request's needs, best score first.
 * When `needsJson` is set (an OpenAI client asked for JSON mode), providers
 * on the Anthropic wire are left out: their wire cannot carry
 * response_format, so the provider would echo JSON mode absent and the
 * request would fail its echo check.
 */
function rankChoices(needs: Requirement, needsJson: boolean): Choice[] {
  const out: Choice[] = [];
  for (const scan of scans) {
    if (!scan.ok) continue;
    if (needsJson && scan.provider.wire === 'anthropic') continue;
    for (const model of scan.models) {
      const row = ratingOfModel(plan.ratings, scan.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(needs, capabilitiesOf(model, row))) continue;
      out.push({ scan, model, score: row?.rating_milli ?? 0 });
    }
  }
  out.sort(compareChoice);
  return out;
}

/** The needs of the request now arriving: the manifest's row by arrival order, else the body's own. */
function needsFor(body: Json): Requirement {
  const manifest = plan.workload?.manifest ?? [];
  const entry = manifest[arrival];
  arrival += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's needs; this fallback is a guess
  // for a brief that holds none.
  return needsFromBody(body);
}

/** Needs read off a request body, when the manifest has no row for it. */
function needsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const vision = messages.some(
    (message) =>
      isRecord(message) &&
      Array.isArray(message.content) &&
      message.content.some((block) => isRecord(block) && (block.type === 'image' || block.type === 'image_url')),
  );
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client request asks for JSON mode, which only some wires can carry. */
function needsJsonCapable(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > BODY_CAP) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function replyJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function replyError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') replyJson(res, status, anthropicError(type, message));
  else replyJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function listModels(req: IncomingMessage, res: ServerResponse): void {
  const list = listedFreeModels();
  if (typeof req.headers['x-api-key'] === 'string') replyJson(res, 200, anthropicModelList(list));
  else replyJson(res, 200, openAIModelList(list));
}

/**
 * POST one inference call to a pick's provider, in its own wire. The
 * client body is converted when the wires differ; the routed model's id
 * replaces the client's "auto". A streamed OpenAI-wire call asks for the
 * usage chunk, so the answer carries its token counts.
 */
async function callUpstream(choice: Choice, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = choice.scan.provider;
  const base = joinBase(ENV_ROOT, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload =
      clientApi === 'openai'
        ? { ...body, model: choice.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, choice.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientApi === 'anthropic' ? { ...body, model: choice.model.id } : openAIRequestAsAnthropic(body, choice.model.id);
  } else {
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(choice.model.id)}:${verb}${stream ? '?alt=sse' : ''}`;
    payload = clientApi === 'openai' ? openAIRequestAsGemini(body, choice.model.id) : anthropicRequestAsGemini(body, choice.model.id);
  }
  if (stream) headers.accept = 'text/event-stream';
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_WAIT_MS) });
}

/** A provider's JSON answer as the client's wire expects it. Same-wire answers never reach here. */
function convertJsonAnswer(clientApi: 'openai' | 'anthropic', wire: 'openai' | 'anthropic' | 'gemini', answer: Json, model: string): Json {
  if (clientApi === 'openai') {
    if (wire === 'anthropic') return anthropicAnswerAsOpenAI(answer, model);
    return answerAsOpenAI(readGeminiAnswer(answer), model);
  }
  if (wire === 'openai') return openAIToAnthropic(answer, model);
  return answerAsAnthropic(readGeminiAnswer(answer), model);
}

/** A provider's event stream, parsed into one answer. Same-wire streams never reach here. */
function readUpstreamEvents(wire: 'openai' | 'anthropic' | 'gemini', text: string): Answer {
  if (wire === 'anthropic') return readAnthropicStream(text);
  if (wire === 'gemini') return readGeminiStream(text);
  return readOpenAIStream(text);
}

/** Copy a provider's event stream to the client as it arrives. */
async function forwardEvents(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * Serve one client request: walk the ranked picks until one answers,
 * then return its answer in the client's wire. A refusal (402, 429, 5xx),
 * a provider that cannot be reached, or an answer that cannot be read or
 * converted moves on to the next pick; a request whose providers all
 * refuse is answered 503.
 */
async function answerRequest(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const picks = rankChoices(needsFor(body), needsJsonCapable(clientApi, body));
  if (picks.length === 0) {
    return replyError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const pick of picks) {
    const wire = pick.scan.provider.wire;
    let upstream: Response;
    try {
      upstream = await callUpstream(pick, clientApi, body);
    } catch {
      // The provider could not be reached at all: try the next pick.
      continue;
    }
    if (!upstream.ok) {
      // A 402, 429 or 5xx is a refusal, not an answer: count it and move on.
      try {
        await upstream.text();
      } catch {
        // The error body is unreadable: the next pick still gets its turn.
      }
      continue;
    }
    try {
      if (stream) {
        if (wire === clientApi) {
          // The wires match: the provider's events are already the client's.
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          if (upstream.body !== null) await forwardEvents(upstream.body, res);
          else res.end();
          return;
        }
        const answer = readUpstreamEvents(wire, await upstream.text());
        const out = clientApi === 'openai' ? answerAsOpenAIStream(answer, pick.model.id) : answerAsAnthropicStream(answer, pick.model.id);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(out);
        return;
      }
      if (wire === clientApi) {
        // The wires match: the provider's JSON is already the client's.
        const text = await upstream.text();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(text);
        return;
      }
      const raw: unknown = await upstream.json();
      replyJson(res, 200, convertJsonAnswer(clientApi, wire, isRecord(raw) ? raw : {}, pick.model.id));
      return;
    } catch {
      // An answer that cannot be read or converted is no answer: try the next pick.
      continue;
    }
  }
  return replyError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return replyError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await answerRequest('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function anthropicMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBody(req);
  if (body === null) return replyError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await answerRequest('anthropic', body, res);
}

async function dispatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    listModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatCompletions(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await anthropicMessages(req, res);
    return;
  }
  replyError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function bindServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function registerWithEnv(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_ROOT}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: API_PROTOCOL, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_ROOT}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') note(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function watchCompletion(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_ROOT}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, WORK_CHECK_MS);
}

async function boot(): Promise<void> {
  await warmUp();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(directory)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_ROOT === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void dispatch(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) replyError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await bindServer(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  note(`listening on 0.0.0.0:${port} with ${directory.providers.length} answered providers and ${listedFreeModels().length} free models`);
  await registerWithEnv(port, directory);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchCompletion(server);
}

try {
  await boot();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}

