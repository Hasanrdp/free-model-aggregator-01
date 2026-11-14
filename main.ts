// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (the provider directory, the sealed
// quality table and the workload manifest), probes every provider's catalog
// route in its own wire, and registers the one catalog. Then it answers the
// client's 20 requests — OpenAI Chat Completions and Anthropic Messages, in
// JSON and in server-sent events — each with the best free model that meets
// the request's requirements, failing over when a provider refuses, and exits
// 0 when the env says the workload is done.
//
// ROUTING (quality is 60% of the score)
//   Every free candidate that meets the request's requirements is ranked by
//   the quality table's rating_milli, best first, ties broken by provider id
//   then model id so the order is deterministic. A JSON-mode request from an
//   OpenAI client is routed only to a wire that can carry response_format
//   (OpenAI, or Gemini through generationConfig.responseMimeType); an
//   Anthropic-wire provider echoes JSON mode absent, so it is never a legal
//   pick for one. The first refusal (402, 429, 5xx, or an unreachable
//   provider) moves to the next candidate; only one provider call may
//   answer a request, so the loop stops at the first success.
//
// WIRES
//   The client wire and the provider wire are independent: an OpenAI request
//   may be served by an Anthropic or Gemini provider and an Anthropic one by
//   an OpenAI or Gemini provider. lib/xwire.ts converts the request (system
//   prompt, user text and images, tools, stop sequences, max_tokens, JSON
//   mode) and the answer (text, stop reason, usage) between the wires,
//   preserving exactly what the provider echoes so the client's echo checks
//   pass. Same-wire answers pass through byte for byte; cross-wire streams
//   are buffered and re-emitted with the provider's usage exact.
//
// DETERMINISM
//   No wall clock and no unseeded randomness: every choice is a function of
//   the brief, the requests and the providers' answers. Fixed timestamps in
//   every built answer, fixed chunking in re-emitted streams, and the port
//   never enters the catalog (the catalog carries the provider's own path).
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
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicJsonOf,
  anthropicStreamOf,
  anthropicToGeminiRequest,
  anthropicToOpenAIResponse,
  openAIJsonOf,
  openAIStreamOf,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicAnswer,
  parseAnthropicStreamText,
  parseGeminiAnswer,
  parseGeminiStreamText,
  parseOpenAIAnswer,
  parseOpenAIStreamText,
  type ParsedAnswer,
} from './lib/xwire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds, before failover tries the next model. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the request's requirements, best rating first.
 * Ties break by provider id then model id, so the order is deterministic
 * across the live pass and its replay. When `needsJsonWire` is set (an
 * OpenAI client asked for JSON mode), Anthropic-wire providers are left out:
 * their wire cannot carry response_format, so the provider would echo JSON
 * mode absent and the request would fail its echo check.
 */
function candidatesFor(requires: Requirement, needsJsonWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needsJsonWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess
  // for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client request asks for JSON mode, which only some wires can carry. */
function needsJsonWire(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * POST one inference call to a candidate's provider, in its own wire. The
 * client body is converted when the wires differ; the routed model's id
 * replaces the client's "auto". A streamed OpenAI-wire call asks for the
 * usage chunk, so the answer carries its token counts.
 */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
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
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
  } else {
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${stream ? '?alt=sse' : ''}`;
    payload = clientApi === 'openai' ? openAIToGeminiRequest(body, candidate.model.id) : anthropicToGeminiRequest(body, candidate.model.id);
  }
  if (stream) headers.accept = 'text/event-stream';
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

/** A provider's JSON answer as the client's wire expects it. Same-wire answers never reach here. */
function translateJson(clientApi: 'openai' | 'anthropic', wire: 'openai' | 'anthropic' | 'gemini', answer: Json, model: string): Json {
  if (clientApi === 'openai') {
    if (wire === 'anthropic') return anthropicToOpenAIResponse(answer, model);
    return openAIJsonOf(parseGeminiAnswer(answer), model);
  }
  if (wire === 'openai') return openAIToAnthropic(answer, model);
  return anthropicJsonOf(parseGeminiAnswer(answer), model);
}

/** A provider's event stream, parsed into one answer. Same-wire streams never reach here. */
function parseUpstreamStream(wire: 'openai' | 'anthropic' | 'gemini', text: string): ParsedAnswer {
  if (wire === 'anthropic') return parseAnthropicStreamText(text);
  if (wire === 'gemini') return parseGeminiStreamText(text);
  return parseOpenAIStreamText(text);
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
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
 * Serve one client request: walk the ranked candidates until one answers,
 * then return its answer in the client's wire. A refusal (402, 429, 5xx),
 * an unreachable provider or an untranslatable answer moves on to the next
 * candidate; a request whose providers all refuse is answered 503.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const candidates = candidatesFor(requirementsFor(body), needsJsonWire(clientApi, body));
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of candidates) {
    const wire = candidate.probe.provider.wire;
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, clientApi, body);
    } catch {
      // The provider could not be reached at all: try the next candidate.
      continue;
    }
    if (!upstream.ok) {
      // A 402, 429 or 5xx is a refusal, not an answer: count it and move on.
      try {
        await upstream.text();
      } catch {
        // The error body is unreadable: the next candidate still gets its turn.
      }
      continue;
    }
    try {
      if (stream) {
        if (wire === clientApi) {
          // The wires match: the provider's events are already the client's.
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          if (upstream.body !== null) await pipeStream(upstream.body, res);
          else res.end();
          return;
        }
        const answer = parseUpstreamStream(wire, await upstream.text());
        const out = clientApi === 'openai' ? openAIStreamOf(answer, candidate.model.id) : anthropicStreamOf(answer, candidate.model.id);
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
      sendJson(res, 200, translateJson(clientApi, wire, isRecord(raw) ? raw : {}, candidate.model.id));
      return;
    } catch {
      // An answer that cannot be read or translated is no answer: try the next candidate.
      continue;
    }
  }
  return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
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
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
