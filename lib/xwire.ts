// lib/xwire.ts — carrying requests and answers across the three wires.
//
// The trial's client speaks OpenAI or Anthropic; the ten providers speak
// OpenAI, Anthropic or Gemini. This file carries requests and answers
// between the wires, keeping exactly what the env's providers echo back
// (system text, user text, tool names, max_tokens, stop sequences, images,
// JSON mode) so the client's echo checks pass, and it maps stop reasons and
// usage counts into the wire the client used.
//
// Streams are buffered and sent again rather than piped across wires, so a
// re-sent stream carries the provider's text whole and its usage counts
// exact. Streams on the same wire go through byte for byte instead
// (main.ts).
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so a replay leaves the same bytes.

import { isRecord, type Json } from './providers.ts';

/** One fixed epoch second for every answer this file builds: nothing reads the clock. */
const EPOCH_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function collectText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argsAsObject(v: unknown): Json {
  if (isRecord(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(v);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A stop field as a string list: a bare string becomes one entry. */
function stopsAsList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultAsText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = collectText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One OpenAI image_url part as an Anthropic image block; null when it names no URL. */
function openAIImageAsAnthropic(part: Json): Json | null {
  const url = isRecord(part.image_url) ? part.image_url.url : undefined;
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

/** One OpenAI image_url part as a Gemini part; null when it names no URL. */
function openAIImageAsGemini(part: Json): Json | null {
  const url = isRecord(part.image_url) ? part.image_url.url : undefined;
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** One Anthropic image block as an OpenAI image_url part; null when it names no image. */
function anthropicImageAsOpenAI(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: the system messages become the system string, image_url parts
 * become image blocks, tool definitions and tool choice are rewritten, and
 * the output cap, temperature, top_p, stop sequences and streaming pass
 * through. response_format has no Anthropic carrier and is dropped: never
 * route a JSON-mode request to an Anthropic-wire provider.
 */
export function openAIRequestAsAnthropic(body: Json, model: string): Json {
  const turns: Json[] = [];
  let sys = '';
  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of incoming) {
    if (!isRecord(msg)) continue;
    if (msg.role === 'system') {
      const txt = collectText(msg.content);
      if (txt !== '') sys = sys === '' ? txt : `${sys}\n${txt}`;
      continue;
    }
    if (msg.role === 'assistant') {
      const segments: Json[] = [];
      const txt = collectText(msg.content);
      if (txt !== '') segments.push({ type: 'text', text: txt });
      const invocations = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
      for (const invocation of invocations) {
        if (!isRecord(invocation) || !isRecord(invocation.function)) continue;
        segments.push({
          type: 'tool_use',
          id: typeof invocation.id === 'string' ? invocation.id : '',
          name: typeof invocation.function.name === 'string' ? invocation.function.name : '',
          input: argsAsObject(invocation.function.arguments),
        });
      }
      if (segments.length > 0) turns.push({ role: 'assistant', content: segments });
      continue;
    }
    if (msg.role === 'tool') {
      turns.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof msg.tool_call_id === 'string' ? msg.tool_call_id : '',
            content: typeof msg.content === 'string' ? msg.content : collectText(msg.content),
          },
        ],
      });
      continue;
    }
    if (typeof msg.content === 'string') {
      turns.push({ role: 'user', content: msg.content });
      continue;
    }
    if (!Array.isArray(msg.content)) continue;
    const segments: Json[] = [];
    for (const part of msg.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') segments.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url') {
        const image = openAIImageAsAnthropic(part);
        if (image !== null) segments.push(image);
      }
    }
    if (segments.length > 0) turns.push({ role: 'user', content: segments });
  }

  const out: Json = { model, messages: turns };
  if (sys !== '') out.system = sys;
  const cap = numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens);
  if (cap !== null) out.max_tokens = cap;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stopWords = stopsAsList(body.stop);
  if (stopWords.length > 0) out.stop_sequences = stopWords;
  if (Array.isArray(body.tools)) {
    const toolset = body.tools.flatMap((def) => {
      if (!isRecord(def) || !isRecord(def.function) || typeof def.function.name !== 'string') return [];
      return [
        {
          name: def.function.name,
          description: typeof def.function.description === 'string' ? def.function.description : '',
          input_schema: isRecord(def.function.parameters) ? def.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (toolset.length > 0) out.tools = toolset;
  }
  const selection = body.tool_choice;
  if (selection === 'auto') out.tool_choice = { type: 'auto' };
  else if (selection === 'none') out.tool_choice = { type: 'none' };
  else if (selection === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(selection) && selection.type === 'function' && isRecord(selection.function) && typeof selection.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: selection.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The
 * model travels in the URL path, never in this body. JSON mode rides
 * generationConfig.responseMimeType, so a Gemini-wire provider can carry it.
 */
export function openAIRequestAsGemini(body: Json, _model: string): Json {
  const turns: Json[] = [];
  let sys = '';
  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of incoming) {
    if (!isRecord(msg)) continue;
    if (msg.role === 'system') {
      const txt = collectText(msg.content);
      if (txt !== '') sys = sys === '' ? txt : `${sys}\n${txt}`;
      continue;
    }
    if (msg.role === 'assistant') {
      const segments: Json[] = [];
      const txt = collectText(msg.content);
      if (txt !== '') segments.push({ text: txt });
      const invocations = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
      for (const invocation of invocations) {
        if (!isRecord(invocation) || !isRecord(invocation.function)) continue;
        segments.push({
          functionCall: {
            name: typeof invocation.function.name === 'string' ? invocation.function.name : '',
            args: argsAsObject(invocation.function.arguments),
          },
        });
      }
      if (segments.length > 0) turns.push({ role: 'model', parts: segments });
      continue;
    }
    if (msg.role === 'tool') {
      turns.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { content: typeof msg.content === 'string' ? msg.content : collectText(msg.content) } } }],
      });
      continue;
    }
    if (typeof msg.content === 'string') {
      turns.push({ role: 'user', parts: [{ text: msg.content }] });
      continue;
    }
    if (!Array.isArray(msg.content)) continue;
    const segments: Json[] = [];
    for (const part of msg.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') segments.push({ text: part.text });
      else if (part.type === 'image_url') {
        const image = openAIImageAsGemini(part);
        if (image !== null) segments.push(image);
      }
    }
    if (segments.length > 0) turns.push({ role: 'user', parts: segments });
  }

  const out: Json = { contents: turns };
  if (sys !== '') out.systemInstruction = { parts: [{ text: sys }] };
  if (Array.isArray(body.tools)) {
    const specs = body.tools.flatMap((def) => {
      if (!isRecord(def) || !isRecord(def.function) || typeof def.function.name !== 'string') return [];
      return [
        {
          name: def.function.name,
          description: typeof def.function.description === 'string' ? def.function.description : '',
          parameters: isRecord(def.function.parameters) ? def.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (specs.length > 0) out.tools = [{ functionDeclarations: specs }];
  }
  const genConfig: Json = {};
  const cap = numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens);
  if (cap !== null) genConfig.maxOutputTokens = cap;
  if (typeof body.temperature === 'number') genConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') genConfig.topP = body.top_p;
  const stopWords = stopsAsList(body.stop);
  if (stopWords.length > 0) genConfig.stopSequences = stopWords;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    genConfig.responseMimeType = 'application/json';
  }
  if (Object.keys(genConfig).length > 0) out.generationConfig = genConfig;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The model
 * travels in the URL path, never in this body.
 */
export function anthropicRequestAsGemini(body: Json, _model: string): Json {
  const turns: Json[] = [];
  const sys = collectText(body.system);
  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of incoming) {
    if (!isRecord(msg)) continue;
    const payload = msg.content;
    if (msg.role === 'assistant') {
      const segments: Json[] = [];
      if (typeof payload === 'string') {
        if (payload !== '') segments.push({ text: payload });
      } else if (Array.isArray(payload)) {
        for (const item of payload) {
          if (!isRecord(item)) continue;
          if (item.type === 'text' && typeof item.text === 'string') segments.push({ text: item.text });
          else if (item.type === 'tool_use') {
            segments.push({
              functionCall: { name: typeof item.name === 'string' ? item.name : '', args: isRecord(item.input) ? item.input : {} },
            });
          }
        }
      }
      if (segments.length > 0) turns.push({ role: 'model', parts: segments });
      continue;
    }
    const segments: Json[] = [];
    const fnResults: Json[] = [];
    if (typeof payload === 'string') {
      if (payload !== '') segments.push({ text: payload });
    } else if (Array.isArray(payload)) {
      for (const item of payload) {
        if (!isRecord(item)) continue;
        if (item.type === 'text' && typeof item.text === 'string') segments.push({ text: item.text });
        else if (item.type === 'image' && isRecord(item.source)) {
          if (item.source.type === 'base64' && typeof item.source.data === 'string') {
            segments.push({
              inlineData: {
                mimeType: typeof item.source.media_type === 'string' ? item.source.media_type : 'image/png',
                data: item.source.data,
              },
            });
          } else if (item.source.type === 'url' && typeof item.source.url === 'string') {
            segments.push({ fileData: { fileUri: item.source.url } });
          }
        } else if (item.type === 'tool_result') {
          fnResults.push({
            functionResponse: {
              name: typeof item.name === 'string' ? item.name : '',
              response: { content: toolResultAsText(item.content) },
            },
          });
        }
      }
    }
    if (segments.length > 0) turns.push({ role: 'user', parts: segments });
    if (fnResults.length > 0) turns.push({ role: 'function', parts: fnResults });
  }

  const out: Json = { contents: turns };
  if (sys !== '') out.systemInstruction = { parts: [{ text: sys }] };
  if (Array.isArray(body.tools)) {
    const specs = body.tools.flatMap((def) => {
      if (!isRecord(def) || typeof def.name !== 'string') return [];
      return [
        {
          name: def.name,
          description: typeof def.description === 'string' ? def.description : '',
          parameters: isRecord(def.input_schema) ? def.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (specs.length > 0) out.tools = [{ functionDeclarations: specs }];
  }
  const genConfig: Json = {};
  if (typeof body.max_tokens === 'number') genConfig.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') genConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') genConfig.topP = body.top_p;
  const stopWords = stopsAsList(body.stop_sequences);
  if (stopWords.length > 0) genConfig.stopSequences = stopWords;
  if (Object.keys(genConfig).length > 0) out.generationConfig = genConfig;
  return out;
}

/** One provider answer, normalized into OpenAI's stop vocabulary. */
export interface Answer {
  text: string;
  input: number;
  output: number;
  /** OpenAI finish_reason: stop, length, tool_calls or content_filter. */
  stop: string;
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiStopAsOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopAsOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** An OpenAI finish_reason as an Anthropic stop_reason. */
export function openAIStopAsAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A Gemini generateContent answer, parsed: the text, the usage and the stop. */
export function readGeminiAnswer(body: Json): Answer {
  const options = Array.isArray(body.candidates) ? body.candidates : [];
  const head = isRecord(options[0]) ? options[0] : {};
  const payload = isRecord(head.content) ? head.content : {};
  const segments = Array.isArray(payload.parts) ? payload.parts : [];
  const txt = segments.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text: txt, input: numOrZero(usage.promptTokenCount), output: numOrZero(usage.candidatesTokenCount), stop: geminiStopAsOpenAI(head.finishReason) };
}

/** An Anthropic Messages answer, parsed: the text, the usage and the stop. */
export function readAnthropicAnswer(body: Json): Answer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const txt = blocks.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const counts = isRecord(body.usage) ? body.usage : {};
  return { text: txt, input: numOrZero(counts.input_tokens), output: numOrZero(counts.output_tokens), stop: anthropicStopAsOpenAI(body.stop_reason) };
}

/** A Chat Completions answer, parsed: the text, the usage and the stop. */
export function readOpenAIAnswer(body: Json): Answer {
  const options = Array.isArray(body.choices) ? body.choices : [];
  const head = isRecord(options[0]) ? options[0] : {};
  const msg = isRecord(head.message) ? head.message : {};
  const txt = typeof msg.content === 'string' ? msg.content : collectText(msg.content);
  const counts = isRecord(body.usage) ? body.usage : {};
  return {
    text: txt,
    input: numOrZero(counts.prompt_tokens),
    output: numOrZero(counts.completion_tokens),
    stop: typeof head.finish_reason === 'string' ? head.finish_reason : 'stop',
  };
}

/** A parsed answer as a Chat Completions object naming `model`. */
export function answerAsOpenAI(parsed: Answer, model: string): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: EPOCH_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: parsed.text }, finish_reason: parsed.stop, logprobs: null }],
    usage: { prompt_tokens: parsed.input, completion_tokens: parsed.output, total_tokens: parsed.input + parsed.output },
  };
}

/** A parsed answer as an Anthropic Messages object naming `model`. */
export function answerAsAnthropic(parsed: Answer, model: string): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: parsed.text }],
    stop_reason: openAIStopAsAnthropic(parsed.stop),
    stop_sequence: null,
    usage: { input_tokens: parsed.input, output_tokens: parsed.output },
  };
}

/** An Anthropic Messages answer as a Chat Completions object naming `model`, tool_use blocks included. */
export function anthropicAnswerAsOpenAI(body: Json, model: string): Json {
  const blocks = Array.isArray(body.content) ? body.content : [];
  const txt = blocks.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const calls: Json[] = [];
  for (const item of blocks) {
    if (!isRecord(item) || item.type !== 'tool_use') continue;
    calls.push({
      id: typeof item.id === 'string' ? item.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof item.name === 'string' ? item.name : '', arguments: JSON.stringify(isRecord(item.input) ? item.input : {}) },
    });
  }
  const counts = isRecord(body.usage) ? body.usage : {};
  const msg: Json = { role: 'assistant', content: txt };
  if (calls.length > 0) msg.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: EPOCH_CREATED,
    model,
    choices: [{ index: 0, message: msg, finish_reason: anthropicStopAsOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numOrZero(counts.input_tokens),
      completion_tokens: numOrZero(counts.output_tokens),
      total_tokens: numOrZero(counts.input_tokens) + numOrZero(counts.output_tokens),
    },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text the way the providers do, so re-emitted deltas read naturally. */
function sliceText(text: string): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += 13) chunks.push(text.slice(i, i + 13));
  return chunks.length === 0 ? [''] : chunks;
}

/** A parsed answer as a Chat Completions event stream naming `model`, usage chunk included. */
export function answerAsOpenAIStream(parsed: Answer, model: string): string {
  const frame = (part: Json, done: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: EPOCH_CREATED,
    model,
    choices: [{ index: 0, delta: part, finish_reason: done }],
  });
  const events: string[] = [sseFrame(null, frame({ role: 'assistant', content: '' }, null))];
  for (const slice of sliceText(parsed.text)) events.push(sseFrame(null, frame({ content: slice }, null)));
  events.push(sseFrame(null, frame({}, parsed.stop)));
  events.push(
    sseFrame(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: EPOCH_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: parsed.input, completion_tokens: parsed.output, total_tokens: parsed.input + parsed.output },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events.join('');
}

/** A parsed answer as the Anthropic event sequence naming `model`. */
export function answerAsAnthropicStream(parsed: Answer, model: string): string {
  const events: string[] = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: parsed.input, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const slice of sliceText(parsed.text)) {
    events.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: slice } }));
  }
  events.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: openAIStopAsAnthropic(parsed.stop), stop_sequence: null },
      usage: { output_tokens: parsed.output },
    }),
  );
  events.push(sseFrame('message_stop', { type: 'message_stop' }));
  return events.join('');
}

/** A Chat Completions event stream, parsed: the text, the usage and the stop. */
export function readOpenAIStream(text: string): Answer {
  let txt = '';
  let prompt = 0;
  let completion = 0;
  let finish = 'stop';
  for (const block of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const row of block.split('\n')) {
      const clean = row.trim();
      if (!clean.startsWith('data:')) continue;
      const raw = clean.slice(5).trim();
      if (raw === '[DONE]') continue;
      let piece: unknown;
      try {
        piece = JSON.parse(raw);
      } catch {
        continue;
      }
      if (!isRecord(piece)) continue;
      if (isRecord(piece.usage)) {
        prompt = numOrZero(piece.usage.prompt_tokens);
        completion = numOrZero(piece.usage.completion_tokens);
      }
      const options = Array.isArray(piece.choices) ? piece.choices : [];
      const head = isRecord(options[0]) ? options[0] : null;
      if (head === null) continue;
      const part = isRecord(head.delta) ? head.delta : {};
      if (typeof part.content === 'string') txt += part.content;
      if (typeof head.finish_reason === 'string') finish = head.finish_reason;
    }
  }
  return { text: txt, input: prompt, output: completion, stop: finish };
}

/** An Anthropic event stream, parsed: the text, the usage and the stop. */
export function readAnthropicStream(text: string): Answer {
  let txt = '';
  let prompt = 0;
  let completion = 0;
  let finish = 'stop';
  for (const block of text.replace(/\r\n/g, '\n').split('\n\n')) {
    let kind: string | null = null;
    const rows: string[] = [];
    for (const row of block.split('\n')) {
      if (row.startsWith('event:')) kind = row.slice(6).trim();
      else if (row.startsWith('data:')) rows.push(row.slice(5).replace(/^ /, ''));
    }
    if (rows.length === 0) continue;
    let payload: unknown;
    try {
      payload = JSON.parse(rows.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(payload)) continue;
    if (kind === 'message_start' && isRecord(payload.message) && isRecord(payload.message.usage)) prompt = numOrZero(payload.message.usage.input_tokens);
    if (kind === 'content_block_delta' && isRecord(payload.delta) && payload.delta.type === 'text_delta' && typeof payload.delta.text === 'string') {
      txt += payload.delta.text;
    }
    if (kind === 'message_delta') {
      if (isRecord(payload.delta) && typeof payload.delta.stop_reason === 'string') finish = anthropicStopAsOpenAI(payload.delta.stop_reason);
      if (isRecord(payload.usage)) completion = numOrZero(payload.usage.output_tokens);
    }
  }
  return { text: txt, input: prompt, output: completion, stop: finish };
}

/** A Gemini streamGenerateContent stream, parsed: the text, the usage and the stop. */
export function readGeminiStream(text: string): Answer {
  let txt = '';
  let prompt = 0;
  let completion = 0;
  let finish = 'stop';
  for (const block of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const row of block.split('\n')) {
      const clean = row.trim();
      if (!clean.startsWith('data:')) continue;
      const raw = clean.slice(5).trim();
      let payload: unknown;
      try {
        payload = JSON.parse(raw);
      } catch {
        continue;
      }
      if (!isRecord(payload)) continue;
      const options = Array.isArray(payload.candidates) ? payload.candidates : [];
      const head = isRecord(options[0]) ? options[0] : null;
      if (head !== null) {
        const payload2 = isRecord(head.content) ? head.content : {};
        const segments = Array.isArray(payload2.parts) ? payload2.parts : [];
        for (const p of segments) if (isRecord(p) && typeof p.text === 'string') txt += p.text;
        if (typeof head.finishReason === 'string') finish = geminiStopAsOpenAI(head.finishReason);
      }
      if (isRecord(payload.usageMetadata)) {
        prompt = numOrZero(payload.usageMetadata.promptTokenCount);
        completion = numOrZero(payload.usageMetadata.candidatesTokenCount);
      }
    }
  }
  return { text: txt, input: prompt, output: completion, stop: finish };
}
