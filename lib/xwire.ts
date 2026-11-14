// lib/xwire.ts — cross-wire translation for the free-model aggregator.
//
// The trial's client speaks OpenAI or Anthropic; the ten providers speak
// OpenAI, Anthropic or Gemini. This file converts requests and answers
// between the wires, preserving exactly what the env's providers echo back
// (system text, user text, tool names, max_tokens, stop sequences, images,
// JSON mode) so the client's echo checks pass, and mapping stop reasons and
// usage counts into the wire the client used.
//
// Streams are buffered and re-emitted rather than piped across wires, so a
// re-emitted stream carries the provider's text whole and its usage counts
// exact. Same-wire streams pass through byte for byte instead (main.ts).
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so a replay leaves the same bytes.

import { isRecord, type Json } from './providers.ts';

/** A fixed epoch second for every answer this file builds: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argsObject(v: unknown): Json {
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
function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One OpenAI image_url part as an Anthropic image block; null when it names no URL. */
function openAIImageToAnthropic(part: Json): Json | null {
  const url = isRecord(part.image_url) ? part.image_url.url : undefined;
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

/** One OpenAI image_url part as a Gemini part; null when it names no URL. */
function openAIImageToGemini(part: Json): Json | null {
  const url = isRecord(part.image_url) ? part.image_url.url : undefined;
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** One Anthropic image block as an OpenAI image_url part; null when it names no image. */
function anthropicImageToOpenAI(block: Json): Json | null {
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
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url') {
        const image = openAIImageToAnthropic(part);
        if (image !== null) blocks.push(image);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const max = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (max !== null) out.max_tokens = max;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The
 * model travels in the URL path, never in this body. JSON mode rides
 * generationConfig.responseMimeType, so a Gemini-wire provider can carry it.
 */
export function openAIToGeminiRequest(body: Json, _model: string): Json {
  const contents: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argsObject(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
      });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role: 'user', parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url') {
        const image = openAIImageToGemini(part);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const max = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (max !== null) config.maxOutputTokens = max;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The model
 * travels in the URL path, never in this body.
 */
export function anthropicToGeminiRequest(body: Json, _model: string): Json {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({
              functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} },
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    const toolResults: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({
              inlineData: {
                mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
                data: block.source.data,
              },
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_result') {
          toolResults.push({
            functionResponse: {
              name: typeof block.name === 'string' ? block.name : '',
              response: { content: toolResultText(block.content) },
            },
          });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (toolResults.length > 0) contents.push({ role: 'function', parts: toolResults });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** One provider answer, normalized into OpenAI's stop vocabulary. */
export interface ParsedAnswer {
  text: string;
  input: number;
  output: number;
  /** OpenAI finish_reason: stop, length, tool_calls or content_filter. */
  stop: string;
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiFinishToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** An OpenAI finish_reason as an Anthropic stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A Gemini generateContent answer, parsed: the text, the usage and the stop. */
export function parseGeminiAnswer(body: Json): ParsedAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount), stop: geminiFinishToOpenAI(first.finishReason) };
}

/** An Anthropic Messages answer, parsed: the text, the usage and the stop. */
export function parseAnthropicAnswer(body: Json): ParsedAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens), stop: anthropicStopToOpenAI(body.stop_reason) };
}

/** A Chat Completions answer, parsed: the text, the usage and the stop. */
export function parseOpenAIAnswer(body: Json): ParsedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    input: numberOrZero(usage.prompt_tokens),
    output: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** A parsed answer as a Chat Completions object naming `model`. */
export function openAIJsonOf(answer: ParsedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: answer.stop, logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A parsed answer as an Anthropic Messages object naming `model`. */
export function anthropicJsonOf(answer: ParsedAnswer, model: string): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: openAIStopToAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** An Anthropic Messages answer as a Chat Completions object naming `model`, tool_use blocks included. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text the way the providers do, so re-emitted deltas read naturally. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A parsed answer as a Chat Completions event stream naming `model`, usage chunk included. */
export function openAIStreamOf(answer: ParsedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, answer.stop)));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A parsed answer as the Anthropic event sequence naming `model`. */
export function anthropicStreamOf(answer: ParsedAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: openAIStopToAnthropic(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/** A Chat Completions event stream, parsed: the text, the usage and the stop. */
export function parseOpenAIStreamText(text: string): ParsedAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = numberOrZero(chunk.usage.prompt_tokens);
        output = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
  }
  return { text: out, input, output, stop };
}

/** An Anthropic event stream, parsed: the text, the usage and the stop. */
export function parseAnthropicStreamText(text: string): ParsedAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    if (dataLines.length === 0) continue;
    let data: unknown;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) input = numberOrZero(data.message.usage.input_tokens);
    if (event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicStopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, input, output, stop };
}

/** A Gemini streamGenerateContent stream, parsed: the text, the usage and the stop. */
export function parseGeminiStreamText(text: string): ParsedAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let data: unknown;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      const candidates = Array.isArray(data.candidates) ? data.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : null;
      if (first !== null) {
        const content = isRecord(first.content) ? first.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
        if (typeof first.finishReason === 'string') stop = geminiFinishToOpenAI(first.finishReason);
      }
      if (isRecord(data.usageMetadata)) {
        input = numberOrZero(data.usageMetadata.promptTokenCount);
        output = numberOrZero(data.usageMetadata.candidatesTokenCount);
      }
    }
  }
  return { text: out, input, output, stop };
}
