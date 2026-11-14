# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, routing each
request to the best free model that can serve it and failing over when a
provider refuses. Built for the make.fun `free-model-aggregator-01`
competition.

## Run

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, registers at $MAKEFUN_ENV_URL
node main.ts --list     # prints the catalog document and exits 0
```

Node 22, standard library only; nothing is installed. The entrypoint is
`main.ts`; the provider directory, quality table and workload come from the
trial brief.

## How it routes

Every free model that meets a request's requirements (tools, vision, JSON
mode, minimum context) is ranked by the brief's quality table
(`rating_milli`), best first, ties broken by provider id then model id so
the order is deterministic across a trial's live pass and its replay. A
JSON-mode request from an OpenAI client goes only to a wire that can carry
`response_format` (OpenAI, or Gemini through
`generationConfig.responseMimeType`); an Anthropic-wire provider echoes JSON
mode absent, so it is never picked for one. The first refusal (402, 429,
5xx, or an unreachable provider) moves to the next candidate; only one
provider call may answer a request.

## Wires

The client wire and the provider wire are independent: an OpenAI request
may be served by an Anthropic or Gemini provider and vice versa.
`lib/xwire.ts` converts requests (system prompt, user text and images,
tools, stop sequences, `max_tokens`, JSON mode) and answers (text, stop
reason, usage) between the wires, preserving exactly what the provider
echoes. Same-wire answers pass through byte for byte; cross-wire streams
are buffered and re-emitted with the provider's usage exact.

## Files

- `main.ts` — the server: brief, catalog, routing, failover, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe.
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation and SSE writers.
- `lib/xwire.ts` — the cross-wire request/answer conversions (OpenAI,
  Anthropic, Gemini).

## License

MIT — see `LICENSE`.
