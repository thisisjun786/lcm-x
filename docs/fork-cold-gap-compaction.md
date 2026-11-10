# Fork: cold-gap compaction at turn start

Status: built, **off by default** (`LCM_COLD_GAP_COMPACTION_SECONDS=0`). This is option D ("turn-start cold-gap
trigger") of section 2 of the fork design note `docs/fork-target-sized-sweep.md` (branch `feat/sweep-target-prompt`,
commit 9a6bbbb7). Part 1 of that note (the target-sized sweep) was not adopted: a cold-gap pass runs the existing
threshold full sweep.

## Why

Prompt caching has a one hour TTL. When a large conversation sits idle past it, the next request rewrites the whole
prompt into the cache. The cache is cold at that point anyway, so compacting at the start of that turn adds no cache
break and makes that rewrite, and every later request of the conversation, smaller. The turn is slower; that is
accepted.

Hermes already calls `should_compress_preflight(messages)` once at turn start when the request is below the threshold
(`agent/turn_context_compaction.py::_engine_preflight_maintenance`, trigger `engine_preflight`) and calls
`compress(messages, current_tokens=...)` when it returns True. No Hermes change is needed. Hermes'
`compression.idle_compact_after_seconds` cannot be used: the gateway stamps its activity clock at turn entry.

## Settings

| Key | Default | Valid |
|---|---|---|
| `LCM_COLD_GAP_COMPACTION_SECONDS` / `lcm.cold_gap_compaction_seconds` | `0` (off) | `0`, or `300`-`86400` |
| `LCM_COLD_GAP_MIN_PROMPT_TOKENS` / `lcm.cold_gap_min_prompt_tokens` | `300000` | `1`-`10000000` |

Env wins over config.yaml. A non-integer or out-of-range value is ignored with a config warning and the next source
applies. The token unit is the provider-reported prompt (system prompt + tools + messages), not LCM's message count.

## Trigger (`_should_compress_preflight_impl` -> `_cold_gap_compaction_due`)

Preflight returns True with the one-shot `CompactionRequest.cold_gap` flag when all hold:

- the setting is `> 0` and `threshold_tokens > 0`;
- the request is below the threshold (at or over it, the threshold path runs unchanged);
- this turn's ingest succeeded, and the session is managed (not ignored, stateless or auxiliary);
- no boundary cooldown, #608 sweep budget hold or #651 no-progress hold is active, and a summary route is available;
- `now - last_api_call_at` of this conversation's compaction telemetry record (written after each provider response,
  durable) is at least the setting; no record or no timestamp means no trigger;
- that record's `last_observed_prompt_tokens` (or the in-memory `last_prompt_tokens` only when the record has none) is at
  least the min setting;
- no cold-gap compaction was requested yet for this `last_api_call_at`.

When it fires, the metadata key `cold_gap_compaction:<conversation_id>` records the `last_api_call_at` it fired for, so
the same idle gap never fires twice, also after a restart. A newer provider response (newer `last_api_call_at`) re-arms
it. One INFO line is logged:
`LCM cold-gap compaction requested: gap=<s>s observed_prompt_tokens=<n> min=<n> threshold=<n>`.

## Compaction

`compress()` consumes the flag (not for forced or recovery calls, and only while the host count is still below the
threshold) and:

- does not turn the call into below-threshold cleanup-only maintenance (#651/#677);
- arms the threshold full sweep (`threshold_full_sweep_enabled` still applies), bounded by
  `threshold_full_sweep_max_passes` and the foreground soft/hard limits; the fresh tail rules are unchanged;
- skips the #671 stub-first exit (it is tied to the over-threshold no-progress candidate; below the threshold its
  target would usually already be met and the pass would store no leaf);
- arms no #651 no-progress or hidden-only hold and never counts toward the zero-leaf streak: a pass that stores
  nothing just ends.

`lcm_status` reports `cold_gap_compaction` (`seconds`, `min_prompt_tokens`, `last_event` with `requested_at`,
`gap_seconds`, `observed_prompt_tokens`, `leaf_passes`, `tokens_before`, `tokens_after`, `duration_ms`, `stop_reason`;
the sweep numbers come from `threshold_full_sweep`) and the two settings under `config`.

## Before enabling

An operator replay on real data is required before merge. Suggested start: `3600` seconds with the default min.

## Cold-cache causes, interruption and no-progress (lina#455)

Settings: `LCM_COLD_CACHE_CAUSES` / `lcm.cold_cache_causes`, default `idle,model,prompt` (comma list; unknown names
ignored with a warning; a value naming none of them is ignored). `cold_gap_compaction_seconds = 0` still turns the
whole family off; the other settings and defaults are unchanged.

Causes (`_cold_cache_cause`), same floor (`cold_gap_min_prompt_tokens`), holds and exclusions as the idle trigger:

- `idle`: the gap reached the setting (unchanged; its log line is unchanged too);
- `model_change`: `self.model` / `self.provider` (set by `update_model`) differ from the `model` / `provider` of the
  last call's telemetry record;
- `prompt_change`: the sha256 of the canonical JSON of the leading system message(s) of the request about to be sent
  differs from `last_system_prompt_sha` of the last call's record. `LCMEngine.select_context` observes it (always
  returns None, never raises, never mutates, no-op for bypassed traffic); the response hook writes it with
  `last_api_call_at` in one telemetry update. Because the prompt is only known at assembly, this cause is consumed by
  the pre-API gate `should_compress(tokens)`; preflight and the gate call the same trigger.

A compaction committed after the last call (`last_leaf_compaction_at >= last_api_call_at`, or compactions this
process counted that the record has not seen) suppresses `model_change` and `prompt_change`. Non-idle causes log
`LCM cold-gap compaction requested: ... cause=<cause> requests=<n>`.

One-shot bookkeeping (`cold_gap_compaction:<conversation>`, keys kept): `last_event` adds `cause`, `requests`,
`attempts`, `outcome` (`requested` | `shrunk` | `no_progress` | `error`), `pass_started` and `owner`. The event is
consumed when the pass ends with a result (`shrunk` or `no_progress`). A pass that raised (`error`) or started and never
finished (`requested` with `pass_started`, e.g. the process died) may be requested again when this process is not
running it (another owner, or older than the foreground hard budget), at most 2 requests per `last_api_call_at`. A
request whose pass never started stays consumed (the previous once-per-gap behaviour across a restart).

No progress: a cold pass that leaves the same number of rows and less than 5% fewer tokens is retried once inside
the same `compress()` call (same foreground budget) with the pass treated as over the threshold for the #651
candidate: the #671 stub-first free cuts (taken only when they free more than 5%), then the same threshold sweep and
leaf/condensation loop. The exit fit (#668), deterministic L3 truncation and memory-block cuts stay off (the host
count is still below the threshold), and neither attempt arms a no-progress, hidden-only or sweep-budget hold. If the
retry frees nothing too, `compress()` returns the original list, logs one WARNING
`LCM cold-cache compaction freed nothing: cause=... attempts=2 stop=... first_stop=... rows=... tokens=...` and
records `outcome=no_progress`.

Leaf valve (views that carry host uids only, the d8a937e3 scope): when the oldest chunk gives the leaf builder
nothing (an empty leaf input that is not a retained-row block), the selection leaves it unselected and takes the next chunk, at most 3
times per pass, with one WARNING per skip naming store ids and content lengths. Nothing is deleted. A pass whose
every chunk was skipped stops with `no_usable_leaf_input` and the original no-op reason (no progress, not drained).
