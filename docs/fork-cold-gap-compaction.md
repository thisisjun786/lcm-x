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
