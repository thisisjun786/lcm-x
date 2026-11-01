# Fork design: target-sized threshold sweep and idle-gap pre-compaction (G60)

Status: part 1 is implemented on top of T176 (`8f6d1a03`, `3eabe55b`, the condensation reserve R); part 2 is
design only. Part 1 is off by default (`LCM_SWEEP_TARGET_PROMPT_TOKENS=0`).

Decisions from the owner (2026-11-02): (1) sweep until the next request fits a target size, bounded: required.
(2) defer compaction until after the reply: not done (latency is accepted). (3) pre-compact while the session is idle:
do it if it is safe, otherwise report why not. Unchanged: `context_threshold` (0.7), the fresh tail rules, one value
for all eight bots.

## 1. Target-sized sweep

### What exists (daa9e995)

- The sweep is armed when the observed prompt reaches the threshold (`compaction.py:2069-2074`,
  `threshold_full_sweep_active`).
- Leaf budget: `state.max_leaf_passes = threshold_full_sweep_max_passes` (`compaction.py:2137-2142`), minus the
  pre-leaf condensation passes (`compaction.py:2180`).
- The leaf loop runs `while leaf_passes < max_leaf_passes` (`compaction.py:3421`) and stops on `raw_prefix_drained`
  (`compaction.py:2251-2256`), the time bound or the pass budget (`compaction.py:2443-2447`).
- Post-leaf condensation runs only when the raw prefix drained (`compaction.py:2650-2664`), with the passes the
  leaves left. With 14 passes and a long backlog the leaves spend all 14, so condensation runs 0 times and the
  summary head stays at 49.5k-56.4k against its 24k target (T176's finding).
- There is no size target for the whole request. The only targets are the threshold (start) and
  `sweep_target_tokens` (the summary head, `compaction.py:2081-2089`).

### The unit: host tokens, scaled by an observed ratio

The target is the provider's prompt size (what OCX and the API count: system prompt, tools and messages). LCM counts
the same list with its own tokenizer (tiktoken cl100k, or a character estimate). On the Korean-heavy fleet sessions the
two differ by a factor, not by a constant: hub-mac records give host 728,870 vs LCM 454,752 at one moment (1.60), and
LCM 303k after a compaction vs 499k on the next request (1.65). Adding `_survival_host_overhead()` (observed minus
counted) as a constant would treat the whole 274k gap as system prompt and keep it after the sweep, so the projection
would be about 1.5x the real figure at the target. The sweep therefore uses a ratio:

- At sweep entry, `ratio = observed prompt tokens / LCM count of the same list` (`compaction.py`
  `_prompt_token_ratio`). The observation is the host count compress() was given (the last request).
- A ratio outside 0.5-4.0, or no observation, is unknown: no target decision is made, the sweep runs on its pass and
  time bounds as with target 0, and `lcm_status` reports `prompt_token_ratio: null`.
- Projection during leaves: `ratio x (LCM count at entry - tokens freed so far)`; the freed tokens come from the
  running `estimated_active_tokens` (each stored leaf subtracts its source and adds its summary; pre-leaf
  condensation subtracts the frontier it removed). The estimate leaves out how assembly renders summaries and
  under-counts (about 3% in the unit test), so a fit by the estimate is confirmed on a no-write trial assembly
  (`_assemble_committed_compaction_context(persist=False)`) before the leaves stop.
- Reported figure: `projected_prompt_tokens_after = round(ratio x count(assembled context))`.

### Implemented behavior

With `LCM_SWEEP_TARGET_PROMPT_TOKENS` / `lcm.sweep_target_prompt_tokens` > 0 (0-4,000,000) and the threshold
sweep active:

1. Pre-leaf condensation as before (#653, within the T176 bound).
2. Leaves as before, plus one check after each stored leaf: when the projected request fits the target, the leaves stop
   with `target_reached` (no further leaf call). The fresh tail and the leaf candidate rules are unchanged.
3. Condensation after the leaves (post-drain, or T176's reserve after a capped leaf phase) condenses toward
   `min(frontier target, frontier - LCM tokens the request is still over)` instead of the frontier target alone, so a
   request still over the prompt target is condensed below the usual frontier target. Without a target or a ratio the
   condensation target is unchanged.
4. Final stop reason: `target_reached` when the projected request fits (unless a failure reason applies:
   `summary_route_unavailable`, `summary_result_rejected`, `leaf_summary_error`, `condensation_error`);
   otherwise the pass and time stops keep their reasons (`pass_budget_exhausted`, `time_budget_exhausted`,
   `soft_target_reached`), and a sweep that drained the raw prefix and ran out of condensation material
   (`raw_prefix_drained`, `summary_prefix_target_reached`, `condensation_no_progress`,
   `no_same_depth_condensation_group`) while still over reports `target_unreachable` (partial).
5. `lcm_status` `threshold_full_sweep` adds `target_prompt_tokens`, `prompt_token_ratio` and
   `projected_prompt_tokens_after` only when the setting is > 0; with 0 the keys, the no-progress fingerprint and the
   golden traces' events are unchanged.

Not done: a second leaf round after condensation. The leaf phase ends with passes left only on `target_reached`
(done), `raw_prefix_drained` (no raw left) or a time stop (no time left), so another leaf round has nothing to do in
each case; condensation always runs after the leaves.

Merge with T176: the reserve is kept as T176 built it (`_condense_within_sweep_reserve`, budget
`min(R, cap - leaf calls - pre-leaf condensation)`); the target only lowers its `target_tokens`. Lifting the R cap
would change nothing measurable: the leaf phase is capped at `cap - R` (promoted leaves extend it by one each), so the
passes left after a capped leaf phase are exactly R. With R = 0 a capped leaf phase still makes no condensation, so the
target arms need R > 0.

### Bounds

- Pass cap: `LCM_THRESHOLD_FULL_SWEEP_MAX_PASSES` (1-64), a ceiling in target mode. Keep it at or below
  `LCM_SUMMARY_SPEND_MAX_CALLS` (fleet 24 per 600 s), or the spend guard stops the sweep and backs off for 1800 s.
- Time: `LCM_FOREGROUND_SOFT_SECONDS` / `LCM_FOREGROUND_HARD_SECONDS` unchanged (fleet 480 / 600). A leaf call is
  about 21 s on the fleet route (12 leaves in 272 s), so the soft bound admits about 22 calls.
- Host limits (Hermes `main` c627ec84; the deployed version was not checked): the compression wrapper's total ceiling
  `compression.context_total_ceiling_seconds` defaults to 600 s and discards a compaction still running then
  (`agent/conversation_compression.py:1-8`); the fleet's LCM hard bound is already 600, so a sweep that uses all of it
  can meet the host ceiling. Stored summaries stay stored either way (nothing is lost). The gateway inactivity timeout
  is 1800 s and the host compression heartbeat touches activity every 60 s.

## 2. Idle-gap pre-compaction

### Candidates

| Where | Effect on the next request | Risk | Host change |
|---|---|---|---|
| A. Timer thread in the LCM plugin publishing a compaction while idle | None by itself: the host still holds the long message list and sends it, because the host adopts a compaction only from `compress()` it calls | Off-turn canonical writes were rejected for #787 for five reasons (`docs/async-background-compaction.md`, "v0.28.0 decisions"): readers see canonical nodes at once, session end uses the in-process frontier, hygiene runs on a clone, `on_session_end` waits on `_stable_use_lock` without timeout, condensation publication has no fence. Also races the host's per-session durable compression lock and turn lease | none, but unsafe |
| B. Gateway idle hook | Same as A unless the hook calls the agent's `_compress_context` | Hermes has no idle hook for context engines; only session expiry (`on_session_end`, a real boundary) | core change |
| C. launchd job calling a plugin command | Same as A | A second process writes the LCM DB while the gateway holds engine state in memory: SQLite locks, stale frontier, T164-type duplicate rows | none, but unsafe |
| D. Turn-start trigger (proposed) | The first request after a cold gap is sent already compacted | Same path as a threshold compaction: inside the host's turn, its lease and its compression lock | none |

### Proposed: D, the cold-gap trigger

The cache is cold after the gap anyway, so compacting at that turn costs no extra cache break. The owner accepts
the latency.

- New setting `LCM_COLD_GAP_COMPACTION_SECONDS` / `lcm.cold_gap_compaction_seconds` (`0` = off, proposed 3600 = the
  cache TTL; 55 min would sometimes break a cache that is still warm, so the trigger is the TTL, not 55 min).
- In `_should_compress_preflight_impl` (`compaction.py:255`), before the new turn is ingested: if the target is set,
  the gap since the conversation's last stored row (`max(ingested_at)`, durable across restarts; the in-memory
  turn-end time when present) is at least the setting, and the rough prompt is over the target, request an
  automatic compaction that runs the target-sized sweep even below the threshold. Today a below-threshold request is
  cleanup-only (`_mark_below_threshold_maintenance`, `compaction.py:482-486`), so it needs its own request flag.
- Fast path: with `LCM_BACKGROUND_LEAF_PREP_ENABLED` the leaves are prepared right after each turn
  (`engine.py:2746`, #787, nothing durable written off-turn) and promoted at that turn start without calls.
- Not triggered for manual, recovery, bypassed or auxiliary traffic; the existing holds and spend guard apply.

Depends on part 1 (the target), so it lands after part 1 and stays off by default.

### Savings estimate

Per cold-gap event over the target: `(P - T) x cache_write_price` on that request, plus
`(P - T) x cache_read_price` on each following warm request, minus the summary calls. The OCX records and the
scripts (`cache-break-causes.py`, `ocx-opus-policy-sim.py`) live on hub-mac; the estimator to run there is
`precompact_estimate.py` (in the task output). Example: P = 600k, T = 150k, 1 h write at $10/MTok, read at
$0.50/MTok, 2 follow-up requests, 8 summary calls of 10k in / 2k out at $3 / $15 per MTok: rewrite $6.00, saved
$4.50 + $0.45, summary cost $0.48, net $4.47. Prices are inputs, not checked.

## Replay plan (hub-mac)

See the task report for the arm table at the fleet's current values. Fields: `lcm_status` `threshold_full_sweep`
`stop_reason`, `status`, `leaf_passes`, `condensation_passes`, `pre_leaf_condensation_passes`,
`reserve_condensation_passes`, `total_passes`, `duration_ms`, `tokens_before`, `tokens_after`,
`summary_prefix_tokens_before`, `summary_prefix_tokens_after`, `target_prompt_tokens`, `prompt_token_ratio`,
`projected_prompt_tokens_after`, and the next request's real prompt tokens from OCX.
