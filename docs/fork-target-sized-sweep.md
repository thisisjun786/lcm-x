# Fork design: target-sized threshold sweep and idle-gap pre-compaction (G60)

Status: design only. No code in this change. The sweep code (`compaction.py` `_condense_before_leaves`, the leaf
loop, `_condense_after_leaves`, `config.py`) is being changed by T176 ("leave R passes for condensation"); the
implementation of part 1 lands on top of the T176 commit. Base read: `origin/hermes` at `daa9e995` (T168).

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

### Where the next request's size can be counted

Next request = system prompt and tool schemas + summary head + raw tail (everything after the summaries, fresh tail
included).

- Running estimate, free: `state.estimated_active_tokens`. It starts from the host's observed prompt tokens
  (`compaction.py:2064-2068`), which already include the system prompt and tools, and each stored leaf subtracts
  its source and adds its summary (`compaction.py:3063`; promoted leaves too). Condensation does **not** update it
  today: the implementation must subtract `frontier_before - frontier_after` after each condensation pass
  (`engine.py:9167` `_run_threshold_sweep_condensation` already reads both).
- Exact figure, at the end: `count_messages_tokens(assembled context)` + host overhead. Host overhead (system prompt
  + tools) is `_survival_host_overhead()` (`survival_fit.py:133-144`): observed prompt tokens minus the counted
  messages, the binding's peak, capped at half the window.
- Note: today's `tokens_after` in `lcm_status` (`compaction.py:2772`) is the message list only, without the host
  overhead. The target must compare `tokens_after + host overhead`.

### Proposed behavior

New setting `LCM_SWEEP_TARGET_PROMPT_TOKENS` / `lcm.sweep_target_prompt_tokens` (int, `0` = off = today's
behavior; proposed fleet value 150000 after replay). When it is set and the threshold sweep is active:

1. Pre-leaf condensation as today (#653), within the T176 reserve.
2. Leaves while `estimated_active_tokens > target`, passes left beyond T176's reserve R, and time left. A leaf pass
   that brings the estimate to the target or below stops the leaves with `target_reached`. The fresh tail is never
   cut; the leaf candidate rules are unchanged.
3. Condensation runs when the head is over `sweep_target_tokens` **or** the estimate is still over the target, and
   not only after `raw_prefix_drained`: with the reserved R passes plus any passes the leaves did not use.
4. If the estimate is still over the target, raw history remains, and passes and time remain, go back to step 2 (one
   more round). A round that stores nothing ends the sweep (`target_unreachable`: for example a fresh tail plus
   system prompt larger than the target).
5. Final stop reasons: `target_reached` (completed), `raw_prefix_drained`, `target_unreachable`,
   `pass_budget_exhausted`, `time_budget_exhausted`, and the existing condensation reasons. Add
   `target_prompt_tokens`, `projected_prompt_tokens_after` (exact figure above) and `rounds` to
   `_last_threshold_full_sweep`.

Merge with T176: T176's R is the condensation reserve inside the same total cap. The leaf loop's limit becomes
`cap - pre_leaf_condensation_passes - R`; condensation gets `R + unused leaf passes`. With a target set, the reserve
is spent only when the head or the estimate is over target, so a sweep that reached the target by leaves alone
does not spend it.

### Bounds

- Pass cap: the existing `LCM_THRESHOLD_FULL_SWEEP_MAX_PASSES` (1-64). In target mode it is a ceiling, not a fixed
  count. It must also stay at or below `LCM_SUMMARY_SPEND_MAX_CALLS` (default 24 per 600 s, `config.py:1221`), or the
  spend guard stops the sweep and backs off for 1800 s.
- Time cap: the existing `LCM_FOREGROUND_HARD_SECONDS` (default 120) and `LCM_FOREGROUND_SOFT_SECONDS` (default
  60). The soft bound stops a sweep with `soft_target_reached`; raise both with the pass cap.
- Host limits (Hermes `main` c627ec84, 2026-11-01; the deployed Hermes version was not checked):
  - Compression wrapper: `compression.context_timeout_seconds` (idle, default 120, raised to at least the auxiliary
    compression timeout) and `compression.context_total_ceiling_seconds` (default 600). A compaction still running at
    the ceiling is discarded by the host (`agent/conversation_compression.py:1-8`,
    `agent/conversation_compression_deadline.py:20-60`). LCM summaries already stored stay stored (nothing is lost;
    the next turn resumes), but the turn gets the host's fallback. Keep `LCM_FOREGROUND_HARD_SECONDS` well under 600.
  - Gateway turn inactivity watchdog: `agent.gateway_timeout` 1800 s (`gateway/run.py:2723`), warning at 900 s. The
    host's compression heartbeat touches activity every 60 s during a compaction, so a sweep under 600 s does not trip
    it.
  - turn-guard (hub plugin) uses only `transform_tool_result` and `pre_llm_call` hooks (hermes-ops
    `docs/design.md:346`); its source is on hub-mac and was not read here.
- Proposed bounds for replay: cap 24, soft 180 s, hard 240 s.

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

Arms (same session, start at 0.7 x window = 700k):

| Arm | `LCM_SWEEP_TARGET_PROMPT_TOKENS` | `LCM_THRESHOLD_FULL_SWEEP_MAX_PASSES` | `LCM_FOREGROUND_SOFT_SECONDS` / `HARD` |
|---|---|---|---|
| base | 0 | 14 | fleet values |
| t120 | 120000 | 24 | 180 / 240 |
| t150 | 150000 | 24 | 180 / 240 |
| t200 | 200000 | 24 | 180 / 240 |

Keep `LCM_SUMMARY_SPEND_MAX_CALLS` >= 24 for the target arms.

Fields from `lcm_status` `threshold_full_sweep`: `stop_reason`, `leaf_passes`, `condensation_passes`,
`pre_leaf_condensation_passes`, `total_passes`, `duration_ms`, `tokens_before`, `tokens_after`,
`summary_prefix_tokens_before`, `summary_prefix_tokens_after`, `summary_prefix_target_tokens`, and after part 1
`target_prompt_tokens`, `projected_prompt_tokens_after`, `rounds`. Summary calls and tokens: the summariser call
log of the replay tool (count, input and output tokens per call).
