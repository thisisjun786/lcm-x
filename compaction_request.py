"""One-shot preflight instructions consumed by the next foreground compaction."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional


def system_prompt_sha(messages: Any) -> Optional[str]:
    """Fork cold cache (lina#455): sha256 of the canonical JSON of the leading system message(s)' content (a
    string or a list of blocks), or None when the request starts with no system message. Reads only those rows."""
    contents = []
    for message in messages or ():
        if not isinstance(message, dict) or message.get("role") != "system":
            break
        contents.append(message.get("content"))
    if not contents:
        return None
    canonical = json.dumps(contents, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()


# Fork cold cache (lina#455): per conversation, the latest (provider-reported prompt tokens, the engine's own count of
# the request) pair of each route (provider/model) that answered, most recent last, in the telemetry record.
ROUTE_CALIBRATION_FIELD = "route_prompt_calibration"
ROUTE_CALIBRATION_LIMIT = 8


def route_key(provider: Any, model: Any) -> Optional[str]:
    """The calibration key of a route; None without a model."""
    return f"{provider or ''}/{model}" if model else None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def route_calibration_entries(record: Any) -> list:
    """The record's well-formed ``[key, reported, counted]`` entries (a missing or malformed field reads empty)."""
    raw = record.get(ROUTE_CALIBRATION_FIELD) if isinstance(record, dict) else None
    if not isinstance(raw, list):
        return []
    return [[item[0], item[1], item[2]] for item in raw
            if isinstance(item, list) and len(item) == 3 and isinstance(item[0], str) and item[0]
            and _positive_int(item[1]) and _positive_int(item[2])][-ROUTE_CALIBRATION_LIMIT:]


def updated_route_calibration(record: Any, key: Optional[str], reported: Any, counted: Any) -> list:
    """The entries with ``key``'s pair replaced by this call's and moved last; other routes' entries stay."""
    entries = route_calibration_entries(record)
    if key and _positive_int(reported) and _positive_int(counted):
        entries = [entry for entry in entries if entry[0] != key] + [[key, reported, counted]]
    return entries[-ROUTE_CALIBRATION_LIMIT:]


def route_calibration(record: Any, key: Optional[str]) -> Optional[tuple]:
    """``(reported, counted)`` of ``key``'s latest call in this conversation, else None."""
    for entry_key, reported, counted in reversed(route_calibration_entries(record)):
        if key and entry_key == key:
            return reported, counted
    return None


@dataclass
class CompactionRequest:
    """Pending handoffs; each entry point retains its existing reset policy."""

    # Adopt already-durable replay cleanup during boundary cooldown without summary work.
    cleanup_only_due_to_boundary_cooldown: bool = False
    # #651: preflight requested maintenance below the host threshold, so only clean up.
    below_threshold_cleanup_only: bool = False
    # #677: any preflight request; compress() cleans up only below the host's count.
    automatic_request: bool = False
    # #618: a hold is active at the survival ceiling, so this pass only fits.
    hold_fit_only_requested: bool = False
    # Fork: preflight found a cold gap (idle past cold_gap_compaction_seconds) below the threshold; compress() runs
    # the threshold full sweep instead of below-threshold cleanup, and arms no #651 hold.
    cold_gap: bool = False
