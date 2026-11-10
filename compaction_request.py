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
