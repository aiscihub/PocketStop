"""Append-only event log (NEXT_ACTIONS.md sections 4 and 6).

Every observation, recommendation, and execution receipt is keyed by a
deterministic id from (run_uid, checkpoint_ns, kind). A duplicate append is
detected and dropped before it is written, so replaying the same event
twice -- a retried call, a restarted process re-emitting its last step --
cannot create a duplicate decision downstream.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _event_id(run_uid: str, checkpoint_ns: float, kind: str) -> str:
    return hashlib.sha256(f"{run_uid}|{checkpoint_ns}|{kind}".encode()).hexdigest()[:16]


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._seen_ids: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._seen_ids.add(json.loads(line)["event_id"])

    def _append(self, kind: str, run_uid: str, checkpoint_ns: float, **fields) -> bool:
        event_id = _event_id(run_uid, checkpoint_ns, kind)
        if event_id in self._seen_ids:
            return False
        record = {
            "event_id": event_id, "kind": kind, "run_uid": run_uid,
            "checkpoint_ns": checkpoint_ns, **fields,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._seen_ids.add(event_id)
        return True

    def record_observation(self, run_uid: str, checkpoint_ns: float, values, error: str | None = None) -> bool:
        return self._append("observation", run_uid, checkpoint_ns, values=values, error=error)

    def record_recommendation(self, run_uid: str, checkpoint_ns: float, state: str,
                               score: float | None, threshold: float) -> bool:
        return self._append("recommendation", run_uid, checkpoint_ns, state=state,
                             score=score, threshold=threshold)

    def record_execution_receipt(self, run_uid: str, checkpoint_ns: float, action: str,
                                  detail: dict | None = None) -> bool:
        return self._append("execution_receipt", run_uid, checkpoint_ns, action=action, detail=detail)

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
