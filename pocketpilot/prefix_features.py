"""Checkpoint-bounded, order-enforced feature access for replay (NEXT_ACTIONS.md
section 4).

`PrefixCursor` is built from a run's fully materialized historical rows (this
is a *replay* adapter over an already-frozen feature table, not the online
extractor of section 5), but it never discloses a row other than the one
requested, and it refuses out-of-order or repeated requests. Combined, this
means a later checkpoint's data cannot influence an earlier decision no
matter what the caller does with the DataFrame afterward -- the guarantee is
structural, not a documented promise.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class FeatureContractError(Exception):
    """Wrong feature order/units/reference/identity, or a missing/NaN frame.

    Callers must map this to DEFER_MODEL, never to a silent stop.
    """


class PrefixViolation(Exception):
    """Requested a checkpoint not strictly after the last one observed."""


@dataclass(frozen=True)
class FeatureObservation:
    run_uid: str
    checkpoint_ns: float
    values: dict


class PrefixCursor:
    def __init__(self, run_uid: str, rows: pd.DataFrame, feature_order: tuple[str, ...]):
        self.run_uid = run_uid
        self._feature_order = tuple(feature_order)
        self._by_checkpoint = {
            float(r["time_ns"]): r for _, r in rows.sort_values("time_ns").iterrows()
        }
        self.checkpoints_seen: list[float] = []

    def advance_to(self, checkpoint_ns: float) -> FeatureObservation:
        checkpoint_ns = float(checkpoint_ns)
        if self.checkpoints_seen and checkpoint_ns <= self.checkpoints_seen[-1]:
            raise PrefixViolation(
                f"{self.run_uid}: checkpoint {checkpoint_ns} is not after last "
                f"observed {self.checkpoints_seen[-1]}"
            )
        if checkpoint_ns not in self._by_checkpoint:
            raise FeatureContractError(
                f"{self.run_uid}: no frame recorded at checkpoint {checkpoint_ns} ns"
            )
        row = self._by_checkpoint[checkpoint_ns]
        missing = [c for c in self._feature_order if c not in row or pd.isna(row[c])]
        if missing:
            raise FeatureContractError(
                f"{self.run_uid}@{checkpoint_ns}ns: missing/NaN features {missing}"
            )
        values = {c: float(row[c]) for c in self._feature_order}
        self.checkpoints_seen.append(checkpoint_ns)
        return FeatureObservation(self.run_uid, checkpoint_ns, values)

    def state(self) -> dict:
        """Resumable state: exactly what has been observed, nothing derived."""
        return {"run_uid": self.run_uid, "checkpoints_seen": list(self.checkpoints_seen)}

    @classmethod
    def restore(cls, rows: pd.DataFrame, feature_order: tuple[str, ...], state: dict) -> "PrefixCursor":
        cursor = cls(state["run_uid"], rows, feature_order)
        for checkpoint_ns in state["checkpoints_seen"]:
            cursor.advance_to(checkpoint_ns)
        return cursor
