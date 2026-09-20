"""Sequential replay engine (NEXT_ACTIONS.md sections 4-5).

PocketStop v0.1 has exactly one active decision checkpoint per run (section
2). Replay reflects that: it exposes the run's prefix only up through that
checkpoint and issues exactly one decision. "Passing the first checkpoint
without a stop means continuing to the planned endpoint, not applying an
8 ns model at every later frame" (section 5) -- there is no later frame to
apply it to here, by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import events as EV
from .policy_bundle import PolicyBundle
from .prefix_features import FeatureContractError, PrefixCursor

CONTINUE_UNRESOLVED = "CONTINUE_UNRESOLVED"
STOP_CANDIDATE = "STOP_CANDIDATE"
DEFER_MODEL = "DEFER_MODEL"


@dataclass
class ReplayOutcome:
    run_uid: str
    checkpoint_ns: float
    state: str
    score: float | None
    threshold: float
    y: int | None


def replay_run(bundle: PolicyBundle, run_uid: str, rows: pd.DataFrame,
                log: EV.EventLog | None = None) -> ReplayOutcome:
    """Expose `rows` up to `bundle.checkpoint_ns` and issue one decision.

    `rows` may contain frames past the checkpoint (it is a slice of an
    already-materialized historical table); PrefixCursor guarantees they are
    never read to reach this decision.
    """
    cursor = PrefixCursor(run_uid, rows, bundle.feature_order)
    y = int(rows["label_unstable"].iloc[0]) if "label_unstable" in rows.columns and len(rows) else None

    try:
        obs = cursor.advance_to(bundle.checkpoint_ns)
    except FeatureContractError as e:
        if log:
            log.record_observation(run_uid, bundle.checkpoint_ns, None, error=str(e))
            log.record_recommendation(run_uid, bundle.checkpoint_ns, DEFER_MODEL,
                                       score=None, threshold=bundle.threshold)
        return ReplayOutcome(run_uid, bundle.checkpoint_ns, DEFER_MODEL, None, bundle.threshold, y)

    if log:
        log.record_observation(run_uid, bundle.checkpoint_ns, obs.values)

    state, score = bundle.decide(obs.values)
    if log:
        log.record_recommendation(run_uid, bundle.checkpoint_ns, state, score=score,
                                   threshold=bundle.threshold)
    return ReplayOutcome(run_uid, bundle.checkpoint_ns, state, score, bundle.threshold, y)


def replay_cohort(bundle: PolicyBundle, common_support: pd.DataFrame,
                   log: EV.EventLog | None = None) -> pd.DataFrame:
    """Replay every run the bundle is permitted to score."""
    if bundle.artifact_type == "historical_replay":
        bundle.require_historical_replay()
        eligible = common_support[common_support["run_uid"].isin(bundle.permitted_run_uids)]
    else:
        bundle.require_deployment()
        eligible = common_support

    outcomes = [
        replay_run(bundle, run_uid, rows, log=log)
        for run_uid, rows in eligible.groupby("run_uid")
    ]
    return pd.DataFrame([o.__dict__ for o in outcomes])
