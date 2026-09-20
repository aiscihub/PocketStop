"""Acceptance tests from NEXT_ACTIONS.md section 4. Plain-assert script, no
pytest dependency:

    venv/bin/python -m pocketpilot.tests.test_replay_parity

`compare/config.py` hardcodes REPO_ROOT for the machine that holds the raw
trajectories; `_patch_repo_root` rewrites the derived CSV paths onto
whatever machine this actually runs on, in memory only, so the frozen
config file itself is never edited.
"""
from __future__ import annotations

import math
import shutil
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from .. import AI2SCI_P4L3  # triggers the package's sys.path bootstrap for `compare`
from compare import config as C
from compare import data as D

from .. import events as EV
from .. import export_bundles as EX
from .. import replay as R
from ..policy_bundle import BundleTypeError, load_bundle
from ..prefix_features import FeatureContractError, PrefixCursor, PrefixViolation

RESULTS_DIR = AI2SCI_P4L3 / "results" / "p4_l3_100ns_v1"


def _patch_repo_root() -> None:
    actual_root = str(AI2SCI_P4L3.parent)
    if C.REPO_ROOT != actual_root:
        C.P4_FEATURES_CSV = C.P4_FEATURES_CSV.replace(C.REPO_ROOT, actual_root)
        C.L3_FEATURES_CSV = C.L3_FEATURES_CSV.replace(C.REPO_ROOT, actual_root)
        C.MANIFEST_CSV = C.MANIFEST_CSV.replace(C.REPO_ROOT, actual_root)
        C.REPO_ROOT = actual_root


def _ground_truth_decisions() -> pd.DataFrame:
    decisions = pd.read_csv(RESULTS_DIR / "decisions.csv")
    decisions = decisions[
        (decisions["family"] == EX.BLOCK)
        & (decisions["checkpoint_ns"] == EX.CHECKPOINT_NS)
        & (decisions["alpha"].astype(float) == EX.ALPHA)
    ]
    return decisions.set_index("run_uid")


class Fixture:
    def __init__(self):
        _patch_repo_root()
        self.rows, _ = D.load_common_support()
        self.cp = self.rows[self.rows["time_ns"] == EX.CHECKPOINT_NS].reset_index(drop=True)
        self.cols = C.FEATURE_BLOCKS[EX.BLOCK]
        self.manifest_hash = "test-manifest-hash-not-used-for-provenance"
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="pocketpilot_bundles_"))
        self.historical = EX.export_historical_replay_bundles(
            self.tmp_dir, self.cp, self.cols, self.manifest_hash
        )
        self.deployment = EX.export_deployment_bundle(
            self.tmp_dir, self.cp, self.cols, self.manifest_hash
        )

    def full_rows_for(self, run_uid: str) -> pd.DataFrame:
        return self.rows[self.rows["run_uid"] == run_uid].copy()

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def test_export_produced_bundles_and_hashes_verify(fx: Fixture):
    assert fx.historical, "expected at least one historical_replay bundle"
    for path in list(fx.historical.values()) + [fx.deployment]:
        bundle = load_bundle(path)  # raises BundleIntegrityError on mismatch
        assert bundle.schema_version


def test_replay_matches_frozen_scores_and_decisions(fx: Fixture):
    truth = _ground_truth_decisions()
    mismatches = []
    for held, path in fx.historical.items():
        bundle = load_bundle(path)
        outcomes = R.replay_cohort(bundle, fx.cp)
        for _, row in outcomes.iterrows():
            expected = truth.loc[row["run_uid"]]
            expected_stop = bool(expected["stop"])
            got_stop = row["state"] == R.STOP_CANDIDATE
            if got_stop != expected_stop:
                mismatches.append((row["run_uid"], "decision", expected_stop, got_stop))
            if not math.isclose(row["score"], float(expected["score"]), rel_tol=1e-6, abs_tol=1e-9):
                mismatches.append((row["run_uid"], "score", expected["score"], row["score"]))
            # thresholds.csv round-trips the threshold through text, which can
            # lose the last ULP of a float64; compare at float64 precision,
            # not by re-deriving the CSV's own text representation.
            if not math.isclose(row["threshold"], float(expected["threshold"]), rel_tol=1e-12, abs_tol=1e-15):
                mismatches.append((row["run_uid"], "threshold", expected["threshold"], row["threshold"]))
    assert not mismatches, f"replay diverged from frozen decisions: {mismatches[:5]}"


def test_future_suffix_cannot_change_earlier_decision(fx: Fixture):
    run_uid = fx.cp["run_uid"].iloc[0]
    rows = fx.full_rows_for(run_uid)
    cols = tuple(fx.cols)

    cursor_a = PrefixCursor(run_uid, rows, cols)
    obs_a = cursor_a.advance_to(8.0)

    mutated = rows.copy()
    mutated.loc[mutated["time_ns"] > 8.0, list(cols)] = -999.0
    cursor_b = PrefixCursor(run_uid, mutated, cols)
    obs_b = cursor_b.advance_to(8.0)

    assert obs_a.values == obs_b.values, "mutating future frames changed an earlier observation"


def test_missing_or_nan_feature_defers_not_stops(fx: Fixture):
    run_uid = fx.cp["run_uid"].iloc[0]
    rows = fx.cp[fx.cp["run_uid"] == run_uid].copy()
    rows.loc[:, fx.cols[0]] = float("nan")
    cursor = PrefixCursor(run_uid, rows, tuple(fx.cols))
    try:
        cursor.advance_to(8.0)
        raise AssertionError("expected FeatureContractError on NaN feature")
    except FeatureContractError:
        pass


def test_missing_checkpoint_frame_defers(fx: Fixture):
    run_uid = fx.cp["run_uid"].iloc[0]
    rows = fx.cp[fx.cp["run_uid"] == run_uid].copy()
    rows = rows[rows["time_ns"] != 8.0]
    cursor = PrefixCursor(run_uid, rows, tuple(fx.cols))
    try:
        cursor.advance_to(8.0)
        raise AssertionError("expected FeatureContractError on missing frame")
    except FeatureContractError:
        pass


def test_duplicate_events_cannot_create_duplicate_decisions(fx: Fixture):
    log = EV.EventLog(fx.tmp_dir / "events.jsonl")
    first = log.record_recommendation("run1", 8.0, "STOP_CANDIDATE", score=0.9, threshold=0.5)
    second = log.record_recommendation("run1", 8.0, "STOP_CANDIDATE", score=0.9, threshold=0.5)
    assert first is True and second is False
    assert len(log.read_all()) == 1


def test_restart_restores_prefix_state(fx: Fixture):
    run_uid = fx.cp["run_uid"].iloc[0]
    rows = fx.full_rows_for(run_uid)
    cols = tuple(fx.cols)

    cursor = PrefixCursor(run_uid, rows, cols)
    cursor.advance_to(2.0)
    cursor.advance_to(4.0)
    state = cursor.state()

    restored = PrefixCursor.restore(rows, cols, state)
    assert restored.checkpoints_seen == [2.0, 4.0]
    try:
        restored.advance_to(4.0)
        raise AssertionError("expected PrefixViolation re-requesting an already-seen checkpoint")
    except PrefixViolation:
        pass
    restored.advance_to(6.0)  # strictly after the restored position: allowed


def test_historical_and_deployment_bundles_cannot_be_confused(fx: Fixture):
    held, hist_path = next(iter(fx.historical.items()))
    hist_bundle = load_bundle(hist_path)
    dep_bundle = load_bundle(fx.deployment)

    try:
        hist_bundle.require_deployment()
        raise AssertionError("historical_replay bundle must refuse require_deployment()")
    except BundleTypeError:
        pass
    try:
        dep_bundle.require_historical_replay()
        raise AssertionError("deployment bundle must refuse require_historical_replay()")
    except BundleTypeError:
        pass
    assert hist_bundle.permitted_run_uids is not None
    assert dep_bundle.permitted_run_uids is None
    assert hist_bundle.held_out_protein == held


TESTS = [
    test_export_produced_bundles_and_hashes_verify,
    test_replay_matches_frozen_scores_and_decisions,
    test_future_suffix_cannot_change_earlier_decision,
    test_missing_or_nan_feature_defers_not_stops,
    test_missing_checkpoint_frame_defers,
    test_duplicate_events_cannot_create_duplicate_decisions,
    test_restart_restores_prefix_state,
    test_historical_and_deployment_bundles_cannot_be_confused,
]


def main() -> int:
    print("building fixture (common-support join + bundle export)...")
    fx = Fixture()
    failures = 0
    try:
        for test in TESTS:
            name = test.__name__
            try:
                test(fx)
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    finally:
        fx.cleanup()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
