"""First coding ticket (NEXT_ACTIONS.md section 9): the policy exporter.

Builds both artifact types from section 3, for the frozen L3 / 8 ns /
nominal alpha=0.10 spec section 2 designates as the first engineering path
(development-informed choice, not a claim that it is optimal). Reuses
`compare/` for the join, the fixed model spec, and the documented
calibration rule rather than re-deriving any of them -- this exporter must
match the frozen analysis, not produce a second implementation of it.

Usage (from this repo's root, with valleyfevermutation checked out as a sibling):
    venv/bin/python -m pocketstop.export_bundles --out-dir pocketstop_bundles
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import sklearn

from compare import config as C
from compare import data as D
from compare import evaluate as E
from compare import model as M
from compare import thresholding as TH

from .policy_bundle import SCHEMA_VERSION, PolicyBundle, hash_file, save_bundle

BLOCK = "L3"
CHECKPOINT_NS = 8.0
ALPHA = 0.10

KNOWN_LIMITATIONS_COMMON = (
    "L3 self-aligns the ligand: it measures internal conformational "
    "deviation, not translation or reorientation relative to the receptor "
    "(README.md / HANDOFF.md section 3).",
    "Scores are model scores, not calibrated percent risks; calibration "
    "has not been independently evaluated.",
    "Endpoint label provenance: historical scratch drivers for the 100ns "
    "labels are not hash-verified (compare/config.py T_NS comment).",
)


def _tested_environment() -> dict:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }


def _numeric_parity_examples(cp, protein: str, cols: list[str], pipeline, n: int = 3) -> tuple[dict, ...]:
    sample = cp[cp["protein"] == protein].head(n)
    if not len(sample):
        return ()
    proba = pipeline.predict_proba(sample[cols].to_numpy(dtype=float))[:, 1]
    examples = []
    for (_, row), p in zip(sample.iterrows(), proba):
        examples.append({
            "run_uid": row["run_uid"],
            "checkpoint_ns": float(row["time_ns"]),
            "feature_values": {c: float(row[c]) for c in cols},
            "expected_score": float(p),
        })
    return tuple(examples)


def _common_fields(cols: list[str], training_manifest_hash: str) -> dict:
    return dict(
        schema_version=SCHEMA_VERSION,
        model_id=f"logreg-C{C.LOGREG_C}-{BLOCK}",
        training_manifest_hash=training_manifest_hash,
        feature_order=tuple(cols),
        feature_units={c: "angstrom_or_angstrom_derived" for c in cols},
        positive_class_meaning=(
            "label_unstable=1 (endpoint not reached by the full-budget outcome); "
            "high score recommends stop"
        ),
        comparison_operator=">=",
        tie_rule="score == threshold recommends STOP (compare/thresholding.py confusion_at)",
        reference_convention="self-aligned ligand heavy-atom RMSD (L3 block); see compare/config.py",
        sampling_convention="checkpoint grid 2,4,...,30 ns per manifest_106runs.csv / config.CHECKPOINTS_NS",
        endpoint_version="label_unstable = f_contact_20ns<=0.35 OR rmsd_late_20ns>=1.0 (100ns cohort)",
        measurement_version=Path(C.L3_FEATURES_CSV).name,
        checkpoint_ns=CHECKPOINT_NS,
        planned_duration_ns=C.T_NS,
        intended_domain="milbemycin / ABC-transporter 100ns cohort, same launches as training",
        allowed_mode="shadow",
        tested_environment=_tested_environment(),
    )


def export_historical_replay_bundles(out_dir: Path, cp, cols: list[str],
                                      training_manifest_hash: str) -> dict[str, Path]:
    result = E.run_learned_block(cp, BLOCK, CHECKPOINT_NS)
    pol = E.apply_policy(result, BLOCK, ALPHA)
    thresholds_by_fold = {t["held_out_group"]: t for t in pol["thresholds"]}

    written = {}
    for held in sorted(cp["protein"].unique()):
        fitted = E._fit_excluding(cp, cols, frozenset({held}), {})
        if fitted.insufficient_training:
            continue  # documented no-stop fold; no numeric model to export
        pipeline = fitted.pipeline
        scaler = pipeline.named_steps["standardscaler"]
        clf = pipeline.named_steps["logisticregression"]
        thr_row = thresholds_by_fold[held]
        permitted = tuple(sorted(cp.loc[cp["protein"] == held, "run_uid"].unique()))

        bundle = PolicyBundle(
            artifact_type="historical_replay",
            policy_id=f"pocketstop-v0.1-{BLOCK}-{int(CHECKPOINT_NS)}ns-alpha{ALPHA}-held_out_{held}",
            scaler_mean=tuple(float(x) for x in scaler.mean_),
            scaler_scale=tuple(float(x) for x in scaler.scale_),
            coefficients=tuple(float(x) for x in clf.coef_[0]),
            intercept=float(clf.intercept_[0]),
            threshold=float(thr_row["threshold"]),
            known_limitations=KNOWN_LIMITATIONS_COMMON + (
                f"Fitted excluding only protein {held}; scores only that "
                "protein's own runs (permitted_run_uids). Not a deployment policy.",
            ),
            numerical_parity_examples=_numeric_parity_examples(cp, held, cols, pipeline),
            held_out_protein=held,
            permitted_run_uids=permitted,
            calibration_note=(
                "threshold selected on pooled protein-cross-fitted calibration "
                "scores within (all data \\ held-out protein); see "
                "compare/evaluate.py apply_policy"
            ),
            **_common_fields(cols, training_manifest_hash),
        )
        path = out_dir / f"historical_{held}.json"
        save_bundle(bundle, path)
        written[held] = path
    return written


def export_deployment_bundle(out_dir: Path, cp, cols: list[str],
                              training_manifest_hash: str) -> Path:
    result = E.run_learned_block(cp, BLOCK, CHECKPOINT_NS)
    pooled_scores = result["oof"]["score"].to_numpy()
    pooled_y = result["oof"]["y"].to_numpy()
    sel = TH.select_threshold(pooled_scores, pooled_y, ALPHA)

    all_X = cp[cols].to_numpy(dtype=float)
    all_y = cp[C.Y_COLUMN].to_numpy(dtype=int)
    all_groups = cp[C.GROUP_COLUMN].to_numpy()
    fitted_all = M.fit(all_X, all_y, all_groups)
    if fitted_all.insufficient_training:
        raise RuntimeError("all-development-data fit is single-class; cannot export a deployment bundle")
    pipeline = fitted_all.pipeline
    scaler = pipeline.named_steps["standardscaler"]
    clf = pipeline.named_steps["logisticregression"]

    bundle = PolicyBundle(
        artifact_type="deployment",
        policy_id=f"pocketstop-v0.1-{BLOCK}-{int(CHECKPOINT_NS)}ns-alpha{ALPHA}-deployment",
        scaler_mean=tuple(float(x) for x in scaler.mean_),
        scaler_scale=tuple(float(x) for x in scaler.scale_),
        coefficients=tuple(float(x) for x in clf.coef_[0]),
        intercept=float(clf.intercept_[0]),
        threshold=float(sel["threshold"]),
        known_limitations=KNOWN_LIMITATIONS_COMMON + (
            "This model is refit on ALL designated development data; its score "
            "on its own training trajectories is NOT a held-out performance "
            "estimate. Held-out performance for this spec (block=L3, "
            "checkpoint=8ns, alpha=0.10) is reported separately in "
            "results/p4_l3_100ns_v1/metrics.csv.",
        ),
        numerical_parity_examples=_numeric_parity_examples(cp, cp["protein"].iloc[0], cols, pipeline),
        held_out_protein=None,
        permitted_run_uids=None,
        calibration_note=(
            "threshold selected once on pooled leave-one-protein-out "
            "cross-fitted calibration scores across the full development "
            "cohort (compare/thresholding.py select_threshold); NOT averaged "
            "or cherry-picked from individual outer folds"
        ),
        **_common_fields(cols, training_manifest_hash),
    )
    path = out_dir / "deployment.json"
    save_bundle(bundle, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = D.load_common_support()
    cp = rows[rows["time_ns"] == CHECKPOINT_NS].reset_index(drop=True)
    cols = C.FEATURE_BLOCKS[BLOCK]
    training_manifest_hash = hash_file(C.MANIFEST_CSV)

    historical = export_historical_replay_bundles(out_dir, cp, cols, training_manifest_hash)
    deployment = export_deployment_bundle(out_dir, cp, cols, training_manifest_hash)

    print(f"wrote {len(historical)} historical_replay bundles to {out_dir}")
    print(f"wrote deployment bundle to {deployment}")


if __name__ == "__main__":
    main()
