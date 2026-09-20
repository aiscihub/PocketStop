"""Numeric-only policy bundle: export, hash-verify, score (NEXT_ACTIONS.md
section 3).

A bundle is plain JSON -- feature order/units, scaler mean/scale, logistic
coefficients/intercept, threshold -- never a pickled or otherwise executable
model. "A numeric model representation avoids accepting arbitrary executable
uploaded models" (section 3): `PolicyBundle.score` recomputes the logistic
score by hand from those numbers, it never deserializes or calls into an
uploaded object.

Two artifact types (section 3), never confusable at the type level:
  - "historical_replay": one outer-fold's fitted model + threshold, scoped
    to its own held-out protein's runs. Reproduces held-out historical
    decisions; not a deployment policy.
  - "deployment": the fixed estimator refit on all designated development
    data, with a threshold selected once on pooled leave-one-protein-out
    calibration scores. Its own training-trajectory scores are not a
    held-out performance estimate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "pocketpilot-bundle-v1"

_TUPLE_FIELDS = (
    "feature_order", "scaler_mean", "scaler_scale", "coefficients",
    "known_limitations", "permitted_run_uids", "numerical_parity_examples",
)


class BundleError(Exception):
    """Base error for bundle construction, loading, or scoring."""


class BundleIntegrityError(BundleError):
    """Stored content_hash does not match the bundle's own fields."""


class BundleTypeError(BundleError):
    """Operation requires a different artifact_type than this bundle has."""


@dataclass(frozen=True)
class PolicyBundle:
    schema_version: str
    artifact_type: str  # "historical_replay" | "deployment"
    policy_id: str
    model_id: str
    training_manifest_hash: str
    feature_order: tuple[str, ...]
    feature_units: dict
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    positive_class_meaning: str
    threshold: float
    comparison_operator: str
    tie_rule: str
    reference_convention: str
    sampling_convention: str
    endpoint_version: str
    measurement_version: str
    checkpoint_ns: float
    planned_duration_ns: float
    intended_domain: str
    known_limitations: tuple[str, ...]
    allowed_mode: str
    tested_environment: dict
    numerical_parity_examples: tuple[dict, ...]
    held_out_protein: str | None = None
    permitted_run_uids: tuple[str, ...] | None = None
    calibration_note: str = ""

    def content_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def score(self, feature_vector: dict) -> float:
        missing = [f for f in self.feature_order if f not in feature_vector]
        if missing:
            raise BundleError(f"missing features: {missing}")
        logit = self.intercept
        for name, mean, scale, coef in zip(
            self.feature_order, self.scaler_mean, self.scaler_scale, self.coefficients
        ):
            z = (feature_vector[name] - mean) / scale if scale != 0 else 0.0
            logit += coef * z
        return 1.0 / (1.0 + math.exp(-logit))

    def decide(self, feature_vector: dict) -> tuple[str, float]:
        if self.comparison_operator != ">=":
            raise BundleError(f"unsupported comparison operator {self.comparison_operator!r}")
        s = self.score(feature_vector)
        stop = s >= self.threshold  # tie_rule: score == threshold recommends stop
        return ("STOP_CANDIDATE" if stop else "CONTINUE_UNRESOLVED"), s

    def require_deployment(self) -> None:
        if self.artifact_type != "deployment":
            raise BundleTypeError(
                f"this operation requires a deployment bundle, got artifact_type={self.artifact_type!r}"
            )

    def require_historical_replay(self) -> None:
        if self.artifact_type != "historical_replay":
            raise BundleTypeError(
                "this operation requires a historical_replay bundle, "
                f"got artifact_type={self.artifact_type!r}"
            )


def hash_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_bundle(bundle: PolicyBundle, path: str | Path) -> str:
    payload = asdict(bundle)
    content_hash = bundle.content_hash()
    payload["content_hash"] = content_hash
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return content_hash


def load_bundle(path: str | Path) -> PolicyBundle:
    payload = json.loads(Path(path).read_text())
    stored_hash = payload.pop("content_hash", None)
    for key in _TUPLE_FIELDS:
        if payload.get(key) is not None:
            payload[key] = tuple(payload[key])
    bundle = PolicyBundle(**payload)
    if stored_hash is None or stored_hash != bundle.content_hash():
        raise BundleIntegrityError(
            f"hash mismatch loading {path}: file may be corrupted or hand-edited"
        )
    return bundle
