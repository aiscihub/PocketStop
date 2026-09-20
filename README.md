# PocketPilot

A versioned policy interface for risk-aware early-stopping decisions on molecular-dynamics screening trajectories, plus a historical-prefix replay engine for testing that a fitted model's retrospective evaluation and its deployed decision path agree exactly.

Renamed from an earlier internal name, "PocketStop," after finding that name already in use by an unrelated third-party project.

## Scope

Historical-replay and deployment policy bundles, a checkpoint-bounded prefix reader, a sequential replay engine, and an append-only event log. **No OpenMM integration, no live pilot.** Those are separate, later work that needs a real simulation environment and explicit sign-off this package does not assume.

## Dependency on `valleyfevermutation`

This package is developed against, and its tests depend on, the sibling `valleyfevermutation` repository's `ai2sci_p4l3` pipeline — specifically the `compare` package (feature/label join, the fixed model specification, calibration) and its already-computed `results/p4_l3_100ns_v1/` comparison outputs, which this package's replay path treats as ground truth. That dependency is **not vendored**.

By default, `pocketpilot/__init__.py` looks for it at `../valleyfevermutation/ai2sci_p4l3`, relative to this repo's own root — i.e. the two repos are expected to be checked out as siblings:

```
remotegit/
├── valleyfevermutation/
│   └── ai2sci_p4l3/
│       ├── compare/
│       └── results/p4_l3_100ns_v1/
└── PocketPilot/
    └── pocketpilot/
```

Set `POCKETPILOT_AI2SCI_P4L3` to override this if that layout doesn't hold on a given machine.

## Usage

From this repo's root:

```bash
# regenerate policy bundles (historical + deployment) from the frozen L3/8ns/alpha=0.10 spec
venv/bin/python -m pocketpilot.export_bundles --out-dir pocketpilot_bundles

# run the replay-parity acceptance tests (plain-assert script, not pytest)
venv/bin/python -m pocketpilot.tests.test_replay_parity
```

As of 2026-09-15, the replay-parity suite passes 8/8: batch/prefix score parity, exact decision agreement (including threshold ties), invariance to later trajectory frames, deferral (not a model-driven stop) on missing or NaN features and on a missing checkpoint frame, idempotent handling of duplicate events, preserved state across a restart, and no confusion between historical and deployment bundles.

**Evidence status:** implemented and replay-tested against historical data. Not live-tested — no simulation has been run under PocketPilot's control. See `valleyfevermutation`'s `ai2sci_p4l3/artifacts/corrected_release/2026-09-15_corrected_v1/CLAIM_EVIDENCE.md` for the full evidence-status writeup this repo's tests fed into.
