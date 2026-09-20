# PocketStop

A Risk-Aware AI Framework for Early Molecular-Dynamics Screening.

**Research question:** Can early molecular dynamics identify runs that will fail a 100 ns screening endpoint without prematurely rejecting runs that would pass?

![PocketStop architecture and shadow-run interface](figures/pocketstop.png)

## AI Innovation

Risk-aware AI learns from early ligand and pocket motion, calibrates stop/continue recommendations to a requested false-stop target, and evaluates computational savings from correct stops alongside preservation of endpoint-passing trajectories.

## Methodology

A versioned policy interface for risk-aware early-stopping decisions on molecular-dynamics screening trajectories, plus a historical-prefix replay engine for testing that a fitted model's retrospective evaluation and its deployed decision path agree exactly: historical-replay and deployment policy bundles, a checkpoint-bounded prefix reader, a sequential replay engine, and an append-only event log. **No OpenMM integration, no live pilot.** Those are separate, later work that needs a real simulation environment and explicit sign-off this package does not assume.

## Key Contributions

- **Risk-aware AI:** stop-or-continue decisions under a specified false-stop target, audited on held-out proteins.
- **Reproducible recommendations:** versioned policies preserve features, checkpoint, model, threshold, and decision provenance.

## Usage

From this repo's root:

```bash
# regenerate policy bundles (historical + deployment) from the frozen L3/8ns/alpha=0.10 spec
venv/bin/python -m pocketstop.export_bundles --out-dir pocketstop_bundles

# run the replay-parity acceptance tests (plain-assert script, not pytest)
venv/bin/python -m pocketstop.tests.test_replay_parity
```

As of 2026-09-15, the replay-parity suite passes 8/8: batch/prefix score parity, exact decision agreement (including threshold ties), invariance to later trajectory frames, deferral (not a model-driven stop) on missing or NaN features and on a missing checkpoint frame, idempotent handling of duplicate events, preserved state across a restart, and no confusion between historical and deployment bundles.

**Evidence status:** implemented and replay-tested against historical data. Not live-tested — no simulation has been run under PocketStop's control. See `valleyfevermutation`'s `ai2sci_p4l3/artifacts/corrected_release/2026-09-15_corrected_v1/CLAIM_EVIDENCE.md` for the full evidence-status writeup this repo's tests fed into.
