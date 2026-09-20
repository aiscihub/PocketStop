"""PocketPilot v0.1 replay path (NEXT_ACTIONS.md sections 3-4).

Scope, deliberately: historical-replay and deployment policy bundles, a
checkpoint-bounded prefix reader, a sequential replay engine, and an
append-only event log. No OpenMM integration, no live pilot, no paper
edits -- those are later, separate tickets (sections 5-8) that need a real
OpenMM environment and human sign-off this package does not assume.

PocketPilot is a standalone repository (renamed from PocketStop, which
collides with an existing third-party project name). It is developed
against, and its tests depend on, the sibling `valleyfevermutation`
repository's `ai2sci_p4l3` pipeline -- specifically the `compare` package
and its already-computed `results/p4_l3_100ns_v1/` outputs, which are this
package's ground truth. That dependency is not vendored; the two repos are
expected to be checked out as siblings (`../valleyfevermutation` relative
to this repo's root). Set POCKETPILOT_AI2SCI_P4L3 to override the location
if that layout does not hold on a given machine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent  # this repo root (parent of the pocketpilot/ package dir)
AI2SCI_P4L3 = Path(os.environ.get(
    "POCKETPILOT_AI2SCI_P4L3",
    str(_REPO_ROOT.parent / "valleyfevermutation" / "ai2sci_p4l3"),
)).resolve()
if str(AI2SCI_P4L3) not in sys.path:
    sys.path.insert(0, str(AI2SCI_P4L3))
