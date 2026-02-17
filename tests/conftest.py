from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_IS_CI = bool(os.getenv("CI"))
_SHOW_CUSTOM = True
_TOTAL_TESTS = 0
_CURRENT_TEST = 0


def _emit(message: str) -> None:
    # Write to the real stdout to avoid pytest capture hiding progress output.
    print(message, file=sys.__stdout__, flush=True)


def pytest_configure(config):
    global _SHOW_CUSTOM
    if _IS_CI:
        _SHOW_CUSTOM = False
        return
    verbose = getattr(config.option, "verbose", 0) or 0
    _SHOW_CUSTOM = verbose < 1


def pytest_collection_modifyitems(session, config, items):
    global _TOTAL_TESTS
    _TOTAL_TESTS = len(items)


def pytest_sessionstart(session):
    _emit("\n========== VBN Test Suite ==========\n")


def pytest_sessionfinish(session, exitstatus):
    _emit("\n========== Test Session Finished ==========\n")


def pytest_runtest_logstart(nodeid, location):
    if not _SHOW_CUSTOM:
        return
    global _CURRENT_TEST
    _CURRENT_TEST += 1
    progress = f" [{_CURRENT_TEST}/{_TOTAL_TESTS}]" if _TOTAL_TESTS else ""
    _emit(f"\n▶ Running: {nodeid}{progress}")


def pytest_runtest_logreport(report):
    if not _SHOW_CUSTOM:
        return
    if report.when == "call":
        outcome = report.outcome.upper()
        if getattr(report, "wasxfail", None):
            outcome = "XFAIL" if report.failed else "XPASS"
        _emit(f"\n[{outcome}][{report.when}] {report.nodeid}")
    elif report.failed:
        _emit(f"\n[FAILED][{report.when}] {report.nodeid}")
    elif report.skipped and report.when != "call":
        _emit(f"\n[SKIPPED][{report.when}] {report.nodeid}")
