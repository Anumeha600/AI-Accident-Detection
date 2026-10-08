"""
Shared test setup.

The project is a flat set of scripts that use CWD-relative paths (e.g.
"accident_classifier_model_expanded.joblib", "expanded_synthetic_dataset.csv"),
so tests run from the project root and put it on sys.path.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _run_from_project_root(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def _isolated_recordings_dir(monkeypatch, tmp_path_factory):
    """Phase 10: app.py records runs; keep tests from writing into the project's recordings/ folder."""
    monkeypatch.setenv("ACCIDENT_RECORDINGS_DIR", str(tmp_path_factory.mktemp("recordings")))
