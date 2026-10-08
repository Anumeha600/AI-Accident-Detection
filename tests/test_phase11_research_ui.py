"""RESEARCH LAB tab: reads real result files, never shows placeholder numbers, never auto-runs an experiment."""

import json

import pytest

from experiments import experiment_runner as er
from experiments import research_ui as ru
from test_app_smoke import manual_ticks, new_app  # noqa: F401  (manual_ticks is an autouse fixture)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def results_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("p11ui")
    cfg = er.load_config()
    cfg["runs_per_scenario"] = 1
    er.run_experiment(cfg, root, experiment_id="p11_ui_fixture")
    return root


def test_no_results_message_and_warnings_and_nothing_runs_on_load(tmp_path, monkeypatch):
    monkeypatch.setenv(ru.RESULTS_DIR_ENV, str(tmp_path / "none"))
    at = new_app()
    assert not at.exception and "RESEARCH LAB" in [t.label for t in at.tabs]
    assert any(i.value == ru.NO_RESULTS for i in at.info)
    text = " ".join(w.value for w in at.warning)
    assert "SYNTHETIC CONTROLLED EXPERIMENT" in text and "do not establish real-world accident-detection accuracy" in text
    assert "frozen" in text and "not retrained" in text
    assert not (tmp_path / "none").exists()                              # loading the page ran nothing
    assert any(b.label == "RUN EXPERIMENT" for b in at.main.button)


def test_page_shows_the_numbers_from_the_result_files(results_root, monkeypatch):
    monkeypatch.setenv(ru.RESULTS_DIR_ENV, str(results_root))
    at = new_app()
    assert not at.exception and not any(i.value == ru.NO_RESULTS for i in at.info)
    summary = json.loads((results_root / "p11_ui_fixture" / "phase11_summary.json").read_text())
    expected = ru.metrics_table(summary)
    frames = [d.value for d in at.dataframe]
    assert any(list(f.columns) == ["Metric", "ML", "Threshold"] and f.equals(expected) for f in frames)
    assert any("Scenario" in f.columns and len(f) == 8 for f in frames)
    assert any(f.columns[0] == "" and "Median (ms)" in set(f.iloc[:, 0]) for f in frames)
    scen = next(f for f in frames if "Scenario" in f.columns)
    assert any("out-of-distribution" in s for s in scen["Scenario"])
    assert any("Out-of-distribution for the ML model" in w.value and "Rollover" in w.value for w in at.warning)


def test_pure_tables_are_built_only_from_the_summary(results_root):
    summary = ru.load_summary(results_root / "p11_ui_fixture")
    m = ru.metrics_table(summary)
    assert list(m["Metric"]) == ["TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1",
                                 "False positive rate", "False negative rate"]
    rec = summary["summary"]["recording_level"]["ml"]
    assert m.loc[m["Metric"] == "TP", "ML"].item() == str(rec["TP"])
    lat = ru.latency_table(summary)
    assert list(lat.columns) == ["", "ML", "Threshold"] and "Pre-event false alarms" in set(lat[""])
    conf = ru.confusion_tables(summary)
    assert len(conf) == 3 and conf["ML multiclass (window level, model classes only)"].shape == (6, 6)
    assert set(ru.ood_table(summary)["Scenario (no ML class)"]) == {"Rollover", "Multi-Impact Collision"}
    assert ru.list_experiments(results_root)[0].name == "p11_ui_fixture"
    assert ru.list_experiments(results_root / "missing") == []
