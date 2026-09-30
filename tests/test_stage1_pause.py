"""Exercise seed 0 pause/resume control flow without any GPU or real COCO assets."""
import json
import sys
from dataclasses import replace

import pytest
import torch

from coco_kd import models, stage1, utils
from coco_kd.config import Config
from coco_kd.synthetic import synthetic_data
from coco_kd.train import train, run_path
from scripts import check_teacher_outputs, plan_stage1


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    # main() changes cwd; restore it automatically after each CLI test.
    monkeypatch.chdir(tmp_path)
    events, completed, training = [], set(), []
    scores = dict(zip(stage1.STUDENTS, (60., 59., 60., 62., 61.5)))
    rows = {name: {"mean_pp": score, "initial_model_sha256": stage1.INITIAL_SHA256}
            for name, score in scores.items()}
    monkeypatch.setattr(plan_stage1, "ROOT", tmp_path)
    monkeypatch.setattr(plan_stage1, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(plan_stage1, "require_tests", lambda: None)
    monkeypatch.setattr(plan_stage1, "check_assets", lambda: {"synthetic_fixture": True})
    monkeypatch.setattr(plan_stage1, "validation_fingerprint", lambda: "fixture")
    monkeypatch.setattr(plan_stage1, "config_audit", lambda: None)
    monkeypatch.setattr(plan_stage1, "read_config", stage1.recipe)
    monkeypatch.setattr(plan_stage1, "read_student_rows", lambda seed: rows)
    monkeypatch.setattr(plan_stage1, "summary", lambda: events.append("summary"))
    monkeypatch.setattr(plan_stage1, "require_teacher_checks", lambda seed: None)
    monkeypatch.setattr(check_teacher_outputs, "check_teachers", lambda seed: events.append(("check", seed)))
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic GPU fixture")
    monkeypatch.setattr(utils, "setup_device", lambda cfg: None)
    monkeypatch.setattr(models, "build_model", lambda *args, **kwargs: object())

    def batch(seed, names):
        events.append(("batch", seed, tuple(names)))
        for name in names:
            if (seed, name) not in completed:
                training.append((seed, name))
                completed.add((seed, name))

    monkeypatch.setattr(plan_stage1, "run_batch", batch)
    return events, rows, training


@pytest.mark.parametrize("selected", ["LS0_T1", "LS0_T4"])
def test_cli_seed0_pause_is_successful_and_resume_starts_only_new_seed1_runs(pipeline, monkeypatch, capsys, selected):
    events, rows, training = pipeline
    rows[selected]["mean_pp"] = 63.
    monkeypatch.setattr(sys, "argv", ["plan_stage1.py", "run", "--jobs", "5", "--stop-after-seed0"])
    assert plan_stage1.main() is None  # normal CLI return => exit 0
    assert f"SEED0_DONE: RUN_SEED1, selected={selected.removeprefix('LS0_')}" in capsys.readouterr().out
    assert events[-1] == "summary"
    assert events[:3] == [("batch", 0, stage1.TEACHERS), ("check", 0), ("batch", 0, stage1.STUDENTS)]
    assert all(seed == 0 for seed, _ in training)
    assert len(training) == 7
    assert not (plan_stage1.OUTPUT / "STOP.json").exists()
    # Same source and configs; the option is not persisted into any run config.
    snapshot = list(training)
    monkeypatch.setattr(sys, "argv", ["plan_stage1.py", "run", "--jobs", "5"])
    assert plan_stage1.main() is None
    assert training[:7] == snapshot
    assert training[7:] == [(1, "T_LS0"), (1, "R1_ce"), (1, selected)]
    assert "FINAL_PASS" in capsys.readouterr().out
    assert not (plan_stage1.OUTPUT / "STOP.json").exists()


@pytest.mark.parametrize("decision", ["STOP_SEED0", "STOP_REGRESSION"])
def test_seed0_pause_preserves_stop_file_and_failure_exit(pipeline, monkeypatch, capsys, decision):
    events, rows, training = pipeline
    if decision == "STOP_SEED0":
        rows["LS0_T1"]["mean_pp"] = rows["LS0_T4"]["mean_pp"] = 61.
    else:
        rows["R2_full_LS01best_T1"]["mean_pp"] = 55.
    monkeypatch.setattr(sys, "argv", ["plan_stage1.py", "run", "--stop-after-seed0"])
    with pytest.raises(RuntimeError, match=decision):
        plan_stage1.main()
    record = json.loads((plan_stage1.OUTPUT / "STOP.json").read_text())
    assert record["decision"] == decision
    assert "summary" in events and all(seed == 0 for seed, _ in training)
    assert "SEED0_DONE" not in capsys.readouterr().out


def test_completed_run_reuse_skips_training_and_preserves_files(tmp_path, monkeypatch):
    import coco_kd.train as training
    synthetic_data(tmp_path / "data")
    cfg = Config(teacher_label_smoothing=0., label_smoothing=.1, data_root=str(tmp_path / "data"),
                 output_root=str(tmp_path / "runs"), device="cpu", model_scale="debug", num_classes=2,
                 student_init="scratch", num_workers=0, batch_size=2, eval_batch_size=2,
                 accumulation_steps=2, epochs=1, warmup_epochs=0, probe_enabled=False)
    checkpoint = train(cfg, method="ce")
    directory = run_path(cfg, method="ce")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in directory.iterdir() if p.is_file() and p.name != ".run.lock"}
    monkeypatch.setattr(training, "train_epoch", lambda *args, **kwargs: pytest.fail("Completed run retrained"))
    assert train(replace(cfg), method="ce") == checkpoint
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
             for p in directory.iterdir() if p.is_file() and p.name != ".run.lock"}
    assert after == before
