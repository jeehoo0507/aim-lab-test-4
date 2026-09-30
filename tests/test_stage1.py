import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from coco_kd.config import Config
from coco_kd import stage1
from coco_kd.synthetic import synthetic_data
from coco_kd.train import distillation_loss, train_epoch, train, run_path, evaluate_test
from coco_kd.utils import sha256, load_checkpoint, write_json
from scripts.check_teacher_outputs import kl_to_ls, nontarget_entropy, measure


def test_teacher_step_is_unsmoothed_ce_and_student_step_uses_point_one(monkeypatch):
    import coco_kd.train as training
    cfg = Config(teacher_label_smoothing=0.0, label_smoothing=0.1, device="cpu", batch_size=2,
                 accumulation_steps=1, amp=False)
    labels = torch.tensor([0, 3])
    fixed = torch.tensor([[2., -1., 0., 1., -2., 3., 1., 0., -1., 0.],
                          [1., 2., -2., 0., 1., 0., 3., 1., -1., 0.]])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(fixed.clone())

        def forward(self, x, return_attention=False):
            return self.logits, torch.zeros(len(x), 196)

    batch = {"image": torch.zeros(2, 1), "label": labels, "foreground": torch.zeros(2, 196, dtype=torch.bool)}
    monkeypatch.setattr(training, "loader", lambda *args, **kwargs: [batch])
    original = training.distillation_loss
    monkeypatch.setattr(training, "distillation_loss", lambda *args: pytest.fail("Teacher must not call student KD loss"))
    model = Model()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    teacher = train_epoch(model, None, [0, 1], torch.optim.SGD(model.parameters(), lr=0), scaler,
                          cfg, torch.device("cpu"), "teacher", 1, role="teacher")
    assert teacher["loss"] == F.cross_entropy(fixed, labels).item()
    assert teacher["kd"] == 0
    monkeypatch.setattr(training, "distillation_loss", original)
    model = Model()
    student = train_epoch(model, None, [0, 1], torch.optim.SGD(model.parameters(), lr=0), scaler,
                          cfg, torch.device("cpu"), "ce", 1)
    assert student["loss"] == F.cross_entropy(fixed, labels, label_smoothing=0.1).item()
    assert student["loss"] != teacher["loss"]


def test_teacher_ls_is_required_even_if_override_would_supply_it(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"label_smoothing": 0.1}')
    with pytest.raises(ValueError, match="teacher_label_smoothing"):
        Config.load(path)
    with pytest.raises(ValueError, match="teacher_label_smoothing"):
        Config.load(path, teacher_label_smoothing=0.1)
    with pytest.raises(ValueError, match="teacher_label_smoothing"):
        Config.load()
    with pytest.raises(TypeError):
        Config()


@pytest.mark.parametrize("temperature", [1., 4.])
def test_kd_matches_manual_teacher_to_student_batchmean_and_zero(temperature):
    cfg = Config(teacher_label_smoothing=0.0, temperature=temperature, kd_alpha=1.0)
    torch.manual_seed(13)
    student, teacher = torch.randn(3, 10), torch.randn(3, 10)
    labels = torch.tensor([0, 1, 2])
    actual, _, _ = distillation_loss(student, labels, teacher, cfg)
    p = (teacher / temperature).softmax(1)
    expected = temperature**2 * (p * (p.log() - (student / temperature).log_softmax(1))).sum() / 3
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    same, _, _ = distillation_loss(student, labels, student, cfg)
    assert abs(same.item()) < 2e-6


def test_ls_teacher_kd_gradient_equals_ls_ce_failure_mode():
    cfg = Config(teacher_label_smoothing=0.1, temperature=1., label_smoothing=0.1, kd_alpha=1.)
    torch.manual_seed(7)
    labels = torch.tensor([0, 3, 9, 2])
    logits = torch.randn(4, 10, requires_grad=True)
    q = 0.9 * F.one_hot(labels, 10).float() + 0.01
    kd, _, _ = distillation_loss(logits, labels, q.log(), cfg)
    kd_gradient = torch.autograd.grad(kd, logits)[0]
    ce_gradient = torch.autograd.grad(F.cross_entropy(logits, labels, label_smoothing=0.1), logits)[0]
    torch.testing.assert_close(kd_gradient, ce_gradient, atol=2e-8, rtol=1e-5)


def test_output_kl_reference_distributions_and_normalized_entropy():
    labels = torch.tensor([0, 3, 9])
    onehot = F.one_hot(labels, 10).double()
    q = 0.9 * onehot + 0.01
    torch.testing.assert_close(kl_to_ls(q, labels), torch.zeros(3, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(kl_to_ls(onehot, labels), torch.full((3,), -math.log(.91), dtype=torch.float64))
    assert -math.log(.91) == pytest.approx(.09431067947)
    for t in (1, 4):
        # Huge target logit must not cause loss of precision in non-target normalization.
        logits = onehot * 10000
        torch.testing.assert_close(nontarget_entropy(logits, labels, t), torch.ones(3, dtype=torch.float64))


def test_frozen_configs_paths_roles_and_unused_temperature(tmp_path):
    for seed, name in stage1.specs():
        cfg = stage1.read_config(seed, name)
        assert cfg.data_root == "data/coco_single"
        if cfg.teacher_checkpoint and name != "R2_full_LS01best_T1":
            assert Path(cfg.teacher_checkpoint).name == "last.pt"
        assert cfg.label_smoothing == .1 and cfg.num_workers == 0 and not cfg.probe_enabled
        assert cfg.temperature == (4 if name.endswith("T4") else 1)
    first, second = (stage1.read_config(0, name) for name in stage1.TEACHERS)
    assert run_path(first, "teacher") != run_path(second, "teacher")
    changes = {k for k, v in first.to_dict().items() if second.to_dict()[k] != v}
    assert changes == {"teacher_label_smoothing", "stage1_run"}
    r2 = stage1.read_config(0, "R2_full_LS01best_T1")
    assert r2.teacher_checkpoint == str(run_path(second, "teacher") / "best.pt")
    assert r2.temperature == 1 and r2.teacher_label_smoothing == .1
    assert stage1.BASELINES == {"R1_ce": 60.78, "R2_full_LS01best_T1": 58.91}
    with pytest.raises(ValueError, match="fixed config"):
        stage1.validate_recipe(replace(first, batch_size=16))
    missing = first.to_dict()
    del missing["temperature"]
    write_json(tmp_path / "missing.json", missing)
    with pytest.raises(ValueError, match="temperature"):
        Config.load(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="forbids test"):
        evaluate_test(first)


def test_last_checkpoint_is_loaded_from_explicit_path_and_ce_needs_no_teacher(tmp_path):
    synthetic_data(tmp_path / "data")
    cfg = Config(teacher_label_smoothing=0.0, data_root=str(tmp_path / "data"),
                 output_root=str(tmp_path / "teachers"), device="cpu", num_classes=2,
                 teacher_pretrained=False, student_init="scratch", model_scale="debug", num_workers=0,
                 batch_size=2, eval_batch_size=2, accumulation_steps=2, teacher_epochs=2, epochs=1,
                 warmup_epochs=0, probe_enabled=False)
    best = train(cfg, role="teacher")
    last = best.with_name("last.pt")
    before = sha256(last)
    best.write_bytes(b"Not a usable checkpoint: Full KD must explicitly load last.pt")
    full = replace(cfg, output_root=str(tmp_path / "student"), teacher_checkpoint=str(last))
    train(full, method="full")
    result = json.loads((run_path(full, method="full") / "result.json").read_text())
    assert result["teacher_sha256"] == before == sha256(last)
    assert not (run_path(full, method="full") / "probe").exists()
    start = json.loads((run_path(full, method="full") / "run_start.json").read_text())
    assert start["teacher_checkpoint"] == str(last) and start["teacher_sha256"] == before
    ce = replace(full, output_root=str(tmp_path / "ce"), teacher_checkpoint=None)
    train(ce, method="ce")
    assert load_checkpoint(run_path(ce, method="ce") / "last.pt")["teacher_sha256"] is None


def test_teacher_measurement_uses_full_train_and_val_only_with_fixed_seed(tmp_path):
    synthetic_data(tmp_path / "data", classes=10)
    cfg = Config(teacher_label_smoothing=0.0, data_root=str(tmp_path / "data"), device="cpu", seed=9,
                 num_workers=0, batch_size=7, eval_batch_size=7, amp=False)

    class Model(torch.nn.Module):
        def forward(self, x):
            # Sensitive to augmentation; checking repetition tests seed reset after prior RNG use.
            score = x[:, :, :, :30].mean((1, 2, 3))
            return score[:, None] * torch.arange(10).float()[None, :]

    first = measure(Model(), cfg, "train", torch.device("cpu"))
    torch.rand(100)
    second = measure(Model(), replace(cfg, seed=1), "train", torch.device("cpu"))
    assert first == second and first["samples"] == 30
    assert measure(Model(), cfg, "val", torch.device("cpu"))["samples"] == 20


def test_epoch_window_and_gate_boundaries():
    history = [{"epoch": e, "validation": {"macro_accuracy": .6 if e > 90 else .1}} for e in range(1, 101)]
    assert stage1.mean_last10(history) == pytest.approx(60.)
    with pytest.raises(ValueError, match="exactly"):
        stage1.mean_last10(history[1:])
    rows = {name: {"mean_pp": value, "initial_model_sha256": stage1.INITIAL_SHA256}
            for name, value in zip(stage1.STUDENTS, (60., 59., 60., 61.5, 61.5))}
    decision = stage1.decide_seed0(rows)
    assert decision["decision"] == "RUN_SEED1" and decision["selected"] == "LS0_T1"
    assert stage1.decide_seed1(rows, "LS0_T1")["decision"] == "FINAL_PASS"
    rows["LS0_T1"]["mean_pp"] = rows["LS0_T4"]["mean_pp"] = 61.49
    assert stage1.decide_seed0(rows)["decision"] == "STOP_SEED0"
    rows["R2_full_LS01best_T1"]["mean_pp"] = 56.
    assert stage1.decide_seed0(rows)["decision"] == "STOP_REGRESSION"
    rows["R2_full_LS01best_T1"]["mean_pp"] = 59.
    rows["R1_ce"]["initial_model_sha256"] = "wrong"
    assert stage1.decide_seed0(rows)["decision"] == "STOP_REGRESSION"
    assert stage1.teacher_pass("T_LS0", .02)
    assert not stage1.teacher_pass("T_LS01", .02)
    assert not stage1.teacher_pass("T_LS0", float("nan"))


def test_hash_guard_does_not_modify_input(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"immutable")
    before = path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        stage1.require_hash(path, "wrong")
    assert path.read_bytes() == b"immutable" and path.stat().st_mtime_ns == before


def test_setup_forces_all_install_and_cache_paths_inside_clone(tmp_path):
    import os
    import shutil
    import subprocess
    clone = tmp_path / "clone with spaces"
    clone.mkdir()
    shutil.copyfile(stage1.ROOT / "setup.sh", clone / "setup.sh")
    tools = tmp_path / "tools"
    tools.mkdir()
    fake_uv = tools / "uv"
    fake_uv.write_text('#!/bin/sh\nenv > "$PWD/environment.txt"\nprintf "%s\\n" "$*" >> "$PWD/commands.txt"\n')
    fake_uv.chmod(0o755)
    keys = ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_TOOL_DIR", "TORCH_HOME", "XDG_CACHE_HOME", "TMPDIR",
            "UV_TOOL_BIN_DIR", "UV_PYTHON_BIN_DIR", "UV_PROJECT_ENVIRONMENT", "MPLCONFIGDIR", "CUDA_CACHE_PATH")
    environment = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", **{key: "/outside/forbidden" for key in keys})
    subprocess.run(["bash", "setup.sh", "install"], cwd=clone, env=environment, check=True, capture_output=True)
    actual = dict(line.split("=", 1) for line in (clone / "environment.txt").read_text().splitlines() if "=" in line)
    for key in keys:
        assert Path(actual[key]).is_relative_to(clone)
    assert (clone / "commands.txt").read_text().splitlines() == ["python install 3.11 --no-bin", "sync --frozen --managed-python --python 3.11"]


def test_batch_failure_terminates_peer_processes(tmp_path, monkeypatch):
    from scripts import plan_stage1

    class Process:
        def __init__(self, code):
            self.code, self.terminated = code, False

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated, self.code = True, -15

        def wait(self, timeout=None):
            return self.code

    bad, peer = Process(1), Process(None)
    processes = iter((bad, peer))
    monkeypatch.setattr(plan_stage1, "OUTPUT", tmp_path)
    monkeypatch.setattr(plan_stage1.subprocess, "Popen", lambda *args, **kwargs: next(processes))
    with pytest.raises(RuntimeError, match="exited 1"):
        plan_stage1.run_batch(0, ("T_LS0", "T_LS01"))
    assert peer.terminated


def test_report_generates_tables_and_five_curves_without_reading_test(tmp_path, monkeypatch):
    from coco_kd import stage1_report
    output, report = tmp_path / "outputs", tmp_path / "reports"
    report.mkdir()
    rows = {}
    for name, value in zip(stage1.STUDENTS, (.60, .59, .60, .62, .615)):
        history = [{"epoch": e, "validation": {"macro_accuracy": value}} for e in range(1, 101)]
        rows[name] = {"mean_pp": stage1.mean_last10(history), "history": history,
                      "initial_model_sha256": stage1.INITIAL_SHA256}
        directory = output / "seed_0" / name
        for filename in ("config.json", "result.json", "run_start.json"):
            write_json(directory / filename, {"synthetic_test_fixture": True})
        write_json(directory / "history.json", history)
        (directory / "test_metrics.json").write_text("MUST NOT BE READ")
    teacher_metric = {"accuracy": .9, "macro_accuracy": .9, "kl_to_ls": .1,
                      "nontarget_entropy@T1": .8, "nontarget_entropy@T4": .9}
    teachers = {}
    for name in ("T_LS0", "T_LS01", "T_LS01_best"):
        metric = dict(teacher_metric, kl_to_ls=.1 if name == "T_LS0" else .001)
        teachers[name] = {"train": metric, "val": metric, "gate": "PASS" if name != "T_LS01_best" else "N/A"}
    write_json(output / "seed_0/teacher_checks.json", {"teachers": teachers})
    monkeypatch.setattr(stage1_report, "OUTPUT", output)
    monkeypatch.setattr(stage1_report, "REPORT", report)
    monkeypatch.setattr(stage1_report, "config_audit", lambda: None)
    monkeypatch.setattr(stage1_report, "read_student_rows", lambda seed: rows if seed == 0 else {})
    result = stage1_report.summary()
    text = (report / "SUMMARY.md").read_text()
    assert result["decision"] == "WAIT_SEED1" and result["selected"] == "LS0_T1"
    assert all(name in text for name in stage1.STUDENTS)
    assert "T_LS01_best" in text and "experiment2_best" not in text
    assert "old_teacher_sha256" not in text
    assert (report / "val_curves.png").stat().st_size > 0
    assert not list(report.rglob("test_metrics.json"))
