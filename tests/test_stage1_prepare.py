"""New local data preparation and LS01 best substitution; no network/GPU use."""
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from coco_kd import prepare as preparation, stage1
from coco_kd.utils import sha256
from scripts import plan_stage1
from scripts.check_teacher_outputs import validate_checkpoint


def test_original_prepare_source_is_unchanged():
    # SHA-256 of git show 8ebf7af:coco_kd/prepare.py; checked before editing.
    assert sha256(stage1.ROOT / "coco_kd/prepare.py") == "a51e46ee10b1b1818d66777c509fd7e614ca852dc195e5d5106c1bc82a95e43d"
    assert stage1.MANIFEST_SHA256 == "ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e"


@pytest.mark.parametrize("scenario", ["match", "wrong_manifest", "changed_prepare"])
def test_prepare_cli_checks_manifest_and_stops_on_mismatch(tmp_path, monkeypatch, capsys, scenario):
    original = (stage1.ROOT / "coco_kd/prepare.py").read_bytes()
    source = tmp_path / "coco_kd/prepare.py"
    source.parent.mkdir()
    source.write_bytes(original + (b"\n# changed\n" if scenario == "changed_prepare" else b""))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plan_stage1, "ROOT", tmp_path)
    monkeypatch.setattr(plan_stage1, "OUTPUT", tmp_path / "outputs/stage1_ls_gate")
    monkeypatch.setattr(sys, "argv", ["plan_stage1.py", "prepare"])
    monkeypatch.setattr(plan_stage1, "execute", lambda *a, **kw: pytest.fail("Preparation must not start training"))
    summaries, calls = [], []
    monkeypatch.setattr(plan_stage1, "summary", lambda: summaries.append("summary"))
    payload = b'{"fixture": "prepared manifest"}\n'
    actual = hashlib.sha256(payload).hexdigest()
    if scenario == "match":
        monkeypatch.setattr(stage1, "MANIFEST_SHA256", actual)

    def fake_prepare(*args, **kwargs):
        calls.append((args, kwargs))
        directory = Path(args[0])
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_bytes(payload)

    monkeypatch.setattr(preparation, "prepare", fake_prepare)
    if scenario == "match":
        assert plan_stage1.main() is None
        assert f"manifest_sha256={actual}" in capsys.readouterr().out
        record = json.loads((plan_stage1.OUTPUT / "preparation.json").read_text())
        assert record == {"data_root": "data/coco_single", "prepare_sha256": stage1.PREPARE_SHA256,
                          "assets": {"manifest_sha256": actual}}
        assert stage1.check_assets() == {"manifest_sha256": actual}  # No teacher exists yet.
        assert not (plan_stage1.OUTPUT / "STOP.json").exists()
    else:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            plan_stage1.main()
        record = json.loads((plan_stage1.OUTPUT / "STOP.json").read_text())
        assert record["decision"] == "STOP_PREPARE"
        assert not (plan_stage1.OUTPUT / "preparation.json").exists()
        assert "PREPARED:" not in capsys.readouterr().out
        assert summaries == ["summary"]
    assert calls == ([] if scenario == "changed_prepare" else [(("data/coco_single",), {})])


def test_replacement_teacher_specs_and_epoch_ls_checks():
    specs = stage1.teacher_specs(0)
    assert specs == [("T_LS0", stage1.run_dir(0, "T_LS0") / "last.pt", 0.),
                     ("T_LS01", stage1.run_dir(0, "T_LS01") / "last.pt", .1),
                     ("T_LS01_best", stage1.run_dir(0, "T_LS01") / "best.pt", .1)]
    assert stage1.teacher_specs(1) == [("T_LS0", stage1.run_dir(1, "T_LS0") / "last.pt", 0.)]
    best = {"role": "teacher", "metadata_sha256": stage1.MANIFEST_SHA256, "epoch": 7,
            "config": {"seed": 0, "stage1_run": "T_LS01", "teacher_label_smoothing": .1, "label_smoothing": .1}}
    validate_checkpoint(best, 0, "T_LS01_best", .1)
    with pytest.raises(ValueError, match="epoch 30"):
        validate_checkpoint(best, 0, "T_LS01", .1)
    validate_checkpoint({**best, "epoch": 30}, 0, "T_LS01", .1)
    with pytest.raises(ValueError, match="epochs 1..30"):
        validate_checkpoint({**best, "epoch": 31}, 0, "T_LS01_best", .1)
    incorrect = deepcopy(best)
    incorrect["config"]["teacher_label_smoothing"] = 0.
    with pytest.raises(ValueError, match="LS mismatch"):
        validate_checkpoint(incorrect, 0, "T_LS01_best", .1)
    assert stage1.teacher_pass("T_LS01_best", .2)  # Third row remains descriptive, no new KL gate.
