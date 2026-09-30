"""Frozen Stage 1 recipe and gates. Never reads held-out test results."""
import hashlib
import json
import math
import platform
from pathlib import Path

from .config import Config
from .utils import sha256, source_fingerprint, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("outputs/stage1_ls_gate")
REPORT = Path("reports/stage1")
DATA = "data/coco_single"
PREPARE_SHA256 = "a51e46ee10b1b1818d66777c509fd7e614ca852dc195e5d5106c1bc82a95e43d"
MANIFEST_SHA256 = "ffbd7a44425184e36efc3f51821cb1fb7126d5fea1dbfeaf0c3229a90528e62e"
INITIAL_SHA256 = "543d4714549b4a1318384fd102ebcfc4b65a0ccf73da68d63e1507d15e5a4e8a"
STUDENTS = ("R1_ce", "R2_full_LS01best_T1", "LS01_T4", "LS0_T1", "LS0_T4")
TEACHERS = ("T_LS0", "T_LS01")
BASELINES = {"R1_ce": 60.78, "R2_full_LS01best_T1": 58.91}


def run_dir(seed, name):
    return OUTPUT / f"seed_{seed}" / name


def config_path(seed, name):
    return ROOT / "configs/stage1" / f"seed_{seed}" / f"{name}.json"


def recipe(seed, name):
    allowed = (*TEACHERS, *STUDENTS) if seed == 0 else ("T_LS0", "R1_ce", "LS0_T1", "LS0_T4")
    if seed not in (0, 1) or name not in allowed:
        raise ValueError("Unknown Stage 1 seed/run")
    teacher_ls = 0.1 if name in ("T_LS01", "R2_full_LS01best_T1", "LS01_T4") else 0.0
    checkpoint = None
    if name == "R2_full_LS01best_T1":
        checkpoint = str(run_dir(0, "T_LS01") / "best.pt")
    elif name.startswith("LS"):
        checkpoint = str(run_dir(seed, "T_LS01" if name == "LS01_T4" else "T_LS0") / "last.pt")
    return Config(teacher_label_smoothing=teacher_ls, data_root=DATA, output_root=str(OUTPUT),
                  seed=seed, student_init="scratch", num_workers=0, temperature=4.0 if name.endswith("T4") else 1.0,
                  teacher_checkpoint=checkpoint, probe_enabled=False, stage1_run=name)


def specs():
    return [(seed, name) for seed, names in ((0, (*TEACHERS, *STUDENTS)),
                                           (1, ("T_LS0", "R1_ce", "LS0_T1", "LS0_T4"))) for name in names]


def role_method(name):
    return ("teacher", "teacher") if name in TEACHERS else ("student", "ce" if name == "R1_ce" else "full")


def validate_recipe(cfg):
    expected = recipe(cfg.seed, cfg.stage1_run).to_dict()
    differences = {k: (expected[k], v) for k, v in cfg.to_dict().items() if v != expected[k]}
    if differences:
        raise ValueError(f"Stage 1 fixed config changed; stop and ask: {differences}")


def read_config(seed, name):
    cfg = Config.load(config_path(seed, name))
    validate_recipe(cfg)
    return cfg


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: expected {expected}, got {actual}")
    return actual


def check_assets():
    return {"manifest_sha256": require_hash(Path(DATA) / "manifest.json", MANIFEST_SHA256)}


def validation_fingerprint():
    digest = hashlib.sha256()
    paths = [ROOT / "setup.sh", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    for folder, pattern in (("coco_kd", "*.py"), ("scripts", "*.py"), ("tests", "*.py"), ("configs", "*.json")):
        paths.extend((ROOT / folder).rglob(pattern))
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_tests():
    import torch
    record = json.loads((OUTPUT / "tests.json").read_text())
    if (record["returncode"] != 0 or record["validation_sha256"] != validation_fingerprint()
            or record["platform"] != platform.platform() or record["torch"] != torch.__version__
            or record["python"] != platform.python_version()):
        raise ValueError("Stage 1 tests failed or source/config changed; run stage1-test before GPU work")


def teacher_specs(seed):
    items = [(name, run_dir(seed, name) / "last.pt", 0.0 if name == "T_LS0" else 0.1)
             for name in (TEACHERS if seed == 0 else ("T_LS0",))]
    if seed == 0:
        items.append(("T_LS01_best", run_dir(0, "T_LS01") / "best.pt", 0.1))
    return items


def teacher_pass(name, kl_to_ls):
    if not math.isfinite(kl_to_ls):
        return False
    return kl_to_ls >= 0.02 if name == "T_LS0" else kl_to_ls < 0.02 if name == "T_LS01" else True


def require_teacher_checks(seed):
    path = OUTPUT / f"seed_{seed}" / "teacher_checks.json"
    record = json.loads(path.read_text())
    if record["source_sha256"] != source_fingerprint() or record["measurement_seed"] != 0:
        raise ValueError("Teacher output checks are stale")
    if record["manifest_sha256"] != MANIFEST_SHA256 or record["seed"] != seed:
        raise ValueError("Teacher output check provenance mismatch")
    for name, checkpoint, _ in teacher_specs(seed):
        row = record["teachers"][name]
        require_hash(checkpoint, row["checkpoint_sha256"])
        if not teacher_pass(name, row["train"]["kl_to_ls"]):
            raise ValueError(f"Teacher output check failed: {name}")
    return record


def validate_training_run(cfg, role, method):
    if (OUTPUT / "STOP.json").exists():
        raise ValueError("A prior STOP is recorded; ask the user before any retry")
    validate_recipe(cfg)
    if (role, method) != role_method(cfg.stage1_run):
        raise ValueError("Stage 1 role/method mismatch")
    require_tests()
    check_assets()
    if cfg.seed == 1:
        decision = decide_seed0(read_student_rows(0))
        if decision["decision"] != "RUN_SEED1" or cfg.stage1_run not in ("T_LS0", "R1_ce", decision["selected"]):
            raise ValueError("Seed 1 requires a passing seed 0 gate and the selected temperature")
        require_teacher_checks(0)
    if role == "student":
        require_teacher_checks(cfg.seed)


def read_student_rows(seed):
    names = STUDENTS if seed == 0 else ("R1_ce", "LS0_T1", "LS0_T4")
    rows = {}
    for name in names:
        directory = run_dir(seed, name)
        if not (directory / "result.json").exists():
            continue
        result = json.loads((directory / "result.json").read_text())
        cfg = read_config(seed, name)
        if result["config"] != cfg.to_dict() or result["metadata_sha256"] != MANIFEST_SHA256:
            raise ValueError(f"Result config/data mismatch: {directory}")
        if result["code_sha256"] != source_fingerprint():
            raise ValueError(f"Result source changed: {directory}")
        if (result["role"], result["method"]) != role_method(name) or result["seed"] != seed or result["student_init"] != "scratch":
            raise ValueError(f"Result role/method/seed mismatch: {directory}")
        if result["last_epoch"] != 100 or result["partial_training"] or result["test_evaluated"]:
            raise ValueError(f"Incomplete or test-evaluated Stage 1 run: {directory}")
        expected_teacher = sha256(cfg.teacher_checkpoint) if cfg.teacher_checkpoint else None
        if result["teacher_sha256"] != expected_teacher:
            raise ValueError(f"Result teacher mismatch: {directory}")
        history = json.loads((directory / "history.json").read_text())
        score = mean_last10(history)
        rows[name] = {"mean_pp": score, "initial_model_sha256": result["initial_model_sha256"],
                      "history": history, "teacher_sha256": expected_teacher}
    return rows


def mean_last10(history):
    if [row["epoch"] for row in history] != list(range(1, 101)):
        raise ValueError("Expected exactly epochs 1..100, without gaps/duplicates")
    values = [row["validation"]["macro_accuracy"] for row in history]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid validation macro accuracy")
    return math.fsum(values[90:100]) * 10  # mean x 100, in percentage points


def decide_seed0(rows):
    rules = []
    for name, baseline in BASELINES.items():
        row = rows.get(name)
        for label, passed in ((f"{name}: regression ±2.0 pp", None if row is None else abs(row["mean_pp"] - baseline) <= 2.0),
                              (f"{name}: initial checksum", None if row is None else row["initial_model_sha256"] == INITIAL_SHA256)):
            rules.append({"rule": label, "status": "NOT_RUN" if passed is None else "PASS" if passed else "FAIL"})
    if any(r["status"] == "FAIL" for r in rules):
        return {"decision": "STOP_REGRESSION", "selected": None, "rules": rules}
    if any(name not in rows for name in STUDENTS):
        return {"decision": "WAIT_SEED0", "selected": None, "rules": rules}
    selected = "LS0_T1" if rows["LS0_T1"]["mean_pp"] >= rows["LS0_T4"]["mean_pp"] else "LS0_T4"
    delta = rows[selected]["mean_pp"] - rows["R1_ce"]["mean_pp"]
    passed = delta >= 1.5
    rules.append({"rule": "seed 0 KD − CE ≥1.5 pp", "status": "PASS" if passed else "FAIL", "delta_pp": delta})
    return {"decision": "RUN_SEED1" if passed else "STOP_SEED0", "selected": selected, "rules": rules}


def decide_seed1(rows, selected):
    if any(name not in rows for name in ("R1_ce", selected)):
        return {"decision": "WAIT_SEED1", "rules": [{"rule": "seed 1 KD − CE ≥1.0 pp", "status": "NOT_RUN"}]}
    delta = rows[selected]["mean_pp"] - rows["R1_ce"]["mean_pp"]
    paired = rows[selected]["initial_model_sha256"] == rows["R1_ce"]["initial_model_sha256"]
    return {"decision": "FINAL_PASS" if delta >= 1.0 and paired else "STOP_SEED1", "rules": [
        {"rule": "seed 1 paired initial checksum", "status": "PASS" if paired else "FAIL"},
        {"rule": "seed 1 KD − CE ≥1.0 pp", "status": "PASS" if delta >= 1.0 else "FAIL", "delta_pp": delta}]}


def config_audit():
    lines = ["# Stage 1 config audit", "", "Generated from committed config JSONs. Missing assets are UNAVAILABLE, never a measured hash.", "",
             "| seed | run | role | teacher LS | student LS | T | α | LR | epochs | augmentation | teacher file | actual SHA-256 | schedule |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for seed, name in specs():
        cfg = read_config(seed, name)
        role, method = role_method(name)
        teacher = role == "teacher"
        unused = lambda value: f"{value:g} (미사용)"
        tls = unused(cfg.teacher_label_smoothing) if method == "ce" else f"{cfg.teacher_label_smoothing:g}"
        sls = unused(cfg.label_smoothing) if teacher else f"{cfg.label_smoothing:g}"
        temp = unused(cfg.temperature) if teacher or method == "ce" else f"{cfg.temperature:g}"
        alpha = unused(cfg.kd_alpha) if teacher or method == "ce" else f"{cfg.kd_alpha:g}"
        path = cfg.teacher_checkpoint
        digest = sha256(path) if path and Path(path).is_file() else "UNAVAILABLE" if path else "미사용"
        schedule = "seed 0" if seed == 0 else "seed 0 통과 시; KD는 선택된 T만"
        lines.append(f"| {seed} | {name} | {role} | {tls} | {sls} | {temp} | {alpha} | "
                     f"{cfg.teacher_lr if teacher else cfg.scratch_lr:g} | {cfg.teacher_epochs if teacher else cfg.epochs} | "
                     f"224 resize + flip | {path or '미사용'} | {digest} | {schedule} |")
    lines.extend(["", f"Required manifest SHA-256: `{MANIFEST_SHA256}`.",
                  f"Data: `{DATA}`, prepared with unchanged `8ebf7af:coco_kd/prepare.py`.",
                  "Teachers: epoch 30 `last.pt`; R2 only: newly trained seed 0 `T_LS01/best.pt`.",
                  "R2 retains the historical 58.91 ±2.0 pp reference by user instruction; it does not replay the original experiment-2 teacher.",
                  "All runs: AdamW wd 0.05, warmup 5 + cosine to 1e-6, batch 32 × accumulation 4, clip 1.0, AMP, drop_path 0.1.",
                  "Student: DeiT-Ti scratch. Teacher: DeiT-S ImageNet. Probe disabled. workers 0, threads 2.",
                  "Teacher measurement seed 0, train gate only; validation is descriptive. No test evaluation."])
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "CONFIG_AUDIT.md").write_text("\n".join(lines) + "\n")
