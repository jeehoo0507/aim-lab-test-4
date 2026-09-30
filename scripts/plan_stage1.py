"""Audit, test, and execute the fixed Stage 1 plan (validation only)."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coco_kd.stage1 import (ROOT, OUTPUT, REPORT, STUDENTS, TEACHERS, check_assets, config_audit,
                            decide_seed0, decide_seed1, read_config, read_student_rows,
                            require_teacher_checks, require_tests, role_method, validation_fingerprint)
from coco_kd.stage1_report import summary
from coco_kd.utils import RunLock, write_json


def run_tests():
    import torch
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {"validation_sha256": validation_fingerprint(), "python": platform.python_version(),
              "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "command": [sys.executable, "-m", "pytest", "-q", "-x", "tests"], "returncode": None}
    # Invalidate an earlier pass before starting, including if the process is interrupted.
    write_json(OUTPUT / "tests.json", record)
    log = OUTPUT / "tests.log"
    with log.open("w") as file:
        process = subprocess.Popen(record["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            file.write(line)
        record["returncode"] = process.wait()
    record["pytest_output"] = log.read_text()
    write_json(OUTPUT / "tests.json", record)
    write_json(REPORT / "tests.json", record)
    if record["returncode"]:
        raise RuntimeError("Tests failed; stop and ask. No automatic fixes or GPU run.")


def run_batch(seed, names):
    """One subprocess per run; fail fast and terminate peers on any failure/OOM."""
    log_root = OUTPUT / "_jobs"
    log_root.mkdir(parents=True, exist_ok=True)
    running, files = [], []
    try:
        for name in names:
            log = (log_root / f"seed_{seed}_{name}.log").open("a", buffering=1)
            files.append(log)
            cmd = [sys.executable, "-u", "-m", "scripts.plan_stage1", "worker", "--seed", str(seed), "--name", name]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
            running.append((name, proc))
        while running:
            for name, proc in running[:]:
                code = proc.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"seed {seed} {name} exited {code}; inspect {log_root}/seed_{seed}_{name}.log")
                running.remove((name, proc))
                print(f"Completed seed {seed} {name}", flush=True)
                if seed == 0 and name in ("R1_ce", "R2_full_old_T1"):
                    decision = decide_seed0(read_student_rows(0))
                    if decision["decision"] == "STOP_REGRESSION":
                        write_json(OUTPUT / "STOP.json", {"decision": "STOP_REGRESSION", "reason": "Regression failed; remaining workers stopped"})
                        raise RuntimeError("Regression failed; stopping all remaining workers")
            if running:
                time.sleep(1)
    finally:
        for _, proc in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in running:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in files:
            log.close()


def execute(jobs, stop_after_seed0=False):
    if jobs != 5:
        raise ValueError("Approved Stage 1 student concurrency is --jobs 5; stop and ask before changing")
    if (OUTPUT / "STOP.json").exists():
        raise ValueError("A prior STOP is recorded; ask the user before any retry")
    require_tests()
    assets = check_assets()
    write_json(OUTPUT / "preflight.json", {"assets": assets, "validation_sha256": validation_fingerprint()})
    config_audit()
    import torch
    from coco_kd.models import build_model
    from coco_kd.utils import setup_device
    if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("Stage 1 GPU run requires locked torch 2.5.1+cu124 / CUDA 12.4")
    cfg = read_config(0, "T_LS0")
    setup_device(cfg)
    # Cache once before the two teachers start to avoid a concurrent download race.
    model = build_model("teacher", cfg, pretrained=True)
    del model
    write_json(OUTPUT / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "assets": check_assets(), "jobs": jobs, "teacher_jobs": 2})
    from scripts.check_teacher_outputs import check_teachers
    run_batch(0, TEACHERS)
    check_teachers(0)
    require_teacher_checks(0)
    config_audit()
    run_batch(0, STUDENTS)
    decision = decide_seed0(read_student_rows(0))
    summary()
    if decision["decision"] != "RUN_SEED1":
        write_json(OUTPUT / "STOP.json", {"decision": decision["decision"], "reason": "Seed 0 gate failed; ask user for next step"})
        raise RuntimeError(f"{decision['decision']}: no further experiments")
    selected = decision["selected"]
    if stop_after_seed0:
        print(f"SEED0_DONE: {decision['decision']}, selected={selected.removeprefix('LS0_')}", flush=True)
        return
    run_batch(1, ("T_LS0",))
    check_teachers(1)
    require_teacher_checks(1)
    run_batch(1, ("R1_ce", selected))
    final = decide_seed1(read_student_rows(1), selected)
    summary()
    if final["decision"] != "FINAL_PASS":
        write_json(OUTPUT / "STOP.json", {"decision": final["decision"], "reason": "Seed 1 gate failed; ask user for next step"})
        raise RuntimeError(f"{final['decision']}: no further experiments")
    print("FINAL_PASS. No RRC/CUB/test work is scheduled.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "plan", "run", "summary", "worker"))
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--stop-after-seed0", action="store_true",
                        help="After the seed 0 decision and summary, exit successfully without starting seed 1")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--name")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == "worker":
        from coco_kd.train import train
        cfg = read_config(args.seed, args.name)
        role, method = role_method(args.name)
        train(cfg, role=role, method=method)
        return
    if args.command == "plan":
        config_audit()
        print(REPORT / "CONFIG_AUDIT.md")
    elif args.command == "summary":
        print(summary()["decision"])
    elif args.command == "test":
        run_tests()
    else:
        with RunLock(OUTPUT / ".stage1.lock"):
            try:
                execute(args.jobs, stop_after_seed0=args.stop_after_seed0)
            except Exception as error:
                if not (OUTPUT / "STOP.json").exists():
                    write_json(OUTPUT / "STOP.json", {"decision": "STOP", "reason": str(error)})
                try:
                    summary()
                except Exception as report_error:
                    print(f"Report unavailable: {report_error}", file=sys.stderr)
                raise


if __name__ == "__main__":
    main()
