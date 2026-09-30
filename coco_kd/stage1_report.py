"""Generate Stage 1 tables and validation curves from recorded result files."""
import json

from .stage1 import (INITIAL_SHA256, MANIFEST_SHA256, OUTPUT, REPORT, STUDENTS,
                     BASELINES, config_audit, decide_seed0, decide_seed1, read_student_rows,
                     teacher_pass, validation_fingerprint)
from .utils import write_json


def summary():
    config_audit()
    seed0, seed1 = read_student_rows(0), read_student_rows(1)
    decision = decide_seed0(seed0)
    rules = list(decision["rules"])
    if decision["decision"] == "RUN_SEED1":
        confirmation = decide_seed1(seed1, decision["selected"])
        rules.extend(confirmation["rules"])
        decision["decision"] = confirmation["decision"]
    teacher_records = {}
    for seed in (0, 1):
        path = OUTPUT / f"seed_{seed}" / "teacher_checks.json"
        if path.exists():
            teacher_records[seed] = json.loads(path.read_text())
    for seed, names in ((0, ("T_LS0", "T_LS01")), (1, ("T_LS0",))):
        for name in names:
            row = teacher_records.get(seed, {}).get("teachers", {}).get(name)
            status = "NOT_RUN" if row is None else "PASS" if teacher_pass(name, row["train"]["kl_to_ls"]) else "FAIL"
            rules.append({"rule": f"seed {seed} {name}: train KL gate", "status": status})
            if status == "FAIL":
                decision["decision"] = "STOP_TEACHER"
    test_path = OUTPUT / "tests.json"
    if test_path.exists():
        tests = json.loads(test_path.read_text())
        passed = tests["returncode"] == 0 and tests["validation_sha256"] == validation_fingerprint()
        rules.insert(0, {"rule": "local/server test suite (current source)", "status": "PASS" if passed else "FAIL"})
    else:
        rules.insert(0, {"rule": "local/server test suite", "status": "NOT_RUN"})
    preflight = OUTPUT / "preflight.json"
    assets = json.loads(preflight.read_text())["assets"] if preflight.exists() else {}
    for key, expected in (("manifest_sha256", MANIFEST_SHA256),):
        rules.append({"rule": key, "status": "NOT_RUN" if key not in assets else "PASS" if assets[key] == expected else "FAIL"})
    stop_path = OUTPUT / "STOP.json"
    stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
    if stop:
        decision["decision"] = stop["decision"]
        rules.append({"rule": stop["reason"].replace("|", "/").replace("\n", " "), "status": "FAIL"})
    lines = ["# Stage 1 — KD gate", "", f"Decision: **{decision['decision']}**", "",
             "Only validation is used for student decisions. NOT_RUN means no measurement; it is not a failure or a pass.", "",
             "R2 uses newly trained seed 0 T_LS01 best.pt. The historical 58.91 ±2.0 pp reference is retained by user instruction; the original experiment-2 teacher is not replayed.", "",
             "## Table 1: teacher outputs", "",
             "Train uses augmentation once with measurement seed 0; KL gates apply only to train. Entropy is normalized by log(9).", "",
             "| seed | teacher | split | accuracy % | macro % | KL to LS (nats) | nontarget entropy T=1 | T=4 | train gate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for seed, names in ((0, ("T_LS0", "T_LS01", "T_LS01_best")), (1, ("T_LS0",))):
        for name in names:
            row = teacher_records.get(seed, {}).get("teachers", {}).get(name)
            for split in ("train", "val"):
                if row is None:
                    lines.append(f"| {seed} | {name} | {split} | — | — | — | — | — | NOT_RUN |")
                    continue
                m = row[split]
                lines.append(f"| {seed} | {name} | {split} | {100*m['accuracy']:.4f} | {100*m['macro_accuracy']:.4f} | "
                             f"{m['kl_to_ls']:.6f} | {m['nontarget_entropy@T1']:.6f} | {m['nontarget_entropy@T4']:.6f} | "
                             f"{row['gate'] if split == 'train' else 'report only'} |")
    lines.extend(["", "## Table 2: students", "", "Means use epochs 91–100; differences use unrounded values.", "",
                  "| seed | run | val macro mean % | Δ CE (pp) | Δ regression reference (pp) | initial checksum |",
                  "|---|---|---|---|---|---|"])
    selected = decision.get("selected")
    for seed, rows, names in ((0, seed0, STUDENTS), (1, seed1, ("R1_ce", selected) if selected else ("R1_ce",))):
        for name in names:
            row = rows.get(name)
            if row is None:
                lines.append(f"| {seed} | {name} | — | — | — | NOT_RUN |")
                continue
            delta = f"{row['mean_pp'] - rows['R1_ce']['mean_pp']:+.4f}" if "R1_ce" in rows else "—"
            baseline = f"{row['mean_pp'] - BASELINES[name]:+.4f}" if seed == 0 and name in BASELINES else "미사용"
            expected = INITIAL_SHA256 if seed == 0 else rows.get("R1_ce", {}).get("initial_model_sha256")
            match = "PASS" if row["initial_model_sha256"] == expected else "FAIL" if expected else "NOT_RUN"
            lines.append(f"| {seed} | {name} | {row['mean_pp']:.4f} | {delta} | {baseline} | {match} |")
    lines.extend(["", "## Table 3: decisions", "", "| rule | status |", "|---|---|"])
    lines.extend(f"| {row['rule']} | {row['status']} |" for row in rules)
    lines.extend(["", f"Selected seed 0 candidate: `{selected or 'NOT_SELECTED'}`.", ""])
    if seed0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for name in STUDENTS:
            if name in seed0:
                history = seed0[name]["history"]
                ax.plot([h["epoch"] for h in history], [100*h["validation"]["macro_accuracy"] for h in history], label=name)
        ax.axvspan(91, 100, color="grey", alpha=.12)
        ax.set(xlabel="Epoch", ylabel="Validation macro accuracy (%)", xlim=(1, 100))
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(REPORT / "val_curves.png", dpi=160)
        plt.close(fig)
        lines.append("![Seed 0 validation curves](val_curves.png)")
    else:
        lines.append("Validation curves: NOT_RUN (GPU experiment has not run).")
    lines.extend(["", "해석: 위 결정은 사전 고정된 임계값만 적용한 결과이며, 미실행 항목의 성능은 추정하지 않는다."])
    (REPORT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    write_json(REPORT / "decision.json", {**decision, "rules": rules})
    # Export only measured train/val data needed to independently regenerate the report.
    for seed, record in teacher_records.items():
        write_json(REPORT / f"seed_{seed}" / "teacher_checks.json", record)
    for seed, rows in ((0, seed0), (1, seed1)):
        for name in rows:
            for filename in ("config.json", "history.json", "result.json", "run_start.json"):
                path = OUTPUT / f"seed_{seed}" / name / filename
                write_json(REPORT / f"seed_{seed}" / name / filename, json.loads(path.read_text()))
    if test_path.exists():
        write_json(REPORT / "tests.json", json.loads(test_path.read_text()))
    return decision
