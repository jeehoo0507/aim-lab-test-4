"""Generate only Stage 1b reports; Stage 1 records are read-only pinned inputs."""
import json

from .stage1b import (OUTPUT, REPORT, RUNS, PAIRS, BASELINE_COMMIT, read_config,
                     read_rows, read_baseline_rows, decisions, run_dir, config_audit)
from .utils import write_json


def summary():
    config_audit()
    baseline, rows = read_baseline_rows(), read_rows()
    result = decisions(rows)
    stop = OUTPUT / 'STOP.json'
    if stop.exists():
        result['execution_stop'] = json.loads(stop.read_text())
    lines = ['# Stage 1b — augmentation gate', '', f'Execution: **{result["decision"]}**', '',
             'Validation macro accuracy, mean of epochs 91–100. Test is never evaluated.',
             f'Stage 1 reference: `{BASELINE_COMMIT}`; existing reports are pinned and read only.', '',
             '| Stage | run | augmentation | val mean % | delta vs same-augmentation CE (pp) | gate |',
             '|---|---|---|---|---|---|']
    for name, row in baseline.items():
        delta = row['mean_pp'] - baseline['R1_ce']['mean_pp']
        lines.append(f'| 1 | {name} | flip | {row["mean_pp"]:.4f} | {delta:+.4f} | reference |')
    for name in RUNS:
        cfg = read_config(name)
        row = rows.get(name)
        metric = 'NOT_RUN' if row is None else f'{row["mean_pp"]:.4f}'
        rule = result['runs'].get(name)
        delta = 'baseline' if rule is None else 'NOT_RUN' if rule['delta_pp'] is None else f'{rule["delta_pp"]:+.4f}'
        gate = 'baseline' if rule is None else rule['status']
        lines.append(f'| 1b | {name} | {cfg.train_augmentation} | {metric} | {delta} | {gate} |')
    lines += ['', 'KD PASS requires delta >=1.5 pp against its own augmentation CE. No automatic seed 1 or further training.',
              'Train accuracy for rrc_mix is not recorded; the CE target is the LS=0.1 mixture.',
              'Crop/mix stream hashes must match between each RRC / RRC-mix KD run and its CE for every epoch.']
    gate_path = OUTPUT / 'flip_verification.json'
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    lines += ['', f'Server historical flip gate: **{gate.get("status", "NOT_RUN")}**.',
              'CPU parity is a separate same-platform test on synthetic data. Its checksum and metrics are not server reference values.']
    teacher_path = OUTPUT / 'teacher_checks.json'
    teacher = json.loads(teacher_path.read_text()) if teacher_path.exists() else {}
    lines += ['', f'New T_LS0 output check: **{teacher.get("status", "NOT_RUN")}**.',
              f'Teacher actual SHA-256: `{teacher.get("assets", {}).get("teacher_sha256", "NOT_RUN")}`.',
              '', '| split | accuracy | macro accuracy | kl_to_ls (nats) | non-target entropy T1 | non-target entropy T4 |',
              '|---|---|---|---|---|---|']
    for split in ('train', 'val'):
        row = teacher.get(split)
        if row:
            values = [row[k] for k in ('accuracy', 'macro_accuracy', 'kl_to_ls', 'nontarget_entropy@T1', 'nontarget_entropy@T4')]
            lines.append('| ' + split + ' | ' + ' | '.join(f'{v:.8f}' for v in values) + ' |')
    lines += ['', 'Teacher gate uses train kl_to_ls >=0.02 only; validation is report-only.',
              '', 'New-server flip diagnostics: initialization checksum is strict; historical epoch metrics are report-only.',
              '| epoch | metric (raw units) | Stage 1 reference | new server | difference | within 1e-6 |',
              '|---|---|---|---|---|---|']
    for row in gate.get('differences', []):
        lines.append(f'| {row["epoch"]} | {row["metric"]} | {row["reference"]:.10f} | {row["actual"]:.10f} | {row["delta"]:+.10f} | {row["within_1e6"]} |')
    lines += ['', '| flip reproduction | Stage 1 val mean % | new server val mean % | difference pp (report-only) |',
              '|---|---|---|---|']
    for name, reference in (('ce_flip', 'R1_ce'), ('LS0_T4_flip', 'LS0_T4')):
        old = baseline[reference]['mean_pp']
        row = rows.get(name)
        metric = 'NOT_RUN' if row is None else f'{row["mean_pp"]:.4f}'
        delta = 'NOT_RUN' if row is None else f'{row["mean_pp"] - old:+.4f}'
        lines.append(f'| {name} | {old:.4f} | {metric} | {delta} |')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for name in RUNS:
            if name in rows:
                history = rows[name]['history']
                ax.plot([r['epoch'] for r in history], [100 * r['validation']['macro_accuracy'] for r in history], label=name)
        ax.axvspan(91, 100, color='grey', alpha=.12)
        ax.set(xlabel='Epoch', ylabel='Validation macro accuracy (%)', xlim=(1, 100))
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(REPORT / 'val_curves.png', dpi=160)
        plt.close(fig)
        lines += ['', '![Stage 1b validation curves](val_curves.png)']
    else:
        lines += ['', 'Validation curves: NOT_RUN. No Stage 1b GPU results yet.']
    lines += ['', '해석: 각 KD 결과는 같은 증강 CE와만 비교한다. 미실행 결과는 추정하지 않는다.']
    (REPORT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    write_json(REPORT / 'decision.json', result)
    for name in rows:
        for filename in ('config.json', 'history.json', 'result.json', 'run_start.json'):
            write_json(REPORT / 'seed_0' / name / filename, json.loads((run_dir(name) / filename).read_text()))
    for filename in ('tests.json', 'cpu_flip_verification.json', 'flip_verification.json', 'preflight.json', 'teacher_checks.json'):
        if (OUTPUT / filename).exists():
            write_json(REPORT / filename, json.loads((OUTPUT / filename).read_text()))
    return result
