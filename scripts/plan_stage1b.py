"""Stage 1b launcher: new teacher followed by seven parallel student runs; keeps historical Stage 1 results unchanged."""
import argparse
import json
import os
import subprocess
import sys
import time

from coco_kd.stage1b import (ROOT, OUTPUT, REPORT, RUNS, check_assets, read_config, method,
                            runtime, validation_fingerprint, require_tests, config_audit, read_baseline_rows)
from coco_kd.stage1b_report import summary
from coco_kd.utils import RunLock, write_json


def run_tests():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record = {**runtime(), 'validation_sha256': validation_fingerprint(), 'returncode': None,
              'command': [sys.executable, '-m', 'pytest', '-q', '-x', 'tests']}
    write_json(OUTPUT / 'tests.json', record)
    with (OUTPUT / 'tests.log').open('w') as file:
        process = subprocess.Popen(record['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env=dict(os.environ, STAGE1B_CPU_PARITY_RECORD=str((OUTPUT / 'cpu_flip_verification.json').resolve())))
        for line in process.stdout:
            print(line, end='', flush=True)
            file.write(line)
        record['returncode'] = process.wait()
    record['pytest_output'] = (OUTPUT / 'tests.log').read_text()
    write_json(OUTPUT / 'tests.json', record)
    write_json(REPORT / 'tests.json', record)
    if record['returncode']:
        raise RuntimeError('Tests failed; stop and ask before proceeding')


def run_batch():
    logs = OUTPUT / '_jobs'
    logs.mkdir(parents=True, exist_ok=True)
    running, files = [], []
    try:
        for name in RUNS:
            file = (logs / f'{name}.log').open('a', buffering=1)
            files.append(file)
            cmd = [sys.executable, '-u', '-m', 'scripts.plan_stage1b', 'worker', '--name', name]
            proc = subprocess.Popen(cmd, stdout=file, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'))
            running.append((name, proc))
            print(f'Started {name}, pid={proc.pid}, log={logs}/{name}.log', flush=True)
        while running:
            for name, proc in running[:]:
                code = proc.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f'{name} exited {code}; inspect {logs}/{name}.log')
                running.remove((name, proc))
                print(f'Completed {name}', flush=True)
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
        for file in files:
            file.close()


def execute(jobs, verify_only=False):
    if jobs != 7:
        raise ValueError('Stage 1b requires --jobs 7; stop and ask before changing')
    if (OUTPUT / 'STOP.json').exists():
        raise ValueError('Stage 1b STOP recorded; stop and ask before retry')
    require_tests()
    assets = check_assets(require_teacher=False)
    read_baseline_rows()
    config_audit()
    import torch
    from coco_kd.utils import setup_device
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4':
        raise ValueError('Stage 1b requires locked torch 2.5.1+cu124 / CUDA 12.4')
    setup_device(read_config('ce_rrc'))
    write_json(OUTPUT / 'preflight.json', {'assets': assets, 'runtime': runtime(),
               'gpu': torch.cuda.get_device_name(), 'jobs': jobs, 'validation_sha256': validation_fingerprint()})
    from scripts.check_stage1b_teacher import train_and_check
    train_and_check()
    config_audit()
    from scripts.verify_stage1b_flip import verify
    verify()
    if verify_only:
        return
    torch.cuda.empty_cache()
    run_batch()
    result = summary()
    if result['decision'] != 'DONE':
        raise RuntimeError('Missing completed runs after Stage 1b execution')
    print('STAGE1B_DONE: ' + ', '.join(f'{name}={rule["status"]}' for name, rule in result['runs'].items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'test', 'plan', 'verify-flip', 'run', 'summary', 'worker'))
    parser.add_argument('--jobs', type=int, default=7)
    parser.add_argument('--name', choices=RUNS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'worker':
        if args.name is None:
            parser.error('worker requires --name')
        from coco_kd.train import train
        train(read_config(args.name), role='student', method=method(args.name))
    else:
        with RunLock(OUTPUT / '.stage1b.lock'):
            try:
                if (OUTPUT / 'STOP.json').exists() and args.command not in ('plan', 'summary'):
                    raise ValueError('Stage 1b STOP recorded; stop and ask before retry')
                if args.command == 'prepare':
                    # Reuse the exact Stage 1 preparation entry point and both strict hashes.
                    from scripts.plan_stage1 import prepare_data
                    prepare_data()
                elif args.command == 'test':
                    run_tests()
                elif args.command == 'plan':
                    config_audit()
                    print(REPORT / 'CONFIG_AUDIT.md')
                elif args.command == 'summary':
                    print(summary()['decision'])
                else:
                    execute(args.jobs, verify_only=args.command == 'verify-flip')
            except Exception as error:
                if not (OUTPUT / 'STOP.json').exists():
                    write_json(OUTPUT / 'STOP.json', {'decision': 'STOP_ERROR', 'reason': str(error)})
                try:
                    summary()
                except Exception as report_error:
                    print(f'Report unavailable: {report_error}', file=sys.stderr)
                raise


if __name__ == '__main__':
    main()
