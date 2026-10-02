"""Separate local CPU parity from strict server initialization and reported hardware differences."""
import json
import math
import subprocess
import types
from dataclasses import replace
from pathlib import Path

from coco_kd.config import Config
from coco_kd.stage1 import INITIAL_SHA256, recipe as stage1_recipe
from coco_kd.stage1b import (ROOT, OUTPUT, BASELINE_COMMIT, check_assets, read_baseline_rows,
                            runtime, validation_fingerprint, require_tests)
from coco_kd.stage1b_config import Stage1bConfig
from coco_kd.utils import RunLock, load_checkpoint, write_json


def reference_module():
    # Read immutable Git object; no second maintained implementation of the loop.
    source = subprocess.check_output(['git', 'show', f'{BASELINE_COMMIT}:coco_kd/train.py'], cwd=ROOT)
    import hashlib
    pins = json.loads((ROOT / 'configs/stage1b/BASELINE_SHA256.json').read_text())
    if hashlib.sha256(source).hexdigest() != pins['train_source_sha256']:
        raise ValueError('Frozen Stage 1 reference source hash mismatch')
    module = types.ModuleType('coco_kd._stage1_reference')
    module.__package__ = 'coco_kd'
    exec(compile(source, f'{BASELINE_COMMIT}:coco_kd/train.py', 'exec'), module.__dict__)
    return module


def compare_flip(cfg: Config, destination: Path):
    if cfg.device != 'cpu':
        raise ValueError('Same-platform parity is CPU-only; use the separate server gate on CUDA')
    from coco_kd import train as current
    old = reference_module()
    # Disable only the historical run identifier's server-specific hash guard.
    # Both CPU paths retain identical model/training settings and seeds.
    cfg = replace(cfg, stage1_run=None)
    new_cfg = Stage1bConfig(**{**cfg.to_dict(), 'stage1_run': None}, train_augmentation='flip')
    directories = [destination / 'reference', destination / 'shared_flip']
    for module, config, directory in zip((old, current), (cfg, new_cfg), directories):
        with RunLock(directory / '.run.lock'):
            module._train(config, 'student', 'ce', directory, stop_after=2)
    histories = [json.loads((d / 'history.json').read_text()) for d in directories]
    states = [load_checkpoint(d / 'last.pt') for d in directories]
    if states[0]['initial_model_sha256'] != states[1]['initial_model_sha256']:
        raise ValueError('Flip initial checksum differs from Stage 1')
    for a, b in zip(*histories):
        if a['epoch'] != b['epoch'] or a['lr'] != b['lr'] or a['validation'] != b['validation']:
            raise ValueError('Flip epoch/LR/validation differs from Stage 1')
        for key in a['train']:
            if key != 'seconds' and a['train'][key] != b['train'].get(key):
                raise ValueError(f'Flip epoch {a["epoch"]} train {key} differs from Stage 1')
    if [h['epoch'] for h in histories[0]] != [1, 2] or [h['epoch'] for h in histories[1]] != [1, 2]:
        raise ValueError('Flip check requires exactly the first two epochs')
    import torch
    if any(not torch.equal(states[0]['model'][k], states[1]['model'][k]) for k in states[0]['model']):
        raise ValueError('Flip final weights differ from Stage 1')
    record = {'scope': 'same_platform_cpu', 'runtime': runtime(),
            'config': cfg.to_dict(),
            'reference_initial_model_sha256': states[0]['initial_model_sha256'],
            'initial_model_sha256': states[1]['initial_model_sha256'],
            'epochs': [{'epoch': h['epoch'], 'loss': h['train']['loss'], 'validation': h['validation']}
                       for h in histories[1]], 'status': 'PASS', 'comparison': 'exact; no platform reference hash'}
    write_json(destination / 'verification.json', record)
    return record


def compare_server_record(record, baseline, report_only=False):
    if record['initial_model_sha256'] != INITIAL_SHA256:
        raise ValueError('Flip checksum does not match recorded Stage 1 R1_ce')
    if [h['epoch'] for h in record['epochs']] != [1, 2] or [h['epoch'] for h in baseline[:2]] != [1, 2]:
        raise ValueError('Server flip gate requires epochs 1 and 2')
    differences = []
    for actual, expected in zip(record['epochs'], baseline[:2]):
        for label, a, b in (('train loss', actual['loss'], expected['train']['loss']),
                            ('val macro', actual['validation']['macro_accuracy'], expected['validation']['macro_accuracy'])):
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('Non-finite flip metric')
            matched = math.isclose(a, b, rel_tol=0., abs_tol=1e-6)
            differences.append({'epoch': actual['epoch'], 'metric': label, 'actual': a,
                                'reference': b, 'delta': a-b, 'within_1e6': matched})
            if not report_only and not matched:
                raise ValueError(f'Flip epoch {actual["epoch"]} {label} differs from Stage 1 beyond 1e-6; stop and ask')
    return differences


def verify():
    import torch
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4' or not torch.cuda.is_available():
        raise ValueError('Historical flip gate requires the CUDA server, not the Mac CPU parity record')
    require_tests()
    assets = check_assets()
    baseline = read_baseline_rows()['R1_ce']['history']
    stamp = OUTPUT / 'flip_verification.json'
    identity = {'scope': 'new_server', 'metric_policy': 'report_only', 'gpu': torch.cuda.get_device_name(), 'runtime': runtime(), 'validation_sha256': validation_fingerprint(),
                'assets': assets, 'reference_commit': BASELINE_COMMIT, 'absolute_tolerance': 1e-6}
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if old.get('status') == 'PASS' and all(old.get(k) == v for k, v in identity.items()):
            print('REUSED flip equivalence verification', flush=True)
            return old
    write_json(stamp, {**identity, 'status': 'RUNNING'})
    cfg = stage1_recipe(0, 'R1_ce')
    values = cfg.to_dict()
    values['output_root'] = str(OUTPUT / '_flip_verification')
    values['stage1_run'] = None
    # This diagnostic uses the shared loop directly and writes exclusively under Stage 1b.
    cfg = Stage1bConfig(**values, train_augmentation='flip', stage1b_run='ce_flip')
    directory = OUTPUT / '_flip_verification' / 'server_shared_flip'
    print('Verifying server flip: shared loop, two full epochs vs recorded Stage 1 R1_ce', flush=True)
    from coco_kd import train as current
    with RunLock(directory / '.run.lock'):
        current._train(cfg, 'student', 'ce', directory, stop_after=2)
    start = json.loads((directory / 'run_start.json').read_text())
    history = json.loads((directory / 'history.json').read_text())
    record = {'initial_model_sha256': start['initial_model_sha256'],
              'epochs': [{'epoch': h['epoch'], 'loss': h['train']['loss'], 'validation': h['validation']} for h in history]}
    # Keep the measured values even if the comparison fails.
    write_json(stamp, {**record, **identity, 'status': 'CHECKING'})
    differences = compare_server_record(record, baseline, report_only=True)
    record.update(identity, status='PASS', differences=differences,
                  comparison='exact server checksum; new hardware train loss / val macro report-only (1e-6 comparison recorded)')
    write_json(stamp, record)
    print('FLIP_INITIALIZATION: PASS; epoch metrics report-only', flush=True)
    return record
