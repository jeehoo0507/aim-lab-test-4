"""Fixed Stage 1b recipes, asset provenance and validation-only decisions."""
import hashlib
import json
import platform
from pathlib import Path

from .stage1 import ROOT, MANIFEST_SHA256, INITIAL_SHA256, mean_last10, recipe as stage1_recipe, require_hash
from .stage1b_config import Stage1bConfig
from .utils import sha256, source_fingerprint, write_json

OUTPUT = Path('outputs/stage1b_aug')
REPORT = Path('reports/stage1b')
BASELINE_COMMIT = 'c5a826d'
TEACHER = 'outputs/stage1_ls_gate/seed_0/T_LS0/last.pt'
TEACHER_SHA256 = '6e31acca311a19b307a722008ef6c7f65259445ec04217f7d86f8486faf293d6'
RUNS = ('ce_rrc', 'LS0_T1_rrc', 'LS0_T4_rrc', 'ce_rrc_mix', 'LS0_T4_rrc_mix')
PAIRS = {'LS0_T1_rrc': 'ce_rrc', 'LS0_T4_rrc': 'ce_rrc', 'LS0_T4_rrc_mix': 'ce_rrc_mix'}


def recipe(name):
    if name not in RUNS:
        raise ValueError('Unknown Stage 1b run')
    values = stage1_recipe(0, 'R1_ce').to_dict()
    values.update(output_root=str(OUTPUT), stage1_run=None, stage1b_run=name,
                  train_augmentation='rrc_mix' if name.endswith('_mix') else 'rrc',
                  temperature=4. if '_T4' in name else 1.,
                  teacher_checkpoint=None if name.startswith('ce_') else TEACHER)
    return Stage1bConfig(**values)


def config_path(name):
    return ROOT / 'configs/stage1b' / f'{name}.json'


def read_config(name):
    cfg = Stage1bConfig.load(config_path(name))
    if cfg.to_dict() != recipe(name).to_dict():
        raise ValueError(f'Stage 1b fixed config changed: {name}; stop and ask')
    return cfg


def method(name):
    return 'ce' if name.startswith('ce_') else 'full'


def run_dir(name):
    return OUTPUT / 'seed_0' / name


def training_fingerprint():
    digest = hashlib.sha256(source_fingerprint().encode())
    paths = list((ROOT / 'coco_kd').glob('stage1b*.py'))
    paths += [ROOT / 'scripts/plan_stage1b.py', ROOT / 'scripts/verify_stage1b_flip.py']
    paths += list((ROOT / 'configs/stage1b').glob('*.json'))
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validation_fingerprint():
    digest = hashlib.sha256(training_fingerprint().encode())
    for path in sorted([ROOT / 'setup_stage1b.sh', ROOT / 'pyproject.toml', *ROOT.glob('tests/*.py')]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime():
    import torch
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda': torch.version.cuda}


def require_tests():
    record = json.loads((OUTPUT / 'tests.json').read_text())
    if record.get('returncode') != 0 or record.get('validation_sha256') != validation_fingerprint():
        raise ValueError('Run Stage 1b tests on current source before GPU work')
    if any(record.get(k) != v for k, v in runtime().items()):
        raise ValueError('Stage 1b tests must pass on this server/environment')


def require_flip_check():
    record = json.loads((OUTPUT / 'flip_verification.json').read_text())
    if (record.get('status') != 'PASS' or record.get('scope') != 'server_a5000'
            or record.get('validation_sha256') != validation_fingerprint()
            or record.get('initial_model_sha256') != INITIAL_SHA256):
        raise ValueError('Stage 1b flip equivalence check is required')
    if record.get('runtime') != runtime():
        raise ValueError('Flip equivalence check must pass on this server')


def check_assets():
    return {'manifest_sha256': require_hash(Path('data/coco_single/manifest.json'), MANIFEST_SHA256),
            'teacher_sha256': require_hash(TEACHER, TEACHER_SHA256)}


def validate_training_run(cfg, role, run_method):
    if (OUTPUT / 'STOP.json').exists():
        raise ValueError('Stage 1b STOP recorded; stop and ask before retry')
    if cfg.to_dict() != read_config(cfg.stage1b_run).to_dict() or (role, run_method) != ('student', method(cfg.stage1b_run)):
        raise ValueError('Stage 1b config/role/method mismatch')
    require_tests()
    require_flip_check()
    check_assets()


def read_baseline_rows():
    pins = json.loads((ROOT / 'configs/stage1b/BASELINE_SHA256.json').read_text())
    for relative, expected in pins['files'].items():
        require_hash(ROOT / relative, expected)
    rows = {}
    for name in ('R1_ce', 'R2_full_LS01best_T1', 'LS01_T4', 'LS0_T1', 'LS0_T4'):
        directory = ROOT / 'reports/stage1/seed_0' / name
        history = json.loads((directory / 'history.json').read_text())
        rows[name] = {'mean_pp': mean_last10(history), 'history': history}
    return rows


def read_rows():
    rows = {}
    for name in RUNS:
        directory = run_dir(name)
        if not (directory / 'result.json').exists():
            continue
        result = json.loads((directory / 'result.json').read_text())
        expected = read_config(name).to_dict()
        teacher_hash = None if method(name) == 'ce' else TEACHER_SHA256
        if (result['config'] != expected or result['metadata_sha256'] != MANIFEST_SHA256
                or result['code_sha256'] != training_fingerprint()
                or result['teacher_sha256'] != teacher_hash or result['initial_model_sha256'] != INITIAL_SHA256
                or result['last_epoch'] != 100 or result['partial_training'] or result['test_evaluated']
                or result['role'] != 'student' or result['method'] != method(name)):
            raise ValueError(f'Invalid Stage 1b result provenance: {name}')
        history = json.loads((directory / 'history.json').read_text())
        rows[name] = {'mean_pp': mean_last10(history), 'history': history}
    for kd, ce in PAIRS.items():
        if kd in rows and ce in rows:
            a, b = rows[kd]['history'], rows[ce]['history']
            if any(not x['train'].get('augmentation_sha256') or
                   x['train']['augmentation_sha256'] != y['train'].get('augmentation_sha256') for x, y in zip(a, b)):
                raise ValueError(f'Crop/mix stream differs between {kd} and {ce}')
    return rows


def decisions(rows):
    rules = {}
    for kd, ce in PAIRS.items():
        delta = rows[kd]['mean_pp'] - rows[ce]['mean_pp'] if kd in rows and ce in rows else None
        rules[kd] = {'baseline': ce, 'delta_pp': delta,
                     'status': 'NOT_RUN' if delta is None else 'PASS' if delta >= 1.5 else 'FAIL'}
    return {'decision': 'DONE' if all(name in rows for name in RUNS) else 'WAIT_RUNS', 'runs': rules}


def config_audit():
    lines = ['# Stage 1b config audit', '',
             '| run | augmentation | T | alpha | LS | LR | epochs | teacher SHA-256 |',
             '|---|---|---|---|---|---|---|---|']
    for name in RUNS:
        cfg = read_config(name)
        unused = method(name) == 'ce'
        temp = '1 (미사용)' if unused else str(cfg.temperature)
        alpha = '0.5 (미사용)' if unused else str(cfg.kd_alpha)
        digest = '미사용' if unused else sha256(TEACHER) if Path(TEACHER).is_file() else 'UNAVAILABLE'
        lines.append(f'| {name} | {cfg.train_augmentation} | {temp} | {alpha} | {cfg.label_smoothing} | {cfg.scratch_lr} | {cfg.epochs} | {digest} |')
    lines.extend(['', f'Teacher required SHA-256: `{TEACHER_SHA256}`.',
                  f'Manifest required SHA-256: `{MANIFEST_SHA256}`.',
                  'seed 0; scratch DeiT-Ti; batch 32 × accumulation 4; workers 0; threads 2; AMP; drop_path 0.1.',
                  'AdamW wd 0.05; warmup 5; cosine min 1e-6; grad clip 1.0; Probe off; test forbidden.',
                  'RRC: 224, scale (0.08, 1), ratio (3/4, 4/3), bicubic antialias=True; paired nearest mask; horizontal flip p=0.5.',
                  'Mix: batch mode, probability 1, mixup alpha 0.8 / cutmix alpha 1.0 selected 50/50, reverse-batch pairs, corrected area lambda.',
                  'CE soft targets contain LS=0.1 once; teacher sees the identical mixed image; mixed train accuracy is null.',
                  'Each KD PASS iff delta against same-augmentation CE >=1.5 pp; all five runs finish; no seed 1.'])
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'CONFIG_AUDIT.md').write_text('\n'.join(lines) + '\n')
