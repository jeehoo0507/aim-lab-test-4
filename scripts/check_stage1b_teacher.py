"""Train and inspect the new LS=0 teacher using the shared Stage 1 code."""
import json

from coco_kd.stage1 import teacher_pass
from coco_kd.stage1b import (OUTPUT, TEACHER, check_assets, read_config, runtime,
                            validation_fingerprint, training_fingerprint, require_tests)
from coco_kd.utils import load_checkpoint, sha256, setup_device, write_json
from scripts.check_teacher_outputs import measure, validate_checkpoint


def train_and_check():
    from coco_kd.train import train
    from coco_kd.models import build_model
    require_tests()
    cfg = read_config('T_LS0')
    print('TEACHER: training seed 0 T_LS0, 30 epochs, flip, LS=0, last.pt', flush=True)
    train(cfg, role='teacher', method='teacher')
    assets = check_assets()
    identity = {'assets': assets, 'runtime': runtime(), 'validation_sha256': validation_fingerprint()}
    path = OUTPUT / 'teacher_checks.json'
    if path.exists():
        old = json.loads(path.read_text())
        if (old.get('status') == 'PASS' and all(old.get(k) == v for k, v in identity.items())
                and teacher_pass('T_LS0', old['train']['kl_to_ls'])):
            print('REUSED teacher output check', flush=True)
            return old
    write_json(path, {**identity, 'status': 'RUNNING'})
    state = load_checkpoint(TEACHER)
    validate_checkpoint(state, 0, 'T_LS0', 0.)
    if state['config'] != cfg.to_dict() or state['code_sha256'] != training_fingerprint():
        raise ValueError('New teacher checkpoint config/source mismatch')
    device = setup_device(cfg)
    model = build_model('teacher', cfg).to(device).requires_grad_(False).eval()
    model.load_state_dict(state['model'])
    del state
    print('TEACHER_CHECK: augmented train once (measurement seed 0), then val', flush=True)
    record = {**identity, 'checkpoint': TEACHER, 'measurement_seed': 0, 'gate_split': 'train',
              'train': measure(model, cfg, 'train', device), 'val': measure(model, cfg, 'val', device)}
    # Reject a replaced file even if it changed while the output check was running.
    if sha256(TEACHER) != assets['teacher_sha256']:
        raise ValueError('Teacher checkpoint changed during output check')
    record['status'] = 'PASS' if teacher_pass('T_LS0', record['train']['kl_to_ls']) else 'FAIL'
    write_json(path, record)
    if record['status'] != 'PASS':
        raise ValueError('Teacher train kl_to_ls <0.02 or non-finite; no student may start')
    print(f"TEACHER_CHECK: PASS, SHA-256={assets['teacher_sha256']}", flush=True)
    return record
