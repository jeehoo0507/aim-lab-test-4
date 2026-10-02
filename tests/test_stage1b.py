import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from coco_kd import stage1b
from coco_kd.config import Config
from coco_kd.data import CocoSubset, loader
from coco_kd.stage1 import INITIAL_SHA256, recipe as stage1_recipe
from coco_kd.stage1b_config import Stage1bConfig
from coco_kd.stage1b_augmentation import (AugmentedCocoSubset, crop_parameters, paired_crop,
                                        mix_batch, mix_parameters)
from coco_kd.synthetic import synthetic_data
from coco_kd.train import train_epoch, distillation_loss, evaluate_test, _train, evaluate
from coco_kd.utils import write_json, load_checkpoint, sha256


def debug_config(tmp_path, augmentation='rrc'):
    return Stage1bConfig(teacher_label_smoothing=0., train_augmentation=augmentation,
                        data_root=str(tmp_path / 'data'), output_root=str(tmp_path / 'out'),
                        device='cpu', model_scale='debug', num_classes=2, student_init='scratch',
                        teacher_pretrained=False, batch_size=2, eval_batch_size=2,
                        accumulation_steps=2, num_workers=0, amp=False, probe_enabled=False,
                        epochs=3, warmup_epochs=0, checkpoint_every=2)


def test_stage1b_required_field_and_fixed_recipes(tmp_path):
    cfg = stage1b.read_config('ce_rrc')
    values = cfg.to_dict()
    del values['train_augmentation']
    write_json(tmp_path / 'missing.json', values)
    with pytest.raises(ValueError, match='train_augmentation'):
        Stage1bConfig.load(tmp_path / 'missing.json', train_augmentation='flip')
    with pytest.raises(TypeError):
        Stage1bConfig(teacher_label_smoothing=0.)
    with pytest.raises(ValueError, match='Invalid train_augmentation'):
        Stage1bConfig.load(teacher_label_smoothing=0., train_augmentation='crop')
    for name in stage1b.RUNS:
        c = stage1b.read_config(name)
        expected = stage1_recipe(0, 'R1_ce').to_dict()
        actual = c.to_dict()
        changes = {k for k in expected if expected[k] != actual[k]}
        assert changes <= {'stage1_run', 'output_root', 'teacher_checkpoint', 'temperature'}
        assert c.stage1_run is None and c.stage1b_run == name
        assert c.temperature == (4. if '_T4' in name else 1.)
        assert c.train_augmentation == ('rrc_mix' if name.endswith('_mix') else 'rrc')
        assert c.teacher_checkpoint == (None if name.startswith('ce_') else stage1b.TEACHER)
        with pytest.raises(ValueError, match='forbids test'):
            evaluate_test(c)
    # Historical JSON still loads with its original schema and values.
    original = Config.load('configs/stage1/seed_0/R1_ce.json')
    assert not hasattr(original, 'train_augmentation')


def test_image_mask_same_geometry_and_nearest_mask():
    mask = np.zeros((260, 280), dtype=np.uint8)
    mask[55:130, 90:155] = 255
    image = np.repeat(mask[..., None], 3, axis=2)
    a, b = paired_crop(Image.fromarray(image), Image.fromarray(mask), (15, 25, 224, 224), True)
    expected = mask[15:239, 25:249][:, ::-1]
    np.testing.assert_array_equal(np.asarray(a)[..., 0], expected)
    np.testing.assert_array_equal(np.asarray(b), expected)
    _, resized = paired_crop(Image.fromarray(image), Image.fromarray(mask), (20, 35, 137, 111), False)
    assert set(np.unique(resized)) <= {0, 255}
    torch.manual_seed(15)
    state = torch.random.get_rng_state().clone()
    first = crop_parameters(Image.fromarray(image), 0, 1, 42)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(123)
    assert first == crop_parameters(Image.fromarray(image), 0, 1, 42)


@pytest.mark.parametrize('augmentation', ['rrc', 'rrc_mix'])
def test_identical_crops_mixes_across_runs_despite_model_rng(tmp_path, augmentation):
    synthetic_data(tmp_path / 'data')
    cfg = debug_config(tmp_path, augmentation)
    baseline = []
    for run in range(3):
        data = AugmentedCocoSubset(cfg.data_root, seed=0)
        data.set_epoch(2)
        records = []
        for step, batch in enumerate(loader(data, cfg, train=True, epoch=2)):
            images, target, params = mix_batch(batch['image'], batch['label'], cfg, 2, step)
            records.append((batch, images, target, params))
            torch.rand(11 + run * 19)  # Different model/teacher RNG consumption.
            np.random.rand(17 + run)
        if run == 0:
            baseline = records
        else:
            for ref, actual in zip(baseline, records):
                assert ref[3] == actual[3]
                for key in ref[0]:
                    assert torch.equal(ref[0][key], actual[0][key])
                assert torch.equal(ref[1], actual[1]) and torch.equal(ref[2], actual[2])
    # Original val/probe loader ignores the training augmentation entirely.
    for split in ('val', 'probe'):
        a = list(loader(CocoSubset(cfg.data_root, split), cfg))
        b = list(loader(CocoSubset(cfg.data_root, split), replace(cfg, train_augmentation='flip')))
        assert all(torch.equal(x['image'], y['image']) for x, y in zip(a, b))


def test_soft_ce_and_mixup_cutmix_area_targets(tmp_path):
    cfg = debug_config(tmp_path, 'rrc_mix')
    images = torch.arange(4 * 3 * 16 * 16).reshape(4, 3, 16, 16).float()
    labels = torch.tensor([0, 1, 0, 1])
    kinds = set()
    for step in range(20):
        mixed, targets, params = mix_batch(images, labels, cfg, 1, step)
        kinds.add(params['kind'])
        lam, box = params['lambda'], params['box']
        assert params['pairs'] == [3, 2, 1, 0]
        if box:
            y0, y1, x0, x1 = box
            assert lam == 1 - ((y1-y0) * (x1-x0) / 256)
            expected = images.clone()
            expected[:, :, y0:y1, x0:x1] = images.flip(0)[:, :, y0:y1, x0:x1]
        else:
            expected = lam * images + (1-lam) * images.flip(0)
        torch.testing.assert_close(mixed, expected)
        q = F.one_hot(labels, 2).float() * .9 + .05
        torch.testing.assert_close(targets, lam*q + (1-lam)*q.flip(0))
        logits = torch.tensor([[2., -1.], [.3, .1], [-2., .5], [.2, .4]], requires_grad=True)
        loss, ce, kd = distillation_loss(logits, targets, None, cfg)
        expected_ce = lam * F.cross_entropy(logits, labels, label_smoothing=.1) + (1-lam)*F.cross_entropy(logits, labels.flip(0), label_smoothing=.1)
        torch.testing.assert_close(loss, expected_ce)
        torch.testing.assert_close(torch.autograd.grad(loss, logits, retain_graph=True)[0], torch.autograd.grad(expected_ce, logits)[0])
        assert kd == 0 and ce == loss.detach()
        teacher_logits = logits.detach().flip(0)
        combined, _, _ = distillation_loss(logits, targets, teacher_logits, replace(cfg, temperature=4.))
        expected_kd = 16 * F.kl_div((logits/4).log_softmax(1), (teacher_logits/4).softmax(1), reduction='batchmean')
        torch.testing.assert_close(combined, .5 * expected_ce + .5 * expected_kd)
    assert kinds == {'cutmix', 'mixup'}
    with pytest.raises(ValueError, match='even'):
        mix_batch(images[:3], labels[:3], cfg, 1, 0)


@pytest.mark.parametrize('augmentation', ['rrc', 'rrc_mix'])
def test_shared_train_epoch_identical_teacher_input_and_run_stream(tmp_path, augmentation):
    synthetic_data(tmp_path / 'data')
    cfg = debug_config(tmp_path, augmentation)
    student_inputs, teacher_inputs = [], []

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, 2))

        def forward(self, x, return_attention=False):
            student_inputs.append(x.detach().clone())
            score = F.dropout(x.mean((2, 3)), p=.1, training=self.training) @ self.weight
            return (score, x.new_zeros((len(x), 196))) if return_attention else score

    class Teacher(torch.nn.Module):
        def forward(self, x, indices=None):
            assert indices is None  # Full KD, all 196 patches.
            teacher_inputs.append(x.detach().clone())
            torch.rand(29)  # Stress independence of subsequent crop/mix draws.
            return x.mean((2, 3))[:, :2]

    model = Student()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    ce = train_epoch(model, None, AugmentedCocoSubset(cfg.data_root, seed=0),
                     torch.optim.SGD(model.parameters(), lr=0), scaler, cfg, torch.device('cpu'), 'ce', 1)
    ce_inputs = list(student_inputs)
    student_inputs.clear()
    kd = train_epoch(model, Teacher(), AugmentedCocoSubset(cfg.data_root, seed=0),
                     torch.optim.SGD(model.parameters(), lr=0), scaler, cfg, torch.device('cpu'), 'full', 1)
    assert len(teacher_inputs) == len(ce_inputs) == 3
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(ce_inputs, student_inputs, teacher_inputs))
    assert ce['augmentation_sha256'] == kd['augmentation_sha256']
    if augmentation == 'rrc_mix':
        assert ce['correct'] is None and kd['correct'] is None
        assert 'not recorded' in kd['accuracy_definition']


def test_flip_matches_pinned_stage1_first_two_epochs_and_checksum(tmp_path):
    from scripts.verify_stage1b_flip import compare_flip
    synthetic_data(tmp_path / 'data', classes=10)
    # Actual DeiT-Ti, same seed and recipe; small synthetic input for local CPU verification.
    # No platform reference hash in this test. A separate server gate checks real COCO history.
    cfg = replace(stage1_recipe(0, 'R1_ce'), device='cpu', amp=False,
                  data_root=str(tmp_path / 'data'), output_root=str(tmp_path / 'out'),
                  batch_size=2, eval_batch_size=2, max_train_batches=2)
    record = compare_flip(cfg, tmp_path / 'comparison')
    assert record['initial_model_sha256'] == record['reference_initial_model_sha256']
    assert record['scope'] == 'same_platform_cpu'
    assert [row['epoch'] for row in record['epochs']] == [1, 2]
    assert all(row['validation']['samples'] == 20 for row in record['epochs'])
    import os
    if os.environ.get('STAGE1B_CPU_PARITY_RECORD'):
        write_json(os.environ['STAGE1B_CPU_PARITY_RECORD'], record)


def test_augmented_resume_matches_uninterrupted(tmp_path):
    synthetic_data(tmp_path / 'data')
    cfg = debug_config(tmp_path, 'rrc_mix')
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    _train(cfg, 'student', 'ce', full, None)
    _train(cfg, 'student', 'ce', resumed, 1)
    _train(cfg, 'student', 'ce', resumed, None)
    a, b = load_checkpoint(full / 'last.pt'), load_checkpoint(resumed / 'last.pt')
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    x, y = [json.loads((d / 'history.json').read_text()) for d in (full, resumed)]
    assert all(a['train']['augmentation_sha256'] == b['train']['augmentation_sha256'] for a, b in zip(x, y))


def test_stage1b_gate_boundaries_and_all_five_completion():
    rows = {name: {'mean_pp': v} for name, v in zip(stage1b.RUNS, [60., 61.5, 61.49, 63., 64.5])}
    result = stage1b.decisions(rows)
    assert result['decision'] == 'DONE'
    assert [v['status'] for v in result['runs'].values()] == ['PASS', 'FAIL', 'PASS']
    assert result['runs']['LS0_T4_rrc_mix']['baseline'] == 'ce_rrc_mix'
    rows.pop('LS0_T4_rrc_mix')
    assert stage1b.decisions(rows)['decision'] == 'WAIT_RUNS'


def test_baseline_and_assets_are_pinned_and_read_only(tmp_path, monkeypatch):
    baseline = stage1b.read_baseline_rows()
    assert baseline['R1_ce']['mean_pp'] == pytest.approx(60.78)
    monkeypatch.chdir(tmp_path)
    p = tmp_path / 'data/coco_single/manifest.json'
    p.parent.mkdir(parents=True)
    p.write_text('wrong')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        stage1b.check_assets()
    assert p.read_text() == 'wrong'


def test_stage1b_report_only_writes_own_reports_and_five_curves(tmp_path, monkeypatch):
    from coco_kd import stage1b_report as report
    before = {p: sha256(p) for p in Path('reports/stage1').rglob('*') if p.is_file()}
    rows = {}
    for name, value in zip(stage1b.RUNS, [.60, .62, .615, .63, .645]):
        history = [{'epoch': e, 'validation': {'macro_accuracy': value}} for e in range(1, 101)]
        rows[name] = {'mean_pp': value*100, 'history': history}
        for filename in ('config.json', 'result.json', 'run_start.json'):
            write_json(tmp_path / 'out/seed_0' / name / filename, {'synthetic': True})
        write_json(tmp_path / 'out/seed_0' / name / 'history.json', history)
        (tmp_path / 'out/seed_0' / name / 'test_metrics.json').write_text('MUST NOT READ')
    dest = tmp_path / 'report'
    dest.mkdir()
    monkeypatch.setattr(report, 'REPORT', dest)
    monkeypatch.setattr(report, 'OUTPUT', tmp_path / 'out')
    monkeypatch.setattr(report, 'config_audit', lambda: None)
    monkeypatch.setattr(report, 'read_rows', lambda: rows)
    monkeypatch.setattr(report, 'run_dir', lambda name: tmp_path / 'out/seed_0' / name)
    result = report.summary()
    assert result['decision'] == 'DONE'
    text = (dest / 'SUMMARY.md').read_text()
    assert all(name in text for name in stage1b.RUNS)
    assert '60.7800' in text and (dest / 'val_curves.png').stat().st_size > 0
    assert not list(dest.rglob('test_metrics.json'))
    assert all(sha256(p) == digest for p, digest in before.items())


def test_all_five_launch_before_poll_and_failure_stops_peers(tmp_path, monkeypatch):
    from scripts import plan_stage1b as plan
    monkeypatch.setattr(plan, 'OUTPUT', tmp_path)
    launched = []

    class Process:
        def __init__(self, code):
            self.code, self.pid, self.terminated = code, len(launched) + 1, False

        def poll(self):
            assert len(launched) == 5  # Concurrent launch, not five serial jobs.
            return self.code

        def terminate(self):
            self.terminated, self.code = True, -15

        def wait(self, timeout=None):
            return self.code

    def launch(cmd, **kwargs):
        p = Process(0)
        launched.append((cmd, p))
        return p

    monkeypatch.setattr(plan.subprocess, 'Popen', launch)
    plan.run_batch()
    assert [cmd[-1] for cmd, _ in launched] == list(stage1b.RUNS)
    launched.clear()

    def fail(cmd, **kwargs):
        p = Process(1 if not launched else None)
        launched.append((cmd, p))
        return p

    monkeypatch.setattr(plan.subprocess, 'Popen', fail)
    with pytest.raises(RuntimeError, match='exited 1'):
        plan.run_batch()
    assert all(p.terminated for _, p in launched[1:])


def test_all_gate_failures_still_finish_normally(tmp_path, monkeypatch, capsys):
    from scripts import plan_stage1b as plan, verify_stage1b_flip as verification
    from coco_kd import utils
    monkeypatch.setattr(plan, 'OUTPUT', tmp_path)
    monkeypatch.setattr(plan, 'require_tests', lambda: None)
    monkeypatch.setattr(plan, 'check_assets', lambda: {})
    monkeypatch.setattr(plan, 'read_baseline_rows', lambda: {})
    monkeypatch.setattr(plan, 'config_audit', lambda: None)
    monkeypatch.setattr(utils, 'setup_device', lambda cfg: None)
    monkeypatch.setattr(torch, '__version__', '2.5.1+cu124')
    monkeypatch.setattr(torch.version, 'cuda', '12.4')
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'mock GPU')
    events = []
    monkeypatch.setattr(verification, 'verify', lambda: events.append('verify'))
    monkeypatch.setattr(plan, 'run_batch', lambda: events.append('five runs'))
    monkeypatch.setattr(plan, 'summary', lambda: {'decision': 'DONE', 'runs': {k: {'status': 'FAIL'} for k in stage1b.PAIRS}})
    assert plan.execute(5) is None
    assert events == ['verify', 'five runs']
    assert 'STAGE1B_DONE:' in capsys.readouterr().out
    assert not (tmp_path / 'STOP.json').exists()
    with pytest.raises(ValueError, match='--jobs 5'):
        plan.execute(4)


def test_stage1b_setup_keeps_environment_inside_clone(tmp_path):
    import os
    import shutil
    import subprocess
    clone = tmp_path / 'clone with spaces'
    clone.mkdir()
    shutil.copyfile(stage1b.ROOT / 'setup_stage1b.sh', clone / 'setup_stage1b.sh')
    tools = tmp_path / 'tools'
    tools.mkdir()
    fake_uv = tools / 'uv'
    fake_uv.write_text('#!/bin/sh\nenv > "$PWD/environment.txt"\n')
    fake_uv.chmod(0o755)
    keys = ('UV_CACHE_DIR', 'UV_PYTHON_INSTALL_DIR', 'UV_TOOL_DIR', 'TORCH_HOME', 'XDG_CACHE_HOME', 'TMPDIR')
    env = dict(os.environ, PATH=f'{tools}:{os.environ["PATH"]}', **{k: '/outside' for k in keys})
    subprocess.run(['bash', 'setup_stage1b.sh', 'install'], cwd=clone, env=env, check=True, capture_output=True)
    values = dict(line.split('=', 1) for line in (clone / 'environment.txt').read_text().splitlines() if '=' in line)
    assert all(Path(values[k]).is_relative_to(clone) for k in keys)


def test_server_flip_gate_exact_hash_and_absolute_tolerance():
    from copy import deepcopy
    from scripts.verify_stage1b_flip import compare_server_record
    baseline = [{'epoch': e, 'train': {'loss': 2. / e}, 'validation': {'macro_accuracy': .2 * e}}
                for e in (1, 2)]
    record = {'initial_model_sha256': INITIAL_SHA256,
              'epochs': [{'epoch': r['epoch'], 'loss': r['train']['loss'], 'validation': dict(r['validation'])}
                         for r in baseline]}
    compare_server_record(record, baseline)
    for key in ('loss', 'macro_accuracy'):
        within = deepcopy(record)
        if key == 'loss':
            within['epochs'][0]['loss'] += .9e-6
        else:
            within['epochs'][0]['validation'][key] += .9e-6
        compare_server_record(within, baseline)
        bad = deepcopy(record)
        if key == 'loss':
            bad['epochs'][1]['loss'] += 1.1e-6
        else:
            bad['epochs'][1]['validation'][key] += 1.1e-6
        with pytest.raises(ValueError, match='beyond 1e-6'):
            compare_server_record(bad, baseline)
    with pytest.raises(ValueError, match='checksum'):
        compare_server_record({**record, 'initial_model_sha256': 'Mac-local-only'}, baseline)
    with pytest.raises(ValueError, match='epochs 1 and 2'):
        compare_server_record({**record, 'epochs': record['epochs'][:1]}, baseline)
    nan = deepcopy(record)
    nan['epochs'][0]['loss'] = float('nan')
    with pytest.raises(ValueError, match='beyond 1e-6'):
        compare_server_record(nan, baseline)


def test_cpu_parity_stamp_cannot_authorize_server_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(stage1b, 'OUTPUT', tmp_path)
    monkeypatch.setattr(stage1b, 'validation_fingerprint', lambda: 'source')
    write_json(tmp_path / 'flip_verification.json', {'status': 'PASS', 'scope': 'same_platform_cpu',
               'validation_sha256': 'source', 'runtime': stage1b.runtime(), 'initial_model_sha256': INITIAL_SHA256})
    with pytest.raises(ValueError, match='equivalence'):
        stage1b.require_flip_check()


def test_failed_server_flip_gate_records_stop_without_starting_runs(tmp_path, monkeypatch):
    import sys
    from scripts import plan_stage1b as plan, verify_stage1b_flip as verification
    from coco_kd import utils
    monkeypatch.setattr(plan, 'OUTPUT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['plan_stage1b', 'run'])
    monkeypatch.setattr(plan, 'summary', lambda: {})
    monkeypatch.setattr(plan, 'require_tests', lambda: None)
    monkeypatch.setattr(plan, 'check_assets', lambda: {})
    monkeypatch.setattr(plan, 'read_baseline_rows', lambda: {})
    monkeypatch.setattr(plan, 'config_audit', lambda: None)
    monkeypatch.setattr(utils, 'setup_device', lambda cfg: None)
    monkeypatch.setattr(torch, '__version__', '2.5.1+cu124')
    monkeypatch.setattr(torch.version, 'cuda', '12.4')
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'mock A5000')
    launched = []
    monkeypatch.setattr(plan, 'run_batch', lambda: launched.append(True))

    def fail():
        raise ValueError('Flip checksum does not match recorded Stage 1 R1_ce')

    monkeypatch.setattr(verification, 'verify', fail)
    with pytest.raises(ValueError, match='checksum'):
        plan.main()
    assert not launched
    assert json.loads((tmp_path / 'STOP.json').read_text())['decision'] == 'STOP_ERROR'
