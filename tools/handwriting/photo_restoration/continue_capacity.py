"""Resume the wide micro model through epoch 500 with 100-epoch evaluations."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from paths import APP_ASSETS, RESULTS
from .evaluate import FrozenHanjaEvaluator, _save_examples, evaluate_known_photos
from .micro_capacity import SEED, SOURCE, TEST_CONDITIONS, _evaluate, _evaluation_pair
from .model import PhotoInkRestorer, restoration_loss
from .synthesis import synthesize_photo

MICRO = RESULTS / 'photo-restoration-micro-capacity-v1'
DEFAULT_OUTPUT = RESULTS / 'photo-restoration-wide-500-v1'


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def _atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _load_sources():
    selected = json.loads((MICRO / 'split.json').read_text())
    original = json.loads((SOURCE / 'split.json').read_text())['records']
    data = {}
    for split, rows in selected['records'].items():
        index = {r['sample_id']: i for i, r in enumerate(original[split])}
        data[split] = np.load(SOURCE / f'{split}.npy', mmap_mode='r')[[index[r['sample_id']] for r in rows]].copy()
    return data, selected['records']


class _EpochPairs(Dataset):
    def __init__(self, images, epoch, thickness_scale=1.0):
        self.images = images
        self.epoch = epoch
        self.thickness_scale = thickness_scale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        seed = SEED + self.epoch * 100_000 + index * 101
        # A separate RNG leaves the original color/geometry seed unchanged.
        variant = np.random.default_rng(seed + 77_000_000).integers(6)
        if self.thickness_scale == .5 and variant < 2:
            photo, target = _evaluation_pair(self.images[index], seed,
                'thick' if variant == 0 else 'thick_blur', self.thickness_scale)
        else:
            photo, target = synthesize_photo(self.images[index], seed)
        return torch.from_numpy(photo), torch.from_numpy(target)


def _checkpoint(model, optimizer, epoch, settings, history, best):
    return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), epoch=epoch, base_channels=48, settings=settings,
        history=history, best=best, torch_rng=torch.get_rng_state())


def _write_report(output, settings, history, evaluations, state):
    lines = ['# 큰 복원 모델 장기 학습', '', f'상태: {state}', '',
        '50자·원본 1,000개, 채널 48/96/192/384, 4,378,753 파라미터. 기존 B 모델의 1 epoch 가중치에서 이어서 총 500 epoch. 매 100 epoch에 고정 검증·모니터링 평가를 실시한다.', '',
        '원래 B.pt에는 optimizer 상태가 없어 최초 이어 학습 시 AdamW 상태만 새로 시작했다. 이후에는 모델·optimizer·완료 epoch·난수 상태를 매 epoch 원자적으로 저장해 재개할 수 있다. 기존 1회와 완전히 연속된 optimizer trajectory는 아니다.', '',
        '기존과 같은 학습률 0.001, weight decay 0.0001, batch 16, 증강 규칙을 유지한다. 원본 수·채널 수·학습률은 평가 결과를 보고 변경하지 않는다. 매 epoch 색·배경·조명·노이즈를 새로 합성한다.', '',
        '## 100회 단위 평가', '', '|epoch|검증 Top-1|검증 Top-10|모니터링 Top-1|모니터링 Top-10|사진 변형 Top-1|사진 변형 Top-10|', '|---|---:|---:|---:|---:|---:|---:|']
    for result in evaluations:
        v, t, p = result['validation'], result['test'], result['photo_variants']
        lines.append(f"|{result['epoch']}|{v['top1']*100:.1f}%|{v['top10']*100:.1f}%|{t['top1']*100:.1f}%|{t['top10']*100:.1f}%|{p['top1']*100:.1f}%|{p['top10']*100:.1f}%|")
    lines += ['', '검증은 기존 검증 원본 50개×6조건=300개로 최선의 체크포인트를 선택한다. 모니터링 평가는 원본 100개×6조건=600개다. 사진 변형은 앞서 관찰한 32개이며 사진 원본 두 글자는 별도로 저장한다. 반복 관찰하는 이 데이터들은 독립적인 최종 실사 정확도 추정용이 아니다.', '',
        'best.pt 선택은 검증 Top-1, Top-10, Dice 순서다. 모니터링 평가 및 사진 변형 점수로 모델을 선택하거나 학습 설정을 바꾸지 않는다.', '',
        f"완료한 추가 학습 epoch: {len(history)}. 재시작은 같은 명령을 실행하면 last.pt에서 이어진다. 평가는 {settings['evaluate_every']}회 간격과 마지막 회차에 수행한다. 평가 도중 중단되면 재시작 시 해당 평가부터 다시 수행한다.", '',
        '앱의 기존 한자 인식기는 고정되어 있고 배포 파일을 변경하지 않는다. 매 epoch progress.json과 last.pt, 매 100 epoch 평가 이미지·예측·모델을 저장한다. 비정상 수치나 실행 오류가 발생하면 상태를 error로 기록하고 중단한다.', '',
        '실행: `PYTHONPATH=tools/handwriting .venv/bin/python -u -m photo_restoration.continue_capacity`']
    if settings.get('thickness_scale') == .5:
        lines[0] = '# 절반 굵기 증강 100회 실험'
        lines[4] = '기존 B 모델 1회 가중치에서 시작해 총 100회까지 학습. 같은 원본 1,000개·분할·순서·학습률을 사용한다. 입력의 약 1/3에 획 폭 +6px 증강(기존 +12px의 절반), 그중 절반에는 blur radius 3을 추가한다. 정답은 증강된 획 폭을 보존한다.'
        lines[-1] = f'실행: `PYTHONPATH=tools/handwriting .venv/bin/python -u -m photo_restoration.continue_capacity --output {output} --max-epochs 100 --thickness-scale 0.5`'
        lines[8] = '평가도 절반 굵기로 수행하며 test-original-thickness.json에 기존 굵기 조건 평가를 함께 저장한다. 기존 100회 모델을 절반 굵기로 재평가한 baseline-half-thickness.json과 비교해야 학습 효과와 평가 난도 변화를 구분할 수 있다.'
    (output / 'REPORT.md').write_text('\n'.join(lines))


def _evaluate_milestone(model, output, epoch, data, records, classifier, thickness_scale=1.0):
    directory = output / f'epoch-{epoch:03d}'
    directory.mkdir(exist_ok=True)
    validation, examples = _evaluate(model, data['validation'], records['validation'], TEST_CONDITIONS, SEED + 5_000_000, classifier, thickness_scale)
    _atomic_json(directory / 'validation.json', validation)
    _save_examples(examples, directory / 'validation.png')
    test, examples = _evaluate(model, data['test'], records['test'], TEST_CONDITIONS, SEED + 9_000_000, classifier, thickness_scale)
    _atomic_json(directory / 'test.json', test)
    _save_examples(examples, directory / 'test.png')
    if thickness_scale != 1.0:
        original, _ = _evaluate(model, data['test'], records['test'], TEST_CONDITIONS,
            SEED + 9_000_000, classifier)
        _atomic_json(directory / 'test-original-thickness.json', original)
    # The shared diagnostic locates fixtures beside its output directory.
    photos = evaluate_known_photos(model, output, classifier)
    _atomic_json(directory / 'known-photos.json', photos)
    shutil.copyfile(output / 'known-photo-examples.png', directory / 'known-photo-examples.png')
    variants = [r for r in photos['rows'] if r['condition'] != 'original']
    metrics = dict(n=len(variants), top1=float(np.mean([r['methods']['restored']['rank'] == 1 for r in variants])),
        top10=float(np.mean([r['methods']['restored']['rank'] <= 10 for r in variants])))
    return dict(epoch=epoch, validation=validation['summary']['all'], test=test['summary']['all'], photo_variants=metrics,
        original_photos=[r for r in photos['rows'] if r['condition'] == 'original'])


def run(output, max_epochs=500, evaluate_every=100, thickness_scale=1.0):
    torch.set_num_threads(4)
    assert torch.backends.mps.is_available(), 'Apple MPS is required'
    data, records = _load_sources()
    classifier_hash = hashlib.sha256((APP_ASSETS / 'hanja.onnx').read_bytes()).hexdigest()
    settings = dict(seed=SEED, max_epochs=max_epochs, evaluate_every=evaluate_every, base_channels=48,
        parameters=4_378_753, train_sources=1000, batch_size=16, learning_rate=.001,
        weight_decay=.0001, initial_epoch=1, optimizer_reset_at_continuation=True,
        source_checkpoint_sha256=hashlib.sha256((MICRO / 'B.pt').read_bytes()).hexdigest(),
        source_split_sha256=hashlib.sha256((MICRO / 'split.json').read_bytes()).hexdigest(),
        classifier_sha256=classifier_hash)
    if thickness_scale != 1.0:
        settings.update(thickness_scale=thickness_scale, stroke_growth_pixels=6,
            thickness_training_fraction=1/3, thick_blur_training_fraction=1/6,
            blur_radius=3, target_policy='preserve augmented stroke width')
    model = PhotoInkRestorer(48).to('mps')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, evaluations, best = [], [], None
    if (output / 'last.pt').exists():
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['settings'] == settings, 'Resume configuration or source hashes changed'
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        epoch, history, best = saved['epoch'], saved['history'], saved['best']
        if (output / 'evaluations.json').exists():
            evaluations = json.loads((output / 'evaluations.json').read_text())
    else:
        saved = torch.load(MICRO / 'B.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        epoch = int(saved['epochs'])
        assert epoch == 1
        torch.manual_seed(SEED)
        _atomic_json(output / 'settings.json', settings)
        _atomic_checkpoint(output / 'last.pt', _checkpoint(model, optimizer, epoch, settings, history, best))
    _atomic_json(output / 'runtime.json', dict(pid=os.getpid(), started_at=time.time(), resumed_epoch=epoch))
    classifier = FrozenHanjaEvaluator()
    while True:
        if (epoch % evaluate_every == 0 or epoch == max_epochs) and not any(r['epoch'] == epoch for r in evaluations):
            _atomic_json(output / 'progress.json', dict(state='evaluating', epoch=epoch, max_epochs=max_epochs))
            result = _evaluate_milestone(model, output, epoch, data, records, classifier, thickness_scale)
            score = tuple(result['validation'][key] for key in ('top1', 'top10', 'dice'))
            if best is None or score > tuple(best['score']):
                best = dict(epoch=epoch, score=score)
                _atomic_checkpoint(output / 'best.pt', _checkpoint(model, optimizer, epoch, settings, history, best))
            _atomic_checkpoint(output / f'epoch-{epoch:03d}.pt', dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=epoch, base_channels=48, settings=settings))
            evaluations.append(result)
            _atomic_checkpoint(output / 'last.pt', _checkpoint(model, optimizer, epoch, settings, history, best))
            _atomic_json(output / 'evaluations.json', evaluations)
            _write_report(output, settings, history, evaluations, '학습 중' if epoch < max_epochs else '완료')
            print('EVALUATION', json.dumps(result, ensure_ascii=False), flush=True)
        if epoch >= max_epochs:
            break
        epoch += 1
        model.train()
        started = time.time()
        dataset = _EpochPairs(data['train'], epoch, thickness_scale)
        # Preserve the exact per-epoch permutation used in the micro experiment.
        order = np.random.default_rng(SEED + epoch).permutation(len(dataset)).tolist()
        loader = DataLoader(dataset, batch_size=16, sampler=order, num_workers=2,
            multiprocessing_context='spawn', persistent_workers=False)
        total_loss, samples = 0.0, 0
        for step, (photo, target) in enumerate(loader, 1):
            photo, target = photo.to('mps'), target.to('mps')
            optimizer.zero_grad(set_to_none=True)
            loss = restoration_loss(model(photo), target)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}, step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(photo)
            samples += len(photo)
            if step == 1 or step % 16 == 0:
                _atomic_json(output / 'progress.json', dict(state='training', epoch=epoch, step=step, steps=len(loader), max_epochs=max_epochs, loss=total_loss / samples, elapsed_seconds=time.time() - started))
        record = dict(epoch=epoch, samples=samples, loss=total_loss / samples, seconds=time.time() - started)
        history.append(record)
        _atomic_checkpoint(output / 'last.pt', _checkpoint(model, optimizer, epoch, settings, history, best))
        _atomic_json(output / 'history.json', history)
        recent_seconds = float(np.mean([r['seconds'] for r in history[-10:]]))
        progress = dict(state='training' if epoch < max_epochs else 'evaluating', epoch=epoch, completed_epoch=epoch,
            max_epochs=max_epochs, loss=record['loss'], seconds_per_epoch=recent_seconds,
            remaining_training_seconds=(max_epochs - epoch) * recent_seconds, next_evaluation=(epoch // evaluate_every + 1) * evaluate_every)
        _atomic_json(output / 'progress.json', progress)
        _write_report(output, settings, history, evaluations, f'{epoch}회 학습 완료')
        print('EPOCH', json.dumps(record), flush=True)
    assert hashlib.sha256((APP_ASSETS / 'hanja.onnx').read_bytes()).hexdigest() == classifier_hash
    _atomic_json(output / 'result.json', dict(epoch=epoch, best=best, evaluations=evaluations,
        additional_training_seconds=sum(r['seconds'] for r in history), classifier_unchanged=True))
    _atomic_json(output / 'progress.json', dict(state='complete', epoch=epoch, best=best))
    print('COMPLETE', epoch, best, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--thickness-scale', type=float, choices=[1.0, .5], default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.output, max_epochs=args.max_epochs, thickness_scale=args.thickness_scale)
        except BaseException as error:
            _atomic_json(args.output / 'error.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), at=time.time()))
            progress_path = args.output / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            progress.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'error', message=str(error))
            _atomic_json(progress_path, progress)
            raise


if __name__ == '__main__':
    main()
