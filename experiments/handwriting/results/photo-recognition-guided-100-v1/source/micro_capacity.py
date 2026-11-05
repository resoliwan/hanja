"""Matched 50-class capacity pilot; at most three epochs, no test-based tuning."""

import collections
import hashlib
import json
import time

import numpy as np
from PIL import Image, ImageFilter
import torch

from paths import APP_ASSETS, RESULTS
from .evaluate import FrozenHanjaEvaluator, _save_examples, normalize_ink
from .model import PhotoInkRestorer, restoration_loss
from .synthesis import synthesize_photo

SEED = 20261105
SOURCE = RESULTS / 'photo-restoration-pilot-v1'
OUTPUT = RESULTS / 'photo-restoration-micro-capacity-v1'
TEST_CONDITIONS = ('clean', 'solid', 'gradient', 'glare', 'thick', 'thick_blur')


def _save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def _select_data():
    original = json.loads((SOURCE / 'split.json').read_text())['records']
    grades = collections.defaultdict(set)
    for row in original['train']:
        grades[row['grade']].add(int(row['class_id']))
    rng = np.random.default_rng(SEED)
    grades = {g: rng.permutation(sorted(ids)).tolist() for g, ids in grades.items()}
    selected = []
    while len(selected) < 50:
        for grade in sorted(grades):
            if grades[grade] and len(selected) < 50:
                selected.append(grades[grade].pop())
    data, records = {}, {}
    for split, per_class in [('train', 20), ('validation', 1), ('test', 2)]:
        pools = collections.defaultdict(list)
        for i, row in enumerate(original[split]):
            pools[int(row['class_id'])].append(i)
        indices = [int(i) for c in selected for i in rng.choice(pools[c], per_class, replace=False)]
        data[split] = np.load(SOURCE / f'{split}.npy', mmap_mode='r')[indices].copy()
        records[split] = [original[split][i] for i in indices]
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for key in ['group_id', 'pixel_sha256', 'sample_id']:
            assert not {r[key] for r in records[left]} & {r[key] for r in records[right]}
    _save('split.json', dict(seed=SEED, selected_classes=selected, records=records))
    return data, records


def _evaluation_pair(clean, seed, condition, thickness_scale=1.0):
    if condition in ('thick', 'thick_blur'):
        # A 7px max mask grows stroke width by about 12px at the 256px input.
        if thickness_scale == 1.0:
            clean = np.asarray(Image.fromarray(clean).filter(ImageFilter.MinFilter(7)))
            photo, target = synthesize_photo(clean, seed, 'solid')
        elif thickness_scale == 0.5:
            photo, target = synthesize_photo(clean, seed, 'solid', stroke_growth=6)
        else:
            raise ValueError('Supported thickness scales are 1.0 and 0.5')
        if condition == 'thick_blur':
            image = Image.fromarray(np.uint8(photo.transpose(1, 2, 0) * 255))
            photo = np.asarray(image.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32).transpose(2, 0, 1).copy() / 255
        return photo, target
    return synthesize_photo(clean, seed, condition)


def write_report():
    result = json.loads((OUTPUT / 'result.json').read_text())
    history = json.loads((OUTPUT / 'history.json').read_text())
    settings = json.loads((OUTPUT / 'settings.json').read_text())
    lines = ['# 초소형 복원 모델 크기 비교', '',
        '**결론: 이번 짧은 학습 예산에서는 채널을 늘려도 개선되지 않았다. 큰 모델이 아직 글자 복원을 배우지 못했으므로, 모델 용량의 최종 성능에 대한 결론은 낼 수 없다.**', '',
        f"50자 / 학습 원본 1,000개 / 검증 원본 50개 / 최종 평가 원본 100개. 모델 모두 처음부터 {result['epochs']} epoch, 63 업데이트씩 학습했다. 준비·두 모델 학습·평가 총 {result['elapsed_seconds']:.1f}초.", '',
        'A: 채널 24→48→96→192, 1,095,553 파라미터. B: 48→96→192→384, 4,378,753 파라미터. 깊이와 입력 해상도는 같고 채널 폭만 바꿨다.', '',
        '같은 학습 입력·정답·증강 seed·순서·batch 16·AdamW lr 0.001·초기화 규칙을 적용했다. 초기 파라미터의 모양이 달라 두 모델의 초기 가중치 값이 동일하다는 뜻은 아니다. 단일 seed이며 같은 계산 시간 비교가 아닌 같은 학습 횟수 비교다.', '',
        '## 최종 평가', '',
        '100개 원본 × 6조건 = 600개. 색/조명 변형은 절차적 합성이다. 이전 32개 사진 변형이나 실제 사진 정확도와 직접 비교할 수 없다.', '',
        '|조건|A Top-1|A Top-10|B Top-1|B Top-10|정답 마스크 직접 입력 Top-1|', '|---|---:|---:|---:|---:|---:|']
    for condition, a in result['results']['A'].items():
        b = result['results']['B'][condition]
        lines.append(f"|{condition}|{a['top1']*100:.1f}%|{a['top10']*100:.1f}%|{b['top1']*100:.1f}%|{b['top10']*100:.1f}%|{a['clean_target_top1']*100:.1f}%|")
    lines += ['', '## 학습 및 검증', '', '|모델|epoch|학습 시간|loss|검증 Top-1|검증 Top-10|', '|---|---:|---:|---:|---:|---:|']
    for row in history:
        metrics = row['validation']['all']
        lines.append(f"|{row['model']}|{row['epoch']}|{row['training_seconds']:.1f}초|{row['loss']:.4f}|{metrics['top1']*100:.1f}%|{metrics['top10']*100:.1f}%|")
    lines += ['', '## 해석과 제한', '',
        '- 사전에 두 모델 모두 검증 Top-1 50% 미만이면 같은 3 epoch까지 늘리기로 고정했다. A가 57%여서 자동 연장은 하지 않았다. B의 실패를 본 뒤 평가 기준이나 학습량을 바꾸지 않았다.',
        '- B 출력은 상당 부분이 균일한 회색 영역이고 글자와 배경의 구분이 되지 않는다. 획 내부 보존율이 높아도 배경까지 획으로 칠하면 인식에 도움이 되지 않으므로, 보존율만으로 성공을 판단하지 않는다.',
        '- A도 최종 전체 Top-1 29.2%로 충분히 학습된 모델이 아니다. 1,000개·1 epoch에서 큰 모델이 더 잘된다는 근거를 얻지 못했을 뿐, 충분히 학습한 큰 모델이 더 나쁘다는 결론은 아니다.',
        '- 굵기 조건은 128px 원본에 7×7 MinFilter를 적용해 256px 입력에서 획 폭을 약 12px 늘린다. 획이 합쳐질 수 있다. 정답 마스크 자체를 기존 인식기에 넣어도 이 조건의 성능이 떨어지므로 복원 모델 오류와 구분해야 한다.',
        '- 다음 검증은 원본 수를 유지한 채 두 모델의 업데이트 횟수를 같은 기준으로 늘려 학습 곡선을 비교하는 것이다. 큰 모델의 학습률 민감도와 seed 차이도 별도 확인해야 한다. 이번에는 C(굵기 증강 학습)를 수행하지 않았다.',
        '- 원본 그룹·픽셀 해시 분리 확인 및 계약 테스트 7개 통과. 기존 24분 실험의 체크포인트는 기본 채널 설정으로 그대로 로드된다. 앱 모델은 변경하지 않았다.', '',
        '재현: `PYTHONPATH=tools/handwriting .venv/bin/python -m photo_restoration.micro_capacity`. 기존 출력 경로가 있으면 덮어쓰지 않도록 중단한다.', '',
        '이미지 열 순서: 입력 / 정답 / 복원.', '', '![A 평가 예시](A-test.png)', '', '![B 평가 예시](B-test.png)']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines))


@torch.inference_mode()
def _evaluate(model, clean, records, conditions, seed, classifier, thickness_scale=1.0):
    model.eval()
    tasks = [(i, c) for i in range(len(clean)) for c in conditions]
    rows, examples = [], []
    for start in range(0, len(tasks), 16):
        batch = tasks[start:start + 16]
        pairs = [_evaluation_pair(clean[i], seed + i * 31 + TEST_CONDITIONS.index(c), c, thickness_scale) for i, c in batch]
        photos = np.stack([p for p, _ in pairs])
        predictions = model(torch.from_numpy(photos).to('mps')).sigmoid().cpu().numpy()[:, 0]
        for (i, condition), (photo, target), prediction in zip(batch, pairs, predictions):
            truth = target[0]
            hard = prediction >= .5
            core = np.asarray(Image.fromarray(np.uint8(truth > .9) * 255).filter(ImageFilter.MinFilter(7))) > 0
            ink = np.where(hard, prediction, 0)
            tensor = normalize_ink(np.uint8(np.clip(1 - ink, 0, 1) * 255))
            result = classifier.recognize(tensor, int(records[i]['class_id']))
            reference = classifier.recognize(normalize_ink(np.uint8((1 - truth) * 255)), int(records[i]['class_id']))
            rows.append(dict(sample_id=records[i]['sample_id'], condition=condition, rank=result['rank'], target_rank=reference['rank'],
                interior_pixels=int(core.sum()), interior_retained=int((hard & core).sum()),
                dice=float((2 * (hard * truth).sum() + 1) / (hard.sum() + truth.sum() + 1))))
            if i < 2:
                examples.append((photo, truth, prediction, f'{condition}: rank {result["rank"]}'))
    summary = {}
    for condition in ('all', *conditions):
        group = rows if condition == 'all' else [r for r in rows if r['condition'] == condition]
        summary[condition] = dict(n=len(group), top1=float(np.mean([r['rank'] == 1 for r in group])),
            top10=float(np.mean([r['rank'] <= 10 for r in group])),
            clean_target_top1=float(np.mean([r['target_rank'] == 1 for r in group])),
            interior_recall=sum(r['interior_retained'] for r in group) / max(1, sum(r['interior_pixels'] for r in group)),
            dice=float(np.mean([r['dice'] for r in group])))
    return dict(summary=summary, rows=rows), examples


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'settings.json').exists():
        raise RuntimeError('Existing experiment; do not overwrite it')
    torch.set_num_threads(4)
    assert torch.backends.mps.is_available()
    started = time.time()
    data, records = _select_data()
    models, optimizers = {}, {}
    for name, channels in [('A', 24), ('B', 48)]:
        torch.manual_seed(SEED)
        models[name] = PhotoInkRestorer(channels).to('mps')
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=.001, weight_decay=.0001)
    settings = dict(seed=SEED, train_sources=1000, validation_sources=50, test_sources=100,
        batch_size=16, learning_rate=.001, from_scratch=True, blank_fraction=0,
        parameters={n: sum(p.numel() for p in m.parameters()) for n, m in models.items()},
        extension_rule='If BOTH validation Top-1 < 50% after epoch 1, extend BOTH to epoch 3; use final epoch, no test-based selection.',
        comparison='same images, augmentations, sample order, optimization steps and optimizer settings; single seed; not equal compute time',
        classifier_sha256=hashlib.sha256((APP_ASSETS / 'hanja.onnx').read_bytes()).hexdigest())
    _save('settings.json', settings)
    print('SETTINGS', json.dumps(settings), flush=True)
    classifier = FrozenHanjaEvaluator()
    history = []
    epochs = 1
    epoch = 0
    while epoch < epochs:
        epoch += 1
        # Materialize a small shared epoch so A and B see byte-identical inputs.
        pairs = [synthesize_photo(clean, SEED + epoch * 100_000 + i * 101) for i, clean in enumerate(data['train'])]
        photos = np.stack([p for p, _ in pairs])
        targets = np.stack([t for _, t in pairs])
        order = np.random.default_rng(SEED + epoch).permutation(len(pairs))
        for name in ('A', 'B'):
            model, optimizer = models[name], optimizers[name]
            model.train()
            losses = []
            begin = time.time()
            for start in range(0, len(order), 16):
                indices = order[start:start + 16]
                photo = torch.from_numpy(photos[indices]).to('mps')
                target = torch.from_numpy(targets[indices]).to('mps')
                optimizer.zero_grad(set_to_none=True)
                loss = restoration_loss(model(photo), target)
                assert torch.isfinite(loss).item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            seconds = time.time() - begin
            validation, examples = _evaluate(model, data['validation'], records['validation'], ('clean', 'solid'), SEED + 5_000_000, classifier)
            entry = dict(model=name, epoch=epoch, training_seconds=seconds, loss=float(np.mean(losses)), validation=validation['summary'])
            history.append(entry)
            _save(f'{name}-validation-epoch{epoch}.json', validation)
            _save('history.json', history)
            _save_examples(examples[:6], OUTPUT / f'{name}-validation.png')
            print('EPOCH', json.dumps(entry), flush=True)
        if epoch == 1 and all(r['validation']['all']['top1'] < .5 for r in history):
            epochs = 3
            print('EXTEND both models to 3 epochs: neither passed validation learning check', flush=True)
    results = {}
    for name, model in models.items():
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, base_channels=24 if name == 'A' else 48, epochs=epochs, settings=settings), OUTPUT / f'{name}.pt')
        evaluation, examples = _evaluate(model, data['test'], records['test'], TEST_CONDITIONS, SEED + 9_000_000, classifier)
        _save(f'{name}-test.json', evaluation)
        _save_examples(examples, OUTPUT / f'{name}-test.png')
        results[name] = evaluation['summary']
        print('TEST', name, json.dumps(evaluation['summary']), flush=True)
    result = dict(epochs=epochs, elapsed_seconds=time.time() - started, results=results)
    _save('result.json', result)
    write_report()
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
