"""Export the current trained checkpoint and verify mobile inference parity."""
from paths import APP_ASSETS, DATASETS, RESULTS, ROOT
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(ROOT / 'vendor/GlyphWeave'))
from handwriting.inference import Recognizer
from hanja3500_dataset import Hanja3500Dataset

CHECKPOINT = RESULTS / 'hanja5000-head-only-v1/head-only-best.pt'


def _samples(recognizer):
    for name, strokes in [
        ('一', [[(30, 100), (220, 100)]]),
        ('十', [[(30, 100), (220, 100)], [(125, 30), (125, 220)]]),
        ('大', [[(30, 90), (220, 90)], [(130, 30), (115, 140), (35, 220)], [(125, 120), (210, 220)]]),
    ]:
        image = Image.new('L', (256, 256), 255)
        draw = ImageDraw.Draw(image)
        for points in strokes:
            draw.line(points, fill=0, width=7)
        yield name, recognizer.transform(image).unsqueeze(0)
    dataset = Hanja3500Dataset(DATASETS / 'korean-hanja5000-v1', 'evaluation', transform=recognizer.transform)
    try:
        # Exercise new output classes with both real handwriting and held-out fonts.
        for source in ['fonts', 'scut']:
            indices = [i for i, row in enumerate(dataset.rows) if row['grade'] == '사범' and row['source'] == source]
            for i in indices[::max(1, len(indices)//10)][:10]:
                tensor, _ = dataset[i]
                yield dataset.rows[i]['sample_id'], tensor.unsqueeze(0)
    finally:
        dataset.close()


def main():
    APP_ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    recognizer = Recognizer(CHECKPOINT, device='cpu')
    assert len(recognizer.classes) == len(set(recognizer.classes)) == 4998
    expected = (DATASETS / 'korean-hanja5000-v1/classes.txt').read_text().splitlines()
    assert recognizer.classes == expected
    with tempfile.TemporaryDirectory(prefix='hanja-export-') as temporary:
        staged = Path(temporary)
        model_path = staged / 'hanja.onnx'
        torch.onnx.export(recognizer.model, torch.zeros(1, 1, 128, 128), str(model_path),
                          input_names=['image'], output_names=['logits'], opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        assert session.get_outputs()[0].shape == [1, len(recognizer.classes)]
        checks = []
        for name, tensor in _samples(recognizer):
            with torch.inference_mode():
                original = recognizer.model(tensor).numpy()[0]
            exported = session.run(None, {'image': tensor.numpy()})[0][0]
            top = np.argsort(-exported)[:10]
            difference = float(np.max(np.abs(original-exported)))
            assert difference < .001 and np.array_equal(np.argsort(-original)[:10], top), (name, difference)
            checks.append(dict(sample=name, top10=[recognizer.classes[i] for i in top], max_logit_difference=difference))
        (staged / 'classes.json').write_text(json.dumps(recognizer.classes, ensure_ascii=False))
        validation = dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                          checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                          model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                          classes=len(recognizer.classes), checks=checks)
        (staged / 'export_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
        # Publish the model and matching labels only after every conversion check passes.
        for name in ['hanja.onnx', 'classes.json', 'export_validation.json']:
            target = APP_ASSETS / name
            target.write_bytes((staged / name).read_bytes())
        print(json.dumps(dict(classes=len(recognizer.classes), checks=len(checks),
                              max_logit_difference=max(c['max_logit_difference'] for c in checks)), indent=2))


if __name__ == '__main__':
    main()
