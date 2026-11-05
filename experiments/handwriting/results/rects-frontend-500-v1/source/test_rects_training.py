"""Real-photo partition isolation and early-stopping contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from . import rects_training_data as data
from .train_rects_frontend import _improved


class ReCTSTrainingContracts(unittest.TestCase):
    def test_decoded_duplicate_cannot_leak_from_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);train=root/'train';held=root/'held';out=root/'out'
            out.mkdir()
            for p in (train,held):
                (p/'img').mkdir(parents=True);(p/'gt_unicode').mkdir()
            names=[]
            for i in range(10):
                name=f'img/{i}.jpg';names.append(name)
                image=Image.new('RGB',(48,48),(30+i*20,220,240));ImageDraw.Draw(image).line((10,24,38,24),fill='black',width=3)
                image.save(train/name)
                (train/'gt_unicode'/f'{i}.json').write_text(json.dumps({'chars':[{'transcription':'一','points':[10,20,38,20,38,28,10,28],'ignore':0}]}))
            (held/'img/duplicate.jpg').write_bytes((train/names[0]).read_bytes())
            (held/'manifest.json').write_text(json.dumps({'selected_images':['img/duplicate.jpg']}))
            (train/'manifest.json').write_text(json.dumps({'selected_images':names}))
            with patch.object(data,'DATA',train),patch.object(data,'HELD_OUT',held):
                result=data.prepare(out,['一'])
            self.assertEqual(result['excluded_duplicate_photos'],[names[0]])
            training={r['pixel_sha256'] for r in result['photos']['train']}
            validation={r['pixel_sha256'] for r in result['photos']['validation']}
            self.assertFalse(training & validation)
            self.assertEqual(len(training)+len(validation),9)
            dataset=data.ReCTSCharacters(out/'train-rgb.npy',result['records']['train'])
            rgb,label=dataset[0]
            self.assertEqual(tuple(rgb.shape),(3,256,256));self.assertEqual(label,0)

    def test_improvement_requires_fixed_minimum_delta(self):
        self.assertFalse(_improved(.501,.5))
        self.assertTrue(_improved(.502,.5))
        self.assertFalse(_improved(.49,.5))


if __name__=='__main__':unittest.main()
