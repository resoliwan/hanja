"""Small correctness checks for paired geometry, empty inputs and inference scale."""

import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from .evaluate import _binary, normalize_ink
from .model import PhotoInkRestorer, restoration_loss
from .synthesis import CONDITIONS, synthesize_photo
from paths import ROOT


class RestorationContracts(unittest.TestCase):
    def setUp(self):
        image = Image.new('L', (128, 128), 255)
        draw = ImageDraw.Draw(image)
        draw.line((20, 40, 100, 40), fill=0, width=5)
        draw.line((60, 20, 60, 110), fill=0, width=5)
        self.clean = np.asarray(image)

    def test_clean_rgb_and_target_have_identical_geometry(self):
        photo, target = synthesize_photo(self.clean, 7, 'clean')
        np.testing.assert_allclose(photo[0], 1 - target[0], atol=1 / 255)
        np.testing.assert_array_equal(photo[0], photo[1])

    def test_each_condition_is_reproducible_and_valid(self):
        for condition in CONDITIONS:
            a, target = synthesize_photo(self.clean, 41, condition)
            b, repeated_target = synthesize_photo(self.clean, 41, condition)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(target, repeated_target)
            self.assertEqual(a.shape, (3, 256, 256))
            self.assertEqual(target.shape, (1, 256, 256))
            self.assertTrue(0 <= a.min() <= a.max() <= 1)
            self.assertTrue(0 < target.sum() < target.size)

    def test_half_thickness_adds_six_pixels_and_preserves_pair_alignment(self):
        clean = np.full((128, 128), 255, np.uint8)
        clean[32:96, 60:64] = 0
        _, original = synthesize_photo(clean, 7, 'clean')
        photo, target = synthesize_photo(clean, 7, 'clean', stroke_growth=6)
        original_width = int((original[0, 128] > .5).sum())
        grown_width = int((target[0, 128] > .5).sum())
        self.assertEqual(grown_width - original_width, 6)
        np.testing.assert_allclose(photo[0], 1 - target[0], atol=1 / 255)

    def test_blank_target_and_empty_normalization(self):
        _, target = synthesize_photo(self.clean, 9, 'texture', blank=True)
        self.assertEqual(float(target.sum()), 0)
        self.assertTrue(np.all(normalize_ink(np.full((256, 256), 255, np.uint8)) == -1))

    def test_binary_polarity_handles_white_on_black(self):
        np.testing.assert_array_equal(_binary(self.clean, True), _binary(255 - self.clean, True))
        self.assertEqual(normalize_ink(self.clean)[0, 0], -1)
        self.assertEqual(float(normalize_ink(self.clean).max()), 1)

    def test_normalization_matches_existing_mobile_reference(self):
        fixtures = ROOT / 'apps/dic/test/fixtures'
        gray = np.fromfile(fixtures / 'ink.rgba', dtype=np.uint8).reshape(64, 64, 4)[:, :, 0]
        expected = np.fromfile(fixtures / 'expected.f32', dtype=np.float32).reshape(128, 128)
        np.testing.assert_allclose(normalize_ink(gray), expected, atol=2 / 127.5)

    def test_model_gradients_include_rgb_input_and_empty_mask(self):
        torch.set_num_threads(2)
        model = PhotoInkRestorer()
        photo = torch.rand(2, 3, 32, 32, requires_grad=True)
        target = torch.zeros(2, 1, 32, 32)
        target[0, :, 10:20, 10:20] = 1
        logits = model(photo)
        self.assertEqual(logits.shape, target.shape)
        loss = restoration_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(photo.grad).all().item())
        self.assertGreater(float(photo.grad.abs().sum()), 0)

    def test_wider_model_keeps_output_contract(self):
        torch.set_num_threads(2)
        small = PhotoInkRestorer()
        wide = PhotoInkRestorer(base_channels=48)
        self.assertGreater(sum(p.numel() for p in wide.parameters()), 3.9 * sum(p.numel() for p in small.parameters()))
        with torch.no_grad():
            self.assertEqual(wide(torch.zeros(1, 3, 32, 32)).shape, (1, 1, 32, 32))


if __name__ == '__main__':
    unittest.main()
