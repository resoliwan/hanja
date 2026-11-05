"""Checkpoint and deterministic continuation contracts for the long-running job."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .continue_capacity import _EpochPairs, _atomic_checkpoint, _load_sources
from .model import PhotoInkRestorer, restoration_loss
from .synthesis import synthesize_photo
from .micro_capacity import SEED


class ContinuationContracts(unittest.TestCase):
    def test_same_source_images_and_augmentation_as_micro_run(self):
        data, records = _load_sources()
        self.assertEqual(data['train'].shape, (1000, 128, 128))
        self.assertEqual(len(records['validation']), 50)
        self.assertEqual(len(records['test']), 100)
        photo, target = _EpochPairs(data['train'], 2)[31]
        expected = synthesize_photo(data['train'][31], SEED + 2 * 100_000 + 31 * 101)
        np.testing.assert_array_equal(photo.numpy(), expected[0])
        np.testing.assert_array_equal(target.numpy(), expected[1])

    def test_atomic_checkpoint_preserves_optimizer_next_update(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        model = PhotoInkRestorer(4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        photo = torch.rand(2, 3, 32, 32)
        target = (torch.rand(2, 1, 32, 32) > .85).float()

        def step(network, update):
            update.zero_grad(set_to_none=True)
            restoration_loss(network(photo), target).backward()
            update.step()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            _atomic_checkpoint(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=2))
            self.assertFalse(path.with_suffix('.tmp').exists())
            saved = torch.load(path, map_location='cpu', weights_only=False)
            restored = PhotoInkRestorer(4)
            restored.load_state_dict(saved['model'])
            resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
            resumed_optimizer.load_state_dict(saved['optimizer'])
            step(model, optimizer)
            step(restored, resumed_optimizer)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
