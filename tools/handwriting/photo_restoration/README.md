# Photo ink restoration pilot

Run from the repository root with the existing Python environment:

```sh
PYTHONPATH=tools/handwriting .venv/bin/python -m unittest photo_restoration.test_contracts
PYTHONPATH=tools/handwriting .venv/bin/python -u -m photo_restoration.train
PYTHONPATH=tools/handwriting .venv/bin/python -m photo_restoration.report
```

The pilot uses Apple MPS, two CPU augmentation workers and batch size 16. It
trains a new 1.1M-parameter RGB-to-ink model for three epochs. The existing ONNX
classifier is read-only. A fixed 0.5 foreground cutoff is used for evaluation;
the best checkpoint is chosen by validation recognition Top-1, then mask Dice.

Preparation reuses the existing original-image splits, checks group and pixel
hash isolation, selects 500 classes across grade buckets, and caches clean
128px images. The 20,000 training slots include 5% blank backgrounds, so 19,000
distinct source glyphs actually appear in each epoch. Appearance corruptions
change each epoch; geometry is shared by the RGB input and its target mask.

This is a procedural-background experiment. Real material photos, JPEG,
perspective distortion and nearby glyphs are not included yet. The 6,000 final
test variants represent 1,000 original glyph images, not 6,000 independent
real-world photos. The previously inspected two-glyph photo is a diagnostic.

Results are written under `experiments/handwriting/results/photo-restoration-pilot-v1`.
Use `--output` for a separate run. Existing training checkpoints are not
overwritten; automatic resume is not implemented. The two-glyph diagnostic
expects the existing `photo-color-probe-20261104` directory beside the output.

Training and validation make no changes to the application or its model assets.
