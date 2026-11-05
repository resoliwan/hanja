# GlyphWeave v1.0.0 — inference package

Clone the matching source release and install its dependencies, then extract this package into the project root. Single-image inference does not require the dataset.

```bash
.venv/bin/python predict.py path/to/glyph.png --checkpoint glyphweave-v1.0.0-inference/best.pt
.venv/bin/python app.py --checkpoint glyphweave-v1.0.0-inference/best.pt
```

On Windows, replace `.venv/bin/python` with `.\.venv-win\Scripts\python.exe`. Complete evaluation and training require the entire `data/processed/` directory.

This checkpoint contains EMA weights and inference metadata only. It cannot resume training.

Only one seed was run. No baseline, ablations, calibrated confidence, timing, or external-photo benchmark is claimed. Check SHA256SUMS.txt before loading downloaded checkpoints.
