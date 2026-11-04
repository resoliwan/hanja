# GlyphWeave v1.0.0 — inference 包

克隆对应版本的源码并安装依赖，将本包解压到项目根目录。单图识别不需要数据集。

```bash
.venv/bin/python predict.py path/to/glyph.png --checkpoint glyphweave-v1.0.0-inference/best.pt
.venv/bin/python app.py --checkpoint glyphweave-v1.0.0-inference/best.pt
```

Windows 将 `.venv/bin/python` 替换为 `.\.venv-win\Scripts\python.exe`。完整评估与训练需要整个 `data/processed/` 目录。

此检查点只包含 EMA 权重与推理元信息，不能用于断点续训。

仅完成单个随机种子实验，未宣称完成基线、消融、分数校准、计时或外部照片基准。加载下载的检查点前，请核对 SHA256SUMS.txt。
