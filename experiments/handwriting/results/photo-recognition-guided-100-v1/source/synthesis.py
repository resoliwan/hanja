"""Deterministic paired augmentations; geometry precedes photo-only corruption."""

import numpy as np
from PIL import Image, ImageFilter

CONDITIONS = ('clean', 'solid', 'gradient', 'texture', 'glare', 'low_contrast')


def synthesize_photo(clean, seed, condition=None, blank=False, stroke_growth=0):
    rng = np.random.default_rng(seed)
    condition = condition or rng.choice(CONDITIONS, p=[.12, .20, .16, .20, .22, .10])
    mask = Image.fromarray(255 - clean).resize((256, 256), Image.Resampling.BICUBIC)
    if stroke_growth:
        if stroke_growth < 0 or stroke_growth % 2:
            raise ValueError('stroke_growth must be a nonnegative even pixel count')
        mask = mask.filter(ImageFilter.MaxFilter(stroke_growth + 1))
    if condition != 'clean':
        scale = rng.uniform(.80, 1.08)
        offset_x, offset_y = rng.uniform(-12, 12, 2)
        mask = mask.transform((256, 256), Image.Transform.AFFINE,
            (1 / scale, rng.uniform(-.04, .04), 128 * (1 - 1 / scale) + offset_x,
             0, 1 / scale, 128 * (1 - 1 / scale) + offset_y),
            resample=Image.Resampling.BICUBIC)
        mask = mask.rotate(rng.uniform(-7, 7), Image.Resampling.BICUBIC)
    alpha = np.asarray(mask, dtype=np.float32) / 255
    if blank:
        alpha = np.zeros_like(alpha)
    target = alpha.copy()
    y, x = np.mgrid[0:256, 0:256].astype(np.float32) / 255
    background = rng.uniform(.05, .95, 3).astype(np.float32)
    foreground = rng.uniform(0, 1, 3).astype(np.float32)
    # Both polarities and primaries occur; avoid making color imply glyph class.
    if rng.random() < .35:
        foreground = np.eye(3, dtype=np.float32)[rng.integers(3)]
    if condition == 'clean':
        background, foreground = np.ones(3), np.zeros(3)
    elif condition == 'low_contrast':
        foreground = np.clip(background + rng.normal(0, .13, 3), 0, 1)
    elif np.linalg.norm(foreground - background) < .45:
        foreground = 1 - background
    bg = np.broadcast_to(background, (256, 256, 3)).copy()
    fg = np.broadcast_to(foreground, (256, 256, 3)).copy()
    if condition == 'gradient':
        fg = foreground * (1 - x[..., None]) + rng.uniform(0, 1, 3) * x[..., None]
        bg += (.25 * (x - .5))[..., None]
    if condition == 'texture':
        wave = np.sin(x * rng.uniform(20, 80) + y * rng.uniform(5, 25))
        coarse = Image.fromarray(rng.integers(0, 256, (12, 12), dtype=np.uint8))
        noise = np.asarray(coarse.resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32) / 255 - .5
        bg += (.08 * wave + .25 * noise)[..., None]
    photo = np.clip(bg, 0, 1) * (1 - alpha[..., None]) + fg * alpha[..., None]
    if condition == 'glare':
        center = rng.uniform(.25, .75)
        axis = x if rng.random() < .5 else y
        field = rng.uniform(.35, .85) * np.exp(-((axis - center) / rng.uniform(.10, .30)) ** 2)
        light = 1.0 if rng.random() < .7 else 0.0
        photo = photo * (1 - field[..., None]) + light * field[..., None]
    if condition != 'clean':
        photo += rng.normal(0, rng.uniform(0, .018), photo.shape).astype(np.float32)
    photo = Image.fromarray(np.uint8(np.clip(photo, 0, 1) * 255))
    if condition != 'clean' and rng.random() < .5:
        photo = photo.filter(ImageFilter.GaussianBlur(rng.uniform(.2, 1.0)))
    return np.asarray(photo, dtype=np.float32).transpose(2, 0, 1).copy() / 255, target[None]
