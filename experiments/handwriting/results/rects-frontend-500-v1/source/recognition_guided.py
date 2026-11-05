"""Continuous RGB frontend supervised through a frozen deployed classifier."""
import hashlib
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

from paths import ROOT, RESULTS, APP_ASSETS
from .model import PhotoInkRestorer

sys.path.insert(0, str(ROOT / 'vendor/GlyphWeave'))
from handwriting.inference import Recognizer

CLASSIFIER_CHECKPOINT = RESULTS / 'hanja5000-head-only-v1/head-only-best.pt'
FRONTEND_CHECKPOINT = RESULTS / 'photo-restoration-half-thickness-100-v1/epoch-100.pt'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecognitionGuidedFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.restorer = PhotoInkRestorer(48)

    def forward(self, rgb):
        # Keep the complete frame and continuous ink probabilities. This exact
        # path is used at train and inference time: no threshold or hard crop.
        ink = self.restorer(rgb).sigmoid()
        return F.interpolate(ink, size=(128, 128), mode='bilinear', align_corners=False) * 2 - 1


def load_frozen_classifier(device):
    export = json.loads((APP_ASSETS / 'export_validation.json').read_text())
    assert file_hash(CLASSIFIER_CHECKPOINT) == export['checkpoint_sha256']
    assert file_hash(APP_ASSETS / 'hanja.onnx') == export['model_sha256']
    recognizer = Recognizer(CLASSIFIER_CHECKPOINT, device=device)
    assert recognizer.classes == json.loads((APP_ASSETS / 'classes.json').read_text())
    model = recognizer.model.eval()
    model.requires_grad_(False)
    return model, recognizer.classes


def initialize_frontend(device):
    frontend = RecognitionGuidedFrontend()
    saved = torch.load(FRONTEND_CHECKPOINT, map_location='cpu', weights_only=False)
    frontend.restorer.load_state_dict(saved['model'])
    return frontend.to(device)
