"""Verify deployed-model identity and gradient flow without updating its weights."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .recognition_guided import load_frozen_classifier, initialize_frontend
from .continue_capacity import _load_sources, _EpochPairs
from .evaluate import FrozenHanjaEvaluator


class GuidedContracts(unittest.TestCase):
    def test_gradient_flow_and_frozen_weights(self):
        torch.set_num_threads(4)
        classifier,classes=load_frozen_classifier('mps')
        frontend=initialize_frontend('mps')
        data,records=_load_sources()
        photo,_=_EpochPairs(data['train'],101,.5)[0]
        label=int(records['train'][0]['class_id'])
        self.assertEqual(classes[label],records['train'][0]['character'])
        before={k:v.detach().cpu().clone() for k,v in classifier.state_dict().items()}
        frontend_before=frontend.restorer.output.weight.detach().cpu().clone()
        optimizer=torch.optim.AdamW(frontend.parameters(),lr=1e-4)
        normalized=frontend(photo[None].to('mps'))
        self.assertEqual(tuple(normalized.shape),(1,1,128,128))
        logits=classifier(normalized)
        evaluator=FrozenHanjaEvaluator()
        onnx=evaluator.session.run(None,{'image':normalized.detach().cpu().numpy()})[0]
        np.testing.assert_allclose(logits.detach().cpu().numpy(),onnx,atol=.002,rtol=.001)
        F.cross_entropy(logits,torch.tensor([label],device='mps')).backward()
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in frontend.parameters()),0)
        torch.nn.utils.clip_grad_norm_(frontend.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        self.assertFalse(torch.equal(frontend_before,frontend.restorer.output.weight.detach().cpu()))
        self.assertTrue(all(torch.equal(v.detach().cpu(),before[k]) for k,v in classifier.state_dict().items()))


if __name__=='__main__':unittest.main()
