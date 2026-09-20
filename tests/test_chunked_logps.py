"""Compare chunked projection/logps and gradients against the unchunked path."""
import unittest

import torch
from torch.nn import functional as F

from spade.swift_backend.chunked_logps import chunked_linear_logps


class ChunkedLogpsTests(unittest.TestCase):
    def compare(self, *, dtype=torch.float64, device="cpu", frozen=False, entropy=False):
        torch.manual_seed(17)
        h = torch.randn(2, 19, 13, dtype=dtype, device=device, requires_grad=True)
        w = (torch.randn(71, 13, dtype=dtype, device=device) * 0.1).requires_grad_(not frozen)
        b = torch.randn(71, dtype=dtype, device=device, requires_grad=not frozen)
        ids = torch.randint(71, (2, 19), device=device)
        scale = torch.randn(2, 19, dtype=dtype, device=device)
        inputs = [h] if frozen else [h, w, b]
        raw = F.linear(h, w, b) / 0.6
        reference = raw.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        reference_entropy = -(raw.log_softmax(-1).exp() * raw.log_softmax(-1)).sum(-1)
        reference_loss = (reference * scale).sum()
        if entropy:
            reference_loss += reference_entropy.sum() * 0.1
        grads = torch.autograd.grad(reference_loss, inputs)
        actual, actual_entropy = chunked_linear_logps(h, w, ids, bias=b, temperature=0.6,
                                                     chunk_size=7, compute_entropy=entropy)
        loss = (actual * scale).sum()
        if entropy:
            loss += actual_entropy.sum() * 0.1
        actual_grads = torch.autograd.grad(loss, inputs)
        tol = dict(atol=0.04, rtol=0.04) if dtype == torch.bfloat16 else dict(atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(actual, reference, **tol)
        if entropy:
            torch.testing.assert_close(actual_entropy, reference_entropy, **tol)
        else:
            self.assertIsNone(actual_entropy)
        for a, r in zip(actual_grads, grads):
            torch.testing.assert_close(a, r, **tol)
        with torch.no_grad():
            inference, _ = chunked_linear_logps(h, w, ids, bias=b, temperature=0.6, chunk_size=5)
        torch.testing.assert_close(inference, reference, **tol)

    def test_logps_and_all_parameter_gradients(self):
        self.compare()

    def test_frozen_head_still_backpropagates_to_backbone(self):
        self.compare(frozen=True)

    def test_entropy_gradients(self):
        self.compare(entropy=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_bfloat16_cuda(self):
        self.compare(dtype=torch.bfloat16, device="cuda", frozen=True)

    def test_training_does_not_save_vocabulary_sized_activations(self):
        h = torch.randn(2, 29, 11, requires_grad=True)
        w = torch.randn(103, 11)
        ids = torch.randint(103, (2, 29))
        saved_shapes = []

        def save(tensor):
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            logps, _ = chunked_linear_logps(h, w, ids, chunk_size=7)
        self.assertFalse(any(len(shape) > 1 and shape[-1] == 103 for shape in saved_shapes), saved_shapes)
        logps.sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())


if __name__ == "__main__":
    unittest.main()
