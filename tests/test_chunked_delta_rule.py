"""Parity across recurrent-state boundaries in the real Qwen fallback."""
import unittest

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from spade.swift_backend.chunked_delta_rule import checkpointed_delta_rule


class DeltaRuleTests(unittest.TestCase):
    def test_no_grad_still_segments_and_preserves_state(self):
        torch.manual_seed(32)
        q, k, v = [torch.randn(1, 529, 2, 8) for _ in range(3)]
        g, beta = -torch.rand(1, 529, 2), torch.rand(1, 529, 2)
        state = torch.randn(1, 2, 8, 8) * 0.1
        lengths = []

        def tracked(*args, **kwargs):
            lengths.append(args[0].shape[1])
            return torch_chunk_gated_delta_rule(*args, **kwargs)

        with torch.no_grad():
            for final_state in (False, True):
                options = dict(initial_state=state, output_final_state=final_state,
                               use_qk_l2norm_in_kernel=True)
                expected, expected_state = torch_chunk_gated_delta_rule(q, k, v, g, beta, **options)
                actual, actual_state = checkpointed_delta_rule(tracked, q, k, v, g, beta, **options)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                if final_state:
                    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-5)
                else:
                    self.assertIsNone(actual_state)
        self.assertEqual(lengths, [256, 256, 17] * 2)

    def compare(self, device="cpu", dtype=torch.float32):
        torch.manual_seed(31)
        shape = (1, 529, 2, 8)  # Two full segments and a padded final segment.
        tensors = [torch.randn(shape, device=device, dtype=dtype).requires_grad_() for _ in range(3)]
        tensors += [(-torch.rand(shape[:-1], device=device)).requires_grad_(),
                    torch.rand(shape[:-1], device=device, dtype=dtype).requires_grad_(),
                    (torch.randn(1, 2, 8, 8, device=device) * 0.1).requires_grad_()]
        q, k, v, g, beta, state = tensors
        options = dict(initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        expected, expected_state = torch_chunk_gated_delta_rule(q, k, v, g, beta, **options)
        scale = torch.randn_like(expected)
        state_scale = torch.randn_like(expected_state)
        gradients = torch.autograd.grad((expected * scale).sum() + (expected_state * state_scale).sum(), tensors)
        actual, actual_state = checkpointed_delta_rule(torch_chunk_gated_delta_rule, q, k, v, g, beta, **options)
        actual_gradients = torch.autograd.grad((actual * scale).sum() + (actual_state * state_scale).sum(), tensors)
        tol = dict(rtol=0.025, atol=0.025) if dtype == torch.bfloat16 else dict(rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(actual, expected, **tol)
        torch.testing.assert_close(actual_state, expected_state, **tol)
        for result, reference in zip(actual_gradients, gradients):
            torch.testing.assert_close(result, reference, **tol)

    def test_recurrent_state_and_all_input_gradients(self):
        self.compare()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_bfloat16_cuda(self):
        self.compare("cuda", torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
