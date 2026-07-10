import torch
from modeling.masking import causal_mask


class TestCausalMask:
    """Tests for the basic causal mask function."""

    def test_causal_mask_basic(self):
        """Test basic causal masking behavior."""
        b = 0
        h = 0
        q_idx = torch.tensor([5])
        kv_idx = torch.arange(10)

        result = causal_mask(b, h, q_idx, kv_idx)

        # Position 5 can attend to positions 0-5 (causal)
        expected = torch.tensor([True, True, True, True, True, True, False, False, False, False])
        assert torch.equal(result, expected), f"Result: {result}, Expected: {expected}"

    def test_causal_mask_full_sequence(self):
        """Test causal mask for a full sequence."""
        b = 0
        h = 0
        q_idx = torch.arange(10).unsqueeze(1)  # [10, 1]
        kv_idx = torch.arange(10).unsqueeze(0)  # [1, 10]

        result = causal_mask(b, h, q_idx, kv_idx)

        # Lower triangular matrix (including diagonal)
        expected = torch.tensor([
            [True, False, False, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False, False, False],
            [True, True, True, True, False, False, False, False, False, False],
            [True, True, True, True, True, False, False, False, False, False],
            [True, True, True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, True, True, False],
            [True, True, True, True, True, True, True, True, True, True],
        ])
        assert torch.equal(result, expected), f"Result: {result}, Expected: {expected}"
