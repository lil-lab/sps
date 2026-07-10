import torch
import pytest

from modeling.tests._helpers import (
    make_full_model,
    forward_logits,
)


@pytest.mark.parametrize(("make_model", "atol", "rtol"), [(make_full_model, 1e-5, 1e-5)])
def test_left_padding_is_invariant_on_real_suffix_logits(make_model, atol: float, rtol: float):
    torch.manual_seed(7)
    model = make_model()
    pad = model.config.pad_token_id
    eos = model.config.eos_token_id

    unpadded = torch.tensor([[3, eos, 4, 5]], dtype=torch.long)
    padded = torch.tensor([[pad, pad, 3, eos, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        logits_unpadded = forward_logits(model, unpadded)
        logits_padded = forward_logits(model, padded)

    torch.testing.assert_close(logits_unpadded, logits_padded[:, -unpadded.size(1) :, :], atol=atol, rtol=rtol)
