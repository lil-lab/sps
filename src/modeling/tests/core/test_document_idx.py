import torch
from modeling.models.full_attention_model import Model, ModelConfig


def create_test_model():
    """Create a minimal model instance for testing."""
    config = ModelConfig(
        block_size=128,
        vocab_size=50257,
        n_layer=2,
        n_head=2,
        hidden_size=64,
        dropout=0.0,
        bias=False,
        pad_token_id=50255,
    )
    model = Model(config)
    model.eval()
    return model


def test_no_eos():
    """Test with no EOS tokens - all tokens in document 0."""
    model = create_test_model()
    idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_single_eos():
    """Test with one EOS token splitting two documents."""
    model = create_test_model()
    EOS_TOKEN_INDEX = model.config.eos_token_id
    idx = torch.tensor([[1, 2, 3, EOS_TOKEN_INDEX, 4, 5, 6, 7, 8, 9]])

    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_multiple_eos():
    """Test with multiple EOS tokens creating multiple documents."""
    model = create_test_model()
    EOS_TOKEN_INDEX = model.config.eos_token_id
    idx = torch.tensor([[1, 2, EOS_TOKEN_INDEX, 3, 4, EOS_TOKEN_INDEX, 5, 6, 7, 8]])

    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]])

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_eos_at_boundaries():
    """Test EOS at start and end positions."""
    model = create_test_model()
    EOS_TOKEN_INDEX = model.config.eos_token_id

    idx = torch.tensor([[EOS_TOKEN_INDEX, 1, 2, 3, 4]])
    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 1, 1, 1, 1]])
    assert torch.equal(result, expected), f"EOS at start: Expected {expected}, got {result}"

    idx = torch.tensor([[1, 2, 3, 4, EOS_TOKEN_INDEX]])
    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 0, 0, 0, 0]])
    assert torch.equal(result, expected), f"EOS at end: Expected {expected}, got {result}"


def test_consecutive_eos():
    """Test consecutive EOS tokens."""
    model = create_test_model()
    EOS_TOKEN_INDEX = model.config.eos_token_id
    idx = torch.tensor([[1, EOS_TOKEN_INDEX, EOS_TOKEN_INDEX, EOS_TOKEN_INDEX, 2, 3]])

    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 0, 1, 2, 3, 3]])

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_batch_processing():
    """Test that batching works correctly (each batch element independent)."""
    model = create_test_model()
    EOS_TOKEN_INDEX = model.config.eos_token_id
    idx = torch.tensor([
        [1, 2, EOS_TOKEN_INDEX, 3, 4, 5, 6, 7, 8, 9],
        [1, EOS_TOKEN_INDEX, 2, 3, EOS_TOKEN_INDEX, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ])

    result = model.generate_document_idx(idx)
    expected = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_left_padding_creates_fake_prefix_documents():
    """Left-pad tokens should not share a document with the real suffix."""
    model = create_test_model()
    EOS_TOKEN_INDEX = model.config.eos_token_id
    PAD_TOKEN_INDEX = model.config.pad_token_id
    idx = torch.tensor([[PAD_TOKEN_INDEX, PAD_TOKEN_INDEX, 1, 2, EOS_TOKEN_INDEX, 3, 4]])

    result = model.generate_document_idx(idx)
    expected = torch.tensor([[0, 1, 2, 2, 2, 3, 3]])

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
