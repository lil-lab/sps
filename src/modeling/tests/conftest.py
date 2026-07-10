"""Shared pytest configuration: custom markers and auto-skip hooks."""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (stability/mismatch diagnostics).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA GPU")
    config.addinivalue_line("markers", "triton: requires Triton (implies CUDA)")
    config.addinivalue_line("markers", "slow: long-running stability/mismatch diagnostics")


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    run_slow = config.getoption("--run-slow")

    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_triton = pytest.mark.skip(reason="Triton not available (requires CUDA)")
    skip_slow = pytest.mark.skip(reason="Slow test; pass --run-slow to enable")

    for item in items:
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if "triton" in item.keywords and not has_cuda:
            item.add_marker(skip_triton)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


@pytest.fixture
def device():
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
