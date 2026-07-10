import torch

from modeling.models.utils.masked_stats import add_distribution_stats, add_empty_distribution_stats


def test_add_distribution_stats_matches_torch_reference():
    values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)

    stats = {}
    add_distribution_stats(stats, "sample", values)

    expected = {
        "sample_mean": values.mean(),
        "sample_var": values.var(unbiased=False),
        "sample_min": values.min(),
        "sample_max": values.max(),
        "sample_p25": values.quantile(0.25),
        "sample_p50": values.quantile(0.50),
        "sample_p75": values.quantile(0.75),
        "sample_p90": values.quantile(0.90),
        "sample_p99": values.quantile(0.99),
    }

    assert set(stats) == set(expected)
    for key, expected_value in expected.items():
        torch.testing.assert_close(stats[key], expected_value)


def test_add_distribution_stats_survives_torch_compile_with_dynamic_masked_lengths():
    def compute_stats(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stats = {}
        filtered = values[mask]
        add_distribution_stats(stats, "sample", filtered)
        return (
            stats["sample_mean"],
            stats["sample_var"],
            stats["sample_min"],
            stats["sample_max"],
            stats["sample_p25"],
            stats["sample_p50"],
            stats["sample_p75"],
            stats["sample_p90"],
            stats["sample_p99"],
        )

    compiled = torch.compile(compute_stats)
    values = torch.arange(1, 17, dtype=torch.float32)

    mask_a = torch.tensor(
        [True, False, True, False, True, False, True, False, True, False, True, False, True, False, False, False],
        dtype=torch.bool,
    )
    mask_b = torch.tensor(
        [False, True, True, True, False, False, True, True, False, True, False, True, False, True, True, True],
        dtype=torch.bool,
    )

    eager_a = compute_stats(values, mask_a)
    eager_b = compute_stats(values, mask_b)
    compiled_a = compiled(values, mask_a)
    compiled_b = compiled(values, mask_b)

    for eager, compiled_out in ((eager_a, compiled_a), (eager_b, compiled_b)):
        for eager_value, compiled_value in zip(eager, compiled_out):
            torch.testing.assert_close(compiled_value, eager_value)


def test_add_empty_distribution_stats_matches_distribution_key_schema():
    values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)

    populated = {}
    add_distribution_stats(populated, "sample", values)

    empty = {}
    add_empty_distribution_stats(empty, "sample")

    assert set(empty) == set(populated)
    for key, value in empty.items():
        assert torch.isnan(value), f"Expected NaN for empty stat {key}, got {value!r}"
