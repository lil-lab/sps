from modeling.evaluation.metric_selection import choose_primary_metric_name


def test_choose_primary_metric_prefers_acc_norm() -> None:
    metrics = {
        "acc,none": 0.31,
        "acc_norm,none": 0.42,
        "acc_norm_stderr,none": 0.01,
    }

    assert choose_primary_metric_name("hellaswag", metrics) == "acc_norm,none"


def test_choose_primary_metric_prefers_lambada_perplexity() -> None:
    metrics = {
        "acc,none": 0.12,
        "word_perplexity,none": 8.5,
    }

    assert choose_primary_metric_name("lambada_openai", metrics) == "word_perplexity,none"


def test_choose_primary_metric_prefers_bits_per_byte_for_rolling_perplexity_tasks() -> None:
    metrics = {
        "word_perplexity,none": 12.0,
        "byte_perplexity,none": 1.2,
        "bits_per_byte,none": 0.27,
    }

    assert choose_primary_metric_name("wikitext", metrics) == "bits_per_byte,none"


def test_choose_primary_metric_uses_word_perplexity_fallback_for_rolling_perplexity_tasks() -> None:
    metrics = {
        "word_perplexity,none": 12.0,
        "byte_perplexity,none": 1.2,
    }

    assert choose_primary_metric_name("paloma_falcon-refinedweb", metrics) == "word_perplexity,none"
