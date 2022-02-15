from banditalgorithms import adaptive_thompson_sampling, bandit_types


def test_adaptive_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)

    _ = new_bandit()
    assert True
