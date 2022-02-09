from banditalgorithms import bandit_types, linear_thompson_sampling


def test_linear_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return linear_thompson_sampling.LinearThompsonSampling(1, 1)

    _ = new_bandit()
    assert True
