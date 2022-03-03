from banditalgorithms import bandit_types, time_varying_thompson_sampling


def test_time_varying_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return time_varying_thompson_sampling.TimeVaryingThompsonSampling(1, 1)

    _ = new_bandit()
    assert True
