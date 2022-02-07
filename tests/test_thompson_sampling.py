from banditalgorithms import bandit_types, thompson_sampling


def test_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.BanditType:
        return thompson_sampling.ThompsonSampling(1)

    _ = new_bandit()
    assert True


def test_thompson_sampling_select() -> None:
    num_arms = 2
    algo = thompson_sampling.ThompsonSampling(num_arms, seed=1)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 0.0)
        algo.update(idx_arm + 1, 1.0)

    assert algo.select() == idx_arm + 1


def test_thompson_sampling_update() -> None:
    rewards = [1.0, 1.0, 1.0]
    cum_rewards = [1.0, 2.0, 3.0]
    idx_arm = 0

    algo = thompson_sampling.ThompsonSampling(1)

    assert algo.counts[idx_arm] == 0.0
    assert algo.rewards[idx_arm] == 0.0

    for i in range(len(rewards)):
        reward = rewards[i]
        cum_reward = cum_rewards[i]

        algo.update(idx_arm, reward)

        assert algo.counts[idx_arm] == i + 1.0
        assert algo.rewards[idx_arm] == cum_reward


def test_thompson_sampling_samples() -> None:
    num_arms = 3
    algo = thompson_sampling.ThompsonSampling(num_arms)
    rewards = [1.0, 1.0, 1.0]
    idx_arm = 0
    for reward in rewards:
        algo.update(idx_arm, reward)

    samples = algo._samples()
    assert len(samples) == num_arms
    assert samples[idx_arm] >= 0.0
    assert samples[idx_arm] <= 1.0


def test_thompson_sampling_samples_with_zero_counts() -> None:
    algo = thompson_sampling.ThompsonSampling(1)
    samples = algo._samples()
    assert samples[0] >= 0.0
    assert samples[0] <= 1.0
