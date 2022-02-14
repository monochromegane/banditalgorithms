import numpy as np
from banditalgorithms import bandit_types, linear_thompson_sampling


def test_linear_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return linear_thompson_sampling.LinearThompsonSampling(1, 1)

    _ = new_bandit()
    assert True


def test_linear_thompson_sampling_select() -> None:
    num_arms = 2
    dim_context = 1
    ctx = [1.0]
    algo = linear_thompson_sampling.LinearThompsonSampling(
        num_arms, dim_context, seed=1
    )
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 0.0, ctx)
        algo.update(idx_arm + 1, 1.0, ctx)

    assert algo.select(ctx) == idx_arm + 1


def test_linear_thompson_sampling_update() -> None:
    rewards = [1.0, 1.0, 1.0]
    cum_rewards = [1.0, 2.0, 3.0]
    ctx = [1.0]
    dim_context = 1
    idx_arm = 0

    algo = linear_thompson_sampling.LinearThompsonSampling(1, dim_context)

    assert np.allclose(algo.invAs[idx_arm].data, np.eye(dim_context))
    assert np.allclose(algo.bs[idx_arm], np.c_[np.zeros(dim_context)])

    for i in range(len(rewards)):
        reward = rewards[i]
        cum_reward = cum_rewards[i]

        algo.update(idx_arm, reward, ctx)

        assert np.allclose(
            algo.invAs[idx_arm].data, np.linalg.inv(np.eye(dim_context) + i + 1.0)
        )
        assert np.allclose(algo.bs[idx_arm], np.c_[np.zeros(dim_context) + cum_reward])


def test_linear_thompson_sampling_samples() -> None:
    num_arms = 3
    dim_context = 1

    algo = linear_thompson_sampling.LinearThompsonSampling(num_arms, dim_context)
    rewards = [1.0, 1.0, 1.0]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    mat = np.eye(1)
    for reward in rewards:
        algo.update(idx_arm, reward, ctx)
        mat += x.dot(x.T)

    count = len(rewards) + 1  # +1 is due to np.eye when its initialize
    actual_reward = sum(rewards) / count

    algo.sigma2 = 0.0  # due to fix multivariate_normal's mu
    samples = algo._samples(x)
    assert len(samples) == num_arms
    assert samples[idx_arm] == actual_reward
