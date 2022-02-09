import math

import numpy as np
from banditalgorithms import bandit_types, linucb


def test_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return linucb.LinUCB(1, 1)

    _ = new_bandit()
    assert True


def test_linucb_select() -> None:
    num_arms = 2
    dim_context = 1
    ctx = [1.0]
    algo = linucb.LinUCB(num_arms, dim_context)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 10.0, ctx)
        algo.update(idx_arm + 1, 20.0, ctx)

    assert algo.select(ctx) == idx_arm + 1


def test_linucb_update() -> None:
    rewards = [10.0, 20.0, 30.0]
    cum_rewards = [10.0, 30.0, 60.0]
    ctx = [1.0]
    idx_arm = 0
    dim_context = 1

    algo = linucb.LinUCB(1, dim_context)

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


def test_linucb_ucb_scores() -> None:
    num_arms = 3
    dim_context = 1
    algo = linucb.LinUCB(num_arms, dim_context)
    rewards = [10.0, 20.0, 30.0]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    mat = np.eye(1)
    for reward in rewards:
        algo.update(idx_arm, reward, ctx)
        mat += x.dot(x.T)

    count = len(rewards) + 1  # +1 is due to np.eye when its initialize
    actual_reward = sum(rewards) / count
    actual_term_ucb = math.sqrt(1.0 / count)

    ucb_scores = algo._ucb_scores(x)
    assert len(ucb_scores) == num_arms
    assert ucb_scores[idx_arm] == actual_reward + actual_term_ucb
