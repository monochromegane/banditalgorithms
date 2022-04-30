import math

import numpy as np
from banditalgorithms import bandit_types, decay_linucb


def test_decay_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return decay_linucb.DecayLinUCB(1, 1)

    _ = new_bandit()
    assert True


def test_decay_linucb_select() -> None:
    num_arms = 2
    dim_context = 1
    gamma = 0.99
    ctx = [1.0]
    algo = decay_linucb.DecayLinUCB(num_arms, dim_context, gamma=gamma)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 10.0, ctx)
        algo.update(idx_arm + 1, 20.0, ctx)

    assert algo.select(ctx) == idx_arm + 1


def test_decay_linucb_ucb_scores() -> None:
    num_arms = 3
    dim_context = 1
    gamma = 0.99
    algo = decay_linucb.DecayLinUCB(num_arms, dim_context, gamma=gamma)
    rewards = [10.0, 20.0, 30.0]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    exp_sum_reward = 0.0
    exp_sum_count = 0.0
    for reward in rewards:
        exp_sum_reward = (exp_sum_reward * gamma) + reward
        exp_sum_count = (exp_sum_count * gamma) + 1.0
    exp_sum_count += 1.0  # +1 is due to add np.eye before its inverse

    for reward in rewards:
        algo.update(idx_arm, reward, ctx)

    actual_reward = exp_sum_reward / exp_sum_count
    actual_term_ucb = math.sqrt(1.0 / exp_sum_count)

    ucb_scores = algo._ucb_scores(x)
    assert len(ucb_scores) == num_arms
    assert ucb_scores[idx_arm] == actual_reward + actual_term_ucb
