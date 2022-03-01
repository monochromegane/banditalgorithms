from unittest.mock import patch

import numpy as np
from banditalgorithms import bandit_types, dynamic_linucb


def test_dynamic_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return dynamic_linucb.DynamicLinUCB(1, 1)

    _ = new_bandit()
    assert True


def test_dynamic_linucb_slave_ucb_scores() -> None:
    num_arms = 3
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)
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
    slave = algo.models[0]

    with patch.object(slave, "B") as mock_slave:
        mock_slave.return_value = 0.0

        ucb_scores = slave._ucb_scores(x)
        assert len(ucb_scores) == num_arms
        assert ucb_scores[idx_arm] == actual_reward  # + actual_term_ucb


def test_dynamic_linucb_slave_B_with_zero_observation() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context, sigma2=1.0, delta1=1.0)
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]
    slave = algo.models[0]

    expected_B = 1.0  # alpha(= 1.0) * ||x||_A^{-1}(= 1.0)
    assert slave.B(idx_arm, x) == expected_B


def test_dynamic_linucb_slave_B() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context, sigma2=1.0, delta1=1.0)
    rewards = [10.0 for _ in range(9)]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    for reward in rewards:
        algo.update(idx_arm, reward, ctx)
    slave = algo.models[0]

    expected_B = (
        0.5746812313093125  # alpha(= 1.8173015965970112) * ||x||_A^{-1}(= 0.31622777)
    )
    np.allclose(slave.B(idx_arm, x), expected_B)
