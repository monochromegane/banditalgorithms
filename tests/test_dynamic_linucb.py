from unittest.mock import patch

import numpy as np
from banditalgorithms import bandit_types, dynamic_linucb


def test_dynamic_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return dynamic_linucb.DynamicLinUCB(1, 1)

    _ = new_bandit()
    assert True


def test_dynamic_linucb_select() -> None:
    num_arms = 2
    dim_context = 1
    ctx = [1.0]
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context, sigma2=1.0, tau=100)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 0.1, ctx)
        algo.update(idx_arm + 1, 1.0, ctx)

    assert algo.select(ctx) == idx_arm + 1


def test_dynamic_linucb_update_keep_model() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)

    rewards = [10.0 for _ in range(5)]
    idx_arm = 0
    ctx = [1.0]
    with patch.object(algo, "_keep_model") as mock_keep:
        mock_keep.return_value = True
        with patch.object(algo, "_discard_model") as mock_discard:
            mock_discard.return_value = True
            with patch.object(algo, "_create_new_slave_model") as mock_creator:
                for reward in rewards:
                    algo.update(idx_arm, reward, ctx)
                mock_creator.assert_not_called()

    assert len(algo.models) == 1


def test_dynamic_linucb_update_discard_model() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)

    rewards = [10.0 for _ in range(5)]
    idx_arm = 0
    ctx = [1.0]

    with patch.object(algo, "_keep_model") as mock_keep:
        mock_keep.return_value = False
        with patch.object(algo, "_discard_model") as mock_discard:
            mock_discard.return_value = True
            with patch.object(algo, "_create_new_slave_model") as mock_creator:
                for reward in rewards:
                    algo.update(idx_arm, reward, ctx)
                mock_creator.assert_called()

    assert len(algo.models) == 1


def test_dynamic_linucb_update_create_model() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)

    rewards = [10.0 for _ in range(5)]
    idx_arm = 0
    ctx = [1.0]

    with patch.object(algo, "_keep_model") as mock_keep:
        mock_keep.return_value = False
        with patch.object(algo, "_discard_model") as mock_discard:
            mock_discard.return_value = False
            with patch.object(algo, "_create_new_slave_model") as mock_creator:
                for reward in rewards:
                    algo.update(idx_arm, reward, ctx)
                mock_creator.assert_called()

    assert len(algo.models) == 1 + len(rewards)


def test_dynamic_linucb_minimum_error_model() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)

    rewards = [10.0 for _ in range(3)]
    idx_arm = 0
    ctx = [1.0]

    with patch.object(algo, "_keep_model") as mock_keep:
        mock_keep.return_value = False
        with patch.object(algo, "_discard_model") as mock_discard:
            mock_discard.return_value = False
            for reward in rewards:
                algo.update(idx_arm, reward, ctx)

    algo.models[0].e_hat = 1.0
    algo.models[1].e_hat = 0.5
    algo.models[2].e_hat = 0.1

    assert algo._minimum_error_model() == 2


def test_dynamic_linucb_slave_ucb_scores() -> None:
    num_arms = 3
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)
    slave = algo.models[0]
    rewards = [10.0, 20.0, 30.0]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    mat = np.eye(1)
    with patch.object(slave, "_exceed_confidence_bound") as mock_exceed:
        mock_exceed.return_value = False
        for reward in rewards:
            algo.update(idx_arm, reward, ctx)
            mat += x.dot(x.T)

    count = len(rewards) + 1  # +1 is due to np.eye when its initialize
    actual_reward = sum(rewards) / count

    with patch.object(slave, "B") as mock_slave:
        mock_slave.return_value = 0.0

        ucb_scores = slave._ucb_scores(x)
        assert len(ucb_scores) == num_arms
        assert ucb_scores[idx_arm] == actual_reward  # + actual_term_ucb


def test_dynamic_linucb_slave_update_when_confidence_bound_exceed() -> None:
    num_arms = 3
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context)
    slave = algo.models[0]
    rewards = [10.0, 20.0, 30.0]
    idx_arm = 0
    ctx = [1.0]

    with patch.object(slave, "_exceed_confidence_bound") as mock_exceed:
        mock_exceed.return_value = True
        for reward in rewards:
            algo.update(idx_arm, reward, ctx)

    assert np.allclose(slave.bs[idx_arm], np.zeros([dim_context, 1]))
    assert np.allclose(slave.invAs[idx_arm].data, np.linalg.inv(np.eye(dim_context)))
    assert slave.counts[idx_arm] == 0


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
    slave = algo.models[0]
    rewards = [10.0 for _ in range(9)]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    with patch.object(slave, "_exceed_confidence_bound") as mock_exceed:
        mock_exceed.return_value = False
        for reward in rewards:
            algo.update(idx_arm, reward, ctx)

    expected_B = (
        0.5746812313093125  # alpha(= 1.8173015965970112) * ||x||_A^{-1}(= 0.31622777)
    )
    assert np.isclose(slave.B(idx_arm, x), expected_B)


def test_dynamic_linucb_slave_badness() -> None:
    num_arms = 1
    dim_context = 1
    algo = dynamic_linucb.DynamicLinUCB(num_arms, dim_context, delta2=0.9, tau=5)
    slave = algo.models[0]

    rewards = [10.0 for _ in range(5)]
    idx_arm = 0
    ctx = [1.0]
    x = np.c_[np.array(ctx)]

    with patch.object(slave, "_exceed_confidence_bound") as mock_slave:
        mock_slave.return_value = 1.0

        for reward in rewards:
            slave.update(idx_arm, reward, x)

        expected_d = 0.10264527054756413
        assert len(slave._recently_es()) == len(rewards)
        assert slave.e_hat == 1.0
        assert np.allclose(slave.d, expected_d)

        mock_slave.return_value = 0.0
        slave.update(idx_arm, reward, x)

        assert len(slave._recently_es()) == len(rewards)
        assert slave.e_hat == 0.8
        assert np.allclose(slave.d, expected_d)
