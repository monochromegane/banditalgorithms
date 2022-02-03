from banditalgorithms import bandit_types, ucb1


def test_ucb1_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.BanditType:
        return ucb1.UCB1(1)

    _ = new_bandit()
    assert True


def test_ucb1_select_with_count_zero() -> None:
    num_arms = 2
    algo = ucb1.UCB1(num_arms)
    idx_arm = 0

    assert algo.select() == idx_arm
    algo.update(idx_arm, 10.0)
    assert algo.select() == idx_arm + 1


def test_ucb1_select() -> None:
    num_arms = 2
    algo = ucb1.UCB1(num_arms)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 10.0)
        algo.update(idx_arm + 1, 20.0)

    assert algo.select() == idx_arm + 1


def test_ucb1_update() -> None:
    rewards = [10.0, 20.0, 30.0]
    cum_rewards = [10.0, 30.0, 60.0]
    idx_arm = 0

    algo = ucb1.UCB1(1)

    assert algo.counts[idx_arm] == 0.0
    assert algo.rewards[idx_arm] == 0.0

    for i in range(len(rewards)):
        reward = rewards[i]
        cum_reward = cum_rewards[i]

        algo.update(idx_arm, reward)

        assert algo.counts[idx_arm] == i + 1.0
        assert algo.rewards[idx_arm] == cum_reward


def test_ucb1_ucb_scores() -> None:
    num_arms = 3
    algo = ucb1.UCB1(num_arms)
    rewards = [10.0, 20.0, 30.0]
    idx_arm = 0
    for reward in rewards:
        algo.update(idx_arm, reward)

    ucb_scores = algo._ucb_scores()
    assert len(ucb_scores) == num_arms
    assert ucb_scores[idx_arm] > sum(rewards) / len(rewards)


def test_ucb_scores_with_zero_counts() -> None:
    algo = ucb1.UCB1(1)
    ucb_scores = algo._ucb_scores()
    assert ucb_scores[0] == 0.0
