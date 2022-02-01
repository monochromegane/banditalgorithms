from banditalgorithms import epsilon_greedy


def test_epsilon_greedy_update() -> None:
    rewards = [10.0, 20.0, 30.0]
    cum_rewards = [10.0, 30.0, 60.0]
    idx_arm = 0

    algo = epsilon_greedy.EpsilonGreedy(1)

    assert algo.counts[idx_arm] == 0.0
    assert algo.rewards[idx_arm] == 0.0

    for i in range(len(rewards)):
        reward = rewards[i]
        cum_reward = cum_rewards[i]

        algo.update(idx_arm, reward)

        assert algo.counts[idx_arm] == i + 1.0
        assert algo.rewards[idx_arm] == cum_reward


def test_epsilon_greedy_theta_hats() -> None:
    num_arms = 3
    algo = epsilon_greedy.EpsilonGreedy(num_arms)
    rewards = [10.0, 20.0, 30.0]
    idx_arm = 0
    for reward in rewards:
        algo.update(idx_arm, reward)

    theta_hats = algo._theta_hats()
    assert len(theta_hats) == num_arms
    assert theta_hats[idx_arm] == sum(rewards) / len(rewards)


def test_epsilon_greedy_theta_hats_with_zero_counts() -> None:
    algo = epsilon_greedy.EpsilonGreedy(1)
    theta_hats = algo._theta_hats()
    assert theta_hats[0] == 0.0
