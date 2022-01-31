from banditalgorithms import epsilon_greedy


def test_epsilon_greedy() -> None:
    algo = epsilon_greedy.EpsilonGreedy(1)
    selected = algo.select()
    algo.update(selected, 1.0)
    assert True
