from banditalgorithms import bandit_types, decay_linucb


def test_decay_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return decay_linucb.DecayLinUCB(1, 1)

    _ = new_bandit()
    assert True
