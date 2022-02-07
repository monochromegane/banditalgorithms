from banditalgorithms import bandit_types, linucb


def test_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return linucb.LinUCB(1, 1)

    _ = new_bandit()
    assert True
