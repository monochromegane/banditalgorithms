from banditalgorithms import bandit_types, dynamic_linucb


def test_dynamic_linucb_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return dynamic_linucb.DynamicLinUCB(1, 1)

    _ = new_bandit()
    assert True
