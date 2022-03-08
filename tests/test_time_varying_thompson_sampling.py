import numpy as np
from banditalgorithms import bandit_types, time_varying_thompson_sampling


def test_time_varying_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return time_varying_thompson_sampling.TimeVaryingThompsonSampling(1, 1)

    _ = new_bandit()
    assert True


def test_time_varying_thompson_sampling_particles_resampling() -> None:
    weights = [0.1, 0.1, 0.1, 0.7]  # The sum should be 1

    algo = time_varying_thompson_sampling.TimeVaryingThompsonSampling(1, 1)
    particles = algo.filters[0]

    # if u0 = 0.01 then
    #   u = [0.01, 0.26, 0.51, 0.76]
    #   w_cumsum = [0.1, 0.2, 0.3, 1.0]
    # so, f_invs = [0, 2, 3, 3]
    assert particles._resampling(weights, u0=0.01) == [0, 2, 3, 3]

    # if u0 = 0.2 then
    #   u = [0.2, 0.45, 0.7, 0.95]
    #   w_cumsum = [0.1, 0.2, 0.3, 1.0]
    # so, f_invs = [1, 3, 3, 3]
    assert particles._resampling(weights, u0=0.2) == [1, 3, 3, 3]


def test_time_varying_thompson_sampling_particles_f_inv() -> None:
    weights = [0.1, 0.1, 0.8]  # The sum should be 1
    w_cumsum = np.cumsum(weights)  # 0.1, 0.2, 1.0
    idx = np.array(list(range(3)))

    algo = time_varying_thompson_sampling.TimeVaryingThompsonSampling(1, 1)
    particles = algo.filters[0]

    assert particles._f_inv(w_cumsum, idx, 0.0) == 0
    assert particles._f_inv(w_cumsum, idx, 0.1 + 1e-10) == 1
    assert particles._f_inv(w_cumsum, idx, 0.2 + 1e-10) == 2
    assert particles._f_inv(w_cumsum, idx, 1.0 - 1e-10) == 2
