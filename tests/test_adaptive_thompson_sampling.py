import math

import numpy as np
from banditalgorithms import adaptive_thompson_sampling, bandit_types


def test_adaptive_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)

    _ = new_bandit()
    assert True


def test_estimator_mahalanobis_distance() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)
    estimator = algo.estimators[0]

    SIGMA_0 = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    SIGMA_1 = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])

    mu_0 = np.c_[np.array([1.0, 0.0, 0.0])]
    mu_1 = np.c_[np.array([0.0, 1.0, 0.0])]

    distance = estimator._mahalanobis_distance(mu_0, SIGMA_0, mu_1, SIGMA_1)
    assert distance == 1.0

    mu_0 = np.c_[np.array([0.0, 2.0, 0.0])]
    mu_1 = np.c_[np.array([0.0, 1.0, 0.0])]

    distance = estimator._mahalanobis_distance(mu_0, SIGMA_0, mu_1, SIGMA_1)
    assert distance == 1.0

    mu_0 = np.c_[np.array([2.0, 0.0, 0.0])]
    mu_1 = np.c_[np.array([0.0, 1.0, 0.0])]

    distance = estimator._mahalanobis_distance(mu_0, SIGMA_0, mu_1, SIGMA_1)
    assert np.allclose(distance, math.sqrt(3.0))
