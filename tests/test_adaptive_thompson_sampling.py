import math

import numpy as np
from banditalgorithms import adaptive_thompson_sampling, bandit_types


def test_adaptive_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)

    _ = new_bandit()
    assert True


def test_estimator_params_from_with_zero() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)
    estimator = algo.estimators[0]

    mu_theta, SIGMA_theta = estimator._params_from(0, 0)

    assert np.allclose(mu_theta, estimator.mu_theta)
    assert np.allclose(SIGMA_theta, estimator.SIGMA_theta)


def test_estimator_params_from() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)
    estimator = algo.estimators[0]

    A0 = np.eye(1)
    b0 = np.zeros([1, 1])
    for _ in range(5):
        ctx = np.ones([1, 1])
        reward = 1.0
        estimator.update(reward, ctx)
        A0 += ctx.dot(ctx.T)
        b0 += reward

    mu_theta, SIGMA_theta = estimator._params_from(0, 5)
    assert np.allclose(np.linalg.inv(A0).dot(b0), mu_theta)
    assert np.allclose(np.linalg.inv(A0), SIGMA_theta)

    A1 = np.eye(1)
    b1 = np.zeros([1, 1])
    for _ in range(5):
        ctx = np.ones([1, 1])
        reward = 2.0
        estimator.update(reward, ctx)
        A1 += ctx.dot(ctx.T)
        b1 += reward

    mu_theta, SIGMA_theta = estimator._params_from(5, 10)
    assert np.allclose(np.linalg.inv(A1).dot(b1), mu_theta)
    assert np.allclose(np.linalg.inv(A1), SIGMA_theta)


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
