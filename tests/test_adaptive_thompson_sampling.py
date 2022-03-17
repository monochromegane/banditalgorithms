import math
from unittest.mock import patch

import numpy as np
from banditalgorithms import adaptive_thompson_sampling, bandit_types

def test_adaptive_thompson_sampling_compatible_with_bandit_type() -> None:
    def new_bandit() -> bandit_types.ContextualBanditType:
        return adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)

    _ = new_bandit()
    assert True


def test_adaptive_thompson_sampling_select() -> None:
    num_arms = 2
    dim_context = 1
    ctx = [1.0]
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(num_arms, dim_context)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 0.0, ctx)
        algo.update(idx_arm + 1, 1.0, ctx)

    assert algo.select(ctx) == idx_arm + 1


def test_adaptive_thompson_sampling_update() -> None:
    num_arms = 2
    dim_context = 1
    ctx = [1.0]
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(num_arms, dim_context)
    idx_arm = 0

    for _ in range(100):
        algo.update(idx_arm, 0.0, ctx)

    assert len(algo.estimators[idx_arm].rewards) == 100
    assert len(algo.estimators[idx_arm + 1].rewards) == 0


def test_estimator_update_when_observations_is_fewer_than_threshold() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(
        1, 1, N=1, splitting_threshold=5
    )
    estimator = algo.estimators[0]
    with patch.object(estimator, "_mahalanobis_distance") as mock_estimator:
        for _ in range(5):
            ctx = np.ones([1, 1])
            reward = 1.0
            estimator.update(reward, ctx)

        mock_estimator.assert_not_called()


def test_estimator_update_when_observations_is_fewer_than_2N() -> None:
    N = 5
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(
        1, 1, N=N, splitting_threshold=1
    )
    estimator = algo.estimators[0]
    with patch.object(estimator, "_mahalanobis_distance") as mock_estimator:
        for _ in range(2 * N - 1):
            ctx = np.ones([1, 1])
            reward = 1.0
            estimator.update(reward, ctx)

        mock_estimator.assert_not_called()


def test_estimator_update() -> None:
    N = 5
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(
        1, 1, N=N, splitting_threshold=1
    )
    estimator = algo.estimators[0]
    for _ in range(2 * N):
        ctx = np.ones([1, 1])
        reward = 1.0
        estimator.update(reward, ctx)

    assert len(estimator.distances) == 1


def test_estimator_update_with_discarding() -> None:
    N = 5
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(
        1, 1, N=N, splitting_threshold=1, seed=1
    )
    estimator = algo.estimators[0]

    for _ in range(10):
        ctx0 = np.ones([1, 1])
        reward0 = 1.0
        estimator.update(reward0, ctx0)

    for _ in range(15):
        ctx1 = np.c_[np.array([0.1])]
        reward1 = 2.0
        estimator.update(reward1, ctx1)
        # Detect change at i == 6, so 8 of observations and 4 of distances increase after detection.

    assert len(estimator.rewards) == N + 8
    assert np.all(np.array(estimator.rewards) == reward1)
    assert len(estimator.xs) == N + 8
    assert np.all(np.array(estimator.xs) == ctx1)
    assert len(estimator.distances) == 4


def test_estimator_params_from_with_zero() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)
    estimator = algo.estimators[0]

    mu_theta, SIGMA_theta = estimator._params_from(0, 0)

    assert np.allclose(mu_theta, estimator.mu_theta)
    assert np.allclose(SIGMA_theta, estimator.SIGMA_theta)


def test_estimator_params_from() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1, N=5)
    estimator = algo.estimators[0]

    A0 = np.eye(1)
    b0 = np.zeros([1, 1])
    for _ in range(5):
        ctx = np.ones([1, 1])
        reward = 1.0
        estimator.update(reward, ctx)
        A0 += ctx.dot(ctx.T)
        b0 += ctx * reward

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
        b1 += ctx * reward

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


TS = [
    10.7,
    13.0,
    11.4,
    11.5,
    12.5,
    14.1,
    14.8,
    14.1,
    12.6,
    16.0,
    11.7,
    10.6,
    10.0,
    11.4,
    7.9,
    9.5,
    8.0,
    11.8,
    10.5,
    11.2,
    9.2,
    10.1,
    10.4,
    10.5,
]


def test_estimator_magnitude_change() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1)
    estimator = algo.estimators[0]

    m = estimator._magnitude_change(TS)
    assert np.allclose(m, 17.74167)


def test_estimator_change_detection_confidence_level_zero() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1, num_bootstrap=10)
    estimator = algo.estimators[0]

    S_diff = estimator._magnitude_change(TS)

    with patch.object(estimator, "_magnitude_change_bootstrap") as mock_estimator:
        mock_estimator.return_value = S_diff

        assert estimator._change_detection_confidence_level(TS) == 0.0


def test_estimator_change_detection_confidence_level_one() -> None:
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(1, 1, num_bootstrap=10)
    estimator = algo.estimators[0]

    S_diff = estimator._magnitude_change(TS)

    with patch.object(estimator, "_magnitude_change_bootstrap") as mock_estimator:
        mock_estimator.return_value = S_diff - 1e-1

        assert estimator._change_detection_confidence_level(TS) == 1.0


def test_estimator_detect_change_true() -> None:
    level_threshold = 0.95
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(
        1, 1, change_detection_confidence=level_threshold
    )
    estimator = algo.estimators[0]

    with patch.object(
        estimator, "_change_detection_confidence_level"
    ) as mock_estimator:
        mock_estimator.return_value = level_threshold

        assert estimator._detect_change(TS)


def test_estimator_detect_change_false() -> None:
    level_threshold = 0.95
    algo = adaptive_thompson_sampling.AdaptiveThompsonSampling(
        1, 1, change_detection_confidence=level_threshold
    )
    estimator = algo.estimators[0]

    with patch.object(
        estimator, "_change_detection_confidence_level"
    ) as mock_estimator:
        mock_estimator.return_value = level_threshold - 1e-1

        assert not estimator._detect_change(TS)
