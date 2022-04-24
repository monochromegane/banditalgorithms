from typing import List, Optional, Tuple, cast

import numpy as np


class AdaptiveThompsonSampling:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        sigma_theta: float = 1.0,
        sigma_r: float = 1.0,
        N: int = 10,
        splitting_threshold: int = 20,
        num_bootstrap: int = 10,
        change_detection_confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms
        self.dim_context = dim_context
        self.estimators = [
            AdaptiveThompsonSamplingEstimator(
                dim_context,
                sigma_theta,
                sigma_r,
                N,
                splitting_threshold,
                num_bootstrap,
                change_detection_confidence,
                self.random,
            )
            for _ in range(num_arms)
        ]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        return cast(
            int, np.argmax([estimator.estimate(x) for estimator in self.estimators])
        )

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]
        self.estimators[idx_arm].update(reward, x)


class AdaptiveThompsonSamplingEstimator:
    def __init__(
        self,
        k: int,
        sigma_theta: float,
        sigma_r: float,
        N: int,
        splitting_threshold: int,
        num_bootstrap: int,
        change_detection_confidence: float,
        rs: np.random.RandomState,
    ) -> None:
        self.rs = rs
        self.dim_context = k
        self.sigma_theta = sigma_theta
        self.sigma_r = sigma_r
        self.N = N
        self.splitting_threshold = splitting_threshold
        self.num_bootstrap = num_bootstrap
        self.change_detection_confidence = change_detection_confidence

        self.rewards: List[float] = []
        self.xs: List[np.ndarray] = []
        self.distances: List[float] = []

        self.A = np.eye(self.dim_context) / self.sigma_theta
        self.b = np.zeros([self.dim_context, 1])
        self.recent_A = np.eye(self.dim_context) / self.sigma_theta
        self.recent_b = np.zeros([self.dim_context, 1])
        self.past_A = np.eye(self.dim_context) / self.sigma_theta
        self.past_b = np.zeros([self.dim_context, 1])

        self.cache_params: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def estimate(self, x: np.ndarray) -> float:
        mu_theta_r, SIGMA_theta_r = self._cache_params()

        theta = self.rs.multivariate_normal(mu_theta_r.reshape(-1), SIGMA_theta_r)
        return cast(float, x.reshape(-1).dot(theta))

    def update(self, reward: float, x: np.ndarray) -> None:
        self._update(reward, x)

        N = self.N
        if len(self.rewards) <= self.splitting_threshold or len(self.rewards) < (2 * N):
            return

        t = len(self.xs)
        mu_theta_r_t, SIGMA_theta_r_t = self._params_from(t - N, t)
        mu_theta_r_tN, SIGMA_theta_r_tN = self._params_from(t - N * 2, t - N)

        self.distances.append(
            self._mahalanobis_distance(
                mu_theta_r_t, SIGMA_theta_r_t, mu_theta_r_tN, SIGMA_theta_r_tN
            )
        )

        if self._detect_change(self.distances):
            self.rewards = self.rewards[t - N : t]
            self.xs = self.xs[t - N : t]
            self.distances = []

            self.A = self.recent_A
            self.b = self.recent_b
            self.past_A = np.eye(self.dim_context) / self.sigma_theta
            self.past_b = np.zeros([self.dim_context, 1])

        self.cache_params = None

    def _update(self, reward: float, x: np.ndarray) -> None:
        self.rewards.append(reward)
        self.xs.append(x)

        self.A += x.dot(x.T) / self.sigma_r
        self.b += x * reward / self.sigma_r
        self.recent_A += x.dot(x.T) / self.sigma_r
        self.recent_b += x * reward / self.sigma_r

        if len(self.xs) > self.N:
            discard_reward = self.rewards[-self.N - 1]
            discard_x = self.xs[-self.N - 1]
            self.recent_A -= discard_x.dot(discard_x.T) / self.sigma_r
            self.recent_b -= discard_x * discard_reward / self.sigma_r

            self.past_A += discard_x.dot(discard_x.T) / self.sigma_r
            self.past_b += discard_x * discard_reward / self.sigma_r

        if len(self.xs) > self.N * 2:
            discard_reward = self.rewards[-self.N * 2 - 1]
            discard_x = self.xs[-self.N * 2 - 1]
            self.past_A -= discard_x.dot(discard_x.T) / self.sigma_r
            self.past_b -= discard_x * discard_reward / self.sigma_r

    def _params_from(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        if start == 0 and end == len(self.xs):
            invA = np.linalg.inv(self.A)
            return invA.dot(self.b), invA
        elif end == len(self.xs):
            invA = np.linalg.inv(self.recent_A)
            return invA.dot(self.recent_b), invA
        else:
            invA = np.linalg.inv(self.past_A)
            return invA.dot(self.past_b), invA

    def _mahalanobis_distance(
        self,
        mu_t: np.ndarray,
        SIGMA_t: np.ndarray,
        mu_tN: np.ndarray,
        SIGMA_tN: np.ndarray,
    ) -> float:
        SIGMA = (SIGMA_t + SIGMA_tN) / 2.0
        residual = mu_t - mu_tN
        return cast(
            float, np.sqrt(residual.T.dot(np.linalg.inv(SIGMA)).dot(residual))[0][0]
        )

    def _detect_change(self, data: List[float]) -> bool:
        level = self._change_detection_confidence_level(data)
        return level >= self.change_detection_confidence

    def _change_detection_confidence_level(self, data: List[float]) -> float:
        N = 0
        S_diff = self._magnitude_change(data)

        for _ in range(self.num_bootstrap):
            S_diff_i = self._magnitude_change_bootstrap(data)
            if S_diff_i < S_diff:
                N += 1

        return N / self.num_bootstrap

    def _magnitude_change_bootstrap(self, data: List[float]) -> float:
        samples = self.rs.choice(data, len(data), replace=False)
        return self._magnitude_change(samples.tolist())

    def _magnitude_change(self, data: List[float]) -> float:
        average = sum(data) / len(data)
        diff = [d - average for d in data]
        S = np.cumsum(diff)
        S_diff = S.max() - S.min()
        return cast(float, S_diff)

    def _cache_params(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_params is None:
            mu_theta_r, SIGMA_theta_r = self._params_from(0, len(self.xs))
            self.cache_params = (mu_theta_r, SIGMA_theta_r)
        return self.cache_params
