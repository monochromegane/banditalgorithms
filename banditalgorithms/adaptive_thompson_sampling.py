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
        self.k = k
        self.sigma_theta = sigma_theta
        self.sigma_r = sigma_r
        self.SIGMA_theta = np.eye(k) * sigma_theta
        self.mu_theta = np.zeros([k, 1])
        self.N = N
        self.splitting_threshold = splitting_threshold
        self.num_bootstrap = num_bootstrap
        self.change_detection_confidence = change_detection_confidence

        self.rewards: List[float] = []
        self.xs: List[np.ndarray] = []
        self.distances: List[float] = []

    def estimate(self, x: np.ndarray) -> float:
        mu_theta_r, SIGMA_theta_r = self._params_from(0, len(self.xs))

        theta = self.rs.multivariate_normal(mu_theta_r.reshape(-1), SIGMA_theta_r)
        return cast(float, x.reshape(-1).dot(theta))

    def update(self, reward: float, x: np.ndarray) -> None:
        self.rewards.append(reward)
        self.xs.append(x)

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

    def _params_from(self, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.xs) == 0:
            return self.mu_theta, self.SIGMA_theta

        xs = self.xs[start:end]
        rewards = self.rewards[start:end]
        r = np.c_[rewards]

        F = np.concatenate(xs).T.reshape(-1, self.k)
        SIGMA_r = np.eye(len(rewards)) * self.sigma_r
        invSIGMA_r = np.linalg.inv(SIGMA_r)
        invSIGMA_theta = np.linalg.inv(self.SIGMA_theta)
        invSIGMA_theta_r = invSIGMA_theta + F.T.dot(invSIGMA_r).dot(F)

        SIGMA_theta_r = np.linalg.inv(invSIGMA_theta_r)
        mu_theta_r = SIGMA_theta_r.dot(
            (F.T.dot(invSIGMA_r).dot(r) + invSIGMA_theta.dot(self.mu_theta))
        )

        return mu_theta_r, SIGMA_theta_r

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
