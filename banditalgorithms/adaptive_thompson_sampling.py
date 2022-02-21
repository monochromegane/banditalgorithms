from typing import List, Optional, cast

import numpy as np


class AdaptiveThompsonSampling:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        sigma_theta: float = 1.0,
        sigma_r: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms
        self.dim_context = dim_context
        self.estimators = [
            AdaptiveThompsonSamplingEstimator(
                dim_context, sigma_theta, sigma_r, self.random
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
        self, k: int, sigma_theta: float, sigma_r: float, rs: np.random.RandomState
    ) -> None:
        self.random = rs
        self.k = k
        self.sigma_theta = sigma_theta
        self.sigma_r = sigma_r
        self.SIGMA_theta = np.eye(k) * sigma_theta
        self.mu_theta = np.zeros([k, 1])

        self.rewards: List[float] = []
        self.xs: List[np.ndarray] = []

    def estimate(self, x: np.ndarray) -> float:
        F = np.concatenate(self.xs).T.reshape(-1, self.k)
        SIGMA_r = np.eye(len(self.rewards)) * self.sigma_r
        invSIGMA_r = np.linalg.inv(SIGMA_r)
        invSIGMA_theta = np.linalg.inv(self.SIGMA_theta)
        invSIGMA_theta_r = invSIGMA_theta + F.T.dot(invSIGMA_r).dot(F)

        SIGMA_theta_r = np.linalg.inv(invSIGMA_theta_r)
        mu_theta_r = SIGMA_theta_r.dot(
            (F.T.dot(invSIGMA_r).dot(self.rewards) + invSIGMA_theta.dot(self.mu_theta))
        )

        theta = self.random.multivariate_normal(mu_theta_r.reshape(-1), SIGMA_theta_r)
        return cast(float, x.reshape(-1).dot(theta)[0][0])

    def update(self, reward: float, x: np.ndarray) -> None:
        self.rewards.append(reward)
        self.xs.append(x)
