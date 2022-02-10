from typing import List, Optional, cast

import numpy as np

from . import inverse_matrix as mat


class LinearThompsonSampling:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        sigma2: float = 1.0,
        sigma2_0: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms
        self.dim_context = dim_context
        self.sigma2 = sigma2
        self.sigma2_0 = sigma2_0 if sigma2_0 is not None else sigma2

        lambda_ = self.sigma2 / self.sigma2_0
        self.invAs = [
            mat.InverseMatrix(dim_context, lambda_=lambda_) for _ in range(num_arms)
        ]
        self.bs = [np.zeros([dim_context, 1]) for _ in range(num_arms)]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        return cast(int, np.argmax(self._samples(x)))

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]

        self.bs[idx_arm] += reward * x
        self.invAs[idx_arm].update(x)

    def _samples(self, x: np.ndarray) -> List[float]:
        return [self._sample(i, x) for i in range(self.num_arms)]

    def _sample(self, idx_arm: int, x: np.ndarray) -> float:
        invA = self.invAs[idx_arm].data
        b = self.bs[idx_arm]

        mu = invA.dot(b).reshape(-1)
        SIGMA = self.sigma2 * invA
        theta_hat = self.random.multivariate_normal(mu, SIGMA)

        return cast(float, theta_hat.dot(x))
