import math
from typing import List, Optional, cast

import numpy as np

from . import inverse_matrix as mat


class DynamicLinUCB:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        lambda_: float = 1.0,
        delta1: float = 0.1,
        delta2: float = 0.1,
        delta1_tilde: Optional[float] = None,
        sigma2: float = 1e-2,
    ) -> None:
        self.num_arms = num_arms
        self.dim_context = dim_context

        self.lambda_ = lambda_
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta1_tilde = delta1_tilde
        self.sigma2 = sigma2

        self.models = [
            DynamicLinUCBSlave(
                self.num_arms,
                self.dim_context,
                lambda_=self.lambda_,
                delta1=self.delta1,
                delta2=self.delta2,
                delta1_tilde=self.delta1_tilde,
                sigma2=self.sigma2,
            )
        ]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        idx_model = 0
        return self.models[idx_model].select(x)

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]
        idx_model = 0
        self.models[idx_model].update(idx_arm, reward, x)


class DynamicLinUCBSlave:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        lambda_: float = 1.0,
        delta1: float = 0.1,
        delta2: float = 0.1,
        delta1_tilde: Optional[float] = None,
        sigma2: float = 1e-2,
    ) -> None:
        self.num_arms = num_arms
        self.dim_context = dim_context

        self.lambda_ = lambda_
        self.delta1 = delta1
        self.delta2 = delta2
        if delta1_tilde is None:
            self.delta1_tilde = delta1
        self.sigma2 = sigma2

        self.invAs = [
            mat.InverseMatrix(dim_context, lambda_=lambda_) for _ in range(num_arms)
        ]
        self.bs = [np.zeros([dim_context, 1]) for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]

    def select(self, x: np.ndarray) -> int:
        return cast(
            int, np.argmax([self._ucb_score(i, x) for i in range(self.num_arms)])
        )

    def update(self, idx_arm: int, reward: float, x: np.ndarray) -> None:
        self.bs[idx_arm] += reward * x
        self.invAs[idx_arm].update(x)
        self.counts[idx_arm] += 1

    def _ucb_score(self, idx_arm: int, x: np.ndarray) -> float:
        theta_hat = self.invAs[idx_arm].data.dot(self.bs[idx_arm])
        reward_hat = cast(float, x.T.dot(theta_hat)[0][0])

        return reward_hat + self.B(idx_arm, x)

    def B(self, idx_arm: int, x: np.ndarray) -> float:
        d = self.dim_context
        size = self.counts[idx_arm]
        alpha = self.sigma2 * math.sqrt(
            d * math.log(1.0 + (size / self.lambda_ * self.delta1))
        )

        return alpha * math.sqrt(x.T.dot(self.invAs[idx_arm].data).dot(x))
