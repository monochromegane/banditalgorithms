from typing import List, Optional

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
        ...

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        ...
