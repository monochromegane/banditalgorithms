from typing import List, Optional

import numpy as np


class AdaptiveThompsonSampling:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms
        self.dim_context = dim_context

    def select(self, ctx: List[float]) -> int:
        ...

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        ...
