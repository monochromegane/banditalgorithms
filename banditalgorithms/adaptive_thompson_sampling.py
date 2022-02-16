from typing import List, Optional, cast

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
        self.estimators = [AdaptiveThompsonSamplingEstimator() for _ in range(num_arms)]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        return cast(
            int, np.argmax([estimator.estimate(x) for estimator in self.estimators])
        )

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]
        self.estimators[idx_arm].update(reward, x)


class AdaptiveThompsonSamplingEstimator:
    def __init__(self) -> None:
        ...

    def estimate(self, x: np.ndarray) -> float:
        ...

    def update(self, reward: float, x: np.ndarray) -> float:
        ...
