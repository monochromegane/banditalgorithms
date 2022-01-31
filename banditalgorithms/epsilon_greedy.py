from typing import Optional

import numpy as np


class EpsilonGreedy:
    def __init__(
        self, num_arms: int, *, epsilon: float = 0.1, seed: Optional[int] = None
    ) -> None:
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.rs = np.random.RandomState(seed)

    def select(self) -> int:
        ...

    def update(self, idx_arm: int, reward: float) -> None:
        ...
