from typing import List, Optional, cast

import numpy as np


class ThompsonSampling:
    def __init__(self, num_arms: int, *, seed: Optional[int] = None) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms

        self.counts: List[float] = [0.0 for _ in range(self.num_arms)]
        self.rewards: List[float] = [0.0 for _ in range(self.num_arms)]

    def select(self) -> int:
        return cast(int, np.argmax(self._samples()))

    def update(self, idx_arm: int, reward: float) -> None:
        self.counts[idx_arm] += 1.0
        self.rewards[idx_arm] += reward

    def _samples(self) -> List[float]:
        return [self._sample(i) for i in range(self.num_arms)]

    def _sample(self, idx_arm: int) -> float:
        reward = self.rewards[idx_arm]
        count = self.counts[idx_arm]

        return self.random.beta(1.0 + reward, 1.0 + count - reward)
