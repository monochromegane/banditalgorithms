from typing import List, Optional, cast

import numpy as np


class EpsilonGreedy:
    def __init__(
        self, num_arms: int, *, epsilon: float = 0.1, seed: Optional[int] = None
    ) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms
        self.epsilon = epsilon

        self.counts: List[float] = [0.0 for _ in range(self.num_arms)]
        self.rewards: List[float] = [0.0 for _ in range(self.num_arms)]

    def select(self) -> int:
        if self._is_exploitation():
            return cast(int, np.argmax(self._theta_hats()))
        else:
            return self.random.choice(len(self._theta_hats()))

    def update(self, idx_arm: int, reward: float) -> None:
        self.counts[idx_arm] += 1.0
        self.rewards[idx_arm] += reward

    def _is_exploitation(self) -> bool:
        return self.random.random() > self.epsilon

    def _theta_hats(self) -> List[float]:
        return [
            reward / count if count > 0.0 else 0.0
            for count, reward in zip(self.counts, self.rewards)
        ]
