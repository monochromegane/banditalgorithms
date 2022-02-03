import math
from typing import List, cast

import numpy as np


class UCB1:
    def __init__(self, num_arms: int) -> None:
        self.num_arms = num_arms

        self.counts: List[float] = [0.0 for _ in range(self.num_arms)]
        self.rewards: List[float] = [0.0 for _ in range(self.num_arms)]

    def select(self) -> int:
        for i, count in enumerate(self.counts):
            if count == 0.0:
                return i

        return cast(int, np.argmax(self._ucb_scores()))

    def update(self, idx_arm: int, reward: float) -> None:
        self.counts[idx_arm] += 1.0
        self.rewards[idx_arm] += reward

    def _ucb_scores(self) -> List[float]:
        return [self._ucb_score(i) for i in range(self.num_arms)]

    def _ucb_score(self, idx_arm: int) -> float:
        reward = self.rewards[idx_arm]
        count = self.counts[idx_arm]
        n = sum(self.counts)

        if count == 0.0:
            return 0.0
        else:
            return (reward / count) + math.sqrt((2.0 * math.log(n)) / count)
