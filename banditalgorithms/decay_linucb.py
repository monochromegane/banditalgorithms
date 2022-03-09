from typing import List, cast

import numpy as np


class DecayLinUCB:
    def __init__(
        self, num_arms: int, dim_context: int, *, alpha: float = 1.0, gamma: float = 1.0
    ) -> None:
        self.num_arms = num_arms
        self.dim_context = dim_context
        self.alpha = alpha
        self.gamma = gamma

        self.As = [np.eye(dim_context) for _ in range(num_arms)]
        self.bs = [np.zeros([dim_context, 1]) for _ in range(num_arms)]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        return cast(int, np.argmax(self._ucb_scores(x)))

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]

        self.As[idx_arm] *= self.gamma
        self.bs[idx_arm] *= self.gamma
        self.As[idx_arm] += x.dot(x.T)
        self.bs[idx_arm] += reward * x

    def _ucb_scores(self, x: np.ndarray) -> List[float]:
        return [self._ucb_score(i, x) for i in range(self.num_arms)]

    def _ucb_score(self, idx_arm: int, x: np.ndarray) -> float:
        invA = np.linalg.inv(self.As[idx_arm])
        b = self.bs[idx_arm]
        reward_hat = x.T.dot(invA.dot(b))
        term_ucb = self.alpha * np.sqrt(x.T.dot(invA).dot(x))
        return cast(float, (reward_hat + term_ucb)[0][0])
