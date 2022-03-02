import math
from typing import List, Optional, cast

import numpy as np
from scipy import special

from . import inverse_matrix as mat


class DynamicLinUCBSlave:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        lambda_: float = 1.0,
        delta1: float = 0.1,
        delta2: float = 0.1,
        sigma2: float = 1e-2,
        tau: int = 10,
    ) -> None:
        self.num_arms = num_arms
        self.dim_context = dim_context

        self.lambda_ = lambda_
        self.delta1 = delta1
        self.delta2 = delta2
        self.sigma2 = sigma2
        self.sigma = math.sqrt(sigma2)
        self.tau = tau
        self.epsilon = cast(
            float, math.sqrt(2.0) * self.sigma * special.erfinv(1.0 - self.delta1)
        )

        self.invAs = [
            mat.InverseMatrix(dim_context, lambda_=lambda_) for _ in range(num_arms)
        ]
        self.bs = [np.zeros([dim_context, 1]) for _ in range(num_arms)]
        self.counts = [0 for _ in range(num_arms)]
        self.es: List[float] = []
        self.e_hat = 0.0
        self.d = 0.0

    def select(self, x: np.ndarray) -> int:
        return cast(int, np.argmax([score for score in self._ucb_scores(x)]))

    def update(self, idx_arm: int, reward: float, x: np.ndarray) -> None:
        e = 1.0 if self._exceed_confidence_bound(idx_arm, reward, x) else 0.0
        self.es.append(e)

        self.bs[idx_arm] += reward * x
        self.invAs[idx_arm].update(x)
        self.counts[idx_arm] += 1

        recently_es = self._recently_es()
        self.e_hat = sum(recently_es) / len(recently_es)
        self.d = math.sqrt(math.log(1.0 / self.delta2) / (2 * len(recently_es)))

    def _ucb_scores(self, x: np.ndarray) -> List[float]:
        return [self._ucb_score(i, x) for i in range(self.num_arms)]

    def _ucb_score(self, idx_arm: int, x: np.ndarray) -> float:
        return self._reward_hat(idx_arm, x) + self.B(idx_arm, x)

    def _reward_hat(self, idx_arm: int, x: np.ndarray) -> float:
        theta_hat = self.invAs[idx_arm].data.dot(self.bs[idx_arm])
        return cast(float, x.T.dot(theta_hat)[0][0])

    def _exceed_confidence_bound(
        self, idx_arm: int, reward: float, x: np.ndarray
    ) -> bool:
        return abs(self._reward_hat(idx_arm, x) - reward) > (
            self.B(idx_arm, x) + self.epsilon
        )

    def _recently_es(self) -> List[float]:
        return self.es[max(0, len(self.es) - self.tau) :]

    def B(self, idx_arm: int, x: np.ndarray) -> float:
        d = self.dim_context
        size = self.counts[idx_arm]
        alpha = self.sigma2 * math.sqrt(
            d * math.log(1.0 + (size / self.lambda_ * self.delta1))
            + math.sqrt(self.lambda_)
        )

        return alpha * math.sqrt(x.T.dot(self.invAs[idx_arm].data).dot(x))


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
        tau: int = 10,
    ) -> None:
        self.num_arms = num_arms
        self.dim_context = dim_context

        self.lambda_ = lambda_
        self.delta1 = delta1
        self.delta2 = delta2
        if delta1_tilde is None:
            self.delta1_tilde = delta1
        self.sigma2 = sigma2
        self.tau = tau

        self.models = [self._create_new_slave_model()]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        idx_model = 0
        return self.models[idx_model].select(x)

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]

        models = []
        create_new_flag = True
        for idx_model in range(len(self.models)):
            m = self.models[idx_model]
            m.update(idx_arm, reward, x)

            if self._keep_model(m):
                create_new_flag = False
                models.append(m)
            elif self._discard_model(m):
                # Discard slave model m
                pass

        if create_new_flag or len(models) == 0:
            models.append(self._create_new_slave_model())

        self.models = models

    def _keep_model(self, m: DynamicLinUCBSlave) -> bool:
        return m.e_hat < self.delta1_tilde + m.d

    def _discard_model(self, m: DynamicLinUCBSlave) -> bool:
        return m.e_hat >= self.delta1 + m.d

    def _create_new_slave_model(self) -> DynamicLinUCBSlave:
        return DynamicLinUCBSlave(
            self.num_arms,
            self.dim_context,
            lambda_=self.lambda_,
            delta1=self.delta1,
            delta2=self.delta2,
            sigma2=self.sigma2,
            tau=self.tau,
        )
