import copy
from typing import List, Optional, cast

import numpy as np


class Particle:
    def __init__(self, dim_context: int, rs: np.random.RandomState) -> None:
        self.dim_context = dim_context
        self.random = rs

        self.sigma_epsilon = 1.0

    def mu_w(self) -> np.ndarray:
        ...

    def SIGMA_w(self) -> np.ndarray:
        ...

    def rho(self, reward: float, x: np.ndarray) -> float:
        ...

    def update_eta(self, reward: float, x: np.ndarray) -> None:
        ...

    def update_params(self, reward: float, x: np.ndarray) -> None:
        ...


class Particles:
    def __init__(
        self, dim_context: int, num_particles: int, rs: np.random.RandomState
    ) -> None:
        self.dim_context = dim_context
        self.random = rs
        self.num_particles = num_particles
        self.P = [Particle(dim_context, rs) for p in range(num_particles)]

    def eval(self, x: np.ndarray) -> float:
        mu = self._mu_wk().reshape(-1)
        SIGMA = self._SIGMA_wk()
        wk = self.random.multivariate_normal(mu, SIGMA)
        return cast(float, x.T.dot(wk)[0][0])

    def update(self, reward: float, x: np.ndarray) -> None:
        rhos = [p.rho(reward, x) for p in self.P]
        weights = [rho / sum(rhos) for rho in rhos]
        resamples = self._resampling(weights)

        P = [copy.deepcopy(self.P[idx]) for idx in resamples]
        self.P = P

        for p in self.P:
            p.update_eta(reward, x)
            p.update_params(reward, x)

    def _mu_wk(self) -> np.ndarray:
        mu = sum([p.mu_w() for p in self.P]) / self.num_particles
        return cast(np.ndarray, mu)

    def _SIGMA_wk(self) -> np.ndarray:
        SIGMA = sum([p.sigma_epsilon * p.SIGMA_w() for p in self.P]) / (
            self.num_particles ** 2
        )
        return cast(np.ndarray, SIGMA)

    def _resampling(
        self, weights: List[float], u0: Optional[float] = None
    ) -> List[int]:
        n_particles = len(weights)
        idx = np.array(list(range(n_particles)))
        if u0 is None:
            u0 = self.random.uniform(0, 1.0 / n_particles)
        u = [1.0 / n_particles * i + u0 for i in range(n_particles)]
        w_cumsum = np.cumsum(weights)
        return [self._f_inv(w_cumsum, idx, val) for val in u]

    def _f_inv(self, w_cumsum: np.ndarray, idx: np.ndarray, u: float) -> int:
        if not np.any(w_cumsum < u):
            return 0
        k = np.max(idx[w_cumsum < u])
        return cast(int, k + 1)


class TimeVaryingThompsonSampling:
    def __init__(
        self,
        num_arms: int,
        dim_context: int,
        *,
        num_particles: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.random = np.random.RandomState(seed)

        self.num_arms = num_arms
        self.dim_context = dim_context
        self.filters = [
            Particles(dim_context, num_particles, self.random) for _ in range(num_arms)
        ]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        return cast(int, np.argmax([f.eval(x) for f in self.filters]))

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]
        self.filters[idx_arm].update(reward, x)
