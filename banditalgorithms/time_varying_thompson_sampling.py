from typing import List, Optional, cast

import numpy as np


class Particle:
    def __init__(self, dim_context: int, rs: np.random.RandomState) -> None:
        self.dim_context = dim_context
        self.random = rs


class Particles:
    def __init__(
        self, dim_context: int, num_particles: int, rs: np.random.RandomState
    ) -> None:
        self.dim_context = dim_context
        self.random = rs
        self.num_particles = num_particles
        self.P = [Particle(dim_context, rs) for p in range(num_particles)]

    def eval(self, x: np.ndarray) -> float:
        ...

    def update(self, reward: float, x: np.ndarray) -> None:
        ...


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
