import copy
from typing import List, Literal, Optional, cast

import numpy as np
from scipy import stats


class Particle:
    def __init__(
        self,
        dim_context: int,
        sigma2_xi: float,
        rs: np.random.Generator,
        multivariate_normal_method: Literal["svd", "eigh", "cholesky"] = "svd",
    ) -> None:
        self.dim_context = dim_context
        self.random = rs
        self.multivariate_normal_method = multivariate_normal_method

        # sigma
        self.alpha = 1.0
        self.beta = 1.0
        sigma2 = stats.invgamma.rvs(
            self.alpha, scale=self.beta, random_state=self.random
        )

        # c_w
        self.mu_c = np.c_[np.zeros(dim_context)]
        self.SIGMA_c = np.identity(dim_context)
        c_w = self.random.multivariate_normal(
            self.mu_c.reshape(-1),
            sigma2 * self.SIGMA_c,
            method=self.multivariate_normal_method,
        )
        self.c_w = np.c_[c_w]

        # theta
        self.mu_theta = np.c_[np.zeros(dim_context)]
        self.SIGMA_theta = np.identity(dim_context)
        theta = self.random.multivariate_normal(
            self.mu_theta.reshape(-1),
            sigma2 * self.SIGMA_theta,
            method=self.multivariate_normal_method,
        )
        self.theta = np.c_[theta]

        # eta (Kalman filter)
        self.sigma2_epsilon = sigma2  # Variance of observation error
        self.sigma2_xi = sigma2_xi  # Variance of state error

        self.mu_eta = np.c_[np.zeros(dim_context)]
        self.SIGMA_eta = np.identity(dim_context) * self.sigma2_xi
        eta = self.random.multivariate_normal(
            self.mu_eta.reshape(-1),
            self.SIGMA_eta,
            method=self.multivariate_normal_method,
        )
        self.eta = np.c_[eta]

    def mu_w(self) -> np.ndarray:
        # Orignal formula. However this value is different from the value of the true coefficient vector.
        # sigma2 = self.sigma2_epsilon
        # mu = self.mu_c + np.linalg.inv(self.SIGMA_eta + sigma2 * self.SIGMA_theta).dot(
        #     self.SIGMA_eta.dot(self.mu_theta)
        #     + sigma2 * self.SIGMA_theta.dot(self.mu_eta)
        # )

        # Modified formula.
        mu = self.mu_c + np.multiply(self.mu_eta, self.mu_theta)
        return cast(np.ndarray, mu)

    def SIGMA_w(self) -> np.ndarray:
        sigma2 = self.sigma2_epsilon
        SIGMA = (sigma2 * self.SIGMA_c) + (
            sigma2 * self.SIGMA_theta.dot(self.SIGMA_eta)
        ).dot(np.linalg.inv(self.SIGMA_eta + sigma2 * self.SIGMA_theta))
        return cast(np.ndarray, SIGMA)

    def rho(self, reward: float, x: np.ndarray) -> float:
        m = x.T.dot(self.c_w + np.multiply(self.theta, self.mu_eta))
        Q = self._Q(x)
        return cast(float, stats.norm.pdf(reward, loc=m[0][0], scale=Q[0][0]))

    def update_eta(self, reward: float, x: np.ndarray) -> None:
        Id = np.identity(self.dim_context) * self.sigma2_xi
        Q = self._Q(x)
        G = (Id + self.SIGMA_eta).dot(np.multiply(self.theta, x)).dot(np.linalg.inv(Q))

        self.mu_eta += G.dot(
            reward - x.T.dot(self.c_w + np.multiply(self.theta, self.eta))
        )
        self.SIGMA_eta += Id - G.dot(Q).dot(G.T)

        eta = self.random.multivariate_normal(
            self.mu_eta.reshape(-1),
            self.SIGMA_eta,
            method=self.multivariate_normal_method,
        )
        self.eta = np.c_[eta]

    def update_params(self, reward: float, x: np.ndarray) -> None:
        dim = self.dim_context
        z_t = np.hstack([x.T, np.multiply(x, self.eta).T]).T
        SIGMA = np.block(
            [
                [self.SIGMA_c, np.zeros([dim, dim])],
                [np.zeros([dim, dim]), self.SIGMA_theta],
            ]
        )
        mu = np.hstack([self.mu_c.T, self.mu_theta.T]).T

        SIGMA_p = np.linalg.inv(np.linalg.inv(SIGMA) + z_t.dot(z_t.T))
        mu_p = SIGMA_p.dot((z_t * reward) + np.linalg.inv(SIGMA).dot(mu))

        alpha_p = self.alpha + 0.5
        beta_p = (
            self.beta
            + 0.5
            * (
                mu.T.dot(np.linalg.inv(SIGMA)).dot(mu)
                + (reward * reward)
                - mu_p.T.dot(np.linalg.inv(SIGMA_p)).dot(mu_p)
            )[0][0]
        )

        sigma2 = stats.invgamma.rvs(alpha_p, scale=beta_p, random_state=self.random)
        v = self.random.multivariate_normal(
            mu_p.reshape(-1), sigma2 * SIGMA_p, method=self.multivariate_normal_method
        )

        self.mu_c, self.mu_theta = [np.c_[mu_] for mu_ in np.split(mu_p, 2)]
        self.SIGMA_c, _, _, self.SIGMA_theta = sum(
            [np.split(s, 2, axis=1) for s in np.split(SIGMA_p, 2, axis=0)], []
        )
        self.alpha = alpha_p
        self.beta = beta_p
        self.sigma2_epsilon = sigma2
        self.c_w, self.theta = [np.c_[v_] for v_ in np.split(v, 2)]

    def _Q(self, x: np.ndarray) -> np.ndarray:
        Id = np.identity(self.dim_context) * self.sigma2_xi
        x_theta = np.multiply(x, self.theta)
        Q = self.sigma2_epsilon + x_theta.T.dot(Id + self.SIGMA_eta).dot(x_theta)
        return cast(np.ndarray, Q)


class Particles:
    def __init__(
        self,
        dim_context: int,
        num_particles: int,
        sigma2_xi: float,
        rs: np.random.Generator,
        multivariate_normal_method: Literal["svd", "eigh", "cholesky"] = "svd",
    ) -> None:
        self.dim_context = dim_context
        self.random = rs
        self.multivariate_normal_method = multivariate_normal_method
        self.num_particles = num_particles
        self.P = [
            Particle(dim_context, sigma2_xi, self.random) for p in range(num_particles)
        ]

    def eval(self, x: np.ndarray) -> float:
        mu = self._mu_wk().reshape(-1)
        SIGMA = self._SIGMA_wk()
        wk = self.random.multivariate_normal(
            mu, SIGMA, method=self.multivariate_normal_method
        )
        return cast(float, x.T.dot(wk)[0])

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
        SIGMA = sum([p.sigma2_epsilon * p.SIGMA_w() for p in self.P]) / (
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
        sigma2_xi: float = 1.0,
        seed: Optional[int] = None,
        multivariate_normal_method: Literal["svd", "eigh", "cholesky"] = "svd",
    ) -> None:
        self.random = np.random.default_rng(seed)

        self.num_arms = num_arms
        self.dim_context = dim_context
        self.filters = [
            Particles(
                dim_context,
                num_particles,
                sigma2_xi,
                self.random,
                multivariate_normal_method,
            )
            for _ in range(num_arms)
        ]

    def select(self, ctx: List[float]) -> int:
        x = np.c_[np.array(ctx)]
        return cast(int, np.argmax([f.eval(x) for f in self.filters]))

    def update(self, idx_arm: int, reward: float, ctx: List[float]) -> None:
        x = np.c_[np.array(ctx)]
        self.filters[idx_arm].update(reward, x)
