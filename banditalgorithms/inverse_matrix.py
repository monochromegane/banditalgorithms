import numpy as np


class InverseMatrix:
    def __init__(self, dim: int, lambda_: float = 1.0) -> None:
        self.dim = dim
        self._inv = (1.0 / lambda_) * np.eye(dim)
        self.identity = np.eye(1)
        self.out = np.zeros((self.dim, self.dim))

    @property
    def data(self) -> np.ndarray:
        return self._inv

    def update(self, x: np.ndarray) -> None:
        inv = self._inv

        xTinv = x.T.dot(inv)
        np.outer(
            inv.dot(x).dot(np.linalg.inv(self.identity + xTinv.dot(x))),
            xTinv,
            out=self.out,
        )  # Woodbury
        self._inv -= self.out
