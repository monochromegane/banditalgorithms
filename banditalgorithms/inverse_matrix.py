import numpy as np


class InverseMatrix:
    def __init__(self, dim: int, lambda_: float = 1.0) -> None:
        self.dim = dim
        self._inv = np.linalg.inv(lambda_ * np.eye(dim))

    @property
    def data(self) -> np.ndarray:
        return self._inv

    def update(self, x: np.ndarray) -> None:
        inv = self._inv

        self._inv -= (
            inv.dot(x)
            .dot(np.linalg.inv(np.eye(1) + x.T.dot(inv).dot(x)))
            .dot(x.T)
            .dot(inv)
        )  # Woodbury
