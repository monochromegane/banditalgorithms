import numpy as np
from banditalgorithms import inverse_matrix

def test_inverse_matrix_initialize() -> None:
    dim = 3
    inv = inverse_matrix.InverseMatrix(dim)
    actual = np.eye(dim)
    assert np.allclose(inv.data, actual)


def test_inverse_matrix() -> None:
    dim = 3
    inv = inverse_matrix.InverseMatrix(dim)
    x = np.c_[np.array([1., 2., 3.])]

    mat = np.eye(dim)

    mat += x.dot(x.T)
    actual = np.linalg.inv(mat)
    inv.update(x)

    assert np.allclose(inv.data, actual)
