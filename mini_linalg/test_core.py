import pytest

from mini_linalg.core import Matrix, Vector


def test_vector_creation():
    assert list(Vector([1, 2, 3]).data) == [1, 2, 3]


def test_matrix_creation():
    matrix = Matrix([[1, 2], [3, 4]])
    assert [list(row) for row in matrix.data] == [[1, 2], [3, 4]]


def test_matvec():
    result = Matrix([[1, 2], [3, 4]]).matvec(Vector([5, 6]))
    assert list(result.data) == [17, 39]


def test_matmul():
    left = Matrix([[1, 2], [3, 4]])
    right = Matrix([[5, 6], [7, 8]])
    result = left.matmul(right)
    assert [list(row) for row in result.data] == [[19, 22], [43, 50]]


def test_matvec_dimension_error():
    with pytest.raises(ValueError):
        Matrix([[1, 2]]).matvec(Vector([1]))


def test_matmul_dimension_error():
    with pytest.raises(ValueError):
        Matrix([[1, 2]]).matmul(Matrix([[1, 2]]))
