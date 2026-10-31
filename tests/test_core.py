import pytest

from mini_linalg import Matrix, Vector


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


def test_matrix_unequal_row_lengths():
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])


def test_matrix_empty():
    with pytest.raises(ValueError):
        Matrix([])


def test_matvec_dimension_error_extra_elements():
    matrix = Matrix([[1, 2], [3, 4]])
    vector = Vector([1, 2, 3])
    with pytest.raises(ValueError):
        matrix.matvec(vector)


def test_matmul_dimension_error_extra_rows():
    left = Matrix([[1, 2], [3, 4]])
    right = Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        left.matmul(right)


def test_matmul_rectangular():
    left = Matrix([[1, 2, 3], [4, 5, 6]])
    right = Matrix([[7, 8], [9, 10], [11, 12]])
    result = left.matmul(right)
    assert [list(row) for row in result.data] == [[58, 64], [139, 154]]


def test_matvec_rectangular():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    vector = Vector([2, 3, 4])
    result = matrix.matvec(vector)
    assert list(result.data) == [20, 47]


def test_vector_repr():
    vector = Vector([1, 2, 3])
    assert repr(vector) == "Vector([1.0, 2.0, 3.0])"


def test_matrix_repr():
    matrix = Matrix([[1, 2], [3, 4]])
    assert repr(matrix) == "Matrix([[1.0, 2.0], [3.0, 4.0]])"


def test_batch_matvec():
    matrices = [Matrix([[1, 2]]), Matrix([[3, 4], [5, 6]])]
    vectors = [Vector([2, 3]), Vector([1, 2])]
    results = Matrix.batch_matvec(matrices, vectors)
    assert [list(result.data) for result in results] == [[8], [11, 17]]


def test_batch_matmul():
    left = [Matrix([[1, 2]]), Matrix([[1, 0], [0, 1]])]
    right = [Matrix([[3], [4]]), Matrix([[2, 3], [4, 5]])]
    results = Matrix.batch_matmul(left, right)
    assert [
        [list(row) for row in result.data] for result in results
    ] == [[[11]], [[2, 3], [4, 5]]]


def test_batch_matvec_empty():
    assert Matrix.batch_matvec([], []) == []


def test_batch_matmul_empty():
    assert Matrix.batch_matmul([], []) == []


def test_batch_matvec_different_batch_sizes():
    with pytest.raises(ValueError):
        Matrix.batch_matvec([Matrix([[1]])], [])


def test_batch_matmul_different_batch_sizes():
    with pytest.raises(ValueError):
        Matrix.batch_matmul([Matrix([[1]])], [])


def test_batch_matvec_dimension_mismatch():
    with pytest.raises(ValueError):
        Matrix.batch_matvec([Matrix([[1, 2]])], [Vector([1])])


def test_batch_matmul_dimension_mismatch():
    with pytest.raises(ValueError):
        Matrix.batch_matmul([Matrix([[1, 2]])], [Matrix([[1, 2]])])
