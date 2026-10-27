from mini_linalg.core import Matrix, Vector


def test_vector_creation():
    vector = Vector([1, 2, 3])
    assert vector.data == [1, 2, 3]


def test_matrix_creation():
    matrix = Matrix([[1, 2], [3, 4]])
    assert matrix.data == [[1, 2], [3, 4]]
