import numpy as np


class Vector:
    def __init__(self, data: list[float]):
        self.data = np.array(data, dtype=float)

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"


class Matrix:
    def __init__(self, data: list[list[float]]):
        rows = [list(row) for row in data]

        if not rows:
            raise ValueError("Matrix must have at least one row")

        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Matrix rows must have equal length")

        if not rows[0]:
            raise ValueError("Matrix must have at least one column")

        self.data = np.array(rows, dtype=float)

    def __repr__(self) -> str:
        return f"Matrix({self.data.tolist()})"

    def matvec(self, vector: Vector) -> Vector:
        if self.data.shape[1] != vector.data.shape[0]:
            raise ValueError("Incompatible matrix and vector dimensions")

        values = self.data @ vector.data

        return Vector(values)

    def matmul(self, other: "Matrix") -> "Matrix":
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError("Incompatible matrix dimensions")

        values = self.data @ other.data

        return Matrix(values)
