import numpy as np

from .vector import Vector


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

    @staticmethod
    def batch_matvec(
        matrices: list["Matrix"], vectors: list[Vector]
    ) -> list[Vector]:
        if len(matrices) != len(vectors):
            raise ValueError("Batch sizes must match")
        return [
            matrix.matvec(vector)
            for matrix, vector in zip(matrices, vectors)
        ]

    @staticmethod
    def batch_matmul(
        left_matrices: list["Matrix"], right_matrices: list["Matrix"]
    ) -> list["Matrix"]:
        if len(left_matrices) != len(right_matrices):
            raise ValueError("Batch sizes must match")
        return [
            left.matmul(right)
            for left, right in zip(left_matrices, right_matrices)
        ]
