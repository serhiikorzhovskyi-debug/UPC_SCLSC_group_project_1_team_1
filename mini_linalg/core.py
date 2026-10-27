class Vector:
    def __init__(self, data: list[float]):
        self.data = list(data)

    def __repr__(self) -> str:
        return f"Vector({self.data})"


class Matrix:
    def __init__(self, data: list[list[float]]):
        rows = [list(row) for row in data]
        if not rows:
            raise ValueError("Matrix must have at least one row")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Matrix rows must have equal length")
        self.data = rows

    def __repr__(self) -> str:
        return f"Matrix({self.data})"

    def matvec(self, vector: Vector) -> Vector:
        if len(self.data[0]) != len(vector.data):
            raise ValueError("Incompatible matrix and vector dimensions")
        values = [
            sum(a * b for a, b in zip(row, vector.data))
            for row in self.data
        ]
        return Vector(values)

    def matmul(self, other: "Matrix") -> "Matrix":
        if len(self.data[0]) != len(other.data):
            raise ValueError("Incompatible matrix dimensions")
        columns = list(zip(*other.data))
        values = [
            [sum(a * b for a, b in zip(row, col)) for col in columns]
            for row in self.data
        ]
        return Matrix(values)
