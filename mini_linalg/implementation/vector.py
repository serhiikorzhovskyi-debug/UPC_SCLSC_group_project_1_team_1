import numpy as np


class Vector:
    def __init__(self, data: list[float]):
        self.data = np.array(data, dtype=float)

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"
