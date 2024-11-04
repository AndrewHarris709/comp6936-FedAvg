import numpy as np

class NormalVariable:

    def __init__(self, mean: float, std: float):
        self.__mean = mean
        self.__std = std
        self.__rng = np.random.default_rng()
    
    def get(self, n: int = 1) -> float | np.ndarray[float]:
        result = np.zeros(n) + self.__mean + (self.__rng.normal(size=n) * self.__std)
        return result[0] if n == 1 else result
