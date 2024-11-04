import numpy as np

class NormalVariable:

    def __init__(self, mean: float, std: float):
        self.__mean = mean
        self.__std = std
        self.__rng = np.random.default_rng()
    
    def set_mean(self, new_mean: float):
        self.__mean = new_mean
    
    def get(self, n: int = 1) -> float | np.ndarray[float]:
        result = np.zeros(n) + self.__mean + (self.__rng.normal(size=n) * self.__std)
        return result[0] if n == 1 else result


class VariableRelation:

    def __init__(self, inputs: list[tuple[NormalVariable, float]], output: NormalVariable):
        if type(inputs[0]) is NormalVariable:
            raise TypeError("Weights must be included in `inputs`!")

        self.__inputs = inputs
        self.__output = output
    
    def get(self, n: int = 1) -> np.ndarray[float]:
        result = np.zeros((len(self.__inputs) + 1, n))
        for i, (input, weight) in enumerate(self.__inputs):
            result[i] = input.get(n)
            result[-1] += result[i] * weight

        return result
