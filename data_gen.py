import numpy as np

class NormalVariable:

    def __init__(self, mean: float, std: float):
        self.__mean = mean
        self.__std = std
        self.__rng = np.random.default_rng()
    
    def set_mean(self, new_mean: float):
        self.__mean = new_mean
    
    def get_base(self) -> float:
        return self.__mean
    
    def get(self, n: int = 1, means: np.ndarray[float] = None) -> float | np.ndarray[float]:
        if means is None:
            result = np.zeros(n) + self.__mean
        else:
            result = np.zeros(n) + means

        result += (self.__rng.normal(size=n) * self.__std)
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
            result[-1] += input.get_base() * weight
        
        result[-1] = self.__output.get(n, result[-1])

        return result
