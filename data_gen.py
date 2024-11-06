import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class GeneratorVariable(ABC):
    lower_base: float
    upper_base: float
    weight: float = 1
    rng: np.random.Generator = np.random.default_rng()

    def get_bases(self, n: int = 1) -> np.ndarray[float]:
        return self.rng.uniform(self.lower_base, self.upper_base, n)

    @abstractmethod
    def generate(self, bases: np.ndarray[float]) -> np.ndarray[float]:
        pass
    
    #def set_mean(self, new_mean: float):
        #self.__mean = new_mean
    
    #def get_base(self) -> float:
        #return self.__mean
    
    #def get(self, n: int = 1, means: np.ndarray[float] = None) -> float | np.ndarray[float]:
    #    if means is None:
    #        result = np.zeros(n) + self.__mean
    #    else:
    #        result = np.zeros(n) + means

    #    result += (self.__rng.normal(size=n) * self.__std)
    #    return result[0] if n == 1 else result


@dataclass
class NormalVariable(GeneratorVariable):
    std: float = 0
    noise: float = 0

    def generate(self, bases: np.ndarray[float]) -> np.ndarray[float]:
        result = self.rng.normal(loc=bases, size=len(bases), scale=self.std)
        result += self.rng.uniform(low=-self.noise, high=self.noise, size=len(bases))
        return result


class VariableRelation:

    def __init__(self, inputs: list[tuple[NormalVariable]], output: NormalVariable):
        if type(inputs[0]) is NormalVariable:
            raise TypeError("Weights must be included in `inputs`!")

        self.__inputs = inputs
        self.__output = output
    
    def get(self, n: int = 1) -> np.ndarray[float]:
        result = np.zeros((len(self.__inputs) + 1, n))
        for i, (input, weight) in enumerate(self.__inputs):
            bases = input.get_bases()
            result[i] = input.generate(bases)
            result[-1] += bases * weight
        
        result[-1] = self.__output.generate(result[-1])

        return result
