import numpy as np

class CholeskyGenerator:

    def __init__(self, corr: np.ndarray[float], shifts: np.ndarray[float] = None, mults: np.ndarray[float] = None,
                 rng: np.random.Generator = np.random.default_rng()):
        
        if type(corr) is list:
            corr = np.array(corr)
        if type(shifts) is list:
            shifts = np.array(shifts)
        if type(mults) is list:
            mults = np.array(mults)

        if mults is None:
            mults = np.zeros(len(corr))
        if shifts is None:
            shifts = np.zeros(len(corr))

        if not (corr==corr.T).all():
            raise TypeError("Correlation Matrix is not symmetric!")

        self.__L = np.linalg.cholesky(corr)
        self.__shifts = shifts.reshape((len(self.__L), 1))
        self.__mults = mults.reshape((len(self.__L), 1))
        self.__rng = rng

    def get(self, n: int = 1):
        result = self.__rng.normal(loc=0, scale=1, size=(len(self.__L), n)).reshape((len(self.__L), n))
        result = self.__L @ result
        result = result * self.__mults
        result = self.__shifts + result

        return result
