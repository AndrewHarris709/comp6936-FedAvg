import numpy as np

result = None
matrix_shape = (17, 17)

generator = np.random.default_rng()

while result is None:
    result = generator.uniform(0, 0.51, matrix_shape)
    result = np.triu(result)
    np.fill_diagonal(result, 1)
    result = result + result.T - np.diag(np.diag(result))

    print(result)

    try:
        np.linalg.cholesky(result)
    except:
        result = None

print(repr(result))
