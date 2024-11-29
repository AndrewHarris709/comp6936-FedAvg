# Unit Tests

9 unit tests are provided for testing Cholesky generation.

To run the tests use the following in the main folder of the project:

```
pytest tests/
```

The tests cover:

- ```test_corr_not_symmetric```

    Checks if symmetry is handled properly in ```CholeskyGenerator```

- ```test_corr_excessive_values```

    Checks if wrong correlation values are handled properly in ```CholeskyGenerator```

- ```test_corr_impossible_correlation```

    Checks if the correlation matrix validity is handled properly in ```CholeskyGenerator```

- ```test_non_numpy_arrays```

    Checks if ```CholeskyGenerator``` is functioning properly with lists

- ```test_simple_gen```

    Checks if ```CholeskyGenerator``` is correctly generating data

- ```test_simple_gen_shift```

    Checks if ```CholeskyGenerator``` is correctly generating data with a shift

- ```test_full_shift```

    Checks if `test_mult is correctly shifting the data

- ```test_distributions```

    Checks generated data distribution in ```CholeskyGenerator```

- ```test_mult```

    Checks matrix multiplication in ```CholeskyGenerator``` for changing the range of generated data.