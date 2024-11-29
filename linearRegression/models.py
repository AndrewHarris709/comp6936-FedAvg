from sklearn import linear_model

def get_model(max_iter: int = 1):
    return linear_model.SGDRegressor(
        penalty = None,
        alpha = 0.0,
        l1_ratio = 0,
        max_iter = max_iter,
        tol = None,
    )