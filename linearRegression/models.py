from sklearn import linear_model

def get_model():
    return linear_model.SGDRegressor(
        penalty = None,
        alpha = 0.0,
        l1_ratio = 0,
        max_iter = 5,
        tol = None,
    )