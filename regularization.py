def lasso (alpha, weights_vector):
    return alpha * sum([abs(value) for value in weights_vector])