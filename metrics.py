# оценка ошибки аппроксимации точек (нормализованная)
def NRMSE(y_predicted, y_actual):
    n = len(y_actual)
    sum_squares = sum([(y_predicted_i - y_actual_i) ** 2 for y_predicted_i, y_actual_i in zip(y_predicted, y_actual)])
    return (
          (sum_squares / n) ** (1/2)
    ) / (max(y_actual) - min(y_actual))
