def rmse(y_pred, y_actual):
    return (sum([(y_pred_i - y_actual_i)**2 for y_pred_i, y_actual_i in zip(y_pred, y_actual)])/len(y_actual))**(1/2)


def nrmse(y_pred, y_actual):
    return rmse(y_pred, y_actual)/(max(y_actual) - min(y_actual))
