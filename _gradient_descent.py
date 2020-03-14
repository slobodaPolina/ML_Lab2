import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from metrics import NRMSE
from parameters import gradient_descent as parameters
import prepare_data


# случайным образом делим наши данные на тернировочные и часть для кросс-валидации. k - какую (1/k) часть отделить
def cross_validation_split(data, k):
    np.random.shuffle(data)
    features = data[:, :-1]
    labels = data[:, -1]
    split_point = len(labels) // k
    training_features = np.array(features[split_point:])
    cross_val_features = np.array(features[:split_point])
    training_labels = np.array(labels[split_point:])
    cross_val_labels = np.array(labels[:split_point])
    return training_features, cross_val_features, training_labels, cross_val_labels


# предсказываем ответы для нескольких запросов, пользуясь существующими весами
def predict(features, weights_vector):
    return [sum([weight * feature_val for weight, feature_val in zip(weights_vector, features[i])]) for i in range(len(features))]


def gradient_descent(parameters):
    train_losses, test_losses = [], []
    training_object_count, feature_count, training_features, training_labels, test_features, test_labels = prepare_data.read_from_file()
    training_features = preprocessing.normalize(training_features, axis=0)
    test_features = preprocessing.normalize(test_features, axis=0)

    # merge features with their own labels to get a normal array of vectors ^)
    train_data = np.hstack((training_features, np.reshape(training_labels, (-1, 1))))

    # array of random initial weights from 0 to 1
    weights = np.random.rand(feature_count + 1)
    cross_val_parameter = parameters['cross_val_parameter']

    for iteration in range(parameters['num_iterations']):
        print('Iteration {}'.format(iteration + 1))

        # разделяем данные, по части из них будет проходить кросс-валидация
        cross_val_training_features, cross_val_test_features, cross_val_training_labels, cross_val_test_labels = cross_validation_split(train_data, cross_val_parameter)

        # добавляю к массиву фич столбик единичек, для тренировочной и тестовой кросс-валидационной части
        cross_val_training_features = np.hstack((
            cross_val_training_features,
            np.ones((cross_val_training_features.shape[0], 1), dtype=cross_val_training_features.dtype)
        ))
        cross_val_test_features = np.hstack((
            cross_val_test_features,
            np.ones((cross_val_test_features.shape[0], 1), dtype=cross_val_test_features.dtype)
        ))

        predicted_values = predict(cross_val_training_features, weights)

        # абсолютные значения ошибок - разница между тем, что было значением cross_val_training и тем, что предсказали по cross_val_training_features
        absolute_error = [predicted_value - cross_val_training_label for predicted_value, cross_val_training_label in zip(predicted_values, cross_val_training_labels)]

        # T.dot перемножение матриц cross_val_training_features и absolute_error
        # градиент считается по cross_val_training_features, проверки будут по cross_val_test_features
        gradient = cross_val_training_features.T.dot(absolute_error) / cross_val_training_features.shape[0] + parameters['regularization_strength'] * weights

        # и обновляю веса на правильные
        weights = weights * (1 - parameters['learning_rate'] * parameters['regularization_strength']) + parameters['learning_rate'] * (-gradient)

        predicted_labels = predict(cross_val_test_features, weights)
        train_loss = NRMSE(predicted_labels, cross_val_test_labels)
        train_losses.append(train_loss)

        # и по всему тестовому набору
        predicted_labels = predict(test_features, weights)
        test_loss = NRMSE(predicted_labels, test_labels)
        test_losses.append(test_loss)
        print('Cross validation loss: {}, Test loss: {}'.format(train_loss, test_loss))

    return train_losses, test_losses


best_test_losses = []
best_train_losses = []

for max_iterations in range(parameters['num_iterations']):
    print("-------------Max iterations {} out of {} -----------------------".format(max_iterations + 1, parameters['num_iterations']))
    train_loss, test_loss = gradient_descent({
            'learning_rate': parameters['learning_rate'],
            'regularization_strength': parameters['regularization_strength'],
            'cross_val_parameter': parameters['cross_val_parameter'],
            'num_iterations': max_iterations + 1
    })
    best_test_losses.append(min(test_loss))
    best_train_losses.append(min(train_loss))

plt.plot(best_test_losses)
plt.xlabel('maximum iterations')
plt.ylabel('best_test_NRMSE')
plt.show()

plt.plot(best_train_losses)
plt.xlabel('maximum iterations')
plt.ylabel('best_train_NRMSE')
plt.show()
