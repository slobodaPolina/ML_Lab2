import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import prepare_data
from metrics import NRMSE

# считываем и нормализуем данные
training_object_count, feature_count, training_features, training_labels, test_features, test_labels = prepare_data.read_from_file()
training_features = preprocessing.normalize(training_features, axis=0)
test_features = preprocessing.normalize(test_features, axis=0)

# ПОДБОР РЕГУЛЯРИЗАЦИИ:
metrics = []
# на самом деле тут шаг в 0.1 от 0 до 150
for i in range(0, 1500, 1):
    regularization = i / 10
    # на примере 100 итераций
    model = Ridge(solver='lsqr', max_iter=100, alpha=regularization)
    model.fit(training_features, training_labels)
    training_NRMSE = NRMSE(model.predict(training_features), training_labels)
    test_NRMSE = NRMSE(model.predict(test_features), test_labels)
    metrics.append({'regularization': regularization, 'training_NRMSE': training_NRMSE, 'test_NRMSE': test_NRMSE})

plt.plot([point['regularization'] for point in metrics], [point['training_NRMSE'] for point in metrics])
plt.xlabel('regularization')
plt.ylabel('training_NRMSE')
plt.show()

plt.plot([point['regularization'] for point in metrics], [point['test_NRMSE'] for point in metrics])
plt.xlabel('regularization')
plt.ylabel('test_NRMSE')
plt.show()
