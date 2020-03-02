from scipy.sparse.linalg import lsqr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from parameters import filename
from parameters import least_squares as parameters
import prepare_data
from metrics import nrmse

training_object_count, feature_count, training_features, training_labels, test_features, test_labels\
    = prepare_data.read_from_file(filename)

training_features, training_labels = prepare_data.prepare_data(training_features, training_labels)
test_features, test_labels = prepare_data.prepare_data(test_features, test_labels)


#model = lsqr(training_features, training_labels, iter_lim=parameters['iterations'])[0]
#predicted_values = np.dot(model, np.transpose(test_features))

model = Ridge(solver='lsqr', max_iter=parameters['iterations'], alpha=parameters['regularization_strength'])

model.fit(training_features, training_labels)

training_predicted_values = model.predict(training_features)
training_nrmse = nrmse(training_predicted_values, training_labels)

test_predicted_values = model.predict(test_features)
test_nrmse = nrmse(test_predicted_values, test_labels)

#plt.plot(predicted_values)
#plt.plot(test_labels)
#plt.show()

print('Training NRMSE: {}'.format(training_nrmse))
print('Test NRMSE: {}'.format(test_nrmse))