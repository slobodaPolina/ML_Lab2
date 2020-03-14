import matplotlib.pyplot as plt
from _gradient_descent import gradient_descent
import parameters

results = []

for learning_rate in parameters.gradient_descent_calculation['learning_rate']:
    for regularization_strength in parameters.gradient_descent_calculation['regularization_strength']:
        for cross_val_parameter in parameters.gradient_descent_calculation['cross_val_parameter']:
            parameters_combination = {
                'learning_rate': learning_rate,
                'regularization_strength': regularization_strength,
                'cross_val_parameter': cross_val_parameter,
                'num_iterations': parameters.gradient_descent_calculation['num_iterations']
            }
            train_loss, test_loss = gradient_descent(parameters_combination)
            results.append({'train_loss': train_loss, 'test_loss': test_loss, 'last_test_loss': test_loss[-1], 'parameters': parameters_combination})

# сортировать по всему test_loss массиву не получится, будем по характерному последнему значению
results = sorted(results, key=lambda k: k['last_test_loss'])

print(results[0])

plt.plot(results[0]['train_loss'], label='train_loss')
plt.plot(results[0]['test_loss'], label='test_loss')
plt.xlabel('iterations')
plt.ylabel('loss(NRMSE)')
plt.legend()
plt.show()
