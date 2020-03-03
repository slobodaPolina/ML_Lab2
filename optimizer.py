import matplotlib.pyplot as plt

from genetic import genetic
from gradient_descent import gradient_descent
import parameters

results = []

#for learning_rate in parameters.gradient_descent['learning_rate']:
#    for regularization_strength in parameters.gradient_descent['regularization_strength']:
#        train_loss, test_loss, hyperparameters = gradient_descent({'learning_rate': learning_rate, 'regularization_strength': regularization_strength, 'num_iterations': parameters.gradient_descent['num_iterations']})

for mutation_amplitude in parameters.genetic['mutation_amplitude']:
    for regularization_strength in parameters.genetic['regularization_strength']:
        train_loss, test_loss, hyperparameters = genetic({'mutation_amplitude': mutation_amplitude, 'regularization_strength': regularization_strength,
                                                          'num_chromosomes': parameters.genetic['num_chromosomes'], 'num_generations': parameters.genetic['num_generations'],
                                                          'parent_proportion': parameters.genetic['parent_proportion'], 'random_gene_amplitude': parameters.genetic['random_gene_amplitude']})

        results.append({'train_loss': train_loss, 'test_loss': test_loss, 'last_test_loss': test_loss[-1], 'hyperparameters': hyperparameters})

results = sorted(results, key=lambda k: k['last_test_loss'])

print(results[0])

plt.plot(results[0]['train_loss'], label='train_loss')
plt.plot(results[0]['test_loss'], label='test_loss')
plt.xlabel('iterations')
plt.ylabel('nrmse')
plt.legend()
plt.show()