import matplotlib.pyplot as plt

import parameters
from gradient_descent import gradient_descent
from genetic import genetic

#model = gradient_descent
model = genetic
num_iterations = parameters.num_iterations
best_losses = []

for max_iterations in range(num_iterations):
    #train_loss, test_loss, _ = model({'learning_rate': parameters.gradient_descent['learning_rate'],
    #                               'regularization_strength': parameters.gradient_descent['regularization_strength'],
    #                               'num_iterations': max_iterations + 1})

    train_loss, test_loss, _ = model({'mutation_amplitude': parameters.genetic['mutation_amplitude'], 'regularization_strength': parameters.genetic['regularization_strength'],
                                                          'num_chromosomes': parameters.genetic['num_chromosomes'], 'num_generations': num_iterations + 1,
                                                          'parent_proportion': parameters.genetic['parent_proportion'], 'random_gene_amplitude': parameters.genetic['random_gene_amplitude']})

    best_losses.append(min(test_loss))

plt.plot(best_losses)
plt.xlabel('max_iterations')
plt.ylabel('best_nrmse')
plt.show()
