import matplotlib.pyplot as plt

from _genetic import genetic
import parameters

results = []

for mutation_amplitude in parameters.genetic_calculation['mutation_amplitude']:
    for regularization_strength in parameters.genetic_calculation['regularization_strength']:
        for size_population in parameters.genetic_calculation['size_population']:
            for random_gene_amplitude in parameters.genetic_calculation['random_gene_amplitude']:
                for parent_proportion in parameters.genetic_calculation['parent_proportion']:
                    parameters_combination = {
                        'mutation_amplitude': mutation_amplitude,
                        'regularization_strength': regularization_strength,
                        'size_population': size_population,
                        'num_generations': parameters.genetic_calculation['num_generations'],
                        'parent_proportion': parent_proportion,
                        'random_gene_amplitude': random_gene_amplitude
                    }
                    train_loss, test_loss = genetic(parameters_combination)
                    results.append({'train_loss': train_loss, 'test_loss': test_loss, 'last_test_loss': test_loss[-1], 'parameters': parameters_combination})

results = sorted(results, key=lambda k: k['last_test_loss'])

print(results[0])

plt.plot(results[0]['train_loss'], label='train_loss')
plt.plot(results[0]['test_loss'], label='test_loss')
plt.xlabel('iterations')
plt.ylabel('NRMSE')
plt.legend()
plt.show()
