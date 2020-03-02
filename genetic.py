import numpy as np
import matplotlib.pyplot as plt

import prepare_data
from choose_parents import choose_parents, predict, fitness
from reproduce import reproduce
from parameters import genetic as parameters
from parameters import filename
from mutate import mutate
from metrics import nrmse

training_object_count, feature_count, training_features, training_labels, test_features, test_labels\
    = prepare_data.read_from_file(filename)

num_generations = parameters['num_generations']
num_chromosomes = parameters['num_chromosomes']
num_parents = int(num_chromosomes * parameters['parent_proportion'])
num_children = num_chromosomes - num_parents
num_genes = feature_count

chromosomes_shape = (num_chromosomes, num_genes + 1)
chromosomes = np.random.uniform(low=-parameters['random_gene_amplitude'], high=parameters['random_gene_amplitude'], size=chromosomes_shape)

for generation in range(num_generations):
    parents = choose_parents(chromosomes, training_features, training_labels, num_parents)
    children = reproduce(parents, num_children, num_parents, num_genes + 1)

    children = mutate(children, parameters['mutation_amplitude'])
    chromosomes = np.concatenate((parents, children), axis=0)

    print('Generation {}. Best training NRMSE: {}'.format(generation+1, nrmse(predict(training_features, parents[0]), training_labels)))
    print('Generation {}. Best fitness: {}'.format(generation+1, fitness(predict(training_features, parents[0]), training_labels, parents[0])))

test_chromosome = choose_parents(chromosomes, training_features, training_labels, num_parents)[0]
print('Test NRMSE: {}'.format(nrmse(predict(test_features, test_chromosome), test_labels)))

plt.plot(predict(test_features, test_chromosome))
plt.plot(test_labels)
plt.show()
