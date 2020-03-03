import metrics
import numpy as np

from regularization import lasso
from parameters import genetic as parameters


def choose_parents(chromosomes, features, labels, num_parents, regularization_strength):
    fitnesses = []

    for chromosome in chromosomes:
        predicted_labels = predict(features, chromosome)
        fitnesses.append({'chromosome': chromosome, 'fitness': fitness(predicted_labels, labels, chromosome, regularization_strength)})

    fitnesses = sorted(fitnesses, key=lambda k: k['fitness'], reverse=True)

    parents = fitnesses[:num_parents]

    return np.array([parent['chromosome'] for parent in parents])


def predict(features, genes):
    predicted_labels = []

    for x in features:
        predicted_labels.append(sum([x_i * genes_i for x_i, genes_i in zip(x, genes)]) + genes[-1])

    return predicted_labels


def fitness(predicted_labels, actual_labels, chromosome, regularization_strength):
    return -(metrics.nrmse(predicted_labels, actual_labels) + lasso(regularization_strength, chromosome))

