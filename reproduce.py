import numpy as np


def reproduce(parents, num_children, num_parents, num_genes):
    children = []

    for index in range(num_children):
        parent1_index = index % num_parents
        parent2_index = (index + 1) % num_parents

        child = np.empty(num_genes)
        crossover_point = round(num_genes / 2)

        child[0:crossover_point] = parents[parent1_index, 0:crossover_point]
        child[crossover_point:] = parents[parent2_index, crossover_point:]

        children.append(child)

    return np.array(children)
