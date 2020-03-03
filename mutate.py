import random
import numpy as np


def mutate(children, mutation_amplitude):
    children = np.array([[gene + (random.random() - 0.5) * 2 * (gene * mutation_amplitude) for gene in child] for child in children])

    return children
