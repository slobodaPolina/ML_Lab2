import random


def mutate(children, mutation_amplitude):
    for child in children:
        child[random.randint(0, len(children)-1)] += (random.random() - 0.5) * 2 * mutation_amplitude

    return children
