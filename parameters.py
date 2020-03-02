filename = './data/5.txt'

least_squares = {
    'iterations': 10,
    'regularization_strength': 100.0
}

gradient_descent = {
    'num_iterations': 10,
    'regularization_strength': 100.0,
    'learning_rate': 0.1
}

genetic = {
    'num_chromosomes': 16,
    'parent_proportion': 1/2,
    'num_generations': 10,
    'random_gene_amplitude': 1000,
    'mutation_amplitude': 10000
}

cross_val_folds = 4