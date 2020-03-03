filename = './data/5.txt'

least_squares = {
    'iterations': 10
}

gradient_descent = {
    'num_iterations': 10,
    'regularization_strength': 0.001,
    'learning_rate': 0.1
}

genetic = {
    'num_chromosomes': 20,
    'parent_proportion': 1/2,
    'num_generations': 10,
    'random_gene_amplitude': 1000,
    'mutation_amplitude': 0.5,
    'regularization_strength': 0.000001
    #'mutation_amplitude': [0.5, 1, 2, 3],
    #'regularization_strength': [0.0000001, 0.000001, 0.00001]
}

cross_val_folds = 4

num_iterations = 15