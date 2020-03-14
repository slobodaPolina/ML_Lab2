# расчитанные предпочтительные параметры модели
gradient_descent = {
    'num_iterations': 50,
    'regularization_strength': 0.001,
    'learning_rate': 0.5,
    'cross_val_parameter': 50
}

# данные для расчета параметров модели gradient_descent
gradient_descent_calculation = {
    'learning_rate': [0.001, 0.01, 0.5, 1, 5],
    'regularization_strength': [0.001, 0.01, 0.5, 1, 5, 10, 30],
    'num_iterations': 10,
    'cross_val_parameter': [5, 10, 20, 50]
}

genetic = {
    'size_population': 50,  # сколько в популяции особей (индивидуумов)
    'parent_proportion': 0.9,  # какая доля из оных сохраняется, оставшаяся будет составлена из потомков выживших
    'num_generations': 100,
    'random_gene_amplitude': 1000,  # параметр, отвечающий за разброс начальных значений у хромосом (то есть весов)
    'mutation_amplitude': 1,  # насколько сильно могут мутировать хромосомы в каждом поколении (+- mutation_amplitude * value)
    'regularization_strength': 0.00001  # штраф на переобучение
}

genetic_calculation = {
    'size_population': [20, 50],
    'parent_proportion': [0.1, 0.5, 0.9],
    'num_generations': 10,  # для подбора это пока не важный параметр
    'random_gene_amplitude': [10, 100, 1000, 10000],
    'mutation_amplitude': [0.1, 0.5, 1, 5],
    'regularization_strength': [0.00001, 0.001, 0.1]
}
