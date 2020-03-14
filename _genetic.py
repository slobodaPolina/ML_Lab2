import numpy as np
import matplotlib.pyplot as plt
import random
import prepare_data
from parameters import genetic as parameters
from metrics import NRMSE


# создать num_children детей, у каждого num_chromosomes_per_individual хромосом
def reproduce(parents, num_children, num_parents, num_chromosomes_per_individual):
    children = []

    for index in range(num_children):
        # выбираем для него родителей (тут берется 2 соседа в зацикленном массиве)
        parent1_index = index % num_parents
        parent2_index = (index + 1) % num_parents

        # создаем ребенка, первую половину хромосом от 1 родителя, вторую от второго
        child = np.empty(num_chromosomes_per_individual)
        crossover_point = round(num_chromosomes_per_individual / 2)
        child[0:crossover_point] = parents[parent1_index, 0:crossover_point]
        child[crossover_point:] = parents[parent2_index, crossover_point:]

        children.append(child)

    return np.array(children)


# отобрать num_parents лучших комбинаций хромосом
def select_parents(chromosomes, features, labels, num_parents, regularization_strength):
    fitnessed = []
    # проверим каждого индивидуума (то есть каждый набор весов), посчитаем для него фитнес- функцию (функцию, по котороый мы вектора оцениваем)
    for chromosome in chromosomes:
        predicted_labels = predict(features, chromosome)
        fitnessed.append({
            'chromosome': chromosome,
            # функция ошибки и штраф лассо
            'fitness': -(NRMSE(predicted_labels, labels) + regularization_strength * sum([abs(value) for value in chromosome]))
        })

    # выбираем несколько лучших векторов весов (наборов хромосом)
    parents = sorted(fitnessed, key=lambda k: k['fitness'], reverse=True)[:num_parents]
    return np.array([parent['chromosome'] for parent in parents])


def predict(features, chromosomes):
    predicted_labels = []
    for feature in features:
        predicted_labels.append(sum([parameter * chromosome for parameter, chromosome in zip(feature, chromosomes)]) + chromosomes[-1])
    return predicted_labels


# мутируем детей (вносим в их хромосомы мутации)
def mutate(children, mutation_amplitude):
    return np.array([[
        # каждая хромосома может изменить свое значение на +-(chromosome * mutation_amplitude)
        chromosome + random.uniform(-1, 1) * (chromosome * mutation_amplitude)
        for chromosome in child] for child in children])


def genetic(parameters):
    train_losses, test_losses = [], []
    training_object_count, feature_count, training_features, training_labels, test_features, test_labels = prepare_data.read_from_file()

    # количество индивидуумов в популяции (всего)
    population_size = parameters['size_population']
    num_parents = int(population_size * parameters['parent_proportion'])
    num_children = population_size - num_parents
    num_chromosomes_per_individual = feature_count + 1

    # хромосомы - по сути, набор векторов весов, который как раз нужно подобрать. У каждого индивидуума он свой, в итоге выберем один
    chromosomes = np.random.uniform(
        low=-parameters['random_gene_amplitude'],
        high=parameters['random_gene_amplitude'],
        # сколько индивидуумов на количество хромосом у каждого (набор хромосом по 1 на каждую фичу для оценки веса ее входа)
        size=(population_size, num_chromosomes_per_individual)
    )

    for i in range(parameters['num_generations']):
        print(i + 1)
        # num_parents лучших наборов хромосом из имеющихся (выбирвем по training - сету)
        parents = select_parents(chromosomes, training_features, training_labels, num_parents, parameters['regularization_strength'])
        # создаем детей посредством скрещивания родителей, мутируем их
        children = reproduce(parents, num_children, num_parents, num_chromosomes_per_individual)
        children = mutate(children, parameters['mutation_amplitude'])
        # обновляем банк хромомсом по родителям (выжившим из предыдущего поколения) и их потомству
        chromosomes = np.concatenate((parents, children), axis=0)
        # делаем предсказания по обучающему и тестировочному множеству, оцениваем его по лучшему родителю (детей из последнего поколения в расчет уже не возьмем:))
        train_losses.append(NRMSE(predict(training_features, parents[0]), training_labels))
        test_losses.append(NRMSE(predict(test_features, parents[0]), test_labels))

    return train_losses, test_losses


max_generations = parameters['num_generations']

train_loss, test_loss = genetic({
    'mutation_amplitude': parameters['mutation_amplitude'],
    'regularization_strength': parameters['regularization_strength'],
    'size_population': parameters['size_population'],
    'num_generations': max_generations + 1,
    'parent_proportion': parameters['parent_proportion'],
    'random_gene_amplitude': parameters['random_gene_amplitude']
})

plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.xlabel('maximum generations')
plt.ylabel('NRMSE')
plt.legend()
plt.show()
