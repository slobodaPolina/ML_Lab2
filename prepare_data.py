def read_from_file():
    file = open('./data/5.txt')
    # число признаков
    feature_count = int(file.readline())
    # число объектов в тренировочном наборе
    training_object_count = int(file.readline())

    # данные тренировочного набора, ответы (значения y) тренировочного набора
    training_features, training_labels = [], []

    for i in range(training_object_count):
        object_data = file.readline().split()
        training_features.append([int(value) for value in object_data[:-1]])
        training_labels.append(int(object_data[-1]))

    # число объектов в тестовом наборе
    test_object_count = int(file.readline())

    # данные тестового набора, ответы (значения y) тестового набора
    test_features, test_labels = [], []

    for i in range(test_object_count):
        object_data = file.readline().split()
        test_features.append([int(value) for value in object_data[:-1]])
        test_labels.append(int(object_data[-1]))

    return training_object_count, feature_count, training_features, training_labels, test_features, test_labels
