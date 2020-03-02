from sklearn import preprocessing
import numpy as np


def read_from_file(filename):
    file = open(filename)

    feature_count = int(file.readline())
    training_object_count = int(file.readline())

    training_features, training_labels = [], []

    for i in range(training_object_count):
        object_data = file.readline().split()
        training_features.append([int(value) for value in object_data[:-1]])
        training_labels.append(int(object_data[-1]))

    test_object_count = int(file.readline())

    test_features, test_labels = [], []

    for i in range(test_object_count):
        object_data = file.readline().split()
        test_features.append([int(value) for value in object_data[:-1]])
        test_labels.append(int(object_data[-1]))

    return training_object_count, feature_count, training_features, training_labels, test_features, test_labels


def prepare_data(features, labels):

    features = normalize(features)

    return features, labels


def normalize(features):
    return preprocessing.normalize(features, axis=0)