import numpy as np


def cross_val_split (data, k):
    np.random.shuffle(data)

    features = data[:, :-1]
    labels = data[:, -1]

    split_point = len(labels) // k

    training_features, cross_val_features = np.array(features[split_point:]), np.array(features[:split_point])
    training_labels, cross_val_labels = np.array(labels[split_point:]), np.array(labels[:split_point])

    return training_features, cross_val_features, training_labels, cross_val_labels
