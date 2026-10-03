import numpy as np


def information_gain(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
) -> float:
    """
    parent_labels: labels of every sample in the node before splitting.
    left_labels, right_labels: parent_labels partitioned by the
    candidate split, every sample in exactly one of the two.

    Returns:
        how much the split reduces Gini impurity, a float.
    """
    # TODO: gain = gini_impurity(parent_labels) minus the size-weighted
    # average of gini_impurity(left_labels) and gini_impurity(right_labels).
    parent_gini = calcgini(parent_labels)
    right_gini = calcgini(right_labels)
    left_gini = calcgini(left_labels)
    w_r = (len(right_labels)/len(parent_labels)) * right_gini
    w_l = (len(left_labels)/len(parent_labels)) * left_gini
    result = parent_gini - (w_r + w_l)
    return result


def calcgini(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return 1 - np.sum(probs ** 2)
