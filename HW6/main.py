from copy import deepcopy
from unicodedata import numeric
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def L2_squared_matrix(points1, points2):
    diff = points1[:, np.newaxis] - points2
    sum_squared_diffs = np.sum(np.square(diff), axis = -1)
    return sum_squared_diffs

def L1_squared_matrix(points1, points2):
    diff = points1[:, np.newaxis] - points2
    sum_squared_diffs = np.sum(np.abs(diff), axis = -1)
    return sum_squared_diffs

def knn_main(train: np.ndarray, test: np.ndarray, k: int, distance: str = "euclidean", voting: "str" = "weighted") -> np.ndarray:
    """
    Vectorized implementation of K-Nearest Neighbor 

    All distance metrics are squared / left squared; inequality is maintained and it reduces computational complexity, as sqrt is expensive.

    Args:
        train:           training data, target variable included
        test:            test data w/out target variable
        k:               int number of neighbors to consider
        distance:        str of desired distance metric ("euclidean" or "manhattan" currently)
        voting:          str of desired voting method ("weighted" or "majority" currently)

    Returns: Class labels if target variable categorical, or numeric predictions if regression data
    """
    # X indicates predictor variables, y is for target variable
    X_train, y_train = train[..., :-1], train[..., -1]
    X_test = test
    predictor_dims = X_train.shape[0]
    
    num_test_points = X_test.shape[0]

    if distance == "euclidean":
        distance_matrix = L2_squared_matrix(X_train, X_test)

    elif distance == "manhattan":
        distance_matrix = L1_squared_matrix(X_train, X_test)

    sorted_distance_matrix = np.argsort(distance_matrix, axis=0)[:k]
    rows = sorted_distance_matrix.transpose().flatten()
    cols = np.repeat(np.arange(start = 0, stop = predictor_dims), k)
    k_nearest_neighbor_distances = np.reshape(distance_matrix[rows, cols], newshape = (-1, k))

    
    if voting == "weighted":
        weights = 1 / (k_nearest_neighbor_distances)
        # If test point and train point share position, distance is 0; weight for that neighbor is 1, all other neighbors 0
        # Code from scikit-learn: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_base.py
        inf_mask = np.isinf(weights)
        inf_row = np.any(inf_mask, axis=1)
        weights[inf_row] = inf_mask[inf_row]

    elif voting == "majority":
        weights = np.ones_like(k_nearest_neighbor_distances)

    unique_labels, label_positions = np.unique(y_train, return_inverse = True)
    num_labels = unique_labels.shape[0]
    votes = np.zeros(shape = (num_test_points, k, num_labels))
    x, y = np.indices(weights.shape)
    x, y = x.flatten(), y.flatten()
    votes[x, y, label_positions[rows]] = weights[x, y]
    summed_votes = np.sum(votes, axis = 1)
    predicted_label_indices = np.argmax(summed_votes, axis =1)
    predicted = unique_labels[predicted_label_indices]

    return predicted

def k_fold(train, algorithm, params, k, rng):
    length = train.shape[0]
    indices = np.arange(length)
    rng.shuffle(indices)

    tmp_length = deepcopy(length)
    partitions = [(length // k) + 1 if _ < length % k else (length // k) for _ in range(k)]
    pass

if __name__ == "__main__":
    rng = np.random.default_rng()
    # path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/HW6/car.data"
    # df = pd.read_csv(path, header = None, names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "acceptability"])
    # print(df)
    # encoding_dict = {
    #     "buying": {"vhigh": 0, "high": 1, "med": 2, "low": 3},
    #     "maint": {"vhigh": 0, "high": 1, "med": 2, "low": 3}, 
    #     "doors": {"2": 0, "3": 1, "4": 2, "5more": 3}, 
    #     "persons": {"2": 0, "4": 1, "more": 2}, 
    #     "lug_boot": {"small": 0, "med": 1, "big": 2}, 
    #     "safety": {"low": 0, "med": 1, "high": 2},
    #     "acceptability": {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    # }
    # numeric_df = df.copy()
    # numeric_df = numeric_df.replace(encoding_dict, value = None)


    # print(numeric_df)
    arr_train = np.array([[0, 0, 1], [15, 15, 2], [10, 10, 3]])
    arr_test = np.array([[16, 16], [9, 9], [0, 0]])
    k = 2
   
    print(k_fold(arr_train, knn_main, params = {"stuff": "other"}, k = 3, rng = rng))
