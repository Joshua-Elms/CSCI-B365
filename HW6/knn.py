import numpy as np

def L2_squared_matrix(points1, points2, dims):
    # calculate the difference between points at each dimensions, add axis into one set of points to allow operation
    diff = points2 - points1[:, np.newaxis, :]

    # Calculate L2 norm for each group of points
    norms = sum([diff[..., i] ** 2 for i in range(dims)])
    return norms

def L1_squared_matrix(points1, points2, dims):
    # calculate the difference between points at each dimensions, add axis into one set of points to allow operation
    diff = points2 - points1[:, np.newaxis, :]

    # Calculate L1 norm for each group of points
    norms = np.square(sum([abs(diff[..., i]) for i in range(dims)]))
    return norms

def knn_main(train: np.ndarray, test: np.ndarray, k: int, classification: bool, distance: str = "euclidean", voting: "str" = "weighted") -> np.ndarray:
    """
    Vectorized implementation of K-Nearest Neighbor 

    All distance metrics are squared / left squared; inequality is maintained and it reduces computational complexity, as sqrt is expensive.

    Args:
        train:           training data, target variable included
        test:            test data w/out target variable
        k:               int number of neighbors to consider
        classification:  boolean, False indicates a regression problem. Class label must be cast to int even if qualitative.
        distance:        str of desired distance metric ("euclidean" or "manhattan" currently)
        voting:          str of desired voting method ("weighted" or "majority" currently)

    Returns: Class labels if target variable categorical, or numeric predictions if regression data
    """
    # X indicates predictor variables, y is for target variable
    X_train, y_train = train[..., :-1], train[..., 1]
    X_test = test
    predictor_dims = X_train.shape[0]
    unique_labels, label_positions = np.unique(y_train, return_inverse = True)
    num_labels = unique_labels.shape[0]
    num_test_points = X_test.shape[0]


    if distance == "euclidean":
        distance_matrix = L2_squared_matrix(X_train, X_test, predictor_dims)

    elif distance == "manhattan":
        distance_matrix = L1_squared_matrix(X_train, X_test, predictor_dims)

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

    
    if classification:
        votes = np.zeroes(shape = (num_test_points, num_labels))
        predicted_label_indices = np.argmax(votes, axis =1)
    

if __name__ == "__main__":
    k = 2
    labs = np.array([[1], [2], [3]])
    unique_labels, label_positions = np.unique(labs, return_inverse = True)
    num_labels = unique_labels.shape[0]
    arr_train = np.array([[0, 0], [15, 15], [10, 10]])
    arr_test = np.array([[16, 16], [9, 9], [0, 0]])
    distances = L1_squared_matrix(arr_train, arr_test, 2)
    print(distances)
    sorted = np.argsort(distances, axis=0)[:k]
    print(sorted)
    # print(sorted.transpose().flatten())
    rows = sorted.transpose().flatten()
    cols = np.repeat(np.arange(start = 0, stop = 3), k)
    weights = 1 / (np.reshape(distances[rows, cols], newshape = (-1, k)))
    # print(weights)
    inf_mask = np.isinf(weights)
    inf_row = np.any(inf_mask, axis=1)
    weights[inf_row] = inf_mask[inf_row]
    print(np.round(weights, 4))
    votes = np.zeros(shape = (3, k, num_labels))
    k_lab_positions = label_positions[sorted]
    votes = weights[np.arange(3), np.indices(weights.shape)[-1].flatten(), ]
    print()
#    points1 = np.array([[1, 2, 3], [4, 5, 6]])
#    points2 = np.array([[1, 2, 3], [4, 5, 6]])
#    print(points2 - points1[:, np.newaxis, :])
#    print(L2_squared_matrix(points1, points2, 3))
