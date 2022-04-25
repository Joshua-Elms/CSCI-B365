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

def knn_main(train: np.ndarray, test: np.ndarray, k: int, distance: str = "euclidean", voting: "str" = "weighted") -> np.ndarray:
    """
    Vectorized implementation of K-Nearest Neighbor 

    All distance metrics are squared / left squared; inequality is maintained and it reduces computational complexity, as sqrt is expensive.

    Args:
        train:     training data, target variable included
        test:      test data w/out target variable
        k:         int number of neighbors to consider
        distance:  str of desired distance metric ("euclidean" or "manhattan" currently)
        voting:    str of desired voting method ("weighted" or "majority" currently)

    Returns: Class labels if target variable categorical, or numeric predictions if regression data
    """
    # X indicates predictor variables, y is for target variable
    X_train, y_train = train[..., :-1], train[..., 1]
    X_test = test
    predictor_dims = X_train.shape[0]

    if distance == "euclidean":
        distance_matrix = L2_squared_matrix(X_train, X_test, predictor_dims)

    elif distance == "manhattan"
    pass

if __name__ == "__main__":
    k = 2
    arr_train = np.array([[0, 0], [15, 15], [10, 10]])
    arr_test = np.array([[-1, -1], [9, 9]])
    distances = L1_squared_matrix(arr_train, arr_test, 2)
    print(distances)
    sorted = np.argsort(distances, axis=0)[:k]
    print(sorted)
    weights = 1 / np.reshape(distances[[0, 2, 2, 1], [0, 0, 1, 1]], newshape = (k, -1))
    print(weights)
#    points1 = np.array([[1, 2, 3], [4, 5, 6]])
#    points2 = np.array([[1, 2, 3], [4, 5, 6]])
#    print(points2 - points1[:, np.newaxis, :])
#    print(L2_squared_matrix(points1, points2, 3))
