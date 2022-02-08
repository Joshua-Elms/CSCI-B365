import numpy as np
from numpy.core.multiarray import asarray
from sklearn.metrics import pairwise_distances


def get_minkowski_distance(p1: tuple, p2: tuple, r: int = 2) -> float:
    """
    Generates Minkowski Distance between two points, r is... dimensionality?
    """

    if isinstance(r, str):
        if r.lower() == "inf" or r.lower() == "infinity":
            r = 100

    cols = len(p1) 
    tmp = 0
    for i in range(cols):
        tmp += (abs(p1[i] - p2[i])**r)

    return tmp**(1/r)


def create_minkowski_matrix(tpl1: tuple, tpl2: tuple, r: int|str) -> np.ndarray:
    """
    
    """
    distance_matrix = [[] for i in range(len(tpl1))]

    for i, p1 in enumerate(zip(tpl1, tpl2)):
        for p2 in zip(tpl1, tpl2):
            distance_matrix[i].append(round(get_minkowski_distance(p1, p2, r), 3))

    return np.asarray(distance_matrix)


if __name__=="__main__":
    # tpl1 = tuple(np.random.randint(0, 10, 10))
    # tpl2 = tuple(np.random.randint(0, 10, 10))
    tpl1 = (0, 2, 3, 5)
    tpl2 = (2, 0, 1, 1)
    arr1 = asarray(tpl1).reshape(1, -1)
    arr2 = asarray(tpl2).reshape(1, -1)

    zip1 = zip(tpl1, tpl2)
    r = 2
    # distance_matrix = create_minkowski_matrix(tpl1, tpl2, r)
    # print(arr1)
    print(pairwise_distances(arr1, arr2, metric="euclidean"))
    # print(distance_matrix)
    print(get_minkowski_distance(tpl1, tpl2))
