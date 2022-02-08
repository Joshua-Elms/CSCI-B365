import numpy as np
from sklearn.metrics import pairwise_distances


def get_minkowski_distance(p1: tuple, p2: tuple, r: int = 2) -> float:
    """
    Generates Minkowski Distance between two points, r is... dimensionality?
    """
    tmp = 0 # all values will be added to this variable, will be returned
    cols = len(p1) 

    if isinstance(r, str):
        if r.lower() == "inf" or r.lower() == "infinity":
            tmp_lst = []
            for i in range(cols):
                tmp_lst.append(abs(p1[i] - p2[i]))

            return max(tmp_lst)

    else: 
        for i in range(cols):
            tmp += (abs(p1[i] - p2[i])**r)

        return tmp**(1/r)


def create_minkowski_matrix(arr: np.ndarray, r: int|str) -> np.ndarray:
    """
    
    """
    d_arr = np.empty((len(arr), len(arr)))# [[] for i in range(len(tpl1))]

    for i, p1 in enumerate(arr):
        for j, p2 in enumerate(arr):
            d_arr[i][j] = (round(get_minkowski_distance(p1, p2, r), 8))

    return d_arr


if __name__=="__main__":
    r = "inf"
    for arr in [np.random.rand(2, 4) for i in range(4)]:
        print("-" * 50)
        print(create_minkowski_matrix(arr*10, r))
        print("\n")
        print(pairwise_distances(arr*10, metric="chebyshev"))
        print("-" * 50)
        print("\n\n")
