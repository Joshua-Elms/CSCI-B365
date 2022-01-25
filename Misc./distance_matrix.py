import numpy as np


def get_minkowski_distance(p1: tuple, p2: tuple, r: int) -> float:
    """
    Generates Minkowski Distance between two points, r is... dimensionality?
    """
    x1, y1 = p1
    x2, y2 = p2

    if isinstance(r, str):
        if r.lower() == "inf" or r.lower() == "infinity":
            r = 100

    return (abs(x2 - x1)**r + abs(y2 - y1)**r)**(1/r)


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
    r = 1
    distance_matrix = create_minkowski_matrix(tpl1, tpl2, r)

    print(distance_matrix)

