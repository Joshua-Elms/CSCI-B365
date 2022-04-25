import numpy as np

def L2_squared(points1, points2):
    diff = points1[:, np.newaxis] - points2
    sum_squared_diffs = np.sum(np.square(diff), axis = -1)

    return sum_squared_diffs

def partitions(length, k):
    return [(length // k) + 1 if _ < length % k else (length // k) for _ in range(k)]

# arr1 = np.array([[0,0], [1,1], [2,2], [3,3]])
# arr2 = np.array([[0,0], [4, 4], [0, 2]])

# step1 = L2_squared(arr2, arr1)
# print(step1)

for len in range(30, 34):
    for k in range(2, 5):
        part = partitions(len, k)
        print(f"Sum: {sum(part)} \n {part} \n\n")