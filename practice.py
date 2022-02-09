import matplotlib.pyplot as plt
# import pandas as pd
# import sys
# sys.path.insert(0, "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/Miscellaneous")
import homemade_stats as hstat

# t_points = [(0,0), (2,2), (4,4)]
# df = pd.DataFrame(t_points, columns=("X", "Y"))
# print(df.describe())
# fig, ax = plt.subplots()
# ax.plot(t_points, "ro", linewidth=0)

# print(hstat.std(df["X"]))
# print(hstat.std(df["Y"]))
# plt.show()
def get_column_major(arr): 
    cm_arr = []
    for c in range(len(arr[0])):
        new_col = []
        for row in arr:
            new_col.append(row[c])
        cm_arr.append(new_col)

    return cm_arr


def multiply_matrices(arr1, arr2): 
    arr_out = []
    cm_arr2 = get_column_major(arr2)
    for r, row_arr1 in enumerate(arr1): 
        new_row = []
        for col_arr2 in cm_arr2:
            dp = 0
            for i in range(len(col_arr2)):
                dp += row_arr1[i]*col_arr2[i]
            new_row.append(dp)
        arr_out.append(new_row)

    return arr_out

# for n in range(9):
#     if n == 10:
#         print("- "*10)
#     print(q5(n))

arr1 = [[2, -1], [4, 1], [5, -3]]
arr2 = [[3, 1], [-2, -1]]

for row in multiply_matrices(arr1, arr2):
    print(row)

# for row in get_column_major(arr2):
#     print(row)


