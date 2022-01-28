import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from statistics import variance

def q1_1(teach_path: str) -> pd.DataFrame:
    """
    Read in teach.txt as a dataframe
    Be sure to provide a functional path to teach.txt for your environment
    """
    return pd.read_csv(teach_path, sep= ", " if teach_path == "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/teach.txt" else ",", header=0, index_col=0, engine="python")


def q1_2(df0: pd.DataFrame) -> list:
    """
    Selects only the "Ratio" column from the Dataframe; this maintains the "Year" labels
    To remove "Year" labels, the Series created from the operation is cast to a list
    """
    return list(df0["Ratio"])


def q1_3(df0: pd.DataFrame) -> pd.DataFrame:
    """
    Selects rows that have "Ratio" values 18 < val < 22
    """
    return df0[(df0["Ratio"] > 18) & (df0["Ratio"] < 22)]


def q1_4(df0: pd.DataFrame):
    """
    Plots and displays a histogram of "Ratio" in df
    Uses pandas library
    """
    plt.style.use('seaborn-pastel')
    kwargs = {"bins": tuple(range(0, 121, 20))}
    ax = df0["Ratio"].hist(grid=False, **kwargs)
    ax.set_title("Histogram of teach$Ratio")
    ax.set_xlabel("teach$Ratio")
    ax.set_ylabel("Frequency")
    plt.show()
    pass


def q2_1(df: pd.DataFrame) -> int: 
    """
    Finds # of entries in df
    """
    return df.size[0]


def q2_2(df: pd.DataFrame) -> tuple:
    """
    Calculates mean and median for V2 of df, returns tuple of (mean, median)
    """
    V2 = df["V2"]
    return V2.mean(), V2.median()


def q2_2(df: pd.DataFrame) -> tuple:
    """
    Calculates variance and std dev for V1 of df, returns tuple of (variance, std dev)
    """
    V1 = df["V1"]
    return V1.var(), V1.std()


def q2_3(df: pd.DataFrame):
    """
    Plots and displays a histogram of "V5" in df
    Uses matplotlib.pyplot exclusively for graphing
    """
    V5 = df["V5"]

    plt.style.use("seaborn-bright")
    fig, ax = plt.subplots()
    fig.set_size_inches(5.5,5)
    ax.hist(V5, bins=(0.5, 1.5, 2.5), rwidth=.95)
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    ax.set_ylim(0, 1600)
    ax.set_xticks(np.linspace(0.5, 2.5, 5))
    ax.set_yticks(np.linspace(0, 1500, 4))
    ax.grid(which="major", axis="both", **{"color": "xkcd:grey", "lw": .5})
    ax.set_axisbelow(True)
    plt.show()
    pass


def dist_Euclidean(points: tuple|list) -> tuple: 
    """
    Uses Euclidean distance formula to determine which to points on a list 
    (ex. [(1,1), (2, 2), (4, 4)]) are least dissimilar.

    Args: 
        points: Any iterable containing at least 2 objects which can be interpreted 
        as points, values can be int|float

    Returns:
        results: 2 points from input that are closest via Euclidean distance and 
        their distance, ex. ((1,1), (2,2), 1.414)
    """
    ### Inline def of distance equation ###
    euc_dist_eq = lambda x1, y1, x2, y2: sqrt(abs(x2 - x1)**2 + abs(y2 - y1)**2)

    ### Create minkowski matrix ###
    dist_list = [[] for i in range(len(points))]
    for i, p1 in enumerate(points): # will append dist to row and column of list
        for p2 in points:
            dist_list[i].append(euc_dist_eq(p1[0], p1[1], p2[0], p2[1]))

    ### Pretty Print Matrix ###
    # dist_matrix = np.asarray(dist_list)
    # print(np.array_str(dist_matrix, precision=2, suppress_small=True))
    
    ### Search for smallest distance, get location and value ###
    length = len(points)
    min = dist_list[0][1]
    loc = (0,1)
    for i in range(length-1):
        for j in range(i+1, length):
            if dist_list[i][j] < min:
                min = dist_list[i][j]
                loc = (i,j)

    return (points[i], points[j], min)


def sample_mean(sample: list|tuple) -> float: 
    """
    Calculates the arithmetic mean of given sample
    """
    return sum(sample)/len(sample)


def sample_variance(sample: list|tuple) -> float:
    """
    Calls sample_mean() to calculate the sample variance of a given sample
    """
    s_mean = sample_mean(sample)
    n = len(sample)
    variance = sum(((i - s_mean)**2 for i in sample))/(n-1)

    return variance


def main():
    ### Problem 1 ###
    # path1 = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/teach.txt"
    # df0 = q1_1(path1)
    # ratio_vals = q1_2(df0)
    # filtered_df = q1_3(df0)
    # q1_4(df0)
    ##################

    ### Problem 2 ###
    # path2 = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/mydata.txt"
    # df1 = q1_1(path2)
    # V2_attrs = q2_2(df1)
    # print(f"Mean: {V2_attrs[0]} \nMedian: {V2_attrs[1]}")
    # V1_attrs = q2_2(df1)
    # print(f"Variance: {V1_attrs[0]} \nStandard Deviation: {V1_attrs[1]}")
    # q2_3(df1)
    ##################

    ## Problem 3 ###
    # lst1 = [(1,2), (3,4), (6,4)]
    # lst2 = [(1,1), (2,5), (3,3)]
    # lst3 = [(5,5), (10,10), (20,20)]
    # print(dist_Euclidean(lst1))
    ################

    ### Problem 4 ###
    # sample1 = (15, 2, 44, 21, 40, 20, 19, 18)
    # mean = sample_mean(sample1)
    # var = sample_variance(sample1)
    # print(f"Sample: {sample1}")
    # print(f"Mean: {mean}")
    # print(f"Sample Variance: {var}")
    #################
    pass


if __name__=="__main__":
    main()

