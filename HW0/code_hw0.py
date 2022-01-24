import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


if __name__=="__main__":
    # path1 = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/teach.txt"
    # df0 = q1_1(path1)
    # ratio_vals = q1_2(df0)
    # filtered_df = q1_3(df0)
    # q1_4(df0)

    path2 = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/mydata.txt"
    df1 = q1_1(path2)
    # V2_attrs = q2_2(df1)
    # print(f"Mean: {V2_attrs[0]} \nMedian: {V2_attrs[1]}")
    # V1_attrs = q2_2(df1)
    # print(f"Variance: {V1_attrs[0]} \nStandard Deviation: {V1_attrs[1]}")
    q2_3(df1)

