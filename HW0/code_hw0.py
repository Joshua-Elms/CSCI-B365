import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def q1_1(teach_path: str) -> pd.DataFrame:
    """
    Read in teach.txt as a dataframe
    Be sure to provide a functional path to teach.txt for your environment
    """
    return pd.read_csv(teach_path, sep= ", ", header=0, index_col=0, engine="python")


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


if __name__=="__main__":
    path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/teach.txt"
    df0 = q1_1(path)
    ratio_vals = q1_2(df0)
    filtered_df = q1_3(df0)
    q1_4(df0)
