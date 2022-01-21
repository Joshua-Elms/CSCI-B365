import pandas as pd

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


if __name__=="__main__":
    path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/HW0/Resources/teach.txt"
    df0 = q1_1(path)
    ratio_vals = q1_2(df0)