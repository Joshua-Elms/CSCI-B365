import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def data():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin//breast-cancer-wisconsin.data",
 names=[f"V{i}" for i in range(1,12)])
    df.index += 1
    return df

def cov(x: list|tuple, y: list|tuple, r: int = 9) -> float: 
    """
    Calculate sample covariance matrix for vectors x and y (must be same length)
    r is the desired numbers of digits after decimal to round to
    """
    try: 
        N = max(len(x), len(y))
        x_mean = sum(x) / N
        y_mean = sum(y) / N
        cov = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(N)]) / (N - 1)
        return round(cov, r)

    except IndexError: 
        raise ValueError("Vectors must be of same length")

    except TypeError:
        raise TypeError("Vectors must only contain ints or floats")


def corr(x: list|tuple, y: list|tuple, r: int = 9) -> float:
    """
    Calculates the correlation between vectors x and y (must be same length)
    Calls cov(), std()
    Output float bounded by [-1, 1] inclusive
    """
    return round(cov(x, y) / (std(x) * std(y)), r)

def var(x: list|tuple, r: int = 9) -> float:
    """
    Calculate population variance matrix for vector x
    Calls cov()
    r is the desired numbers of digits after decimal to round to
    """
    return cov(x, x, r)


def std(x: list|tuple, r: int = 9) -> float: 
    """
    Calculate standard deviation of x
    Calls var()
    r is the desired numbers of digits after decimal to round to
    """
    return round(sqrt(var(x)), r)


if __name__ == "__main__": 
    df = data()
    # q1 = df["V11"][df.V11 == 4].count()
    # print(q1)

    # print(df.isnull().count())

    ### histograms ###
    # print(df.describe())
    # df.rename(columns={"V2": "Clump Thickness", "V3": "Uniformity of Cell Shape", "V4": "Uniformity of Cell Shape",
    # "V5": "Marginal Adhesion", "V6": "Single Epithelial Cell Size", "V7": "Bare Nuclei", "V8": "Bland Chromatin", "V9": "Normed Nuclei", 
    # "V10": "Mitoses", "V11": "Class (2 benign, 4 malignant)"}, inplace=True)
    # print(df.head())
    # df.hist(df.columns[1:]) # start at second column, first is ID field
    # plt.show()

    xlst, ylst = [1, 2, 3, 4], [10, 20, 30, 40]
    r = 1
    cov1, cov2, cov3 = corr(xlst, xlst, r), corr(xlst, ylst, r), corr(ylst, ylst, r)
    print(f"[[{cov1}, {cov2}], \n[{cov2}, {cov3}]]")

    # print(np.cov([xlst, ylst]))
