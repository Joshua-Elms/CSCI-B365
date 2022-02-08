import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/Miscellaneous")
import homemade_stats as hstat

def data():
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin//breast-cancer-wisconsin.data",
 names=[f"V{i}" for i in range(1,12)])
    df.index += 1
    return df



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

    # xlst, ylst = [1, 2, 3, 4], [10, 20, 30, 40]
    # r = 1
    # cov1, cov2, cov3 = hstat.corr(xlst, xlst, r), hstat.corr(xlst, ylst, r), hstat.corr(ylst, ylst, r)
    # print(f"[[{cov1}, {cov2}], \n[{cov2}, {cov3}]]")

    