import pandas as pd
import scipy
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions, get_distributions

path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
df = pd.read_csv(path)


# Fitter to find best distribution 
hail = df["Hailstone Size"].to_numpy()
common = get_common_distributions()
good = ["gennorm", "dgamma", "dweibull", "cauchy"]
f = Fitter(hail, distributions=good)

fig, ax = plt.subplots()
scipy.stats.probplot(df["Hailstone Size"], sparams=(0.47774409138777574, 1.0, 0.47774409138777574), dist='gennorm', fit=True, plot=ax, rvalue=False)
plt.show()



# f.fit()

# print(f.summary())

# print(f.get_best(method = "sumsquare_error"))

