import pickle
import pandas as pd
from time import perf_counter


"""
Pickles are significantly faster; with just one trial I can see that
pandas can read a dataframe from pickle in about 1/20th of the time for a csv

However, since I am only working with ~20MB, the csv time is only .5 seconds, so I don't care.
"""

data_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
data_out = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.pkl"

df = pd.read_csv(data_in)

### write out to pickle ###
with open(data_out, "wb") as f:
    pickle.dump(df, f)
############################


### read pickle, just for testing to ensure it works ###
# with open(data_out, "rb") as f:
#     df = pickle.load(f)
#########################################################
# print(df)



### Benchmark to compare time req's for csv vs pickle ###
# # 
# start1 = perf_counter()
# df1 = pd.read_csv(data_in)
# stop1 = perf_counter()


# start2 = perf_counter()
# with open(data_out, "rb") as f:
#     df2 = pickle.load(f)
# stop2 = perf_counter()

# t1, t2 = stop1 - start1, stop2 - start2

# print(f"Time to read from csv: {t1} seconds")
# print(f"Time to read from pickle: {t2} seconds")
###################################################