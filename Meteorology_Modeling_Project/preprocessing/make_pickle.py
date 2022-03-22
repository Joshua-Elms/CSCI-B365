import pickle
import pandas as pd

data_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
data_out = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.pkl"

df = pd.read_csv(data_in)

# write out to pickle
with open(data_out, "wb") as f:
    pickle.dump(df, f)

# # read pickle, just for testing to ensure it works
# with open(data_out, "rb") as f:
#     df = pickle.load(f)

# print(df)