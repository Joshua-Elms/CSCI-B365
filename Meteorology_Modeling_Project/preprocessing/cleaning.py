import pandas as pd
from sklearn.impute import KNNImputer


path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
df = pd.read_csv(path)
df_knn = KNNImputer().fit_transform(df)
df_knn_actual = pd.DataFrame(df_knn)
df_knn_actual.columns = df.columns

write_path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
df_knn_actual.to_csv(write_path)