import pandas as pd
from itertools import combinations, chain
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from numpy import corrcoef
from random import shuffle

path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
df = pd.read_csv(path)

def power_set(iterable):
    pset = chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))
    return list(list(combo) for combo in pset if len(combo) > 0)

indices = [i for i in range(21, 37)] # + [i for i in range(41, 50)]

field_pset = power_set([df.columns[i] for i in indices])
# shuffle(field_pset)

print(field_pset)

df_knn = KNNImputer().fit_transform(df)
df_knn_actual = pd.DataFrame(df_knn)
df_knn_actual.columns = df.columns

with open("/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/log_file.txt", "w") as f:
    max = 0
    for combination in field_pset:
        arr = df_knn_actual[combination].to_numpy()
        target = df_knn_actual["Hailstone Size"].to_numpy()
        obj = LinearRegression().fit(X=arr, y=target)
        coefficients = obj.coef_
        linear_combination = (df_knn_actual[combination]*coefficients).sum(axis=1)
        correlation = corrcoef(linear_combination, target)

        corr = correlation[0][1]

        if corr > max:
            max = corr
            variable_str = f'{", ".join(combination[:-1])}, and {combination[-1]}' if len(combination) > 2 else f'{combination[0]} and {combination[-1]}' if len(combination) == 2 else combination[0]
            f.write(f"Variable{'s' if len(combination) > 1 else ''} {variable_str} yield{'' if len(combination) > 1 else 's'} a correlation of {corr}\n")