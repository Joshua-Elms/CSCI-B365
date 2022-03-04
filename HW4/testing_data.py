from importlib.resources import path
import numpy as np 
import pandas as pd

path_wbcd = "breast-cancer-wisconsin.data"
path_wd = "wine.data"
path_synthetic = "synthetic.data"

wisconsin_bc = pd.read_csv(path_wbcd, header=None)
wisconsin_bc.columns = ["SCN", *[f"A{i}" for i in range(2, 11)], "Class"]
wine = pd.read_csv(path_wd)
synthetic = pd.read_csv(path_synthetic)


manual_cost = 2097000
beep_boop_cost = 215991

reduction = ( manual_cost - beep_boop_cost ) / manual_cost
print(reduction)

# print(wisconsin_bc[wisconsin_bc["Class"] == 4]["Class"].count() * .7)

# print(209700 + 6291)