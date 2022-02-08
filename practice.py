import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, "/Users/joshuaelms/Desktop/github_repos/CSCI-B365_repo/CSCI-B365/Miscellaneous")
import homemade_stats as hstat

t_points = [(0,0), (2,2), (4,4)]
df = pd.DataFrame(t_points, columns=("X", "Y"))
# print(df.describe())
fig, ax = plt.subplots()
ax.plot(t_points, "ro", linewidth=0)

print(hstat.std(df["X"]))
print(hstat.std(df["Y"]))
# plt.show()

