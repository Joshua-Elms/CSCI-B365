import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def set_params(k=2, half_of_points=20, dims=2, means=(5, 15), stdevs=(1, 1)): 
    """
    Determine the parameters of the algorithm

    Args:
        None

    Returns: 
        k: int, determines # of centroids that will be chosen (final # of clusters)
    """
    # k = input("# of centroids:")
    return k, half_of_points, dims, means, stdevs


def generate_data(k, half_of_points, dims, distr_means, stdevs):
    """
    Uses random generator from NumPy to create two clusters for us to play with
    Based on the min and max of each cluster, determine the initial conditions for our centroids

    Args:
        k: num of centroids/clusters to be found
        half_of_points: Multiply this by 2 to find the number of clusterable data points from which we will make a data frame, 1/2 for each cluster
        dims: How many dimensions the data will span (each will be a column in final df); separate from clusters, as this will only generate 3 clusters
        distr_means: Where the clusters will be centered around (mean for normal distr.)
        stdevs: Measure of spread for each cluster (stdev for normal distr.)

    Returns:
        init_data_matrix: nd.array containing our points for clustering
        means: initial values for each of the k centroids, "dim" dimensional random values from ranges of init_data_matrix
    """
    rng = np.random.default_rng()
    clust1 = rng.normal(loc=distr_means[0], scale=stdevs[0], size=(half_of_points, dims))
    clust2 = rng.normal(loc=distr_means[1], scale=stdevs[1], size=(half_of_points, dims))

    data = np.concatenate((clust1, clust2)) # create nd.array from data

    columns = [data[:, i] for i in range(dims)] # get list of all columns

    dim_ranges = [(column.min(), column.max()) for column in columns] # get min and max of each column into tuple

    means = [rng.uniform(*dim_range, size=dims).tolist() for dim_range in dim_ranges] # create dim sized, randomly selected points to represent centroids

    return data, means


def get_column_major(arr): 
    cm_arr = []
    for c in range(len(arr[0])):
        new_col = []
        for row in arr:
            new_col.append(row[c])
        cm_arr.append(new_col)

    return cm_arr


def build_df(matrix, rm_means, dims, k_num): 
    df1 = pd.DataFrame(matrix)
    col_names = (f"Dim_{i+1}" for i in range(dims))
    df1.columns = col_names # set names for all of our points dimensions
    df1["Type"] = "data" # Type will be either data or centroid
    df1["Cluster"] = "None" # Cluster will be one of the centroids (k1, k2, ...)
    cm_means = get_column_major(rm_means)
#, "Y":k_y, "Type":"centroid"
    k_dict = {col_names[i]:cm_means[i] for i in range(dims)}
    k_dict["Cluster"] =  [f"C{i+1}" for i in range(k_num)]


    df_k = pd.DataFrame(k_dict)
    df = pd.concat([df1, df_k], ignore_index=True)

    return 
    
    


### Find column ranges, choose random centroids from ranges

# df = df.append(k_dict, ignore_index = True)

# ###  Graph data
# sns.set_theme()

# sns.relplot(data=df, x="X", y="Y", hue="Cluster", style="Type")
# plt.show()


def test_main():
    k_num, half_of_points, dims, distr_means, stdevs = set_params()
    init_matrix, init_means = generate_data(k_num, half_of_points, dims, distr_means, stdevs)
    df = build_df(init_matrix, init_means, dims)
    # print(init_matrix)
    # print(init_means)

    pass


def main():
    k_num = set_params()
    init_matrix, init_means = generate_data()


    pass

if __name__ == "__main__": 
    test_main()
