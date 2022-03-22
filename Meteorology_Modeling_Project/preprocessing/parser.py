data_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/raw_data.txt"
sizes_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/raw_sizes.txt"
data_out = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"

# data_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/test_data.txt"
# sizes_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/test_sizes.txt"

with open(data_in, "r") as f1:
    with open(sizes_in, "r") as f2:
        with open(data_out, "w") as f3:
            pos = 2
            line = f1.readline().strip().rstrip("\n")
            entry_lst = [line]

            while line:                    
                line = f1.readline().strip().rstrip("\n")
                entry_lst.append(line)
                pos += 1

                if pos == 54:
                    pos = 1
                    entry_lst.append(f2.readline().strip().rstrip("\n"))
                    f3.write(",".join(entry_lst) + "\n")
                    print(f"That entry was {len(entry_lst) + 1} long.")
                    entry_lst = []

                
