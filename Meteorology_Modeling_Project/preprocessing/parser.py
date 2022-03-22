"""
This parser is intended to take in both the serial data containing parameters for hailstorm events and
serial data containing the actual size of hail in hailstorms, then combine the two into a csv w/ hail size as the last column

This is written out to the file in "data_out"

This uses a lazy-load method of not reading or writing entire files at once, so it is not liable to fail with massive files.
"""

data_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/raw_data.txt"
sizes_in = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/raw_sizes.txt"
data_out = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/pretty_data.csv"
col_names_path = "/Users/joshuaelms/Desktop/github_repos/CSCI-B365/Meteorology_Modeling_Project/data/53parameters_matlab.txt"

with open(data_in, "r") as f1:
    with open(sizes_in, "r") as f2:
        with open(data_out, "w") as f3:
            with open(col_names_path, "r") as f4:
                ### Write Field Names ###
                col_names = [line.strip().rstrip("\n") for line in f4.readlines()]
                col_names.append("Hailstone Size")
                f3.write(",".join(col_names) + "\n")

                ### Read every 53 items, then read size and write all 54 into a output file ###
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
                        # print(f"That entry was {len(entry_lst) + 1} long.")
                        entry_lst = []