import pandas as pd
import json

input_file = "results/bb_sim_basic_no_pc_50000_100_10000_7_cycle_d_debug_all_errors.csv"
output_file = "bb_sim_basic_no_pc_50000_100_10000_7_cycle_d_debug_all_errors_Z.csv"

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

X = []
Y = []

for _, row in df.iterrows():
    meta = json.loads(row["json_metadata"])

    if meta["basis"] == "Z":
        p = meta["p"]
        shots = row["shots"]
        errors = row["errors"]

        X.append(p)
        Y.append(errors / shots)

out = pd.DataFrame({"X": X, "Y": Y})

out = out.sort_values("X")

out.to_csv(output_file, index=False)
