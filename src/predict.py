import json, joblib, pandas as pd
from density import classify

tree = joblib.load("models/tree.joblib")
counts = json.load(open("data/counts.json"))     # output of Step 1

for lane in counts:
    row = pd.DataFrame([{ "count": lane["total"], "density": classify(lane["total"]),
                          "direction": lane["direction"] }])
    row = pd.get_dummies(row).reindex(columns=tree.feature_names_in_, fill_value=0)
    lane["green_time"] = int(tree.predict(row)[0])
    print(f'{lane["direction"].upper()}: {lane["total"]} vehicles ({classify(lane["total"])}) → {lane["green_time"]} s')