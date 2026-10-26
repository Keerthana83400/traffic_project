import pandas as pd, joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from density import classify

BASE = {"LOW": 10, "MEDIUM": 20, "HIGH": 35}          # your policy: base green per density
BAND_START = {"LOW": 0, "MEDIUM": 6, "HIGH": 16}
DIR_WEIGHT = {"north": 2, "east": 0, "south": 1, "west": -1}

def policy_green(count, direction):
    d = classify(count)
    raw = BASE[d] + 0.6 * (count - BAND_START[d]) + DIR_WEIGHT[direction]
    return min(max(round(raw), 5), 60)

# dataset rows become:
rows = [{"count": c, "density": classify(c), "direction": d,
         "green_time": policy_green(c, d)}
        for d in ["north", "east", "south", "west"] for c in range(41)]
df = pd.DataFrame(rows)

X = pd.get_dummies(df[["count", "density", "direction"]])   # one-hot encode categories
y = df["green_time"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_tr, y_tr)
print("MAE:", mean_absolute_error(y_te, tree.predict(X_te)))   # report this number
joblib.dump(tree, "models/tree.joblib")

