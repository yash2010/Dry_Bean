import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# load and preprocess the data
df = pd.read_excel("Dry_Bean_Dataset.xlsx", sheet_name=0).drop("Class", axis=1)
cols = df.columns
X = df[cols].values

# train KMeans model
kmeans = KMeans(n_clusters=6).fit(X)

# ask user to enter feature values for new data
new_data = []
for col in cols:
    val = float(input(f"Enter {col}: "))
    new_data.append(val)

# predict the cluster for new data
new_data_cluster = kmeans.predict(np.array(new_data).reshape(1, -1))

print(f"The predicted cluster for the new data point is {new_data_cluster[0]}")
