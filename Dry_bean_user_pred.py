import pandas as pd
import numpy as np
from keras.callbacks import TensorBoard 
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

df_original = pd.read_excel("Dry_Bean_Dataset.xlsx", sheet_name=0)
df = pd.read_excel("Dry_Bean_Dataset.xlsx", sheet_name=0).drop("Class", axis=1)
print(df.head)
cols = df.columns

for i in range(len(cols)-1):
  for j in range(i+1, len(cols)-1):
    x_label = cols[i]
    y_label= cols[j]
    sns.scatterplot(x=x_label, y=y_label, data=df, hue="Class")
    plt.title("Original Data point")
    plt.show()

X = df[cols].values
kmeans = KMeans(n_clusters=6).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1,1))), columns = cols.tolist()+["Cluster"])

print(cluster_df["Cluster"].value_counts())

for i in range(len(cols)-1):
    for j in range(i+1, len(cols)):
        x_label = cols[i]
        y_label = cols[j]
        sns.scatterplot(x=x_label, y=y_label, data=cluster_df, hue='Cluster')
        plt.title("Clustered")
        plt.show()


new_data = []
for col in cols:
    val = float(input(f"Enter {col}: "))
    new_data.append(val)


new_data_cluster = kmeans.predict(np.array(new_data).reshape(1, -1))


print(f"The predicted cluster for the new data point is {new_data_cluster[0]}")



