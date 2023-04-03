import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import silhouette_score

df = pd.read_excel("Dry_Bean_Dataset.xlsx", sheet_name=0)
cols = df.columns

df.head()

for i in range(len(cols)-1):
  for j in range(i+1, (len(cols)-1)):
    x_label = df.columns[i]
    y_label = df.columns[j]
    sns.scatterplot(x=x_label, y=y_label, data = df, hue = "Class")
    plt.show()



print(df.dtypes)

print(df["Class"])

x = "Perimeter"
y = "ShapeFactor3"

X = df[df.columns[:-1]].values

scaler = StandardScaler()
scaler = scaler.fit_transform(X)

model = KMeans(n_clusters=7, random_state=0 )
model = model.fit(X)

cluster = model.labels_

cluster_df = pd.DataFrame(np.hstack((X, cluster.reshape(-1,1))), columns=df.columns)

sns.scatterplot(x=x, y=y, hue = "Class", data = cluster_df )

sns.scatterplot(x=x, y=y, hue = "Class", data = df )

silhouette_avg = silhouette_score(X, cluster)

print(silhouette_avg)

Area = float(input("Enter Area value: "))
perimeter = float(input("Enter Perimeter value: "))
MajorAxisLength = float(input("Enter MajorAxisLength value: "))
AspectRation = float(input("Enter AspectRation value: "))
Eccentricity = float(input("Enter Eccentricity value: "))
ConvexArea = float(input("Enter ConvexArea value: "))
EquviDiameter = float(input("Enter EquviDiameter value: "))
Extent = float(input("Enter Extent value: "))
Solidity = float(input("Enter Solidity value: "))
Roundness = float(input("Enter Roundness value: "))
Compactness = float(input("Enter Compactness value: "))
ShapeFactor1 = float(input("Enter ShapeFactor1 value: "))
ShapeFactor2 = float(input("Enter ShapeFactor2 value: "))
shape_factor3 = float(input("Enter ShapeFactor3 value: "))


user_data = np.array(["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength" , "AspectRation", "Eccentricity", "ConvexArea", 
                      "EquviDiameter", "Extent", "Solidity", "Roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", 
                      "ShapeFactor3", "ShapeFactor4"])
user_data_scaled = scaler.transform(user_data)


user_cluster_label = model.predict(user_data_scaled)

print("The predicted cluster label for the input data is:", user_cluster_label[0])

