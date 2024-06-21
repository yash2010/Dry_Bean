# Dry Bean Dataset Clustering 🌱

This repository contains a clustering model implemented using KMeans to analyze and classify data from the Dry Bean Dataset. The model clusters the data based on various morphological features of dry beans.

## Dataset 🧾
The dataset used for this project is stored in an Excel file named Dry_Bean_Dataset.xlsx. This file contains various features of dry beans including:
+ Area
+ Perimeter
+ MajorAxisLength
+ MinorAxisLength
+ AspectRatio
+ Eccentricity
+ ConvexArea
+ EquviDiameter
+ Extent
+ Solidity
+ Roundness
+ Compactness
+ ShapeFactor1
+ ShapeFactor2
+ ShapeFactor3
+ ShapeFactor4
+ Class (Bean Type)

The dataset is sourced from [kaggle](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset).

## Files 📁
+ Dry_Bean_Clustering.py: Python script containing the implementation of the clustering model.
+ Dry_Bean_Dataset.xlsx: Dataset containing features of dry beans.
  
## Requirements
+ Python 3.6 or higher
+ Pandas
+ NumPy
+ Scikit-learn
+ Seaborn
+ Matplotlib
+ Openpyxl

1. You can install the required libraries using the following command:
```sh
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl
```

## Usage 
1. Clone the repository:
```sh
git clone https://github.com/yash2010/Dry_Bean_Clustering.git
```

2. Navigate to the project directory:
```sh
cd Dry_Bean_Clustering
```

3. Ensure that Dry_Bean_Dataset.xlsx is in the project directory.

4. Run the Python script:
```sh
python Dry_Bean_Clustering.py
```
5. The script will display scatter plots of various feature combinations and perform KMeans clustering on the dataset. It will also prompt you to enter feature values for a new dry bean sample to predict its cluster label.

## Functions
### Data Visualization
The script reads the Dry_Bean_Dataset.xlsx dataset and displays scatter plots for combinations of features, colored by the bean class.

### Data Preprocessing
Features are scaled using StandardScaler to standardize the data for clustering.

### Model Training
The KMeans clustering model is trained on the dataset with a specified number of clusters (7 in this case).

### Model Evaluation
The silhouette score is calculated to evaluate the clustering performance.

### User Input for Prediction
+ The script prompts the user to input values for various features of a dry bean.
+ The input data is scaled, and the KMeans model predicts the cluster label for the input data.

## Visualization 📊
The script uses Seaborn and Matplotlib to plot scatter plots for combinations of features, showing the distribution of bean classes and clusters.

## Example
After running the script, the system will:
  + Display scatter plots for various feature combinations.
  + Perform KMeans clustering on the dataset.
  + Calculate and print the silhouette score.
  + Prompt the user to enter feature values and predict the cluster label for the new dry bean sample.
