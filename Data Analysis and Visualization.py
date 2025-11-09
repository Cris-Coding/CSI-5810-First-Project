#Data Analysis and Visualization
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

data_path= "C:\\Users\\lucas\\Desktop\\CSI 5810 Final Project\\train.csv"
training_data = pd.read_csv(data_path)
X= training_data.drop(columns=["Id", "Cover_Type"])
Y= training_data["Cover_Type"]

selector = VarianceThreshold(threshold=0.0001)
selector.fit(X)
mask = selector.get_support()
cols_to_drop = X.columns[~mask]
print("Low variance features removed:", list(cols_to_drop))

# compute stats
stats = X.agg(['mean','std'], numeric_only=True).T
stats['cv'] = stats['std'] / (np.abs(stats['mean']) + 1e-12) # avoid div by zero

# sort by CV (relative variation)
cv_sorted = stats['cv'].sort_values(ascending=False)

# CV Bar Plot
plt.figure(figsize=(10,8))
plt.barh(cv_sorted.index, cv_sorted.values)
plt.gca().invert_yaxis()     # highest CV at top
plt.xlabel("Coefficient of Variation (std / |mean|)")
plt.title("Relative Variation Per Feature")
plt.tight_layout()
plt.show()


# visualizing distributions of continuous features
continuous = ['Elevation','Aspect','Slope',
              'Horizontal_Distance_To_Hydrology',
              'Vertical_Distance_To_Hydrology',
              'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
              'Horizontal_Distance_To_Roadways',
              'Horizontal_Distance_To_Fire_Points']

rows = 3
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(12,8))
axes = axes.flatten()
for i, col in enumerate(continuous):
    axes[i].hist(X[col], bins=40)
    axes[i].set_title(col)
for j in range(len(continuous), rows*cols):
    axes[j].axis('off')
plt.tight_layout()
plt.show()

# frequency bars for soil types
binary_cols = [c for c in X.columns if c.startswith("Soil_Type")]
counts = X[binary_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(10,6))
plt.bar(counts.index, counts.values)
plt.xticks(rotation=90)
plt.title("Soil Type Frequencies")
plt.show()

# correlation heatmap
corr = X.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

