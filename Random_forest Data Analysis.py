#Random Forest Classification Confusion Matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns

#Training Data
data_path= "C:\\Users\\lucas\\Desktop\\CSI 5810 Final Project\\train.csv"
training_data= pd.read_csv(data_path)
X= training_data.drop(columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15", "Soil_Type8", "Soil_Type25"])
Y= training_data["Cover_Type"]

best_rf = RandomForestClassifier(
    n_estimators=20,
    max_depth=None,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X, Y)
preds = best_rf.predict(X)

cm = confusion_matrix(Y, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Get importance values
importances = best_rf.feature_importances_
feat_names = X.columns


fi = pd.DataFrame({
    'feature': feat_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(fi.head(10))   # top 10 most important features
plt.figure(figsize=(10,8))
plt.barh(fi.head(10)['feature'], fi.head(10)['importance'])
plt.gca().invert_yaxis()
plt.xlabel("Feature Importance (Random Forest)")
plt.title("Top 10 RF Feature Importances")
plt.tight_layout()
plt.show()