#Random Forest Classification
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


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

n_estimators_values = [5, 10, 20]
max_depth_values    = [None, 10, 20]
for n in n_estimators_values:
    for d in max_depth_values:

        pipe_rf = Pipeline([
            ('rf', RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ))
        ])

        scores = cross_validate(
            pipe_rf, X, Y,
            cv=cv,
            scoring={'acc':'accuracy', 'f1':'f1_macro'},
            n_jobs=-1,
            return_train_score=False
        )

        acc_mean = scores['test_acc'].mean()
        acc_std  = scores['test_acc'].std()
        f1_mean  = scores['test_f1'].mean()
        f1_std   = scores['test_f1'].std()

        print(f"n_estimators={n:>3}, max_depth={str(d):>4}  ->  "
              f"Acc {acc_mean:.4f} ± {acc_std:.4f}   |   "
              f"F1 {f1_mean:.4f} ± {f1_std:.4f}")