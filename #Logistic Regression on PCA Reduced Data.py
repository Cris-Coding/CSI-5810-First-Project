#Logistic Regression on PCA Reduced Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

#Training Data
data_path= "C:\\Users\\lucas\\Desktop\\CSI 5810 Final Project\\train.csv"
training_data= pd.read_csv(data_path)
X= training_data.drop(columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15", "Soil_Type8", "Soil_Type25"])
Y= training_data["Cover_Type"]

#Pipeline with PCA and Logistic Regression
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

solvers  = ["lbfgs", "saga", "sag"]   # all support multiclass; sag/lbfgs -> L2, saga -> L1/L2/elasticnet
C_values = [1, 1000, 100000]

for solver in solvers:
    for C in C_values:
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('pca', PCA(n_components=30)),
            ('logreg', LogisticRegression(
                solver=solver,
                C=C,
                max_iter=5000,
                class_weight='balanced',
                n_jobs=None  # (no effect for lbfgs/sag; safe to leave)
            ))
        ])

        # get accuracy and macro-F1 with mean ± std
        scores = cross_validate(
            pipe, X, Y, cv=cv,
            scoring={'acc':'accuracy', 'f1':'f1_macro'},
            n_jobs=-1, return_train_score=False, error_score='raise'
        )
        acc_mean, acc_std = scores['test_acc'].mean(), scores['test_acc'].std()
        f1_mean,  f1_std  = scores['test_f1'].mean(),  scores['test_f1'].std()

        print(f"solver={solver:5s}, C={C:>6}  |  "
              f"Acc {acc_mean:.4f} +/- {acc_std:.4f}  |  "
              f"Macro-F1 {f1_mean:.4f} +/- {f1_std:.4f}")