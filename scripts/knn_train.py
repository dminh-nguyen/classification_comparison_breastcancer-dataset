# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

# split data (80 training, 20 testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# knn with initial k=5
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# eval
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# confusion matrix base model
plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# tuning hyperparameters
param_grid = {"n_neighbors": np.arange(1, 21), "weights": ["uniform", "distance"]}
grid_search = GridSearchCV(
    KNeighborsClassifier(metric="euclidean"), param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_["n_neighbors"]
best_weights = grid_search.best_params_["weights"]
print(f"Best k: {best_k}, Best Weights: {best_weights}")

# train and eval optimized model
optimized_knn = KNeighborsClassifier(
    n_neighbors=best_k, metric="euclidean", weights=best_weights
)
optimized_knn.fit(X_train, y_train)
y_pred_optimized = optimized_knn.predict(X_test)

print("Optimized KNN Accuracy:", accuracy_score(y_test, y_pred_optimized))
print(
    "Optimized Classification Report:\n",
    classification_report(y_test, y_pred_optimized),
)

# confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred_optimized),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

roc_auc = roc_auc_score(y_test, optimized_knn.predict_proba(X_test)[:, 1])
print("ROC-AUC Score:", roc_auc)
