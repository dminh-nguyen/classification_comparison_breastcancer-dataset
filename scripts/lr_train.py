import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# split data (80 training, 20 testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# base model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]  # Probability for class 1

# eval base model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f" Basic Logistic Regression Accuracy: {accuracy:.4f}")
print(f" Basic Logistic Regression ROC-AUC: {roc_auc:.4f}")
print(" Classification Report:\n", classification_report(y_test, y_pred))

# tuning hyperparameters
param_grid = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l1", "l2"],
    "solver": [
        "liblinear",
    ],
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=5,
    scoring="accuracy",
)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_log_reg = grid_search.best_estimator_

# optimized model
y_pred_optimized = best_log_reg.predict(X_test_scaled)
y_pred_prob_optimized = best_log_reg.predict_proba(X_test_scaled)[:, 1]

# eval optimized model
optimized_accuracy = accuracy_score(y_test, y_pred_optimized)
optimized_roc_auc = roc_auc_score(y_test, y_pred_prob_optimized)
print(f"\n Optimized Logistic Regression Accuracy: {optimized_accuracy:.4f}")
print(f" Optimized Logistic Regression ROC-AUC: {optimized_roc_auc:.4f}")
print(f" Best Parameters: {best_params}")
print(
    " Optimized Classification Report:\n",
    classification_report(y_test, y_pred_optimized),
)

conf_matrix = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Optimized Logistic Regression")
plt.show()
