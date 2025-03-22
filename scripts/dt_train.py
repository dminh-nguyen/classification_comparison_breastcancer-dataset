import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# split data (80 training, 20 testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# base model
basic_dt = DecisionTreeClassifier(random_state=42)  # Default settings
basic_dt.fit(X_train, y_train)

y_pred_basic = basic_dt.predict(X_test)
accuracy_basic = accuracy_score(y_test, y_pred_basic)
roc_auc_basic = roc_auc_score(y_test, y_pred_basic)
report_basic = classification_report(y_test, y_pred_basic)

print("\n Basic Decision Tree Performance")
print(f"Accuracy: {accuracy_basic:.4f}")
print(f"ROC-AUC Score: {roc_auc_basic:.4f}")
print("Classification Report:\n", report_basic)

# tuning hyperparameters
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 8, 10, None],
    "min_samples_split": [1.0, 2, 5, 7, 10],
    "min_samples_leaf": [1, 2, 5],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# optimized model
best_dt = grid_search.best_estimator_
y_pred_best = best_dt.predict(X_test)

# eval
accuracy_best = accuracy_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_best)
report_best = classification_report(y_test, y_pred_best)

print("\n Optimized Decision Tree Performance")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Optimized Accuracy: {accuracy_best:.4f}")
print(f"Optimized ROC-AUC Score: {roc_auc_best:.4f}")
print("Optimized Classification Report:\n", report_best)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Optimized Decision Tree")
plt.show()

# visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(
    best_dt,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
)
plt.title("Decision Tree Visualization")
plt.show()
