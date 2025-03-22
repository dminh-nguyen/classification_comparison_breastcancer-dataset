import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


def load_and_preprocess_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)  # 0 = malignant, 1 = benign
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_svm(X_train, y_train):
    svm = SVC(kernel="linear", random_state=42)
    svm.fit(X_train, y_train)
    return svm


def evaluate_model(model, X_test, y_test, title="SVM Model Performance"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
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
    plt.title(title)
    plt.show()


def optimize_svm(X_train, y_train):
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", verbose=1)
    grid_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data()

    basic_svm = train_svm(X_train_scaled, y_train)
    evaluate_model(basic_svm, X_test_scaled, y_test, "Basic SVM Performance")

    optimized_svm = optimize_svm(X_train_scaled, y_train)
    evaluate_model(optimized_svm, X_test_scaled, y_test, "Optimized SVM Performance")
