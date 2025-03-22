# 🩺 Breast Cancer Classification with Machine Learning

This project evaluates and compares multiple **machine learning algorithms** for **breast cancer classification** using the **Wisconsin Breast Cancer Dataset**. The models analyzed include:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Decision Trees**
- **Logistic Regression**

Each model is optimized and evaluated using key performance metrics such as **accuracy, precision, recall, F1-score, and ROC-AUC**.

---

## 🛠 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. **Create a virtual environment (optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # For MacOS/Linux
   venv\Scripts\activate  # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run a specific model**

   ```bash
   python scripts/knn_train.py
   python scripts/svm_train.py
   python scripts/dt_train.py
   python scripts/lr_train.py
   ```

5. **Run all models and compare results**
   ```bash
   python main.py
   ```

---

## 📈 Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: How many predicted positives are actually positive.
- **Recall (Sensitivity)**: How many actual positives were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC Score**: Measures how well the model differentiates between classes.

---

## 📌 Key Takeaways

- **Logistic Regression and SVM performed best for breast cancer classification.**
- **Hyperparameter tuning significantly improved model performance.**
- **Decision Trees underperformed but could be improved with ensemble methods.**
- **Machine learning models can play a crucial role in medical diagnostics.**

---
