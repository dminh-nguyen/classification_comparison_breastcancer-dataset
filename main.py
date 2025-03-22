import subprocess
import os

# Define script paths
scripts = {
    "K-Nearest Neighbors": "scripts/knn_train.py",
    "Support Vector Machine": "scripts/svm_train.py",
    "Decision Tree": "scripts/dt_train.py",
    "Logistic Regression": "scripts/lr_train.py",
}

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# File to save the report
report_file = "results/report.txt"

# Run each script and collect output
results = {}

for model, script in scripts.items():
    print(f"🔹 Running {model}...")
    process = subprocess.Popen(
        ["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        results[model] = stdout
        print(f"✅ {model} completed successfully.\n")
    else:
        print(f"❌ {model} failed to run. Error:\n{stderr}\n")

# Write the summary report
with open(report_file, "w") as f:
    f.write("# Model Comparison Report\n\n")
    f.write(
        "This report summarizes the performance of four classification models on the Breast Cancer dataset.\n\n"
    )

    for model, output in results.items():
        f.write(f"## {model} Results\n")
        f.write("```\n" + output + "\n```\n")

print(f"📄 Report generated at: {report_file}")
