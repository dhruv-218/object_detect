import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# ------------------------
# 💡 Step 1: Remove .cache if corrupted
# ------------------------
CACHE_PATH = "valid/labels.cache"  # ← update if your validation path is different
if os.path.exists(CACHE_PATH):
    print(f"🧹 Removing corrupted cache: {CACHE_PATH}")
    os.remove(CACHE_PATH)

# ------------------------
# 🚀 Step 2: Load the trained model
# ------------------------
model = YOLO("yolo11n.pt")  # Adjust path if needed

# ------------------------
# 🧪 Step 3: Run evaluation
# ------------------------
results = model.val(data="data_kaggle.yaml")  # Your dataset YAML

# ------------------------
# 📊 Step 4: Plot PR Curve
# ------------------------
def plot_pr_curve(pr_curves, class_names):
    plt.figure(figsize=(8, 6))
    for i, pr in enumerate(pr_curves):
        if pr is not None:
            plt.plot(pr[:, 0], pr[:, 1], label=class_names[i])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pr_curve.png")
    plt.show()

plot_pr_curve(results.pr_curves, results.names)

# ------------------------
# 📈 Step 5: F1 vs Confidence
# ------------------------
def plot_f1_vs_conf(f1_data):
    plt.figure(figsize=(8, 6))
    conf_thresholds = f1_data[:, 0]
    f1_scores = f1_data[:, 1]
    plt.plot(conf_thresholds, f1_scores, color='purple')
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Confidence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("f1_vs_conf.png")
    plt.show()

plot_f1_vs_conf(results.f1)

# ------------------------
# 🔲 Step 6: Confusion Matrix
# ------------------------
def plot_confusion_matrix(results):
    matrix = results.confusion_matrix
    if matrix is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(results.names.values()))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

plot_confusion_matrix(results)

# ------------------------
# 🧾 Step 7: Print mAP and metrics
# ------------------------
print("\n📊 Model Evaluation Summary:")
for metric, value in results.metrics.items():
    print(f"{metric}: {value:.4f}")
