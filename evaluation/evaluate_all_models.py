import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# PATHS
# =========================
TEST_DIR = r"E:\Desktop\PPE_Project\dataset_split\test"
MODELS_DIR = r"E:\Desktop\PPE_Project\saved_models"

CONF_MATRIX_DIR = r"E:\Desktop\PPE_Project\outputs\confusion_matrices"
GRAPHS_DIR = r"E:\Desktop\PPE_Project\outputs\graphs"
REPORTS_DIR = r"E:\Desktop\PPE_Project\outputs\reports"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs(CONF_MATRIX_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================
# DATA LOADER
# =========================
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

# =========================
# LOAD MODELS
# =========================
models = {
    "CNN": tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "cnn_scratch.keras")
    ),
    "MobileNet": tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "best_mobilenet.keras")
    )
}

results = {}

# =========================
# EVALUATION LOOP
# =========================
for name, model in models.items():

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(test_gen)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    # =====================
    # METRICS
    # =====================
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    results[name] = [acc, prec, rec, f1]

    print(f"\n{name} Results")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # =====================
    # CONFUSION MATRIX
    # =====================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(CONF_MATRIX_DIR, f"{name}_cm.png")
    plt.savefig(cm_path)
    plt.close()

    # =====================
    # SAVE REPORT
    # =====================
    report_path = os.path.join(REPORTS_DIR, f"{name}_report.txt")

    with open(report_path, "w") as f:
        f.write(f"{name} Evaluation Report\n\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1-score: {f1}\n\n")
        f.write(classification_report(y_true, y_pred))

print("\n===== FINAL COMPARISON =====")

for k, v in results.items():
    print(f"\n{k}")
    print(f"Accuracy : {v[0]:.4f}")
    print(f"Precision: {v[1]:.4f}")
    print(f"Recall   : {v[2]:.4f}")
    print(f"F1-score : {v[3]:.4f}")