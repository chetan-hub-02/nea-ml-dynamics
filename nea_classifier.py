import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, ReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Reproducibility
# =====================
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0]}")
else:
    print("No GPU found, running on CPU.")

# =====================
# Paths
# =====================
class_0_path = "/nea_dataset_v1/class_0"
class_1_path = "/nea_dataset_v1/class_1"

# =====================
# Output Directory
# =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("trained_GRU_models", f"model_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Outputs will be saved in: {output_dir}")

# =====================
# Data Loading
# =====================
def load_and_downsample(path, label):
    data = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path)
            if "a" in df.columns and "e" in df.columns:
                a_downsampled = df["a"].values[::10]
                e_downsampled = df["e"].values[::10]
                combined = np.stack((a_downsampled, e_downsampled), axis=-1)
                data.append((combined, label))
    return data

data_class_0 = load_and_downsample(class_0_path, 0)
data_class_1 = load_and_downsample(class_1_path, 1)
all_data = data_class_0 + data_class_1

X = np.array([seq for seq, _ in all_data])
y = np.array([label for _, label in all_data])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.17, random_state=seed, stratify=y
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
)

# =====================
# Hyperparameter Grid
# =====================
# The first GRU width is tuned.
# The second GRU layer uses half the units of the first to progressively reduce the representation dimensionality across layers.
units_grid = [32, 64, 128, 256]
lr_grid = [1e-4, 1e-3, 1e-2]
dropout_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
batch_size_grid = [32, 64, 128, 256]
l2_grid = [1e-4, 1e-3, 1.5e-3, 1e-2]
depth = 2

# =====================
# Model Builder
# =====================
def build_model(input_shape, units, dropout_rate, l2_weight_decay, learning_rate):
    second_units = units // 2

    model = Sequential([
        GRU(
            units,
            activation="tanh",
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l2(l2_weight_decay)
        ),
        BatchNormalization(),
        Dropout(dropout_rate),

        GRU(
            second_units,
            activation="tanh",
            kernel_regularizer=l2(l2_weight_decay)
        ),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, kernel_regularizer=l2(l2_weight_decay)),
        ReLU(),
        Dropout(dropout_rate),

        Dense(32, kernel_regularizer=l2(l2_weight_decay)),
        ReLU(),

        Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_weight_decay))
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# =====================
# Evaluation Helper
# =====================
def evaluate_split(model, X_split, y_split):
    loss, accuracy = model.evaluate(X_split, y_split, verbose=0)
    y_prob = model.predict(X_split, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype("int32")

    if len(np.unique(y_split)) > 1:
        fpr, tpr, _ = roc_curve(y_split, y_prob)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_split, y_prob)
        pr_auc = auc(recall, precision)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "y_prob": y_prob,
        "y_pred": y_pred
    }

# =====================
# Hyperparameter Search
# =====================
results = []
best_overall_val_accuracy = -np.inf
best_overall_run_dir = None

for units in units_grid:
    for lr in lr_grid:
        for dropout_rate in dropout_grid:
            for batch_size in batch_size_grid:
                for l2_weight_decay in l2_grid:
                    tf.keras.backend.clear_session()
                    random.seed(seed)
                    np.random.seed(seed)
                    tf.random.set_seed(seed)

                    second_units = units // 2
                    run_name = (
                        f"units_{units}"
                        f"_secondunits_{second_units}"
                        f"_depth_{depth}"
                        f"_lr_{lr:.0e}"
                        f"_dropout_{dropout_rate:.1f}"
                        f"_batch_{batch_size}"
                        f"_l2_{l2_weight_decay:.4g}"
                    )
                    run_dir = os.path.join(output_dir, run_name)
                    os.makedirs(run_dir, exist_ok=True)

                    print(f"\nStarting run: {run_name}")

                    model = build_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        units=units,
                        dropout_rate=dropout_rate,
                        l2_weight_decay=l2_weight_decay,
                        learning_rate=lr
                    )

                    # Save a model summary for each run
                    summary_path = os.path.join(run_dir, "model_summary.txt")
                    with open(summary_path, "w") as f:
                        model.summary(print_fn=lambda x: f.write(x + "\n"))

                    checkpoint_path = os.path.join(run_dir, "best_weights.h5")
                    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_path,
                        save_best_only=True,
                        monitor="val_accuracy",
                        mode="max",
                        verbose=1
                    )

                    earlystop_cb = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=7,
                        restore_best_weights=True,
                        verbose=1
                    )

                    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=1
                    )

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=batch_size,
                        verbose=2,
                        callbacks=[checkpoint_cb, earlystop_cb, lr_schedule]
                    )

                    # Save final model too
                    final_model_path = os.path.join(run_dir, "final_model.h5")
                    model.save(final_model_path)

                    history_df = pd.DataFrame(history.history)
                    history_df.to_csv(os.path.join(run_dir, "training_history.csv"), index=False)

                    # Save run configuration
                    run_config = {
                        "units": units,
                        "second_units": second_units,
                        "depth": depth,
                        "learning_rate": lr,
                        "dropout": dropout_rate,
                        "batch_size": batch_size,
                        "l2_weight_decay": l2_weight_decay,
                        "seed": seed,
                        "train_test_split_random_state": seed,
                        "model_checkpoint_monitor": "val_accuracy",
                        "model_checkpoint_mode": "max",
                        "early_stopping_monitor": "val_loss",
                        "early_stopping_patience": 7
                    }
                    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
                        json.dump(run_config, f, indent=4)

                    # Accuracy plot
                    plt.figure(figsize=(8, 6))
                    plt.plot(history.history["accuracy"], label="Train Accuracy")
                    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
                    plt.xlabel("Epochs", fontsize=16)
                    plt.ylabel("Accuracy", fontsize=16)
                    plt.title("Training and Validation Accuracy", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.tick_params(axis="both", which="major", labelsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "accuracy_plot.jpg"), dpi=300)
                    plt.savefig(os.path.join(run_dir, "accuracy_plot.eps"), format="eps")
                    plt.close()

                    # Loss plot
                    plt.figure(figsize=(8, 6))
                    plt.plot(history.history["loss"], label="Train Loss")
                    plt.plot(history.history["val_loss"], label="Validation Loss")
                    plt.xlabel("Epochs", fontsize=16)
                    plt.ylabel("Loss", fontsize=16)
                    plt.title("Training and Validation Loss", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.tick_params(axis="both", which="major", labelsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "loss_plot.jpg"), dpi=300)
                    plt.savefig(os.path.join(run_dir, "loss_plot.eps"), format="eps")
                    plt.close()

                    # Load best checkpoint weights for evaluation
                    model.load_weights(checkpoint_path)

                    train_metrics = evaluate_split(model, X_train, y_train)
                    val_metrics = evaluate_split(model, X_val, y_val)
                    test_metrics = evaluate_split(model, X_test, y_test)

                    # ROC Curve on test set
                    fpr, tpr, _ = roc_curve(y_test, test_metrics["y_prob"])
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
                    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
                    plt.xlabel("False Positive Rate", fontsize=16)
                    plt.ylabel("True Positive Rate (Recall)", fontsize=16)
                    plt.title("ROC Curve", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.tick_params(axis="both", which="major", labelsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "roc_curve.jpg"), dpi=300)
                    plt.savefig(os.path.join(run_dir, "roc_curve.eps"), format="eps")
                    plt.close()

                    # Precision-Recall Curve on test set
                    precision, recall, _ = precision_recall_curve(y_test, test_metrics["y_prob"])
                    pr_auc = auc(recall, precision)
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, color="green", lw=2, label=f"AUC = {pr_auc:.3f}")
                    plt.xlabel("Recall", fontsize=16)
                    plt.ylabel("Precision", fontsize=16)
                    plt.title("Precision-Recall Curve", fontsize=16)
                    plt.legend(fontsize=14)
                    plt.tick_params(axis="both", which="major", labelsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "pr_curve.jpg"), dpi=300)
                    plt.savefig(os.path.join(run_dir, "pr_curve.eps"), format="eps")
                    plt.close()

                    # Confusion Matrix on test set
                    cm = confusion_matrix(y_test, test_metrics["y_pred"])
                    cm_df = pd.DataFrame(cm, index=["Class 0", "Class 1"], columns=["Pred 0", "Pred 1"])
                    cm_df.to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

                    plt.figure(figsize=(6, 6))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=["Class 0", "Class 1"]
                    )
                    plt.title("Confusion Matrix", fontsize=16)
                    plt.xlabel("Predicted Label", fontsize=16)
                    plt.ylabel("True Label", fontsize=16)
                    plt.tick_params(axis="both", which="major", labelsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "confusion_matrix.jpg"), dpi=300)
                    plt.savefig(os.path.join(run_dir, "confusion_matrix.eps"), format="eps")
                    plt.close()

                    # Full evaluation metrics
                    metrics_rows = [
                        {
                            "split": "train",
                            "loss": train_metrics["loss"],
                            "accuracy": train_metrics["accuracy"],
                            "roc_auc": train_metrics["roc_auc"],
                            "pr_auc": train_metrics["pr_auc"]
                        },
                        {
                            "split": "val",
                            "loss": val_metrics["loss"],
                            "accuracy": val_metrics["accuracy"],
                            "roc_auc": val_metrics["roc_auc"],
                            "pr_auc": val_metrics["pr_auc"]
                        },
                        {
                            "split": "test",
                            "loss": test_metrics["loss"],
                            "accuracy": test_metrics["accuracy"],
                            "roc_auc": test_metrics["roc_auc"],
                            "pr_auc": test_metrics["pr_auc"]
                        }
                    ]
                    metrics_df = pd.DataFrame(metrics_rows)
                    metrics_df.to_csv(os.path.join(run_dir, "evaluation_metrics.csv"), index=False)

                    run_result = {
                        "run_name": run_name,
                        "units": units,
                        "second_units": second_units,
                        "depth": depth,
                        "learning_rate": lr,
                        "dropout": dropout_rate,
                        "batch_size": batch_size,
                        "l2_weight_decay": l2_weight_decay,
                        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "train_roc_auc": train_metrics["roc_auc"],
                        "train_pr_auc": train_metrics["pr_auc"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_roc_auc": val_metrics["roc_auc"],
                        "val_pr_auc": val_metrics["pr_auc"],
                        "test_loss": test_metrics["loss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_roc_auc": test_metrics["roc_auc"],
                        "test_pr_auc": test_metrics["pr_auc"],
                        "run_dir": run_dir,
                        "best_weights_path": checkpoint_path,
                        "final_model_path": final_model_path
                    }
                    results.append(run_result)

                    if val_metrics["accuracy"] > best_overall_val_accuracy:
                        best_overall_val_accuracy = val_metrics["accuracy"]
                        best_overall_run_dir = run_dir

                    print(
                        f"Completed {run_name} | "
                        f"val_accuracy={val_metrics['accuracy']:.6f} | "
                        f"val_loss={val_metrics['loss']:.6f} | "
                        f"test_accuracy={test_metrics['accuracy']:.6f}"
                    )

# =====================
# Save Search Summary
# =====================
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "hyperparameter_search_summary.csv"), index=False)

best_idx = results_df["val_accuracy"].astype(float).idxmax()
best_run = results_df.loc[best_idx]
best_run.to_frame().T.to_csv(os.path.join(output_dir, "best_run_summary.csv"), index=False)

print("\nTraining complete.")
print(f"Best run saved in: {best_overall_run_dir}")
print("Hyperparameter search summary, per-run best_weights, final_model.h5, evaluation metrics, and plots are saved.")
