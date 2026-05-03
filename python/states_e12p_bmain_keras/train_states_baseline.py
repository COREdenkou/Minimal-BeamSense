#!/usr/bin/env python3
"""Train the three-state Minimal BeamSense baseline model."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from dataGenerator_states import DataGenerator, DEFAULT_STATE_ORDER


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a three-state Minimal BeamSense baseline model on fused 10x234x12 windows."
    )
    p.add_argument(
        "--train-csv",
        default=r"<PROJECT_ROOT>\state_training\state_train.csv",
        help="Path to state_train.csv",
    )
    p.add_argument(
        "--val-csv",
        default=r"<PROJECT_ROOT>\state_training\state_val.csv",
        help="Path to state_val.csv",
    )
    p.add_argument(
        "--out-dir",
        default=r"<PROJECT_ROOT>\state_training\models\states_baseline",
        help="Directory to save model and training artifacts",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=111)
    return p.parse_args()


def build_states_model(input_shape=(10, 234, 12), classes=3):
    model = models.Sequential()

    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation="softmax"))
    return model


def save_history_csv(history, out_dir: Path):
    """
    Save Keras History object to train_history.csv for external plotting scripts.

    Output columns:
      epoch, accuracy, loss, val_accuracy, val_loss, learning_rate, ...
    The function keeps all keys from history.history automatically.
    """
    history_csv_path = out_dir / "train_history.csv"

    hist = history.history
    keys = list(hist.keys())

    # Determine number of epochs from the first history key.
    n_epochs = 0
    if keys:
        n_epochs = len(hist[keys[0]])

    with open(history_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)

        for i in range(n_epochs):
            row = [i + 1]
            for k in keys:
                values = hist.get(k, [])
                row.append(values[i] if i < len(values) else "")
            writer.writerow(row)

    return history_csv_path


def save_history_plots(history, out_dir: Path):
    # Accuracy
    plt.figure(figsize=(8, 5))
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_accuracy.png", dpi=300)
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_loss.png", dpi=300)
    plt.close()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.is_file():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    if not val_csv.is_file():
        raise FileNotFoundError(f"val_csv not found: {val_csv}")

    # Build generators
    train_gen = DataGenerator(
        dataset_csv=str(train_csv),
        batchsize=args.batch_size,
        shuffle=True,
        to_categorical=True,
        label_order=DEFAULT_STATE_ORDER,
        normalize_by_180=True,
    )

    val_gen = DataGenerator(
        dataset_csv=str(val_csv),
        batchsize=args.batch_size,
        shuffle=False,
        to_categorical=True,
        label_order=train_gen.label_order,
        normalize_by_180=True,
    )

    num_classes = train_gen.num_classes
    label_order = train_gen.label_order

    # Save label map
    label_map_path = out_dir / "label_map.json"
    train_gen.export_label_map(label_map_path)

    # Build model
    model = build_states_model(input_shape=(10, 234, 12), classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_model_path = out_dir / "states_baseline_best.keras"
    last_model_path = out_dir / "states_baseline_last.keras"

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            verbose=1,
            factor=0.5,
            min_lr=1e-5,
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        x=train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model after training loop
    model.save(last_model_path)

    # Save raw history
    history_path = out_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)

    # Save CSV history for external plotting scripts
    history_csv_path = save_history_csv(history, out_dir)

    # Save plots
    save_history_plots(history, out_dir)

    # Save run summary
    summary = {
        "version": "beamsense_v2_states",
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "out_dir": str(out_dir),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_classes": num_classes,
        "label_order": label_order,
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "label_map_path": str(label_map_path),
        "train_history_json_path": str(history_path),
        "train_history_csv_path": str(history_csv_path),
    }

    summary_path = out_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] train_csv:", train_csv)
    print("[INFO] val_csv:", val_csv)
    print("[INFO] num_classes:", num_classes)
    print("[INFO] label_order:", label_order)
    print("[INFO] best_model_path:", best_model_path)
    print("[INFO] last_model_path:", last_model_path)
    print("[INFO] history_path:", history_path)
    print("[INFO] history_csv_path:", history_csv_path)
    print("[INFO] summary_path:", summary_path)


if __name__ == "__main__":
    main()