#!/usr/bin/env python3
"""Data generator for the three-state Minimal BeamSense Keras pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as spio
from tensorflow import keras


DEFAULT_STATE_ORDER = [
    "empty",
    "one_person",
    "two_person",
]


def load_label_map_from_rows(labels, preferred_order=None):
    preferred_order = preferred_order or DEFAULT_STATE_ORDER
    labels = list(labels)

    unknown = [x for x in labels if x not in preferred_order]
    if unknown:
        raise RuntimeError(
            f"Found labels not present in preferred_order: {unknown}\n"
            f"Either rename labels in your segments CSV or extend DEFAULT_STATE_ORDER."
        )

    label_order = [x for x in preferred_order if x in labels]
    label_to_index = {label: i for i, label in enumerate(label_order)}
    return label_order, label_to_index


def read_fused_mat(file_path):
    obj = spio.loadmat(file_path)
    if "bf_matrix" not in obj:
        raise KeyError(f"bf_matrix missing in {file_path}")

    bf = obj["bf_matrix"].astype(np.float32)
    if bf.shape != (10, 234, 12):
        raise RuntimeError(
            f"Unexpected fused bf_matrix shape in {file_path}: got {bf.shape}, expected (10, 234, 12)"
        )
    return bf


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for three-state fused multi-peer Minimal BeamSense windows.

    Expected CSV columns:
      - fused_path
      - state_label

    Each fused_path should point to a .mat containing:
      - bf_matrix with shape (10, 234, 12)
    """

    def __init__(
        self,
        dataset_csv,
        batchsize=32,
        shuffle=True,
        to_categorical=True,
        label_order=None,
        normalize_by_180=True,
    ):
        self.dataset_csv = str(dataset_csv)
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.to_categorical = to_categorical
        self.normalize_by_180 = normalize_by_180

        df = pd.read_csv(self.dataset_csv)
        required = {"fused_path", "state_label"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing required columns in {self.dataset_csv}: {sorted(missing)}")

        self.df = df.copy()
        self.paths = self.df["fused_path"].astype(str).tolist()
        self.labels = self.df["state_label"].astype(str).tolist()

        if label_order is None:
            self.label_order, self.label_to_index = load_label_map_from_rows(sorted(set(self.labels)))
        else:
            label_order = list(label_order)
            self.label_order, self.label_to_index = load_label_map_from_rows(sorted(set(self.labels)), label_order)

        self.num_classes = len(self.label_order)

        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batchsize))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batchsize : (idx + 1) * self.batchsize]
        X, y = self._load_batch(indexes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_batch(self, indexes):
        batch_data = np.empty((len(indexes), 10, 234, 12), dtype=np.float32)
        batch_label = np.empty(len(indexes), dtype=int)

        for i, k in enumerate(indexes):
            file_path = self.paths[k]
            label = self.labels[k]

            x = read_fused_mat(file_path)
            if self.normalize_by_180:
                x = x / 180.0

            batch_data[i] = x
            batch_label[i] = self.label_to_index[label]

        if self.to_categorical:
            batch_label = keras.utils.to_categorical(batch_label, num_classes=self.num_classes)

        return batch_data, batch_label

    def export_label_map(self, out_json_path):
        obj = {
            "label_order": self.label_order,
            "label_to_index": self.label_to_index,
            "num_classes": self.num_classes,
            "dataset_csv": self.dataset_csv,
        }
        out_path = Path(out_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return str(out_path)