from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate

from src.models.mctrcm_data import prepare_multicorpus_data
from src.models.mctrcm_v2 import (
    MCTRCMV2,
    OrderedThresholdOrdinalHead,
    TaskHeadSpecV2,
    _ordinal_probabilities_torch,
)
from src.utils.constants import CANONICAL_MODALITIES, PROJECT_ROOT


def _task_specs(task_metadata):
    specs = []
    for meta in task_metadata:
        if meta.label_type == "continuous":
            output_dim = 1
            num_classes = None
        elif meta.label_type == "binary":
            output_dim = 1
            num_classes = len(meta.class_space or [0, 1])
        elif meta.label_type == "ordinal":
            num_classes = len(meta.class_space or [])
            output_dim = max(num_classes - 1, 1)
        else:
            num_classes = len(meta.class_space or [])
            output_dim = num_classes
        specs.append(
            TaskHeadSpecV2(
                task_index=meta.task_index,
                label_type=meta.label_type,
                num_classes=num_classes,
                output_dim=output_dim,
            )
        )
    return specs


class ProtocolAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        split_path = PROJECT_ROOT / "data_interim" / "window_tables" / "studentlife" / "splits.json"
        cls.prepared = prepare_multicorpus_data(["studentlife"]) if split_path.exists() else None

    def _require_prepared_data(self):
        if self.prepared is None:
            self.skipTest("Prepared data_interim tables are not present in this checkout.")
        return self.prepared

    def test_ordinal_probabilities_are_valid(self):
        logits = torch.tensor([[2.0, 0.5, -0.2, -1.0], [-0.5, 1.0, 0.2, -2.0]], dtype=torch.float32)
        probabilities = _ordinal_probabilities_torch(logits)
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.size(0)), atol=1e-6))

    def test_ordered_thresholds_are_monotonic(self):
        torch.manual_seed(123)
        head = OrderedThresholdOrdinalHead(
            feature_dim=5,
            num_classes=4,
            hidden_dim=8,
            dropout=0.0,
            activation="gelu",
        )
        logits = head(torch.randn(3, 5))
        thresholds = head.thresholds()
        self.assertEqual(tuple(logits.shape), (3, 3))
        self.assertTrue(torch.all(thresholds[1:] > thresholds[:-1]))
        probabilities = _ordinal_probabilities_torch(logits)
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-6))

    def test_participant_splits_are_disjoint(self):
        any_present = False
        for dataset_id in ("studentlife", "deprest_cat", "psyche_d", "depresjon", "obf"):
            path = PROJECT_ROOT / "data_interim" / "window_tables" / dataset_id / "splits.json"
            if not path.exists():
                continue
            any_present = True
            splits = json.loads(path.read_text(encoding="utf-8"))
            train = {str(x) for x in splits.get("train_subjects", [])}
            valid = {str(x) for x in splits.get("valid_subjects", [])}
            test = {str(x) for x in splits.get("test_subjects", [])}
            self.assertFalse(train & valid, dataset_id)
            self.assertFalse(train & test, dataset_id)
            self.assertFalse(valid & test, dataset_id)
        if not any_present:
            self.skipTest("Prepared split files are not present in this checkout.")

    def test_standardization_stats_use_training_split_only(self):
        prepared = self._require_prepared_data()
        train_frame = prepared.train.frame
        for modality, columns in prepared.modality_feature_columns.items():
            values = train_frame[columns].apply(pd.to_numeric, errors="coerce")
            expected_mean = values.mean(axis=0, skipna=True).fillna(0.0).to_numpy(dtype=np.float32)
            expected_std = values.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
            got_mean = np.asarray(prepared.feature_stats[modality]["mean"], dtype=np.float32)
            got_std = np.asarray(prepared.feature_stats[modality]["std"], dtype=np.float32)
            self.assertTrue(np.allclose(got_mean, expected_mean, equal_nan=False), modality)
            self.assertTrue(np.allclose(got_std, expected_std, equal_nan=False), modality)

    def test_model_forward_no_nan_and_output_dimensions(self):
        torch.manual_seed(123)
        data = self._require_prepared_data()
        modality_dims = {m: len(data.modality_feature_columns[m]) for m in CANONICAL_MODALITIES}
        model = MCTRCMV2(
            modality_input_dims=modality_dims,
            num_datasets=len(data.dataset_to_index),
            task_head_specs=_task_specs(data.task_metadata),
            native_input_dim=len(data.native_feature_columns),
            token_dim=16,
            encoder_hidden_dim=24,
            transformer_layers=1,
            transformer_heads=4,
            transformer_ff_dim=32,
            latent_dim=24,
            dataset_embedding_dim=8,
            task_embedding_dim=8,
            concept_dim=0,
            concept_hidden_dim=16,
            task_feature_dim=16,
            output_refine_dim=8,
            recursion_steps=2,
            dropout=0.0,
        )
        model.eval()
        batch = default_collate([data.train[0], data.train[1]])
        with torch.no_grad():
            out1 = model(batch)
            out2 = model(batch)
        self.assertEqual(len(out1["steps"]), 2)
        self.assertTrue(torch.allclose(out1["base_latent"], out2["base_latent"], atol=1e-7))
        for step in out1["steps"]:
            self.assertFalse(torch.isnan(step["predictive_latent"]).any())
            for task_id, logits in step["logits_by_task"].items():
                spec = {s.task_index: s for s in _task_specs(data.task_metadata)}[task_id]
                self.assertEqual(logits.shape[-1], spec.output_dim)
                self.assertFalse(torch.isnan(logits).any())

    def test_missing_modalities_do_not_create_nan(self):
        data = self._require_prepared_data()
        modality_dims = {m: len(data.modality_feature_columns[m]) for m in CANONICAL_MODALITIES}
        model = MCTRCMV2(
            modality_input_dims=modality_dims,
            num_datasets=len(data.dataset_to_index),
            task_head_specs=_task_specs(data.task_metadata),
            native_input_dim=len(data.native_feature_columns),
            token_dim=8,
            encoder_hidden_dim=16,
            transformer_layers=1,
            transformer_heads=2,
            transformer_ff_dim=16,
            latent_dim=16,
            dataset_embedding_dim=4,
            task_embedding_dim=4,
            concept_dim=0,
            concept_hidden_dim=8,
            task_feature_dim=8,
            output_refine_dim=4,
            recursion_steps=1,
            dropout=0.0,
        )
        model.eval()
        batch = default_collate([data.train[0], data.train[1]])
        batch["modality_mask"] = torch.zeros_like(batch["modality_mask"])
        with torch.no_grad():
            out = model(batch)
        self.assertFalse(torch.isnan(out["base_latent"]).any())
        self.assertFalse(torch.isnan(out["steps"][-1]["predictive_latent"]).any())


if __name__ == "__main__":
    unittest.main()
