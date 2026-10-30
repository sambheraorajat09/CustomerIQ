"""
inference.py - Serialization, model pipeline persistence, and batch/single-record inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.feature_engineering import RFMTransformer


@dataclass
class CustomerIQPipeline:
    """Serialized clustering pipeline container containing preprocessing, model, and metadata."""
    transformer: RFMTransformer
    model: Any
    algorithm: str
    features: List[str]
    segment_meta: Dict[int, Dict[str, Any]]
    reference_date: str
    metrics: Dict[str, Any]
    created_at: str
    version: str = "1.0.0"

    def predict(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign customer records to segments using the fitted preprocessing and clustering components.

        Parameters:
            rfm_df: Customer DataFrame containing required features (Recency, Frequency, Monetary).

        Returns:
            pd.DataFrame: Augmented DataFrame with Cluster, Segment, and Segment_Tag.
        """
        missing = [f for f in self.features if f not in rfm_df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required features for inference: {missing}")

        df_out = rfm_df.copy()

        # Transform and scale
        scaled = self.transformer.transform(df_out)

        # Predict cluster
        if hasattr(self.model, "predict"):
            cluster_preds = self.model.predict(scaled)
        elif hasattr(self.model, "cluster_centers_"):
            # Fallback Euclidean nearest centroid
            dists = np.linalg.norm(scaled[:, np.newaxis] - self.model.cluster_centers_, axis=2)
            cluster_preds = np.argmin(dists, axis=1)
        else:
            raise NotImplementedError(
                f"Model type '{type(self.model)}' does not support out-of-sample prediction."
            )

        df_out["Cluster"] = cluster_preds
        df_out["Segment"] = [
            self.segment_meta.get(c, {}).get("name", f"Cluster {c}") for c in cluster_preds
        ]
        df_out["Segment_Tag"] = [
            self.segment_meta.get(c, {}).get("tag", f"Cluster {c}") for c in cluster_preds
        ]

        return df_out


def save_pipeline(
    pipeline: CustomerIQPipeline,
    filepath: Union[str, Path],
) -> Path:
    """
    Safely serialize and save CustomerIQPipeline using joblib.

    Parameters:
        pipeline: Instantiated CustomerIQPipeline.
        filepath: Destination file path (typically inside models/ directory).

    Returns:
        Path: Absolute path to the saved pipeline artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() != ".joblib":
        path = path.with_suffix(".joblib")

    joblib.dump(pipeline, str(path))
    return path.resolve()


def load_pipeline(filepath: Union[str, Path]) -> CustomerIQPipeline:
    """
    Safely deserialize and load CustomerIQPipeline artifact.

    Parameters:
        filepath: Path to .joblib artifact.

    Returns:
        CustomerIQPipeline: Restored pipeline object ready for inference.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    loaded = joblib.load(str(path))
    if not isinstance(loaded, CustomerIQPipeline):
        raise TypeError(
            f"Loaded object is of type {type(loaded)}, expected CustomerIQPipeline."
        )

    return loaded
