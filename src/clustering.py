"""
clustering.py - K-Means and Hierarchical (Agglomerative) clustering models and optimization routines.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Avoid Windows MKL thread leak warning in KMeans
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def validate_clustering_inputs(
    scaled_data: np.ndarray,
    n_clusters: int,
) -> None:
    """Validate clustering data shape and cluster count parameters."""
    if scaled_data is None or len(scaled_data) == 0:
        raise ValueError("Scaled data array is empty or None.")

    n_samples = scaled_data.shape[0]
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
    if n_clusters >= n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be strictly less than the number of samples ({n_samples})."
        )


def fit_kmeans(
    scaled_data: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit K-Means clustering algorithm on scaled customer features.

    Parameters:
        scaled_data: Preprocessed and standardized feature matrix.
        n_clusters: Number of clusters (k).
        random_state: Random state for centroid initialization reproducibility.
        n_init: Number of time the k-means algorithm will be run with different centroid seeds.
        max_iter: Maximum number of iterations for a single run.

    Returns:
        Tuple[KMeans, np.ndarray]: Fitted KMeans estimator and cluster label assignments.
    """
    validate_clustering_inputs(scaled_data, n_clusters)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(scaled_data)
    return kmeans, labels


def fit_hierarchical(
    scaled_data: np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward",
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fit Agglomerative Hierarchical Clustering on scaled customer features.

    Parameters:
        scaled_data: Preprocessed and standardized feature matrix.
        n_clusters: Target number of customer clusters.
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single').
                 'ward' minimizes the total within-cluster variance.

    Returns:
        Tuple[AgglomerativeClustering, np.ndarray]: Fitted model and cluster label assignments.
    """
    validate_clustering_inputs(scaled_data, n_clusters)

    # Ward linkage requires Euclidean metric
    metric = "euclidean" if linkage == "ward" else "euclidean"

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
    )
    labels = model.fit_predict(scaled_data)
    return model, labels


def run_elbow_and_silhouette_sweep(
    scaled_data: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate K-Means across a range of k values to determine the optimal number of clusters.
    Calculates Inertia (WCSS), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.

    Returns:
        pd.DataFrame: Table of evaluation metrics indexed by k.
    """
    n_samples = len(scaled_data)
    results = []

    for k in k_range:
        if k >= n_samples:
            break

        start_t = time.perf_counter()
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(scaled_data)
        elapsed = time.perf_counter() - start_t

        inertia = float(kmeans.inertia_)
        sil = float(silhouette_score(scaled_data, labels))
        db = float(davies_bouldin_score(scaled_data, labels))
        ch = float(calinski_harabasz_score(scaled_data, labels))

        results.append({
            "k": k,
            "Inertia_WCSS": round(inertia, 2),
            "Silhouette_Score": round(sil, 4),
            "Davies_Bouldin_Index": round(db, 4),
            "Calinski_Harabasz_Index": round(ch, 2),
            "Fit_Time_Seconds": round(elapsed, 4),
        })

    return pd.DataFrame(results)
