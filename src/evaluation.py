"""
evaluation.py - Clustering metrics computation, model diagnostics, and algorithm comparison.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


def compute_cluster_metrics(
    scaled_data: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics for a cluster assignment.

    Returns:
        dict: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score,
              Cluster count, Min cluster size, Max cluster size, Size std dev.
    """
    n_samples = len(scaled_data)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        raise ValueError("Cannot evaluate metrics for fewer than 2 clusters.")
    if n_clusters >= n_samples:
        raise ValueError("Number of clusters cannot be >= number of samples.")

    sil = float(silhouette_score(scaled_data, labels))
    db = float(davies_bouldin_score(scaled_data, labels))
    ch = float(calinski_harabasz_score(scaled_data, labels))

    sizes = [int(np.sum(labels == c)) for c in unique_labels]

    return {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_index": round(ch, 2),
        "min_cluster_size": int(min(sizes)),
        "max_cluster_size": int(max(sizes)),
        "size_std_dev": round(float(np.std(sizes)), 2),
        "cluster_sizes": sizes,
    }


def compute_per_sample_silhouette(
    scaled_data: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute individual silhouette coefficients for all data points to evaluate
    cluster separation and identify misassigned or borderline instances.
    """
    sample_sils = silhouette_samples(scaled_data, labels)
    return pd.DataFrame({
        "Cluster": labels,
        "Silhouette_Coefficient": sample_sils,
    })


def compare_algorithms(
    scaled_data: np.ndarray,
    k_values: Optional[List[int]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Directly compare K-Means against Agglomerative Hierarchical Clustering
    across multiple k values using Silhouette, Davies-Bouldin, Calinski-Harabasz, and fit runtime.

    Parameters:
        scaled_data: Standardized feature matrix.
        k_values: List of k values to evaluate (defaults to [3, 4, 5, 6]).
        random_state: Random state for K-Means.

    Returns:
        pd.DataFrame: Comparative performance table.
    """
    if k_values is None:
        k_values = [3, 4, 5, 6]

    records = []

    for k in k_values:
        if k >= len(scaled_data):
            continue

        # 1. K-Means
        km_start = time.perf_counter()
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km_labels = km.fit_predict(scaled_data)
        km_time = time.perf_counter() - km_start

        km_sil = silhouette_score(scaled_data, km_labels)
        km_db = davies_bouldin_score(scaled_data, km_labels)
        km_ch = calinski_harabasz_score(scaled_data, km_labels)

        records.append({
            "Algorithm": "K-Means",
            "k": k,
            "Silhouette Score (↑)": round(float(km_sil), 4),
            "Davies-Bouldin Index (↓)": round(float(km_db), 4),
            "Calinski-Harabasz Index (↑)": round(float(km_ch), 2),
            "Fit Time (ms)": round(km_time * 1000, 2),
        })

        # 2. Hierarchical (Agglomerative)
        h_start = time.perf_counter()
        agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        agg_labels = agg.fit_predict(scaled_data)
        h_time = time.perf_counter() - h_start

        agg_sil = silhouette_score(scaled_data, agg_labels)
        agg_db = davies_bouldin_score(scaled_data, agg_labels)
        agg_ch = calinski_harabasz_score(scaled_data, agg_labels)

        records.append({
            "Algorithm": "Hierarchical (Ward)",
            "k": k,
            "Silhouette Score (↑)": round(float(agg_sil), 4),
            "Davies-Bouldin Index (↓)": round(float(agg_db), 4),
            "Calinski-Harabasz Index (↑)": round(float(agg_ch), 2),
            "Fit Time (ms)": round(h_time * 1000, 2),
        })

    return pd.DataFrame(records)
