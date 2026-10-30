"""
test_clustering.py - Unit tests for K-Means, Hierarchical clustering, and clustering evaluation metrics.
"""

import numpy as np
import pytest
from src.clustering import (
    fit_hierarchical,
    fit_kmeans,
    run_elbow_and_silhouette_sweep,
    validate_clustering_inputs,
)
from src.evaluation import compare_algorithms, compute_cluster_metrics


@pytest.fixture
def sample_scaled_data():
    np.random.seed(42)
    # 3 distinct 2D Gaussian clusters
    c1 = np.random.normal(loc=[-2, -2], scale=0.5, size=(40, 2))
    c2 = np.random.normal(loc=[2, 2], scale=0.5, size=(40, 2))
    c3 = np.random.normal(loc=[-2, 2], scale=0.5, size=(40, 2))
    return np.vstack([c1, c2, c3])


def test_fit_kmeans(sample_scaled_data):
    model, labels = fit_kmeans(sample_scaled_data, n_clusters=3, random_state=42)
    assert len(labels) == len(sample_scaled_data)
    assert len(np.unique(labels)) == 3
    assert hasattr(model, "cluster_centers_")
    assert model.cluster_centers_.shape == (3, 2)


def test_fit_hierarchical(sample_scaled_data):
    model, labels = fit_hierarchical(sample_scaled_data, n_clusters=3, linkage="ward")
    assert len(labels) == len(sample_scaled_data)
    assert len(np.unique(labels)) == 3


def test_compute_cluster_metrics(sample_scaled_data):
    _, labels = fit_kmeans(sample_scaled_data, n_clusters=3, random_state=42)
    metrics = compute_cluster_metrics(sample_scaled_data, labels)

    assert metrics["n_clusters"] == 3
    # Distinct well-separated clusters should have strong silhouette score (>0.4)
    assert metrics["silhouette_score"] > 0.4
    assert metrics["davies_bouldin_index"] > 0
    assert metrics["calinski_harabasz_index"] > 0
    assert metrics["min_cluster_size"] > 0


def test_elbow_and_silhouette_sweep(sample_scaled_data):
    sweep_df = run_elbow_and_silhouette_sweep(sample_scaled_data, k_range=range(2, 6))
    assert len(sweep_df) == 4
    assert "Inertia_WCSS" in sweep_df.columns
    assert "Silhouette_Score" in sweep_df.columns

    # WCSS should generally decrease as k increases
    inertias = sweep_df["Inertia_WCSS"].values
    assert inertias[-1] < inertias[0]


def test_invalid_k_raises():
    data = np.random.randn(20, 2)
    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        validate_clustering_inputs(data, n_clusters=1)

    with pytest.raises(ValueError, match="must be strictly less than the number of samples"):
        validate_clustering_inputs(data, n_clusters=25)


def test_compare_algorithms(sample_scaled_data):
    comp_df = compare_algorithms(sample_scaled_data, k_values=[2, 3])
    assert not comp_df.empty
    assert "K-Means" in comp_df["Algorithm"].values
    assert "Hierarchical (Ward)" in comp_df["Algorithm"].values
    assert "Silhouette Score (↑)" in comp_df.columns
