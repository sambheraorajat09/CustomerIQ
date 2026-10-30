"""
test_inference.py - Unit tests for model serialization, loading, and inference prediction.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.clustering import fit_kmeans
from src.data_loader import generate_synthetic_retail_data
from src.feature_engineering import RFMTransformer, calculate_rfm
from src.inference import CustomerIQPipeline, load_pipeline, save_pipeline
from src.preprocessing import clean_retail_data
from src.recommendations import generate_cluster_recommendations
from src.segment_interpretation import assign_evidence_based_names, compute_cluster_profiles


@pytest.fixture
def trained_pipeline(tmp_path):
    raw_df = generate_synthetic_retail_data(n_customers=60, n_transactions=400, random_state=42)
    clean_df, _ = clean_retail_data(raw_df)
    rfm_df = calculate_rfm(clean_df)

    transformer = RFMTransformer(method="log1p")
    scaled = transformer.fit_transform(rfm_df)

    kmeans, labels = fit_kmeans(scaled, n_clusters=4, random_state=42)
    profiles = compute_cluster_profiles(rfm_df, labels)
    segment_meta = assign_evidence_based_names(profiles)

    pipeline = CustomerIQPipeline(
        transformer=transformer,
        model=kmeans,
        algorithm="kmeans",
        features=["Recency", "Frequency", "Monetary"],
        segment_meta=segment_meta,
        reference_date="2025-12-05",
        metrics={"silhouette_score": 0.45},
        created_at="2025-12-05 12:00:00",
    )
    return pipeline


def test_pipeline_predict(trained_pipeline):
    new_customers = pd.DataFrame({
        "CustomerID": ["99901", "99902"],
        "Recency": [5, 250],
        "Frequency": [20, 1],
        "Monetary": [3500.0, 45.0],
    })

    pred_df = trained_pipeline.predict(new_customers)

    assert "Cluster" in pred_df.columns
    assert "Segment" in pred_df.columns
    assert "Segment_Tag" in pred_df.columns
    assert len(pred_df) == 2


def test_save_and_load_pipeline(trained_pipeline, tmp_path):
    save_path = tmp_path / "test_model.joblib"
    saved_file = save_pipeline(trained_pipeline, save_path)

    assert saved_file.exists()

    loaded_pipeline = load_pipeline(saved_file)
    assert loaded_pipeline.algorithm == "kmeans"
    assert len(loaded_pipeline.segment_meta) == 4

    # Verify prediction consistency between original and loaded
    sample_df = pd.DataFrame({
        "Recency": [10],
        "Frequency": [5],
        "Monetary": [500.0],
    })
    orig_pred = trained_pipeline.predict(sample_df)
    loaded_pred = loaded_pipeline.predict(sample_df)

    assert orig_pred["Cluster"].iloc[0] == loaded_pred["Cluster"].iloc[0]
    assert orig_pred["Segment"].iloc[0] == loaded_pred["Segment"].iloc[0]


def test_segment_recommendations(trained_pipeline):
    recs = generate_cluster_recommendations(trained_pipeline.segment_meta)
    assert len(recs) == 4
    for cid, rec in recs.items():
        assert "strategic_goal" in rec
        assert "actionable_tactics" in rec
        assert len(rec["actionable_tactics"]) > 0
        assert "ab_test_hypothesis" in rec
