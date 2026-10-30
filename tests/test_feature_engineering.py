"""
test_feature_engineering.py - Unit tests for data cleaning, RFM aggregation, and feature transformations.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_loader import generate_synthetic_retail_data
from src.feature_engineering import (
    RFMTransformer,
    calculate_rfm,
    get_feature_skewness,
)
from src.preprocessing import clean_retail_data


@pytest.fixture
def sample_raw_data():
    return generate_synthetic_retail_data(n_customers=60, n_transactions=400, random_state=42)


def test_clean_retail_data(sample_raw_data):
    clean_df, summary = clean_retail_data(sample_raw_data)

    assert not clean_df.empty
    assert summary.raw_rows == len(sample_raw_data)
    assert summary.cleaned_rows == len(clean_df)

    # All CustomerIDs must be non-null and string
    assert clean_df["CustomerID"].isna().sum() == 0
    assert pd.api.types.is_string_dtype(clean_df["CustomerID"])

    # No cancelled invoices
    assert not clean_df["InvoiceNo"].str.upper().str.startswith("C").any()

    # All quantities and prices positive
    assert (clean_df["Quantity"] > 0).all()
    assert (clean_df["UnitPrice"] > 0).all()

    # TotalPrice calculated
    assert "TotalPrice" in clean_df.columns
    assert (clean_df["TotalPrice"] > 0).all()


def test_calculate_rfm(sample_raw_data):
    clean_df, _ = clean_retail_data(sample_raw_data)
    ref_date = "2025-12-05"
    rfm = calculate_rfm(clean_df, reference_date=ref_date)

    assert not rfm.empty
    assert "CustomerID" in rfm.columns
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert "AvgOrderValue" in rfm.columns

    # Check non-negative recency and positive spend
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()


def test_rfm_transformer_fit_transform(sample_raw_data):
    clean_df, _ = clean_retail_data(sample_raw_data)
    rfm = calculate_rfm(clean_df)

    transformer = RFMTransformer(method="log1p")
    scaled = transformer.fit_transform(rfm)

    assert scaled.shape == (len(rfm), 3)
    # Check approximately standardized (mean ~0, std ~1)
    np.testing.assert_allclose(np.mean(scaled, axis=0), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.std(scaled, axis=0), 1.0, atol=1e-2)


def test_rfm_transformer_unfitted_raises():
    transformer = RFMTransformer(method="log1p")
    dummy_rfm = pd.DataFrame({
        "Recency": [10, 20],
        "Frequency": [2, 5],
        "Monetary": [100.0, 500.0]
    })
    with pytest.raises(RuntimeError):
        transformer.transform(dummy_rfm)


def test_get_feature_skewness(sample_raw_data):
    clean_df, _ = clean_retail_data(sample_raw_data)
    rfm = calculate_rfm(clean_df)

    # Test None input (defaults to ["Recency", "Frequency", "Monetary"])
    skew_none = get_feature_skewness(rfm, features=None)
    assert "Recency" in skew_none
    assert "Frequency" in skew_none
    assert "Monetary" in skew_none
    assert isinstance(skew_none["Monetary"], float)

    # Test pandas Index input (regression test for ambiguous Index truth value)
    skew_index = get_feature_skewness(rfm, features=rfm.columns[:3])
    assert len(skew_index) <= 3
    for k in skew_index:
        assert isinstance(skew_index[k], float)

    # Test list input
    skew_list = get_feature_skewness(rfm, features=["Recency", "Monetary"])
    assert set(skew_list.keys()) == {"Recency", "Monetary"}

    # Test tuple input
    skew_tuple = get_feature_skewness(rfm, features=("Frequency", "Monetary"))
    assert set(skew_tuple.keys()) == {"Frequency", "Monetary"}

    # Test empty list (should fall back to default)
    skew_empty = get_feature_skewness(rfm, features=[])
    assert "Recency" in skew_empty
    assert "Frequency" in skew_empty
    assert "Monetary" in skew_empty


def test_rfm_transformer_features_input_types(sample_raw_data):
    clean_df, _ = clean_retail_data(sample_raw_data)
    rfm = calculate_rfm(clean_df)

    # Test with pandas Index
    cols_index = pd.Index(["Recency", "Frequency", "Monetary"])
    transformer_index = RFMTransformer(method="log1p", features=cols_index)
    scaled_index = transformer_index.fit_transform(rfm)
    assert scaled_index.shape == (len(rfm), 3)

    # Test with tuple
    cols_tuple = ("Recency", "Frequency", "Monetary")
    transformer_tuple = RFMTransformer(method="log1p", features=cols_tuple)
    scaled_tuple = transformer_tuple.fit_transform(rfm)
    assert scaled_tuple.shape == (len(rfm), 3)

    # Test with None
    transformer_none = RFMTransformer(method="log1p", features=None)
    assert transformer_none.features == ["Recency", "Frequency", "Monetary"]

    # Test with empty list
    transformer_empty = RFMTransformer(method="log1p", features=[])
    assert transformer_empty.features == ["Recency", "Frequency", "Monetary"]

