"""
test_data_loader.py - Unit tests for data loading, validation, and synthetic data generation.
"""

import io
import pytest
import pandas as pd
from src.data_loader import (
    validate_raw_columns,
    load_dataset,
    generate_synthetic_retail_data,
    REQUIRED_COLUMNS,
)


def test_validate_raw_columns_success():
    df = pd.DataFrame(columns=[
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    ])
    is_valid, missing, mapping = validate_raw_columns(df)
    assert is_valid is True
    assert len(missing) == 0
    assert len(mapping) == 8


def test_validate_raw_columns_case_insensitive():
    df = pd.DataFrame(columns=[
        "invoiceno", "stockcode", "description", "quantity",
        "invoicedate", "unitprice", "customerid", "country"
    ])
    is_valid, missing, mapping = validate_raw_columns(df)
    assert is_valid is True
    assert len(missing) == 0


def test_validate_raw_columns_missing():
    df = pd.DataFrame(columns=["InvoiceNo", "Quantity", "UnitPrice"])
    is_valid, missing, _ = validate_raw_columns(df)
    assert is_valid is False
    assert "CustomerID" in missing
    assert "InvoiceDate" in missing


def test_generate_synthetic_retail_data():
    df = generate_synthetic_retail_data(n_customers=50, n_transactions=200, random_state=42)
    assert not df.empty
    assert len(df) == 200
    for col in REQUIRED_COLUMNS:
        assert col in df.columns

    # Verify synthetic metadata flag
    assert df.attrs.get("is_synthetic") is True

    # Verify presence of cancellations
    has_cancelled = df["InvoiceNo"].str.startswith("C").any()
    assert bool(has_cancelled) is True

    # Verify presence of missing CustomerIDs
    has_missing_cids = df["CustomerID"].isna().any()
    assert bool(has_missing_cids) is True


def test_load_dataset_csv_buffer():
    csv_data = io.StringIO(
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "536365,85123A,WHITE HEART,6,2025-10-01 08:26:00,2.55,17850,United Kingdom\n"
    )
    df = load_dataset(csv_data, file_extension=".csv")
    assert len(df) == 1
    assert df["CustomerID"].iloc[0] == 17850


def test_load_dataset_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_retail_file_xyz.csv")
