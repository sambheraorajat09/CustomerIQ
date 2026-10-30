"""
preprocessing.py - Retail data sanitization, outlier filtering, and transaction-level cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleaningSummary:
    """Summary of operations performed during data cleaning."""
    raw_rows: int
    raw_unique_customers: int
    dropped_missing_customer_id: int
    dropped_cancelled_invoices: int
    dropped_invalid_quantity: int
    dropped_invalid_price: int
    dropped_duplicates: int
    cleaned_rows: int
    cleaned_unique_customers: int
    earliest_date: Optional[str]
    latest_date: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Raw Rows": self.raw_rows,
            "Raw Unique Customers": self.raw_unique_customers,
            "Dropped (Missing CustomerID)": self.dropped_missing_customer_id,
            "Dropped (Cancelled Invoices 'C')": self.dropped_cancelled_invoices,
            "Dropped (Quantity <= 0)": self.dropped_invalid_quantity,
            "Dropped (UnitPrice <= 0)": self.dropped_invalid_price,
            "Dropped (Duplicates)": self.dropped_duplicates,
            "Cleaned Valid Rows": self.cleaned_rows,
            "Cleaned Unique Customers": self.cleaned_unique_customers,
            "Observation Window": f"{self.earliest_date} to {self.latest_date}",
        }


def clean_retail_data(
    df: pd.DataFrame,
    drop_missing_customer_id: bool = True,
    remove_cancelled: bool = True,
    remove_invalid_quantity: bool = True,
    remove_invalid_price: bool = True,
    remove_duplicates: bool = True,
) -> Tuple[pd.DataFrame, CleaningSummary]:
    """
    Clean raw retail transaction dataframe according to rigorous RFM segmentation standards.

    Parameters:
        df: Raw retail DataFrame with canonical columns.
        drop_missing_customer_id: If True, drops rows without CustomerID.
        remove_cancelled: If True, drops cancellations (InvoiceNo starting with 'C' or 'c').
        remove_invalid_quantity: If True, drops Quantity <= 0.
        remove_invalid_price: If True, drops UnitPrice <= 0.
        remove_duplicates: If True, drops identical duplicate transaction rows.

    Returns:
        Tuple[pd.DataFrame, CleaningSummary]: Sanitized DataFrame and summary metrics.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot clean empty dataset.")

    # Work on a copy
    df_clean = df.copy()

    # Track metrics
    raw_rows = len(df_clean)
    raw_unique_cust = df_clean["CustomerID"].dropna().nunique() if "CustomerID" in df_clean.columns else 0

    dropped_missing_cid = 0
    dropped_cancelled = 0
    dropped_invalid_qty = 0
    dropped_invalid_prc = 0
    dropped_dupes = 0

    # 1. Handle missing CustomerID
    if drop_missing_customer_id and "CustomerID" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=["CustomerID"])
        dropped_missing_cid = before - len(df_clean)

    # 2. Standardize CustomerID as integer string
    if "CustomerID" in df_clean.columns:
        df_clean["CustomerID"] = (
            df_clean["CustomerID"]
            .astype(float)
            .astype(int)
            .astype(str)
        )

    # 3. Handle cancellations (InvoiceNo starting with 'C' or 'c')
    if remove_cancelled and "InvoiceNo" in df_clean.columns:
        before = len(df_clean)
        # Convert InvoiceNo to string for safe checking
        is_cancelled = df_clean["InvoiceNo"].astype(str).str.upper().str.startswith("C")
        df_clean = df_clean[~is_cancelled]
        dropped_cancelled = before - len(df_clean)

    # 4. Handle non-positive quantities (<= 0)
    if remove_invalid_quantity and "Quantity" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["Quantity"] > 0]
        dropped_invalid_qty = before - len(df_clean)

    # 5. Handle non-positive unit prices (<= 0)
    if remove_invalid_price and "UnitPrice" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean["UnitPrice"] > 0]
        dropped_invalid_prc = before - len(df_clean)

    # 6. Remove duplicate transactions
    if remove_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped_dupes = before - len(df_clean)

    # 7. Convert InvoiceDate to datetime
    if "InvoiceDate" in df_clean.columns:
        df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce")
        df_clean = df_clean.dropna(subset=["InvoiceDate"])

    # 8. Calculate TotalPrice (spend per line item)
    if "Quantity" in df_clean.columns and "UnitPrice" in df_clean.columns:
        df_clean["TotalPrice"] = (df_clean["Quantity"] * df_clean["UnitPrice"]).round(2)

    df_clean = df_clean.reset_index(drop=True)

    cleaned_rows = len(df_clean)
    cleaned_unique_cust = df_clean["CustomerID"].nunique() if "CustomerID" in df_clean.columns else 0

    earliest_date = df_clean["InvoiceDate"].min().strftime("%Y-%m-%d") if not df_clean.empty else None
    latest_date = df_clean["InvoiceDate"].max().strftime("%Y-%m-%d") if not df_clean.empty else None

    summary = CleaningSummary(
        raw_rows=raw_rows,
        raw_unique_customers=raw_unique_cust,
        dropped_missing_customer_id=dropped_missing_cid,
        dropped_cancelled_invoices=dropped_cancelled,
        dropped_invalid_quantity=dropped_invalid_qty,
        dropped_invalid_price=dropped_invalid_prc,
        dropped_duplicates=dropped_dupes,
        cleaned_rows=cleaned_rows,
        cleaned_unique_customers=cleaned_unique_cust,
        earliest_date=earliest_date,
        latest_date=latest_date,
    )

    # Preserve any dataset attributes (e.g. is_synthetic)
    if hasattr(df, "attrs"):
        df_clean.attrs.update(df.attrs)

    return df_clean, summary
