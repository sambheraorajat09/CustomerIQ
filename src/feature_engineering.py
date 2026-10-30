"""
feature_engineering.py - Customer-level RFM calculation, skewness transformations, and feature standardization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler


def calculate_rfm(
    df: pd.DataFrame,
    reference_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Aggregate sanitized transactional retail data into customer-level RFM features.

    Parameters:
        df: Cleaned transaction DataFrame with canonical columns (InvoiceDate, InvoiceNo, CustomerID, TotalPrice).
        reference_date: Reference snapshot date. If None, defaults to max(InvoiceDate) + 1 day.

    Returns:
        pd.DataFrame: Customer-level RFM DataFrame indexed by CustomerID.
    """
    if df.empty:
        raise ValueError("Cannot calculate RFM on empty DataFrame.")

    required_cols = {"CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for RFM calculation: {missing}")

    # Set snapshot reference date
    if reference_date is None:
        ref_dt = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        ref_dt = pd.to_datetime(reference_date)

    # Customer-level aggregations
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: int((ref_dt - dates.max()).days)),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        TenureDays=("InvoiceDate", lambda dates: int((ref_dt - dates.min()).days)),
        TotalItems=("Quantity", "sum") if "Quantity" in df.columns else ("TotalPrice", "count"),
        UniqueProducts=("StockCode", "nunique") if "StockCode" in df.columns else ("TotalPrice", "count"),
    ).reset_index()

    # Round Monetary and compute Average Order Value
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AvgOrderValue"] = (rfm["Monetary"] / rfm["Frequency"]).round(2)

    # Filter out customers with non-positive Monetary value
    rfm = rfm[rfm["Monetary"] > 0].copy()
    rfm = rfm.reset_index(drop=True)

    # Carry dataset attributes if present
    if hasattr(df, "attrs"):
        rfm.attrs.update(df.attrs)

    return rfm


class RFMTransformer:
    """
    Transforms and scales customer RFM features for distance-based clustering algorithms.
    Handles right-skewness via Log1p or Yeo-Johnson Power Transformation, followed by StandardScaler.
    """

    def __init__(
        self,
        method: str = "log1p",
        features: Optional[List[str]] = None,
    ):
        """
        Parameters:
            method: Transformation method ('log1p' or 'yeo-johnson').
            features: List of feature column names to transform. Defaults to ['Recency', 'Frequency', 'Monetary'].
        """
        self.method = method.lower()
        if features is None:
            self.features = ["Recency", "Frequency", "Monetary"]
        else:
            self.features = list(features)
            if len(self.features) == 0:
                self.features = ["Recency", "Frequency", "Monetary"]
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method="yeo-johnson") if self.method == "yeo-johnson" else None
        self.is_fitted = False

    def fit(self, rfm_df: pd.DataFrame) -> "RFMTransformer":
        """Fit transformer and scaler on customer RFM data."""
        data_to_transform = rfm_df[self.features].values.astype(float)

        if self.method == "log1p":
            transformed = np.log1p(np.maximum(0, data_to_transform))
        elif self.method == "yeo-johnson":
            transformed = self.power_transformer.fit_transform(data_to_transform)
        elif self.method == "none":
            transformed = data_to_transform
        else:
            raise ValueError(f"Unsupported transformation method: '{self.method}'. Use 'log1p', 'yeo-johnson', or 'none'.")

        self.scaler.fit(transformed)
        self.is_fitted = True
        return self

    def transform(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """Apply transformation and scaling to RFM data."""
        if not self.is_fitted:
            raise RuntimeError("RFMTransformer must be fitted before calling transform().")

        data_to_transform = rfm_df[self.features].values.astype(float)

        if self.method == "log1p":
            transformed = np.log1p(np.maximum(0, data_to_transform))
        elif self.method == "yeo-johnson":
            transformed = self.power_transformer.transform(data_to_transform)
        elif self.method == "none":
            transformed = data_to_transform

        scaled = self.scaler.transform(transformed)
        return scaled

    def fit_transform(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in a single call."""
        return self.fit(rfm_df).transform(rfm_df)


def get_feature_skewness(
    rfm_df: pd.DataFrame,
    features: Optional[Union[List[str], pd.Index, Tuple[str, ...], Any]] = None,
) -> Dict[str, float]:
    """
    Calculate Fisher-Pearson skewness coefficient for the given RFM features.
    Supports list, tuple, pandas Index, and None inputs.
    """
    if features is None:
        feat_list = ["Recency", "Frequency", "Monetary"]
    else:
        feat_list = list(features)
        if len(feat_list) == 0:
            feat_list = ["Recency", "Frequency", "Monetary"]

    skew_dict = {}
    for col in feat_list:
        if col in rfm_df.columns and pd.api.types.is_numeric_dtype(rfm_df[col]):
            val = rfm_df[col].skew()
            skew_dict[str(col)] = float(val) if pd.notna(val) else 0.0
    return skew_dict
