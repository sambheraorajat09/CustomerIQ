"""
data_loader.py - Data ingestion, schema validation, synthetic data generator, and dataset downloader.
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def validate_raw_columns(df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, str]]:
    """
    Validate whether the input DataFrame contains all required columns for retail analysis.
    Supports case-insensitive column matching.

    Returns:
        is_valid (bool): True if all required columns are present.
        missing_columns (list): List of missing required column names.
        col_mapping (dict): Mapping from raw column names to canonical names.
    """
    col_map = {}
    normalized_input_cols = {col.strip().lower(): col for col in df.columns}

    missing = []
    for req in REQUIRED_COLUMNS:
        req_norm = req.lower()
        if req_norm in normalized_input_cols:
            col_map[normalized_input_cols[req_norm]] = req
        else:
            missing.append(req)

    is_valid = len(missing) == 0
    return is_valid, missing, col_map


def load_dataset(
    file_path_or_buffer: Union[str, Path, io.BytesIO, io.StringIO],
    file_extension: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load retail dataset from CSV or Excel file / buffer.

    Parameters:
        file_path_or_buffer: File path string, Path object, or file buffer.
        file_extension: Optional extension override ('.csv' or '.xlsx').

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        path_str = str(file_path_or_buffer)
        ext = Path(path_str).suffix.lower() if not file_extension else file_extension.lower()

        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Dataset file not found at: {path_str}")

        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path_str, engine="openpyxl")
        elif ext == ".csv":
            # Attempt multiple common encodings
            for enc in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
                try:
                    return pd.read_csv(path_str, encoding=enc, low_memory=False)
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Failed to decode CSV file {path_str} with common encodings.")
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .xlsx / .xls.")

    else:
        # Buffer input (e.g. Streamlit UploadedFile)
        ext = file_extension.lower() if file_extension else ".csv"
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path_or_buffer, engine="openpyxl")
        else:
            try:
                return pd.read_csv(file_path_or_buffer, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                if hasattr(file_path_or_buffer, "seek"):
                    file_path_or_buffer.seek(0)
                return pd.read_csv(file_path_or_buffer, encoding="latin1", low_memory=False)


def generate_synthetic_retail_data(
    n_customers: int = 500,
    n_transactions: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic retail transactional data mimicking UCI Online Retail.
    Produces realistic clusters (Champions, Loyal, At-Risk, Occasional, Hibernating),
    including cancellations ('C' prefix with negative quantities) and unassigned CustomerIDs.

    Parameters:
        n_customers (int): Number of unique registered customer IDs to generate.
        n_transactions (int): Total number of transaction rows to generate.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic dataset with canonical retail schema.
    """
    rng = np.random.default_rng(random_state)
    end_date = datetime(2025, 12, 1, 17, 0, 0)
    start_date = end_date - timedelta(days=365)

    customer_ids = [10000 + i for i in range(n_customers)]

    # Assign archetypes to customers to create realistic behavioral clusters
    # Archetype distribution:
    # 1. Champions (10%): low recency (1-30 days), high frequency (15-30 orders), high monetary
    # 2. Loyal Regulars (25%): low-mid recency (10-90 days), mid-high frequency (8-15 orders), mid spend
    # 3. New / Promising (20%): very recent (1-45 days), low frequency (1-3 orders), moderate spend
    # 4. At Risk (20%): dormant (90-250 days), previously moderate/high frequency (5-12 orders)
    # 5. Hibernating / Lost (25%): very dormant (180-365 days), low frequency (1-2 orders), low spend
    archetype_weights = [0.10, 0.25, 0.20, 0.20, 0.25]
    archetypes = rng.choice(
        ["champions", "loyal", "new", "at_risk", "hibernating"],
        size=n_customers,
        p=archetype_weights,
    )
    customer_archetype = dict(zip(customer_ids, archetypes))

    product_catalog = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("71053", "WHITE METAL LANTERN", 3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
        ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
        ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
        ("22633", "HAND WARMER UNION JACK", 1.85),
        ("22632", "HAND WARMER RED RETROSPOT", 1.85),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.55),
        ("22960", "JAM MAKING SET WITH JARS", 3.75),
        ("22423", "REGENCY CAKESTAND 3 TIER", 10.95),
        ("47566", "PARTY BUNTING", 4.65),
        ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
        ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
        ("20727", "LUNCH BAG BLACK SKULL.", 1.65),
        ("20728", "LUNCH BAG CARS BLUE", 1.65),
        ("22382", "LUNCH BAG SPACEBOY DESIGN", 1.65),
        ("22961", "JAM MAKING SET PRINTED", 1.45),
    ]

    countries = ["United Kingdom"] * 85 + ["Germany"] * 5 + ["France"] * 4 + ["EIRE"] * 3 + ["Spain"] * 3

    rows = []
    invoice_seq = 536365

    # Generate customer transactions based on archetypes
    for cid in customer_ids:
        arch = customer_archetype[cid]
        c_country = rng.choice(countries)

        if arch == "champions":
            n_invoices = rng.integers(12, 28)
            recency_days = rng.integers(1, 25)
            qty_mult = rng.uniform(2.0, 5.0)
        elif arch == "loyal":
            n_invoices = rng.integers(6, 14)
            recency_days = rng.integers(15, 75)
            qty_mult = rng.uniform(1.2, 2.5)
        elif arch == "new":
            n_invoices = rng.integers(1, 3)
            recency_days = rng.integers(2, 40)
            qty_mult = rng.uniform(0.8, 1.8)
        elif arch == "at_risk":
            n_invoices = rng.integers(4, 10)
            recency_days = rng.integers(100, 240)
            qty_mult = rng.uniform(1.0, 2.5)
        else:  # hibernating
            n_invoices = rng.integers(1, 2)
            recency_days = rng.integers(180, 360)
            qty_mult = rng.uniform(0.5, 1.2)

        last_purchase_date = end_date - timedelta(days=int(recency_days))

        for inv_i in range(n_invoices):
            inv_no = str(invoice_seq)
            invoice_seq += 1

            # Spread invoices backwards from the last purchase date
            if inv_i == 0:
                inv_date = last_purchase_date
            else:
                days_prior = rng.integers(5, max(6, int(360 - recency_days)))
                inv_date = last_purchase_date - timedelta(days=int(days_prior))
                if inv_date < start_date:
                    inv_date = start_date + timedelta(days=int(rng.integers(1, 60)))

            # Items per invoice
            n_items = rng.integers(1, 6)
            chosen_items = rng.choice(len(product_catalog), size=n_items, replace=False)

            for idx in chosen_items:
                stock_code, desc, base_price = product_catalog[idx]
                qty = max(1, int(rng.geometric(p=0.25) * qty_mult))
                unit_price = round(base_price * rng.uniform(0.95, 1.05), 2)

                rows.append({
                    "InvoiceNo": inv_no,
                    "StockCode": stock_code,
                    "Description": desc,
                    "Quantity": qty,
                    "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": unit_price,
                    "CustomerID": cid,
                    "Country": c_country,
                })

            # 5% chance of cancellation order
            if rng.random() < 0.05:
                cancel_item = product_catalog[chosen_items[0]]
                rows.append({
                    "InvoiceNo": f"C{inv_no}",
                    "StockCode": cancel_item[0],
                    "Description": f"Cancelled: {cancel_item[1]}",
                    "Quantity": -max(1, int(rng.integers(1, 4))),
                    "InvoiceDate": (inv_date + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice": cancel_item[2],
                    "CustomerID": cid,
                    "Country": c_country,
                })

    # Add ~10% guest / missing CustomerID transactions (mimicking real dataset)
    n_guest_rows = int(len(rows) * 0.10)
    for _ in range(n_guest_rows):
        stock_code, desc, base_price = product_catalog[rng.integers(0, len(product_catalog))]
        g_inv_no = str(invoice_seq)
        invoice_seq += 1
        g_date = start_date + timedelta(days=int(rng.integers(1, 350)), hours=int(rng.integers(8, 18)))
        rows.append({
            "InvoiceNo": g_inv_no,
            "StockCode": stock_code,
            "Description": desc,
            "Quantity": int(rng.integers(1, 8)),
            "InvoiceDate": g_date.strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": base_price,
            "CustomerID": np.nan,
            "Country": rng.choice(countries),
        })

    df = pd.DataFrame(rows)
    # Shuffle transactions
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Trim to approximate target size if needed, but keep representative
    if len(df) > n_transactions:
        df = df.iloc[:n_transactions].copy()

    # Tag dataframe with metadata
    df.attrs["is_synthetic"] = True
    df.attrs["data_source"] = "Synthetic Generator (CustomerIQ Demo)"
    return df


def download_uci_online_retail(save_dir: Union[str, Path] = "data") -> Optional[Path]:
    """
    Attempt to download and extract the UCI Online Retail dataset.
    Returns path to downloaded file or None if download failed.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    target_xlsx = save_path / "online_retail.xlsx"

    if target_xlsx.exists():
        return target_xlsx

    zip_path = save_path / "online_retail.zip"
    headers = {"User-Agent": "Mozilla/5.0 CustomerIQ/1.0"}

    try:
        response = requests.get(UCI_DATASET_URL, headers=headers, timeout=25, stream=True)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)

            # Extract zip
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(save_path)

            if zip_path.exists():
                zip_path.unlink()

            for extracted_file in save_path.glob("*Online*Retail*"):
                if extracted_file.suffix.lower() in [".xlsx", ".xls", ".csv"]:
                    return extracted_file

            return target_xlsx if target_xlsx.exists() else None
        else:
            return None
    except Exception:
        # Gracefully handle connection errors or timeout
        return None
