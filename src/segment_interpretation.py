"""
segment_interpretation.py - Evidence-based cluster profiling, segment naming, and methodological transparency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

METHODOLOGICAL_DISCLOSURE = """
### Methodological Disclosures & Operational Scope

1. **Unsupervised Nature**: K-Means and Hierarchical clustering are exploratory, unsupervised methods.
   They partition records based solely on geometric distances in transformed RFM space.
2. **Absence of Demographic Data**: The underlying transaction records do not contain customer age, gender,
   income, occupation, or household structure. No demographic inferences should be fabricated.
3. **Correlation vs Causality**: Past transaction recency, frequency, and monetary spend do not prove causal loyalty
   or guarantee future retention. Customer churn risk is an empirical observation, not a deterministic certainty.
4. **Observation Window & Seasonality**: Analysis is constrained by the historical transaction window. Seasonality
   (e.g., holiday peaks in November-December) strongly affects transaction volume and recency measurements.
5. **Preprocessing Sensitivity**: Log1p transformation and standardization compress extreme monetary outliers,
   which stabilizes centroid estimation but alters Euclidean geometry relative to original currency values.
""".strip()


def compute_cluster_profiles(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute empirical summary statistics (mean, median, customer count, revenue share) for each cluster.

    Parameters:
        rfm_df: Customer-level RFM DataFrame.
        labels: Cluster assignment array.

    Returns:
        pd.DataFrame: Comprehensive cluster summary table.
    """
    df = rfm_df.copy()
    df["Cluster"] = labels

    total_revenue = df["Monetary"].sum()
    total_customers = len(df)

    profile_records = []
    for cluster_id in sorted(np.unique(labels)):
        cdf = df[df["Cluster"] == cluster_id]
        c_size = len(cdf)
        c_spend = cdf["Monetary"].sum()

        profile_records.append({
            "Cluster": cluster_id,
            "Customer_Count": c_size,
            "Customer_Pct": round((c_size / total_customers) * 100, 2),
            "Total_Spend": round(c_spend, 2),
            "Revenue_Pct": round((c_spend / total_revenue) * 100, 2),
            "Recency_Median": round(float(cdf["Recency"].median()), 1),
            "Recency_Mean": round(float(cdf["Recency"].mean()), 1),
            "Frequency_Median": round(float(cdf["Frequency"].median()), 1),
            "Frequency_Mean": round(float(cdf["Frequency"].mean()), 1),
            "Monetary_Median": round(float(cdf["Monetary"].median()), 2),
            "Monetary_Mean": round(float(cdf["Monetary"].mean()), 2),
            "AvgOrderValue_Median": round(float(cdf["AvgOrderValue"].median()), 2) if "AvgOrderValue" in cdf.columns else 0.0,
        })

    return pd.DataFrame(profile_records)


def assign_evidence_based_names(
    cluster_profiles: pd.DataFrame,
) -> Dict[int, Dict[str, Any]]:
    """
    Dynamically assign descriptive, evidence-based segment titles and descriptions
    derived strictly from empirical RFM rank percentiles without demographic assumptions.

    Parameters:
        cluster_profiles: Cluster summary dataframe from compute_cluster_profiles.

    Returns:
        Dict[int, Dict[str, Any]]: Mapping of cluster_id -> {name, tag, summary, evidence}.
    """
    profiles = cluster_profiles.copy()

    # Calculate rank percentiles across clusters for R, F, M
    # Note: For Recency, LOWER value is BETTER (more recent)
    r_med = profiles["Recency_Median"].values
    f_med = profiles["Frequency_Median"].values
    m_med = profiles["Monetary_Median"].values

    r_p50 = np.median(r_med)
    f_p50 = np.median(f_med)
    m_p50 = np.median(m_med)

    segment_meta = {}

    for _, row in profiles.iterrows():
        cid = int(row["Cluster"])
        r = row["Recency_Median"]
        f = row["Frequency_Median"]
        m = row["Monetary_Median"]
        rev_share = row["Revenue_Pct"]

        # Rule-based archetype assignment based on empirical medians
        is_recent = r <= r_p50
        is_frequent = f >= f_p50
        is_high_spend = m >= m_p50

        if is_recent and is_frequent and is_high_spend:
            name = "High-Value Active (Champions)"
            tag = "Champions"
            summary = (
                f"Highly engaged customers with recent purchases (median {r} days ago), "
                f"above-average order counts ({f} orders), and high total spend (${m:,.2f}). "
                f"Generates {rev_share}% of total portfolio revenue."
            )
        elif is_recent and is_frequent and not is_high_spend:
            name = "Frequent Low-Basket Shoppers"
            tag = "Core Active"
            summary = (
                f"Regular transactional frequency ({f} orders) and recent activity ({r} days ago), "
                f"but smaller transaction totals (${m:,.2f}). Steady operational revenue base."
            )
        elif is_recent and not is_frequent:
            name = "Recent / New Buyers"
            tag = "New Shoppers"
            summary = (
                f"Customers who transacted recently ({r} days ago) but currently have low lifetime frequency "
                f"({f} orders). May represent newly acquired accounts in early customer lifecycle."
            )
        elif not is_recent and is_high_spend:
            name = "At-Risk Former High Spenders"
            tag = "At-Risk"
            summary = (
                f"Historically high spenders (${m:,.2f}) who have not placed an order in an extended period "
                f"({r} days). Represents potential revenue loss if dormancy continues."
            )
        elif not is_recent and is_frequent:
            name = "Dormant Regulars"
            tag = "Slipping"
            summary = (
                f"Repeat buyers ({f} past orders) who have lapsed into inactivity ({r} days). "
                f"Familiar with catalog but disengaged recently."
            )
        else:
            name = "Hibernating / Infrequent"
            tag = "Hibernating"
            summary = (
                f"Lapsed customers with long dormancy ({r} days) and low historical engagement "
                f"({f} orders, ${m:,.2f} total spend). Minimal contribution to total revenue ({rev_share}%)."
            )

        segment_meta[cid] = {
            "name": name,
            "tag": tag,
            "summary": summary,
            "evidence": {
                "Recency_Median_Days": r,
                "Frequency_Median_Orders": f,
                "Monetary_Median_Spend": m,
                "Customer_Share_Pct": row["Customer_Pct"],
                "Revenue_Share_Pct": rev_share,
            },
        }

    return segment_meta


def get_labeled_rfm_data(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    segment_meta: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Attach cluster labels and descriptive segment names to the RFM customer dataframe.
    """
    df = rfm_df.copy()
    df["Cluster"] = labels
    df["Segment"] = df["Cluster"].map(lambda c: segment_meta[c]["name"])
    df["Segment_Tag"] = df["Cluster"].map(lambda c: segment_meta[c]["tag"])
    return df
