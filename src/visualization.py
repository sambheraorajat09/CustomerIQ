"""
visualization.py - Interactive Plotly and Seaborn charts for RFM distributions, clustering diagnostics, and customer exploration.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


PALETTE = [
    "#3B82F6",  # Blue
    "#10B981",  # Green
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
    "#F97316",  # Orange
]


def plot_rfm_distributions(rfm_df: pd.DataFrame) -> go.Figure:
    """
    Plot interactive distribution histograms with marginal box plots for Recency, Frequency, and Monetary.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency (Days)", "Frequency (Orders)", "Monetary Spend ($)"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Histogram(x=rfm_df["Recency"], nbinsx=35, marker_color="#3B82F6", name="Recency"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=rfm_df["Frequency"], nbinsx=35, marker_color="#10B981", name="Frequency"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(x=rfm_df["Monetary"], nbinsx=35, marker_color="#F59E0B", name="Monetary"),
        row=1, col=3,
    )

    fig.update_layout(
        title_text="Raw RFM Feature Distributions (Notice Extreme Positive Skewness)",
        showlegend=False,
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_elbow_and_silhouette(eval_df: pd.DataFrame) -> go.Figure:
    """
    Plot dual-panel diagnostic chart showing the Elbow curve (Inertia) and Silhouette Score across k.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (WCSS / Inertia)", "Silhouette Score Analysis (Separation Quality)"),
        horizontal_spacing=0.10,
    )

    # Panel 1: Inertia
    fig.add_trace(
        go.Scatter(
            x=eval_df["k"],
            y=eval_df["Inertia_WCSS"],
            mode="lines+markers",
            marker=dict(size=8, color="#3B82F6"),
            line=dict(color="#3B82F6", width=2.5),
            name="Inertia (WCSS)",
        ),
        row=1, col=1,
    )

    # Panel 2: Silhouette Score
    fig.add_trace(
        go.Scatter(
            x=eval_df["k"],
            y=eval_df["Silhouette_Score"],
            mode="lines+markers",
            marker=dict(size=8, color="#10B981"),
            line=dict(color="#10B981", width=2.5),
            name="Silhouette Score",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2, dtick=1)
    fig.update_yaxes(title_text="Within-Cluster Sum of Squares", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Coefficient (-1 to +1)", row=1, col=2)

    fig.update_layout(
        title_text="K-Means Cluster Optimization Diagnostics (k=2 to 10)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=50, r=40, t=60, b=50),
    )
    return fig


def plot_clusters_3d(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    segment_names: Optional[Dict[int, str]] = None,
) -> go.Figure:
    """
    Generate an interactive 3D scatter plot of customer clusters in RFM space.
    Uses log-scaled axes for Frequency and Monetary to display clusters clearly.
    """
    df = rfm_df.copy()
    df["Cluster"] = labels
    df["Segment"] = [segment_names.get(c, f"Cluster {c}") if segment_names else f"Cluster {c}" for c in labels]
    df["Log_Frequency"] = np.log1p(df["Frequency"])
    df["Log_Monetary"] = np.log1p(df["Monetary"])

    fig = px.scatter_3d(
        df,
        x="Recency",
        y="Log_Frequency",
        z="Log_Monetary",
        color="Segment",
        hover_data={
            "CustomerID": True,
            "Recency": ":.0f days",
            "Frequency": ":.0f orders",
            "Monetary": ":$.2f",
            "Log_Frequency": False,
            "Log_Monetary": False,
        },
        color_discrete_sequence=PALETTE,
        title="3D Customer Clusters in RFM Space (Frequency & Spend Log-Scaled for Clarity)",
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Recency (Days)",
            yaxis_title="Log(Frequency + 1)",
            zaxis_title="Log(Monetary Spend + 1)",
        ),
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02),
    )
    return fig


def plot_clusters_pca(
    scaled_data: np.ndarray,
    labels: np.ndarray,
    segment_names: Optional[Dict[int, str]] = None,
) -> Tuple[go.Figure, List[float]]:
    """
    Project scaled customer features into 2D via Principal Component Analysis (PCA)
    and plot cluster separation with variance explained.
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_data)
    exp_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    df_pca = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cluster": labels,
        "Segment": [segment_names.get(c, f"Cluster {c}") if segment_names else f"Cluster {c}" for c in labels],
    })

    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color="Segment",
        color_discrete_sequence=PALETTE,
        title=f"2D PCA Projection of Customer Clusters (Total Explained Variance: {sum(exp_var):.1f}%)",
        labels={
            "PC1": f"Principal Component 1 ({exp_var[0]}% var)",
            "PC2": f"Principal Component 2 ({exp_var[1]}% var)",
        },
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    return fig, exp_var


def plot_cluster_radar(
    cluster_profiles: pd.DataFrame,
    segment_names: Optional[Dict[int, str]] = None,
) -> go.Figure:
    """
    Plot radar/spider chart comparing normalized RFM medians across customer segments.
    """
    profiles = cluster_profiles.copy()
    categories = ["Recency (Inverted)", "Frequency", "Monetary Spend", "Customer Share %"]

    # Normalize metrics between 0.1 and 1.0 for balanced radar visualization
    # Invert Recency so higher score = better (more recent)
    r_vals = profiles["Recency_Median"].values
    f_vals = profiles["Frequency_Median"].values
    m_vals = profiles["Monetary_Median"].values
    share_vals = profiles["Customer_Pct"].values

    r_norm = 1.0 - (r_vals - r_vals.min()) / (r_vals.max() - r_vals.min() + 1e-6)
    f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-6)
    m_norm = (m_vals - m_vals.min()) / (m_vals.max() - m_vals.min() + 1e-6)
    s_norm = (share_vals - share_vals.min()) / (share_vals.max() - share_vals.min() + 1e-6)

    fig = go.Figure()

    for i, row in profiles.iterrows():
        cid = int(row["Cluster"])
        name = segment_names.get(cid, f"Cluster {cid}") if segment_names else f"Cluster {cid}"
        color = PALETTE[i % len(PALETTE)]

        values = [r_norm[i], f_norm[i], m_norm[i], s_norm[i]]
        # Close polygon
        values.append(values[0])
        radar_cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_cats,
            fill="toself",
            name=name,
            line=dict(color=color, width=2),
            opacity=0.65,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05], showticklabels=False),
        ),
        title="Segment Relative RFM Radar Profile (Normalized Centroids)",
        template="plotly_dark",
        height=480,
        margin=dict(l=50, r=50, t=60, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    return fig


def plot_cluster_boxplots(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    segment_names: Optional[Dict[int, str]] = None,
) -> go.Figure:
    """
    Plot side-by-side interactive boxplots for Recency, Frequency, and Monetary by segment.
    """
    df = rfm_df.copy()
    df["Segment"] = [segment_names.get(c, f"Cluster {c}") if segment_names else f"Cluster {c}" for c in labels]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Recency (Days)", "Frequency (Order Count, Log Scale)", "Monetary Spend ($, Log Scale)"),
        horizontal_spacing=0.08,
    )

    segments = sorted(df["Segment"].unique())

    for idx, seg in enumerate(segments):
        sdf = df[df["Segment"] == seg]
        color = PALETTE[idx % len(PALETTE)]

        fig.add_trace(
            go.Box(y=sdf["Recency"], name=seg, marker_color=color, showlegend=(idx == 0)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(y=sdf["Frequency"], name=seg, marker_color=color, showlegend=False),
            row=1, col=2,
        )
        fig.add_trace(
            go.Box(y=sdf["Monetary"], name=seg, marker_color=color, showlegend=False),
            row=1, col=3,
        )

    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=3)

    fig.update_layout(
        title_text="RFM Metric Distributions by Customer Segment",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
