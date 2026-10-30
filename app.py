"""
app.py - CustomerIQ: AI-Powered Customer Segmentation Interactive Streamlit Dashboard.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import (
    fit_hierarchical,
    fit_kmeans,
    run_elbow_and_silhouette_sweep,
)
from src.data_loader import (
    generate_synthetic_retail_data,
    load_dataset,
    validate_raw_columns,
)
from src.evaluation import (
    compare_algorithms,
    compute_cluster_metrics,
    compute_per_sample_silhouette,
)
from src.feature_engineering import (
    RFMTransformer,
    calculate_rfm,
    get_feature_skewness,
)
from src.inference import (
    CustomerIQPipeline,
    load_pipeline,
    save_pipeline,
)
from src.preprocessing import clean_retail_data
from src.recommendations import (
    RECOMMENDATION_DISCLAIMER,
    generate_cluster_recommendations,
)
from src.segment_interpretation import (
    METHODOLOGICAL_DISCLOSURE,
    assign_evidence_based_names,
    compute_cluster_profiles,
    get_labeled_rfm_data,
)
from src.visualization import (
    plot_cluster_boxplots,
    plot_cluster_radar,
    plot_clusters_3d,
    plot_clusters_pca,
    plot_elbow_and_silhouette,
    plot_rfm_distributions,
)

# Set page configuration
st.set_page_config(
    page_title="CustomerIQ — AI-Powered Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .badge-real {
        background-color: #065F46;
        color: #6EE7B7;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .badge-synth {
        background-color: #78350F;
        color: #FDE68A;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .alert-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38BDF8;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_preprocess(
    source_type: str,
    uploaded_file_bytes: Optional[bytes] = None,
    file_name: Optional[str] = None,
    synthetic_customers: int = 500,
    synthetic_tx: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, bool]:
    """Load raw dataset, clean transactions, and calculate customer-level RFM features."""
    is_synthetic = False

    if source_type == "Demo: Synthetic Data":
        raw_df = generate_synthetic_retail_data(
            n_customers=synthetic_customers,
            n_transactions=synthetic_tx,
            random_state=42,
        )
        is_synthetic = True
    elif source_type == "Local: UCI Online Retail":
        csv_path = PROJECT_ROOT / "data" / "Online_Retail.csv"
        xlsx_path = PROJECT_ROOT / "data" / "Online Retail.xlsx"

        if csv_path.exists():
            raw_df = pd.read_csv(csv_path, low_memory=False)
        elif xlsx_path.exists():
            raw_df = pd.read_excel(xlsx_path, engine="openpyxl")
        else:
            # Fallback to synthetic if neither exists
            raw_df = generate_synthetic_retail_data(n_customers=500, n_transactions=4000, random_state=42)
            is_synthetic = True
    else:  # Upload
        if uploaded_file_bytes is None:
            raw_df = generate_synthetic_retail_data(n_customers=500, n_transactions=4000, random_state=42)
            is_synthetic = True
        else:
            ext = Path(file_name).suffix.lower() if file_name else ".csv"
            raw_df = load_dataset(io.BytesIO(uploaded_file_bytes), file_extension=ext)

    # Validate schema
    is_valid, missing, mapping = validate_raw_columns(raw_df)
    if not is_valid:
        raise ValueError(f"Uploaded dataset is missing required columns: {', '.join(missing)}")

    # Rename canonical
    if mapping:
        raw_df = raw_df.rename(columns=mapping)

    # Clean data
    clean_df, summary = clean_retail_data(raw_df)
    rfm_df = calculate_rfm(clean_df)

    return raw_df, clean_df, rfm_df, summary, is_synthetic


def main():
    # Sidebar: Header & Configuration
    with st.sidebar:
        st.title("🎯 CustomerIQ")
        st.markdown("**AI-Powered Unsupervised Customer Segmentation**")
        st.markdown("---")

        st.subheader("📁 Data Source")
        data_choice = st.radio(
            "Select Input Source",
            options=["Local: UCI Online Retail", "Demo: Synthetic Data", "Upload File (CSV / XLSX)"],
            index=0,
        )

        uploaded_bytes = None
        uploaded_name = None

        if data_choice == "Upload File (CSV / XLSX)":
            uploaded_file = st.file_uploader("Upload Retail Transactions", type=["csv", "xlsx", "xls"])
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_name = uploaded_file.name

        st.markdown("---")
        st.subheader("⚙️ Segmentation Settings")
        n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, step=1)
        transform_method = st.selectbox(
            "Feature Transformation",
            options=["log1p", "yeo-johnson", "none"],
            index=0,
            format_func=lambda x: "Log1p (Recommended for skewed RFM)" if x == "log1p" else ("Yeo-Johnson Power Transform" if x == "yeo-johnson" else "None (Raw Standardization)"),
        )
        random_seed = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)

        st.markdown("---")
        st.subheader("💾 Pipeline Persistence")
        save_pipeline_btn = st.button("Save Current Pipeline", use_container_width=True)
        load_pipeline_btn = st.button("Load Saved Pipeline", use_container_width=True)

    # Main Header
    st.markdown('<div class="main-title">CustomerIQ — Customer Segmentation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">RFM Feature Extraction • K-Means & Hierarchical Clustering • Evidence-Based Action Plans</div>', unsafe_allow_html=True)

    # Load and process data
    with st.spinner("Processing transaction data and aggregating RFM features..."):
        try:
            raw_df, clean_df, rfm_df, summary, is_synthetic = load_and_preprocess(
                source_type=data_choice,
                uploaded_file_bytes=uploaded_bytes,
                file_name=uploaded_name,
            )
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Falling back to synthetic demonstration mode.")
            raw_df, clean_df, rfm_df, summary, is_synthetic = load_and_preprocess(
                source_type="Demo: Synthetic Data"
            )

    # Active dataset banner
    if is_synthetic:
        st.markdown(
            '<div class="badge-synth">⚠️ SYNTHETIC DEMONSTRATION DATA ACTIVE — Simulated Customer Archetypes</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-real">🟢 REAL DATASET ACTIVE — UCI Online Retail Historical Transactions</div>',
            unsafe_allow_html=True,
        )

    # Fit Transformer and Cluster Model
    transformer = RFMTransformer(method=transform_method)
    scaled_data = transformer.fit_transform(rfm_df)

    kmeans_model, kmeans_labels = fit_kmeans(scaled_data, n_clusters=n_clusters, random_state=random_seed)
    cluster_profiles = compute_cluster_profiles(rfm_df, kmeans_labels)
    segment_meta = assign_evidence_based_names(cluster_profiles)
    labeled_rfm = get_labeled_rfm_data(rfm_df, kmeans_labels, segment_meta)
    cluster_metrics = compute_cluster_metrics(scaled_data, kmeans_labels)
    recommendations = generate_cluster_recommendations(segment_meta)

    # Build Current Pipeline Object
    pipeline = CustomerIQPipeline(
        transformer=transformer,
        model=kmeans_model,
        algorithm="kmeans",
        features=["Recency", "Frequency", "Monetary"],
        segment_meta=segment_meta,
        reference_date=str(clean_df["InvoiceDate"].max()),
        metrics=cluster_metrics,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Handle Save / Load Pipeline Actions
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = model_dir / "customeriq_pipeline.joblib"

    if save_pipeline_btn:
        save_path = save_pipeline(pipeline, pipeline_file)
        st.sidebar.success(f"Pipeline successfully saved to: `{save_path.name}`")

    if load_pipeline_btn:
        if pipeline_file.exists():
            loaded_p = load_pipeline(pipeline_file)
            st.sidebar.success(f"Pipeline restored! Created: `{loaded_p.created_at}`, k={len(loaded_p.segment_meta)}")
        else:
            st.sidebar.error("No saved pipeline found at `models/customeriq_pipeline.joblib`. Save one first!")

    # Tabs Interface
    tab_overview, tab_eda, tab_eval, tab_clusters, tab_strategy, tab_lookup, tab_mgmt = st.tabs([
        "🏢 Executive KPIs",
        "📊 RFM & EDA",
        "⚙️ Model Diagnostics",
        "🎯 Cluster Visualizations",
        "📋 Strategic Interpretation",
        "🔍 Customer Explorer & Export",
        "💾 Model & Inference",
    ])

    # -------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW & KPIS
    # -------------------------------------------------------------
    with tab_overview:
        st.subheader("Portfolio Performance & Health Summary")

        # Top KPI Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        total_rev = rfm_df["Monetary"].sum()
        total_cust = len(rfm_df)
        total_orders = rfm_df["Frequency"].sum()
        avg_aov = rfm_df["AvgOrderValue"].mean()
        med_recency = rfm_df["Recency"].median()

        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Revenue</div>
                <div class="metric-value">${total_rev:,.0f}</div>
                <div class="metric-sub">Sum of all valid spend</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Unique Customers</div>
                <div class="metric-value">{total_cust:,}</div>
                <div class="metric-sub">Active in window</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Transactions</div>
                <div class="metric-value">{total_orders:,}</div>
                <div class="metric-sub">Unique invoice events</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Avg Order Value</div>
                <div class="metric-value">${avg_aov:,.2f}</div>
                <div class="metric-sub">Per transaction mean</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Median Recency</div>
                <div class="metric-value">{med_recency:.0f} d</div>
                <div class="metric-sub">Days since last order</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Data Sanitization Waterfall & Methodology
        col_clean, col_method = st.columns([1, 1])

        with col_clean:
            st.markdown("#### 🧹 Data Sanitization & Cleaning Audit")
            st.write(
                "In accordance with rigorous RFM methodology, transactions with missing customer identities, "
                "order cancellations, and invalid quantities/prices were removed to prevent centroid distortion:"
            )
            summary_dict = summary.to_dict()
            summary_df = pd.DataFrame(list(summary_dict.items()), columns=["Audit Check", "Metric Value"])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with col_method:
            st.markdown("#### 📜 Methodological Bounds & Scope")
            st.markdown(METHODOLOGICAL_DISCLOSURE)

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS & RFM
    # -------------------------------------------------------------
    with tab_eda:
        st.subheader("RFM Feature Distributions & Skewness Analysis")
        st.write(
            "Retail customer spend and order frequency typically follow heavy-tailed Pareto distributions. "
            "Because K-Means uses Euclidean distance, un-transformed skewed data causes extreme outliers to dominate."
        )

        fig_dist = plot_rfm_distributions(rfm_df)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Skewness before and after
        skew_raw = get_feature_skewness(rfm_df)
        transformed_df = pd.DataFrame(scaled_data, columns=["Recency_Scaled", "Frequency_Scaled", "Monetary_Scaled"])
        skew_trans = get_feature_skewness(transformed_df, features=transformed_df.columns)

        col_sk1, col_sk2 = st.columns(2)
        with col_sk1:
            st.markdown("##### Raw Feature Skewness (Fisher-Pearson)")
            st.json(skew_raw)
            st.caption("Skewness > 1.0 indicates severe right-skewness violating normality.")

        with col_sk2:
            st.markdown(f"##### Transformed & Standardized Skewness ({transform_method.upper()})")
            st.json(skew_trans)
            st.caption("Transformation stabilizes variance and prevents large spenders from distorting distances.")

        st.markdown("---")
        st.markdown("##### RFM Summary Statistics (Customer Level)")
        st.dataframe(
            rfm_df[["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"]].describe().round(2),
            use_container_width=True,
        )

    # -------------------------------------------------------------
    # TAB 3: MODEL DIAGNOSTICS & EVALUATION
    # -------------------------------------------------------------
    with tab_eval:
        st.subheader("Optimal Cluster Selection & Algorithm Comparison")

        col_diag_btn, col_diag_status = st.columns([1, 3])
        with col_diag_btn:
            run_eval_sweep = st.button("Run Full Optimization Sweep (k=2 to 8)", use_container_width=True)

        if run_eval_sweep or "eval_sweep_df" in st.session_state:
            if run_eval_sweep:
                with st.spinner("Computing Inertia and Silhouette across candidate k values..."):
                    # Use sample if dataset is huge for instant responsiveness on 8GB laptop
                    eval_sample = scaled_data if len(scaled_data) <= 3000 else scaled_data[np.random.choice(len(scaled_data), 3000, replace=False)]
                    st.session_state["eval_sweep_df"] = run_elbow_and_silhouette_sweep(eval_sample, k_range=range(2, 9))
                    st.session_state["algo_comp_df"] = compare_algorithms(eval_sample, k_values=[2, 3, 4, 5, 6])

            eval_df = st.session_state["eval_sweep_df"]
            fig_elbow = plot_elbow_and_silhouette(eval_df)
            st.plotly_chart(fig_elbow, use_container_width=True)

            st.markdown("##### 🔍 K-Means vs Hierarchical (Agglomerative) Benchmark")
            st.dataframe(st.session_state["algo_comp_df"], use_container_width=True, hide_index=True)
            st.info(
                "**Diagnostic Takeaway**: Look for the 'elbow' where WCSS rate-of-decrease tapers off, and choose k "
                "where Silhouette Score is maximized and Davies-Bouldin is minimized. For retail portfolios, k=4 or k=5 "
                "typically strikes the optimal balance between mathematical separation and business interpretability."
            )
        else:
            st.info("Click **Run Full Optimization Sweep** to compute Elbow curves, Silhouette scores, and algorithm comparisons.")

        st.markdown("---")
        st.markdown(f"##### Current Model Performance (k={n_clusters})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette Score (↑)", f"{cluster_metrics['silhouette_score']:.4f}")
        m2.metric("Davies-Bouldin Index (↓)", f"{cluster_metrics['davies_bouldin_index']:.4f}")
        m3.metric("Calinski-Harabasz Index (↑)", f"{cluster_metrics['calinski_harabasz_index']:,.1f}")
        m4.metric("Cluster Size Std Dev", f"{cluster_metrics['size_std_dev']:.1f}")

    # -------------------------------------------------------------
    # TAB 4: CLUSTER VISUALIZATIONS
    # -------------------------------------------------------------
    with tab_clusters:
        st.subheader("Multi-Dimensional Customer Cluster Projections")

        segment_name_map = {cid: meta["name"] for cid, meta in segment_meta.items()}

        col_viz_left, col_viz_right = st.columns([1.3, 1.0])

        with col_viz_left:
            st.markdown("##### 3D Interactive RFM Space")
            fig_3d = plot_clusters_3d(rfm_df, kmeans_labels, segment_name_map)
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_viz_right:
            st.markdown("##### 2D PCA Cluster Projection")
            fig_pca, exp_var = plot_clusters_pca(scaled_data, kmeans_labels, segment_name_map)
            st.plotly_chart(fig_pca, use_container_width=True)

        col_rad, col_box = st.columns([1, 1])
        with col_rad:
            st.markdown("##### Normalized RFM Centroid Radar")
            fig_radar = plot_cluster_radar(cluster_profiles, segment_name_map)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_box:
            st.markdown("##### Segment Feature Boxplots")
            fig_boxes = plot_cluster_boxplots(rfm_df, kmeans_labels, segment_name_map)
            st.plotly_chart(fig_boxes, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: STRATEGIC INTERPRETATION & RECOMMENDATIONS
    # -------------------------------------------------------------
    with tab_strategy:
        st.subheader("Evidence-Based Segment Profiles & Operational Strategy")
        st.markdown(f'<div class="alert-box">{RECOMMENDATION_DISCLAIMER}</div>', unsafe_allow_html=True)

        # Cluster Profiles Table
        st.markdown("##### Cluster Summary Matrix (Medians and Spend Contributions)")
        display_profiles = cluster_profiles.copy()
        display_profiles["Segment_Name"] = display_profiles["Cluster"].map(lambda c: segment_meta[c]["name"])
        cols_order = [
            "Cluster", "Segment_Name", "Customer_Count", "Customer_Pct", "Total_Spend",
            "Revenue_Pct", "Recency_Median", "Frequency_Median", "Monetary_Median", "AvgOrderValue_Median"
        ]
        st.dataframe(display_profiles[cols_order], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### Strategic Playbooks by Segment")

        for cid in sorted(segment_meta.keys()):
            meta = segment_meta[cid]
            rec = recommendations[cid]
            name = meta["name"]
            tag = meta["tag"]
            evidence = meta["evidence"]

            with st.expander(f"📌 Cluster {cid}: {name} ({tag}) — {evidence['Customer_Share_Pct']}% of Customers, {evidence['Revenue_Share_Pct']}% of Revenue", expanded=True):
                col_exp_l, col_exp_r = st.columns([1.2, 1.0])

                with col_exp_l:
                    st.markdown(f"**Empirical Evidence**: {meta['summary']}")
                    st.markdown(f"**Priority Level**: `{rec['priority_level']}`")
                    st.markdown(f"**Strategic Objective**: {rec['strategic_goal']}")
                    st.markdown("**Actionable Marketing Tactics:**")
                    for tactic in rec["actionable_tactics"]:
                        st.markdown(f"- {tactic}")

                with col_exp_r:
                    st.markdown(f"**Pricing & Discount Approach:** {rec['pricing_strategy']}")
                    st.markdown(f"**🧪 A/B Testing Validation Hypothesis:**\n> *{rec['ab_test_hypothesis']}*")
                    st.warning(f"**Operational Risk:** {rec['risk_warning']}")

    # -------------------------------------------------------------
    # TAB 6: CUSTOMER EXPLORER & CSV EXPORT
    # -------------------------------------------------------------
    with tab_lookup:
        st.subheader("Customer Level Explorer & Data Export")
        st.write("Inspect individual customer records, filter by segment or RFM boundaries, and export labeled cohorts.")

        col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 2.0])
        with col_f1:
            segment_options = ["All Segments"] + sorted(list(set(labeled_rfm["Segment"])))
            selected_seg = st.selectbox("Filter by Segment", options=segment_options)

        with col_f2:
            search_cid = st.text_input("Search Customer ID", placeholder="e.g. 17850")

        with col_f3:
            spend_range = st.slider(
                "Filter Monetary Spend ($)",
                min_value=float(rfm_df["Monetary"].min()),
                max_value=float(min(50000.0, rfm_df["Monetary"].max())),
                value=(float(rfm_df["Monetary"].min()), float(min(20000.0, rfm_df["Monetary"].max()))),
            )

        # Apply filters
        filtered_df = labeled_rfm.copy()
        if selected_seg != "All Segments":
            filtered_df = filtered_df[filtered_df["Segment"] == selected_seg]
        if search_cid.strip():
            filtered_df = filtered_df[filtered_df["CustomerID"].astype(str).str.contains(search_cid.strip())]
        filtered_df = filtered_df[
            (filtered_df["Monetary"] >= spend_range[0]) & (filtered_df["Monetary"] <= spend_range[1])
        ]

        st.markdown(f"Showing **{len(filtered_df):,}** customers matching criteria:")

        st.dataframe(
            filtered_df[[
                "CustomerID", "Segment", "Cluster", "Recency", "Frequency",
                "Monetary", "AvgOrderValue", "TenureDays"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        # Export CSV Button
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Segmented Customers (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"customeriq_segments_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -------------------------------------------------------------
    # TAB 7: MODEL INSPECTION & SINGLE RECORD INFERENCE
    # -------------------------------------------------------------
    with tab_mgmt:
        st.subheader("Model Artifacts & Real-Time Inference")

        col_inf_l, col_inf_r = st.columns([1, 1])

        with col_inf_l:
            st.markdown("##### 🧪 Real-Time Single Customer Classifier")
            st.write("Score a new or hypothetical customer against the currently fitted clustering pipeline:")

            inf_r = st.number_input("Customer Recency (Days since last order)", min_value=0, max_value=1000, value=15)
            inf_f = st.number_input("Customer Frequency (Total orders placed)", min_value=1, max_value=500, value=8)
            inf_m = st.number_input("Customer Monetary Spend (Total $ spend)", min_value=1.0, max_value=500000.0, value=1250.0)

            if st.button("Classify Customer", use_container_width=True):
                new_cust_df = pd.DataFrame([{
                    "CustomerID": "NEW_CUST_001",
                    "Recency": inf_r,
                    "Frequency": inf_f,
                    "Monetary": inf_m,
                }])
                pred_res = pipeline.predict(new_cust_df)
                pred_c = pred_res["Cluster"].iloc[0]
                pred_seg = pred_res["Segment"].iloc[0]
                pred_tag = pred_res["Segment_Tag"].iloc[0]

                st.success(f"**Predicted Segment**: `{pred_seg}` (Cluster {pred_c} - {pred_tag})")
                st.info(f"**Recommended Action**: {recommendations[pred_c]['strategic_goal']}")

        with col_inf_r:
            st.markdown("##### 🔍 Pipeline Serialization State")
            st.json({
                "Algorithm": pipeline.algorithm,
                "Number of Clusters": len(pipeline.segment_meta),
                "Transformation Method": pipeline.transformer.method,
                "Features Used": pipeline.features,
                "Reference Date": pipeline.reference_date,
                "Pipeline Version": pipeline.version,
                "Silhouette Score": pipeline.metrics.get("silhouette_score"),
                "Davies-Bouldin Index": pipeline.metrics.get("davies_bouldin_index"),
            })

            st.write(f"Artifact location: `{pipeline_file}`")


if __name__ == "__main__":
    main()
