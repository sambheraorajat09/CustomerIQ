# CustomerIQ: Viva & Technical Defense Guide

This document contains in-depth viva questions, theoretical foundations, mathematical explanations, and defense strategies for academic or industry project evaluations.

---

### Q1: Why did you choose RFM (Recency, Frequency, Monetary) for customer segmentation?
**Answer**:
RFM is an established, quantitative framework derived from database marketing. Rather than relying on subjective traits, RFM directly models fundamental behavioral dynamics:
- **Recency** measures engagement freshness; recent purchasers have high brand salience and higher response probability.
- **Frequency** captures habituation and loyalty; repeat buyers exhibit lower acquisition overhead and higher lifetime retention.
- **Monetary** measures gross revenue contribution; it quantifies the commercial value of each customer.
By compressing complex line-item transactional logs into these three continuous features, we obtain an orthogonal, high-signal representation suitable for unsupervised distance algorithms.

---

### Q2: Why is data transformation (such as `log1p`) necessary before applying K-Means clustering to RFM features?
**Answer**:
K-Means uses Euclidean distance as its loss function objective ($J = \sum_{k=1}^K \sum_{x \in C_k} \| x - \mu_k \|^2$). Euclidean distance makes two implicit structural assumptions:
1. Features have comparable scales.
2. Clusters are isotropic (spherical) with roughly equal variances.

In transactional retail data, Frequency and Monetary follow power-law / Pareto distributions with long right tails (skewness often exceeds 15-20). Without transformation:
- Extreme spenders act as severe leverage points, pulling cluster centroids toward themselves.
- The majority of standard customers get compressed into a single uninformative cluster.
Applying $\ln(x + 1)$ compresses the dynamic range, pulls in long right tails, stabilizes variance, and ensures that when standardized with `StandardScaler`, all dimensions contribute proportionally to distance metrics.

---

### Q3: How did you select the optimal number of clusters ($k$)?
**Answer**:
We utilized a multi-criteria diagnostic strategy rather than relying on a single heuristic:
1. **Elbow Method (Inertia/WCSS)**: Plotting WCSS against $k$ identifies the point of diminishing returns (the "elbow") where adding another centroid ceases to explain substantially more within-cluster variance.
2. **Silhouette Coefficient Analysis**: Measures how well-separated clusters are ($s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$). We evaluated both the global average silhouette score and per-sample silhouette plots to verify that clusters have balanced thicknesses and minimal negative coefficients (which indicate misassigned border samples).
3. **Davies-Bouldin & Calinski-Harabasz Indices**: Davies-Bouldin assesses cluster separation relative to dispersion (lower is better); Calinski-Harabasz measures variance ratio (higher is better).
4. **Business Actionability**: A mathematically optimal $k=12$ is impractical for marketing execution. A choice of $k=4$ or $k=5$ provides high silhouette separation while mapping directly onto distinct operational playbooks (Champions, Loyal Regulars, New Buyers, At-Risk Spenders, Hibernating).

---

### Q4: Compare K-Means and Hierarchical Clustering. What are their relative trade-offs?
**Answer**:
- **Computational Complexity**:
  - K-Means runs in $\mathcal{O}(k \cdot N \cdot d \cdot I)$, where $N$ is sample size, $d$ is dimension, $k$ is clusters, and $I$ is iterations. It scales linearly with $N$ and executes in milliseconds.
  - Agglomerative Hierarchical Clustering runs in $\mathcal{O}(N^2 \log N)$ or $\mathcal{O}(N^3)$ and requires an $\mathcal{O}(N^2)$ distance matrix. For large customer bases ($N > 50,000$), Hierarchical clustering causes memory exhaustion on typical 8 GB laptops.
- **Determinism & Reproducibility**:
  - K-Means depends on centroid initialization (mitigated using `k-means++` and multiple `n_init=10` runs).
  - Agglomerative clustering is deterministic for a given linkage criterion (e.g. Ward's linkage).
- **Out-of-Sample Inference**:
  - K-Means provides explicit centroids ($\mu_k$), allowing instant $\mathcal{O}(k \cdot d)$ classification of new incoming customer records via minimum Euclidean distance.
  - Hierarchical clustering does not learn parametric centroids and natively requires re-clustering the entire dataset when new observations arrive.

---

### Q5: How do you handle cancellations (invoices starting with 'C') and returns in your preprocessing pipeline?
**Answer**:
In retail datasets, cancellations have negative quantities and credit invoices. If naively retained:
- Summing total spend without filtering could produce zero or negative lifetime monetary values.
- Cancellation events could falsely inflate order frequency if counted as separate purchase occasions.

Our pipeline separates transactional integrity into a multi-step audit:
1. We drop records where `CustomerID` is null (guest checkouts cannot be clustered longitudinally).
2. We filter out cancellation records (`InvoiceNo` starting with 'C') and negative/zero quantities (`Quantity <= 0`) for positive transaction frequency analysis.
3. We filter out zero or negative unit prices (`UnitPrice <= 0`).
4. Any remaining negative spend totals at the customer level are purged to ensure mathematical validity of logarithmic transformation.

---

### Q6: What are the primary scientific and methodological limitations of your customer segmentation project?
**Answer**:
1. **No Demographic Grounding**: The dataset contains transaction metadata only. We do not possess customer age, gender, income, or psychological traits, and must strictly avoid fabricating demographic personas.
2. **Correlation vs Causality**: Past transaction recency and spend are descriptive correlations, not causal determinants. An "At-Risk" label does not guarantee a customer will churn, nor does a "Champion" label guarantee loyalty.
3. **Observation-Window Truncation & Seasonality**: Transaction data spans approximately 12 months. Infrequent buyers (e.g. holiday shoppers) may appear dormant simply because their annual purchase cycle falls outside the window.
4. **Generalizability**: Findings are specific to a UK-based wholesale/gift retailer in 2010-2011 and cannot be blindly generalized to subscription, B2B SaaS, or daily FMCG grocery retail without recalibration.

---

### Q7: How does CustomerIQ support model persistence and production inference?
**Answer**:
We encapsulate preprocessing, feature transformations, trained estimator, and empirical segment metadata into a serializable `CustomerIQPipeline` dataclass:
- Components stored: `RFMTransformer` (with fitted log and `StandardScaler` parameters), fitted `KMeans` model, feature column ordering, reference snapshot date, and dynamically derived segment mappings.
- Saved using `joblib.dump()` and restored via `joblib.load()` with strict type and integrity validation.
- When new customer data arrives, `pipeline.predict(df)` applies the exact identical transformations and assigns cluster labels and marketing playbooks in real-time.
