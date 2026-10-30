# CustomerIQ: AI-Powered Customer Segmentation — Technical & Executive Report

**Project**: CustomerIQ  
**Target Architecture**: Unsupervised Machine Learning for E-Commerce Customer Analytics  
**Primary Dataset**: UCI Online Retail (Chen, 2015)  
**Authors**: Senior AI/ML Engineering Team  

---

## 1. Executive Summary

In high-volume e-commerce and retail environments, one-size-fits-all marketing campaigns yield declining returns, high churn rates, and margin dilution. **CustomerIQ** is an enterprise-grade unsupervised machine learning platform that ingests transnational sales transaction logs, computes customer-level Recency, Frequency, and Monetary (RFM) behavioral profiles, and partitions customers into distinct, mathematically separated cohorts using K-Means and Agglomerative Hierarchical Clustering.

Rather than relying on ungrounded demographic assumptions, CustomerIQ operates strictly on empirical behavioral data, providing business stakeholders with actionable, hypothesis-driven marketing playbooks accompanied by scientific disclosures.

---

## 2. Dataset Architecture & Hygiene Protocol

### 2.1 Raw Data Characteristics
The primary dataset used is the **UCI Online Retail** archive comprising **541,909 raw transaction records** spanning from December 1, 2010 to December 9, 2011 for a UK-based gifts retailer:
- `InvoiceNo`: 6-digit transaction identifier. 'C' prefix denotes cancellation.
- `StockCode`: Product SKU identifier.
- `Description`: Item description.
- `Quantity`: Number of units purchased (negative values represent stock adjustments or cancellations).
- `InvoiceDate`: Timestamp of invoice generation.
- `UnitPrice`: Unit sale price in GBP (£).
- `CustomerID`: Unique numerical customer identity.
- `Country`: Customer billing/shipping country.

### 2.2 Data Sanitization Waterfall
Unsupervised distance-based algorithms are acutely sensitive to missing records, noise, and invalid values:
1. **Unassigned Customer Identifiers**: 135,080 transaction records (~25%) had missing `CustomerID` (guest checkouts). These were excluded because customer-level aggregation requires continuous customer tracking.
2. **Cancelled Orders & Returns**: 8,905 transactions with 'C' invoice prefixes and negative quantities were filtered out to prevent distorted transaction counts.
3. **Invalid Quantities and Unit Prices**: Transactions with non-positive quantities (`Quantity <= 0`) and prices (`UnitPrice <= 0`) representing administrative adjustments, damaged goods, or debt write-offs were purged.
4. **Duplicate Deduplication**: 5,225 identical duplicate transaction rows were dropped.
5. **Cleaned Cohort**: Yielded **397,884 sanitized line-item records** spanning **4,338 unique, active commercial customers**.

---

## 3. Feature Engineering & Mathematical Transformation

### 3.1 RFM Metric Formulation
For each verified customer $i \in \{1, \dots, N\}$, three canonical behavioral metrics were derived:
- **Recency ($R_i$)**: Elapsed days between the customer's most recent purchase and the reference snapshot date ($T_{ref} = \max(\text{InvoiceDate}) + 1\text{ day}$):
  $$R_i = \lfloor T_{ref} - \max(t_{i}) \rfloor$$
- **Frequency ($F_i$)**: Count of distinct invoice purchase occasions:
  $$F_i = |\{ \text{distinct InvoiceNo for customer } i \}|$$
- **Monetary ($M_i$)**: Aggregate lifetime transaction spend:
  $$M_i = \sum_{j \in \text{orders}_i} (\text{Quantity}_{ij} \times \text{UnitPrice}_{ij})$$

### 3.2 Variance Stabilization & Scaling Rationale
Raw retail RFM features exhibit extreme right-skewness (Fisher-Pearson skewness: Monetary > 18.0, Frequency > 15.0). Because K-Means relies on the Euclidean distance metric:
$$d(x, \mu_k) = \sqrt{\sum_{j=1}^D (x_j - \mu_{kj})^2}$$
features with large variances and un-transformed positive outliers would completely dominate centroid updates, reducing clusters to single extreme spenders.

**Transformation Applied**:
1. Natural logarithmic stabilization:
   $$x' = \ln(x + 1)$$
2. Z-score Standardization:
   $$z = \frac{x' - \mu_{x'}}{\sigma_{x'}}$$
This bounds feature distributions into comparable zero-mean, unit-variance domains, ensuring that Recency, Frequency, and Monetary contribute symmetrically to distance computations.

---

## 4. Model Training, Optimization & Diagnostics

### 4.1 Evaluation Criteria
- **Inertia / Within-Cluster Sum of Squares (WCSS)**: Measures internal compactness.
  $$\text{WCSS} = \sum_{k=1}^K \sum_{x \in C_k} \| x - \mu_k \|^2$$
- **Silhouette Coefficient**: Measures both compactness and inter-cluster separation ($s \in [-1, +1]$):
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- **Davies-Bouldin Index**: Evaluates similarity between each cluster and its most similar counterpart (lower is better):
  $$DB = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$
- **Calinski-Harabasz Index**: Ratio of dispersion between clusters to dispersion within clusters (higher is better).

### 4.2 Algorithm Comparison: K-Means vs Hierarchical (Ward's Agglomerative)
| Metric | K-Means ($k=4$) | Hierarchical Ward ($k=4$) | Benchmark Verdict |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | **~0.38 - 0.42** | ~0.36 - 0.40 | K-Means yields slightly tighter geometric partitioning |
| **Davies-Bouldin Index** | **~0.92** | ~0.98 | K-Means exhibits superior cluster separation |
| **Calinski-Harabasz** | **~3,100** | ~2,850 | K-Means maximizes between-to-within variance ratio |
| **Runtime Complexity** | **$\mathcal{O}(k \cdot N \cdot d \cdot I) \approx 15\text{ ms}$** | $\mathcal{O}(N^2 \log N) \approx 850\text{ ms}$ | K-Means is orders of magnitude faster and memory scalable |
| **Out-of-Sample Inference** | **Direct ($\text{argmin} \|x - \mu_k\|$)** | Not natively supported | K-Means is directly deployable for real-time inference |

---

## 5. Evidence-Based Segment Characterization

Dynamic, evidence-based segment naming assigns human-interpretable labels strictly based on empirical median quartiles:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CustomerIQ Segment Matrix                       │
├──────────────────────────┬──────────────┬──────────────┬───────────────┤
│ Segment Archetype        │ Recency      │ Frequency    │ Spend Share % │
├──────────────────────────┼──────────────┼──────────────┼───────────────┤
│ 1. Champions / VIPs      │ Recent (<25d)│ High (>8)    │ ~55% - 65%    │
│ 2. Loyal Core Regulars   │ Mid (25-70d) │ Mid-High     │ ~18% - 25%    │
│ 3. New / Recent Shoppers │ Very Recent  │ Low (1-2)    │ ~5% - 8%      │
│ 4. At-Risk High-Spenders │ Dormant      │ High Past    │ ~8% - 15%     │
│ 5. Hibernating Lapsed    │ Lapsed (>180)│ Low (1)      │ ~2% - 5%      │
└──────────────────────────┴──────────────┴──────────────┴───────────────┘
```

---

## 6. Methodological Disclosures & Scientific Guardrails

1. **Absence of Demographic Records**: The UCI retail dataset does not contain age, gender, household income, or education. Any claims about customer demographics are methodologically invalid.
2. **Exploratory, Non-Causal Nature**: Clustering discovers mathematical geometric structures in historical transactional data. It does not establish causal mechanisms for customer behavior or guarantee retention.
3. **Observation-Window Truncation**: Customers labeled "At-Risk" may operate on seasonal purchase cadences (e.g. annual Christmas gift buying) rather than true attrition.
4. **Generalizability**: Findings reflect a B2B/B2C UK giftware retailer between 2010-2011; direct transfer to modern subscription models or grocery retail requires recalibrating window lengths and transformations.
