# CustomerIQ - Dataset Repository

This directory stores the transactional retail datasets used for customer segmentation.

## Primary Dataset: UCI Online Retail
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/352/online+retail
- **Reference**: Chen, D. (2015). Online Retail. UCI Machine Learning Repository.
- **Description**: Contains 541,909 transnational transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based non-store online retail company specializing in all-occasion gifts.
- **Attributes**:
  1. `InvoiceNo`: 6-digit integral number uniquely assigned to each transaction. Starts with 'C' for cancellations.
  2. `StockCode`: 5-digit integral number uniquely assigned to each distinct product.
  3. `Description`: Product name.
  4. `Quantity`: The quantities of each product per transaction.
  5. `InvoiceDate`: The day and time each transaction was generated.
  6. `UnitPrice`: Product price per unit in sterling (£).
  7. `CustomerID`: 5-digit integral number uniquely assigned to each customer.
  8. `Country`: The name of the country where each customer resides.

## Data Cleaning Rules
- Drop records with missing `CustomerID` (~25% of raw records in UCI dataset represent guest checkouts).
- Remove cancellations (Invoice numbers prefixed with 'C').
- Remove transactions with non-positive quantities (`Quantity <= 0`) and prices (`UnitPrice <= 0`).
- Remove identical duplicate transactions.

## Synthetic Demonstration Mode
When no local file is available or for quick exploration, CustomerIQ includes an in-memory realistic synthetic generator (`generate_synthetic_retail_data()`) that mimics real-world customer RFM archetypes with realistic cancellations and price distributions.
