# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** Random forest (RF)
- **Rationale:** Our dataset is tabular. Our target variable (RTT) is influenced by different nonlinear factors like Geographic distance, ISP routing policies, Network congestion, ASN/network provider etc - a Random Forest model seems like an ideal candidate. The reasons behind are RF can - 
  1. Handle Nonlinearity Naturally
  2. Handle Mixed Data Types
  3. Is robust to Outliers
  4. Needs little preprocessing
  5. Provides Feature Importance

### Model Performance
- **Evaluation Metric:** RMSE, MAE, MAPE, R²
- **Performance Score:** TODO
- **Cross-Validation Score:** TODO

### Evaluation Methodology
- **Data Split:** TODO
- **Evaluation Metrics:** RMSE, MAE, MAPE, R²

### Metric Practical Relevance
- practical relevance
- business impact of each chosen evaluation metric.
- translate to real-world performance and decision-making?

## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.
