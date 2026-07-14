# Baseline Model

**[Notebook](atlas-rtt.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** Random forest (RF)
- **Rationale:** Our dataset is tabular. Our target variable (RTT) is influenced by different nonlinear factors like Geographic distance, ISP routing policies, Network congestion, ASN/network provider etc - a RF model seems like an ideal candidate. As a RF can -

  1. Handle Nonlinearity Naturally
  2. Handle Mixed Data Types
  3. Is robust to Outliers
  4. Needs little preprocessing
  5. Provides Feature Importance

So this a good candidate for a *Baseline Model*

### Model Performance
- **Evaluation Metric:** RMSE, MAE, R²
- **Performance Score:**

| Metric |     Value | Interpretation                                                                        |
| ------ | -------- | ------------------------------------------------------------------------------------- |
| MAE    | 12.813 ms | On average, predictions differ from actual RTT by about 12.8 ms.                      |
| RMSE   | 22.954 ms | Typical prediction error is about 23.0 ms, with larger errors penalized more heavily. |
| R²     |    0.9393 | The model explains approximately 93.93% of the variance in RTT values.                |

- **Cross-Validation Score:**  TODO ()

### Evaluation Methodology
- **Data Split:** 80/20 Train-Test at random on the dataset. No new feature enginnering for now.
- **Evaluation Metrics:** RMSE, MAE, R²

### Practical Relevance of Metrics
- **MAE (Mean Absolute Error)** : Directly interpretable in RTT units. Easy for stakeholders to understand.
- **RMSE (Root Mean Squared Error)** : Directly interpretable in RTT units. Penalizes large errors more heavily.Very important for latency-sensitive systems (e.g. video conference, online gaming, realtime trading).
- **R2** : How much network behavior is explained by the model.

For RTT estimation projects, **MAE** and **RMSE** are typically the two most informative metrics to present in reports and stakeholder presentations.

## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.

> On the next step we applied a group-based split on the dataset, which was a more realistic modelling of the internet.