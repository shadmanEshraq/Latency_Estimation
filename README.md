# Latency Predictions

Final project for the Applied Machine Learning Course (Summer Semester 2026), organised by *opencampus.sh*.

## Repository Link

[Here](https://github.com/shadmanEshraq/Latency_Estimation)

## Description

This project builds a regression pipeline to predict global internet **Round-Trip Time (RTT)**. By streaming and decompressing massive bz2 telemetry dumps from RIPE Atlas, we will establish a **baseline model** for estimating the physical propagation delay of the internet. Our approach uses a filtered dataset of over 33 million successful measurements to ensure the model learns the structure of the internet connection rather than transient operational noise.

At first we have build a baseline model based on Random Forest. Then we have applied "Node-disjointed split" on the dataset, which is a more realistic model of the internet. After that we made predictions using three seperate models - XGBoost, LightGBM and MLP (classic neural network).

### Task Type

Regression

## Results Summary

The performance of the various models is summarized as follows.

### Model Comparison

| Model | Configuration | RMSE | $R^2$ | Improvement over Baseline<br>(RMSE / $R^2$) |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | 300 Trees • Depth=25 | 39.56 ms | 0.829 | Baseline |
| XGBoost | 300 Trees • Depth=8 | 37.46 ms | 0.847 | +5.31% / +2.17% |
| **LightGBM** | **500 Trees • 127 Leaves** | **36.96 ms** | **0.851** | **+6.57% / +2.65%** |
| MLP | 256 &rarr; 128 &rarr; 64 • LR= $10^-3$ | 40.20 ms | 0.824 | -1.62% / -0.60% |

#### Best Model Performance
- **Best Model:** LightGBM
- **Evaluation Metric:** Root Mean Square Error (RMSE), $R^2$
- **Final Performance:** We have **+6.57%** improvement on RMSE and **+2.65%** improvement on $R^2$, compared to the Random Forest Baseline.
- **Best Alternative Model:** XGBoost.

#### Key Insights

- **Most Important Features:** We calculated Mutual Information among the features before training and also looked at feature importance after we build various models.In all cases, *distance_km* is overwhelmingly the most critical feature. The following features also play some role : *src_continent, dst_continent, src_country, dst_country* .

- **Model Strengths:** LightGBM has :
    - Best performance among the evaluated models
    - Efficient for large tabular datasets
    - Captures complex non-linear relationships

- **Model Limitations:** However, we have also seen the following limitations :
    - Performance varies across geographic regions
    - Higher errors under challenging network conditions
    - Does not explicitly model network topology

- **Business Impact:** Some possible real-world use cases in the networking industry could be : 
    - Optimizing CDN and Edge Server Selection
    - Smart Traffic Engineering and Multi-Cloud Routing
    - Real-Time Bidding (RTB) and Ad Tech

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/README.md)**
3. **[Baseline Model](2_BaselineModel/README.md)**
4. **[Model Definition and Evaluation](3_Model/README.md)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/project_cover_image.png)
