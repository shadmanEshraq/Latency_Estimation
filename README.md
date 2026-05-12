# Global Latency Oracle: RIPE Atlas Baseline MVP

## Repository Link

[Here](https://github.com/shadmanEshraq/Latency_Estimation)

## Description

This project builds a regression pipeline to predict global internet **Round-Trip Time (RTT)**. By streaming and decompressing massive bz2 telemetry dumps from RIPE Atlas, we establish a clean baseline for the physical propagation delay of the internet. We try to solve the problem of "Network Tomography" by predicting latency between nodes where no direct measurement exists. Our approach uses a filtered dataset of over 33 million successful measurements to ensure the model learns the structural physics of the internet rather than transient operational noise.

### Task Type

[Regression]

### Results Summary

#### Best Model Performance
- **Best Model:** ["In Progress"]
- **Evaluation Metric:** [Mean Absolute Error (MAE)]
- **Final Performance:** ["In Progress"]

#### Model Comparison
- **Baseline Performance:** ["In Progress"]
- **Improvement Over Baseline:** ["In Progress"]
- **Best Alternative Model:** ["In Progress"]

#### Key Insights
- **Most Important Features:** ["In Progress"]
- **Model Strengths:** ["In Progress"]
- **Model Limitations:** ["In Progress"]
- **Business Impact:** ["In Progress"]

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

![Project Cover Image](CoverImage/Cover_Image.png)
