# Air Quality Forecasting and Pollution Hotspot Detector

## Quick start

This repo now includes a runnable baseline implementation for:

- temporal PM2.5 forecasting with an LSTM
- spatial hotspot generation from station observations
- evaluation metrics for regression and hotspot segmentation
- a lightweight demo script to exercise the pipeline

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the demo:

```bash
python demo.py
```

Run the smoke tests:

```bash
python -m pytest -q
```

Train the PM2.5 LSTM and evaluate it on the held-out chronological test set:

```bash
python -m src.train_lstm --epochs 30 --batch-size 256 --patience 5 --sequence-length 24
python -m src.evaluate_lstm
```

The sequence length can be set to 6, 12, 24, or 48 hours for the roadmap
experiments. Evaluation reads the sequence length from the saved model.
Training saves the best validation-loss checkpoint under `models/`.
Evaluation saves metrics, persistence-baseline results, test predictions,
inference latency, and an actual-versus-predicted plot under
`results/lstm/`. The first full-dataset training run can take time,
especially on a CPU.

## 1. Project Overview

This project develops a deep-learning-based air-quality analytics system
with two independent prediction pipelines:

1.  **LSTM temporal forecasting** -- predicts future PM2.5 concentration
    from previous hourly air-quality and weather observations.
2.  **CNN spatial hotspot detection** -- converts multi-station
    observations into a regular spatial grid and predicts a 2D
    pollution-hotspot probability map.

The two models address complementary questions:

> **LSTM:** What will the pollution level be in the near future?

> **CNN:** Where are the likely pollution hotspots?

The final system visualizes the CNN output as a heatmap and reports
quantitative evaluation metrics for both models.

------------------------------------------------------------------------

## 2. Objectives

-   Forecast PM2.5 using historical time-series observations.
-   Detect spatial regions with elevated pollution.
-   Convert multi-station observations into spatial grids suitable for
    CNN processing.
-   Generate a 2D hotspot probability map.
-   Visualize pollution hotspots as heatmaps.
-   Evaluate temporal forecasting and spatial hotspot detection
    independently.
-   Maintain leakage-safe training, validation, and test partitions.
-   Provide a reproducible Python-based implementation.

------------------------------------------------------------------------

## 3. System Architecture

``` text
                         Air Quality Data
                                |
                         Data Preprocessing
                                |
                +---------------+---------------+
                |                               |
                v                               v
         Temporal Sequence                 Spatial Grid
                |                               |
                v                               v
              LSTM                             CNN
                |                               |
                v                               v
         Future PM2.5                  Hotspot Probability
           Forecast                          Map
                                                |
                                                v
                                           Hotspot Mask
                                                |
                                                v
                                            Heatmap
```

The LSTM and CNN are intentionally implemented as independent models
rather than as one combined CNN-LSTM architecture.

------------------------------------------------------------------------

## 4. Dataset

The primary dataset is the **UCI Beijing Multi-Site Air Quality
Dataset**.

It contains hourly air-quality measurements from multiple monitoring
stations together with meteorological variables. The multi-station
structure makes it suitable for both temporal forecasting and spatial
hotspot analysis.

Dataset source:

https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti-site%2Bair%2Bquality%2Bdata

Main pollutant variables include:

-   PM2.5
-   PM10
-   SO2
-   NO2
-   CO
-   O3

Meteorological variables include measurements such as:

-   Temperature
-   Pressure
-   Dew point
-   Rainfall
-   Wind speed

The project should retain the original dataset provenance and licensing
information in the project documentation.

------------------------------------------------------------------------

## 5. Temporal Pipeline

The temporal branch uses sliding time windows.

For example, with a 24-hour window:

``` text
Previous 24 hours
       |
       v
      LSTM
       |
       v
Next-hour PM2.5 prediction
```

A sample input contains multiple features for each hour, such as
pollutant and weather measurements.

Example input shape:

``` text
(24, number_of_features)
```

### Suggested model

``` text
Input sequence
      |
LSTM(64)
      |
Dropout
      |
LSTM(32)
      |
Dense(32, ReLU)
      |
Dense(1)
      |
Predicted PM2.5
```

### Temporal metrics

-   MAE
-   RMSE
-   R²

An actual-vs-predicted time-series graph will be generated for
qualitative analysis.

------------------------------------------------------------------------

## 6. Spatial Pipeline

The spatial branch starts with measurements associated with
monitoring-station coordinates.

``` text
Station measurements
       |
Latitude / Longitude
       |
Spatial interpolation
       |
Regular 2D grid
       |
CNN
       |
Hotspot probability map
       |
Thresholding
       |
Hotspot mask
       |
Heatmap
```

The initial implementation can use PM2.5 as the main spatial channel. A
multi-channel version can subsequently include other pollutants and
meteorological variables.

Example:

``` text
32 x 32 x 1   -> PM2.5 only
32 x 32 x 6   -> PM2.5 + PM10 + NO2 + SO2 + temperature + wind speed
```

### CNN concept

The CNN preserves spatial structure instead of collapsing the grid into
a single class prediction. Its output is a 2D map in which each grid
cell represents the predicted probability of being a pollution hotspot.

### Spatial metrics

-   Intersection over Union (IoU)
-   Precision
-   Recall
-   F1-score

------------------------------------------------------------------------

## 7. Hotspot Definition

A hotspot label is generated from the spatial pollution grid using a
documented threshold.

A project-defined data-driven threshold, such as an upper percentile of
observed PM2.5 values, can be used for the initial implementation. If a
health-based threshold is used, the relevant official standard must be
cited in the report.

The threshold is used consistently to generate:

``` text
Observed PM2.5 grid
        |
        v
Hotspot criterion
        |
        v
Ground-truth hotspot mask
```

------------------------------------------------------------------------

## 8. Data Leakage Prevention

Temporal data must be split chronologically rather than randomly.

Conceptually:

``` text
Earlier period  -> Training
Later period    -> Validation
Latest period   -> Test
```

Spatial and temporal preprocessing should also avoid using information
from the test period when fitting scalers, interpolation parameters, or
other learned preprocessing components.

------------------------------------------------------------------------

## 9. Evaluation

The final system should report:

  Component                Metrics
  ------------------------ ----------------------------
  LSTM PM2.5 Forecasting   MAE, RMSE, R²
  CNN Hotspot Detection    IoU, Precision, Recall, F1
  System                   Inference latency

Additional evaluation should include error analysis and representative
failure cases.

------------------------------------------------------------------------

## 10. Experiments

Recommended experiments include:

### LSTM experiments

-   Compare different sequence lengths such as 6, 12, 24 and 48 hours.
-   Compare the LSTM against a simple persistence baseline.
-   Study the effect of input feature selection.
-   Monitor overfitting using training and validation curves.

### CNN experiments

-   PM2.5-only spatial input versus multi-channel input.
-   Compare spatial grid resolutions where practical.
-   Evaluate sensitivity to the selected hotspot threshold.
-   Compare model performance using IoU and F1.

------------------------------------------------------------------------

## 11. Expected Outputs

The project should produce:

-   Trained LSTM model.
-   Trained CNN hotspot model.
-   PM2.5 forecast results.
-   Actual-vs-predicted forecast plot.
-   Spatial hotspot probability map.
-   Binary hotspot mask.
-   Pollution heatmap.
-   IoU, F1, precision and recall results.
-   MAE, RMSE and R² results.
-   Training curves.
-   Error-analysis results.
-   Reproducible preprocessing and inference code.

------------------------------------------------------------------------

## 12. Project Structure

``` text
air_quality_project/
|
+-- data/
|   +-- raw/
|   +-- processed/
|
+-- notebooks/
|   +-- 01_eda.ipynb
|   +-- 02_spatial_grid.ipynb
|   +-- 03_model_experiments.ipynb
|
+-- src/
|   +-- data_loader.py
|   +-- preprocessing.py
|   +-- spatial_grid.py
|   +-- lstm_model.py
|   +-- cnn_model.py
|   +-- train_lstm.py
|   +-- train_cnn.py
|   +-- evaluate.py
|   +-- visualization.py
|
+-- models/
|   +-- lstm_pm25.keras
|   +-- cnn_hotspot.keras
|
+-- results/
|
+-- app.py
+-- requirements.txt
+-- README.md
+-- PRD.md
+-- techstack.md
+-- RoadMap.md
```

------------------------------------------------------------------------

## 13. Project Scope

The core implementation consists of two independent deep-learning
models:

-   LSTM for PM2.5 time-series forecasting.
-   CNN for spatial pollution hotspot detection.

The project does not require a combined CNN-LSTM architecture for the
core implementation.

Advanced features such as transformer forecasting, generative models, or
a combined spatial-temporal architecture should only be considered as
optional extensions after the core system is complete.

------------------------------------------------------------------------

## 14. Final System

The completed system should allow the user to inspect:

1.  Current/recent pollution measurements.
2.  Predicted future PM2.5.
3.  Spatial hotspot predictions.
4.  Pollution heatmap.
5.  Model performance metrics.

The final result is an environmental analytics prototype that combines
**temporal forecasting** and **spatial hotspot detection** in one
system.
