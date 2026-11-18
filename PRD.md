# Product Requirements Document (PRD)

## 1. Product Name

**Air Quality Forecasting and Pollution Hotspot Detector**

## 2. Product Goal

Build a deep-learning system that uses historical multi-station
air-quality and weather data to:

1.  Predict future PM2.5 concentration.
2.  Detect spatial pollution hotspots.
3.  Visualize predicted hotspots as a 2D heatmap.
4.  Evaluate the forecasting and hotspot-detection models using
    appropriate quantitative metrics.

------------------------------------------------------------------------

## 3. Core Requirements

### 3.1 PM2.5 Forecasting

The system shall:

-   Accept historical hourly air-quality observations as input.
-   Construct temporal sequences using a configurable historical window.
-   Use an LSTM-based model for temporal forecasting.
-   Predict future PM2.5 concentration.
-   Support configurable forecast horizons, with next-hour prediction as
    the initial target.
-   Report MAE, RMSE and R².
-   Provide actual-vs-predicted visualization.

### 3.2 Pollution Hotspot Detection

The system shall:

-   Use measurements from multiple monitoring stations.
-   Associate measurements with geographic coordinates.
-   Convert station observations into a regular spatial grid.
-   Generate hotspot labels using a documented PM2.5 threshold.
-   Use a CNN to learn spatial pollution patterns.
-   Produce a 2D hotspot probability map.
-   Convert the probability map into a hotspot mask using a configurable
    threshold.

### 3.3 Heatmap Visualization

The system shall:

-   Visualize the spatial pollution prediction as a heatmap.
-   Clearly distinguish low- and high-pollution regions.
-   Support visualization of predicted hotspot regions.
-   Allow representative prediction results to be saved for reporting.

### 3.4 Model Evaluation

The system shall evaluate the two models independently.

For PM2.5 forecasting:

-   MAE
-   RMSE
-   R²

For hotspot detection:

-   IoU
-   Precision
-   Recall
-   F1-score

The system should also record inference latency.

------------------------------------------------------------------------

## 4. Data Requirements

The system shall:

-   Use the selected multi-site air-quality dataset.
-   Preserve dataset provenance and licensing information.
-   Handle missing observations.
-   Remove or appropriately handle invalid/duplicate records.
-   Convert observations into a consistent timestamp format.
-   Preserve station/location information required for spatial
    processing.
-   Use leakage-safe training, validation and test partitions.

------------------------------------------------------------------------

## 5. Model Requirements

### LSTM

The LSTM model shall:

-   Process sequential hourly observations.
-   Learn temporal dependencies.
-   Produce a numerical PM2.5 forecast.
-   Support configurable sequence length.

### CNN

The CNN model shall:

-   Process 2D spatial pollution grids.
-   Preserve spatial information throughout the prediction pipeline.
-   Produce a dense spatial probability map rather than only a single
    classification label.
-   Support one or multiple spatial input channels.

------------------------------------------------------------------------

## 6. Visualization Requirements

The project shall provide:

-   PM2.5 forecast graph.
-   Training/validation loss curves.
-   Spatial pollution grid visualization.
-   CNN hotspot probability map.
-   Final hotspot heatmap.
-   Representative prediction and failure-case visualizations.

------------------------------------------------------------------------

## 7. Testing Requirements

The system shall be tested for:

-   Data preprocessing correctness.
-   Temporal sequence generation.
-   Spatial grid generation.
-   Model training and inference.
-   Metric calculation.
-   Visualization generation.
-   Leakage between training and test data.
-   Robustness to missing or incomplete observations.

------------------------------------------------------------------------

## 8. Success Criteria

The project is considered successful when:

-   The LSTM produces valid future PM2.5 predictions.
-   The CNN produces a valid spatial hotspot probability map.
-   Hotspot predictions can be converted into a spatial heatmap.
-   Both models can be evaluated quantitatively.
-   The complete pipeline can be reproduced from the documented
    preprocessing and training steps.
