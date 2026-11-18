# Tech Stack

## 1. Programming Language

### Python

Python is the primary programming language for the complete project.

It is used for:

-   Data loading and preprocessing.
-   Numerical computation.
-   Temporal sequence generation.
-   Spatial grid generation.
-   LSTM development.
-   CNN development.
-   Model evaluation.
-   Visualization.
-   Optional application/dashboard integration.

------------------------------------------------------------------------

## 2. Deep Learning Framework

### TensorFlow

TensorFlow provides the underlying deep-learning framework used to train
and evaluate the models.

It is responsible for:

-   Tensor operations.
-   Model training.
-   Automatic differentiation.
-   GPU acceleration where available.
-   Model saving and loading.

Dataset/API documentation:

https://www.tensorflow.org/

### Keras

Keras is used as the high-level neural-network API within TensorFlow.

It will be used to implement:

-   LSTM layers.
-   Dense layers.
-   Dropout.
-   Conv2D layers.
-   Pooling/upsampling layers.
-   Optimizers.
-   Loss functions.
-   Training callbacks.
-   Model evaluation.

Documentation:

https://keras.io/

------------------------------------------------------------------------

## 3. Data Processing

### NumPy

Used for:

-   Numerical arrays.
-   Tensor preparation.
-   Mathematical operations.
-   Spatial grid manipulation.
-   Data transformation.

https://numpy.org/

### Pandas

Used for:

-   Loading the air-quality dataset.
-   Cleaning tabular data.
-   Timestamp processing.
-   Missing-value handling.
-   Feature selection.
-   Time-series preparation.

https://pandas.pydata.org/

------------------------------------------------------------------------

## 4. Data Visualization

### Matplotlib

Used for:

-   PM2.5 forecasting plots.
-   Training/validation loss curves.
-   Spatial grids.
-   CNN probability maps.
-   Hotspot heatmaps.
-   Error analysis.

https://matplotlib.org/

### Seaborn

Used for statistical and exploratory visualizations such as:

-   Correlation matrices.
-   Feature distributions.
-   Model-performance plots.
-   Exploratory data analysis.

https://seaborn.pydata.org/

------------------------------------------------------------------------

## 5. Machine Learning Utilities

### Scikit-learn

Used where appropriate for:

-   Feature scaling.
-   Train/validation/test preprocessing utilities.
-   Evaluation metrics.
-   Baseline models.
-   Data preprocessing utilities.

https://scikit-learn.org/

Potentially relevant metrics include:

-   MAE
-   RMSE
-   R²
-   Precision
-   Recall
-   F1-score
-   IoU-related evaluation utilities or custom implementation

------------------------------------------------------------------------

## 6. Spatial Processing

### SciPy

SciPy can be used for spatial interpolation and numerical processing.

Potential uses include:

-   Spatial interpolation.
-   Distance calculations.
-   Numerical preprocessing.

https://scipy.org/

The initial implementation should prefer a simple, explainable
interpolation method such as inverse-distance weighting or another
suitable method before introducing more complex geostatistical
techniques.

------------------------------------------------------------------------

## 7. Optional Geographic Visualization

### Folium

Folium can be added if the final system needs geographic map
visualization.

It can be used to:

-   Display station locations.
-   Overlay pollution heatmaps.
-   Show hotspot regions on a geographic map.

https://python-visualization.github.io/folium/

This is optional; Matplotlib is sufficient for the core PBL heatmap
requirement.

------------------------------------------------------------------------

## 8. Optional Application Layer

### Streamlit

Streamlit can be used to create a lightweight interactive interface for
the final demonstration.

Possible dashboard elements:

-   PM2.5 forecast.
-   Forecast graph.
-   Hotspot heatmap.
-   Model metrics.
-   Input/time selection.

https://streamlit.io/

Streamlit is an optional presentation layer and is not required for
training the models.

------------------------------------------------------------------------

## 9. Dataset

### UCI Beijing Multi-Site Air Quality Dataset

The primary dataset is:

**Beijing Multi-Site Air Quality Data**

Official UCI dataset page:

https://archive.ics.uci.edu/dataset/501/beijing%2Bmulti-site%2Bair%2Bquality%2Bdata

The dataset provides multi-site hourly air-quality and meteorological
observations, making it appropriate for:

-   PM2.5 temporal forecasting.
-   Multi-station spatial analysis.
-   Pollution hotspot detection.

------------------------------------------------------------------------

## 10. Model Summary

``` text
Python
  |
  +-- Pandas / NumPy
  |       |
  |       +-- Data preprocessing
  |
  +-- TensorFlow / Keras
  |       |
  |       +-- LSTM
  |       |     +-- PM2.5 forecasting
  |       |
  |       +-- CNN
  |             +-- Spatial hotspot prediction
  |
  +-- Scikit-learn
  |       |
  |       +-- Scaling / evaluation / baselines
  |
  +-- Matplotlib / Seaborn
  |       |
  |       +-- Graphs / heatmaps / EDA
  |
  +-- SciPy
  |       |
  |       +-- Spatial processing
  |
  +-- Optional Folium / Streamlit
          |
          +-- Geographic visualization / dashboard
```

------------------------------------------------------------------------

## 11. Core Stack

The minimum required stack is:

-   Python
-   NumPy
-   Pandas
-   TensorFlow
-   Keras
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   SciPy

Optional:

-   Folium
-   Streamlit
