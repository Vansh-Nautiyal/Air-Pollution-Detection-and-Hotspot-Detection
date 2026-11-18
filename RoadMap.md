# Project Roadmap

## 10-Stage Development Roadmap

The project should be developed in ten sequential stages. Each stage
should produce a usable output before moving to the next stage.

------------------------------------------------------------------------

## Stage 1 --- Environment and Library Setup

### Objective

Set up the Python development environment and install the required
libraries.

### Tasks

-   Install/configure Python.
-   Create a virtual environment.
-   Install NumPy.
-   Install Pandas.
-   Install TensorFlow/Keras.
-   Install Scikit-learn.
-   Install Matplotlib.
-   Install Seaborn.
-   Install SciPy.
-   Optionally install Folium and Streamlit.
-   Verify that TensorFlow can detect the available hardware
    configuration.

### Output

A working Python environment with all required dependencies.

------------------------------------------------------------------------

## Stage 2 --- Dataset Acquisition and Inspection

### Objective

Acquire and understand the air-quality dataset.

### Tasks

-   Download the UCI Beijing Multi-Site Air Quality dataset.
-   Store the raw dataset without modifying it.
-   Inspect the available stations.
-   Inspect pollutant variables.
-   Inspect meteorological variables.
-   Inspect timestamps.
-   Identify missing values and invalid records.
-   Verify station/location information.

### Output

A documented raw dataset and initial understanding of its structure.

------------------------------------------------------------------------

## Stage 3 --- Data Cleaning and Preprocessing

### Objective

Prepare clean data for deep-learning models.

### Tasks

-   Combine and organize station records.
-   Create consistent timestamps.
-   Sort observations chronologically.
-   Handle missing values.
-   Detect duplicate records.
-   Handle invalid values.
-   Select relevant pollutant and weather features.
-   Apply appropriate feature scaling.
-   Preserve station coordinates.

### Output

Cleaned and normalized data ready for model-specific processing.

------------------------------------------------------------------------

## Stage 4 --- Exploratory Data Analysis

### Objective

Understand temporal and spatial pollution behavior before modeling.

### Tasks

-   Plot PM2.5 over time.
-   Analyze daily/weekly/seasonal patterns where appropriate.
-   Plot pollutant distributions.
-   Examine pollutant correlations.
-   Examine weather-pollution relationships.
-   Compare pollution levels across stations.
-   Visualize station locations.

### Output

EDA notebook and figures that justify feature selection and modeling
decisions.

------------------------------------------------------------------------

## Stage 5 --- Temporal Dataset and LSTM Pipeline

### Objective

Convert historical observations into sequences suitable for LSTM
training.

### Tasks

-   Select the forecasting target: PM2.5.
-   Define the historical sequence length.
-   Start with a 24-hour window.
-   Construct sliding temporal sequences.
-   Separate training, validation and test periods chronologically.
-   Create input tensors and target values.
-   Build the LSTM architecture.

### Initial architecture

``` text
Input Sequence
      |
LSTM(64)
      |
Dropout
      |
LSTM(32)
      |
Dense
      |
PM2.5 Prediction
```

### Output

Training-ready temporal dataset and initial LSTM model.

------------------------------------------------------------------------

## Stage 6 --- Train and Evaluate the LSTM

### Objective

Build the complete PM2.5 forecasting pipeline.

### Tasks

-   Train the LSTM using Adam.
-   Use an appropriate regression loss such as MSE.
-   Monitor training and validation loss.
-   Save the best model.
-   Generate PM2.5 predictions.
-   Calculate MAE.
-   Calculate RMSE.
-   Calculate R².
-   Generate actual-vs-predicted graphs.

### Experiments

Compare different sequence lengths such as:

``` text
6 hours
12 hours
24 hours
48 hours
```

Also compare the LSTM with a simple persistence baseline.

### Output

Trained LSTM model, forecasting plots and quantitative results.

------------------------------------------------------------------------

## Stage 7 --- Spatial Grid Generation

### Objective

Convert monitoring-station observations into CNN-compatible spatial
data.

### Tasks

-   Extract station latitude and longitude.
-   Select the pollution variables required for spatial analysis.
-   Define the geographic grid.
-   Start with a practical grid resolution such as 16×16 or 32×32.
-   Interpolate station observations onto the grid.
-   Generate spatial pollution tensors.
-   Generate hotspot labels using the documented hotspot criterion.

### Example

``` text
Station observations
       |
Latitude + Longitude
       |
Spatial interpolation
       |
16×16 / 32×32 grid
       |
Hotspot label generation
```

### Output

Spatial training samples and corresponding hotspot masks.

------------------------------------------------------------------------

## Stage 8 --- CNN Hotspot Detection

### Objective

Train the CNN to predict spatial pollution hotspots.

### Tasks

-   Build the CNN spatial model.
-   Start with PM2.5 as the input channel.
-   Train the CNN using the generated hotspot masks.
-   Preserve spatial dimensions in the output.
-   Produce a 2D hotspot probability map.
-   Apply probability thresholding to obtain a binary hotspot mask.

### Initial architecture

``` text
Spatial Grid
     |
Conv2D
     |
Pooling
     |
Conv2D
     |
Conv2D
     |
Upsampling
     |
Conv2D
     |
Sigmoid
     |
Hotspot Probability Map
```

### Output

Trained CNN model and predicted hotspot maps.

------------------------------------------------------------------------

## Stage 9 --- CNN Evaluation and System Integration

### Objective

Evaluate the spatial model and connect both model pipelines.

### Tasks

### CNN evaluation

Calculate:

-   IoU.
-   Precision.
-   Recall.
-   F1-score.

Generate:

-   Ground-truth hotspot map.
-   Predicted probability map.
-   Predicted hotspot mask.
-   Final heatmap.

### Integration

Combine:

``` text
LSTM
  |
PM2.5 Forecast
```

and:

``` text
CNN
  |
Hotspot Map
```

into one project workflow.

### Output

An integrated air-quality forecasting and hotspot-detection system.

------------------------------------------------------------------------

## Stage 10 --- Final Testing, Validation and Documentation

### Objective

Validate the complete project and prepare the final deliverables.

### Tasks

-   Run the complete pipeline from raw data to predictions.
-   Verify preprocessing reproducibility.
-   Test the LSTM on unseen temporal data.
-   Test the CNN on unseen spatial samples.
-   Verify metric calculations.
-   Measure inference latency.
-   Perform error analysis.
-   Identify failure cases.
-   Check robustness to incomplete/missing input data.
-   Save final model weights.
-   Organize project files.
-   Complete README and technical documentation.
-   Prepare final graphs and heatmaps.
-   Prepare the project demonstration.

### Final evaluation package

``` text
LSTM:
  MAE
  RMSE
  R²

CNN:
  IoU
  Precision
  Recall
  F1

System:
  Inference latency
  Failure cases
  Robustness observations
```

### Final Output

A reproducible, documented air-quality forecasting and pollution hotspot
detection project with trained LSTM and CNN models, evaluation results,
visualizations and demonstration-ready outputs.
