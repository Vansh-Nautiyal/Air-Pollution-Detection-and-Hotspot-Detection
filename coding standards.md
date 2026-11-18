# Coding Standards

## 1. Project Development Context

This project is an **undergraduate student deep-learning project**.

All code must be written with the following context in mind:

- The developer is an undergraduate student building a PBL/academic project.
- The implementation should prioritize clarity, learning value, reproducibility, and correctness.
- The code should not be written as if it were an industrial production system.
- Avoid unnecessary enterprise-level architecture, abstractions, infrastructure, or design patterns.
- Prefer straightforward Python code that a student can understand, explain, modify, and present during a viva.
- The implementation should remain proportional to the scope and timeline of the project.

---

## 2. Follow the Project Roadmap

The coding process must follow the stages defined in `RoadMap.md`.

The agent should:

- Work on the current roadmap stage before moving to later stages.
- Complete and verify the current stage before introducing the next stage.
- Avoid implementing future-stage functionality prematurely unless it is necessary for the current stage.
- Keep the implementation aligned with the 10-stage development roadmap.
- Do not skip important preprocessing or evaluation steps simply to reach the model-training stage faster.

The roadmap is the primary development sequence for the project.

---

## 3. Follow the PRD

All implementation decisions must follow the requirements defined in `PRD.md`.

The core project requirements are:

- PM2.5 forecasting using an LSTM.
- Spatial pollution hotspot detection using a CNN.
- Spatial heatmap generation.
- Appropriate evaluation of the LSTM.
- Appropriate evaluation of the CNN.
- Data preprocessing and leakage-safe dataset splitting.
- Final testing and validation.

The agent must not silently expand these requirements.

---

## 4. No Unrequested Features

**Do not add new project features unless the user explicitly asks for them.**

For example, do not introduce:

- Transformers.
- GANs.
- VAEs.
- Combined CNN-LSTM architectures.
- Real-time prediction systems.
- Cloud deployment.
- Authentication.
- User accounts.
- Databases.
- APIs.
- Mobile applications.
- Production monitoring.
- Complex MLOps pipelines.
- Microservices.
- Docker/Kubernetes infrastructure.
- Advanced model architectures.
- Extra dashboards or interfaces.

These may be possible extensions in a different project, but they are **outside the current implementation scope unless explicitly requested**.

If a possible improvement is noticed, mention it separately only when it is genuinely relevant. Do not implement it automatically.

---

## 5. Core Model Architecture

### 5.1 LSTM

The LSTM is responsible for **temporal PM2.5 forecasting**.

The code should maintain the separation:

```text
Historical time sequence
        |
        v
       LSTM
        |
        v
Future PM2.5
```

Do not change the LSTM into a different model without the user's instruction.

### 5.2 CNN

The CNN is responsible for **spatial pollution hotspot detection**.

The intended flow is:

```text
Station observations
        |
        v
Spatial grid
        |
        v
CNN
        |
        v
Hotspot probability map
        |
        v
Heatmap / hotspot mask
```

The CNN should preserve spatial information and produce a spatial prediction rather than reducing the task to a single whole-image classification unless the user explicitly changes the requirement.

---

## 6. Keep the Two Models Independent

The core project contains two independent models:

```text
LSTM → PM2.5 forecasting

CNN → Spatial hotspot detection
```

Do not combine them into a CNN-LSTM model by default.

Do not create a shared end-to-end architecture unless the user explicitly requests it.

The two models may share the cleaned source dataset and common preprocessing utilities where appropriate, but their training pipelines and prediction objectives should remain separate.

---

## 7. Coding Style

### 7.1 Prefer Readable Code

Code should be:

- Simple.
- Explicit.
- Well structured.
- Easy for an undergraduate student to understand.
- Easy to explain line-by-line.

Prefer:

```python
model = Sequential([
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation="relu"),
    Dense(1)
])
```

over unnecessarily abstract model-building frameworks.

### 7.2 Avoid Overengineering

Do not create:

- Excessive classes.
- Unnecessary design patterns.
- Generic framework layers.
- Large configuration systems.
- Complex dependency injection.
- Abstractions that make simple code difficult to follow.

Use functions and small modules where they improve clarity.

---

## 8. Python Standards

Use Python as the primary programming language.

Follow normal Python conventions:

- Use meaningful variable names.
- Use `snake_case` for variables and functions.
- Use `PascalCase` for classes when classes are actually necessary.
- Keep functions focused on one logical task.
- Avoid excessively long functions.
- Avoid unnecessary global variables.
- Use comments where they improve understanding.

Example:

```python
def create_sequences(data, sequence_length):
    ...
```

is preferred over unclear names such as:

```python
def cs(d, n):
    ...
```

---

## 9. Comments and Documentation

Comments should explain **why** something is being done when the reason is not obvious.

Avoid commenting every obvious line.

Good:

```python
# Use chronological splitting to prevent future observations
# from leaking into the training data.
train_data = data[:train_end]
```

Avoid:

```python
# Create variable
train_data = data[:train_end]
```

Important preprocessing decisions should be documented, especially:

- Missing-value handling.
- Feature selection.
- Scaling.
- Sequence length.
- Spatial interpolation.
- Hotspot threshold.
- Train/validation/test split.
- Model architecture choices.

---

## 10. Data Handling Standards

The raw dataset must not be modified directly.

Use a structure such as:

```text
data/
├── raw/
└── processed/
```

### Raw data

Keep the original downloaded dataset unchanged.

### Processed data

Store:

- Cleaned data.
- Temporal sequences or intermediate representations.
- Spatial grids.
- Generated labels where appropriate.

The preprocessing code should be reproducible.

---

## 11. Data Leakage Prevention

Data leakage must be avoided.

For temporal forecasting:

```text
Earlier data → Training
Later data   → Validation
Latest data  → Test
```

Do not randomly shuffle the entire time series before creating the train/test split.

Scaling parameters must be learned from the training data and then applied to validation/test data.

Do not use test observations to tune the model.

---

## 12. LSTM Coding Standards

The LSTM code should clearly separate:

1. Data preparation.
2. Sequence generation.
3. Model definition.
4. Training.
5. Prediction.
6. Evaluation.
7. Visualization.

The initial target is:

```text
Previous N hours → Next-hour PM2.5
```

The initial sequence length should be **24 hours**, unless the roadmap or user explicitly changes it.

Recommended initial model:

```text
LSTM(64)
Dropout(0.2)
LSTM(32)
Dense(32, ReLU)
Dense(1)
```

The implementation should use an appropriate regression loss such as MSE and report:

- MAE.
- RMSE.
- R².

Do not add additional forecasting targets unless requested.

---

## 13. CNN Coding Standards

The CNN code should clearly separate:

1. Spatial data preparation.
2. Grid generation.
3. Hotspot-label generation.
4. Model definition.
5. Training.
6. Prediction.
7. Evaluation.
8. Heatmap visualization.

The CNN should initially operate on the project-defined spatial grid.

The initial implementation may use:

```text
PM2.5 → spatial grid → CNN → hotspot probability map
```

A multi-channel spatial input may be implemented when it is part of the planned roadmap experiment.

The CNN should report:

- IoU.
- Precision.
- Recall.
- F1-score.

---

## 14. Hotspot Definition

The hotspot definition must be explicit and reproducible.

If a project-defined threshold is used, document:

- Which variable is thresholded.
- The threshold value or percentile.
- Why that threshold was selected.

Do not describe a project-defined threshold as an official health standard unless the relevant official source is actually being used and cited.

---

## 15. Visualization Standards

Use visualizations to understand and demonstrate the models.

Required/important visualizations include:

### LSTM

- Training/validation loss.
- Actual vs predicted PM2.5.

### CNN

- Spatial pollution grid.
- Hotspot probability map.
- Hotspot mask.
- Final heatmap.

Use:

- Matplotlib for core visualization.
- Seaborn for EDA/statistical plots where useful.

Do not introduce additional visualization frameworks unless required or explicitly requested.

---

## 16. Libraries

Use the project's defined technology stack.

Core libraries:

```text
Python
NumPy
Pandas
TensorFlow / Keras
Scikit-learn
Matplotlib
Seaborn
SciPy
```

Optional libraries already identified by the project:

```text
Folium
Streamlit
```

Optional libraries should only be used when they serve an already-defined project requirement.

Do not add a new dependency merely because it provides a slightly easier way to perform a task.

---

## 17. File and Module Organization

Keep the code organized according to the project structure.

Recommended structure:

```text
src/
├── data_loader.py
├── preprocessing.py
├── spatial_grid.py
├── lstm_model.py
├── cnn_model.py
├── train_lstm.py
├── train_cnn.py
├── evaluate.py
└── visualization.py
```

Responsibilities should remain clear.

For example:

- `data_loader.py` → loading raw data.
- `preprocessing.py` → cleaning/scaling/sequences.
- `spatial_grid.py` → spatial grid generation.
- `lstm_model.py` → LSTM definition.
- `cnn_model.py` → CNN definition.
- `train_lstm.py` → LSTM training.
- `train_cnn.py` → CNN training.
- `evaluate.py` → evaluation utilities.
- `visualization.py` → plots and heatmaps.

Do not create extra modules unless they are actually needed.

---

## 18. Model Saving

Save trained models so that they can be reused without retraining.

Expected models:

```text
models/
├── lstm_pm25.keras
└── cnn_hotspot.keras
```

Do not repeatedly retrain a model simply to generate visualizations if a trained model can be loaded.

---

## 19. Experiments

Experiments should be implemented only when they are part of the roadmap or PRD.

Planned LSTM experiments may include:

- Different sequence lengths.
- Persistence baseline.
- Feature-selection comparison.

Planned CNN experiments may include:

- PM2.5-only versus multi-channel input.
- Grid-resolution comparison where practical.
- Hotspot-threshold sensitivity.

Each experiment should have:

```text
Experiment
    ↓
Changed variable
    ↓
Measured metrics
    ↓
Comparison
    ↓
Conclusion
```

Do not run experiments merely for the sake of adding more results.

---

## 20. Error Handling

Use simple and understandable error handling.

For example:

```python
if data.empty:
    raise ValueError("The processed dataset is empty.")
```

Avoid building complicated error-handling frameworks.

Errors should provide enough information for the student to understand what went wrong.

---

## 21. Reproducibility

Where practical:

- Set random seeds.
- Record important hyperparameters.
- Keep preprocessing consistent.
- Save trained models.
- Keep dataset-processing steps documented.
- Use fixed experiment configurations when comparing models.

Example:

```python
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
```

---

## 22. Model Complexity

Start with the simplest architecture that satisfies the PRD.

Do not increase:

- Number of layers.
- Number of neurons.
- Number of filters.
- Number of input features.
- Training complexity.

unless there is a demonstrated reason to do so.

If the baseline works, improvements should be introduced one at a time and evaluated quantitatively.

---

## 23. Student-Level Explainability

Every important implementation choice should be explainable by an undergraduate student during a project presentation or viva.

The student should be able to explain:

- Why LSTM is used for temporal forecasting.
- Why CNN is used for spatial patterns.
- Why a spatial grid is required.
- How hotspot labels are generated.
- Why the selected metrics are appropriate.
- How train/validation/test splitting prevents leakage.
- What the model input and output shapes are.
- What the main hyperparameters do.

Avoid implementing techniques that cannot reasonably be explained within the project's academic scope.

---

## 24. Testing Standards

Before considering a stage complete, verify that:

- The code runs without errors.
- Input and output shapes are correct.
- Missing-value handling works.
- Temporal sequences are correctly constructed.
- Spatial grids are correctly generated.
- Training/validation/test data are correctly separated.
- Model predictions have the expected format.
- Metrics produce sensible values.
- Visualizations correspond to the correct data.

Testing should be performed progressively rather than waiting until the end.

---

## 25. Scope Control Rule

This is the most important coding rule:

> **Implement exactly what is required by the current roadmap and PRD, at an appropriate undergraduate-project level. Do not expand the project scope without the user's explicit instruction.**

If the agent identifies a potentially useful feature, it must **not implement it automatically**.

Instead, it may say:

> "This could be added as a future extension, but it is outside the current PRD, so I have not implemented it."

---

## 26. Priority Order

When making implementation decisions, follow this priority:

```text
1. User's explicit instruction
          ↓
2. PRD.md requirements
          ↓
3. RoadMap.md development stage
          ↓
4. Existing project architecture
          ↓
5. Coding simplicity and clarity
```

Do not let optional ideas override the defined project scope.

---

## 27. Definition of Done

A coding task is complete when:

- It satisfies the requested requirement.
- It follows the current roadmap stage.
- It does not introduce unrequested features.
- It is understandable at undergraduate level.
- It integrates correctly with the existing project.
- It has been tested appropriately.
- It does not introduce avoidable data leakage.
- Relevant outputs and metrics are produced.
- The code is documented sufficiently for the student to explain it.

The goal is not to build industrial-grade software.

The goal is to build a **correct, understandable, reproducible, and academically defensible undergraduate deep-learning project**.
