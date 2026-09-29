# Gold Price Prediction

A machine learning regression project that predicts the gold price (`GLD`) using related financial market variables.

## Dataset

The notebook loads `gold_price_ml_ready.csv`.

- **Rows:** 2,290
- **Columns:** 5
- **Target:** `GLD`

### Columns

| Column | Role |
|---|---|
| `SPX` | Feature |
| `GLD` | Target (gold price) |
| `USO` | Feature |
| `SLV` | Feature |
| `EUR/USD` | Feature |

All five columns are numeric (`float64`), and the notebook reports **no missing values**.

## Exploratory Data Analysis

The notebook performs the following checks:

- Displays the first and last rows
- Checks dataset shape and data types
- Checks missing values
- Generates descriptive statistics
- Calculates the correlation matrix
- Visualizes correlations with a heatmap
- Examines the correlation of each variable with `GLD`
- Plots the distribution of `GLD`

### Correlation with Gold Price

The notebook reports these correlations with `GLD`:

| Variable | Correlation with GLD |
|---|---:|
| `SPX` | 0.049345 |
| `USO` | -0.186360 |
| `SLV` | 0.866632 |
| `EUR/USD` | -0.024375 |

`SLV` has the strongest correlation with `GLD` among the features in this dataset.

## Feature and Target Selection

The target variable is separated from the input features:

```python
X = gold_price.drop(['GLD'], axis=1)
Y = gold_price['GLD']
```

Therefore:

- **Features:** `SPX`, `USO`, `SLV`, `EUR/USD`
- **Target:** `GLD`

## Train-Test Split

The notebook uses:

```python
train_test_split(
    X, Y,
    test_size=0.2,
    random_state=2
)
```

This creates an 80/20 training and testing split.

## Model

The project uses a **Random Forest Regressor**:

```python
RandomForestRegressor(n_estimators=100)
```

The model is trained using the training portion of the dataset and then used to predict `GLD` values for the test set.

## Results

The notebook evaluates the regression model using **R² score**.

**R² Score: `0.9887634308741609`**

This indicates that the fitted model explains a very high proportion of the variation in the test-set target values according to the R² metric.

The notebook also plots the actual gold prices against the predicted prices for the test observations.

## Visualization

The notebook includes:

1. Correlation heatmap
2. Gold price (`GLD`) distribution plot
3. Actual vs. predicted gold price plot

## Project Workflow

```text
Gold Price Dataset
        ↓
Data Inspection
        ↓
Missing Value Check
        ↓
Descriptive Statistics
        ↓
Correlation Analysis
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Random Forest Regressor
        ↓
Gold Price Prediction
        ↓
R² Evaluation
        ↓
Actual vs Predicted Plot
```

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Main Scikit-learn components:

- `train_test_split`
- `RandomForestRegressor`
- `metrics.r2_score`

## How to Run

1. Place `gold_price_ml_ready.csv` in the project directory.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Update the `pd.read_csv()` path if required.
4. Run the cells from top to bottom.

## Notes

This README is based on the implementation and outputs present in the uploaded notebook. The notebook uses R² as its reported regression evaluation metric and does not report MAE or RMSE.
