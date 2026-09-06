# High-Level Overview of the Entire Solution

The project follows a typical Kaggle quantitative research workflow:

```text
Raw Market Data
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Offline Validation
        ↓
Model Training
        ↓
Model Ensembling
        ↓
Online Inference
        ↓
Competition Submission
```

---

# Notebook 1: Feature Engineering + Offline Validation

## Goal

The purpose of Notebook 1 was to answer:

```text
What features should we use?

What model configuration works best?

How many boosting rounds should we train?
```

---

## Step 1: Load Raw Optiver Data

The original Optiver dataset contains:

```text
Prices
Sizes
Auction information
Targets
```

for:

```text
200 stocks
481 trading days
55 auction timestamps per day
```

---

## Step 2: Create a Large Feature Library

Instead of designing a small number of features, the notebook creates a large candidate pool.

Feature families included:

```text
Scaling features
Ratio features
Imbalance features
Auction liquidity features
Rolling statistics
Difference features
Shift features
Mock-target features
Global market features
MACD features
Historical target features
```

Result:

```text
≈ 407 dataframe columns
```

At this stage nothing has been selected.

The objective is:

```text
Generate many potential signals.
```

---

## Step 3: Create Historical Target Features

The strongest family of features uses previous targets.

Question:

```text
At the same auction second,
how has this stock historically behaved?
```

The notebook creates:

```text
Rolling target means
Rolling target standard deviations
```

using many windows.

These become major predictive features.

---

## Step 4: Select the Best Features

The generated dataset contains approximately:

```text
407 columns
```

Many are redundant.

The author performs repeated validation experiments.

Process:

```text
Generate feature family
      ↓
Train model
      ↓
Evaluate MAE
      ↓
Keep useful features
      ↓
Remove weak features
```

Final result:

```text
157 selected features
```

stored in:

```text
xgb_157_features.json
```

---

## Step 5: Train Offline XGBoost Model

The data is split chronologically:

```text
Train:
date_id < 390

Validation:
date_id >= 390
```

This mimics the future prediction problem.

XGBoost is trained with:

```text
MAE objective
GPU acceleration
Early stopping
```

---

## Step 6: Find Optimal Number of Trees

Training begins with:

```text
10,000 possible trees
```

Early stopping identifies the best point.

Result:

```text
Best iteration ≈ 2964 trees
```

Training beyond this point increases overfitting.

---

## Step 7: Analyse Feature Importance

After training:

```text
XGBoost feature importance
```

is extracted.

This helps confirm:

```text
Which features matter most.
```

---

## Step 8: Apply Prediction Neutralisation

The Optiver target is:

```text
Stock Return
Minus
Market Return
```

Therefore:

```text
Weighted average target ≈ 0
```

The notebook enforces the same constraint on predictions.

This improves validation MAE.

Result:

```text
Raw MAE:
≈ 5.8718

Postprocessed MAE:
≈ 5.8675
```

---

## Notebook 1 Output

Notebook 1 creates:

```text
train_features_157.parquet

xgb_157_features.json

xgb_157_offline.json

xgb_157_training_metadata.json
```

These become the inputs for the next stages.

---

# Notebook 2: Multi-Seed Ensemble

## Goal

Notebook 1 produced:

```text
One good model.
```

Notebook 2 attempts to improve performance.

Question:

```text
Can we reduce prediction variance
by averaging multiple models?
```

---

## Step 1: Reuse Same Features

No feature engineering is repeated.

Notebook 2 loads:

```text
train_features_157.parquet
```

with the already selected:

```text
157 features
```

---

## Step 2: Train Multiple XGBoost Models

The same hyperparameters are used.

Only the random seed changes.

Examples:

```text
Seed 47
Seed 111
Seed 2024
```

Every seed learns slightly different trees.

---

## Step 3: Generate Validation Predictions

Each model predicts on the same validation period.

Produces:

```text
Prediction A
Prediction B
Prediction C
```

---

## Step 4: Ensemble Predictions

Predictions are averaged.

```text
Final Prediction

=
(Pred A + Pred B + Pred C)
/ 3
```

Why?

Because:

```text
Random errors tend to cancel out.
```

---

## Step 5: Apply Neutralisation Again

The weighted-zero-sum adjustment is applied to the ensemble output.

This further improves consistency with the competition target construction.

---

## Notebook 2 Result

Performance improves slightly.

```text
Single model:
≈ 5.8675

Ensemble:
≈ 5.863
```

This becomes the preferred configuration.

---

# Full-Data Training Notebook

## Goal

Now that:

```text
Features are fixed
Hyperparameters are fixed
Best iteration is known
```

the model can be trained using all available data.

---

## Step 1: Load Full Feature Dataset

```text
5.2 million rows
157 selected features
```

---

## Step 2: Apply Recency Weighting

Recent observations receive higher importance.

Example:

```text
Recent dates:
weight = 1.5

Older dates:
weight = 1.0
```

Reason:

```text
Recent market behaviour
is often more relevant.
```

---

## Step 3: Retrain Models

Each ensemble model is trained from scratch using:

```text
All available data
```

instead of the train/validation split.

Produces:

```text
Final model files
```

for inference.

---

# Final Submission Notebook

## Goal

Use trained models to predict the hidden competition test set.

---

## Step 1: Load Saved Assets

Loads:

```text
Model files
Feature list
Historical targets
Scaling statistics
Stock weights
```

---

## Step 2: Recreate Online Feature Engineering

Each new batch arriving from:

```python
iter_test()
```

must undergo the same transformations used during training.

The notebook rebuilds:

```text
Scaling features
Ratios
Rolling features
Mock-target features
Historical target features
Global features
MACD features
```

in real time.

---

## Step 3: Update Historical Target Information

Competition API reveals:

```text
Past targets
```

during inference.

These are added to:

```text
historical_targets
```

allowing target-history features to improve as more information becomes available.

---

## Step 4: Build the 157 Model Features

From the full feature set:

```text
157 selected features
```

are extracted in the exact order expected by XGBoost.

---

## Step 5: Generate Ensemble Prediction

The models predict:

```text
future relative return
```

for every stock.

Predictions are averaged.

---

## Step 6: Neutralise Predictions

Weighted market mean is removed.

```text
Prediction
-
Weighted Average Prediction
```

This enforces:

```text
Market neutrality
```

consistent with the target definition.

---

## Step 7: Submit Predictions

Predictions are sent to:

```python
environment.predict()
```

for every batch.

After all batches:

```text
submission.csv
```

is produced.

---

# One-Sentence Summary

The entire workflow was:

```text
Create ~407 candidate auction and order-book features,
select the best 157 using time-based validation,
train GPU XGBoost models,
ensemble multiple seeds,
apply market-neutral post-processing,
and rebuild the exact feature pipeline online
to generate competition predictions.
```
