# Confirmed Interpretation of the Deep Learning Optiver Solution

## Overview

The full write-up and the author's comments remove the remaining ambiguities. They also confirm one correction to the previous review:

**The live winning submission completed four online-learning updates because the fifth update caused the notebook to exceed the runtime limit.**

---

## 1. Final Ensemble

Before post-processing, the final prediction was:

```text
Final Prediction =
0.5 × CatBoost
+ 0.3 × GRU
+ 0.2 × Transformer
```

The weights were chosen using the holdout validation set.

All three models used the same selected **300 feature definitions**, but the features were organised differently for each model:

```text
CatBoost:
one row × 300 features

GRU:
one stock × 55 timesteps × 300 features

Transformer:
one timestamp × 200 stocks × 300 features
```

Therefore:

> "Same 300 features" does not mean identical input tensors.

---

## 2. Validation Used a Single Chronological Holdout

The author used:

```text
Training:   first 400 days
Validation: final 81 days
```

This was not five-fold cross-validation. It was one time-ordered train-validation split.

This is appropriate for the problem because random splitting could allow information from later market regimes to influence validation indirectly.

The author reported that the validation score aligned well with the leaderboard, so the holdout was used to select:

- Features
- Models
- Ensemble weights

---

## 3. Exact Meaning of the Phase Features

The auction was divided into three intervals:

```python
seconds_in_bucket < 300  -> group 0
seconds_in_bucket < 480  -> group 1
otherwise                -> group 2
```

Therefore:

```text
Group 0: 0 to 290 seconds
Group 1: 300 to 470 seconds
Group 2: 480 to 540 seconds
```

### First-Value Ratio

For each feature:

```text
first_ratio =
(first value within the phase)
/
(current value)
```

The direction is important.

```text
first value / current value
```

not:

```text
current value / first value
```

Example:

```text
Beginning of phase = 800
Current value      = 1000

first_ratio = 800 / 1000 = 0.8
```

A value below 1 means the current value is larger than the phase's initial value.

---

### Rolling-Mean Ratio

The second feature is:

```text
rolling_ratio =
rolling_mean_100(feature)
/
current_feature_value
```

Although the alias says:

```text
group_expanding_mean100
```

the actual implementation is:

```python
rolling_mean(100, min_periods=1)
```

Therefore it is a rolling mean with a maximum window of 100 rows rather than an unlimited expanding mean.

Because the calculation is performed within one:

```text
stock
+
date
+
auction phase
```

it restarts at every phase boundary.

---

## 4. Exact Meaning of the Cross-Sectional Features

These features were calculated separately for every:

```text
date_id + seconds_in_bucket
```

In other words, they compare all stocks observed at the same instant.

### Cross-Sectional Mean Ratio

The code calculates:

```text
mean_ratio =
(mean value across all stocks)
/
(current stock value)
```

Interpretation:

```text
mean_ratio < 1
    stock value is above average

mean_ratio > 1
    stock value is below average

mean_ratio ≈ 1
    stock value is near average
```

---

### Cross-Sectional Ordinal Rank

The rank feature is:

```text
rank =
descending_rank
/
number_of_stocks
```

Because descending ranking is used:

```text
Highest value  -> Rank 1
Second highest -> Rank 2
Lowest value   -> Rank N
```

For 200 stocks:

```text
Highest value :   1 / 200 = 0.005
Middle value  : 100 / 200 = 0.50
Lowest value  : 200 / 200 = 1.00
```

Therefore:

```text
Rank near 0 -> unusually high raw value
Rank near 1 -> unusually low raw value
```

---

## 5. GRU Structure

The GRU input shape was:

```text
(B, 55, F)

B = batch size
55 = auction timesteps
F = number of selected features (~300)
```

The model contained four GRU layers.

Output shape:

```text
(B, 55)
```

This is a causal sequence-to-sequence setup:

```text
t = 0   -> prediction for t = 0
t = 10  -> prediction for t = 10
...
t = 540 -> prediction for t = 540
```

The author explicitly confirmed:

> "Yes, it's seq2seq model but not bidirection for avoiding label leak."

This is important because a bidirectional GRU could use future auction information when predicting an earlier timestep.

A causal GRU only processes information available up to the current timestep.

### Why Use the Last Timestep During Inference?

During inference:

```python
prediction = output[:, -1]
```

If observations currently exist only up to second 200:

```text
Last timestep
=
prediction at second 200
```

It does not necessarily mean second 540.

The hidden state is reset at the beginning of each trading day.

---

## 6. Transformer Structure

Transformer input shape:

```text
(B, 200, F)

B = date-time snapshots
200 = stocks
F = selected features
```

The Transformer attends across stocks, not across time.

At one timestamp:

```text
Token 1   = Stock 0
Token 2   = Stock 1
...
Token 200 = Stock 199
```

Output shape:

```text
(B, 200)
```

The intended separation was:

```text
GRU         -> temporal relationships
Transformer -> cross-sectional relationships
```

The model used four standard Transformer encoder layers.

### Transformer Mean Correction

The implementation applies:

```python
out = out - out.mean(1, keepdim=True)
```

Conceptually:

```text
adjusted_prediction =
prediction
-
(mean prediction across all stocks)
```

Therefore:

```text
Sum of adjusted predictions across stocks = 0
```

This removes the common market component and encourages the model to predict relative stock performance.

---

## 7. Why CatBoost, GRU, and Transformer Complement One Another

Each model receives similar information but applies a different inductive bias.

### CatBoost

CatBoost learns nonlinear feature interactions such as:

```text
high imbalance
+ narrow spread
+ late auction phase
+ unusual rank
```

It views the engineered row as a tabular sample.

### GRU

The GRU learns temporal evolution:

```text
imbalance increasing
spread narrowing
matched volume accelerating
price pressure persisting
```

Two stocks may have identical current features but different histories.

The GRU can distinguish between them.

### Transformer

The Transformer learns cross-stock relationships:

```text
Is this pressure unusual?
Are stocks moving together?
Is this just a market-wide effect?
```

The ensemble therefore combines:

```text
tabular nonlinearities
+
temporal relationships
+
cross-sectional relationships
```

---

## 8. Feature Selection

The author generated a larger candidate feature library and selected the top 300 using CatBoost feature importance.

The comments indicate:

```text
300 features performed better than 200
400 features created memory issues
```

Therefore the final choice was a balance between:

```text
validation performance
+
runtime constraints
+
memory limits
```

rather than a theoretically optimal feature count.

---

## 9. Online Learning

The complete write-up confirms:

```text
Update frequency : every 12 test days
Planned updates  : 5
Completed updates: 4
```

At each update:

- CatBoost retrained from scratch
- GRU fine-tuned
- Transformer fine-tuned

Workflow:

```text
Predict days 1-12
↓
Receive delayed targets
↓
Append newly labelled data
↓
Retrain / fine-tune models
↓
Predict next block
```

The fifth update exceeded notebook runtime limits.

A runtime guard therefore skipped further online retraining beyond a predefined threshold.

As a result:

```text
Completed updates = 4
```

---

## 10. Daily-File Memory Technique

The important idea was not merely HDF5 storage.

The key objective was to avoid building several massive intermediate DataFrames.

The process was:

1. Save engineered features separately for each day.
2. Allocate one large Float32 NumPy array.
3. Load each daily matrix sequentially.
4. Copy it into the correct position.

Example:

```python
res = np.empty(
    (number_of_rows, number_of_features),
    dtype=np.float32,
)

for date_id in all_date_ids:
    daily_data = load_daily_h5(date_id)
    res[start:end, :] = daily_data
```

Benefits:

```text
Lower memory usage
Avoid DataFrame concatenation overhead
Enable online retraining
Support 300-feature dataset
```

---

## 11. Final Weighted Post-Processing

After blending predictions:

```text
weighted_mean =
Σ(weight × prediction)
/
Σ(weight)
```

Then:

```text
final_prediction =
prediction
-
weighted_mean
```

This guarantees:

```text
Weighted sum of final predictions = 0
```

This differs from the Transformer's internal correction:

```text
Transformer:
equal-weight average

Final post-processing:
competition stock weights
```

The weighted correction reportedly improved leaderboard performance.

---

## 12. Approaches That Did Not Work

The author experimented with:

- 1D CNN ensemble members
- MLP ensemble members
- Multi-day GRU sequences
- Larger Transformers such as DeBERTa
- Gradient-boosted trees on bucket averages
- Second-level stacking models

These approaches were ultimately rejected.

The author indicated that stacking added complexity without meaningful improvement.

---

# Final Corrected Summary

The first-place solution was a carefully aligned system:

```text
Time-ordered validation
+
300 selected microstructure features
+
CatBoost for tabular nonlinearities
+
Causal seq2seq GRU for intraday dynamics
+
Cross-stock Transformer for relative behaviour
+
Periodic online retraining using revealed targets
+
Stock-weighted post-processing
```

The strongest lesson is that the win did not come from one unusually complex architecture.

Instead, it came from modelling the problem along its two natural dimensions:

```text
GRU:
    one stock through time

Transformer:
    many stocks at the same time
```

That is why the GRU and Transformer were organised differently even though both were built from the same 300 feature definitions.
