# Confirmed Interpretation of the First-Place Optiver Solution

## Overview

The full write-up and the author’s comments remove the remaining ambiguities. They also confirm one correction to the previous review: **the live winning submission completed four online-learning updates**, because the fifth update caused the notebook to exceed the runtime limit.

---

## 1. Final Ensemble

Before post-processing, the final prediction was:

$$
\hat{y}
=
0.5\hat{y}_{\text{CatBoost}}
+
0.3\hat{y}_{\text{GRU}}
+
0.2\hat{y}_{\text{Transformer}}.
$$

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

> “Same 300 features” does not mean identical input tensors.

---

## 2. Validation Used a Single Chronological Holdout

The author used:

```text
Training:   first 400 days
Validation: final 81 days
```

This was not five-fold cross-validation. It was one time-ordered train-validation split.

This is appropriate for the problem because random splitting could allow information from later market regimes to influence validation indirectly. The author reported that the validation score aligned well with the leaderboard, so the holdout was used to select:

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

- **Group 0:** 0 to 290 seconds
- **Group 1:** 300 to 470 seconds
- **Group 2:** 480 to 540 seconds

### First-Value Ratio

For each feature $x$, date $d$, stock $i$, time $t$, and phase $g$:

$$
F^{\text{first-ratio}}_{d,i,t}
=
\frac{x_{d,i,t_g^{\text{first}}}}{x_{d,i,t}}.
$$

The direction is important. It is:

```text
first value / current value
```

not:

```text
current value / first value
```

For example, suppose imbalance size is 800 at the beginning of the phase and its current value is 1,000:

$$
\frac{800}{1000}=0.8.
$$

A value below 1 means the current value is larger than the phase’s initial value.

### Rolling-Mean Ratio

The second feature is:

$$
F^{\text{rolling-ratio}}_{d,i,t}
=
\frac{\operatorname{RollingMean}_{100}(x_{d,i,t})}{x_{d,i,t}}.
$$

Although its alias says `group_expanding_mean100`, the actual operation is:

```python
rolling_mean(100, min_periods=1)
```

It is therefore a rolling mean with a maximum window of 100 rows, not an unlimited expanding mean.

Because the calculation is performed within one stock, one date, and one auction phase, it restarts at each phase boundary.

---

## 4. Exact Meaning of the Cross-Sectional Features

These features were calculated separately for every:

```text
date_id + seconds_in_bucket
```

In other words, they compare all stocks observed at the same instant.

### Cross-Sectional Mean Ratio

The code calculates:

$$
F^{\text{mean-ratio}}_{i,t}
=
\frac{\overline{x}_t}{x_{i,t}},
$$

where:

$$
\overline{x}_t
=
\frac{1}{N_t}\sum_{j=1}^{N_t}x_{j,t}.
$$

Again, the direction matters. It is:

```text
mean across stocks / current stock value
```

Interpretation:

- $F < 1$: the stock value is above the cross-sectional mean.
- $F > 1$: the stock value is below the cross-sectional mean.
- $F \approx 1$: the stock value is close to the mean.

### Cross-Sectional Ordinal Rank

The rank feature is:

$$
F^{\text{rank}}_{i,t}
=
\frac{\operatorname{RankDescending}(x_{i,t})}{N_t}.
$$

Because `descending=True` is used:

- The highest feature value receives rank 1.
- The second-highest receives rank 2.
- The lowest receives rank $N_t$.

For 200 stocks:

```text
Highest value:  1 / 200   = 0.005
Middle value:   100 / 200 ≈ 0.50
Lowest value:   200 / 200 = 1.00
```

Therefore, a rank near zero means a relatively high raw feature value. A rank near one means a relatively low raw value.

---

## 5. GRU Structure

The GRU input was:

$$
X_{\text{GRU}} \in \mathbb{R}^{B \times 55 \times F},
$$

where:

- $B$ is the batch size.
- 55 is the number of 10-second observations in one auction day.
- $F$ is the selected feature dimension, approximately 300.

The model contained four GRU layers and produced:

$$
\hat{Y}_{\text{GRU}} \in \mathbb{R}^{B \times 55}.
$$

This is a causal sequence-to-sequence setup:

```text
t = 0   -> prediction for t = 0
t = 10  -> prediction for t = 10
...
t = 540 -> prediction for t = 540
```

The author explicitly confirmed:

> “Yes, it’s seq2seq model but not bidirection for avoiding label leak.”

This is important. A bidirectional GRU at an intermediate timestep could inspect later auction observations. The causal GRU processes only information available up to the current timestep.

### Why Use the Last Timestep During Inference?

When the model is passed a sequence containing all currently available observations, the required prediction is the final available output:

```python
prediction = output[:, -1]
```

If observations from seconds 0 through 200 are currently available, “last” means the prediction at second 200. It does not necessarily mean second 540.

The hidden state is zero-initialised at the beginning of each trading day. It is not carried indefinitely between days.

---

## 6. Transformer Structure

The Transformer input was:

$$
X_{\text{Transformer}}
\in
\mathbb{R}^{B \times 200 \times F}.
$$

Here:

- $B$ represents different date-and-time snapshots.
- 200 is the stock dimension.
- $F$ is the selected feature dimension.

The Transformer attends across **stocks**, not across the 55 auction timesteps.

At a particular moment:

```text
Token 1   = Stock 0
Token 2   = Stock 1
...
Token 200 = Stock 199
```

It outputs all stock predictions together:

$$
\hat{Y}_{\text{Transformer}}
\in
\mathbb{R}^{B \times 200}.
$$

The intended division is:

```text
GRU         -> sequence information over time
Transformer -> information across different stocks
```

The Transformer used four standard Transformer encoder layers rather than a specialised tabular model.

### Transformer Mean Correction

The model applies:

```python
out = out - out.mean(1, keepdim=True)
```

For each market snapshot:

$$
\widetilde{y}_{i,t}
=
\hat{y}_{i,t}
-
\frac{1}{N}\sum_{j=1}^{N}\hat{y}_{j,t}.
$$

Consequently:

$$
\sum_i \widetilde{y}_{i,t}=0.
$$

This removes the common level from the Transformer’s cross-sectional predictions and encourages it to predict which stocks outperform or underperform their peers.

---

## 7. Why CatBoost, GRU, and Transformer Complement One Another

Each model receives related information but imposes a different inductive bias.

### CatBoost

CatBoost learns nonlinear interactions such as:

```text
high imbalance
+ narrow spread
+ late auction phase
+ unusual cross-sectional rank
```

It treats the engineered row as a tabular observation.

### GRU

The GRU learns the path followed by one stock:

```text
imbalance increasing
spread narrowing
matched volume accelerating
price pressure persisting
```

It can distinguish two stocks that currently look similar but arrived at the current state through different paths.

### Transformer

The Transformer learns whether a signal is stock-specific or shared across the market:

```text
Is this stock’s pressure unusual relative to other stocks?
Which stocks are moving together?
Is the apparent movement simply a market-wide effect?
```

The blend therefore covers:

```text
nonlinear tabular relationships
+ temporal relationships
+ cross-sectional relationships
```

---

## 8. Feature Selection

The author generated a larger candidate feature pool and selected the top 300 using CatBoost feature importance.

The comments explain the practical choice of 300:

- 300 features performed better than 200 on validation.
- 400 features created memory problems.

The final count was therefore a validation-and-compute trade-off rather than a theoretically optimal number of features.

---

## 9. Online Learning

The complete write-up confirms:

```text
Update frequency:  every 12 test days
Planned updates:   5
Completed updates: 4
```

At each update:

- CatBoost was retrained from scratch.
- The GRU was fine-tuned.
- The Transformer was fine-tuned.

The delayed historical targets available during inference made this possible.

Conceptually:

```text
Predict days 1 to 12
↓
Receive their delayed targets
↓
Append the newly labelled observations
↓
Retrain CatBoost and fine-tune the neural networks
↓
Predict the next block
```

The fifth update caused the best submission to exceed the runtime limit. The author therefore implemented a runtime guard that skipped online retraining once total inference time reached a chosen threshold. This resulted in four completed updates.

Therefore, the earlier explanation stating that **four full updates completed** was correct.

---

## 10. Daily-File Memory Technique

The important element was not simply the use of HDF5. The main idea was to avoid constructing several enormous intermediate DataFrames while combining historical data and revealed test data.

The author:

1. Saved engineered features separately for each day.
2. Preallocated one `float32` NumPy array.
3. Loaded each daily matrix sequentially.
4. Copied it directly into the correct array slice.

```python
res = np.empty(
    (number_of_rows, number_of_features),
    dtype=np.float32,
)

for date_id in all_date_ids:
    daily_data = load_daily_h5(date_id)
    res[start:end, :] = daily_data
```

This reduces the peak memory overhead associated with DataFrame concatenation and allows the 300-feature dataset to be used during online retraining.

---

## 11. Final Weighted Post-Processing

After blending the three models, the author calculated:

$$
\mu_w
=
\frac{\sum_i w_i\hat{y}_i}{\sum_i w_i},
$$

followed by:

$$
\hat{y}^{\text{final}}_i
=
\hat{y}_i-\mu_w.
$$

This guarantees:

$$
\sum_i w_i\hat{y}^{\text{final}}_i=0.
$$

This differs from the Transformer’s internal ordinary-mean correction:

```text
Transformer correction: equal weight for every stock
Final correction:       competition-specific stock weights
```

A comment from the 14th-place participant provides supporting evidence that this weighted correction mattered. Their score reportedly changed from **5.4457** to **5.4405** after applying it.

---

## 12. Approaches That Did Not Work

The author tested and rejected:

- Adding 1D CNN or MLP predictions to the ensemble.
- Feeding multiple days into the GRU rather than one day.
- Using a larger Transformer such as DeBERTa.
- Predicting the target bucket mean with a gradient-boosted decision tree.
- Using a second-level stacking model instead of a weighted sum.

The author’s stated reason for avoiding stacking was that it required more time and did not help substantially in practice.

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

The strongest lesson is that the win did not come from one complex architecture. It came from representing the problem along its two natural dimensions:

$$
\boxed{\text{one stock through time}}
\qquad\text{and}\qquad
\boxed{\text{many stocks at the same time}}.
$$

That is why the GRU and Transformer were organised differently even though they shared the same 300 feature definitions.

## Reference

The relevant first-place write-up link is recorded in the user’s [DS and DL and Options Notes in OneNote](https://arup-my.sharepoint.com/personal/yuvraj_singh_arup_com/_layouts/15/Doc.aspx?action=edit&mobileredirect=true&wdorigin=Sharepoint&DefaultItemOpen=1&sourcedoc=%7B78bdc142-9152-4f48-874a-41d4ee13b7fd%7D&wd=target%28/DS%20and%20DL%20and%20Options%20Notes.one/%29&wdpartid=%7B9de81546-e54f-4aa5-bea5-5080d552a91a%7D%7B1%7D&wdsectionfileid=%7B413c7a93-b710-4f71-a909-84b7303132ae%7D&EntityRepresentationId=6926aae6-28bf-4d23-a134-060abd3824c7).
