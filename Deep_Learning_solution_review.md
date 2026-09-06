# Critical Review of the First-Place Optiver Solution Explanation

## Overview

After reviewing both explanations and cross-checking them against the available first-place write-up material, the overall picture is broadly correct. However, several details are simplified or potentially misleading, so the AI-generated explanation should **not** be treated as an exact description of the implementation.

## What the Winning Solution Actually Did

The best mental model is:

```text
Raw auction and order-book data
↓
Approximately 300 selected features
↓
Three complementary models
├── CatBoost: row-level nonlinear relationships
├── GRU: within-stock temporal relationships
└── Transformer: cross-stock relationships
↓
Weighted ensemble
0.5 CatBoost + 0.3 GRU + 0.2 Transformer
↓
Stock-weighted mean correction
↓
Final prediction
```

### Reported Configuration

- CatBoost weight: **0.5**
- GRU weight: **0.3**
- Transformer weight: **0.2**
- Holdout validation: first 400 days for training, final 81 days for validation
- Shared feature set: approximately 300 CatBoost-selected features
- Online retraining: every 12 new days
- Final ensemble private MAE: **5.4030** (reported five-update experiment)

---

# Important Corrections and Nuances

## 1. GRU Training vs Inference

The common statement:

> “It processes the sequential auction data and uses only the final timestep to make predictions.”

is incomplete.

The GRU receives approximately:

```text
(batch_size, 55, feature_dimension)
```

A four-layer GRU produces predictions across all 55 auction observations.

The author described it as a **causal sequence-to-sequence model**, avoiding bidirectionality to prevent future leakage.

During live inference, only the prediction associated with the latest available timestep is required.

### More accurate wording

> The GRU is trained causally across the intraday sequence, producing timestep-level outputs. During inference, the latest available timestep output is used.

This distinction matters because training across all timesteps provides substantially more supervision than training solely on the final bucket.

---

## 2. “All Models Use the Same Input” Is Slightly Misleading

All three models use the same selected set of roughly 300 feature columns, but the tensor organisation differs.

### CatBoost

```text
One row = one stock at one date and second
Input = [300 features]
Output = one target
```

CatBoost does not treat neighbouring rows as a sequence.

### GRU

```text
One sequence = one stock across the day
Input = [55 timesteps, 300 features]
Output = timestep-level predictions
```

The sequence dimension corresponds to time.

### Transformer

```text
One sequence = all stocks at one date and second
Input = [200 stocks, 300 features]
Output = one prediction per stock
```

The sequence dimension corresponds to stocks, not time.

### Key Insight

> Same feature definitions ≠ same tensor organisation.

This structural diversity is one reason the ensemble gains additional predictive power.

---

## 3. “Group Expanding Mean” Is Not Truly Expanding

The implemented logic is approximately:

```python
pl.col(col).rolling_mean(100, min_periods=1) / pl.col(col)
```

This is actually a **rolling mean with a maximum 100-row window**, not a classical unlimited expanding mean.

The first-ratio feature is:

```text
first_value / current_value
```

not:

```text
current_value / first_value
```

Therefore:

```math
rac{x_{first}}{x_t}=0.8
\Rightarrow
x_t = 1.25 x_{first}
```

A value of 0.8 means the current value is larger than the starting value.

---

## 4. Cross-Sectional Mean Ratio Direction Matters

Implemented as:

```python
pl.col(col).mean() / pl.col(col)
```

Therefore:

```math
	ext{MeanRatio}_{i,t} = rac{ar{x}_t}{x_{i,t}}
```

not:

```math
rac{x_{i,t}}{ar{x}_t}
```

Interpretation:

- Below 1 → stock is above the cross-sectional mean.
- Above 1 → stock is below the cross-sectional mean.
- Near 1 → stock is close to the mean.

Reversing this ratio changes the feature meaning completely.

---

## 5. Rank Direction Must Be Interpreted Correctly

The implementation uses:

```python
rank(descending=True, method="ordinal") / count()
```

This implies:

- Rank near `1/N` → among the largest values.
- Rank near `1` → among the smallest values.

Therefore larger rank values do **not** necessarily indicate stronger pressure.

---

## 6. Transformer Zero-Mean Operation Is Cross-Sectional

Implementation:

```python
out = out - out.mean(1, keepdim=True)
```

With output shape:

```text
(batch, stocks)
```

this subtracts the mean prediction across stocks for each market snapshot.

Conceptually:

```math
	ilde{p}_{i,t}=p_{i,t}-rac{1}{N_t}\sum_{j=1}^{N_t}p_{j,t}
```

This encourages modelling of **relative cross-sectional performance** rather than common market direction.

---

## 7. Transformer Centering vs Final Post-Processing

These are different operations.

### Transformer Internal Centering

```math
	ilde{p}_{i,t}=p_{i,t}-rac{1}{N}\sum_j p_{j,t}
```

Unweighted and applied inside the network.

### Final Ensemble Post-Processing

```math
p^{final}_{i,t}=p^{ensemble}_{i,t}-rac{\sum_j w_j p^{ensemble}_{j,t}}{\sum_j w_j}
```

Weighted and applied after ensembling.

The weighted correction enforces:

```math
\sum_i w_i p^{final}_{i,t}=0
```

This aligns closely with a target defined relative to a weighted market index.

---

## 8. Online Learning Requires Careful Wording

Supported claims:

- Models were updated every 12 days.
- CatBoost was retrained from scratch.
- GRU and Transformer were fine-tuned.
- Time-aware checks could skip retraining to avoid notebook timeout.

What cannot be stated with certainty:

> A fixed number of updates always completed successfully.

The reliable takeaway is that runtime-aware retraining and timeout management formed part of the production submission logic.

---

# The Key Quantitative Insight

The target is approximately:

```math
y_{i,t} = 10000\left[
\left(rac{WAP_{i,t+60}}{WAP_{i,t}}-1ight)
-
\left(rac{I_{t+60}}{I_t}-1ight)
ight]
```

This explains why the solution emphasises both temporal and cross-sectional information.

## Temporal Axis

Questions addressed:

- What changed during this auction?
- Is imbalance increasing?
- Is liquidity disappearing?
- How does the current value compare with the beginning of the auction phase?

Supported by:

- Lag features
- Rolling features
- Phase features
- GRU modelling

## Cross-Sectional Axis

Questions addressed:

- Is imbalance unusually large relative to other stocks?
- Where does liquidity rank?
- Is pressure stock-specific or market-wide?

Supported by:

- Cross-sectional means
- Ranking features
- Transformer architecture
- Weighted post-processing

---

# Bottom Line

The earlier AI explanation captures the high-level strategy reasonably well, but the technically accurate description is:

> The solution combined row-wise nonlinear modelling, causal within-stock sequence modelling, and same-time cross-stock attention. Its strongest edge was not a single model but the alignment between target design, temporal features, cross-sectional features, online updating, ensemble diversity, and index-aware post-processing.

The most important corrections are:

1. Exact ratio directions.
2. The causal sequence-to-sequence nature of the GRU.
3. Shared features but differently organised tensors.
4. The distinction between Transformer centering and weighted final post-processing.
5. Correct interpretation of ranks and cross-sectional ratios.
