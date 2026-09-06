# Optiver Trading at the Close: Understanding even more complex approach

## 1. Competition Overview

### What was the competition?

The Optiver Trading at the Close competition asked participants to predict the future movement of stocks during the NASDAQ closing auction.

The target was:

```text
target = future stock return over a short horizon
```

measured in basis points.

The challenge was unusual because:

- Data arrives only during the final 10 minutes of trading.
- Predictions must be generated every 10 seconds.
- The auction book continuously evolves.
- Each stock interacts with all other stocks through the index.

The competition therefore becomes a mixture of:

1. Time-series forecasting
2. Market microstructure modelling
3. Cross-sectional modelling
4. Efficient online inference

---

# Goal

Given:

```text
date_id
stock_id
seconds_in_bucket
```

and auction-book information:

```text
imbalance_size
matched_size
bid_size
ask_size
reference_price
far_price
near_price
wap
...
```

predict:

```text
future price movement
```

for every stock at every timestamp.

---

# Why the Competition Was Difficult

Most Kaggle competitions use:

```text
One row -> independent prediction
```

This competition does not.

Rows are linked through:

- Time
- Stocks
- Auction mechanics

A prediction for stock A often depends on:

- What happened earlier in the day
- What is happening to stock B
- What is happening to the entire market at that second

The best solutions therefore captured all three views:

```text
1. Row view
2. Temporal view
3. Cross-sectional view
```

---

# 2. High-Level Architecture of the Winning Solution

The 1st place solution used three completely different models.

```text
                Feature Engineering
                         |
                         |
                  ~300 Features
                         |
      -------------------------------------
      |                 |                |
      |                 |                |
   CatBoost           GRU          Transformer
      |                 |                |
      -------------------------------------
                         |
                  Weighted Blend
                         |
               Post-Processing
                         |
                    Prediction
```

Final blend:

```text
0.50 CatBoost
0.30 GRU
0.20 Transformer
```

The key insight:

Each model sees the same market from a different perspective.

---

# 3. Feature Engineering

Feature engineering was responsible for a large portion of the score.

The raw competition data contained only:

```text
reference_price
far_price
near_price
bid_price
ask_price
wap
imbalance_size
matched_size
bid_size
ask_size
```

The winner transformed these into several hundred features.

---

# Feature Family 1: Basic Auction Features

Examples:

```text
volume
spread
mid_price
signed_imbalance
matched_imbalance
liquidity_imbalance
```

Examples:

```python
volume = bid_size + ask_size

spread = ask_price - bid_price

mid_price = (bid_price + ask_price) / 2
```

These describe the current state of the auction.

---

# Feature Family 2: Pairwise Price Imbalances

For every pair of prices:

```text
reference_price
near_price
far_price
bid_price
ask_price
wap
mid_price
```

the winner created:

```python
(a - b) / (a + b)
```

Example:

```python
(reference_price - wap) /
(reference_price + wap)
```

Why?

Because auction pressure is often encoded in the relative differences between prices rather than their absolute values.

---

# Feature Family 3: Lag Features

Auction behaviour is highly temporal.

The winner used:

```text
lag 1
lag 2
lag 3
lag 6
lag 12
```

Examples:

```python
wap_diff_1
wap_ratio_1

spread_diff_6
spread_ratio_6

imbalance_diff_12
```

These capture:

```text
How rapidly is the auction changing?
```

Rather than:

```text
What is it right now?
```

---

# Feature Family 4: Rolling Features

Examples:

```text
rolling mean
rolling std
rolling ratios
```

for:

```text
wap
spread
signed imbalance
market urgency
```

These approximate the recent auction trajectory.

---

# Feature Family 5: Phase Features

The closing auction behaves differently during:

```text
0-300 seconds
300-480 seconds
480-540 seconds
```

The winner therefore created:

```text
phase 0
phase 1
phase 2
```

and computed features relative to each phase.

Examples:

```python
current_value / phase_mean

current_value / phase_first_value
```

These tell the model:

```text
Where am I relative to the auction phase?
```

---

# Feature Family 6: Cross-Sectional Features

This was one of the most important ideas.

At each timestamp:

```text
200 stocks are observed together
```

The winner computed:

```text
cross-sectional mean
cross-sectional rank
cross-sectional z-score
```

Example:

```python
stock_spread - mean_market_spread
```

or

```python
spread_rank_among_200_stocks
```

This converts:

```text
absolute behaviour
```

into

```text
relative market behaviour
```

which is far more predictive.

---

# 4. Why Feature Selection Was Necessary

The initial feature pool contained hundreds of features.

Many were:

```text
Redundant
Correlated
Noisy
```

Using all of them:

```text
Slower training
Worse
