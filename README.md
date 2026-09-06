# Feature-Engineering-Ideas-at-Optiver-Trading-at-Close.-
This repo is for getting the idea for feature engineering for any data set

# Feature Engineering Methodology Behind the Optiver Solution

## Goal

The objective of feature engineering was not simply to create hundreds of columns.

The objective was to systematically transform raw auction order book data into features that capture:

1. Market imbalance
2. Supply-demand pressure
3. Relative liquidity
4. Intraday momentum
5. Mean reversion
6. Market-wide behaviour
7. Historical behaviour at the same auction timestamp
8. Multi-scale temporal structure

The final model used 157 features out of approximately 407 generated features.

---

# Philosophy Used To Create Features

Rather than manually inventing 157 features one-by-one, the approach generated entire groups of features.

The process was:

```text
Raw Data
    ↓
Generate large feature family
    ↓
Generate hundreds of candidate features
    ↓
Validate on time-based holdout
    ↓
Keep useful features
    ↓
Discard noisy features
```

This is how most top Kaggle solutions operate.

The idea is:

```text
Generate more features than necessary
Select only those improving validation
```

---

# Group 1: Normalised Size Features

Raw sizes are not comparable across stocks.

Example:

```text
Stock A median imbalance size = 2 million
Stock B median imbalance size = 100 thousand
```

The same absolute imbalance means different things.

Therefore:

```python
scale_feature =
current_value /
historical_stock_median
```

Generated examples:

```text
scale_imbalance_size
scale_matched_size
scale_bid_size
scale_ask_size
```

Purpose:

```text
Convert absolute quantities
into relative quantities
```

This principle works in any dataset.

Whenever different entities operate at different scales:

```text
Customer
Machine
Store
Country
Stock
```

normalise first.

---

# Group 2: Monetary Features

Raw sizes alone are incomplete.

1000 shares of a £10 stock
differs from

1000 shares of a £100 stock.

Therefore:

```python
money = size × price
```

Examples:

```text
bid_money
ask_money
bid_auc_money
ask_auc_money
volumn_money
volumn_auc_money
```

Purpose:

```text
Liquidity measured in value,
not just quantity.
```

---

# Group 3: Supply-Demand Imbalance Features

This was one of the most important groups.

General formula:

```python
(A - B) / (A + B)
```

Examples:

```text
bid vs ask
auction bid vs auction ask
auction liquidity vs book liquidity
```

Example:

```python
imb1_ask_size_bid_size
=
(ask_size - bid_size)
/
(ask_size + bid_size)
```

Interpretation:

```text
+1    huge ask dominance
 0    perfectly balanced
-1    huge bid dominance
```

This transformation:

```text
Captures direction
Captures magnitude
Is scale-invariant
```

Useful in nearly every ML problem involving competition between two quantities.

---

# Group 4: Pairwise Price Relationships

Prices used:

```text
reference_price
far_price
near_price
bid_price
ask_price
wap
mid_price
```

For every pair:

```python
(price1 - price2)
/
(price1 + price2)
```

Examples:

```text
imb1_reference_price_wap
imb1_reference_price_bid_price
imb1_near_price_wap
imb1_wap_mid_price
```

Purpose:

```text
Measure disagreement between
multiple pricing mechanisms.
```

General lesson:

Whenever multiple measurements represent the same concept:

```text
sensor readings
probability estimates
pricing signals
valuation metrics
```

build pairwise relationships.

---

# Group 5: Pressure Features

These combine multiple signals.

Example:

```python
price_pressure =
spread
× imbalance
```

Idea:

```text
Wide spread
+
Strong imbalance
=
More meaningful
than either alone
```

Purpose:

Generate interaction effects.

General principle:

```text
Feature A
Feature B

Create

A × B
A / B
A - B
```

Interactions often outperform raw variables.

---

# Group 6: Rolling Statistics

Most important feature family.

Rolling windows:

```text
3
6
18
36
60
```

Generated:

```python
rolling_mean
rolling_std
```

Examples:

```text
rolling3_mean
rolling18_mean
rolling60_std
```

Purpose:

Capture:

```text
Trend
Volatility
Regime
Stability
```

General principle:

Every time-series problem should include:

```text
Rolling mean
Rolling std
Rolling min
Rolling max
Rolling range
```

at several window lengths.

---

# Group 7: Difference Features

General formula:

```python
current
-
past
```

Examples:

```text
ask_price_diff_1
ask_price_diff_3
wap_diff_10
```

Measures:

```text
Velocity
Acceleration
Momentum
```

These are analogous to derivatives.

General lesson:

Always create:

```text
Lag
Difference
Percentage change
```

features.

---

# Group 8: Shift Features

General form:

```python
value(t - k)
```

Examples:

```text
shift1
shift3
shift6
shift12
```

Purpose:

Introduce memory.

Model learns:

```text
current value
vs
historical value
```

Useful for almost every sequential dataset.

---

# Group 9: Shift Ratios

General form:

```python
current
/
past
```

Examples:

```text
div_shift3
div_shift6
div_shift12
```

Purpose:

Capture relative growth.

Equivalent to:

```text
Growth rate
Percentage increase
Decay
```

---

# Group 10: Shift Differences

General form:

```python
current
-
past
```

Examples:

```text
diff_shift1
diff_shift6
diff_shift12
```

Measures:

```text
Direction
Change magnitude
```

---

# Group 11: Market-Wide Features

Most competitors only use stock information.

This solution also used cross-sectional information.

For each timestamp:

```python
weighted average
of all stocks
```

Examples:

```text
global_market_urgency
global_bid_pressure
global_target_mock
```

Purpose:

Distinguish:

```text
Market move
vs
Stock-specific move
```

General principle:

Whenever entities coexist:

```text
stores
machines
users
stocks
```

create aggregate context features.

---

# Group 12: Historical Target Features

Arguably the strongest feature family.

Question:

```text
At this exact auction second,
how has this stock behaved historically?
```

Examples:

```text
rolling_mean_5_target_second
rolling_mean_20_target_second
rolling_std_45_target_second
```

Purpose:

Capture recurring behaviour.

General lesson:

If historical labels exist:

```text
Use past labels
carefully and causally.
```

These features are often powerful.

---

# Group 13: Mock Target Features

This was a very clever idea.

Construct:

```python
future_wap /
current_wap
```

historically.

Then transform into:

```text
stock return
minus
market return
```

creating a synthetic target.

Examples:

```text
target_mock_shift1
target_mock_shift3
target_mock_shift6
target_mock_shift12
```

Then rolling averages were built.

Purpose:

Teach model what future behaviour looked like historically.

General principle:

Create proxy versions of the target whenever possible.

---

# Group 14: MACD Features

Borrowed from quantitative trading.

Steps:

```text
EMA short
EMA long

difference

signal

MACD
```

Generated:

```text
rolling_ewm
dif
dea
macd
```

Purpose:

Capture trends across several scales simultaneously.

General lesson:

Multi-scale smoothing works in:

```text
Finance
IoT
Sensor data
Energy
Retail
```

---

# Why Only 157 Features Were Kept

Approximately 407 features were generated.

Most were discarded.

Feature selection was performed using:

```text
Chronological validation
```

Validation
