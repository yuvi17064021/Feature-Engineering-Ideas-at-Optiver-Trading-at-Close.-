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




# 41. Reverse Engineering the ~407 Generated Features

The exact number of generated columns was approximately 407.

The important point is that those 407 features were not hand-crafted individually.

They were generated by repeatedly applying a small set of feature engineering transformations.

The workflow looked like:

```text
19 raw features
        ↓
23 basic engineered features
        ↓
10 ratio features
        ↓
28 pairwise imbalance features
        ↓
3 market urgency features
        ↓
80 rolling features
        ↓
48 difference features
        ↓
48 lag features
        ↓
6 mock-target rolling features
        ↓
14 global features
        ↓
13 MACD features
        ↓
26 historical target features

≈ 400+ total columns
```

The lesson is:

```text
You don't manually invent 400 features.

You invent
10-20 feature generators.

The generators create
the 400 features.
```

---

# 42. Feature Generation Template

The entire feature engineering process can be reduced to:

```text
Raw Variables
    ↓
Ratios
    ↓
Differences
    ↓
Interactions
    ↓
Rolling Statistics
    ↓
Lag Features
    ↓
Cross Entity Features
    ↓
Target History Features
```

For almost any dataset this same framework applies.

---

# 43. Stage 1: Raw Variables (19 Features)

Original market features:

```text
stock_id
date_id
seconds_in_bucket

imbalance_size
imbalance_buy_sell_flag

reference_price
matched_size
far_price
near_price

bid_price
bid_size

ask_price
ask_size

wap
```

Additional engineered raw variables:

```text
weight

scale_imbalance_size
scale_matched_size
scale_bid_size
scale_ask_size

auc_bid_size
auc_ask_size
```

Total:

```text
≈ 19 features
```

---

# 44. Stage 2: Basic Arithmetic Features (23 Features)

Generated using:

```text
Addition
Subtraction
Multiplication
Division
```

Examples:

```python
ask_money
bid_money

ask_size_all
bid_size_all

volume_cont
volumn_auc

mid_price

price_diff_ask_bid

price_div_ask_bid

depth_pressure
```

Feature count:

```text
≈ 23
```

Purpose:

```text
Raw variables rarely contain enough signal.

Interactions often matter
more than the variables themselves.
```

---

# 45. Stage 3: Ratio Features (10 Features)

General template:

```python
feature_A / feature_B
```

Generated:

```text
imbalance_size / bid_size
imbalance_size / ask_size

matched_size / bid_size
matched_size / ask_size

auc_bid_size / bid_size
auc_ask_size / ask_size

etc.
```

Feature count:

```text
10
```

Purpose:

```text
Relative values
usually outperform
absolute values.
```

General lesson:

```text
Build ratios whenever scale differs.
```

---

# 46. Stage 4: Imbalance Features (28 Features)

Core formula:

```python
(A - B)
/
(A + B)
```

This is one of the most powerful feature generators.

Generated for:

```text
Bid vs Ask

Ask Money vs Bid Money

Auction Bid vs Auction Ask

Reference Price vs WAP

Reference Price vs Mid Price

Near Price vs WAP

Far Price vs WAP

etc.
```

Feature count:

```text
7 size imbalance features
+
21 price imbalance features

=
28
```

Purpose:

```text
Measure competition
between two forces.
```

This pattern can be used in:

```text
Finance
Customer behaviour
Fraud
Manufacturing
Healthcare
```

---

# 47. Stage 5: Market Urgency Features (3 Features)

Generated:

```text
market_urgency

market_urgency_v2

market_urgency_v3
```

Template:

```python
pressure
× imbalance

or

pressure
× liquidity
```

Purpose:

Capture nonlinear behaviour.

General lesson:

```text
Feature interactions are often
more predictive than raw values.
```

---

# 48. Stage 6: Rolling Statistics (80 Features)

One of the largest feature groups.

Source variables:

```text
8 columns
```

Windows:

```text
3
6
18
36
60
```

Operations:

```text
Rolling Mean
Rolling Std
```

Feature count:

```text
8 variables
× 5 windows
× 2 operations

=
80 features
```

Formula:

```python
rolling_mean

rolling_std
```

Purpose:

Capture:

```text
Trend
Volatility
Stability
Regime
Momentum
```

---

# 49. Stage 7: Difference Features (48 Features)

Source Variables:

```text
12 variables
```

Windows:

```text
1
2
3
10
```

Formula:

```python
current
-
lagged
```

Feature count:

```text
12 × 4

=
48 features
```

Examples:

```text
ask_price_diff_1
ask_price_diff_2
ask_price_diff_3
ask_price_diff_10
```

Purpose:

Measure:

```text
Velocity
Momentum
Acceleration
```

---

# 50. Stage 8: Mock Target Features (24 Features)

This was one of the cleverest ideas.

Create synthetic targets:

```python
future_wap
/
current_wap
```

Convert into:

```text
Stock Return
Minus
Market Return
```

Generated:

```text
target_mock_shift1
target_mock_shift3
target_mock_shift6
target_mock_shift12
```

Then:

```text
rolling means
```

were applied.

Windows:

```text
1
3
6
12
24
48
```

Feature count:

```text
4 target signals
× 6 windows

=
24 features
```

Only six survived.

---

# 51. Stage 9: Lag Features (60 Features)

Variables:

```text
4 source variables
```

Lags:

```text
1
2
3
6
12
```

For every lag:

Created:

```text
shift

difference

ratio
```

Count:

```text
4 variables
× 5 lags
× 3 operations

=
60 features
```

Examples:

```text
shift6_price_pressure

diff_shift12_price_pressure

div_shift3_price_pressure
```

---

# 52. Stage 10: Global Features (14 Features)

This was the first truly cross-sectional feature group.

Formula:

```python
weighted market average
```

For each timestamp:

```python
Σ(weight × variable)
/
Σ(weight)
```

Generated:

```text
global_market_urgency

global_bid_pressure

global_target_mock

etc.
```

Feature count:

```text
14
```

Purpose:

Distinguish:

```text
Market move

vs

Stock move
```

---

# 53. Stage 11: MACD Features (60 Generated)

Source Variables:

```text
3
```

EWM Windows:

```text
3
6
12
24
48
```

Generated:

```text
EMA
DIF
DEA
MACD
```

For multiple scales.

Feature count:

```text
≈ 60
```

Only 13 survived.

Purpose:

Capture:

```text
Short trend

Medium trend

Long trend
```

simultaneously.

---

# 54. Stage 12: Historical Target Features (26 Features)

Question:

```text
Historically,
how has this stock behaved
at this exact auction second?
```

Windows:

```text
1
2
3
5
10
15
20
25
30
35
40
45
60
```

Generated:

```text
Rolling Mean
Rolling Std
```

Count:

```text
13 means

13 stds

=
26
```

Examples:

```text
rolling_mean_20_target_second

rolling_std_45_target_second
```

---

# 55. Approximate Breakdown

Generated groups:

```text
Raw features                         19

Basic arithmetic                     23

Ratios                               10

Imbalances                           28

Market urgency                        3

Rolling statistics                   80

Diff features                        48

Mock target features                 24

Shift features                       60

Global features                      14

MACD features                        60

Historical target features           26
----------------------------------------

Approx Total                        395+
```

After intermediate helper features and temporary columns:

```text
≈ 407 total columns
```

---

# 56. Why Only 157 Survived

A feature survived if:

```text
Adding it improved
out-of-time validation MAE.
```

A feature was removed if:

```text
Highly correlated

No validation improvement

Overfitted

Too noisy

Computationally expensive
```

The process was:

```text
Generate Group
      ↓
Train XGBoost
      ↓
Measure MAE
      ↓
Keep if MAE improves
      ↓
Discard otherwise
```

This is the key lesson.

Feature selection was NOT:

```text
This feature sounds useful.
```

Feature selection WAS:

```text
Does validation improve?
```

Only those features that repeatedly improved future-date validation performance were retained.

That process reduced:

```text
407 generated features
        ↓
157 selected features
```

which is approximately:

```text
61% reduction
```

while improving generalisation.
