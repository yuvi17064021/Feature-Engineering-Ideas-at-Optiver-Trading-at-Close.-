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

# 8. How an Experienced Kaggle Quant Would Think About Creating 407 Features

The most important lesson from this notebook is not the final 157 features.

The real lesson is how a relatively small set of feature-engineering ideas generated over 400 candidate signals.

The workflow was:

```text
Raw Columns
      ↓
Transformations
      ↓
Feature Families
      ↓
400+ Candidate Features
      ↓
Validation Experiments
      ↓
157 Selected Features
```

The author never attempted to manually invent 407 features.

Instead, he repeatedly applied a small number of feature-engineering templates.

Those templates are reusable across almost every tabular and time-series competition.

---

# Feature Engineering Framework

Almost every generated feature belongs to one of the following families:

```text
1. Scaling Features
2. Arithmetic Features
3. Ratio Features
4. Imbalance Features
5. Interaction Features
6. Rolling Features
7. Difference Features
8. Shift Features
9. Target-Encoding Features
10. Global Context Features
11. Momentum Features
12. Historical Label Features
```

The author essentially asks:

```text
Can I measure:

Level?
Ratio?
Difference?
Momentum?
Volatility?
Relative Position?
Market Context?
History?
```

for every important variable.

---

# Family 1: Scaling Features

Raw values often have different scales.

Example:

```text
Stock A:
imbalance = 10,000

Stock B:
imbalance = 2,000,000
```

A raw imbalance of:

```text
100,000
```

means something very different for each stock.

Therefore:

```python
scaled_feature =
feature /
historical_stock_median
```

Generated:

```text
scale_imbalance_size
scale_matched_size
scale_bid_size
scale_ask_size
```

Idea:

```text
Convert absolute size
into relative size.
```

General lesson:

Whenever entities operate at different scales:

```text
Users
Machines
Countries
Stores
Stocks
```

always create normalised features.

---

# Family 2: Arithmetic Features

The first feature family simply combines columns.

Examples:

```python
A + B
A - B
A × B
A / B
```

Examples:

```text
ask_money
bid_money

ask_size_all
bid_size_all

mid_price

volume_cont

volumn_auc
```

Idea:

Raw columns rarely capture the true quantity of interest.

Instead:

```text
Liquidity
Pressure
Value
Exposure
```

are often combinations of multiple raw variables.

General lesson:

```text
Create domain-specific composites.
```

---

# Family 3: Ratio Features

A common question is:

```text
How large is A relative to B?
```

Examples:

```python
A / B
```

Generated:

```text
imbalance_size / bid_size

imbalance_size / ask_size

matched_size / bid_size

matched_size / ask_size
```

Idea:

Absolute values are often less informative than relative values.

General lesson:

```text
Ratios remove scale.
```

Very common in:

```text
Finance
Credit Risk
Retail
Manufacturing
Healthcare
```

---

# Family 4: Imbalance Features

This is probably the most important feature family.

Formula:

```python
(A - B)
/ (A + B)
```

Examples:

```text
ask_size vs bid_size

ask_money vs bid_money

reference_price vs wap

near_price vs wap

far_price vs wap
```

Idea:

Measure competition between two forces.

Examples:

```text
Supply vs Demand

Buyer vs Seller

Near vs Far

Bid vs Ask
```

Advantages:

```text
Scale invariant

Directional

Magnitude aware
```

Generated:

```text
7 size imbalances
+
21 price imbalances
```

Total:

```text
28 features
```

---

# Family 5: Interaction Features

Once the author had:

```text
Pressure
Spread
Liquidity
Imbalance
```

he multiplied them together.

Example:

```python
market_urgency =
spread
× imbalance
```

Idea:

A feature can be weak on its own.

But:

```text
Feature A
+
Feature B
```

may create a much stronger signal.

General lesson:

Always test:

```text
A × B

A / B

A - B
```

between important variables.

---

# Family 6: Rolling Features

This is where the feature count explodes.

The template is:

```python
rolling_mean
rolling_std
```

for many variables and windows.

Example:

```python
rolling_mean_3

rolling_mean_6

rolling_mean_18

rolling_mean_36

rolling_mean_60
```

Similarly:

```python
rolling_std_3

rolling_std_6

rolling_std_18

rolling_std_36

rolling_std_60
```

This alone generated:

```text
80 features
```

Reason:

The author is asking:

```text
Current Value

vs

Recent Average
```

and

```text
Current Stability
```

General lesson:

The easiest way to create 50–100 strong features is:

```text
Multiple windows
×
Multiple statistics
```

---

# Family 7: Difference Features

Template:

```python
current - past
```

Example:

```python
ask_price_diff_1
ask_price_diff_3
ask_price_diff_10
```

Idea:

Measure:

```text
Velocity

Momentum

Acceleration
```

Generated:

```text
48 features
```

General lesson:

A model often learns better from:

```text
Change
```

than from:

```text
Level
```

---

# Family 8: Shift Features

Template:

```python
feature(t - k)
```

Example:

```python
shift1

shift3

shift6

shift12
```

Idea:

Introduce memory.

Generated:

```text
60 lag-family features
```

including:

```text
shift

differences

ratios
```

General lesson:

Any temporal model should know:

```text
What happened before.
```

---

# Family 9: Mock Target Features

This is one of the most advanced ideas.

The author asks:

```text
Can I build a proxy version
of the target?
```

Construct:

```python
future_wap
/
current_wap
```

Then:

```python
stock_return
−
market_return
```

Creating:

```text
target_mock
```

This is essentially a historical approximation of the target.

Generated:

```text
24 candidate features
```

Idea:

The best feature is often:

```text
Something similar to the label itself.
```

---

# Family 10: Global Context Features

Question:

```text
What is the entire market doing?
```

Example:

```python
weighted_average(
    feature
)
```

Generated:

```text
14 global features
```

Idea:

Every stock should know:

```text
Its own state

and

The market state
```

General lesson:

Create:

```text
group-level
region-level
industry-level
market-level
```

aggregates whenever possible.

---

# Family 11: EWM and MACD Features

The author borrowed a classic trading concept.

Generate:

```text
Fast Average

Slow Average

Difference

Signal

Residual
```

Resulting in:

```text
EWM
DIF
DEA
MACD
```

Generated:

```text
51 features
```

Idea:

Capture trends across multiple timescales.

General lesson:

For any sequential dataset:

```text
Short-term view

Medium-term view

Long-term view
```

often contain different information.

---

# Family 12: Historical Target Features

This was likely among the strongest feature families.

Question:

```text
Historically,
at this exact auction second,
what normally happens?
```

Generated:

```python
rolling_mean_target

rolling_std_target
```

for:

```text
13 windows
```

creating:

```text
26 features
```

Idea:

Historical labels often contain the strongest predictive signal.

General lesson:

Whenever leakage can be avoided:

```text
Past labels
are often more useful
than raw features.
```

---

# Why The Feature Count Became 407

The author did not manually write:

```text
407 unique formulas.
```

Instead:

```text
8 variables
× 5 windows
× 2 statistics
=
80 features
```

or:

```text
12 variables
× 4 lags
=
48 features
```

or:

```text
4 variables
× 5 lags
× 3 transformations
=
60 features
```

The feature count grows automatically once feature generators are created.

This is how strong tabular models are usually built.

---

# The Real Feature Selection Process

The notebook strongly suggests:

```text
Generate many
Select few
```

rather than:

```text
Carefully hand-pick
all final features
```

The likely workflow was:

```text
Generate feature family
       ↓

Train XGBoost
       ↓

Measure Validation MAE
       ↓

Keep family if MAE improves
       ↓

Remove family if MAE worsens
```

Then:

```text
Feature Family
       ↓

Individual Feature Ablation
       ↓

Keep strongest members
```

For example:

```text
80 Rolling Features Generated

Only 18 Retained
```

Similarly:

```text
51 MACD Features Generated

Only 13 Retained
```

and:

```text
60 Shift Features Generated

Only 13 Retained
```

---

# General Blueprint For Future Competitions

Whenever working on a new dataset:

```text
Raw Variables
      ↓
Normalise
      ↓
Create Ratios
      ↓
Create Differences
      ↓
Create Imbalances
      ↓
Create Interactions
      ↓
Create Rolling Statistics
      ↓
Create Lags
      ↓
Create Group Aggregates
      ↓
Create Target History Features
      ↓
Generate Hundreds of Candidates
      ↓
Use Validation To Select Winners
```

This is essentially the framework used by many top Kaggle feature-engineering solutions.
