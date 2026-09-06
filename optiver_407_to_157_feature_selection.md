# Optiver Solution: From 407 Engineered Columns to 157 XGBoost Features

## Overview

The **407 columns represent the complete engineered dataframe**, whereas the **157 features are the manually curated subset ultimately passed to XGBoost**.

The available notebook code allows us to reconstruct the arithmetic behind both numbers. However, it does **not** contain the author's complete feature-selection experiment log. Therefore, this document distinguishes between:

- **What can be verified directly from the code**
- **What can reasonably be inferred from comments and feature-list construction**
- **What cannot be confirmed from the released notebooks**

The final relationship is:

```text
407 total dataframe columns
157 selected XGBoost features
250 unused, temporary, target-related, or intermediate columns
```

---

# Part I: How the Dataframe Reaches 407 Columns

## 1. Columns Present Before the Main Feature Function

Before entering the main feature-generation function, the dataframe contains **25 columns**.

### 1.1 Original training columns: 18

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
target
time_id
row_id
fold
```

### 1.2 Preprocessing columns: 7

Before the main feature function is called, the notebook adds:

```text
scale_imbalance_size
scale_matched_size
scale_bid_size
scale_ask_size
auc_bid_size
auc_ask_size
weight
```

Therefore:

```text
18 original columns + 7 preprocessing columns = 25 starting columns
```

---

## 2. Exact Breakdown of the 382 Additional Columns

The feature function creates another **382 columns**:

```text
25 starting columns + 382 additional columns = 407 total columns
```

## Group A: Basic Market and Auction Features, 23

These features combine prices, visible order-book sizes, auction quantities, and monetary values.

Examples include:

```python
ask_money = ask_size * ask_price
bid_money = bid_size * bid_price
ask_size_all = ask_size + auc_ask_size
bid_size_all = bid_size + auc_bid_size
mid_price = (ask_price + bid_price) / 2
price_diff_ask_bid = ask_price - bid_price
```

The group covers:

- Monetary values on the bid and ask sides
- Combined auction and continuous-book sizes
- Total continuous and auction volume
- Mid-price constructions
- Bid-ask spread measures
- Signed imbalance
- Price pressure
- Depth pressure

```text
Count: 23
Running total: 25 + 23 = 48
```

## Group B: Raw Ratio Features, 10

The notebook calculates ten direct ratios:

```text
imbalance_size / bid_size
imbalance_size / ask_size
matched_size / bid_size
matched_size / ask_size
imbalance_size / volume_cont
matched_size / volume_cont
auc_bid_size / bid_size
auc_ask_size / ask_size
bid_auc_money / bid_money
ask_auc_money / ask_money
```

These ratios compare auction liquidity and imbalance quantities with visible continuous-market liquidity.

```text
Count: 10
Running total: 48 + 10 = 58
```

## Group C: Non-Price Imbalance Features, 7

The standard normalised imbalance formula is:

```text
(x - y) / (x + y)
```

It is applied to seven pairs:

```text
ask_size versus bid_size
ask_money versus bid_money
volumn_money versus volumn_auc_money
volume_cont versus volumn_auc
imbalance_size versus matched_size
auc_ask_size versus auc_bid_size
ask_size_all versus bid_size_all
```

```text
Count: 7
Running total: 58 + 7 = 65
```

## Group D: Pairwise Price-Imbalance Features, 21

The solution uses seven price-like variables:

```python
prices = [
    "reference_price",
    "far_price",
    "near_price",
    "ask_price",
    "bid_price",
    "wap",
    "mid_price",
]
```

For every unique pair, it calculates:

```text
(price_1 - price_2) / (price_1 + price_2)
```

Seven prices produce:

```text
C(7, 2) = 7! / (2! × 5!) = 21 combinations
```

Examples:

```text
imb1_reference_price_far_price
imb1_reference_price_ask_price
imb1_reference_price_wap
imb1_near_price_wap
imb1_ask_price_bid_price
imb1_wap_mid_price
```

```text
Count: 21
Running total: 65 + 21 = 86
```

## Group E: Market-Urgency Features, 3

The author creates three interactions between price dislocation and size imbalance:

```text
market_urgency
market_urgency_v2
market_urgency_v3
```

For example:

```python
market_urgency = (
    price_diff_ask_bid
    * imb1_ask_size_bid_size
)
```

These attempt to distinguish a wide spread with balanced liquidity from a wide spread accompanied by one-sided pressure.

```text
Count: 3
Running total: 86 + 3 = 89
```

## Group F: Rolling Mean and Standard-Deviation Features, 80

Eight source columns are selected:

```python
rolling_features = [
    "bid_auc_money",
    "imb1_reference_price_wap",
    "bid_size_all",
    "imb1_auc_ask_size_auc_bid_size",
    "div_flag_imbalance_size_2_balance",
    "imb1_ask_size_all_bid_size_all",
    "flag_imbalance_size",
    "imb1_reference_price_mid_price",
]
```

Five rolling windows are used:

```python
windows = [3, 6, 18, 36, 60]
```

For each source-window combination, the code creates both a rolling mean and rolling standard deviation:

```text
8 source columns × 5 windows × 2 statistics = 80 features
```

```text
Count: 80
Running total: 89 + 80 = 169
```

## Group G: Momentum and Spread-Dynamics Features, 3

The author adds:

```text
imbalance_momentum_unscaled
spread_intensity
imbalance_momentum
```

Conceptually:

```text
imbalance_momentum_unscaled = signed_imbalance[t] - signed_imbalance[t-1]
```

and:

```text
imbalance_momentum = imbalance_momentum_unscaled / matched_size
```

These features measure whether market pressure is changing, rather than only describing its current level.

```text
Count: 3
Running total: 169 + 3 = 172
```

## Group H: Difference Features, 48

Differences are generated for 12 source columns:

```python
difference_sources = [
    "ask_price",
    "bid_price",
    "imb1_reference_price_near_price",
    "bid_size",
    "scale_bid_size",
    "mid_price",
    "ask_size",
    "price_div_ask_bid",
    "div_bid_size_ask_size",
    "market_urgency",
    "wap",
    "imbalance_momentum",
]
```

Four lags are used:

```python
lags = [1, 2, 3, 10]
```

Therefore:

```text
12 source columns × 4 lags = 48 difference features
```

Examples:

```text
ask_price_diff_1
ask_price_diff_3
bid_size_diff_10
market_urgency_diff_1
wap_diff_3
imbalance_momentum_diff_2
```

```text
Count: 48
Running total: 172 + 48 = 220
```

## Group I: Mock-Target Intermediate Features, 12

Four WAP horizons are used:

```python
mock_horizons = [1, 3, 12, 6]
```

This creates four future-WAP working columns:

```text
wap_shift_n1
wap_shift_n3
wap_shift_n12
wap_shift_n6
```

It also creates four shifted mock-target columns:

```text
target_mock_shift1
target_mock_shift3
target_mock_shift12
target_mock_shift6
```

There are four additional repeatedly overwritten working columns:

```text
target_single
weight_tmp
index_target_mock
target_mock
```

Therefore:

```text
4 future-WAP columns + 4 shifted mock targets + 4 working columns = 12
```

> **Important:** These include intermediate and working columns. Their presence helps explain the dataframe's width, but it does not mean all 12 are valid model inputs.

```text
Count: 12
Running total: 220 + 12 = 232
```

## Group J: Rolling Mock-Target Features, 24

The four shifted mock-target features are rolled over six windows:

```python
mock_target_windows = [1, 3, 6, 12, 24, 48]
```

Only rolling means are generated:

```text
4 shifted mock-target columns × 6 windows = 24 features
```

```text
Count: 24
Running total: 232 + 24 = 256
```

## Group K: Shift, Ratio-to-Shift, and Difference-to-Shift Features, 60

Four source columns are used:

```python
lag_sources = [
    "imb1_auc_ask_size_auc_bid_size",
    "flag_imbalance_size",
    "price_pressure_v2",
    "scale_matched_size",
]
```

Five lag periods are used:

```python
lag_periods = [1, 2, 3, 6, 12]
```

For every source-lag combination, three transformations are created:

```text
shift_h      = x[t-h]
div_shift_h  = x[t] / x[t-h]
diff_shift_h = x[t] - x[t-h]
```

Therefore:

```text
4 source columns × 5 lags × 3 transformations = 60 features
```

```text
Count: 60
Running total: 256 + 60 = 316
```

## Group L: Global Cross-Sectional Features, 14

The author selects 14 row-level features and calculates a stock-weighted market average for every:

```text
date_id × seconds_in_bucket
```

Conceptually:

```text
global_feature = Σ(stock_weight × feature) / Σ(stock_weight)
```

Examples:

```text
global_market_urgency
global_imb1_ask_money_bid_money
global_ask_price_diff_3
global_imb1_ask_size_bid_size
```

These columns allow each stock's row to incorporate information about the market-wide state at the same second.

```text
Count: 14
Running total: 316 + 14 = 330
```

## Group M: EWM and MACD-Style Features, 51

Three source columns are used:

```python
ewm_sources = [
    "mid_price_near_far",
    "imb1_reference_price_wap",
    "near_price",
]
```

### Exponential moving averages

Five spans are used:

```python
spans = [3, 6, 12, 24, 48]
```

```text
3 sources × 5 spans = 15 EWM features
```

### Fast-minus-slow differences

Four consecutive span pairs are used:

```text
3 versus 6
6 versus 12
12 versus 24
24 versus 48
```

```text
3 sources × 4 span pairs = 12 DIF features
```

### Signal-line features

Each DIF receives an exponentially weighted signal line:

```text
3 sources × 4 span pairs = 12 DEA features
```

### MACD residuals

```text
MACD = DIF - DEA
```

```text
3 sources × 4 span pairs = 12 MACD features
```

Total:

```text
15 EWM + 12 DIF + 12 DEA + 12 MACD = 51 features
```

```text
Count: 51
Running total: 330 + 51 = 381
```

## Group N: Historical Target Features, 26

For the target, 13 historical windows are used:

```python
target_windows = [
    1, 2, 3, 5, 10, 15, 20,
    25, 30, 35, 40, 45, 60,
]
```

For each window, two statistics are created:

```text
rolling_mean
rolling_std
```

Therefore:

```text
13 windows × 2 statistics = 26 features
```

```text
Count: 26
Final total: 381 + 26 = 407
```

---

# Part II: Complete 407-Column Reconciliation

| Feature group | Columns added | Running total |
|---|---:|---:|
| Original and preprocessing columns | 25 | 25 |
| Basic market and auction features | 23 | 48 |
| Raw ratios | 10 | 58 |
| Non-price imbalance features | 7 | 65 |
| Pairwise price-imbalance features | 21 | 86 |
| Market-urgency features | 3 | 89 |
| Rolling mean and standard-deviation features | 80 | 169 |
| Momentum and spread-dynamics features | 3 | 172 |
| Difference features | 48 | 220 |
| Mock-target intermediate features | 12 | 232 |
| Rolling mock-target features | 24 | 256 |
| Shift, division, and difference-to-shift features | 60 | 316 |
| Global cross-sectional features | 14 | 330 |
| EWM and MACD-style features | 51 | 381 |
| Historical target features | 26 | **407** |

The arithmetic is therefore:

```text
25 + 23 + 10 + 7 + 21 + 3 + 80 + 3 + 48
+ 12 + 24 + 60 + 14 + 51 + 26
= 407
```

---

# Part III: How the 157 XGBoost Features Were Finalised

The author did **not** automatically pass all 407 dataframe columns into XGBoost. Instead, the model-input list is manually assigned and extended through code similar to:

```python
feas_list = [
    "imb1_wap_mid_price",
    # ... manually selected feature names ...
]
```

This means a column may remain present in the engineered dataframe without being used by the model. Only columns included in the final ordered `feas_list` become XGBoost inputs.

## A. Initial Manually Selected Core and Rolling Features: 34

The main written list contains:

```text
16 core/current-state features
18 selected rolling features
--------------------------------
34 initial selected features
```

### The 16 core features

```python
[
    "imb1_wap_mid_price",
    "imb1_ask_money_bid_money",
    "imb1_volume_cont_volumn_auc",
    "imb1_reference_price_ask_price",
    "imb1_reference_price_mid_price",
    "seconds_in_bucket",
    "div_flag_imbalance_size_2_balance",
    "ask_price",
    "imb1_reference_price_bid_price",
    "scale_matched_size",
    "imb1_near_price_wap",
    "volumn_auc_money",
    "imb1_far_price_wap",
    "bid_size",
    "scale_bid_size",
    "bid_size_all",
]
```

These retain information about:

- Time within the closing auction
- Current bid-side liquidity
- Scaled matched volume
- Auction monetary value
- Relative locations of reference price, WAP, near price, far price, bid, ask, and mid-price

### The 18 selected rolling features

The selected set includes features such as:

```text
rolling18_mean_imb1_auc_ask_size_auc_bid_size
rolling3_mean_div_flag_imbalance_size_2_balance
rolling60_std_div_flag_imbalance_size_2_balance
rolling36_mean_flag_imbalance_size
rolling3_std_imb1_auc_ask_size_auc_bid_size
rolling6_std_bid_size_all
rolling18_std_bid_auc_money
rolling60_mean_imb1_reference_price_wap
```

Only 18 of the 80 first-stage rolling features are included in this initial list.

```text
Selected count: 16 + 18 = 34
```

## B. Momentum Features: 3

All three generated momentum features are selected:

```text
imbalance_momentum_unscaled
spread_intensity
imbalance_momentum
```

```text
Running selected count: 34 + 3 = 37
```

## C. Difference Features: 48

All 48 generated difference features are appended to the model feature list.

```text
Running selected count: 37 + 48 = 85
```

This indicates that short-horizon changes across prices, sizes, pressure, WAP, and momentum were treated as a particularly important feature family.

## D. Selected Mock-Target Rolling Features: 6

Although 24 rolling mock-target means are generated, only six are selected:

```python
[
    "rolling48_mean_target_mock_shift3",
    "rolling48_mean_target_mock_shift1",
    "rolling48_mean_target_mock_shift12",
    "rolling1_mean_target_mock_shift6",
    "rolling24_mean_target_mock_shift6",
    "rolling24_mean_target_mock_shift12",
]
```

```text
Running selected count: 85 + 6 = 91
```

The selection favours a small number of particular horizon-window combinations rather than retaining the complete grid.

## E. Selected Lag-Transformation Features: 13

The function generates 60 shift, division-to-shift, and difference-to-shift columns, but only 13 are retained.

Examples include:

```text
div_shift6_imb1_auc_ask_size_auc_bid_size
diff_shift6_price_pressure_v2
shift1_price_pressure_v2
div_shift3_flag_imbalance_size
div_shift3_scale_matched_size
shift6_flag_imbalance_size
shift12_flag_imbalance_size
```

```text
Running selected count: 91 + 13 = 104
```

## F. Global Cross-Sectional Features: 14

All 14 generated global market features are selected.

```text
Running selected count: 104 + 14 = 118
```

This suggests that market-wide context at the same timestamp was considered useful enough to retain as a complete family.

## G. Selected EWM and MACD-Style Features: 13

Although 51 EWM/MACD-style columns are generated, only 13 are selected.

Examples include:

```text
macd_imb1_reference_price_wap_12_24
dif_imb1_reference_price_wap_3_6
macd_mid_price_near_far_12_24
dif_near_price_3_6
macd_near_price_24_48
rolling_ewm_24_imb1_reference_price_wap
dea_near_price_24_48
```

```text
Running selected count: 118 + 13 = 131
```

## H. Historical Target Features: 26

All 26 historical target rolling-mean and rolling-standard-deviation features are selected.

```text
Final selected count: 131 + 26 = 157
```

---

# Part IV: Complete 157-Feature Reconciliation

| Selected feature group | Features retained | Cumulative count |
|---|---:|---:|
| Manually selected core and rolling features | 34 | 34 |
| Momentum features | 3 | 37 |
| Difference features | 48 | 85 |
| Selected mock-target rolling features | 6 | 91 |
| Selected lag-transformation features | 13 | 104 |
| Global cross-sectional features | 14 | 118 |
| Selected EWM and MACD-style features | 13 | 131 |
| Historical target features | 26 | **157** |

The exact calculation is:

```text
34 + 3 + 48 + 6 + 13 + 14 + 13 + 26 = 157
```

---

# Part V: On What Basis Were the 157 Features Chosen?

## What the Code Directly Shows

### 1. The final feature list is manually maintained

The decisive evidence is the explicitly written assignment:

```python
feas_list = [
    "imb1_wap_mid_price",
    # ...
]
```

The list is then extended with selected feature families or selected individual columns.

The released notebook does not show an automated selector such as:

```text
SelectFromModel
Recursive Feature Elimination (RFE)
permutation_importance
SHAP-based elimination
correlation-threshold filtering
```

Therefore, the final selection occurs through an explicitly curated list, not through an automated selector executed in the provided notebook.

### 2. Only specific members of large generated families are retained

The clearest examples are:

| Generated family | Generated | Retained | Not retained |
|---|---:|---:|---:|
| First-stage rolling features | 80 | 18 | 62 |
| Difference features | 48 | 48 | 0 |
| Rolling mock-target features | 24 | 6 | 18 |
| Lag-transformation features | 60 | 13 | 47 |
| Global cross-sectional features | 14 | 14 | 0 |
| EWM/MACD-style features | 51 | 13 | 38 |
| Historical target features | 26 | 26 | 0 |

This pattern is consistent with feature-family experiments followed by manual retention of either:

- The whole family, when it improved validation sufficiently
- A smaller subset of individual variants, when the full family was redundant or less effective

### 3. Notebook comments record validation comparisons

The code reportedly contains comments of the form:

```python
# [1,2,3,5,10,15,20,25,30,35,40,45,60] 5.8704926 157
# [1,2,3,5,10,15,20,30,45,60] 5.8708683137
```

The first configuration includes three additional historical target windows:

```text
25, 35, and 40
```

Its recorded score is:

```text
5.8704926
```

The smaller configuration records:

```text
5.8708683137
```

Because the Optiver competition metric is mean absolute error and **lower is better**, the first configuration has the better recorded score by approximately:

```text
5.8708683137 - 5.8704926 = 0.0003757137
```

The associated comment also records a 157-feature count. This is direct evidence that at least some feature-window choices were compared through local validation experiments.

## Most Plausible Reconstruction of the Selection Workflow

The exact experimental notebook is unavailable, so the following is a **reconstruction**, not a fully documented fact.

A likely workflow is:

1. Generate a broad library of financially motivated candidate features.
2. Use the notebook's time-based training and validation split:

   ```text
   training:   date_id < 390
   validation: date_id >= 390
   ```

3. Train XGBoost on a baseline feature set.
4. Add or remove feature families or selected variants, including:
   - Rolling windows
   - Difference and lag features
   - Mock-target signals
   - Global cross-sectional features
   - EWM/MACD-style signals
   - Historical target windows
5. Compare local validation MAE after each change.
6. Keep features or feature families that improved the chosen validation score.
7. Manually write the winning feature names into `feas_list`.
8. Save the final ordered list as:

   ```text
   xgb3_feas_v7_157.json
   ```

Feature importance may have informed this shortlist, but the provided notebooks do not demonstrate that step. Consequently, it would be incorrect to claim definitively that the author used gain importance, SHAP, permutation importance, correlation filtering, or a particular elimination threshold.

---

# Part VI: Why These Feature Types Make Sense

The retained 157 features are concentrated in five broad information categories.

## 1. Current Auction State

Examples:

```text
bid_size_all
scale_matched_size
volumn_auc_money
seconds_in_bucket
```

These describe the stock's position and liquidity during the closing-auction process.

## 2. Relative Price Position

Examples:

```text
imb1_wap_mid_price
imb1_reference_price_ask_price
imb1_reference_price_bid_price
imb1_near_price_wap
imb1_far_price_wap
```

Normalised relative-price features are more comparable across stocks than raw absolute price differences.

## 3. Short-Term Intraday Dynamics

Examples:

```text
ask_price_diff_1
wap_diff_3
market_urgency_diff_10
rolling18_mean_*
rolling60_std_*
```

These capture recent movement, persistence, reversal, and volatility rather than just a static snapshot.

## 4. Market-Relative Behaviour

Examples:

```text
global_market_urgency
global_imb1_ask_size_bid_size
global_ask_price_diff_3
```

These help the model distinguish stock-specific pressure from a market-wide move occurring at the same auction second.

## 5. Historical Predictability

Examples:

```text
rolling48_mean_target_mock_shift3
rolling24_mean_target_mock_shift12
rolling_mean_target_second_*
rolling_std_target_second_*
```

These features attempt to represent repeated short-term patterns and historical variation in the target signal.

The selected set therefore preserves a deliberate mixture of:

```text
current market state
+ relative price dislocation
+ short-term movement
+ auction pressure
+ market-wide context
+ historical target behaviour
```

---

# Part VII: Interpreting the 250 Non-Selected Columns

The fact that a column was not included in the final 157 does **not** necessarily mean it was useless in isolation. A generated feature may have been excluded because it was:

- A temporary working column
- A target or fold-management column
- Needed only to calculate another selected feature
- Highly redundant with a retained feature
- A weaker variant of the same signal at another window or lag
- Unhelpful under the author's particular local validation split
- Potentially risky for causal or online inference if used directly

The subtraction is exact:

```text
407 total columns - 157 model inputs = 250 non-model columns
```

However, the available code does not provide a complete reason for the exclusion of each individual one of those 250 columns.

---

# Part VIII: Key Conclusions

1. **The 407 figure is the width of the fully engineered dataframe**, including original, preprocessing, engineered, historical, and intermediate columns.
2. **The 157 figure is the final ordered model-input list**, not the total number of generated features.
3. The arithmetic behind both totals can be reconstructed exactly from the feature-generation and feature-list logic.
4. The final list is **manually curated in code** rather than selected by an automated algorithm shown in the released notebook.
5. Notebook comments provide evidence that the author compared at least some feature-window configurations using local validation MAE.
6. The complete selection history is not available, so the precise reason for every inclusion and exclusion cannot be stated with certainty.
7. The final feature set strongly emphasises price relationships, short-term changes, auction pressure, cross-sectional market context, and historical target behaviour.

## Bottom Line

The author first created a broad pool of **407 dataframe columns**. From this pool, an explicitly maintained `feas_list` selected **157 columns for XGBoost**.

```text
407 total dataframe columns
157 final XGBoost inputs
250 unused, intermediate, or non-model columns
```

The code supports the conclusion that local, time-based MAE experiments were at least part of the finalisation process. It does not expose a complete automated selection method or every individual ablation result, so those parts should not be overstated.
