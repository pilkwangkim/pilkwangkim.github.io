---
title: "S6E5 Driver's High: Driver Feature Engineering"
date: 2026-05-03 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, playground-series, feature-engineering, cross-validation, auc, formula1]
math: true
pin: false
---

# 🏎️ S6E5 Driver's High: Driver Feature Engineering

Kaggle notebook link:  
[S6E5 Driver's High: Driver Feature Engineering](https://www.kaggle.com/code/pilkwang/s6e5-driver-s-high-driver-feature-eng)

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-03-s6e5-drivers-high/cover.png" alt="S6E5 Driver's High cover" width="80%">
</p>

The central question of this notebook is narrow:

> **Does `Driver` contain measurable signal after race-state, tyre-state, compound, and stint information are already modeled?**

In a pit-stop prediction problem, `Driver` is a tempting feature because it looks like a real strategic actor.
But statistically, it is also a high-cardinality categorical variable.
That means a strong result must distinguish between **actual reusable signal** and _category memorization_.

The notebook therefore treats Driver feature engineering as a sequence of controlled experiments:

$$
\Delta_{\text{AUC}}(B)
=
\operatorname{AUC}\!\left(\hat p_{\text{base}+B}^{\text{oof}}, y\right)
-
\operatorname{AUC}\!\left(\hat p_{\text{counterfactual}}^{\text{oof}}, y\right)
$$

where each candidate block \(B\) is promoted only when it improves the correct counterfactual under the same OOF protocol.

## **1. The Modeling Question**

The target is the probability of a pit stop on the next lap:

$$
y_i =
\begin{cases}
1, & \text{pit next lap} \\
0, & \text{otherwise}
\end{cases}
$$

The model produces:

$$
\hat p_i = P(y_i = 1 \mid x_i)
$$

The score is ROC-AUC, which can be interpreted as a ranking probability:

$$
\operatorname{AUC}
=
P(\hat p^+ > \hat p^-)
+ \frac{1}{2}P(\hat p^+ = \hat p^-)
$$

So the Driver question is not whether Driver changes the predicted probability for some rows.
The question is whether Driver improves the **relative ranking** of positive and negative rows.

| Layer | Role | Statistical Risk |
|---|---|---|
| 🧱 **Race-state foundation** | lap, progress, stint, tyre age, degradation | underfitting if the physics-like state is too thin |
| 🛞 **Compound and tyre pressure** | tyre age relative to expected compound life | wrong scale if compound effects are treated as raw IDs only |
| 🧑‍✈️ **Driver representation** | string structure, frequency, target prior, interactions | high-cardinality leakage or memorization |
| 🧪 **OOF protocol** | same split, one changed block | false lift if validation rows encode their own target |
| 🧩 **Stack / residual layer** | compare Driver signal against stronger artifacts | over-crediting Driver when the real gain comes from model family or original data |

The key design choice is that `Driver` is **not allowed to enter as one undifferentiated feature**.
It is split into progressively riskier representations.

## **2. Driver Inventory**

The first diagnostic checks whether Driver behaves like a compact real-world categorical variable or a synthetic high-cardinality one.

| Dataset | Rows | Unique Drivers | `D...` Code Rate | 3-Letter Rate |
|---|---:|---:|---:|---:|
| Train | 439,140 | 887 | 59.58% | 40.42% |
| Test | 188,165 | 801 | 59.34% | 40.66% |
| Original | 101,371 | 31 | 0.00% | 100.00% |

This matters because the train/test data contain many `D###` style synthetic IDs, while the original data is made of 3-letter driver codes.
The notebook therefore separates:

| Driver Signal | Meaning | Leakage Risk |
|---|---|---|
| 🧬 **String structure** | `D###` vs 3-letter code, numeric ID bins | low |
| 📚 **Original vocabulary flag** | whether Driver appears in the original data vocabulary | low to moderate |
| 📊 **Frequency/support** | number of rows behind Driver/Race contexts | low |
| 🎯 **Target encoding** | smoothed empirical pit-stop prior by category | high unless fold-safe |
| 🔀 **Interaction target encoding** | Driver-Compound, Race-Stint, Compound-Stint priors | higher variance |

The support distribution is also important:

| Threshold | Drivers | Covered Rows | Covered Row Rate |
|---|---:|---:|---:|
| `>= 10 rows` | 666 | 438,432 | 99.84% |
| `>= 50 rows` | 535 | 435,485 | 99.17% |
| `>= 100 rows` | 485 | 431,979 | 98.37% |
| `>= 250 rows` | 411 | 419,501 | 95.53% |
| `>= 500 rows` | 337 | 392,580 | 89.40% |

This makes frequency features statistically meaningful.
Most rows are covered by drivers with enough support, but the long tail is still large enough that raw one-hot identity can overfit.

<details markdown="1">
<summary>Show notebook snippet: data loading and Driver inventory</summary>

```python
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission_template = pd.read_csv(submission_path)
original = pd.read_csv(original_path) if original_path is not None else None

TARGET = resolve_target_column(train, TARGET_CANDIDATES, "train")
SUBMISSION_TARGET = resolve_target_column(submission_template, TARGET_CANDIDATES, "sample_submission")

ID_COL = "id"
base_features = [col for col in test.columns if col != ID_COL]
raw_numeric_cols = [col for col in base_features if pd.api.types.is_numeric_dtype(train[col])]
raw_categorical_cols = [col for col in base_features if col not in raw_numeric_cols]
original_driver_set = set(original["Driver"].astype(str)) if original is not None and "Driver" in original else set()

driver_inventory = pd.DataFrame({
    "Dataset": ["Train", "Test", "Original" if original is not None else "Original missing"],
    "Rows": [len(train), len(test), len(original) if original is not None else 0],
    "Unique_Drivers": [
        train["Driver"].nunique(),
        test["Driver"].nunique(),
        original["Driver"].nunique() if original is not None else 0,
    ],
    "D_Code_Rate": [
        train["Driver"].astype(str).str.match(r"^D\d+$").mean(),
        test["Driver"].astype(str).str.match(r"^D\d+$").mean(),
        original["Driver"].astype(str).str.match(r"^D\d+$").mean() if original is not None else np.nan,
    ],
    "Three_Letter_Rate": [
        train["Driver"].astype(str).str.match(r"^[A-Z]{3}$").mean(),
        test["Driver"].astype(str).str.match(r"^[A-Z]{3}$").mean(),
        original["Driver"].astype(str).str.match(r"^[A-Z]{3}$").mean() if original is not None else np.nan,
    ],
})
```

</details>

## **3. Race-State Foundation**

Before measuring Driver, the model needs a strong non-Driver baseline.
The foundation translates raw lap rows into variables that better represent pit-stop pressure.

The first transformation estimates race length:

$$
\widehat L_i =
\frac{\text{LapNumber}_i}{\text{RaceProgress}_i}
$$

Then it derives remaining race state:

$$
R_i = 1 - \text{RaceProgress}_i,
\qquad
\text{NormalizedTyreLife}_i =
\frac{\text{TyreLife}_i}{\widehat L_i}
$$

Compound-specific tyre pressure is expressed as:

$$
u_i =
\frac{\text{TyreLife}_i}{E[\text{Life}\mid \text{Compound}_i]}
$$

and a pit-window indicator is:

$$
\mathbb{1}\!\left(u_i \ge w_{\text{Compound}_i}\right)
$$

This makes the model compare tyre age on a compound-adjusted scale instead of treating all tyre life values as equivalent.

| Feature Block | Main Variables | Interpretation |
|---|---|---|
| 🏁 **Race/tyre algebra** | estimated race laps, laps remaining, normalized tyre life | race phase and tyre age scale |
| 🛞 **Compound structure** | hardness, dry/wet-like flags, tyre-life × hardness | compound-dependent wear pressure |
| ⏱️ **Public domain features** | pit window, degradation per lap, stop urgency | F1-inspired strategy pressure |
| 🔥 **Nonlinear interactions** | tyre life × progress, tyre life × stint, degradation × age | threshold-like pit behavior |

<details markdown="1">
<summary>Show notebook snippet: race-state and compound feature blocks</summary>

```python
COMPOUND_EXPECTED_LIFE = {
    "SOFT": 25.0,
    "MEDIUM": 35.0,
    "HARD": 45.0,
    "INTERMEDIATE": 30.0,
    "WET": 40.0,
}

COMPOUND_WINDOW_START = {
    "SOFT": 0.64,
    "MEDIUM": 0.68,
    "HARD": 0.72,
    "INTERMEDIATE": 0.62,
    "WET": 0.60,
}

def add_race_tyre_algebra(df):
    out = df.copy()
    estimated_laps = _safe_divide(
        out["LapNumber"].astype(float),
        out["RaceProgress"].astype(float),
    ).replace([np.inf, -np.inf], np.nan)

    out["EstimatedRaceLaps"] = estimated_laps.clip(lower=1, upper=120)
    out["LapsRemainingEstimate"] = (out["EstimatedRaceLaps"] - out["LapNumber"]).clip(lower=-10, upper=120)
    out["RemainingRaceProgress"] = (1.0 - out["RaceProgress"].astype(float)).clip(-0.1, 1.1)
    out["TyreLifeToLapNumber"] = _safe_divide(out["TyreLife"].astype(float), out["LapNumber"].astype(float))
    out["TyreLifeToRaceProgress"] = _safe_divide(out["TyreLife"].astype(float), out["RaceProgress"].astype(float))
    out["NormalizedTyreLifeEstimate"] = _safe_divide(out["TyreLife"].astype(float), out["EstimatedRaceLaps"].astype(float))
    return out

def add_public_domain_features(df):
    out = df.copy()
    expected_life = out["Compound"].astype(str).map(COMPOUND_EXPECTED_LIFE).fillna(35.0)
    window_start = out["Compound"].astype(str).map(COMPOUND_WINDOW_START).fillna(0.68)

    tyre_life = out["TyreLife"].astype(float)
    race_progress = out["RaceProgress"].astype(float).clip(0, 1.05)
    lap_delta = out["LapTime_Delta"].astype(float)
    cum_deg = out["Cumulative_Degradation"].astype(float)

    out["TyreLifePct"] = _safe_divide(tyre_life, expected_life).clip(0, 3)
    out["PitWindowStartPct"] = window_start
    out["InCompoundPitWindow"] = (out["TyreLifePct"] >= window_start).astype(int)
    out["WindowOvershootPct"] = (out["TyreLifePct"] - window_start).clip(lower=0, upper=2)
    out["DegPerLap"] = _safe_divide(cum_deg, tyre_life + 1.0)
    out["PositiveLapTimeDelta"] = lap_delta.clip(lower=0)
    out["AgeXDelta"] = tyre_life * out["PositiveLapTimeDelta"]
    out["StopUrgency"] = out["TyreLifePct"] * race_progress * (1.0 + 0.20 * out["Stint"].astype(float).clip(1, 4))
    return out
```

</details>

## **4. Driver Representation**

The Driver feature ladder starts with the safest information and moves toward higher-risk encodings.

### **4.1 String Structure**

The first compressed representation avoids raw one-hot identity:

$$
\text{Driver} \mapsto
\left[
\mathbb{1}_{D\text{-code}},
\mathbb{1}_{3\text{-letter}},
\text{DriverNum},
\text{DriverNumBin},
\mathbb{1}_{\text{missing num}}
\right]
$$

This captures the synthetic-vs-original vocabulary structure without allocating one dummy variable per driver.

### **4.2 Frequency Encoding**

For a categorical key \(c\), frequency encoding adds:

$$
n(c) = \sum_i \mathbb{1}(x_i = c),
\qquad
f(c) = \log(1+n(c))
$$

and a rare-driver flag:

$$
\mathbb{1}\{n(\text{Driver}) < 50\}
$$

This does not say that frequent drivers are inherently better.
It tells the model how much support a category estimate has.

### **4.3 Context Normalization**

Race-relative z-scores compare a row against its local context:

$$
z_{i,g,v}
=
\frac{x_{i,v} - \mu_{g(i),v}}{\sigma_{g(i),v} + \epsilon}
$$

where \(g(i)\) can be `Race`, `Race_Compound`, `Race_Lap`, or `Compound`.
This is useful because pit timing is not purely absolute.
An old tyre on one race/lap context may be ordinary in another.

### **4.4 Target Encoding**

For a category \(c\), the smoothed target prior is:

$$
\operatorname{TE}(c)
=
\frac{\sum_{i:g_i=c} w_i y_i + \alpha \bar y}
{\sum_{i:g_i=c} w_i + \alpha}
$$

The smoothing parameter \(\alpha\) controls shrinkage toward the global mean.
Large \(\alpha\) is useful for small-support interactions such as `Driver_Compound`.

<details markdown="1">
<summary>Show notebook snippet: Driver structure, count maps, and target encoding</summary>

```python
def add_driver_structure(df):
    out = df.copy()
    driver = out["Driver"].astype(str)
    extracted = driver.str.extract(r"^D(\d+)$", expand=False)
    out["Driver_is_D_code"] = driver.str.match(r"^D\d+$").astype(int)
    out["Driver_is_3letter"] = driver.str.match(r"^[A-Z]{3}$").astype(int)
    out["Driver_num"] = pd.to_numeric(extracted, errors="coerce").fillna(-1)
    out["Driver_num_missing"] = (out["Driver_num"] < 0).astype(int)
    bins = [-2, -0.5, 99, 199, 399, 699, 9999]
    out["Driver_num_bin"] = pd.cut(out["Driver_num"], bins=bins, labels=False, include_lowest=True).astype(float)
    return out

def fit_count_maps(fit_df, columns, weight=None):
    w = np.ones(len(fit_df), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    maps = {}
    for col in columns:
        tmp = pd.DataFrame({col: fit_df[col].astype(str), "__w": w})
        maps[col] = tmp.groupby(col, observed=True)["__w"].sum()
    return maps

def apply_count_maps(df, count_maps):
    out = df.copy()
    for col, counts in count_maps.items():
        mapped = out[col].astype(str).map(counts).fillna(0).astype(float)
        out[f"{col}_count"] = mapped
        out[f"{col}_freq_log1p"] = np.log1p(mapped)
        if col == "Driver":
            out["Driver_is_rare_50"] = (mapped < 50).astype(int)
    return out

def fit_target_maps(fit_df, columns, target, weight=None, alpha=80):
    y = fit_df[target].astype(float).to_numpy()
    w = np.ones(len(fit_df), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    prior = np.average(y, weights=w)
    maps = {}
    tmp = fit_df.copy()
    tmp["__y"] = y
    tmp["__w"] = w
    tmp["__yw"] = y * w
    for col in columns:
        grouped = tmp.groupby(col, observed=True).agg(
            weight_sum=("__w", "sum"),
            target_sum=("__yw", "sum"),
        )
        maps[col] = ((grouped["target_sum"] + alpha * prior) / (grouped["weight_sum"] + alpha)).to_dict()
    return maps, prior
```

</details>

## **5. Fold-Safe Validation**

The most important implementation detail is that all fitted transformations are learned inside the fold.

For fold \(k\):

$$
\mathcal{D}_{-k} = \text{training rows outside fold } k,
\qquad
\mathcal{D}_{k} = \text{validation rows inside fold } k
$$

Then any fitted encoder \(T_k\) must satisfy:

$$
T_k = \operatorname{fit}(\mathcal{D}_{-k}),
\qquad
X_k^{\text{valid}} = T_k(\mathcal{D}_k)
$$

For target encoding, the training side itself also needs an inner OOF encoding.
Otherwise a row can encode its own target.

$$
\hat z_i =
\operatorname{TE}_{-h(i)}(x_i)
$$

where \(h(i)\) is the inner fold containing row \(i\).

<details markdown="1">
<summary>Show notebook snippet: fold-safe feature fitting and OOF runner</summary>

```python
def fit_fold_features(fit_df, valid_df, test_df, blocks, target, fit_weight=None):
    fit_x = prepare_static_features(fit_df, blocks)
    valid_x = prepare_static_features(valid_df, blocks)
    test_x = prepare_static_features(test_df, blocks)

    if "frequency_encoding" in blocks:
        count_maps = fit_count_maps(fit_x, FREQUENCY_COLUMNS, weight=fit_weight)
        fit_x = apply_count_maps(fit_x, count_maps)
        valid_x = apply_count_maps(valid_x, count_maps)
        test_x = apply_count_maps(test_x, count_maps)

    if "context_normalization" in blocks:
        value_cols = sorted({value for values in CONTEXT_GROUP_SPECS.values() for value in values})
        global_stats = fit_global_stats(fit_x, value_cols, weight=fit_weight)
        group_stats = fit_group_stats(fit_x, CONTEXT_GROUP_SPECS, weight=fit_weight)
        fit_x = apply_group_stats(fit_x, group_stats, global_stats)
        valid_x = apply_group_stats(valid_x, group_stats, global_stats)
        test_x = apply_group_stats(test_x, group_stats, global_stats)

    if "oof_target_encoding" in blocks:
        fit_x = add_inner_oof_target_encoding(
            fit_x,
            target=target,
            columns=TARGET_ENCODING_COLUMNS,
            sample_weight=fit_weight,
            alpha=80,
            n_splits=INNER_TE_SPLITS,
            suffix="te",
        )
        maps, prior = fit_target_maps(fit_x, TARGET_ENCODING_COLUMNS, target, fit_weight, alpha=80)
        valid_x = apply_target_maps(valid_x, maps, prior, suffix="te")
        test_x = apply_target_maps(test_x, maps, prior, suffix="te")

    return fit_x, valid_x, test_x

def run_oof_experiment(config, train_frame=None, test_frame=None, verbose=True):
    train_frame = cv_train if train_frame is None else train_frame
    test_frame = test if test_frame is None else test_frame
    y = train_frame[TARGET].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(train_frame), dtype=float)
    test_pred = np.zeros(len(test_frame), dtype=float)

    for fold, (fit_idx, valid_idx) in enumerate(skf.split(train_frame, y), start=1):
        fold_fit = train_frame.iloc[fit_idx].copy()
        fold_valid = train_frame.iloc[valid_idx].copy()

        fit_x, valid_x, test_x = fit_fold_features(
            fold_fit,
            fold_valid,
            test_frame,
            config.blocks,
            TARGET,
        )

        numeric_features = select_numeric_features(config.blocks, available_columns=fit_x.columns)
        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=select_categorical_features(config.include_driver_ohe),
            scale_numeric=(config.model_kind == "logreg"),
        )

        X_fit = preprocessor.fit_transform(fit_x)
        X_valid = preprocessor.transform(valid_x)
        X_test = preprocessor.transform(test_x)

        model = make_model(config.model_kind)
        valid_pred, fold_test_pred, fitted_model = fit_predict_model(
            config.model_kind,
            model,
            X_fit,
            fold_fit[TARGET].astype(int).to_numpy(),
            X_valid,
            fold_valid[TARGET].astype(int).to_numpy(),
            X_test,
        )
        oof[valid_idx] = valid_pred
        test_pred += fold_test_pred / N_SPLITS

    return {
        "config": config,
        "oof": oof,
        "test_pred": test_pred,
        "auc": roc_auc_score(y, oof),
        "logloss": log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)),
    }
```

</details>

## **6. Driver Ladder Results**

The Driver ladder is split into two branches:

| Branch | Question | Counterfactual |
|---|---|---|
| 🧑‍✈️ **Raw identity ladder** | Does Driver one-hot help directly? | no-Driver race-state foundation |
| 🧬 **Compressed ladder** | Can Driver signal survive without raw OHE? | same foundation, adding one Driver representation at a time |
| 🔀 **Interaction branch** | Are Driver/compound/stint priors stable? | the B04 target-encoding parent |

### **6.1 Raw Driver OHE**

| Experiment | Added Block | OOF AUC | Delta vs No Driver |
|---|---|---:|---:|
| `A00_no_driver` | Race-state foundation without raw Driver ID | 0.946243 | 0.000000 |
| `A01_driver_ohe` | Add raw Driver one-hot identity | 0.945964 | -0.000279 |

Raw Driver identity did **not** improve this saved CV run.
This is an important negative result because it prevents the notebook from over-trusting a high-cardinality ID.

### **6.2 Compressed Driver Signal**

| Experiment | Added Block | OOF AUC | Delta vs No Driver | CV Read |
|---|---|---:|---:|---|
| `B00_no_driver` | Race-state foundation | 0.946243 | 0.000000 | baseline |
| `B01_driver_string` | Driver string structure | 0.946691 | +0.000448 | improves |
| `B02_original_vocab` | original-driver vocabulary flag | 0.946586 | +0.000343 | below previous |
| `B03_driver_frequency` | Driver/Race frequency features | **0.946818** | **+0.000575** | best compressed step |
| `B04_driver_te` | inner-OOF Driver/Race target encoding | 0.946670 | +0.000427 | below previous |

The best saved Driver representation is:

$$
\texttt{B03\_driver\_frequency}
$$

with:

$$
\Delta_{\text{AUC}}
=
0.946818 - 0.946243
=
0.000575
$$

That is small, but it is directionally consistent in the paired bootstrap:

| Comparison | Mean Delta | P05 | Median | P95 | Probability Positive |
|---|---:|---:|---:|---:|---:|
| `B03_driver_frequency - B00` | 0.000573 | 0.000384 | 0.000572 | 0.000764 | 1.000 |

### **6.3 Interaction Target Encoding**

Interaction TE did not beat the `B04_driver_te` parent:

| Experiment | Parent | Delta vs Parent | Decision |
|---|---|---:|---|
| `B05_driver_compound_te` | `B04_driver_te` | -0.000261 | reject |
| `B06_race_compound_te` | `B04_driver_te` | -0.000076 | reject |
| `B07_race_stint_te` | `B04_driver_te` | -0.000143 | reject |
| `B08_compound_stint_te` | `B04_driver_te` | -0.000125 | reject |

This is the expected failure mode for sparse interactions.
The conditional prior is conceptually reasonable, but the sample support is thinner and the variance penalty dominates.

<details markdown="1">
<summary>Show notebook snippet: Driver ladder configuration and bootstrap check</summary>

```python
DRIVER_FOUNDATION_BLOCKS = [
    "race_tyre_algebra",
    "compound_structure",
    "nonlinear_interactions",
]

identity_driver_configs = [
    ExperimentConfig(
        "A00_no_driver",
        "Race-state foundation without raw Driver ID",
        DRIVER_FOUNDATION_BLOCKS,
        include_driver_ohe=False,
        model_kind=LADDER_MODEL,
    ),
    ExperimentConfig(
        "A01_driver_ohe",
        "Add raw Driver one-hot identity",
        DRIVER_FOUNDATION_BLOCKS,
        include_driver_ohe=True,
        model_kind=LADDER_MODEL,
    ),
]

compressed_driver_configs = [
    ExperimentConfig(
        "B00_no_driver",
        "Race-state foundation without raw Driver ID",
        DRIVER_FOUNDATION_BLOCKS,
        include_driver_ohe=False,
        model_kind=LADDER_MODEL,
    ),
    ExperimentConfig(
        "B01_driver_string",
        "Driver string structure without raw OHE",
        DRIVER_FOUNDATION_BLOCKS + ["driver_structure"],
        include_driver_ohe=False,
        model_kind=LADDER_MODEL,
    ),
    ExperimentConfig(
        "B03_driver_frequency",
        "Add Driver/Race frequency features",
        DRIVER_FOUNDATION_BLOCKS + ["driver_structure", "driver_original_vocab", "frequency_encoding"],
        include_driver_ohe=False,
        model_kind=LADDER_MODEL,
    ),
    ExperimentConfig(
        "B04_driver_te",
        "Add inner-OOF Driver/Race target encoding",
        DRIVER_FOUNDATION_BLOCKS + ["driver_structure", "driver_original_vocab", "frequency_encoding", "oof_target_encoding"],
        include_driver_ohe=False,
        model_kind=LADDER_MODEL,
    ),
]

def paired_auc_delta_bootstrap(y, pred_a, pred_b, comparison, n_boot=800, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    deltas = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], pred_b[idx]) - roc_auc_score(y[idx], pred_a[idx]))

    deltas = np.asarray(deltas, dtype=float)
    return pd.DataFrame([{
        "Comparison": comparison,
        "Mean_Delta": deltas.mean(),
        "P05": np.quantile(deltas, 0.05),
        "Median": np.quantile(deltas, 0.50),
        "P95": np.quantile(deltas, 0.95),
        "Prob_Positive": np.mean(deltas > 0),
        "Bootstraps": len(deltas),
    }])
```

</details>

## **7. Score-Oriented Candidate Stack**

After the Driver ladder, the notebook tests whether public-stack style preprocessing beats the current Driver-frequency selection.

Two matrix styles are compared:

| Candidate | Feature Count | OOF AUC | Delta vs Current Best |
|---|---:|---:|---:|
| `S05_stack_core_preprocessing` | 35 | 0.944899 | -0.0019 |
| `S06_stack_domain_preprocessing` | 111 | 0.946050 | -0.0008 |

The larger domain matrix improves over the core matrix, but neither beats `B03_driver_frequency` in this local LGBM run.
The interpretation is not that sequence/group features are bad.
It is that their value appears more compatible with the heavier XGB/Cat/LGBM stack than with the saved lightweight ladder.

The score gap table makes this explicit:

| Model or Stack | Reference OOF AUC | Gap vs `B03_driver_frequency` | Read |
|---|---:|---:|---|
| `B03_driver_frequency` | 0.946818 | 0.000000 | current local best |
| Domain LGBM public reference | 0.948120 | +0.001302 | small improvement |
| RealMLP public reference | 0.949510 | +0.002692 | small improvement |
| XGB baseline + original | 0.958160 | +0.011342 | leaderboard-scale gap |
| XGB Optuna + original | 0.959940 | +0.013122 | leaderboard-scale gap |
| XGB + CatBoost ensemble | 0.960760 | +0.013942 | leaderboard-scale gap |

This reframes Driver as an **auxiliary signal**.
The largest score movement is coming from model family, original data usage, and ensemble structure.

<details markdown="1">
<summary>Show notebook snippet: stack preprocessing candidates</summary>

```python
SCORE_FOUNDATION_BLOCKS = [
    "race_tyre_algebra",
    "compound_structure",
    "public_domain_features",
    "nonlinear_interactions",
]

SCORE_SAFE_DRIVER_BLOCKS = SCORE_FOUNDATION_BLOCKS + [
    "driver_structure",
    "driver_original_vocab",
    "frequency_encoding",
    "context_normalization",
]

score_candidate_configs = [
    ExperimentConfig(
        "S00_driver_ohe_domain",
        "Driver OHE + public pit-window domain features",
        SCORE_FOUNDATION_BLOCKS,
        include_driver_ohe=True,
        model_kind=LADDER_MODEL,
    ),
    ExperimentConfig(
        "S01_driver_ohe_context",
        "Add weighted support and field-relative context",
        SCORE_SAFE_DRIVER_BLOCKS,
        include_driver_ohe=True,
        model_kind=LADDER_MODEL,
    ),
    ExperimentConfig(
        "S02_driver_ohe_driver_te",
        "Add inner-OOF Driver/Race target encoding",
        SCORE_SAFE_DRIVER_BLOCKS + ["oof_target_encoding"],
        include_driver_ohe=True,
        model_kind=LADDER_MODEL,
    ),
]

def add_stack_domain_features(train_df, test_df):
    train_base = add_stack_row_features(train_df)
    test_base = add_stack_row_features(test_df)
    train_base["__part"] = "train"
    test_base["__part"] = "test"
    train_base["__row"] = np.arange(len(train_base))
    test_base["__row"] = np.arange(len(test_base))
    full = pd.concat([train_base, test_base], ignore_index=True, sort=False)

    sort_cols = [col for col in ["Race", "Driver", "Year", "Stint", "LapNumber"] if col in full.columns]
    full = full.sort_values(sort_cols).copy()
    group_cols = [col for col in ["Race", "Driver", "Year", "Stint"] if col in full.columns]
    g = full.groupby(group_cols, sort=False, observed=True)

    full["stack_lap_in_stint"] = g.cumcount() + 1
    full["stack_stint_len"] = g["LapNumber"].transform("count")
    full["stack_pos_prev1"] = g["Position"].shift(1)
    full["stack_pos_delta1"] = full["Position"] - full["stack_pos_prev1"]
    full["stack_lt_prev1"] = g["LapTime (s)"].shift(1)
    full["stack_lt_delta1"] = full["LapTime (s)"] - full["stack_lt_prev1"]
    full["stack_tyre_life_lag1"] = g["TyreLife"].shift(1)
    full["stack_tyre_life_diff1"] = full["TyreLife"] - full["stack_tyre_life_lag1"]
    full["stack_race_lap_meantyrelife"] = full.groupby(["Race", "LapNumber"], observed=True)["TyreLife"].transform("mean")
    full["stack_tyrelife_vs_field_mean"] = full["TyreLife"] - full["stack_race_lap_meantyrelife"]

    return full
```

</details>

## **8. Driver As A Residual Correction**

The second Driver hypothesis is subtler:

> Driver may be too weak as a main feature, but still useful as a post-model calibration bias.

The residual layer works in logit space.
For a base prediction \(\hat p_i\):

$$
\operatorname{logit}(\hat p_i)
=
\log \frac{\hat p_i}{1-\hat p_i}
$$

For a Driver group \(g\), the notebook estimates a smoothed observed rate and a smoothed predicted rate:

$$
\tilde y_g =
\frac{\sum_{i:g_i=g} y_i + \alpha \bar y}{n_g+\alpha},
\qquad
\tilde p_g =
\frac{\sum_{i:g_i=g} \hat p_i + \alpha \bar p}{n_g+\alpha}
$$

The group residual is:

$$
\delta_g =
\operatorname{logit}(\tilde y_g)
-
\operatorname{logit}(\tilde p_g)
$$

Then the corrected prediction is:

$$
\hat p'_i
=
\sigma\!\left(
\operatorname{logit}(\hat p_i)
+
\lambda
\sum_{g \in G_i} w_g\delta_g
\right)
$$

where \(\lambda\) is selected by OOF AUC.

The saved local residual table:

| Experiment | Groups | Lambda | OOF AUC | Delta vs Base | Decision |
|---|---|---:|---:|---:|---|
| `R00_base_selected_stack` | none | 0.00 | 0.946818 | 0.000000 | base |
| `R01_driver_logit_bias` | Driver | -0.25 | 0.946820 | +0.000002 | watchlist |
| `R02_driver_phase_logit_bias` | Driver + Driver_Phase | -0.25 | 0.946820 | +0.000002 | watchlist |
| `R05_driver_strategy_logit_bias` | Driver + phase + compound + stint | -0.10 | 0.946819 | +0.000001 | watchlist |

The promotion threshold is:

$$
\Delta_{\text{AUC}} \ge 0.0003
$$

The local residual gains are far below that threshold, so they remain watchlist signals rather than final candidates.

<details markdown="1">
<summary>Show notebook snippet: logit residual correction</summary>

```python
def logit_clip(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

def fit_logit_residual_maps(residual_frame, y, base_pred, group_alphas):
    maps = {}
    y = np.asarray(y, dtype=float)
    base_pred = np.clip(np.asarray(base_pred, dtype=float), 1e-6, 1 - 1e-6)
    global_y = float(np.mean(y))
    global_p = float(np.mean(base_pred))

    for col, alpha in group_alphas.items():
        tmp = pd.DataFrame({
            "key": residual_frame[col].astype(str),
            "y": y,
            "p": base_pred,
        })
        grouped = tmp.groupby("key", observed=True).agg(
            n=("y", "size"),
            y_sum=("y", "sum"),
            p_sum=("p", "sum"),
        )
        grouped["y_smooth"] = (grouped["y_sum"] + alpha * global_y) / (grouped["n"] + alpha)
        grouped["p_smooth"] = (grouped["p_sum"] + alpha * global_p) / (grouped["n"] + alpha)
        grouped["delta_logit"] = logit_clip(grouped["y_smooth"]) - logit_clip(grouped["p_smooth"])
        maps[col] = grouped[["delta_logit", "n"]]
    return maps

def evaluate_logit_residual_spec(spec, train_resid_frame, test_resid_frame, y, base_oof, base_test):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 100)
    delta_oof = np.zeros(len(train_resid_frame), dtype=float)
    delta_test = np.zeros(len(test_resid_frame), dtype=float)

    for fold, (fit_idx, valid_idx) in enumerate(skf.split(train_resid_frame, y), start=1):
        maps = fit_logit_residual_maps(
            train_resid_frame.iloc[fit_idx],
            y[fit_idx],
            base_oof[fit_idx],
            spec["alphas"],
        )
        delta_oof[valid_idx] = apply_logit_residual_signal(
            train_resid_frame.iloc[valid_idx],
            maps,
            spec["groups"],
        )
        delta_test += apply_logit_residual_signal(test_resid_frame, maps, spec["groups"]) / N_SPLITS

    base_logit_oof = logit_clip(base_oof)
    lambda_grid = np.array([-1.00, -0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75, 1.00])

    rows = []
    for lam in lambda_grid:
        corrected_oof = sigmoid_array(base_logit_oof + lam * delta_oof)
        rows.append({
            "Lambda": lam,
            "OOF_AUC": roc_auc_score(y, corrected_oof),
        })

    strength_table = pd.DataFrame(rows)
    best_lambda = float(strength_table.loc[strength_table["OOF_AUC"].idxmax(), "Lambda"])
    corrected_oof = sigmoid_array(base_logit_oof + best_lambda * delta_oof)
    return corrected_oof, strength_table
```

</details>

## **9. Heavy-Stack Overlay**

The final section attaches stronger OOF/test artifacts and asks whether Driver residuals add value on top of the real scoring base.

The stack search uses probability and rank blends:

$$
\hat p^{\text{prob}}_i
=
\sum_{m=1}^{M} w_m \hat p_{im},
\qquad
w_m \ge 0,\quad \sum_m w_m = 1
$$

For rank blending:

$$
\hat r_i
=
\sum_{m=1}^{M} w_m
\operatorname{rank}_{01}(\hat p_{im})
$$

The discovered base models had these OOF scores:

| Model | OOF AUC |
|---|---:|
| `realmlp_full_fullrows_5fold` | 0.952177 |
| `lgb_domain_w07_full_fullrows_10fold` | 0.950665 |
| `xgb_core_w1_full_fullrows_10fold` | 0.950331 |
| `cat_core_w1_full_fullrows_10fold` | 0.949682 |
| `xgb_public_te_w1_full_fullrows_10fold` | 0.949381 |

The selected heavy-stack artifact reached:

$$
\operatorname{AUC}_{\text{heavy stack}}
\approx
0.953404
$$

The Driver residual overlay reached about:

$$
0.953409
$$

but the improvement was not large enough to beat the promotion threshold.
So the final blend assigned all weight to the heavy-stack artifact:

| Model | Blend Weight |
|---|---:|
| `B03_driver_frequency` | 0.000 |
| `H00_heavy_stack_artifact` | 1.000 |

This is a useful outcome.
It says Driver frequency is a valid local signal, but once stronger full-row stack predictions are available, Driver residuals do not yet justify replacing the artifact base.

<details markdown="1">
<summary>Show notebook snippet: artifact blending and heavy-stack residual overlay</summary>

```python
def rank01_np(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks / (len(values) + 1)

def simplex_grid(n, step):
    units = int(round(1 / step))
    current = []

    def rec(left, slots):
        if slots == 1:
            yield current + [left]
        else:
            for value in range(left + 1):
                current.append(value)
                yield from rec(left - value, slots - 1)
                current.pop()

    for parts in rec(units, n):
        yield np.array(parts, dtype=float) / units

def search_stack_base_blends(names, oof_matrix, test_matrix, y):
    rank_oof = np.column_stack([rank01_np(oof_matrix[:, idx]) for idx in range(oof_matrix.shape[1])])
    rank_test = np.column_stack([rank01_np(test_matrix[:, idx]) for idx in range(test_matrix.shape[1])])
    candidates = []

    for mode, matrix_oof, matrix_test in [
        ("probability", oof_matrix, test_matrix),
        ("rank", rank_oof, rank_test),
    ]:
        best_auc = -np.inf
        best_weights = None
        best_oof = None
        best_test = None

        for weights in simplex_grid(len(names), STACK_BLEND_GRID_STEP):
            pred = matrix_oof @ weights
            auc = roc_auc_score(y, pred)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights.copy()
                best_oof = pred
                best_test = matrix_test @ weights

        candidates.append({
            "Mode": mode,
            "OOF_AUC": best_auc,
            "Weights": best_weights,
            "OOF": best_oof,
            "Test": best_test,
        })

    best_candidate = max(candidates, key=lambda item: item["OOF_AUC"])
    probability_candidate = next(item for item in candidates if item["Mode"] == "probability")
    return best_candidate, probability_candidate

def run_heavy_stack_residual_overlay(stack_train_frame, stack_test_frame, y, probability_base, best_artifact_base):
    spec_results = [
        evaluate_logit_residual_spec(
            spec,
            build_residual_frame(stack_train_frame),
            build_residual_frame(stack_test_frame),
            y,
            np.clip(probability_base["OOF"], 1e-6, 1 - 1e-6),
            np.clip(probability_base["Test"], 1e-6, 1 - 1e-6),
        )
        for spec in RESIDUAL_SPECS
    ]

    best_residual = max(spec_results, key=lambda item: item["auc"])
    if best_residual["auc"] >= best_artifact_base["OOF_AUC"] + RESIDUAL_PROMOTION_MIN_AUC:
        selected_oof = best_residual["oof"]
        selected_test = best_residual["test_pred"]
        selected_decision = "promoted residual"
    else:
        selected_oof = best_artifact_base["OOF"]
        selected_test = best_artifact_base["Test"]
        selected_decision = "artifact base"

    return selected_oof, selected_test, selected_decision
```

</details>

## **10. Final Submission Logic**

The final file is a standard probability submission.
The notebook checks row count, ID alignment, probability bounds, and missing values:

$$
0 \le \hat p_i \le 1
$$

The saved output summary:

| Rows | Mean Prediction | Min Prediction | Max Prediction | Filename |
|---:|---:|---:|---:|---|
| 188,165 | 0.1982 | 0.0001 | 0.9954 | `submission.csv` |

<details markdown="1">
<summary>Show notebook snippet: final submission checks</summary>

```python
submission = submission_template.copy()
submission[SUBMISSION_TARGET] = np.clip(blend_test, 0, 1)

assert submission.shape[0] == test.shape[0]
assert submission[ID_COL].equals(test[ID_COL])
assert submission[SUBMISSION_TARGET].between(0, 1).all()
assert submission[SUBMISSION_TARGET].isna().sum() == 0

submission.to_csv(SUBMISSION_FILENAME, index=False)

submission_summary = pd.DataFrame({
    "Rows": [len(submission)],
    "Mean_Prediction": [submission[SUBMISSION_TARGET].mean()],
    "Min_Prediction": [submission[SUBMISSION_TARGET].min()],
    "Max_Prediction": [submission[SUBMISSION_TARGET].max()],
    "Filename": [SUBMISSION_FILENAME],
})
```

</details>

## **11. What The Notebook Establishes**

The most useful conclusion is not simply "Driver helps" or "Driver does not help."
The more precise reading is:

| Claim | Evidence | Decision |
|---|---|---|
| Raw Driver one-hot is not enough | `A01 - A00 = -0.000279` AUC | reject in saved CV |
| Compressed Driver support is useful | `B03 - B00 = +0.000575` AUC | keep as local signal |
| Target encoding needs strict guardrails | `B04` falls below `B03` | do not promote by complexity alone |
| Interaction TE is high variance | all tested interactions fall below the parent | reject interaction branches |
| Public-style stack features need stronger models | local LGBM stack candidates trail `B03` | keep for heavier stack |
| Driver residual is not promoted | gains are around a few \(10^{-6}\) AUC | watchlist only |
| Heavy-stack artifact dominates final blend | selected blend weight is 1.0 on artifact stack | final submission base |

### **11.1 Statistical Meaning Of The Conclusion**

The conclusion should not be read as a causal claim about driver skill.
The notebook does **not** establish that a particular driver is intrinsically better at pit strategy.
What it establishes is narrower:

> After race-state, tyre-state, compound, stint, and degradation features are already present, Driver-derived support and frequency variables still contain a small amount of additional OOF ranking signal.

The relevant marginal effect is:

$$
\Delta_{\text{Driver frequency}}
=
\operatorname{AUC}(\text{race-state} + \text{Driver frequency})
-
\operatorname{AUC}(\text{race-state only})
$$

In the saved run:

$$
\Delta_{\text{Driver frequency}}
=
0.946818 - 0.946243
=
0.000575
$$

So the statistically careful interpretation is:

> **Driver frequency has small marginal predictive utility under the controlled OOF protocol.**

This is different from saying that raw Driver identity is useful.
Raw one-hot Driver identity moved in the wrong direction:

$$
\Delta_{\text{raw Driver OHE}}
=
0.945964 - 0.946243
=
-0.000279
$$

That negative result matters.
It suggests that a high-cardinality Driver ID can increase variance or memorize unstable category effects, while lower-variance summaries such as support counts and frequency logs are more robust.

Statistically, the surviving Driver signal is closer to a **support-size / reliability proxy** than a direct identity effect:

$$
\text{Driver} \not\Rightarrow \text{skill claim}
\qquad
\text{Driver frequency} \Rightarrow \text{category support and generator-context signal}
$$

The residual-correction result gives the second half of the interpretation.
Once the stronger stack artifact is used as the base, Driver residual corrections do not clear the promotion threshold:

$$
\operatorname{AUC}_{\text{corrected}}
-
\operatorname{AUC}_{\text{base}}
<
0.0003
$$

This means the small Driver signal is either already absorbed by the stronger model stack or too weak to survive as an independent post-stack correction.
Therefore, Driver is best interpreted as a **low-amplitude auxiliary feature**, not as a primary modeling axis.

The notebook's statistical structure is therefore:

$$
\text{Driver ID}
\rightarrow
\text{Driver compression}
\rightarrow
\text{fold-safe Driver prior}
\rightarrow
\text{residual Driver correction}
\rightarrow
\text{heavy-stack comparison}
$$

The important part is the **counterfactual discipline**.
Every Driver idea must answer:

$$
\operatorname{AUC}_{\text{candidate}}
-
\operatorname{AUC}_{\text{matching counterfactual}}
> 0
$$

and for residual overlays:

$$
\operatorname{AUC}_{\text{corrected}}
-
\operatorname{AUC}_{\text{base}}
\ge
0.0003
$$

That is why the final interpretation is conservative:

> **Driver frequency survives as a small, measurable signal. Driver residual correction does not yet survive promotion once the stronger stack artifact is the base.**
