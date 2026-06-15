---
title: "S6E5 Driver's High: Driver 피처의 잔여 신호 — OOF Ladder 검증"
date: 2026-05-03 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, playground-series, feature-engineering, cross-validation, auc, formula1, korean]
math: true
pin: false
---

# S6E5 Driver's High: Driver 피처의 잔여 신호 — OOF Ladder 검증

Kaggle 노트북 링크:  
[S6E5 Driver's High: Driver Feature Engineering](https://www.kaggle.com/code/pilkwang/s6e5-driver-s-high-driver-feature-eng)

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-05-03-s6e5-drivers-high/cover.png" alt="S6E5 Driver's High cover" width="80%">
</p>

이 노트북의 질문은 좁고 명확합니다.

> **race state, tyre state, compound, stint 정보를 이미 넣은 뒤에도 `Driver`에는 따로 측정 가능한 신호가 남아 있는가?**

Pit stop 예측에서 `Driver`는 매력적인 변수입니다. 실제 전략을 수행하는 주체처럼 보이기 때문입니다. 하지만 통계적으로는 위험한 변수이기도 합니다. Driver는 고카디널리티 categorical feature이고, 잘못 쓰면 재사용 가능한 신호가 아니라 category memorization을 학습할 수 있습니다.

따라서 Driver feature engineering은 하나의 큰 feature를 추가하는 일이 아니라, 위험도가 다른 표현을 차례대로 검증하는 과정입니다.

$$
\Delta_{\text{AUC}}(B)
=
\operatorname{AUC}\!\left(\hat p_{\text{base}+B}^{\text{oof}}, y\right)
-
\operatorname{AUC}\!\left(\hat p_{\text{counterfactual}}^{\text{oof}}, y\right)
$$

여기서 \(B\)는 후보 feature block입니다. 중요한 건 단순히 AUC가 오르는지 보는 게 아니라, 같은 OOF protocol 아래에서 올바른 counterfactual보다 좋아지는지입니다.

## **1. 모델링 질문**

Target은 다음 lap에 pit stop을 하는지 여부입니다.

$$
y_i =
\begin{cases}
1, & \text{pit next lap} \\
0, & \text{otherwise}
\end{cases}
$$

모델은 다음 확률을 예측합니다.

$$
\hat p_i = P(y_i = 1 \mid x_i)
$$

평가 지표는 ROC-AUC입니다. AUC는 양성 row가 음성 row보다 더 높은 점수를 받을 확률로 해석할 수 있습니다.

$$
\operatorname{AUC}
=
P(\hat p^+ > \hat p^-)
+ \frac{1}{2}P(\hat p^+ = \hat p^-)
$$

따라서 Driver의 가치는 어떤 row의 확률을 조금 바꾸는 데 있지 않습니다. 양성 row와 음성 row의 **상대적 순위**를 더 잘 만들 수 있는지가 핵심입니다.

| 층 | 역할 | 통계적 위험 |
|---|---|---|
| **Race-state foundation** | lap, progress, stint, tyre age, degradation | race state가 약하면 underfit이 생깁니다. |
| **Compound and tyre pressure** | compound별 기대 수명 대비 tyre age | compound 효과를 raw ID처럼만 쓰면 scale이 어긋납니다. |
| **Driver representation** | string structure, frequency, target prior, interaction | 고카디널리티 누수 또는 memorization |
| **OOF protocol** | 같은 split, 한 번에 하나의 block만 변경 | validation row가 자기 target을 encode하면 false lift가 생깁니다. |
| **Stack / residual layer** | Driver 신호를 더 강한 artifact와 비교 | 실제 gain이 model family나 original data에서 왔는데 Driver에 과대 부여할 수 있습니다. |

핵심 설계는 `Driver`를 하나의 feature로 통째로 넣지 않는 것입니다. 대신 안전한 표현부터 위험한 표현까지 ladder 형태로 분리합니다.

## **2. Driver Inventory**

먼저 Driver가 현실의 작은 category인지, 아니면 synthetic high-cardinality category인지 확인합니다.

| Dataset | Rows | Unique Drivers | `D...` Code Rate | 3-Letter Rate |
|---|---:|---:|---:|---:|
| Train | 439,140 | 887 | 59.58% | 40.42% |
| Test | 188,165 | 801 | 59.34% | 40.66% |
| Original | 101,371 | 31 | 0.00% | 100.00% |

Train/test에는 `D###` 형태의 synthetic ID가 많고, original data는 3-letter driver code로 구성되어 있습니다. 이 차이는 중요합니다. 같은 `Driver` 컬럼이라도 두 데이터 소스의 의미가 완전히 같지 않기 때문입니다.

Driver signal은 다음처럼 나눠 볼 수 있습니다.

| Driver Signal | 의미 | Leakage Risk |
|---|---|---|
| **String structure** | `D###`인지 3-letter code인지, numeric ID bin | 낮음 |
| **Original vocabulary flag** | original data에 등장한 Driver인지 | 낮음에서 중간 |
| **Frequency/support** | Driver/Race context 뒤에 있는 row 수 | 낮음 |
| **Target encoding** | category별 smoothed pit-stop prior | fold-safe가 아니면 높음 |
| **Interaction target encoding** | Driver-Compound, Race-Stint, Compound-Stint prior | variance가 더 큼 |

Support distribution도 함께 봐야 합니다.

| Threshold | Drivers | Covered Rows | Covered Row Rate |
|---|---:|---:|---:|
| `>= 10 rows` | 666 | 438,432 | 99.84% |
| `>= 50 rows` | 535 | 435,485 | 99.17% |
| `>= 100 rows` | 485 | 431,979 | 98.37% |
| `>= 250 rows` | 411 | 419,501 | 95.53% |
| `>= 500 rows` | 337 | 392,580 | 89.40% |

대부분의 row는 충분한 support를 가진 driver에 속합니다. 그래서 frequency feature는 통계적으로 의미가 있습니다. 다만 long tail도 충분히 크기 때문에, raw one-hot identity는 여전히 overfit하기 쉽습니다.

<details markdown="1">
<summary>코드: 데이터 로딩과 Driver inventory</summary>

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

Driver를 측정하기 전에 non-Driver baseline이 충분히 강해야 합니다. 그렇지 않으면 모델은 race state가 설명해야 할 변동을 Driver에게 잘못 부여할 수 있습니다.

먼저 race length를 추정합니다.

$$
\widehat L_i =
\frac{\text{LapNumber}_i}{\text{RaceProgress}_i}
$$

남은 race state와 tyre life scale은 다음처럼 만듭니다.

$$
R_i = 1 - \text{RaceProgress}_i,
\qquad
\text{NormalizedTyreLife}_i =
\frac{\text{TyreLife}_i}{\widehat L_i}
$$

Compound별 tyre pressure는 compound 기대 수명 대비 tyre age로 표현합니다.

$$
u_i =
\frac{\text{TyreLife}_i}{E[\text{Life}\mid \text{Compound}_i]}
$$

Pit window indicator는 다음과 같습니다.

$$
\mathbb{1}\!\left(u_i \ge w_{\text{Compound}_i}\right)
$$

이렇게 하면 모든 `TyreLife` 값을 같은 scale로 보지 않고, compound별 기대 수명에 맞춰 비교할 수 있습니다.

| Feature Block | 주요 변수 | 해석 |
|---|---|---|
| **Race/tyre algebra** | estimated race laps, laps remaining, normalized tyre life | race phase와 tyre age scale |
| **Compound structure** | hardness, dry/wet-like flags, tyre-life × hardness | compound별 wear pressure |
| **Public domain features** | pit window, degradation per lap, stop urgency | F1 strategy pressure |
| **Nonlinear interactions** | tyre life × progress, tyre life × stint, degradation × age | threshold-like pit behavior |

<details markdown="1">
<summary>코드: race-state와 compound feature block</summary>

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

Driver ladder는 안전한 정보에서 시작해 점점 위험한 encoding으로 넘어갑니다.

### **4.1 String Structure**

가장 안전한 표현은 raw one-hot identity가 아니라 문자열 구조입니다.

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

이 표현은 driver마다 dummy variable을 만들지 않으면서 synthetic ID와 original vocabulary의 구조 차이를 잡습니다.

### **4.2 Frequency Encoding**

Category key \(c\)에 대해 frequency encoding은 다음 값을 추가합니다.

$$
n(c) = \sum_i \mathbb{1}(x_i = c),
\qquad
f(c) = \log(1+n(c))
$$

Rare driver flag도 둡니다.

$$
\mathbb{1}\{n(\text{Driver}) < 50\}
$$

이 feature는 “많이 등장한 driver가 더 좋다”는 뜻이 아닙니다. 해당 category estimate가 얼마나 많은 support 위에 있는지를 모델에게 알려주는 역할입니다.

### **4.3 Context Normalization**

Race-relative z-score는 row를 자기 주변 context와 비교합니다.

$$
z_{i,g,v}
=
\frac{x_{i,v} - \mu_{g(i),v}}{\sigma_{g(i),v} + \epsilon}
$$

여기서 \(g(i)\)는 `Race`, `Race_Compound`, `Race_Lap`, `Compound` 등이 될 수 있습니다. Pit timing은 절대값만으로 결정되지 않습니다. 어떤 race/lap에서는 오래된 tyre가 평범할 수 있고, 다른 context에서는 곧바로 pit pressure가 될 수 있습니다.

### **4.4 Target Encoding**

Category \(c\)에 대한 smoothed target prior는 다음과 같습니다.

$$
\operatorname{TE}(c)
=
\frac{\sum_{i:g_i=c} w_i y_i + \alpha \bar y}
{\sum_{i:g_i=c} w_i + \alpha}
$$

Smoothing parameter \(\alpha\)는 global mean으로 shrink하는 힘입니다. `Driver_Compound`처럼 support가 얇은 interaction에서는 큰 \(\alpha\)가 필요합니다.

<details markdown="1">
<summary>코드: Driver structure, count map, target encoding</summary>

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

가장 중요한 구현 조건은 모든 fitted transformation을 fold 안에서만 학습하는 것입니다.

Fold \(k\)에 대해:

$$
\mathcal{D}_{-k} = \text{training rows outside fold } k,
\qquad
\mathcal{D}_{k} = \text{validation rows inside fold } k
$$

Fitted encoder \(T_k\)는 반드시 다음 조건을 만족해야 합니다.

$$
T_k = \operatorname{fit}(\mathcal{D}_{-k}),
\qquad
X_k^{\text{valid}} = T_k(\mathcal{D}_k)
$$

Target encoding은 training side에도 inner OOF encoding이 필요합니다. 그렇지 않으면 한 row가 자기 자신의 target을 encode할 수 있습니다.

$$
\hat z_i =
\operatorname{TE}_{-h(i)}(x_i)
$$

여기서 \(h(i)\)는 row \(i\)가 속한 inner fold입니다.

<details markdown="1">
<summary>코드: fold-safe feature fitting과 OOF runner</summary>

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
```

</details>

## **6. Driver Ladder Results**

Driver ladder는 세 갈래로 나뉩니다.

| Branch | 질문 | Counterfactual |
|---|---|---|
| **Raw identity ladder** | Driver one-hot이 직접 도움이 되는가? | Driver 없는 race-state foundation |
| **Compressed ladder** | raw OHE 없이도 Driver 신호가 살아남는가? | 같은 foundation에 Driver 표현을 하나씩 추가 |
| **Interaction branch** | Driver/compound/stint prior가 안정적인가? | B04 target-encoding parent |

### **6.1 Raw Driver OHE**

| Experiment | Added Block | OOF AUC | Delta vs No Driver |
|---|---|---:|---:|
| `A00_no_driver` | Race-state foundation without raw Driver ID | 0.946243 | 0.000000 |
| `A01_driver_ohe` | Add raw Driver one-hot identity | 0.945964 | -0.000279 |

Raw Driver identity는 이 saved CV run에서 개선을 만들지 못했습니다. 이 음수 결과는 중요합니다. 고카디널리티 ID를 그대로 믿으면 variance나 memorization을 키울 수 있기 때문입니다.

### **6.2 Compressed Driver Signal**

| Experiment | Added Block | OOF AUC | Delta vs No Driver | CV Read |
|---|---|---:|---:|---|
| `B00_no_driver` | Race-state foundation | 0.946243 | 0.000000 | baseline |
| `B01_driver_string` | Driver string structure | 0.946691 | +0.000448 | improves |
| `B02_original_vocab` | original-driver vocabulary flag | 0.946586 | +0.000343 | below previous |
| `B03_driver_frequency` | Driver/Race frequency features | **0.946818** | **+0.000575** | best compressed step |
| `B04_driver_te` | inner-OOF Driver/Race target encoding | 0.946670 | +0.000427 | below previous |

가장 좋은 saved Driver representation은 다음과 같습니다.

$$
\texttt{B03\_driver\_frequency}
$$

개선폭은 작지만 방향은 안정적입니다.

$$
\Delta_{\text{AUC}}
=
0.946818 - 0.946243
=
0.000575
$$

Paired bootstrap에서도 같은 방향이 확인됩니다.

| Comparison | Mean Delta | P05 | Median | P95 | Probability Positive |
|---|---:|---:|---:|---:|---:|
| `B03_driver_frequency - B00` | 0.000573 | 0.000384 | 0.000572 | 0.000764 | 1.000 |

### **6.3 Interaction Target Encoding**

Interaction TE는 `B04_driver_te` parent를 넘지 못했습니다.

| Experiment | Parent | Delta vs Parent | Decision |
|---|---|---:|---|
| `B05_driver_compound_te` | `B04_driver_te` | -0.000261 | reject |
| `B06_race_compound_te` | `B04_driver_te` | -0.000076 | reject |
| `B07_race_stint_te` | `B04_driver_te` | -0.000143 | reject |
| `B08_compound_stint_te` | `B04_driver_te` | -0.000125 | reject |

Sparse interaction에서 흔히 나오는 실패입니다. 조건부 prior 자체는 그럴듯하지만, support가 얇아지면서 variance penalty가 더 커집니다.

## **7. Score-Oriented Candidate Stack**

Driver ladder 이후에는 public-stack style preprocessing이 현재 Driver-frequency selection을 넘는지 확인합니다.

| Candidate | Feature Count | OOF AUC | Delta vs Current Best |
|---|---:|---:|---:|
| `S05_stack_core_preprocessing` | 35 | 0.944899 | -0.0019 |
| `S06_stack_domain_preprocessing` | 111 | 0.946050 | -0.0008 |

Domain matrix는 core matrix보다 낫지만, 이 local LGBM run에서는 `B03_driver_frequency`를 넘지 못했습니다. Sequence/group feature가 나쁘다는 뜻은 아닙니다. 이 feature들은 lightweight ladder보다 XGB/Cat/LGBM heavy stack과 더 잘 맞는 신호일 수 있습니다.

Score gap을 보면 Driver의 위치가 더 명확해집니다.

| Model or Stack | Reference OOF AUC | Gap vs `B03_driver_frequency` | 해석 |
|---|---:|---:|---|
| `B03_driver_frequency` | 0.946818 | 0.000000 | current local best |
| Domain LGBM public reference | 0.948120 | +0.001302 | small improvement |
| RealMLP public reference | 0.949510 | +0.002692 | small improvement |
| XGB baseline + original | 0.958160 | +0.011342 | leaderboard-scale gap |
| XGB Optuna + original | 0.959940 | +0.013122 | leaderboard-scale gap |
| XGB + CatBoost ensemble | 0.960760 | +0.013942 | leaderboard-scale gap |

따라서 Driver는 primary axis라기보다 auxiliary signal에 가깝습니다. 더 큰 점수 이동은 model family, original data usage, ensemble structure에서 나옵니다.

## **8. Driver As A Residual Correction**

두 번째 가설은 조금 더 섬세합니다.

> Driver는 main feature로는 약해도, post-model calibration bias로는 쓸 수 있을지 모른다.

Residual layer는 logit space에서 작동합니다. Base prediction \(\hat p_i\)에 대해:

$$
\operatorname{logit}(\hat p_i)
=
\log \frac{\hat p_i}{1-\hat p_i}
$$

Driver group \(g\)마다 smoothed observed rate와 smoothed predicted rate를 비교합니다.

$$
\tilde y_g =
\frac{\sum_{i:g_i=g} y_i + \alpha \bar y}{n_g+\alpha},
\qquad
\tilde p_g =
\frac{\sum_{i:g_i=g} \hat p_i + \alpha \bar p}{n_g+\alpha}
$$

Group residual은 다음과 같습니다.

$$
\delta_g =
\operatorname{logit}(\tilde y_g)
-
\operatorname{logit}(\tilde p_g)
$$

Corrected prediction은:

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

\(\lambda\)는 OOF AUC로 선택합니다.

Saved local residual table은 다음과 같습니다.

| Experiment | Groups | Lambda | OOF AUC | Delta vs Base | Decision |
|---|---|---:|---:|---:|---|
| `R00_base_selected_stack` | none | 0.00 | 0.946818 | 0.000000 | base |
| `R01_driver_logit_bias` | Driver | -0.25 | 0.946820 | +0.000002 | watchlist |
| `R02_driver_phase_logit_bias` | Driver + Driver_Phase | -0.25 | 0.946820 | +0.000002 | watchlist |
| `R05_driver_strategy_logit_bias` | Driver + phase + compound + stint | -0.10 | 0.946819 | +0.000001 | watchlist |

Promotion threshold는 다음과 같습니다.

$$
\Delta_{\text{AUC}} \ge 0.0003
$$

Local residual gain은 이 기준보다 훨씬 작습니다. 따라서 final candidate가 아니라 watchlist signal로 남습니다.

## **9. Heavy-Stack Overlay**

마지막 단계에서는 더 강한 OOF/test artifact 위에 Driver residual을 얹어 봅니다. 여기서 질문은 바뀝니다.

```text
Driver signal이 lightweight ladder에서는 살아남았지만,
heavy stack 위에서도 독립적인 correction으로 남는가?
```

Stack search는 probability blend와 rank blend를 모두 봅니다.

$$
\hat p^{\text{prob}}_i
=
\sum_{m=1}^{M} w_m \hat p_{im},
\qquad
w_m \ge 0,\quad \sum_m w_m = 1
$$

Rank blend는 다음입니다.

$$
\hat r_i
=
\sum_{m=1}^{M} w_m
\operatorname{rank}_{01}(\hat p_{im})
$$

발견된 base model들의 OOF score는 다음과 같습니다.

| Model | OOF AUC |
|---|---:|
| `realmlp_full_fullrows_5fold` | 0.952177 |
| `lgb_domain_w07_full_fullrows_10fold` | 0.950665 |
| `xgb_core_w1_full_fullrows_10fold` | 0.950331 |
| `cat_core_w1_full_fullrows_10fold` | 0.949682 |
| `xgb_public_te_w1_full_fullrows_10fold` | 0.949381 |

선택된 heavy-stack artifact는 다음 수준에 도달했습니다.

$$
\operatorname{AUC}_{\text{heavy stack}}
\approx
0.953404
$$

Driver residual overlay는 약:

$$
0.953409
$$

까지 올라갔지만, promotion threshold를 넘을 만큼 충분하지는 않았습니다. 그래서 final blend는 모든 weight를 heavy-stack artifact에 둡니다.

| Model | Blend Weight |
|---|---:|
| `B03_driver_frequency` | 0.000 |
| `H00_heavy_stack_artifact` | 1.000 |

이 결과는 유용합니다. Driver frequency는 local signal로는 유효하지만, 더 강한 full-row stack prediction이 들어오면 Driver residual이 artifact base를 대체할 만큼 독립적인 개선을 만들지는 못한다는 뜻입니다.

## **10. Final Submission Logic**

최종 파일은 표준 probability submission입니다. Row count, ID alignment, probability bounds, missing value를 확인합니다.

$$
0 \le \hat p_i \le 1
$$

저장된 제출 요약은 다음과 같습니다.

| Rows | Mean Prediction | Min Prediction | Max Prediction | Filename |
|---:|---:|---:|---:|---|
| 188,165 | 0.1982 | 0.0001 | 0.9954 | `submission.csv` |

<details markdown="1">
<summary>코드: final submission check</summary>

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

## **11. 결론**

결론은 “Driver가 중요하다” 또는 “Driver가 중요하지 않다”처럼 단순하지 않습니다. 더 정확한 해석은 다음과 같습니다.

| Claim | Evidence | Decision |
|---|---|---|
| Raw Driver one-hot은 충분하지 않습니다 | `A01 - A00 = -0.000279` AUC | reject in saved CV |
| Compressed Driver support는 유용합니다 | `B03 - B00 = +0.000575` AUC | keep as local signal |
| Target encoding은 강한 guardrail이 필요합니다 | `B04`가 `B03`보다 낮음 | complexity만으로 promote하지 않음 |
| Interaction TE는 variance가 큽니다 | 모든 interaction이 parent보다 낮음 | reject interaction branches |
| Public-style stack feature는 더 강한 모델과 맞습니다 | local LGBM stack candidates가 `B03`보다 낮음 | heavier stack에서 유지 |
| Driver residual은 promote되지 않습니다 | gain이 몇 \(10^{-6}\) AUC 수준 | watchlist only |
| Heavy-stack artifact가 final blend를 지배합니다 | artifact stack weight 1.0 | final submission base |

### **11.1 통계적 의미**

이 결과를 driver skill에 대한 causal claim으로 읽으면 안 됩니다. 특정 driver가 pit strategy를 본질적으로 더 잘한다는 걸 증명한 게 아닙니다.

더 좁은 결론은 다음과 같습니다.

> Race-state, tyre-state, compound, stint, degradation feature가 이미 들어간 뒤에도, Driver-derived support와 frequency variable에는 작은 OOF ranking signal이 남아 있다.

관련 marginal effect는 다음과 같습니다.

$$
\Delta_{\text{Driver frequency}}
=
\operatorname{AUC}(\text{race-state} + \text{Driver frequency})
-
\operatorname{AUC}(\text{race-state only})
$$

Saved run에서는:

$$
\Delta_{\text{Driver frequency}}
=
0.946818 - 0.946243
=
0.000575
$$

따라서 조심스러운 통계적 해석은 다음과 같습니다.

> **Driver frequency는 controlled OOF protocol 아래에서 작지만 측정 가능한 marginal predictive utility를 갖는다.**

이건 raw Driver identity가 유용하다는 말과 다릅니다. Raw one-hot Driver identity는 오히려 나쁜 방향으로 움직였습니다.

$$
\Delta_{\text{raw Driver OHE}}
=
0.945964 - 0.946243
=
-0.000279
$$

이 음수 결과가 중요합니다. 고카디널리티 Driver ID는 불안정한 category effect를 외우거나 variance를 키울 수 있습니다. 반면 support count와 frequency log처럼 variance가 낮은 summary는 더 robust합니다.

즉 살아남은 Driver signal은 직접적인 identity effect라기보다, **support-size / reliability proxy**에 가깝습니다.

$$
\text{Driver} \not\Rightarrow \text{skill claim}
\qquad
\text{Driver frequency} \Rightarrow \text{category support and generator-context signal}
$$

Residual correction 결과는 해석의 두 번째 부분을 줍니다. 더 강한 stack artifact를 base로 쓰면 Driver residual correction은 promotion threshold를 넘지 못합니다.

$$
\operatorname{AUC}_{\text{corrected}}
-
\operatorname{AUC}_{\text{base}}
<
0.0003
$$

이는 작은 Driver signal이 강한 stack 안에 이미 흡수되었거나, post-stack correction으로 독립적으로 살아남기에는 너무 약하다는 뜻입니다.

따라서 Driver는 primary modeling axis가 아니라 **low-amplitude auxiliary feature**로 보는 게 맞습니다.

전체 통계 구조는 다음과 같습니다.

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

중요한 건 counterfactual discipline입니다. Driver 아이디어는 매번 다음 질문에 답해야 합니다.

$$
\operatorname{AUC}_{\text{candidate}}
-
\operatorname{AUC}_{\text{matching counterfactual}}
> 0
$$

Residual overlay는 더 강한 기준을 통과해야 합니다.

$$
\operatorname{AUC}_{\text{corrected}}
-
\operatorname{AUC}_{\text{base}}
\ge
0.0003
$$

그래서 최종 해석은 보수적입니다.

> **Driver frequency는 작지만 측정 가능한 신호로 남습니다. 하지만 stronger stack artifact를 base로 둔 뒤에는 Driver residual correction이 아직 promote될 만큼 강하지 않습니다.**
