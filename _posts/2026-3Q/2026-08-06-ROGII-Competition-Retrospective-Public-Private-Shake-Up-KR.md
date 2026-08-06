---
title: "ROGII 대회 회고: Silver Medal까지의 여정과 Public/Private 격차의 교훈"
date: 2026-08-06 09:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, rogii, geosteering, stratigraphy, tvt, competition-retrospective, public-private-gap, silver-medal, korean]
math: true
pin: false
image:
  path: /assets/img/posts/2026-08-06-rogii-competition-retrospective/cover.png
  alt: "ROGII 지층 정렬과 TVT 예측을 설명하는 geosteering 개요도"
---

# ROGII 대회 회고: Silver Medal까지의 여정과 Public/Private 격차의 교훈

ROGII Wellbore Geology Prediction 대회를 **210위, Silver Medal**로 마쳤다. 앞선 두 글에서는 문제를 바라보는 관점과 오차를 나누어 해석하는 방법을 다뤘다. 이번 글에서는 그 분석을 실제 제출 시스템으로 옮긴 과정, 두 달 동안 점수를 줄여온 실험의 흐름, 그리고 마지막 Public/Private 순위 변동이 남긴 교훈을 정리한다.

이전 글:

- [1편: Target-Free 지층 대비로 TVT 복원하기](https://pilkwangkim.github.io/posts/ROGII-Target-Free-Stratigraphic-Alignment-for-TVT-KR/)
- [2편: Target-Free TVT Geosteering의 오차 해부](https://pilkwangkim.github.io/posts/ROGII-Working-Note-2-Target-Free-TVT-Geosteering-KR/)

English version:  
[ROGII Competition Retrospective: A Silver Medal and Lessons from the Public-Private Gap](https://pilkwangkim.github.io/posts/ROGII-Competition-Retrospective-Public-Private-Shake-Up/)

주요 링크:

- [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [최종 제출 노트북: ROGII Development & Tests](https://www.kaggle.com/code/pilkwang/rogii-development-tests)
- [Working Note: Target-Free TVT Geosteering](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)
- [최종 리더보드](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/leaderboard)

---

## 먼저 결과부터

최종 결과는 **210위, Silver Medal**이었다.

| 항목 | 결과 |
|---|---:|
| 팀 Public 점수 / 순위 | **5.952 / 67위** |
| 팀 Private 점수 / 순위 | **8.197 / 210위** |
| 점수 차이, Private - Public | **+2.245** |
| 최종 점수를 결정한 제출 | PS3 bounded TCN dual structural |
| 개인 API 기록 | 339개 |
| Public/Private가 모두 기록된 제출 | 301개 |

RMSE는 낮을수록 좋다. 내 점수는 Public에서 Private으로 넘어가며 `2.245`만큼 나빠졌다. 큰 폭의 변화였지만, 나에게만 일어난 일은 아니었다.

- 전체 6,191개 팀에서 점수 차이의 중앙값은 **+1.929**였다.
- Public 상위 10개 팀 중 Private에서도 상위 10위에 남은 팀은 **1팀**뿐이었다.
- Public 상위 25개 팀과 Private 상위 25개 팀의 교집합은 **11팀**이었다.
- Public 상위 100개 팀에서는 점수 차이의 중앙값이 **+2.213**이었다.
- Public 1위 `shu01`은 `4.608 -> 6.653`, 최종 28위가 되었다.
- Private 1위 `Ruby`는 Public 31위 `5.648`에서 Private `5.639`로 올라왔다.

따라서 내 `+2.245`만 유난히 큰 값은 아니었다. Public 리더보드가 무의미했던 것은 아니지만, 그것만 보고 최종 순위를 예상하기도 어려웠다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-08-06-rogii-competition-retrospective/fig-02-public-private-shakeup.png" alt="ROGII public and private leaderboard shake-up" width="94%">
</p>

이 그림은 Public 상위 500개 팀의 Public/Private 점수를 팀별로 연결한 것이다. 점선은 두 점수가 같은 위치이고, 색은 Private에서 순위가 얼마나 바뀌었는지를 나타낸다. 상위권에서도 점수 차이와 순위 역전이 매우 컸다.

다만 팀 단위 비교에는 주의할 점이 있다. Public 점수는 팀의 Public 최고 제출에서 나오지만, Private 점수는 최종 선택한 두 제출 가운데 하나로 정해진다. 따라서 팀별 점수 차이는 **최종 순위가 얼마나 뒤바뀌었는지**는 보여주지만, 같은 모델이 분포 이동을 겪으며 얼마나 나빠졌는지를 그대로 뜻하지는 않는다. 뒤에서 다룰 개인 제출 비교는 동일한 제출 기록(submission ref)의 Public/Private 점수를 직접 짝지었으므로, 같은 모델이 두 평가 구간에서 어떻게 달라졌는지를 볼 수 있다.

---

## 문제를 다시 한 문장으로 쓰기

이 대회에서는 수평정(horizontal well)의 숨겨진 구간에서 `TVT`를 예측해야 했다. 겉으로는 다음과 같은 행 단위 회귀 문제처럼 보인다.

$$
\widehat T_{w,i}=f(MD_{w,i},X_{w,i},Y_{w,i},Z_{w,i},GR_{w,i}).
$$

하지만 각 행은 서로 독립된 표본이 아니다. 한 well을 따라 이어지는 궤적의 한 지점이다. 그래서 문제를 다음과 같이 다시 썼다.

$$
\widehat T_w(s)=\widehat D_w+\widehat\phi_w(s),
\qquad s\in[0,1].
$$

여기서:

- $\widehat D_w$는 well 전체를 지층 단면의 어느 높이에 놓을지 정하는 **datum**이고,
- $\widehat\phi_w(s)$는 그 위치에서 숨겨진 구간이 어떻게 움직이는지를 나타내는 **shape**다.

이렇게 문제를 쓰자 작업 방향도 달라졌다. `GR`을 단순한 입력 변수 하나로 넣는 대신 typewell 좌표계와의 정렬을 구했고, 관측된 `TVT_input` 구간을 기준점으로 삼았다. 반복되는 지층 무늬 때문에 깊이 후보가 둘로 갈릴 때는 하나를 억지로 고르기보다 두 가능성을 함께 남겨 두는 방법을 고민하게 됐다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-03-rogii-working-note-target-free-tvt-geosteering/fig-02-data-contract.png" alt="ROGII visible prefix and hidden tail data contract" width="94%">
</p>

well $w$에서 TVT가 보이는 앞부분과 숨겨진 뒷부분을 각각

$$
\mathcal P_w=\{i:T^{input}_{w,i}\ \text{is observed}\},\qquad
\mathcal H_w=\{i:T^{input}_{w,i}\ \text{is missing}\}
$$

로 두면, 허용되는 추론기는 다음 정보만 사용할 수 있다.

$$
\widehat T_{w,\mathcal H}
=F\!\left(
X_{w,\mathcal P\cup\mathcal H},
T^{input}_{w,\mathcal P},
\operatorname{typewell}_w
\right).
$$

전체 GR과 궤적 정보는 테스트 입력에 주어진 관측값이므로 사용할 수 있다. 반면 숨겨진 `TVT`나 그 정답에서 계산한 통계량은 사용할 수 없다. 이후 만든 모든 OOF 산출 과정, 특징 생성기, 실행 어댑터는 이 경계를 지키도록 설계했다.

---

## 점수 진행: Public RMSE는 8.336 ft, Private RMSE는 5.546 ft 줄었다

첫 제출의 Public 점수는 `14.288`이었다. 마지막 최고 기록은 `5.952`였다. 두 달 동안 Public RMSE를 **8.336 ft** 줄였다.

대회가 끝난 뒤 동일한 제출 기록끼리 Private 점수를 짝지어 보니 이야기가 조금 달랐다. 첫 제출은 `13.743`, 최종 선택 제출은 `8.197`이었다. Private에서도 **5.546 ft**를 줄였다. 따라서 두 달간의 작업을 전부 Public 과적합이라고 볼 수는 없다. 다만 Public에서 줄인 8.336 ft 가운데 Private으로 이어진 폭은 5.546 ft였다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-08-06-rogii-competition-retrospective/fig-01-score-journey.png" alt="ROGII personal public score journey and paired private transfer" width="96%">
</p>

Public 최고 기록이 바뀐 주요 시점은 다음과 같다.

| 시점 | Public | Private | 당시의 핵심 변화 |
|---|---:|---:|---|
| 5월 6일 | 14.288 | 13.743 | 초기 EDA + 잔차 모델링 |
| 5월 10일 | 10.160 | 9.873 | 같은 입력 행렬의 모델 앙상블 |
| 6월 5일 | 8.072 | 9.982 | target-free 정렬 |
| 6월 8일 | 7.747 | 9.952 | ridge/PF/투영 조합 |
| 6월 27일 | 7.202 | 9.637 | visible-prefix + 이중 후보 헤지 |
| 7월 13일 | 7.022 | 9.648 | 이중 경로 prefix 보정 |
| 7월 20일 | 6.941 | 9.615 | seed-cloud datum 경로 |
| 7월 23일 | 6.517 | 9.456 | 학습과 추론의 출처를 맞춘 엔진 재구축 |
| 7월 31일 | 6.161 | 9.088 | 입력 일치 검증 궤적 전이 |
| 8월 3일 | 6.001 | 8.823 | 쿼리 시계열 잔차 경로 |
| 8월 5일 | 5.952 | 8.523 | 제한된 잔차 + 구조장, PS1 |

이 표에서 가장 중요한 구간은 6월 초다. target-free 정렬을 도입하자 Public은 크게 좋아졌지만 Private은 거의 움직이지 않았다. 공개 테스트의 중복 관계나 공개 쿼리에 특화된 신호와, 처음 보는 well에서도 통하는 신호가 이때부터 갈라지기 시작했다.

반대로 7월 말에 만든 시계열·구조 모델은 학습과 추론에서 같은 정보원을 사용했고, Public에서의 변화는 작았지만 Private을 `9.4` 부근에서 `8점대`까지 낮췄다. 후반부의 학습 모델이 헛수고였던 것은 아니다. 다만 여러 보정을 겹쳐 넣었을 때 각 효과가 Public에서 기대했던 만큼 고스란히 더해지지는 않았다.

---

## Phase 1: 표 형태의 잔차 모델에서 target-free 정렬로

초기에는 표 형태의 입력 변수와 잔차 모델을 늘렸다. 같은 입력 행렬에 CatBoost, LightGBM, ridge를 쌓고, well 단위 분할과 행 단위 특징값을 조정했다. 이 단계에서 점수는 `14.288`에서 `10.160`으로 크게 좋아졌다.

하지만 어느 순간부터 변수를 더 넣어도 같은 종류의 오차가 계속 남았다. 숨겨진 구간 전체가 거의 일정한 값만큼 위나 아래로 밀리는 현상이었다. 이는 행 단위 모델의 표현력보다, well을 잘못된 지층 높이에 놓은 데서 생기는 문제에 가까웠다.

그래서 typewell의 `TVT -> GR` 곡선과 horizontal well의 `MD -> GR` 곡선을 맞추는 target-free 정렬로 방향을 틀었다. 목적함수를 간단히 쓰면 다음과 같다.

$$
J_w(\Delta)
=\frac1{|M_w|}\sum_{i\in M_w}
\rho\!\left(
\frac{G^{hw}_{w,i}-G^{tw}_w(T^{base}_{w,i}+\Delta)}{\sigma_w}
\right),
$$

여기서 $\Delta$는 well 전체에 적용할 datum 이동량이고, $\rho$는 일부 큰 오차가 비용함수를 지배하지 못하게 막는 강건 손실함수다. 구현의 핵심만 추리면 다음과 같다.

```python
def alignment_cost(tvt_path, hw_gr, tw_tvt, tw_gr, scale):
    ref_gr = np.interp(tvt_path, tw_tvt, tw_gr)
    z = (hw_gr - ref_gr) / max(scale, 1e-6)
    huber = np.where(np.abs(z) <= 2.0,
                     0.5 * z**2,
                     2.0 * np.abs(z) - 2.0)
    return float(np.mean(huber))

best_shift = min(
    candidate_shifts,
    key=lambda d: alignment_cost(
        base_tvt + d, hw_gr, tw_tvt, tw_gr, gr_scale
    ),
)
```

이 계산에는 hidden `TVT`가 들어가지 않는다. typewell과 horizontal well에서 관측된 GR 곡선만으로 가능한 datum 후보를 비교한다.

이 변화로 Public 점수는 `8.072`까지 내려갔다. 여기에 ridge 산출물, particle filter, beam search, 저차수 투영을 결합하면서 `7.747`까지 더 줄였다.

그러나 대회가 끝난 뒤 Private 점수를 확인해 보니 이 단계의 개선은 거의 이어지지 않았다. GR이 가장 잘 맞는 위치가 실제 깊이라는 보장이 없고, 공개 테스트의 몇 개 well에서 잘 작동한 정렬이 새로운 well에서도 같은 방식으로 통한다는 보장도 없었다.

---

## Phase 2: Datum, mode, shape를 분리하다

다음 전환점은 “GR 오차가 가장 작은 깊이가 곧 정답은 아니다”라는 사실을 받아들인 일이었다. 비슷한 지층 무늬가 반복되면 서로 멀리 떨어진 두 깊이에서 거의 같은 GR 비용이 나온다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-03-rogii-working-note-target-free-tvt-geosteering/fig-06-error-anatomy.png" alt="Datum mode and shape error anatomy" width="94%">
</p>

두 후보 궤적을 $a_w(s)$와 $b_w(s)$라고 하고, $a$가 맞을 확률을 $p$라고 하자. $a$ 하나를 고정해서 선택했을 때의 제곱오차 위험은

$$
R_{hard}=(1-p)(a-b)^2
$$

이고, 두 후보의 확률 가중 평균 $pa+(1-p)b$를 사용했을 때의 위험은

$$
R_{mean}=p(1-p)(a-b)^2
$$

다. 따라서

$$
R_{hard}-R_{mean}=(1-p)^2(a-b)^2\ge0.
$$

어느 후보가 맞는지 가려낼 신호가 약하다면 하나를 단정해서 고르는 것보다 두 후보를 섞는 편이 제곱오차에 유리하다. 이 관점에서 이중 후보 탐지기, prefix 신뢰도 축소, visible-prefix 보호 규칙을 만들었다. 최종 노트북의 midpoint hedge를 핵심만 남겨 쓰면 다음과 같다.

```python
if branch_mass >= 0.25 and 4.0 <= separation <= 40.0:
    midpoint = 0.5 * (low_branch + high_branch)
    shift = np.clip(0.60 * (midpoint - weighted_path), -2.0, 2.0)
    prediction[eval_mask] += shift
```

분기가 실제로 두 갈래로 나뉘고 각 후보에 충분한 질량이 있을 때만 움직이며, 한 번에 이동할 수 있는 폭도 2 ft로 제한했다.

typewell과 horizontal well은 GR의 크기와 기준점도 다를 수 있다. 이를 맞추기 위해 관측된 heel 구간에서 선형(affine) 보정값을 구했다.

$$
(\alpha_w,\beta_w)
=\arg\min_{\alpha,\beta}
\sum_{i\in\mathcal P_w}
\left[G^{hw}_{w,i}-(\alpha G^{tw}_w(T^{input}_{w,i})+\beta)\right]^2,
$$

$$
\widetilde G^{hw}_{w,i}=\frac{G^{hw}_{w,i}-\beta_w}{\alpha_w}.
$$

이 단계에서 visible-prefix 보정, 이중 후보 헤지, 검증 조건을 둔 접촉점 복원(contact reconstruction)을 도입했고 Public은 `7.2` 부근까지 내려갔다. 그러나 공개 노트북을 조금씩 바꿔 다시 돌리는 방식만으로는 어느 부분이 실제로 효과가 있었는지 판단하기 어려웠다. 사실상 같은 예측 함수도 제출할 때마다 점수가 흔들렸기 때문이다.

그래서 실험의 단위를 “제출 한 번의 점수”가 아니라 “명시적인 가설과 기준 모델 대비 변화”로 바꿨다.

---

## 7점대 군집: Public LB가 공동 탐색셋이 된 순간

이 무렵 참가자 상당수가 비슷한 7점대 공용 노트북을 출발점으로 삼았다. ridge, particle filter, projection, visible-prefix 보정, contact 복원처럼 이름은 조금씩 달랐지만, 실제로는 같은 계열의 예측을 공유하는 경우가 많았다. 그 위에서 spread, projection 차수, branch 임계값, blend weight를 바꾸면 Public 점수가 0.01 단위로 움직였다. 나 역시 이 구간에서 많은 파라미터 조합을 시험했다.

대회가 끝난 뒤 돌아보면 여기에는 서로 다른 두 종류의 가치가 섞여 있었다.

- **구조적 가치**: 공용 노트북은 이 문제가 단순한 행 단위 회귀가 아니라 datum, mode, shape를 함께 추정하는 궤적 문제라는 사실을 빠르게 드러냈다. 이 통찰은 실제로 Private 개선에도 도움이 됐다.
- **선택의 가치**: 같은 예측 계열 안에서 Public 점수가 가장 좋은 cutoff와 weight를 찾는 일은 거의 이어지지 않았다. 보이는 평가 구간에 맞는 조합을 골랐을 뿐, 처음 보는 well에서 더 좋은 조합을 찾았다는 증거는 아니었다.

둘을 구분하지 않으면 “공용 노트북이 유용했다”와 “공용 노트북을 계속 미세조정하는 일이 유용했다”를 같은 주장으로 받아들이게 된다. 이번 대회에서 전자는 대체로 맞았고, 후자는 특히 7점대 이후에는 한계가 뚜렷했다.

이를 간단한 선택 문제로 쓸 수 있다. 후보 $\theta$의 Public 측정값을

$$
\widehat R_{pub}(\theta)
=R_{pub}(\theta)+\varepsilon_{sample}(\theta)
$$

라고 하자. $R_{pub}$는 Public 분포의 위험이고, $\varepsilon_{sample}$은 제한된 Public 표본에서 생기는 측정 오차다. 우리가 실제로 알고 싶은 것은 $R_{priv}$이지만, 많은 후보 중

$$
\widehat\theta
=\arg\min_{\theta\in\Theta_{adaptive}}
\widehat R_{pub}(\theta)
$$

를 고르면, 가장 좋은 모델뿐 아니라 우연히 $\varepsilon_{sample}$이 가장 유리했던 모델도 함께 선택된다. 최종 차이는 다음 두 항으로 나뉜다.

$$
R_{priv}(\widehat\theta)-\widehat R_{pub}(\widehat\theta)
=
\underbrace{R_{priv}(\widehat\theta)-R_{pub}(\widehat\theta)}_{\text{distribution shift}}
+
\underbrace{R_{pub}(\widehat\theta)-\widehat R_{pub}(\widehat\theta)}_{\text{selection optimism}}.
$$

후보의 측정 오차가 독립적인 정규분포라는 단순한 근사에서는 두 번째 항을 만드는 선택 편향의 크기가 대략

$$
\mathbb E\!\left[
\min_{1\le k\le K_{eff}}\varepsilon_{sample,k}
\right]
\approx
-\sigma\sqrt{2\log K_{eff}}
$$

처럼 커진다. 실제 후보들은 서로 강하게 상관되어 있으므로 $K_{eff}$는 제출 횟수보다 작다. 그러나 중요한 점은 $K_{eff}$가 내 실험 횟수로 끝나지 않는다는 것이다. 한 사람이 좋은 Public 조합을 공개하면 수많은 참가자가 그 주변을 다시 탐색하고, 그 결과가 다음 공용 노트북의 시작점이 된다. **Public LB는 개인의 검증셋을 넘어 참가자 전체가 반복해서 조회한 공동 탐색셋**이 되었다.

이번 대회에서는 검증 표본의 겉보기 크기와 실제 크기도 크게 달랐다. Public 쿼리는 14,151개 행이었지만 well은 3개였다. 각 행의 오차를 well별 datum 오차 $\delta_w$와 국소 shape 오차 $s_{w,i}$로 쓰면

$$
RMSE^2
=\sum_w\frac{n_w}{N}
\left(
\delta_w^2
+2\delta_w\overline{s}_w
+\overline{s_w^2}
\right).
$$

즉 datum을 한 번 잘못 잡으면 $\delta_w^2$가 그 well의 긴 tail 전체에 복사된다. 행이 많아도 독립적인 well이 세 개뿐이면 이런 오차가 평균화되지 않는다. 한 Public well에 잘 맞는 파라미터가 순위를 크게 끌어올리고, 다른 성격의 Private well 하나가 그 순서를 뒤집을 수 있었다. **작은 집단 수, well 단위의 강한 상관, RMSE의 큰 오차 민감도, 공동체 차원의 반복 탐색**이 한꺼번에 겹쳤다는 점에서 이번 대회의 shake-up은 특히 컸다.

최종 순위 변동은 이 현상을 극단적으로 보여줬다. `Herra Huu` 팀은 Public **1,591위(6.501)**에서 Private **15위(6.346)**로 **1,576계단** 올라 Gold Medal을 받았다. 절대 점수는 0.155 좋아졌지만 순위는 거의 1,600등 움직였다. 이는 한 팀이 Private에서 갑자기 1,600등어치의 마법 같은 개선을 얻었다는 뜻이 아니다. Public에서 앞서 있던 많은 팀이 비슷한 계열의 오차를 함께 안고 있었고, Private에서 그 군집이 한꺼번에 재배열됐다는 뜻에 가깝다. 점수가 촘촘한 구간에서는 작은 절대 차이도 수백, 수천 계단의 순위 차이로 확대됐다.

따라서 이 결과를 “Public LB는 아무 의미가 없다”거나 “순위 상승은 전부 운이었다”로 읽는 것도 지나치다. Public은 큰 구조적 개선의 방향을 알려줬고, Private에서 크게 오른 팀은 공통 계열과 다른 귀납 편향을 가졌거나 Public에는 불리하지만 숨겨진 well에는 더 잘 맞는 제출을 최종 선택했을 수 있다. 다만 Public **순위**는 모델의 절대적인 품질 순서가 아니라, 그 시점에 참가자들이 집중한 모델 계열 안에서의 상대적 위치였다는 점이 분명해졌다.

| 관측한 현상 | 성급한 해석 | 더 나은 해석과 행동 |
|---|---|---|
| 비슷한 7점대 노트북이 계속 등장 | 해법이 거의 확정됐다 | 보이는 구간에서 한 계열이 포화됐다. 새로운 observable을 찾아야 한다 |
| weight를 바꿔 0.02 개선 | 일반화 성능도 좋아졌다 | 반복 선택의 낙관일 수 있다. 고정 parent와 grouped/pseudo-private holdout으로 확인한다 |
| Public 하위권이 Private Gold로 상승 | 숨은 비법이나 운이 전부다 | 공통 오차의 상관과 순위 압축이 컸다. 독립적인 작동 원리를 포트폴리오에 남긴다 |
| Public/Private 순서가 크게 뒤바뀜 | Public은 폐기해야 한다 | 큰 효과의 방향 확인에는 쓰되, 작은 차이의 tie-breaker로 쓰지 않는다 |

결국 공용 노트북은 **답안**보다 **논문**에 가깝게 읽어야 했다. 점수와 파라미터를 가져오는 대신, 어떤 관측값을 새로 읽었는지, 어떤 가정이 있어야 작동하는지, 그 가정을 별도의 holdout에서 반증할 수 있는지를 가져와야 했다. 이 판단이 다음 단계에서 공개 노트북을 엔진, observable, artifact로 분해해 다시 구축한 이유다.

---

## Phase 3: 공개 노트북을 베끼지 않고 독립 전문가 모델로 분해하다

7월에는 높은 점수를 기록한 공개 노트북이 빠르게 늘어났다. 그렇다고 코드를 통째로 가져다 붙이는 것은 의미가 없었다. 먼저 각 노트북이 새로 가져온 정보가 무엇인지 다음 질문으로 나눠 살펴봤다.

1. 어떤 관측 신호를 추가했는가?
2. 그 신호를 숨겨진 정답 없이 다시 만들 수 있는가?
3. 학습 OOF와 실제 추론이 같은 좌표계의 특징값을 쓰는가?
4. 기준 예측과 보정값의 출처가 일치하는가?
5. 실제 실행 환경에서 well, 행, 열을 동적으로 읽는가?

이 과정을 거치며 공개 노트북을 **실행 엔진(engine)**, **관측 신호(observable)**, **학습 산출물(artifact)**로 나눴다. 실제 추론에 쓸 수 있는 엔진이 있더라도 같은 조건으로 만든 OOF가 없으면 독립 전문가 모델로 인정하지 않았다. 반대로 OOF가 좋아도 추론 시점에 같은 특징값을 다시 만들 수 없으면 제출 경로에 넣지 않았다.

엄격한 통과 조건은 아래처럼 단순했다.

```python
assert np.array_equal(parent_oof.id, feature_oof.id)
assert feature_manifest["target_columns_used"] is False
assert feature_manifest["coordinate_schema"] == query_schema

residual = y_true - parent_oof.prediction
model.fit(feature_oof.values, residual, groups=well_id)

# 배포 전에 같은 생성기로 추론용 특징값을 다시 만든다.
query_feature = build_features(query, target=None)
assert query_feature.columns.tolist() == feature_oof.columns.tolist()
```

OOF 수치가 아무리 좋아도 이 계약을 통과하지 못하면 배포 가능한 개선으로 보지 않았다.

이 시기에 만든 주요 축은 다음과 같다.

- 추론 좌표계에 맞춘 시계열 잔차 라우팅
- 그래프 증류를 이용한 지층 보정
- 각 지점의 주변 정보만 쓰는 잔차 모델
- TCN/HGBR 그래프 전문가 모델
- BiMamba 직접 예측·잔차 보정 상태공간 모델
- dip/formation 구조장(structural field)
- 입력 내용으로 검증한 궤적 전이

이 접근으로 Public을 `6.5`에서 `6.0` 부근까지 낮췄다. 대신 많은 모델이 엄격한 fold 검증을 넘지 못했다. 입력 변수가 많다고 항상 좋아지는 것은 아니었고, 출처가 다른 OOF를 섞으면 실제로는 존재하지 않는 큰 개선이 나타났다. 좌표가 2 ft만 어긋나도 보정 모델이 배우는 대상 자체가 달라졌다.

실패한 실험을 버리지 않고 원장에 기록한 이유도 여기에 있다. 마지막 며칠에는 “무엇을 더 해볼까”보다 “어느 접근이 이미 가능성이 없는 것으로 확인됐는가”가 더 중요한 정보였다.

---

## Phase 4: `private-safe`를 실행 계약으로 만들다

대회 후반에는 Public 점수만 좇지 않는 `private-safe` 경로를 따로 만들었다. 그러나 당시에는 이 이름에 서로 다른 두 뜻이 섞여 있었다.

### 운영 안전성

- 숨겨진 정답을 특징 생성기에 넘기지 않는다.
- `sample_submission.csv`에서 ID와 예측값 열을 동적으로 찾는다.
- well 목록과 행 수를 하드코딩하지 않는다.
- 추론용 특징값과 예측값을 공개 테스트에 고정된 배열로 읽지 않는다.
- 필요한 데이터셋은 내가 관리하는 비공개 저장소에 보존한다.
- 노트북 SHA와 실행 산출물의 SHA를 고정한다.
- T4의 9시간 제한 안에서 전체 추론 경로를 통과한다.

### Private 분포에서의 일반화

- 새로운 well에서도 보정이 유효하다.
- 공개 테스트의 정확·부분 중복 관계에 지나치게 의존하지 않는다.
- OOF 개선이 숨겨진 분포에서도 유지된다.
- 마지막 보정 단계가 기준 예측의 숨겨진 구간 오차를 키우지 않는다.

첫 번째 목표는 상당 부분 달성했다. 최종 노트북은 실행할 때 제출 형식을 읽고, ID 순서와 예측값의 유효성을 확인했으며, 공개 쿼리의 14,151개 행을 동적으로 처리했다. 외부 데이터셋이 사라지는 상황에 대비해 의존성 보관용 저장소도 따로 만들었다.

```python
sample = pd.read_csv(comp_root / "sample_submission.csv", dtype=str)

id_matches = [
    c for c in sample.columns
    if str(c).strip().lower() == "id"
]
id_col = id_matches[0] if len(id_matches) == 1 else sample.columns[0]
target_cols = [c for c in sample.columns if c != id_col]

if len(target_cols) != 1:
    raise RuntimeError("could not derive the active target column")

parts = sample[id_col].str.rsplit("_", n=1, expand=True)
wells = parts.iloc[:, 0].astype(str)
rows = pd.to_numeric(parts.iloc[:, 1], errors="raise")
```

이 계약 덕분에 초기에 몇 차례 겪었던 hidden rerun의 형식 오류를 막을 수 있었다.

하지만 최종 결과를 보면 두 번째 의미의 안전성은 충분히 검증하지 못했다. **재현 가능하고 누수가 없는 모델**과 **Private에서 잘 일반화되는 모델**은 서로 다른 조건이다.

---

## 최종 노트북의 구조

최종 노트북의 내부 제목은 **Bounded Dual State-Space Structural Geosteering**이다. 마지막에 선택한 PS3 경로로, Public `5.998`, Private `8.197`을 기록했다.

전체 경로는 다음과 같이 요약할 수 있다.

```text
ridge/PF + 물리 selector
-> 저차수 U = TVT + Z 투영
-> 학습된 궤적 혼합
-> 검증 조건을 둔 contact / visible-prefix 보정
-> 입력 일치 검증 궤적 전이
-> 위험도에 따라 적용하는 쿼리 시계열 잔차
-> 그래프 증류 보정
-> 기준 궤적별 datum head
-> dual BiMamba 직접 예측 + 잔차 보정
-> dip/formation 구조장
-> 동적 제출 형식 및 SHA 검사
```

### 1. Ridge/PF와 selector 기준 궤적

첫 기준 궤적은 두 경로를 섞어 만들었다.

$$
T_i^A=0.30T_i^{ridge}+0.70T_i^{selector}.
$$

selector는 128개 seed의 PF likelihood 앙상블과 14개 설정의 beam 앙상블을 사용했다. well 길이와 $Z$ 범위에 따라 PF scale과 hold 비율을 골랐다. 한 번의 우연한 입자 경로에 의존하지 않고, 여러 seed의 likelihood를 함께 보려는 설계였다.

### 2. 지층면 투영

TVT와 수직 좌표를 합친

$$
U_i=T_i+Z_i
$$

를 낮은 차수의 함수로 근사했다.

$$
T_i^{proj}
=(1-\lambda_p)T_i^A
+\lambda_p\left(\widehat U_i-Z_i\right),
\qquad \lambda_p=0.75.
$$

행 단위로 흔들리는 잡음은 줄이고, well 전체의 완만한 형태는 남기기 위한 단계다. 실제 코드는 정규화한 MD 위에서 강건 다항식을 반복해서 적합했다.

```python
def robust_polyfit(s, y, degree=5):
    coef = np.polyfit(s, y, degree)
    for _ in range(4):
        residual = y - np.polyval(coef, s)
        scale = 1.4826 * np.median(np.abs(residual)) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.0 * scale))**2)
        coef = np.polyfit(s, y, degree, w=weight)
    return np.polyval(coef, s)

u_fit = anchor_u + robust_polyfit(s, (tvt + z) - anchor_u)
tvt_projected = 0.25 * tvt + 0.75 * (u_fit - z)
```

큰 잔차의 가중치를 반복해서 낮추기 때문에 잘못된 후보 경로의 일부 점이 전체 곡선을 끌고 가는 현상을 줄일 수 있었다.

### 3. 입력 일치 검증 궤적 전이

대상 well의 입력 내용이 정답을 가진 donor와 정확히 또는 부분적으로 일치할 때만 donor 궤적을 가져왔다. 가져온 궤적은 관측된 prefix에 맞춰 datum을 다시 보정했다.

$$
\widehat T_w^{content}(i)
=T_d(MD_w(i))
+\operatorname{median}_{j\in\mathcal P_w}
\left[T_w^{input}(j)-T_d(MD_w(j))\right].
$$

최종 공개 쿼리의 세 well에서는 donor와 입력 행이 정확히 일치하는 관계가 발견됐다. 다만 파일이나 ID가 같다는 이유만으로 덮어쓰지는 않았다. 관측된 prefix에서도 궤적이 충분히 잘 맞는지를 다시 확인했다.

```python
known = test_well[test_well["TVT_input"].notna()]
candidate = interpolate_donor(donor_tvt, known["MD"])
prefix_rmse = rmse(known["TVT_input"], candidate)

if len(known) >= 50 and valid_physics_rows >= 100 and prefix_rmse <= 1.0:
    prediction[hidden_rows] = interpolate_donor(donor_tvt, hidden_md)
else:
    prediction[hidden_rows] = parent_prediction[hidden_rows]
```

이 경로는 Public에서 매우 강했다. 그러나 새로운 Private well에도 같은 donor 관계가 존재할지는 별개의 문제였다. 관측 구간에서 충분히 잘 맞을 때만 사용하도록 검증 조건을 둔 이유가 여기에 있다.

### 4. 시계열 위험도 라우터

다섯 종류의 기준 궤적과 서로 간의 차이를 256개 지점의 공통 좌표로 다시 표본화했다. 그 위에서 15개의 시계열 모델이 5차 Legendre 기저의 보정량을 예측했다.

$$
r_w(s)=\operatorname{clip}\!\left[
0.25\sum_{k=0}^{5}\theta_{w,k}P_k(2s-1),-8,8
\right].
$$

다섯 개의 위험도 모델은 추론 시점에 계산할 수 있는 요약 통계만 보고 $\alpha_w\in[0,1]$를 정했다.

$$
\widehat T_w(s)=c_w(s)+\alpha_w r_w(s).
$$

구현에서는 보정량을 곧바로 더하지 않았다. 모델 간 불일치와 prefix 안정성을 바탕으로 실제 적용 강도를 다시 조절했다.

```python
raw_residual = temporal_ensemble.predict(sequence_features)
router_input = summarize_without_target(parent_paths, raw_residual)
alpha = np.clip(risk_router.predict(router_input), 0.0, 1.0)

move = alpha[:, None] * np.clip(raw_residual, -8.0, 8.0)
prediction = content_parent + move
```

학습 모델의 역할을 “새 궤적을 처음부터 만드는 것”이 아니라 “현재 궤적을 언제, 얼마나 고칠지 제안하는 것”으로 제한한 셈이다.

### 5. 그래프, datum, BiMamba, 구조장

후반부에는 추론까지 같은 정보로 재현되는 graph TCN, 기준 궤적별 datum head, dual BiMamba 보정을 차례로 적용했다. 최종 PS3의 BiMamba 단계는 50개 체크포인트를 사용했고, 직접 예측의 적용 비율은 `0.075`, 잔차 보정의 적용 비율은 `0.5`였다.

마지막 구조장은 dip과 formation 예측을 실행 중에 계산한 뒤, 작은 범위로 제한해 더했다.

$$
\Delta^{struct}_w
=0.5\left[
0.1\,\operatorname{clip}(\Delta^{dip}_w,-4,4)
+0.075\,\operatorname{clip}(\Delta^{formation}_w,-4,4)
\right].
$$

모든 보정은 실제 추론 데이터에서 다시 계산했고, 마지막에는 `sample_submission.csv`를 기준으로 결과를 검사했다.

```python
sample = pd.read_csv(sample_path)
submission = pd.read_csv(submission_path)

identifier = sample.columns[0]
prediction_col = [c for c in sample.columns if c != identifier][0]

if submission.columns.tolist() != sample.columns.tolist():
    raise RuntimeError("submission columns differ from sample")
if not submission[identifier].astype(str).equals(sample[identifier].astype(str)):
    raise RuntimeError("identifier order differs from sample")

values = submission[prediction_col].to_numpy(float)
if not np.isfinite(values).all():
    raise RuntimeError("submission contains non-finite predictions")

audit["sha256_submission_csv"] = sha256_file(submission_path)
```

행 수나 예측값 열 이름을 코드에 박아 두지 않았고, ID 순서와 모든 예측값의 유효성, 파일 SHA를 실행할 때마다 기록했다. 이 운영 계약은 최종 노트북에서 가장 재사용 가치가 높은 부분이다.

---

## 마지막 3+2 포트폴리오

마지막 날에는 이미 점수를 받은 제출을 되풀이하지 않았다. 대신 작동 원리가 다른 세 개의 공격적 후보와 두 개의 `private-safe` 후보를 준비했다.

| 역할 | 후보 | Public | Private | 차이 |
|---|---|---:|---:|---:|
| Moonshot | M4 residual direct structural | 6.103 | 8.202 | +2.099 |
| Moonshot | M5 TCN dual structural | 6.066 | **8.034** | +1.968 |
| Moonshot | M3 HGBR direct structural | 6.060 | 8.132 | +2.072 |
| Private-safe | PS1 bounded residual structural | **5.952** | 8.523 | +2.571 |
| Private-safe | PS3 bounded TCN dual structural | 5.998 | 8.197 | +2.199 |

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-08-06-rogii-competition-retrospective/fig-03-final-portfolio.png" alt="ROGII final three moonshots and two private-safe candidates" width="96%">
</p>

Public 순서는 `PS1 -> PS3 -> M3 -> M5 -> M4`였지만 Private에서는 `M5 -> M3 -> PS3 -> M4 -> PS1`로 완전히 달라졌다. Public에서 가장 좋았던 PS1이 Private에서는 다섯 후보 가운데 가장 나빴다.

최종 팀 Private 점수 `8.197`은 PS3의 점수와 정확히 같다. 마지막에 선택한 두 후보 가운데 PS3가 최종 성적을 결정한 셈이다. 결과를 모두 알고 난 뒤 돌아보면 M5의 `8.034`가 내 제출 중 가장 좋은 Private 점수였다. M5를 골랐다면 최종 순위표에서 약 170위였을 것으로 추정된다. 실제보다 약 40계단 높지만, 대회 전체의 결론을 바꿀 정도의 차이는 아니다.

더 중요한 문제는 `private-safe`라는 이름에 서로 다른 두 뜻을 넣었다는 점이다. PS1과 PS3는 동적 스키마, 제한된 보정, 직접 관리하는 의존성, 숨겨진 정답을 쓰지 않는 계약을 지켰다. 실행과 재현의 관점에서는 안전했다. 그러나 이런 조건만으로 Private 분포에서 더 잘 일반화된다고 말할 수는 없다. 다음 대회에서는 두 개념을 처음부터 따로 관리해야 한다.

- **배포 안전성(deployment-safe)**: 재현 가능하고, 누수가 통제되며, 입력 형식 변화에도 견디는가?
- **분포 강건성(distribution-robust)**: 실제와 비슷한 미관측 집단의 분포 이동에서도 성능이 유지되는가?

---

## 이번 대회에서 얻은 것

메달보다 오래 남을 성과는 문제를 바라보는 방식과 실험 절차를 바꾼 것이다. 특히 다음 다섯 가지는 다른 대회에서도 그대로 활용할 수 있다.

### 1. 행 단위 회귀를 well 전체의 궤적 추정 문제로 다시 정의했다

가장 큰 전환은 행 단위 회귀에서 datum, mode, shape를 따로 보는 방식으로 옮겨간 일이었다. 그 뒤로는 “어떤 변수를 더 넣을까”보다 “well 전체의 높이가 틀렸는가, 두 지층 후보 가운데 잘못 골랐는가, 숨겨진 구간의 모양이 틀렸는가”를 따로 물을 수 있었다. Private도 초기 `13.743`에서 `8.197`까지 낮아졌으므로, 이 관점의 변화는 실제 개선으로 이어졌다.

### 2. Prefix와 typewell을 정답이 아니라 근거로 사용했다

prefix 보정, heel GR 보정, 접촉점 복원은 모두 강한 신호였다. 그러나 항상 맞는 규칙처럼 적용하면 일부 well에서 오차가 크게 튀었다. 근거가 충분할 때만 보정을 허용하고, 애매할 때는 두 후보를 섞거나 원래 예측으로 돌아가게 하면서 큰 실패를 줄였다.

여기서 얻은 원칙은 단순하다. **관측 근거가 강하면 움직이고, 약하면 원래 예측을 지킨다.** 지질학적 사전 지식 역시 정답표처럼 쓰기보다 불확실성을 줄이는 근거로 다뤄야 했다.

### 3. 점수가 좋은 OOF보다 실제로 재현되는 OOF를 먼저 확인했다

후반부에는 숫자만 좋은 모델보다 아래 질문에 답할 수 있는 모델을 우선했다.

- 새 well에서도 같은 특징값을 정확히 다시 만들 수 있는가?
- OOF와 실제 추론에서 같은 기준 예측을 사용하는가?
- 정답 열을 임의로 바꿔도 예측이 달라지지 않는가?
- 실행 시점에 모든 ID와 열을 동적으로 찾을 수 있는가?

이 원칙 덕분에 특징값과 기준 예측의 출처가 달라 생긴 가짜 개선을 여러 번 걸러냈다. 최종 제출도 hidden rerun에서 형식 오류 없이 끝까지 실행됐다. 모델 성능과 별개로, 이 실행 계약은 가장 확실하게 다시 쓸 수 있는 결과물이다.

### 4. 실험을 기억이 아니라 기록으로 관리했다

가설, 예상 점수, 실제 결과, 해석, 다음 행동을 CSV·JSON·Markdown 원장에 남겼다. 대회 막판에는 이 기록이 코드만큼 중요했다. 이미 닫힌 가설을 다시 시험하는 일을 줄였고, 제출 결과가 좋을 때와 나쁠 때의 후속 작업을 미리 준비할 수 있었다.

다만 기록의 양이 늘었다고 실험 설계까지 저절로 좋아지는 것은 아니었다. 앞으로는 “무엇을 했는가”뿐 아니라 **이 실험이 어느 가설을 다른 가설과 구분해 주는가**를 더 엄격히 적어야 한다.

### 5. 공개 노트북을 복사 대상이 아니라 새로운 정보원의 후보로 읽었다

공개 코드를 그대로 덧붙이는 대신 새로 쓰는 관측값과 모델 계열을 분리해 엄격한 OOF에서 다시 검증했다. 많은 경로가 탈락했지만 시계열 잔차, graph TCN, BiMamba처럼 Private을 조금씩 낮춘 독립적인 축도 얻었다. 공개 솔루션을 가장 잘 활용하는 방법은 코드의 모양을 따라가는 것이 아니라, **그 코드가 기존에 없던 어떤 정보를 읽고 있는지 찾아내는 것**이었다.

---

## 아쉬웠던 판단과 놓친 검증

결과를 돌아보면 모델보다 검증 방식과 최종 선택 기준에 더 큰 아쉬움이 남는다.

### 1. 후반까지 Public 점수를 중심 목표로 삼았다

동일한 제출 301개의 Public/Private 점수를 비교하면 Spearman 상관계수는 `0.763`이었다. 큰 방향을 확인하는 데는 도움이 되지만, 최종 후보의 순서를 정하기에는 부족했다. 내 제출 가운데 **90.0%**가 Private에서 나빠졌고, 점수 차이의 중앙값은 **+2.310**이었다.

Public 최고 기록을 갱신하는 과정은 새로운 특징값을 찾는 데 도움이 됐다. 그러나 7월 이후에는 “Public에서 얼마나 좋아졌는가”보다 “이 개선이 처음 보는 well에서도 남을 가능성이 얼마나 되는가”를 중심 지표로 삼았어야 했다.

### 2. 공개 쿼리의 중복 관계를 일반화 신호로 과대평가했다

최종 공개 표본은 3개 well, 14,151개 행이었다. 입력 일치 검증 궤적 전이는 이 세 well에서 매우 강했다. 하지만 Private의 새로운 well에도 같은 donor 관계가 있으리라는 보장은 없었다.

공개 쿼리에서 완벽하게 검증된 규칙이라도 숨겨진 쿼리에 같은 조건이 없다면 일반화 모델이 아니라 특정 상황에서만 통하는 지름길이다. 중복 관계가 있을 때 쓰는 경로와 그렇지 않을 때 쓰는 기본 경로를 처음부터 나누어 평가했어야 했다.

### 3. OOF가 Private의 분포 이동을 충분히 흉내 내지 못했다

well 단위 그룹 분할로 행 사이의 누수는 막았지만, 그것만으로 Private의 분포 변화를 흉내 낼 수는 없었다. 일반적인 grouped CV와 별도로 Private 분포를 가정한 검증 분할이 필요했다.

예를 들면 다음과 같은 well을 의도적으로 검증용 holdout으로 묶을 수 있었다.

- train에 있는 typewell 계열과 특성이 크게 다른 well
- prefix GR calibration이 불안정한 well
- likelihood가 두 개의 뚜렷한 후보로 갈리는 well
- 숨겨진 구간이 길고 $Z$ 범위가 큰 well
- donor와 입력 내용이 전혀 맞지 않는 well

이처럼 분포 이동을 고려한 holdout에서도 안정적인 후보에만 `distribution-robust`라는 이름을 붙였어야 한다.

### 4. 모델을 더 얹는 것과 새로운 정보를 더하는 것을 구분하지 못했다

최종 노트북에는 많은 보정 단계가 들어갔다. 그러나 서로 다른 모델이 같은 기준 예측의 오차를 반복해서 읽는 경우도 많았다. 모델 수가 늘어도 정보원이 같으면 오차는 비슷한 방향으로 움직이고, 앙상블 효과도 제한된다.

M5가 PS1보다 Public 점수는 낮았지만 Private에서는 더 좋았던 이유도 이와 관련됐을 가능성이 크다. 작은 Public 이득보다 서로 다른 작동 원리와 보정 폭의 제한이 더 중요했다.

### 5. 마지막 선택에서 보정의 위험을 충분히 벌점화하지 않았다

최종 후보를 고를 때 Public 점수, OOF 개선, 정보원의 다양성, 실행 안정성을 함께 봤다. 하지만 Private 결과를 보면 평균적으로 얼마나 좋아지는지만큼, 보정이 빗나갔을 때 얼마나 크게 망가지는지도 중요했다.

$$
J_{final}
=\widehat{RMSE}_{shift}
+\lambda_1\,Q_{0.95}(|\Delta|)
+\lambda_2\left|\operatorname{Corr}(\Delta,\Delta_{parent})\right|
+\lambda_3 D_{overlap}.
$$

여기서 $D_{overlap}$은 특정 쿼리의 중복 관계에 성능이 얼마나 기대는지를 나타낸다. 즉 평균 OOF 개선뿐 아니라 보정값의 극단적인 꼬리, 기준 예측과의 중복, 특정 쿼리에 대한 의존성을 직접 벌점으로 줬어야 했다. 평균 0.05 ft를 줄이는 단계라도 일부 well을 3~4 ft 악화시킨다면 최종 포트폴리오에서는 빼는 편이 나을 수 있다.

---

## Public/Private 점수 차이를 어떻게 읽어야 하나

이번 결과만 보고 “Public 리더보드는 쓸모없다”고 결론 내리는 것도 옳지 않다. Public이 14.288에서 5.952로 내려가는 동안 Private도 13.743에서 8점대로 좋아졌다. 문제를 바라보는 방식을 바꾼 큰 개선은 Private에도 이어졌다.

다만 후반부의 작은 순위 차이는 거의 이어지지 않았다. 개선의 종류에 따라 나누어 보면 다음과 같다.

| 개선의 성격 | Private로 이어질 가능성 |
|---|---|
| 문제 표현을 바꾸는 큰 구조적 개선 | 대체로 이어졌다 |
| 특정 쿼리의 정렬이나 중복 관계를 이용한 개선 | 약하게 이어지거나 사라졌다 |
| 막판의 작은 혼합 가중치 조정 | 평가 잡음과 분포 이동을 구분하기 어려웠다 |

따라서 리더보드는 최적화의 정답이 아니라 잡음이 섞인 관측값으로 다뤄야 한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-03-rogii-working-note-target-free-tvt-geosteering/fig-08-evidence-architecture-ladder.png" alt="Evidence to architecture ladder" width="94%">
</p>

더 바람직한 순서는 다음과 같다.

$$
\text{관측}
\rightarrow
\text{추정기}
\rightarrow
\text{분포 이동을 고려한 검증}
\rightarrow
\text{보정 폭이 제한된 배포 정책}
\rightarrow
\text{리더보드 확인}.
$$

이번 대회에서는 내부 검증과 리더보드 확인 사이의 간격을 충분히 좁히지 못했다.

---

## 다시 시작한다면 이렇게 하겠다

가장 크게 바꿀 부분은 모델의 종류가 아니라 **검증 체계를 만드는 순서**다. 다음에는 Public 점수가 좋아진 뒤에야 안전성의 근거를 찾지 않을 것이다. 처음부터 서로 다른 분포를 가정한 검증 환경을 만들고 그 위에서 모델을 고르려 한다.

### 첫 주

1. 제출 형식과 hidden rerun 검사를 가장 먼저 자동화한다.
2. well 단위 OOF와 함께 분포 이동을 가정한 holdout을 만든다.
3. datum, mode, shape의 oracle ladder를 계산해 실제로 줄일 수 있는 오차의 크기부터 확인한다.
4. 공개 노트북은 점수 순으로 모으지 않고, 새로 제공하는 관측 정보별로 분류한다.

### 중반

1. 기준 예측을 고정한 뒤 한 번에 하나의 보정만 검증한다.
2. 보정 벡터와 OOF 행의 순서에 SHA를 붙여 서로 다른 실험 결과가 섞이지 않게 한다.
3. 평균 개선뿐 아니라 좋아진 well의 비율과 p95/p99 악화 폭을 함께 본다.
4. GPU는 첫 fold 검증을 통과하고 추론용 특징값까지 동일하게 재현된 모델에만 사용한다.

### 마지막 2주

1. Public 점수를 공격적으로 노리는 후보와 분포 이동에 강한 후보를 별도의 표로 관리한다.
2. 같은 제출의 반복은 점수 잡음을 추정하는 데 필요한 최소 횟수로 제한한다.
3. 제출권은 하나의 주효과나 두 요소의 상호작용을 구분할 수 있을 때만 쓴다.
4. 마지막 두 개는 Public 순서가 아니라 작동 원리의 다양성과 분포 이동 위험을 기준으로 고른다.

최종 포트폴리오 목적함수도 다음처럼 바꾸고 싶다.

$$
\min_{a,b}
\mathbb E_{\mathcal D_{shift}}
\left[\min\{L(a),L(b)\}\right]
+\lambda\,\operatorname{Corr}(e_a,e_b),
$$

두 후보는 평균 오차가 낮아야 할 뿐 아니라 서로 다른 상황에서 실패해야 한다. 같은 well에서 함께 무너지는 두 모델은 이름이 달라도 제대로 된 포트폴리오가 아니다.

---

## 마치며

최종 성적은 210위, Silver Medal이었다. Public 5.952를 기록했을 때 기대했던 순위에는 미치지 못했다. Private 8.197은 후반부에 얻은 개선 가운데 상당 부분이 새로운 분포에서는 유지되지 않았음을 보여줬다.

그렇다고 이 과정을 실패로만 정리할 생각은 없다.

- 초기 Private 13.743에서 8.197까지 실제로 개선했다.
- 문제를 target-free geosteering 문제로 다시 표현했다.
- datum, mode, shape의 오차 구조를 분리했다.
- hidden target과 관측 공변량의 경계를 코드 계약으로 만들었다.
- 같은 정보원으로 만든 OOF와 동적 추론 경로를 연결했다.
- 마지막에는 50개 체크포인트로 앙상블한 상태공간 모델과 구조장 보정을 하나의 재현 가능한 노트북으로 묶었다.
- 무엇보다 Public 점수와 Private 일반화 성능이 같은 말이 아니라는 사실을 숫자로 확인했다.

최종 노트북은 간결한 해법과는 거리가 멀다. 여러 달 동안 시도한 아이디어를 한 실행 경로 안에 어디까지 묶을 수 있는지 보여주는 연구 기록에 더 가깝다. 다음 대회에 가져갈 가장 중요한 것은 복잡한 모델 조합 자체가 아니라 그 과정에서 얻은 다음 원칙들이다.

1. 근거가 충분할 때만 datum을 움직인다.
2. mode가 애매하면 하나를 단정하지 않고 위험을 나눈다.
3. shape는 처음부터 끝까지 같은 좌표계에서 학습하고 적용한다.
4. 재현 가능성과 일반화 성능을 서로 다른 주장으로 검증한다.
5. 리더보드는 판단을 돕는 증거로 쓰되, 정답 그 자체로 여기지 않는다.

Silver Medal은 긴 실험 과정의 결과였고, Public/Private 점수 차이는 그 과정에서 받은 마지막이자 가장 값비싼 피드백이었다.

---

## 참고 자료

- [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- [ROGII final leaderboard](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/leaderboard)
- Pilkwang Kim, [ROGII Development & Tests](https://www.kaggle.com/code/pilkwang/rogii-development-tests)
- Pilkwang Kim, [Working Note: Target-Free TVT Geosteering](https://www.kaggle.com/code/pilkwang/working-note-target-free-tvt-geosteering)
- Georgy Mamarin, [Stop reforking: the best GR fit is the wrong depth](https://www.kaggle.com/code/georgymamarin/stop-reforking-the-best-gr-fit-is-the-wrong-depth)
- [ROGII Geological Operations / StarSteer overview](https://rogii.com/solutions/geological-operations)
- mycarta, [ROGII Geosteering Toolkit](https://github.com/mycarta/rogii-geosteering-toolkit)

### 리더보드 분석 메모

수치는 2026년 8월 6일 공식 Kaggle API에서 받은 6,191개 팀의 Public/Private 리더보드와 개인 제출 기록을 기준으로 계산했다. 팀 단위 점수는 서로 다른 최종 선택 제출에서 나올 수 있으므로 전체 순위 변동을 보는 데만 사용했다. 개인 제출의 점수 차이는 동일한 제출 기록끼리 직접 비교했다.
