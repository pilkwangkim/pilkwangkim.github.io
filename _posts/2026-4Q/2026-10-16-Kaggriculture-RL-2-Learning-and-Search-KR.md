---
title: "Kaggriculture RL 2 — BC·PPO·탐색을 이용한 전략 개선"
description: "단일 RTX 5090의 BC·PPO와 우승팀의 탐색 교사를 실제 코드로 연결하고, 인력·토지·작물 묶음의 후보 생성부터 미래 수익·임금·시장 가격을 계산하는 과정까지 따라간다."
date: 2026-10-16 10:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggriculture, reinforcement-learning, behavior-cloning, planning, ppo, search]
topic: kaggriculture
lang: ko
translation_key: kaggriculture-rl-02
series: kaggriculture-rl
series_order: 2
published: true
image:
  path: /assets/img/posts/kaggriculture-rl-series/cover-2.webp
  alt: "Replays, behavior cloning, self-play PPO and a planner forming a strategy improvement loop"
  width: 1672
  height: 941
  fit: contain
  hide_caption: true
math: true
toc: true
---

<style>
.kaggriculture-rl-figure .rl-figure-narrow { display: none; }
.kaggriculture-rl-figure img { width: 100%; height: auto; }
.kaggriculture-rl-figure .img-link { display: block; }
@media (max-width: 600px) {
  .kaggriculture-rl-figure .rl-figure-wide { display: none; }
  .kaggriculture-rl-figure .rl-figure-narrow { display: block; }
}
</style>

강한 전략 하나를 가지고 있어도, 그다음에 무엇을 바꿀지는 저절로 드러나지 않는다. 일손을 한 명 더 고용하면 돈만 낭비할 수도 있다. 땅을 사면 돌볼 사람이 부족하고, 토마토를 늘리면 수확 시점에 가격이 무너질 수도 있다. 무엇을 바꿀지 모르는 상태에서 벤치마크만 빨라져도 새 전략은 생기지 않는다.

에이전트가 지금 볼 수 있는 현금·밭·인력·시장 정보를 **observation**, 고용·이동·판매처럼 내리는 명령을 **action**이라고 한다. **policy**는 observation을 받아 action을 고르는 함수다. 규칙으로 작성할 수도 있고 신경망으로 학습할 수도 있다.

Kaggriculture의 강자들은 이 함수를 여러 방법으로 개선했다. 강자의 기록에서 observation과 action의 관계를 배우고, 그렇게 만든 policy로 새 경기를 하며 action 확률을 바꿨다. 경제 규칙을 바탕으로 인력·작물·작업 계획을 탐색하기도 했다. 각각 **BC, PPO, 계획 탐색**이다.

| 방법 | 직접 바꾸는 것 | 좋아졌다는 학습·탐색 신호 |
| --- | --- | --- |
| BC, Behavior Cloning | observation에서 action을 내는 신경망 가중치 | 강자의 action에 주는 확률 |
| PPO, Proximal Policy Optimization | 새 경험을 만든 확률적인 policy의 가중치 | 예상보다 좋거나 나빴던 경기 결과 |
| 계획 탐색 | 인력·작물·작업·이동·거래를 묶은 계획 | 실행 가능성과 예상 수익·비용 |

사람은 이 계산의 재료와 수정 방식을 설계했다. 알고리즘은 그 안에서 수많은 action 확률과 계획 후보를 바꿨다. 사람이 모든 경기의 719수를 눈으로 보고 하나씩 고친 것은 아니다. 그렇다고 알고리즘이 인간의 설계 없이 경제 운영을 전부 발견한 것도 아니다.

## 1. 단일 RTX 5090의 BC·PPO 구현

[10위의 단일 RTX 5090 writeup](https://www.kaggle.com/competitions/kaggriculture/writeups/10th-place-bc-ppo-on-a-single-rtx-5090)은 비교적 작은 장비로 이 연결을 보여 준다. 저자는 강자의 최근 경기 기록으로 BC를 수행한 뒤, JAX 환경에서 64경기씩 새 경험을 만들며 PPO를 이어 갔다. 데이터 준비·BC 약 3.5시간, PPO 약 19시간이라는 시간은 저자 보고다. 공개 코드는 원래 BC 체크포인트가 유실되어 재학습본으로 대체됐다고 설명한다. 따라서 당시 학습 시간과 공개 재현 절차를 똑같은 실행으로 보지는 않는다.

이 사례의 코드 기준은 공개 저장소의 커밋 `8aa3476`이다. 모델은 지금 보이는 농장과 시장 정보를 받아 일손의 작업·목표 위치·수량과 시장 주문을 출력한다. 준비된 전략 몇 개 중 하나를 고르는 데 그치지 않고, 매 turn의 구체적인 운영을 결정한다.

### 2,380경기에서 BC 학습 표본을 만드는 과정

BC에서 따라 배울 시범을 제공하는 에이전트를 **teacher**라고 한다. 여기서는 강한 에이전트들의 2,380경기에서 양쪽 좌석을 teacher 데이터로 사용했다. 한 경기에 각 좌석의 결정이 719번이므로 다음과 같다.

$$
2{,}380\times719\times2=3{,}422{,}440
$$

342만 개는 서로 독립적인 342만 경기가 아니다. **한 경기 안에서 연속으로 기록한 observation–action 표본**이다. 학습 1,891경기와 개발용 489경기로 나누되, 같은 경기의 두 좌석은 같은 쪽에 둔다. 같은 시장과 난수를 공유한 한쪽을 학습에 넣고 다른 쪽을 검증에 넣으면 성능을 낙관적으로 평가하기 쉽다.

변환의 중심인 [`build_episode()`](https://github.com/KawattaTaido/kaggriculture-release/blob/8aa34764564fa94848912a5abede8475ac48d3db/kaggri/tools/bc_lb.py#L348-L477)는 다음 일을 한다.

1. 원래 경기의 설정과 seed로 환경을 만든다.
2. 그 시점에 각 좌석이 볼 수 있는 observation을 인코딩한다.
3. teacher action을 모델의 action 표현으로 변환한다.
4. 원래 양쪽 action으로 환경을 진행한다.
5. 체결·수량·목표 위치 등을 반영해 표본을 저장한다.
6. 마지막 자금이 원래 기록과 맞는지 검사한다.

가령 “5일차, 현금 700, 일손 4명, 토마토 수요 상점 2개”라는 입력에 “일손 추가 고용”이라는 정답이 붙을 수 있다. 같은 날짜라도 자금과 농장 상태가 다르면 고용하지 않는 정답이 붙는다. BC는 날짜별 일정표를 외우는 것보다 넓은 **현재 상황에 따른 선택 규칙**을 배울 수 있다.

신경망에 경기 JSON이나 “현금 700”이라는 문장을 통째로 넣는 것은 아니다. **encoder**가 observation을 정해진 순서의 숫자 배열로 바꾼다. 이 구현의 `map`은 농장의 공간 정보, `global`은 날짜·경제 정보, `unit_feat`와 `unit_pos`는 일손의 상태와 위치를 담는다. 어떤 항목을 어느 척도로 넣는지는 코드가 정한 입력 규칙이다. [실제 입력 배열과 마스크의 선언](https://github.com/KawattaTaido/kaggriculture-release/blob/8aa34764564fa94848912a5abede8475ac48d3db/kaggri/env_const.py#L430-L442)

반대편에는 **decoder**가 있다. 모델이 계산한 action 확률에서 선택을 뽑고, 작업 종류·목표 위치·수량·시장 주문을 엔진이 받는 명령으로 바꾼다. 기록의 action을 학습 정답으로 바꿀 때도 같은 의미를 보존해야 한다.

| 경계 | 한 결정에서 확인할 것 |
|---|---|
| observation → 숫자 배열 | 현금·위치·작물 정보가 언제의 값이며 어느 자리에 들어갔나? |
| teacher 명령 → action 라벨 | 판매할 상품과 수량, 작업 대상이 어떤 번호로 표현됐나? |
| 모델 출력 → 실행 명령 | 같은 번호를 다시 명령으로 바꾸면 원래 의도가 살아 있나? |

**action mask**는 현재 고를 수 없는 선택을 표시한다. 없는 일손의 작업이나 자원이 없어 실행할 수 없는 명령을 제한하고, 남은 선택 사이에서 확률을 계산한다. action 라벨과 mask가 어긋나면 정답 action에 확률을 줄 수 없거나, 학습 때와 실행 때 서로 다른 문제를 풀게 된다.

### BC loss와 teacher action의 확률

다음 숫자는 계산을 위한 가정이다. 어느 상태에서 action을 세 가지로 단순화하고, teacher는 밀을 판매했다고 하자.

| action | 현재 모델의 확률 |
| --- | ---: |
| 밀 판매 | 10% |
| 밀 구매 | 20% |
| 거래하지 않기 | 70% |

**loss**는 정답을 얼마나 못 맞혔는지 수치로 나타낸 것이다. BC는 teacher가 고른 action에 낮은 확률을 줄수록 loss가 커지도록 정의한다. 기본 식은 그 action 확률의 음의 로그다.

$$
L_{\mathrm{BC}}=-\log \pi_\theta(a_{\mathrm{teacher}}\mid s)
$$

여기서 $s$는 앞의 현금 700·일손 4명·밭과 상점 정보 등을 숫자로 바꾼 observation이고, $a$는 teacher가 그 상황에서 선택한 명령이다. $\theta$는 신경망 가중치다. $\log$는 자연로그이며, `exp`는 그 역함수다. teacher action의 확률이 1이면 loss는 0이고, 확률이 0에 가까워질수록 loss는 커진다. 처음에는 $-\log0.1\approx2.30$이다. 학습 뒤 같은 action의 확률이 60%가 되면 $-\log0.6\approx0.51$로 줄어든다.

이 loss에는 승리 여부가 직접 들어 있지 않다. 같은 observation에서 **teacher가 고른 action을 얼마나 낮게 평가했는가**를 센다. 여러 표본의 loss 평균을 줄이는 것이 BC의 학습 목표다. PPO는 teacher의 정답 대신 자기 선택의 결과를 학습 신호로 사용한다.

### Logits·softmax·gradient의 계산 예제

신경망이 처음부터 합계 1인 확률을 출력할 필요는 없다. action마다 제한 없는 실수인 **logit** $z_i$를 출력하고, softmax로 확률을 만든다. 각 값에 지수함수 $e^{z_i}$를 적용하면 모두 양수가 된다. 그 값을 전체 합으로 나누므로 확률의 합은 1이 되고, logit이 큰 action일수록 높은 확률을 받는다.

$$
\begin{aligned}
p_i&=\frac{e^{z_i}}{\sum_k e^{z_k}},\\
\frac{\partial L_{\mathrm{BC}}}{\partial z_i}&=p_i-y_i.
\end{aligned}
$$

$y_i$는 teacher action이면 1, 나머지는 0이다. teacher가 첫 action을 골랐으므로 정답 벡터는 `[1, 0, 0]`이다. 앞의 확률에서 시작하는 logits는 `log([0.1, 0.2, 0.7])`로 놓을 수 있다. softmax를 적용하면 원래 확률이 나온다.

$\partial L/\partial z_i$는 logit 하나를 조금 높였을 때 loss가 어느 방향으로 얼마나 움직이는지 나타내는 **gradient**다. 판매의 gradient −0.9는 판매 logit을 높이는 쪽이 loss를 줄인다는 뜻이다. 확률에 −0.9를 바로 더하라는 명령이 아니다. 확률은 logits를 바꾼 뒤 softmax로 다시 계산한다.

| action | 정답 $y_i$ | 현재 $p_i$ | gradient $p_i-y_i$ | loss를 줄이는 logit 변화 |
| --- | ---: | ---: | ---: | --- |
| 밀 판매 | 1 | 0.1 | −0.9 | 높인다 |
| 밀 구매 | 0 | 0.2 | +0.2 | 낮춘다 |
| 거래하지 않기 | 0 | 0.7 | +0.7 | 더 많이 낮춘다 |

<figure class="kaggriculture-rl-figure">
<div class="rl-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-04-bc-update.png" alt="BC 한 번의 갱신 전후 action 확률과 logit 변화 막대그래프" width="1800" height="992" loading="lazy">
</div>
<div class="rl-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-04-bc-update-mobile.png" alt="BC 한 번의 갱신 전후 action 확률과 logit 변화 막대그래프" width="960" height="1530" loading="lazy">
</div>
<figcaption>그림 2. 세 action 예제에서 학습률 0.5로 logits를 한 번 갱신한 결과다. SELL 확률은 0.100에서 0.189로 오른다. BUY logit은 내려가도 확률은 0.200에서 0.218로 오를 수 있다. 세 확률이 함께 정규화되기 때문이다.</figcaption>
</figure>

gradient descent는 `새 값 = 옛 값 − 학습률 × gradient`다. 학습률을 0.5로 둔 한 번의 계산은 다음과 같다. Python 표준 라이브러리만 사용한다.

```python
import math

def softmax(logits):
    peak = max(logits)
    weights = [math.exp(z - peak) for z in logits]
    return [w / sum(weights) for w in weights]

logits = [math.log(p) for p in [0.1, 0.2, 0.7]]
target, learning_rate = 0, 0.5
before = softmax(logits)
gradient = [p - (i == target) for i, p in enumerate(before)]
updated = [
    z - learning_rate * g for z, g in zip(logits, gradient)
]
after = softmax(updated)
print("before:", [round(p, 4) for p in before])
print("after: ", [round(p, 4) for p in after])
print("loss:  ", round(-math.log(after[target]), 4))
```

출력은 `before: [0.1, 0.2, 0.7]`, `after: [0.1887, 0.2177, 0.5935]`, `loss: 1.6676`이다. 판매 확률은 한 번에 60%가 된 것이 아니라 약 18.9%가 됐다. 구매 logit을 낮췄는데 구매 확률은 조금 높아졌다는 점도 보인다. softmax의 확률은 다른 action들과 함께 정규화되므로 각 logit의 변화만 독립적으로 읽을 수 없다.

코드의 `target = 0`은 첫 action인 판매가 정답이라는 표시다. `(i == target)`은 정답 자리에서 1, 나머지에서 0이 되어 수식의 $y_i$ 역할을 한다. `peak`를 빼는 것은 큰 수에 지수를 취할 때 넘치지 않게 하는 계산 방법이며, 모든 logit에서 같은 값을 빼도 softmax의 확률은 같다.

이 코드는 **한 상태의 logits 자체**를 바꾼다. 실제 BC에서는 logits를 만드는 신경망 가중치를 바꾸며, gradient가 여러 층을 거꾸로 전달된다. 같은 가중치를 공유하는 다른 상태의 action도 함께 달라진다. 이 공유가 학습과 일반화를 가능하게 하지만 다른 운영을 망가뜨릴 가능성도 만든다.

gradient를 거꾸로 전달한다는 것은 “이 가중치가 출력 logit에 미친 영향”과 “그 logit이 loss에 미친 영향”을 이어 계산한다는 뜻이다. 이를 **backpropagation**이라고 한다. 사람이 모든 가중치를 조금씩 바꿔 loss를 재측정하는 대신, 미분 규칙에 따라 각 가중치의 영향을 계산한다.

GPU는 “고용 규칙 1번을 바꾼 프로그램”과 “판매 규칙 2번을 바꾼 프로그램”을 하나씩 대전시키는 것이 아니다. 여러 표본을 동시에 신경망에 넣어 gradient를 구한다. **optimizer**는 그 gradient를 실제 가중치 변경으로 바꾸는 규칙이다. 위 코드는 단순한 gradient descent이고, Adam 같은 optimizer는 최근 gradient의 크기와 방향도 이용한다. 학습률은 한 번에 얼마나 움직일지를 조절한다.

실제 BC에는 상대의 거래를 예측하는 보조 loss도 있다. 상대 거래의 미래 기록을 학습 목표로 쓰는 것이며, 제출 때 미래 거래 정답을 미리 입력받는 것은 아니다. 실제 학습 인수는 [`repro/02_train_bc.sh`](https://github.com/KawattaTaido/kaggriculture-release/blob/8aa34764564fa94848912a5abede8475ac48d3db/repro/02_train_bc.sh)에서 확인할 수 있다.

### 한 수의 모방에서 전체 농장 운영으로

teacher 기록에는 연결이 들어 있다. 땅을 사는 상태 다음에는 넓어진 땅에서 작업하는 상태가, 토마토를 심은 다음에는 물을 주고 수확하는 상태가 따라온다. 모델은 이 연쇄의 여러 지점에서 다음 action을 배운다.

그렇지만 연결이 완벽히 보존된다는 보장은 없다. teacher는 일손을 고용했는데 모델은 고용하지 않았다면, 이후에는 teacher의 기록과 다른 농장을 운영해야 한다. 이런 **자기 action으로 생긴 상태의 차이**가 BC의 중요한 약점이다. 기록 속 상태에서 teacher action을 잘 맞힌다는 결과와, 자기 action으로 30일 농장을 잘 운영한다는 결과를 따로 봐야 하는 이유다.

BC의 일차 목표는 이런 검사까지 통과하는 강한 출발점을 얻는 것이다. teacher보다 강해지는 것은 별도 목표다.

## 2. 새 경험을 이용한 PPO의 policy 개선

PPO에서는 BC policy로 **새 경기**를 치른다. 모델이 낸 확률에 따라 action을 뽑는다. 같은 상황에서도 다른 명령이 나올 수 있으므로 기존 운영에서 벗어난 선택을 경험할 수 있다.

그때 최종 자금만 저장하면 부족하다. “어떤 상태에서, 어떤 확률로, 무엇을 골랐고, 당시 결과를 어느 정도 예상했는가”도 필요하다.

게임 결과를 학습에 사용할 숫자로 정한 것이 **reward**다. **return**은 앞으로 받을 reward를 합한 값이며, 설정에 따라 먼 미래의 reward에는 할인을 적용한다. **actor**는 action의 확률을 계산하고, **critic**은 현재 observation에서 기대할 return을 예측한다. 이 예상값을 **value**라고 한다. critic이 고용 여부를 직접 고르지는 않는다. actor의 선택 결과를 평소 기대와 비교할 기준을 제공한다. 두 출력이 같은 신경망 본체를 공유할 수도 있다.

현재 policy로 실제 경기를 진행하며 연속된 데이터를 모으는 과정과 그 결과를 **rollout**이라고 한다. PPO는 rollout을 모은 다음 그 데이터로 actor와 critic을 갱신한다.

| 경험 데이터 | 의미 |
| --- | --- |
| observation $s_t$ | 그 turn의 현금·밭·일손·시장 정보 |
| 선택한 action $a_t$ | 그 상황에서 고른 작업·고용·시장 주문 |
| `old_logp` | 경험 수집 당시 그 action의 log probability |
| `value` | 그때 critic이 예상한 return |
| reward | 게임 결과를 학습용 숫자로 바꾼 값 |
| policy·상대·seed 식별자 | 누구의 경험이며 어떤 세계였나 |

강자의 공개 리플레이에 우리 policy의 `old_logp`가 들어 있지는 않다. 그래서 공개 기록을 현재 policy가 만든 PPO 경험인 것처럼 넣는 대신, 먼저 BC 교재로 쓰고 그 policy로 새 경험을 만든다.

한 turn에 여러 일손의 작업과 시장 주문을 내면, action 확률은 그 명령들을 함께 고를 확률이다. 순서대로 선택한 조건부 확률들을 곱하고, 로그로 저장할 때는 각 log probability를 더한다. 어떤 선택을 허용했는지에 따라 이 값이 달라지므로 mask도 확률 계산의 일부다.

### Reward·value·advantage의 역할

단일 5090 사례의 최종 설정은 경기 종료 시 두 에이전트의 현금 차이를 30,000으로 나누고 −1~1 범위로 제한한 값을 reward로 사용한다. $M_{\mathrm{self}}$와 $M_{\mathrm{opp}}$는 각각 우리와 상대의 최종 현금이다. 두 값의 차이는 코인 단위의 **bank margin**이며, 승점 차이나 Kaggle 평점 차이와는 다르다.

$$
R=\operatorname{clip}\left(\frac{M_{\mathrm{self}}-M_{\mathrm{opp}}}{30{,}000},-1,1\right)
$$

6,000코인 앞서면 $+0.2$다. 30,000코인 이상 앞서면 모두 $+1$이다. 승패만 사용하는 reward와 달리, 같은 승리라도 현금 차이가 클수록 더 높은 값을 받는 구간이 있다. 이 reward를 높인 변화가 실제 승점도 높이는지는 별도로 확인해야 한다. 설정은 [`ppo_common.sh`](https://github.com/KawattaTaido/kaggriculture-release/blob/8aa34764564fa94848912a5abede8475ac48d3db/repro/ppo_common.sh)에 들어 있다.

가령 critic이 어떤 농장 상태에서 $+0.05$를 예상했는데 실제 마지막 reward가 $+0.20$이었다면, 예상보다 $+0.15$ 좋았다. **advantage**는 이렇게 평소 기대와 비교해 선택 결과가 얼마나 좋았는지를 추정하는 학습 신호다. 기대치 역할을 하는 critic의 value를 여기서는 **baseline**으로 사용한다.

실제 PPO는 매 시점의 reward와 다음 value를 GAE라는 계산으로 연결한다. 마지막 reward 하나를 모든 action에 똑같이 붙이거나, 승리한 경기의 모든 수를 정답이라고 선언하는 방법은 아니다.

<figure class="kaggriculture-rl-figure">
<div class="rl-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-05-ppo-data-contract.png" alt="Old policy의 경기에서 만든 고정 rollout buffer를 current policy의 forward 계산과 gradient 갱신에 쓰는 경로" width="1800" height="1350" loading="lazy">
</div>
<div class="rl-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-05-ppo-data-contract-mobile.png" alt="Old policy의 경기에서 만든 고정 rollout buffer를 current policy의 forward 계산과 gradient 갱신에 쓰는 경로" width="960" height="1918" loading="lazy">
</div>
<figcaption>그림 3. 파란 화살표는 데이터 전달과 신경망의 출력 계산(forward), 주황 화살표는 현재 가중치로 돌아가는 gradient다. 자물쇠는 수집 당시 log probability·value·reward를 보존한다는 뜻이다. Advantage도 이 자료에서 계산해 갱신 중에는 고정한다.</figcaption>
</figure>

### 확률비와 clipping을 이용한 갱신

어떤 상태에서 action의 원래 확률이 20%였고, 갱신 중인 모델의 확률이 22%라고 하자. 새 확률과 옛 확률의 비율은 1.1이다.

$$
\begin{aligned}
r_t(\theta)&=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)},\\
\log r_t(\theta)&=\log\pi_\theta(a_t\mid s_t)\\
&\quad-\log\pi_{\mathrm{old}}(a_t\mid s_t).
\end{aligned}
$$

구현에서는 log probability의 차이를 구한 뒤 `exp`를 취해 확률비로 되돌린다. PPO가 최대화하는 한 표본의 clipped objective는 다음이다. 여기서는 갱신 폭을 조절하는 하이퍼파라미터를 $\varepsilon=0.2$로 두어 비교 구간이 0.8~1.2가 된다. 학습 코드에서는 부호를 뒤집은 loss를 최소화한다.

가령 `old_logp = log(0.20)`은 약 −1.6094, `new_logp = log(0.22)`는 약 −1.5141이다. 두 값의 차이에 `exp`를 취하면 1.1이 된다. 경험을 모았을 때보다 **같은 action을 지금 얼마나 더 자주 또는 덜 자주 고르게 됐는지**를 비교하는 수다. 다음 식의 $A_t$는 해당 경험에 대해 미리 계산해 둔 advantage다.

$$
\ell_t=\min\left(r_t A_t,
\operatorname{clip}(r_t,0.8,1.2)A_t\right).
$$

advantage의 부호에 따라 clip이 작동하는 방향이 달라진다. 다음 표는 옛 확률 0.20을 기준으로 한다. advantage가 양수인 action의 확률을 계속 높이거나, 음수인 action의 확률을 계속 낮춰도 **목적함수 값이 끝없이 좋아지지는 않도록** 한다. 이 판단은 학습 신호의 추정에 따른 것이다.

| $A$ | 새 확률 | 비율 $r$ | 원래 항 $rA$ | clipped 항 | 작은 쪽 $\ell$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| +1 | 0.22 | 1.1 | +1.1 | +1.1 | +1.1 |
| +1 | 0.30 | 1.5 | +1.5 | +1.2 | +1.2 |
| −1 | 0.10 | 0.5 | −0.5 | −0.8 | −0.8 |
| −1 | 0.30 | 1.5 | −1.5 | −1.2 | −1.5 |

<figure class="kaggriculture-rl-figure">
<div class="rl-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-06-ppo-clip.png" alt="양수와 음수 advantage에서 확률비에 따라 달라지는 PPO clipped objective 곡선" width="1800" height="1118" loading="lazy">
</div>
<div class="rl-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-06-ppo-clip-mobile.png" alt="양수와 음수 advantage에서 확률비에 따라 달라지는 PPO clipped objective 곡선" width="960" height="1810" loading="lazy">
</div>
<figcaption>그림 4. 가로축은 확률비 $r$, 세로축은 한 표본의 PPO objective $\ell_t$다. 회색 점선은 $rA$, 주황선은 clip을 적용한 값이다. $\varepsilon=0.2$에서 양의 $A$는 $r>1.2$, 음의 $A$는 $r<0.8$인 쪽이 평평해진다. Reward 자체를 잘라 내는 연산은 아니다.</figcaption>
</figure>

두 번째 행에서는 확률을 0.24 이상으로 높여도 이 항의 값이 더 커지지 않는다. 세 번째 행에서는 확률을 0.16 아래로 내려도 마찬가지다. 네 번째처럼 advantage가 음수인 action의 확률을 오히려 높이면 목적함수는 계속 나빠진다.

```python
def ppo_term(old_p, new_p, advantage, epsilon=0.2):
    ratio = new_p / old_p
    clipped = min(1 + epsilon, max(1 - epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)

for advantage, new_p in [(1, .22), (1, .30), (-1, .10),
                         (-1, .30)]:
    term = ppo_term(.20, new_p, advantage)
    print(advantage, new_p, round(term, 2))
```

마지막 숫자는 차례로 `1.1, 1.2, -0.8, -1.5`다. clip은 확률을 강제로 0.16~0.24에 가두는 명령이 아니다. 신경망 가중치를 공유하는 다른 표본의 gradient나 다른 loss 항 때문에 그 범위를 벗어날 수 있다. 그래서 확률 분포의 변화를 재는 KL을 확인하고 갱신을 중단하는 장치도 사용한다. 작은 평균 KL도 배포 때 가장 높은 확률의 action만 고르는 policy의 운영을 보존하지는 못한다. [PPO 논문](https://arxiv.org/abs/1707.06347)과 [Spinning Up의 수식·해설](https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations)이 이 구분을 설명한다. 실제 5090 사례의 갱신은 [`ppo.py`](https://github.com/KawattaTaido/kaggriculture-release/blob/8aa34764564fa94848912a5abede8475ac48d3db/kaggri/train/ppo.py#L204-L274)에 있다.

### PPO 갱신 중 old probability와 advantage의 고정

한 rollout으로 학습하는 동안 `old_logp`와 advantage는 저장한 숫자로 고정한다. actor는 현재 policy가 계산한 `new_logp`를 통해 학습하고, critic은 value를 예측하는 별도의 loss로 학습한다. advantage까지 가중치와 함께 바뀌면 비교 기준 자체가 움직여 버린다. 자동미분 코드에서는 이 경계를 `stop_gradient` 또는 `.detach()`로 표시한다. 미리 계산한 배열을 buffer에 저장했다면 그 배열은 이미 현재 가중치의 미분 대상이 아니다.

`old_logp`를 매번 현재 policy로 다시 덮어쓰면 비율이 늘 1이 된다. 반대로 오래된 온갖 policy의 경험을 계속 섞으면 현재 수집 분포를 기준으로 하는 PPO의 조건이 무너진다. 현재 policy로 새 rollout을 모으고, 허용한 횟수만 그 묶음을 갱신한 뒤 다시 새 경험을 수집하는 이유다. [Spinning Up의 buffer와 갱신 코드](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html)

이때 “그 고용 하나가 승리의 원인이었다”는 증명이 생기지는 않는다. 상점 추첨과 상대 action, 다른 수많은 선택이 섞여 있다. critic과 많은 새 경험으로 잡음을 줄이며 유리한 방향을 추정한다. 긴 투자 연쇄를 연결해야 하는 게임에서 BC가 중요한 이유도 여기에 있다. 처음부터 합법 action을 무작위로 고르며 고용→확장→생산→수확→판매를 우연히 완성하기는 어렵다.

단일 5090 사례도 PPO만 계속 돌리면 저절로 좋아진다고 가정하지 않았다. teacher 분포에서 급격히 벗어나지 않게 하는 KL, 작은 학습률, 분포 변화 guard를 사용했다. 학습 경험을 재사용하는 횟수와 KL 중단도 관리했다.

여기까지의 한 주기는 `가중치 고정 → 새 rollout 수집 → 고정한 reward·value로 advantage 계산 → 같은 관측의 새 확률 재계산 → 가중치 갱신`이다. PPO loss를 계산하기 위해 후보마다 경기를 다시 치르는 것은 아니다. 경기는 경험 수집 때 진행했고, 갱신할 때는 그 observation과 action을 신경망에 다시 넣는다.

## 3. 우승팀의 BC·PPO·planner 개선 과정

우승팀의 출발도 강자 기록의 BC였다. 하지만 최종 경로는 “BC 한 번, PPO를 끝까지”로 끝나지 않는다. [`training-lineage.md`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/training-lineage.md#L10-L50)에 남은 과정은 더 넓다.

<figure class="kaggriculture-rl-figure">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/winner-training-loop.png" alt="우승팀 원본 그림: 공개 리플레이 BC와 self-play PPO에서 약한 조건을 찾아 heuristic planner와 teacher 데이터를 보강하는 순환" width="3200" height="2100" loading="lazy">
<figcaption>그림 5. 우승팀의 <a href="https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/images/overview.png">Training Loop 원본</a>. 토마토·상점 조건의 약점을 찾은 뒤 heuristic planner를 고치고, 그 조건의 플레이를 BC 데이터에 추가한다. 출처: msdsm, kaggriculture-1st-solution.</figcaption>
</figure>

학습한 policy는 토마토 수요가 높거나 특정 상점이 몰린 세계에서 약했다. 우승팀은 이 조건의 시범을 보강했다. 경제 규칙과 탐색으로 운영 계획을 만드는 프로그램인 **planner**를 개선하고, 해당 조건에서 처음부터 끝까지 경기한 기록을 BC 데이터에 추가했다.

PPO를 더 오래 돌리는 방법은 현재 policy의 시도 안에서 좋은 토마토 운영이 나올 것이라고 기대한다. 반면 teacher를 개선하면 더 나은 토마토 운영을 학습 자료에 직접 넣을 수 있다. policy가 혼자 발견하기 어려운 긴 투자·작업 조합을 planner가 먼저 찾아 주는 셈이다.

### 우승팀의 BC 데이터 전처리

공개 우승 코드는 커밋 `84057a0f`를 기준으로 읽었다. [`prepare_replays.py`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/data/prepare_replays.py#L75-L134)는 teacher 좌석 하나에서 다음 배열을 만든다.

앞의 단일 5090 구현과 입력 형식은 다르다. observation을 숫자로 바꾸고 action 라벨을 붙인다는 원리는 같지만, 같은 배열을 두 모델에 그대로 넣을 수는 없다.

| 배열 | 크기 | 뜻 |
| --- | --- | --- |
| `features` | `719 × 264 × 124` | 시점별 observation 토큰과 feature |
| `labels` | `719 × 30` | 유닛 최대 20칸과 시장 주문 10칸의 정답 |

여기서 **token**은 문장 속 단어가 아니라 밭 한 칸이나 일손 하나처럼 정보를 담는 자리다. 각 자리에는 같은 길이인 124개의 feature, 즉 수치 특징을 넣는다. `719 × 264 × 124`는 “719번의 결정마다 264자리의 정보를 124개 숫자로 표현했다”는 뜻이다. `labels`의 30자리는 20개 유닛 action과 순서 있는 10개 시장 주문에 대응한다. 없는 유닛과 무시할 label은 action cross-entropy 계산에서 제외한다. 다만 실제 유닛과 시장 주문 10칸의 entropy·teacher KL은 해당 observation에서 계속 계산한다. [Label과 regularization 처리](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/docs/data.md#L35-L39)

264개 토큰은 두 농장의 칸 200개, 유닛 최대 40개, 전역 상태, 품목, 상대 재고 추정, 시장 주문 자리 등으로 나뉜다. **attention**은 한 자리의 정보를 해석할 때 다른 자리 중 어디를 참고할지 학습하는 연산이다. 일손의 action을 정할 때 가까운 작물과 시장 수요를 함께 참고하는 식이다. Transformer는 이 연산을 반복하는 신경망 구조다. 상대의 실제 숨은 재고를 입력하는 대신 observation의 변화에서 추정한 값을 사용한다.

데이터 정렬도 중요하다. 기록의 `steps[t]` observation에는 `steps[t+1]`에 저장된 action을 붙인다. action이 실행된 다음 행에 기록되기 때문이다. 이 한 칸을 잘못 맞추면 “action 전에 무엇을 보고 결정했나”라는 학습 문제가 깨진다.

정답도 명령 원문을 모두 그대로 쓰지는 않는다. 효과가 없던 실패 명령은 action cross-entropy에서 제외하고, 판매는 실제 체결 수량을 라벨로 만든다. 주문은 순서를 보존한다. 반대로 모든 구매 명령의 수량을 체결 수량으로 바꾼다고 일반화해서는 안 된다. 구매와 판매의 라벨 처리에는 차이가 있다. [`label_actions()`와 합법성 처리](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/data/prepare_replays.py#L30-L72)는 이런 작은 구현 차이를 명시한다.

### PPO 시작 전 critic 사전 학습

BC는 teacher action을 맞히도록 학습한다. 같은 신경망을 쓴다고 critic도 return을 잘 예측하게 되는 것은 아니다. 부정확한 value를 baseline으로 사용하면 잘 운영하던 policy를 오히려 나쁜 방향으로 바꿀 수 있다.

공개된 절차는 actor와 Transformer 본체의 가중치를 고정하고 self-play 데이터를 모은 뒤, 본체의 출력에서 value를 계산하는 critic만 먼저 학습한다. 본체 가중치가 고정돼 있으므로 그 가중치의 gradient는 구하지 않는다. 계산한 feature를 cache에 저장해 critic 갱신마다 본체의 forward를 반복하지 않도록 한다. best checkpoint를 저장할 때 actor의 hash가 그대로인지 검사한다. 이후 PPO를 시작하면 본체와 actor도 다시 학습한다. [`critic.py`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/training/critic.py#L72-L191)

우승팀 공개 기본 PPO의 reward는 단일 5090 사례와 다르다. 마지막 자금 차이의 **부호**, 즉 승리 $+1$, 무승부 $0$, 패배 $-1$이다. 한 코인 차이 승리와 큰 차이 승리를 동일하게 취급한다. [`rollout.py`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/training/rollout.py#L857-L865)

공개 기본값의 GAE는 다음 형태다.

$$
\begin{aligned}
\delta_t&=R_t+\gamma V(s_{t+1})-V(s_t),\\
A_t&=\delta_t+\gamma\lambda A_{t+1}.
\end{aligned}
$$

여기서 $R_t$는 해당 시점의 reward다. 앞에서 쓴 확률비 $r_t$와 구분한다. $V(s_t)$는 그때 보이는 돈·밭·인력·시장 정보를 입력했을 때 critic이 예측한 value다. $\delta_t$는 한 turn을 진행한 뒤 받은 reward와 다음 value를 기준으로, 이전 예상이 얼마나 빗나갔는지 나타낸다.

여기서는 $\gamma=1$, $\lambda=0.97$이다. $\gamma$는 미래 reward의 할인율이고, $\lambda$는 뒤 시점의 오차를 앞 시점에 얼마나 반영할지 조절한다. 경기가 끝나면 다음 value와 advantage를 0으로 놓고 마지막 turn부터 거꾸로 계산한다. $\delta_t$를 **TD error**라고 부른다. 이 방식은 실제 마지막 결과뿐 아니라 중간 상황에 대한 critic의 추정도 사용한다.

critic의 학습 정답은 advantage에 수집 당시 value를 더한 `return = advantage + value`다. 이 배열은 실제 reward를 끝까지 합친 값과 항상 같지는 않다. 아래 예제처럼 critic의 중간 예측도 반영한 학습 목표다.

결정이 두 번뿐인 경기를 가정하자. critic의 value는 처음 0.1, 다음에 0.2이고, 마지막에 승리 reward 1을 받는다. 마지막 시점부터 계산하면 다음과 같다.

| 시점 | reward $R$ | 수집 당시 $V$ | 다음 $V$ | TD error $\delta$ | advantage $A$ | critic 정답 $A+V$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 마지막 $t=1$ | 1 | 0.2 | 0 | 0.8 | 0.8 | 1.0 |
| 처음 $t=0$ | 0 | 0.1 | 0.2 | 0.1 | $0.1+0.97\times0.8=0.876$ | 0.976 |

마지막 시점에는 다음 상태가 없으므로 `1 − 0.2 = 0.8`이 오차이자 advantage다. 처음 시점에는 우선 `0 + 0.2 − 0.1 = 0.1`을 계산한 다음, 뒤에서 온 신호 `0.97 × 0.8`을 더한다. 이 계산이 종결 결과를 앞선 결정으로 전달한다. 첫 action이 혼자 승리를 만들었다는 뜻은 아니다.

첫 시점의 critic 정답이 1보다 조금 작은 것은 $\lambda=0.97$일 때 다음 value도 반영되기 때문이다. 이처럼 자신의 다른 예측을 이용해 학습 목표를 만드는 방식을 **bootstrapping**이라고 한다. $\lambda=1$이면 이 두 시점 예제의 정답은 모두 마지막에 받은 reward 1이 된다.

```python
rewards, values = [0.0, 1.0], [0.1, 0.2]
gamma, lam = 1.0, 0.97
advantages = [0.0] * len(rewards)
next_value, next_advantage = 0.0, 0.0
for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * next_value - values[t]
    advantage = delta + gamma * lam * next_advantage
    advantages[t] = advantage
    next_value, next_advantage = values[t], advantage
targets = [a + v for a, v in zip(advantages, values)]
mse = sum((v - y)**2 for v, y in zip(values, targets)) / 2
learning_rate = 0.1
predictions = [
    v - learning_rate * 2 * (v - y) / len(values)
    for v, y in zip(values, targets)
]
print("advantage:", [round(a, 3) for a in advantages])
print("target:   ", [round(y, 3) for y in targets])
print("value MSE:", round(mse, 6))
print("new value:", [round(v, 4) for v in predictions])
```

출력은 advantage `[0.876, 0.8]`, 정답 `[0.976, 1.0]`, 평균 제곱오차 `0.703688`이다. 마지막의 작은 gradient descent에서는 예측 `[0.1, 0.2]`가 고정 정답 쪽인 `[0.1876, 0.28]`로 이동한다. logits 예제처럼 여기서는 예측값 자체를 바꿨고, 실제 critic은 그 예측을 만드는 신경망 가중치를 바꾼다. 같은 경험에서 actor는 advantage로 action 확률을, critic은 고정 정답과 현재 예측 차이로 value 예측을 학습한다.

실제 우승팀의 value loss는 이 예제의 MSE 대신 Huber loss다. 둘 다 정답과 예측의 차이를 줄이지만, Huber loss는 아주 큰 오차에 대해 MSE처럼 제곱만큼 빠르게 벌점을 키우지 않는다. policy 학습에는 advantage의 평균과 크기를 맞추는 정규화도 적용한다. 이 처리가 코드의 숫자 척도를 바꾸더라도, 경기 reward 자체를 바꾼 것은 아니다. [실제 GAE와 return 계산](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/training/rollout.py#L408-L433)

실제 학습에서는 저장한 observation 묶음을 현재 신경망에 다시 넣어 **현재 action 확률과 value**를 함께 계산한다. actor loss에는 현재 확률·옛 확률·고정 advantage를 사용하고, critic loss에는 현재 value와 고정 정답의 차이를 사용한다. 두 loss를 가중합하고 backpropagation으로 gradient를 구해 출력층과 공유 본체의 가중치를 바꾼다. 그 가중치로 다음 rollout을 모으므로, 앞서 계산한 확률비와 예측 오차가 실제 농장의 다음 선택에 영향을 준다.

목적함수는 PPO policy 항, value 예측 오차, 고정 teacher와의 KL, action entropy를 결합한다. `jax.value_and_grad`와 Adam이 많은 가중치를 동시에 바꾼다. teacher는 공개 기본 루프에서 주로 **action 분포를 보존하는 기준**이다. 현재 policy의 양쪽 복사본이 싸우는 self-play와, 고정 teacher의 KL 기준을 혼동하면 안 된다. [`objectives.py`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/training/objectives.py#L649-L718)

### Entropy·teacher KL·old/new KL의 차이

| 항 | 비교 대상 | 작용하는 방향 | 지나치게 강하면 |
| --- | --- | --- | --- |
| entropy 보너스 | 현재 policy 자신의 action 분포 | 한 action에 너무 빨리 몰리지 않게 함 | 이미 좋은 action도 불필요하게 흔들림 |
| teacher KL 벌점 | 고정 teacher와 현재 policy | teacher가 주던 확률 분포를 보존함 | teacher의 약점을 벗어나기 어려움 |
| old/new KL 점검 | rollout을 모은 policy와 현재 policy | 한 번의 갱신에서 큰 분포 변화 감시 | 갱신이 너무 일찍 중단될 수 있음 |

세 action의 entropy는 $H(p)=-\sum_i p_i\log p_i$다. 균등한 `[1/3, 1/3, 1/3]`에서 약 1.099로 가장 높고, `[1, 0, 0]`에 가까우면 0에 가까워진다. 높은 entropy만으로 좋은 전략을 알 수는 없다. 가능한 action을 더 탐색하게 만드는 유인이다.

반면 teacher 분포가 $q$일 때 우승 코드의 KL은 $\sum_iq_i\log(q_i/p_i)$다. teacher가 판매에 높은 확률을 줬다면 현재 policy도 그 선택을 쉽게 버리지 못하게 한다. entropy는 현재 분포만 보지만, teacher KL은 따라야 할 구체적인 분포와 비교한다. [`교사를 고정하는 KL 계산`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/training/objectives.py#L172-L176)

KL은 여기서 두 확률 분포가 얼마나 달라졌는지 재는 값이다. 같으면 0이고, 특정 action 하나의 확률 차이만 보는 값은 아니다. actor의 최종 학습 loss에는 좋은 결과의 action 확률을 높이는 PPO 항과 함께, 분포가 너무 빨리 몰리는 것을 막는 entropy 항, teacher로부터 멀어지는 데 비용을 주는 KL 항이 들어간다. 그러므로 PPO 항 하나가 좋아졌다고 전체 갱신이나 대전 강도가 반드시 좋아진 것은 아니다.

## 4. 사람의 설계와 알고리즘의 반복 계산

학습이 자동이어도 실험 설계까지 자동이 되는 것은 아니다.

| 사람이 설계하거나 고른 것 | 알고리즘이 반복해서 계산한 것 |
| --- | --- |
| 어떤 teacher의 어느 버전 기록을 쓸지 | BC 표본의 loss와 가중치 gradient |
| observation·action·합법성 표현 | 각 상태의 action 확률 |
| reward와 KL·entropy·학습률 | 새 경험의 advantage와 PPO 갱신 |
| 어떤 세계에서 약한지 진단 | 주어진 세계의 탐색 계획 후보 |
| 고용·생산을 함께 바꾸는 탐색 연산 | 작업 배치·경로·작물 조합의 비교 |
| 미래 수익과 위험의 근사식 | 바뀐 계획의 근사 점수 |
| 채택용 상대와 검증 절차 | 같은 조건에서의 경기 결과 |

고득점을 만든 세부 action을 사람이 모두 손으로 쓴 것은 아니다. 그러나 **알고리즘이 유망한 변경을 제안할 수 있는 표현과 수정 연산**은 사람이 설계했다. 우리에게 부족했던 부분도 단순히 benchmark 횟수라기보다, 약점을 발견한 다음 어떤 생성기를 어떻게 고칠지 연결하는 방법이었다.

우승팀의 공개 자료는 이 구조를 확인하게 해 주지만, 모든 수동 변경의 실험 일지와 기여도 표를 제공하지는 않는다. 특정 휴리스틱 하나가 몇 점을 만들었다고 역산할 수는 없다.

## 5. 탐색에서 사용하는 계획의 자료구조

우승팀 C++ 탐색기의 핵심 구조는 [`terminal_search.cpp`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/terminal_search.cpp#L35-L67)에 있다. 원문 필드 전체보다 포함 관계를 먼저 이해하는 편이 쉽다.

이 탐색에서 바꾸는 것은 신경망 가중치가 아니라 **오늘 실행할 계획**이다. 현재 observation에서 계획 여러 개를 만들고, 자금과 시간 제약을 지키면서 어느 쪽이 더 유리할지 계산한다. 내일은 바뀐 농장의 observation을 받아 다시 계획한다.

```text
Plan: 오늘의 농장 운영 후보
 ├─ Route: 농부 또는 일손 한 명의 작업 경로
 │   ├─ Task: 어느 칸을 방문할지
 │   │   └─ Option: 그 칸에서 할 작업 묶음
 │   └─ 창고 반입, 시작 시각, 야간 반입 여부
 ├─ 토지 확장 허용 여부
 ├─ 품목별로 남겨 둘 재고와 판매 시점
 └─ 고용·구매 등 준비 주문의 방식
```

`Task`는 좌표와 여러 `Option`을 가진다. 같은 작물 칸에도 “그냥 두기”, “물만 주기”, “비료와 물”, “물 주고 수확”, “수확 뒤 제거” 같은 대안이 생긴다. 빈칸에는 심을 작물별 대안이 생긴다.

`Option`에는 action 목록뿐 아니라 실행 뒤 칸 상태, 필요한 씨앗·먹이·비료, 생산물과 미래 가치 계산에 필요한 정보가 들어 있다. `Route`는 `(task 번호, option 번호)`를 순서대로 가진다. 경로가 하나 더 생기면 그날 사용할 인력도 늘어난다.

이 표현 덕분에 “토마토 세 칸을 늘리고 일손 두 명을 추가한다”는 변경을 action 문자열 719개를 손으로 고치는 대신, **작업 대안과 경로의 재배치**로 다룰 수 있다. 수정된 표현을 실행 명령으로 내보내는 일은 별도의 실행기가 맡는다.

## 6. 초반 투자안 구성: 고용·토지·작물의 동시 변경

이제 코드의 비용 규칙을 사용하는 가상 상태를 놓자. 날짜는 0부터 세므로 코드의 `day=4`는 다섯째 날이다. 그날의 초기 고용이 끝나 농부 포함 4명이 있고, 현금은 4,000코인이다. 다음 구획의 가격은 1,000코인이며, 새 작물 씨앗은 없다. 기존 농장을 유지하는 작업도 수행해야 한다.

| 후보 | 오늘의 총인력 | 토지 확장 | 새 토마토 | 새 딸기 |
| --- | ---: | --- | ---: | ---: |
| A | 4명 | 없음 | 0칸 | 0칸 |
| B | 6명 | 있음 | 3칸 | 3칸 |
| C | 6명 | 있음 | 6칸 | 0칸 |
| D | 6명 | 있음 | 0칸 | 6칸 |

이 표를 사람이 전부 나열해 줘야만 하는 것은 아니다. 실제 `begin_search()`는 여러 인력 수와 토지 확장 여부로 시작 계획을 만든다. 이후 `full_search_step()`은 작업을 일부 지우고 다시 넣으며, 인력을 여러 명 바꾸고 생산 구성을 함께 재구성한다. 코드 주석도 “일손 한 명씩 늘리는 탐색은 중간 후보를 기각해 유리한 조합에 도달하지 못할 수 있다”고 이유를 적는다. [`묶음 재구성 연산`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/search_methods.inc#L286-L307)

인력만 늘렸을 때 손해라고 해서 확장이 불필요한 것은 아니다. 땅만 샀을 때 손해라고 해서 새 생산 체계가 약한 것도 아니다. **땅·작물·작업 인력을 함께 늘려야 수익이 나는 조합**을 하나의 후보로 제안할 수 있어야 한다.

### 후보 B의 당일 추가 비용

코드의 기본 임금은 고용 순서에 따라 피보나치 수열처럼 증가한다. 농부를 포함한 총인력 4명의 누적 고용비는 4, 6명의 누적 고용비는 12다. 이미 4명이 있으므로 추가 2명의 비용은 $12-4=8$이다.

토마토 씨앗은 50, 딸기 씨앗은 100코인이다. 따라서 B의 추가 지출은 다음과 같다.

$$
\underbrace{8}_{\text{추가 고용}}+
\underbrace{1{,}000}_{\text{토지}}+
\underbrace{3\times50}_{\text{토마토}}+
\underbrace{3\times100}_{\text{딸기}}
=1{,}458
$$

C는 1,308, D는 1,608코인이다. B는 당장 A보다 은행이 1,458 줄어든다. 그래서 오늘의 잔액만 최대화하는 평가기는 이 투자를 거절한다.

실행 가능성에는 여유 자금도 필요하다. 이 상태에서 코드의 운영 준비금이 최솟값인 75로 계산된다면 $1{,}458+75\le4{,}000$이므로 현금 검사에 통과한다. 준비금은 **검사에 사용하는 여유분**이지 실제로 지불한 추가 비용이 아니다. 지출 후 현금은 2,542다. 준비금을 수익 계산에서 다시 빼면 이중 차감이 된다. [`고용비·평가·준비금`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/terminal_search.cpp#L548-L564)

### 현금·시간·재고를 고려한 실행 가능성 검사

B는 추가 일손이 실제로 등장하기 전에 그 일손에게 작업을 맡길 수 없다. 구매한 씨앗이 사용 가능해지기 전에 심거나, 구획이 열리기 전에 들어갈 수도 없다. 기존 농장의 수확을 놓치면서 새 밭만 돌보면 경제 계산도 무너진다.

[`preparation_uncached_into()`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/preparation_methods.inc#L20-L150)는 고용·토지·씨앗 등의 주문과 사용 가능 시점을 계산한다. 한 turn에 시장 주문은 최대 10개다. 새로 고용한 일손과 구입한 씨앗이 다음 농장 작업 단계부터 유효해지는 시점도 반영한다.

그다음에는 각 일손의 이동 경로와 작업 시간표를 맞춘다. 어디서 출발해 몇 칸 이동하고, 심고 물을 주고, 기존 작물을 수확하고, 창고까지 돌아갈 수 있는지 계산한다. 총인력이 6명이라는 조건만으로는 부족하다. 시간과 자원 제약을 지키는 `Route` 여섯 개가 있어야 한다.

## 7. 작물별 생산 일정과 미래 수익 계산

`add_plant_variants()`는 현재 심을 수 있는 작물 대안을 만든다. 토마토와 딸기를 코드의 품목 번호와 함께 쓰면 다음과 같다.

| 작물 | 품목 번호 | 씨앗 값 | 첫 생산 나이 | 이후 생산 간격 |
| --- | ---: | ---: | ---: | ---: |
| 토마토 | 2 | 50 | 8일 | 1일 |
| 딸기 | 3 | 100 | 10일 | 2일 |

비료나 물 공급, 누적 수확량에 따라 이후 상태가 달라지므로, “씨앗 가격 × 고정 배수”만으로 가치를 매기지 않는다. [`enumerate()`와 `value()`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/terminal_search.cpp#L339-L432)는 한 칸에 가능한 작업 묶음을 열거하고 다음 날 상태로 넘긴다.

한 칸의 작업안을 비교하는 식을 단순화하면 다음과 같다.

$$
\begin{aligned}
V_{\mathrm{tile}}(x,d)
&=\max_{o\in\mathcal O(x,d)}Q(x,d,o),\\
Q(x,d,o)&=\mathrm{immediate}(o)\\
&\quad-\mathrm{workCost}(o,d)\\
&\quad+\beta_d V_{\mathrm{tile}}(x',d+1),\\
x'&=\mathrm{next}(x,o).
\end{aligned}
$$

$x$는 작물 종류·나이·물·비료·남은 수확량 같은 칸 상태다. $\mathcal O(x,d)$는 그날 가능한 작업 묶음이고, $Q$는 그중 하나를 골랐을 때의 평가값이다. 오늘 물을 주고 기다릴지, 수확할지, 마지막 수확 뒤 제거할지 등을 남은 날짜의 수익과 비용까지 고려해 비교한다. 같은 `(칸 상태, 날짜)`를 다시 만나면 이미 계산한 값을 재사용한다. 이 방법이 **DP(dynamic programming)**다.

이 $V_{\mathrm{tile}}$은 앞의 PPO critic과 다른 함수다. critic은 경험에서 학습한 신경망의 예측이고, 이 함수는 사람이 작성한 작물 규칙과 비용식으로 계산한다. $\beta_d$도 여기서는 다음 날의 계산값을 할인하는 계수다. 같은 $V$라는 글자를 쓴다고 같은 모델을 호출하는 것은 아니다.

예를 들어 지금 수확하면 5를 벌고 작업에 1을 쓰며 이후 수익은 없다고 하자. 이 안의 평가값은 4다. 기다리는 안은 오늘 수입 0, 작업비 1, 내일의 평가값 8이고 $\beta_d=0.9$라면 `0 − 1 + 0.9 × 8 = 6.2`다. 이 가정에서 `max`는 6.2인 안을 고른다. 내일의 8도 그날 가능한 선택을 같은 방식으로 비교해 얻는다. 마지막 날에는 미래 항이 없으므로 마지막 날부터 거꾸로 값을 구할 수 있다. 서로 다른 계획에 같은 칸 상태가 등장하면 계산을 반복하지 않고 cache에서 꺼내 쓴다.

`Option.immediate`는 **어느 칸에 어떤 작업을 넣을지 고르는 데 쓰는 임시 평가값**이다. 농장 전체의 성적은 별도로 계산한다. 칸별 예상 판매액을 시장 계산 뒤에 다시 더하면 같은 수입을 두 번 세게 된다.

### 날짜별 생산량·씨앗 비용·작업량 배열

[`project_uncached()`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/future_methods.inc#L73-L117)는 칸별 이후 작업을 다음 세 종류의 날짜별 배열로 바꾼다.

```text
q[날짜][품목] : 생산물과 소모품의 순증감
labour[날짜]  : 후속 씨앗 구입비
work[날짜]    : 작업과 이동 여유를 포함한 노동량 근사
```

`labour`라는 이름과 달리 이 배열에는 나중에 다시 살 씨앗 비용이 들어간다. 임금은 `work`에서 따로 계산한다. 두 배열을 같은 인건비로 읽으면 비용을 잘못 계산하게 된다.

예제의 토마토와 딸기에 매일 물을 주고, 비료 없이 생산되는 대로 수확한다고 가정하면, 코드의 `day=4`에 심은 토마토는 `day=12, 13, 14, 15`, 딸기는 `day=14, 16, 18, 20`에 수확 기회가 생긴다. 이후 날짜 번호도 모두 0부터 세는 코드의 `day`를 뜻한다. 각각 3칸이면 해당 날짜에 3개씩이다. 실제 DP는 물·비료·수확 시점의 다른 대안도 비교한다.

이 단계에서는 **어떤 상품이 언제 얼마나 나오고, 무엇을 얼마나 소모하는지**를 칸별로 정리한다. 아직 토마토 3개의 최종 판매액을 정하지는 않는다. 농장 전체의 생산량을 시장 계산에 넣고, 필요한 작업량에서 임금을 추정하고, 씨앗 비용까지 빼야 해당 계획의 미래 수입과 지출을 비교할 수 있다.

### 예상 작업량에 따른 미래 고용비 계산

Kaggriculture의 hired hands는 밤마다 사라진다. 다음 날 필요한 만큼 다시 고용해야 한다. “오늘 두 명을 추가했으니 끝까지 두 명을 유지한다”는 장기 고정 인건비 모델이 아니다. 참조 엔진도 밤 처리에서 `hands=[]`, `hires_today=0`으로 초기화한다. [`밤 처리`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/reference_engine.py#L593-L596)

따라서 미래에는 각 날짜의 예상 작업량에서 필요한 인력을 다시 추정한다. 탐색기는 노동량을 1/5 turn 단위로 계산하며, 1명당 유효 작업 용량을 110, 즉 22 turn으로 근사한다.

$$
\widehat{n}_d=\max\left(1,\frac{\mathrm{work}[d]}{110}\right)
$$

이 인력에는 처음부터 있는 농부가 이미 포함된다. 여기에 농부 한 명을 다시 더하면 임금을 과대평가한다. 필요한 인력이 정수가 아니면 양옆 정수 인력의 누적 임금을 선형 보간한다. 예를 들어 전체 예상 작업량이 385면 3.5명이다. 총인력 3명의 임금 2와 4명의 임금 4 사이에서 비용 3을 사용한다.

오늘은 실제 경로와 실제 고용비로 실행 가능성을 확인하고, 미래는 예상 노동량과 근사 임금으로 많은 후보를 빠르게 비교하는 구분이다. [`미래 노동 비용`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/future_methods.inc#L141-L161)

## 8. 생산량과 가격 변화를 함께 고려한 수익 추정

토마토 100개를 더 생산한다고 지금 토마토 가격의 100배를 그대로 버는 것은 아니다. 내가 팔면 시장 재고가 늘고 가격이 내려간다. 상대도 같은 상품을 팔 수 있고, 상점이 그 사이 수요를 만든다. 생산 날짜도 다르다.

이 planner는 칸별 예상 생산을 농장 전체로 합친 뒤, 품목별 시장 재고를 날짜 순으로 진행한다. [`trade_difference()`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/terminal_search.cpp#L192-L230)는 자기와 상대의 예측 거래가 만드는 현금 차이를 계산한다. 내 수입뿐 아니라 내 공급 때문에 달라지는 상대 판매액도 목표에 관여한다.

토마토 시장 재고가 10,000인 시점에 우리만 3개를 연속 판매한다고 놓자. 코드의 가격식과 반올림을 적용하면 판매 가격은 60, 57, 56으로 내려가 총 173이다. 3개 모두 60으로 계산한 180과 다르다. 물량이 커지거나 상대가 함께 팔면 차이가 더 중요해진다.

당일에는 판매 이전에 창고에 물건이 들어와야 한다. 일손이 물건을 들고 밭 끝에 서 있는데 판매액을 먼저 더할 수는 없다. [`market_timeline.inc`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/market_timeline.inc#L1-L80)는 경로에서 필요한 `PICKUP`과 `DROP` 사건을 꺼내고, 창고 용량과 시장 주문 자리를 순서대로 처리한다. 모든 빈 turn의 전체 세계를 다시 만드는 대신, 거래 결과에 영향을 주는 사건을 희소하게 처리한다.

### 아직 보이지 않은 상점의 시나리오 평가

미래 상점 종류는 모른다. 이미 열린 상점은 반영하고, 안 열린 부분에는 여러 수요 시나리오를 놓는다. 열린 상점 수에 따라 32·16·8·1개 시나리오를 사용한다. 모두 공개되면 1개로 줄어든다.

여러 상점 조건에서 구한 미래 평가값을 다음처럼 합친다.

$$
F(p)=\operatorname{mean}_s F_s(p)
-0.25\operatorname{std}_s F_s(p)
$$

이 식의 $p$는 계획이고, $s$는 상점 시나리오 번호다. 앞에서 확률을 뜻하던 $p_i$나 observation을 뜻하던 $s_t$와 구분해야 한다. $F_s(p)$는 “상점 조건 $s$에서 계획 $p$를 실행했을 때 앞으로 벌고 쓸 돈을 반영한 평가값”이다. 이 값들의 평균에서 시나리오에 따른 변동폭인 표준편차의 1/4을 뺀다.

시나리오마다 날짜별 수요, 우리 생산, 상대의 추정 공급을 반영해 현금 차이를 계산한다. 거기에서 미래 씨앗 비용과 일일 고용비도 뺀다. 평균이 좋아도 특정 상점 조합에서만 성립하는 계획은 표준편차 항으로 조금 불리해진다. 이 0.25는 설계자가 정한 위험 계수이며 공식 대회 점수의 규칙이 아니다. [`미래 시나리오 합산`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/future_methods.inc#L163-L205)

먼 미래의 수입에는 할인도 적용한다. 현재가 `day=4`일 때 `day=12`의 금액에는 $0.9^4\times0.97^4\approx0.58084$를 곱한다. 앞의 173코인은 약 100.49로 평가된다. 실제 판매액은 여전히 173코인이다. planner가 당장의 지출과 나중의 불확실한 수입을 비교할 때 먼 수입을 덜 반영하도록 만든 것이다.

## 9. 후보 A·B의 비용·수익 비교

농장 전체 평가기 [`evaluate_plan_summary()`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/terminal_search.cpp#L550-L561)의 핵심은 다음과 같다.

```text
준비 주문과 시간표가 불가능하면 탈락
구매비 + 운영 준비금이 현재 현금보다 크면 탈락
오늘 경로·창고·거래를 계산해 현금 차이 산출
마지막 날이 아니면 미래 가치 추가
새로 필요한 씨앗·가축 구매비 차감
```

이를 개념식으로 쓰면 다음이다.

$$
J(p)=T_{\mathrm{today}}(p)+F(p)-C_{\mathrm{new\ inputs}}(p)
$$

오늘의 $T$에는 추가 고용·토지 비용과 당일 거래를, 미래 $F$에는 이후 생산·시장 거래·고용·작물 갱신의 추정치를 반영한다. 현재 가진 현금은 모든 후보에서 같으므로 더하지 않는다. $J$는 후보를 비교할 평가값이지, 최종 현금 잔액을 그대로 예측한 수치는 아니다. `Option.value`도 여기에 모두 합산하지 않는다. 그것은 후보를 만들고 우선순위를 정할 때 쓰던 값이다.

칸별 작업을 정하면 날짜별 생산량과 작업량이 나온다. 이를 상점 시나리오별 시장 계산에 넣어 $F$를 구하고, 오늘의 거래·지출 $T$와 합쳐 $J$를 만든다. $C_{\mathrm{new\ inputs}}$는 오늘 새로 필요한 씨앗·가축의 구입비다. $F$의 씨앗 비용은 이후 작물을 갱신할 때 드는 돈이므로 같은 구입비를 두 번 빼는 것은 아니다.

A와 B가 기존 농장에서 오늘 올리는 매출은 같고, 새 투자 지출과 미래 평가값만 다르다고 가정하면 다음처럼 비교할 수 있다.

$$
J(B)-J(A)=F(B)-F(A)-1{,}458
$$

미래 평가값 $F(B)-F(A)$가 1,700이면 오늘 지출 1,458을 빼도 B가 242만큼 높다. 1,300이면 B가 158만큼 낮다. 새 밭을 돌볼 인건비까지 계산한 뒤에도, 앞으로 남을 돈이 오늘의 추가 지출보다 많은지가 기준이다.

### 네 후보의 평가값을 계산하는 toy evaluator

아래는 계산 구조만 남긴 두 정산 구간의 가상 평가기다. 생산량과 가격·추가 노동비는 예제용으로 정한다. 토마토 한 칸은 `[3, 3]`, 딸기 한 칸은 `[0, 4]`를 생산한다고 놓는다. 세 가격 시나리오는 토마토가 비쌀 때 딸기가 싸고 그 반대도 있는 조건이다. 실제 planner의 날짜별 DP·가격 곡선·경로 검사는 생략한다.

`plans`의 값은 `(인력, 토마토 칸 수, 딸기 칸 수)`다. `prices`의 첫 시나리오 `((100, 80), (70, 60))`은 토마토의 두 구간 가격이 100·80, 딸기의 가격이 70·60이라는 뜻이다. 후보 B의 첫 구간에는 토마토가 `3칸 × 3개 = 9개` 나오고 딸기는 없다. 따라서 할인 전 순수입은 `9 × 100 − 2 × 6 = 888`, 할인 후에는 `0.9 × 888 = 799.2`가 된다. 코드의 안쪽 반복문이 이 계산을 구간별·시나리오별로 한다.

```python
from statistics import mean, pstdev

plans = {"A": (4, 0, 0), "B": (6, 3, 3),
         "C": (6, 6, 0), "D": (6, 0, 6)}
prices = [((100, 80), (70, 60)),
          ((60, 50), (140, 120)),
          ((20, 20), (220, 180))]
discounts = [0.9, 0.81]

def evaluate(plan):
    workers, tomatoes, strawberries = plan
    plots = tomatoes + strawberries
    cost = {4: 4, 6: 12}[workers] - 4
    if plots:
        cost += 1000 + 50 * tomatoes + 100 * strawberries
    if cost + 75 > 4000:
        return float("-inf")
    q_tomato = [3 * tomatoes, 3 * tomatoes]
    q_strawberry = [0, 4 * strawberries]
    values = []
    for tomato_price, strawberry_price in prices:
        flows = []
        for d in range(2):
            income = q_tomato[d] * tomato_price[d]
            income += q_strawberry[d] * strawberry_price[d]
            future_labor = [2, 3][d] * plots
            flows.append(discounts[d] * (income - future_labor))
        values.append(sum(flows))
    future = mean(values) - 0.25 * pstdev(values)
    return future - cost

for name, plan in plans.items():
    print(name, round(evaluate(plan), 2))
```

이 예제에서는 A가 0, B가 약 525, C가 약 146, D가 약 461이므로 B를 고른다. B의 시나리오별 미래 평가값은 `1951.02, 1991.52, 2032.02`다. 평균 약 1991.52에서 표준편차 약 33.07의 1/4, 그리고 오늘 지출 1,458을 빼면 약 525.25가 된다. B가 가장 저렴하지는 않지만, 두 상품을 섞어 생산한 결과 세 가격 조건에서 고르게 수입을 남긴다. 가격이나 생산량을 바꾸면 다른 후보가 이길 수 있다. **후보를 제안하는 과정과 각 후보를 같은 식으로 평가하는 과정**이 어떻게 연결되는지 보여 주는 계산이다.

<figure class="kaggriculture-rl-figure">
<div class="rl-figure-wide">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-07-plan-and-validation.png" alt="토지 확장과 총인원 4명에서 6명 변경 및 토마토 딸기 구성, 추가 비용과 네 후보의 평가값" width="1800" height="1618" loading="lazy">
</div>
<div class="rl-figure-narrow">
<img src="{{ site.baseurl }}/assets/img/posts/kaggriculture-rl-series/fig-07-plan-and-validation-mobile.png" alt="토지 확장과 총인원 4명에서 6명 변경 및 토마토 딸기 구성, 추가 비용과 네 후보의 평가값" width="960" height="1548" loading="lazy">
</div>
<figcaption>그림 6. 위 toy evaluator의 농장 구성과 계산 결과. A는 기존 운영을 유지하고, B는 토지를 넓혀 토마토·딸기를 3칸씩 배치한다. Total workforce는 농부를 포함한 총인원이다. 추가 지출은 1,458코인이고, 하단은 같은 식으로 계산한 A·B·C·D의 평가값이다. 밭과 경로는 후보 구성을 보여 주는 도식이다.</figcaption>
</figure>

실제 코드에서는 각 칸의 생산량과 소모량을 합산하고, 후보를 수정하면 바뀐 부분만 반영해 `q`를 만든다. 작업량에서는 날짜별 고용비를 추정한다. 시장 가격도 고정된 값이 아니라 우리와 상대의 공급에 따라 변한다. 이 작은 예제는 그런 세부 계산을 생략했지만, 날짜별 수입·비용을 합산해 후보를 비교하는 구조는 같다.

실제 A·B·C·D 중 어느 것이 이기는지는 기존 밭의 상태와 위치, 공개 상점, 재고, 상대 공급 등 전체 입력에 달려 있다. **인력 6명과 토마토 3칸이 정답이라는 고정 규칙을 발견한 계산이 아니다.** 현재 상태에서 가능한 여러 연결 계획을 만들고, 같은 평가식으로 고르는 방법이다.

### 토지의 고정 비용과 부분 계획의 조기 기각

고정 비용에는 별도의 함정이 있다. 새 구획의 첫 칸에 토마토를 심을 때 1,000코인을 전부 부담시키면 대부분 손해처럼 보인다. 첫 칸이 기각되면 두 번째부터 여섯 번째 칸은 검사조차 못 한다.

`repair()`는 **계획에 작업을 하나씩 넣는 동안** 새 토지 비용을 일시적으로 보정한다. 첫 작물 하나 때문에 확장안을 버리지 않고 다음 작업도 붙여 볼 기회를 주는 것이다. 완성된 계획의 평가값에는 토지 비용이 포함된다. 땅을 공짜로 사는 것으로 계산하는 것이 아니라, 일부만 만든 계획을 너무 일찍 버리지 않게 하는 장치다. [`농장 전체 평가값을 이용한 작업 추가`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/terminal_search.cpp#L877-L909)

이처럼 후보 생성 단계에는 부분 계획을 완성할 수 있는 설계가 필요하다. 완성된 전략의 점수를 정확하게 읽는 것만으로는 해결되지 않는 문제다.

## 10. 전체 대전 없이 계획 후보를 평가하는 방법

계획 탐색의 계산 흐름을 의사코드로 쓰면 다음과 같다.

```text
best = 여러 인력·토지·재고 조건에서 만든 초기 계획 중 최선
current = best

while 탐색 시간이 남아 있다:
    proposal = current의 일부 작업·경로·생산·인력을 변경
    proposal = 돈과 시간과 자원 제약에 맞춰 재구성
    score = 오늘의 경량 사건 계산 + 미래 가치 근사

    if 더 좋거나, 온도에 따라 일부 악화를 허용한다면:
        current = proposal
    if proposal이 best보다 좋다면:
        best = proposal

return best를 오늘의 실행 명령으로 변환
```

실제 수정에는 작업 추가·제거, 작업 옵션 교체, 다른 일손에게 옮기기, 방문 순서 바꾸기, 여러 생산 칸의 재구성, 인력 수 변경 등이 있다. 조금 나쁜 중간 계획도 확률적으로 허용하는 simulated annealing이 있다. 의사코드의 온도는 이런 중간 악화를 얼마나 허용할지 조절하는 탐색 파라미터다. 좋은 후보는 따로 저장하며 여러 생산 계열의 시작점도 유지한다. 현재 계획 주변의 아주 작은 개선만 반복하다 막히는 일을 줄이려는 장치다. [`search_methods.inc`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/search_methods.inc#L220-L472)

계산량을 줄이는 방법도 구조에 맞춰져 있다.

| 재사용·근사 | 피하는 계산 |
| --- | --- |
| 같은 칸 상태의 DP 결과 저장 | 남은 기간의 생산과 비용을 같은 조건에서 다시 계산 |
| 칸별 날짜별 생산량과 수정에 따른 차이 저장 | 후보마다 모든 칸의 미래 재계산 |
| 바뀐 작업의 생산·노동 차이만 반영 | 바뀌지 않은 농장 부분 재계산 |
| 품목별 시나리오 경로 저장 | 같은 공급 흐름의 시장 계산 반복 |
| 창고 반입·반출 사건 중심 계산 | 후보마다 30일 공식 대전 |
| 경로별 길이·자원 요약 재사용 | 작은 수정 때 모든 경로 전개 |

가령 딸기 한 칸의 수확 시점만 바뀌면, 기존 합계에서 그 칸의 이전 생산 일정을 빼고 새 일정을 더한다. 이 차이로 인해 필요한 인력과 예상 판매액이 얼마나 변하는지 계산한다. 밭 전체를 처음부터 다시 시뮬레이션할 필요는 없다.

예상 가격도 한 번 정하고 끝내지 않는다. `continuation_prices()`는 특정 날짜에 생산물이 **한 개 더 나왔을 때 우리와 상대의 예상 현금 차이가 얼마나 변하는지**를 계산한다. 이 변화량과 기존 예상 가격을 50:50으로 섞고, 그 가격으로 칸별 작업을 다시 고른다. 한 작물을 너무 많이 생산해도 계속 같은 가격에 팔릴 것처럼 평가한 계획을 조정할 수 있다. [`추가 생산량의 영향과 작업 계획 재계산`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/search/future_methods.inc#L208-L282)

이 미래 모델이 남은 30일의 모든 투자를 설계하는 것은 아니다. 오늘의 투자로 만들어질 농장을 기준으로 관리·수확·일부 작물 갱신을 계산한다. 내일 모든 빈칸에 새 토마토·딸기 투자를 할 수 있다고 가정하지는 않는다. 다음 새벽의 실제 observation을 받으면 다시 계획한다. **오늘 실행 가능한 작업을 자세히 계산하고, 그 이후는 빠른 근사로 비교하는 방식**이다.

## 11. 내부 평가와 실제 대전 성능의 차이

탐색 중에는 많은 계획의 평가값을 구해야 한다. 매번 공식 엔진에서 전체 경기를 치르면 검토할 수 있는 후보 수가 크게 줄어든다. 그래서 앞의 계산 비용이 작은 평가기를 사용한다.

그 평가기의 가정이 틀리면 탐색도 잘못된 계획을 고른다. 상대 공급을 잘못 예측하거나 미래 인건비를 너무 낮게 계산하면, 실제로는 손해인 계획에 높은 값을 줄 수 있다. 계산상 좋은 계획이 실제로도 강한지 확인할 완전한 경기가 필요하다.

따라서 두 단계가 모두 필요하다.

1. **생성 내부:** 계산 비용이 작은 근사로 많은 계획을 비교하고 연결된 후보를 만든다.
2. **policy 검증:** 실제 규칙 엔진에서 완전한 경기를 수행해 별도의 상대·세계에서 확인한다.

우승팀 공개 평가 도구는 공식 엔진 `1.32.7`에서 같은 seed의 좌석을 뒤집어 비교한다. 그 도구만으로 당시 전체 필드의 상대 분포나 모든 승격 기준을 복원할 수는 없다. 우리가 가져올 핵심은 내부 생성 점수와 외부 검증 결과를 같은 것으로 취급하지 않는 구분이다. [`evaluate.py`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/scripts/evaluate.py#L54-L89)

PPO도 마찬가지다. 같은 policy끼리 싸운 결과를 양 좌석 모두 집계하면, 무승부를 0.5로 센 평균 승점은 0.5다. 한쪽이 얻은 승점을 다른 쪽이 잃기 때문이다. 이 숫자만으로 세대가 강해졌는지 알 수 없다. 학습 loss 감소, teacher 일치율, 탐색 점수, 독립 상대 승점은 서로 다른 질문에 답한다.

## 12. Planner의 운영 기록을 이용한 BC 재학습

우승팀의 마지막 BC 자료는 공개 강자의 경기에서 가져온 849좌석과 planner로 생성한 600좌석이었다. 뒤의 600개는 planner가 운영한 300경기의 양쪽 좌석으로, 승리한 쪽만 고른 자료가 아니다.

| 시범을 만든 상점 조건 | 경기 수 | 해결하려는 문제 |
| --- | ---: | --- |
| 일반 상점 추첨 | 100 | 일반 운영을 잃지 않기 |
| 토마토 관련 상점 최소 2개 | 100 | 토마토 수요가 높은 세계의 약점 |
| 같은 종류 상점 최소 3개 | 100 | 수요가 한 상품에 몰리는 세계의 약점 |

학습 자료를 만들 때는 부족했던 상점 조건을 더 자주 나오게 했다. 상점이 공개되는 시점은 유지하고 열린 상점의 종류만 바꾸므로, 미래 정보를 먼저 알려 주는 방식은 아니다. 이렇게 의도적으로 바꾼 조건의 성적을 일반 대회 분포의 성적으로 해석해서는 안 된다. [`shop_scenarios.py`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/data/shop_scenarios.py#L15-L34)

planner가 만든 observation–action 쌍으로 BC를 다시 학습하면, 신경망은 비슷한 상황에서 그 운영을 따라 할 수 있다. 이후 PPO에서는 policy가 직접 플레이하며 방문한 상황의 결과로 action 확률을 조정한다.

여기서 학습은 서로 다른 능력을 연결한다. 탐색기는 사람이 설계한 경제 구조를 이용해 특정 세계에서 좋은 시범을 만들 수 있다. 신경망은 많은 시범의 상태–action 관계를 일반화하고 빠르게 실행한다. PPO는 현재 policy가 실제로 방문한 상태의 결과로 그 함수를 바꾼다. 어느 하나가 다른 둘의 모든 역할을 대신하지 않는다.

## 13. 학습용 teacher와 최종 제출 agent의 실행 방식

우승팀 저장소에는 전체 시즌을 운영하는 C++ 기반 탐색 teacher와, 최종 Agent A의 마지막 날 탐색이 함께 들어 있다.

teacher 데이터를 만들 때는 탐색 policy가 0일부터 29일까지 운영한다. 매일 새 observation으로 하루 계획을 만들고 실행한다. 반면 최종 Agent A는 신경망을 기본으로 쓰다가 코드의 `day=29`, 즉 마지막 날에 탐색 제어기로 넘긴다. 하루 전 라이브러리를 미리 읽고, 탐색 시작 시 남은 초과시간이 부족하거나 실패하면 신경망으로 돌아간다. [`SearchHandoff`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/heuristics/handoff.py#L16-L65)

그래서 “우승팀은 마지막 날 수확만 최적화해서 전체 운영을 배웠다”는 해석은 틀리다. 초반 투자와 중반 생산의 교재를 만든 탐색은 전체 시즌에 사용됐다. “최종 A가 매일 큰 C++ 탐색으로 농장을 운영했다”는 해석도 맞지 않는다.

Agent B에는 또 다른 실행 개선이 있다. 여러 일손이 씨앗 하나를 동시에 쓰려 하거나 같은 칸에 같은 작업을 중복 선택하지 않도록, 앞서 고른 action이 소비한 자원을 다음 일손의 mask에 반영한다. rollout과 확률 재계산, 제출 추론에서 이 순차 제약이 맞아야 한다. action을 뽑은 분포와 학습에 사용하는 분포가 다르면 PPO 확률비부터 틀어진다. [`순차 행동 선택`](https://github.com/msdsm/kaggriculture-1st-solution/blob/84057a0fda4238ccdebc46f9bf5496c6c4b2e00d/python/kaggriculture/actions/sequential.py#L35-L93)

가령 씨앗이 하나이고 서로 다른 빈칸의 일손 둘이 `심기/대기` 중 하나를 고른다고 하자. 첫 일손은 `0.6/0.4`, 둘째는 mask 적용 전 `0.8/0.2`다. 첫 일손이 심었다면 둘째의 심기를 막고 남은 대기의 확률을 1로 다시 정규화한다.

| 첫 일손 | 남은 씨앗 | 둘째의 조건부 확률 | 두 선택의 결합 확률 |
| --- | ---: | --- | ---: |
| 심기 | 0 | 심기 0, 대기 1 | 심기·대기 $0.6\times1=0.6$ |
| 대기 | 1 | 심기 0.8, 대기 0.2 | 대기·심기 $0.4\times0.8=0.32$ |
| 대기 | 1 | 심기 0.8, 대기 0.2 | 대기·대기 $0.4\times0.2=0.08$ |

이 세 경우의 확률은 합계 1이다. mask 없이 독립 선택하면 둘 다 심는 불가능한 경우에 $0.6\times0.8=0.48$을 주게 된다.

PPO에 저장할 첫 경우의 log probability는 `log(0.6) + log(1)`이다. 학습 때 둘째의 mask를 잊고 `log(0.6) + log(0.2)`로 계산하면, 가중치를 전혀 바꾸지 않았는데도 새/옛 확률비가 1 대신 0.2가 된다. 동작이 같다는 검사만으로 부족한 이유다. **샘플링한 action, 그때의 유효 mask, 저장 확률, 학습 중 재계산 확률이 같은 분포를 가리켜야 한다.** 이렇게 확률을 일치시키기 위해 일손마다 본체 신경망을 다시 계산할 필요는 없다. 이 코드는 한 번 계산한 logits에서 앞선 선택에 따른 유효 action을 차례로 제한한다.

명령이 서로 충돌하거나 실패하면 학습한 성능이 나오지 않는다. 어떤 전략을 배웠는지뿐 아니라, 그 선택이 의도대로 실행되는지도 확인해야 한다.

## 14. 실패를 해결하는 데 필요한 전략 변경 범위

우리도 기존 전략을 바꾸려 했고, 기록에서 작은 개선을 찾았으며, 여러 대안 중 더 유리한 것을 고르는 학습을 구현했다. 부족했던 것은 바꿔야 한다는 생각이 아니었다. **프로그램이 시도할 수 있는 변경의 범위가 좁았고, 그 범위를 넓힐 생성기와 학습 과정이 충분히 연결되지 않았다.**

준비된 경로를 고르는 모델은 어느 경로를 선택할지 배운다. 경로에 없는 초반 투자와 새 작업 배치까지 만들지는 못한다. 비료나 씨앗 구매를 한 번 바꾸는 보정 규칙도 해당 상황을 개선할 수는 있지만, 일손·토지·작물 전환을 함께 다시 계획하지는 못한다.

BC·PPO 모델은 매 turn의 action을 바꿀 수 있고, planner는 인력 수와 작업 구성을 함께 바꿀 수 있다. 어느 알고리즘이 무조건 우월한지보다 **지금의 실패를 해결하려면 어떤 결정을 바꿔야 하고, 내 모델이 그 결정을 바꿀 수 있는지**를 물어야 한다.

다음 대회에서 “토마토 운영이 약하다”는 진단을 받았다면 곧바로 모델을 키우거나 GPU를 더 빌리기 전에 다음을 나눠야 한다.

- 이미 좋은 토마토 운영이 있는데 선택하지 못한다면 선택 문제다.
- teacher는 잘 운영하지만 BC policy의 작업·주문이 실패한다면 표현·학습·실행을 확인한다.
- policy가 직접 플레이하다 teacher 데이터에 없던 상황에서 무너진다면, 학습 데이터의 범위와 새 rollout을 확인한다.
- 보유한 후보와 teacher 어느 쪽에도 좋은 운영이 없다면 새로운 teacher·탐색·전략 생성 문제다.

각 문제에 쓰는 계산이 다르다. 첫 문제에 좋은 선택기를 만들 수 있어도 네 번째 문제의 메뉴는 생기지 않는다. 네 번째 문제를 풀 좋은 탐색기가 있어도 실행 오류는 저절로 없어지지 않는다.

## 15. BC·PPO·탐색의 계산 자원 배분

단일 5090 사례는 환경까지 JAX로 옮겨 GPU에서 여러 세계를 진행했다. 우승팀 공개 경로는 Rust batch 환경과 JAX의 policy 추론·학습을 연결한다. C++ 계획 탐색은 CPU에서 돌아간다. 장비보다 **어떤 계산을 병렬화할지**가 먼저 정해져 있다.

BC에서는 많은 observation에 대해 teacher action의 확률과 gradient를 계산한다. PPO에서는 여러 세계에서 policy를 실행해 데이터를 모으고, 그 데이터의 gradient를 계산한다. planner는 작업과 경로로 표현한 후보의 예상 수입·비용을 계산한다. 완전 경기 벤치마크는 이렇게 얻은 결과물이 실제로 강한지 확인한다.

우승팀 A의 채택된 조상 경로에는 저자 기록 기준 약 829만 PPO 경기가 있다. 버린 실험과 teacher 자료 생성, critic 적합, 평가까지 포함한 총작업량은 아니다. 이 대규모 경로를 단일 5090 사례의 비용으로 소개할 수는 없다. 반대로 우승팀만큼의 장비가 없으면 BC→PPO를 시도할 수 없다는 결론도 단일 GPU 사례와 맞지 않는다. 필요한 것은 가용 자원 안에서 **하나의 완전한 개선 루프를 검증하는 것**이다.

처음에는 작은 과정부터 연결할 수 있다. teacher 기록 한 경기에서 observation–action 표본을 만들고, 작은 BC policy가 끝까지 농장을 운영하는지 확인한다. 그 policy로 action 확률과 value를 포함한 rollout을 모으고, PPO 한 번이 실제 선택을 어떻게 바꿨는지 검사한다. 새로운 운영이 필요하면 planner가 그 계획을 표현하고 수정할 수 있는지부터 확인한다. 이 과정이 작동한 뒤에 처리량을 높인다.

벤치마크는 후보를 비교한다. 더 강한 후보를 만들려면 강자의 기록으로 policy를 학습하거나, 자기 rollout으로 그 policy를 개선하거나, 새로운 운영 계획을 탐색해야 한다. 무엇을 바꿀 수 있는 프로그램을 만들고 있는지가 다음 계산의 출발점이다.

---

[← 1편: 벤치마크에서 전략 생성으로](/posts/Kaggriculture-RL-1-From-Benchmarks-to-Strategy-KR/) · [3편: 로그·검증·GPU를 잇는 다음 대회 workflow →](/posts/Kaggriculture-RL-3-Next-Competition-Workflow-KR/)
