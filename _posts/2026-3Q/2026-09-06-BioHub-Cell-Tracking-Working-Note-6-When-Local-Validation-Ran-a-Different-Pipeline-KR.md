---
title: "BioHub Cell Tracking 작업 기록 6: 로컬 검증이 제출 파이프라인과 달랐던 문제"
date: 2026-09-06 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, universe-mismatch, transfer-ratio, detector-augmentation, leakage, oof, working-note, korean]
math: true
pin: false
hide: false
published: false  # keep unpublished until the competition closes (2026-09-29 23:59 UTC)
image:
  path: /assets/img/posts/2026-09-06-biohub-working-note-6/cover.png
  alt: "BioHub 작업 기록 6 표지: 로컬 검증이 제출 파이프라인과 달랐던 문제"
---

<style>
.content .table-wrapper > table {
  table-layout: fixed;
  width: 100%;
  min-width: 36rem;
}
.content .table-wrapper > table th,
.content .table-wrapper > table td {
  white-space: normal;
  overflow-wrap: break-word;
  vertical-align: top;
}
</style>

# BioHub Cell Tracking 작업 기록 6: 로컬 검증이 제출 파이프라인과 달랐던 문제

- 대회: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- 공식 평가지표: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- 이전 글:
  - [작업 기록 1: Lineage Graph 학습과 평가지표에 맞춘 후처리]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)
  - [작업 기록 2: Public 점수가 멈췄을 때 — OOF 기반 오류 분석]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/)
  - [작업 기록 3: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
  - [작업 기록 4: Local Gain이 Public Board에서 보이지 않았던 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
  - [작업 기록 5: 한 칸씩 쌓아 올린 방식이 Local Optimum에 갇힌 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
- 영문판: [BioHub Cell Tracking Working Note 6: When Local Validation Ran a Different Pipeline]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline/)
- 후속 글: [BioHub Cell Tracking 작업 기록 7: 로직으로 판단하려면 로컬 검증이 갖춰야 할 것]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce-KR/)

관련 공개 노트북:

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

> **이 시리즈에 대해.** Kaggle 대회 BioHub Cell Tracking During Development를 진행하며 쓴 작업 기록이다.
> 제브라피시 배아의 3D 타임랩스 현미경 영상에서 세포 계통(lineage)을 복원하는 대회이고, 학습 데이터는 배아 두 개에서
> 나온 영상 199편이다. Public 리더보드는 처음 보는 배아로 구성된 hidden test의 29%만 보여 주기 때문에, 제출을
> 거듭하며 이 점수를 쫓으면 과최적화를 피할 수 없다. 그래서 판단은 미리 설명할 수 있는 로직에 두고, 그 로직은 로컬
> 검증으로 확인했다. 이 시리즈는 그 로컬 검증을 믿을 수 있게 만든 과정과, 탐색 방식 자체가 부족했던 지점, 그리고
> 최종 제출을 고른 과정을 기록한다.
{: .prompt-info }

5편은 대회가 한 달 남은 시점에 남은 시간을 분열 채널에 쓰기로 하면서 끝났다. 손으로 조정한 게이트 대신, 세포가 둘로 갈라지는 지점(lineage graph의 분기, fork)을 학습된 verifier가 정하게 했다. 분열 항은 점수의 10분의 1이다.
이 verifier는 로컬 리플레이에서 OOF로 $+0.0053$을 얻었고, v79로 제출해 sanity check를 했다. sanity check는 학습용 배아 두 개에서 고른 이득이 처음 보는 배아에서도 유지되는지 확인하는 Public 제출이다.

08-29에 나온 v79의 Public 점수는 $0.935$로, 직전 제출본의 $0.921$보다 $+0.014$ 높았다. $0.002$ 이내를 동점으로 보는 리더보드에서 분명한 차이이고, 로컬 이득의 약 $2.6$배다.
이 글(2026-08-29 ~ 2026-09-04)은 로컬 검증이 왜 분열 단계를 hidden test보다 훨씬 작게 봤는지를 묻는다.
제출 두 번으로 잰 hidden 분열 항은 로컬 수준의 약 다섯 배였다. 이어서 제출 파이프라인을 리플레이해 보니, leak 없이 배아 단위로 나뉜 로컬 OOF 리플레이는 다른 파이프라인을 리플레이하고 있었고 같은 영상과 같은 scorer에서 $0.149$ 낮았다.
검증을 실제 제출 파이프라인 위에 다시 만들었고, 거기서 다시 학습한 verifier를 v83으로 냈다.

핵심은 다음과 같다.

```text
v79의 Public 상승 폭은 로컬 이득의 2.6배였다. 로컬 검증이 무언가를 놓치고 있었다.
첫 해석은 verifier의 데이터 부족이었고, 네 가지로 시험했지만 달라진 것이 없었다.
제출 두 번으로 잰 hidden 분열 Jaccard는 0.32 부근으로 로컬 0.062의 다섯 배였다.
로컬 검증은 다른 파이프라인을 리플레이하고 있었고, 같은 영상에서 0.149 낮았다.
검증을 제출 파이프라인 위에 다시 만들고 verifier를 다시 학습했다. v83은 새 base로 삼는 구간에 들어왔다.
예전 개선책 세 가지와 detector 보완 하나는 새 검증에서 다시 재 보고 중단했다.
```

| 절 | 질문 |
|---|---|
| 0 | Public이 크게 오른 것이 왜 로컬 검증의 문제였나? |
| 1 | 첫 해석: verifier의 데이터가 부족했나? detector에서 더 얻을 것이 있었나? |
| 2 | hidden test에서 분열 항은 얼마인가? |
| 3 | hidden test의 분열 수준은 왜 로컬의 다섯 배였나? |
| 4--5 | 제출 파이프라인 위에서 검증을 다시 만들자 무엇이 달라졌나? |
| 6 | 예전 개선책 중 새 검증에서도 유지된 것이 있었나? |
| 7 | 새 검증으로 detector tail을 보면 어떤가? |
| 8--9 | 무엇을 결정했고, 무엇이 확인됐나? |

---

## 0. 로컬 검증과 맞지 않았던 sanity check

제출한 Kaggle 노트북(커널)은 seed가 다른 detector 두 개의 field를 섞어 세포 노드를 만들고, transformer로 프레임 사이 링크에 점수를 매기고, ILP(integer linear program)로 그래프를 고른 뒤 후처리 단계를 거친다.
가장 최근에 붙인 후처리 단계가 verifier다. 함께 제출한 후보 테이블로 노트북 안에서 다시 학습하는 gradient boosting 모델이고, 어떤 분기를 추가할지 정한다.

대회가 제공하는 라벨 달린 예시 영상 네 편은 학습 영상의 사본이다. 그래서 전체 학습 데이터로 학습한 배포 모델을 이 영상으로 채점하면 hold-in 확인이 된다. 망가진 것은 잡아내지만 후보를 고르는 데는 쓸 수 없다.
선택은 embryo-out 리플레이로 한다. 학습 영상 199편은 배아 두 개(71편, 128편)에서 나왔고, 두 fold가 각각 한 배아를 다른 배아로 학습한 모델로 채점한다. 변경은 두 배아 모두에서 음수가 아니어야 한다.

| 2026-08-29 시작 시점의 항목 | 값 |
|---|---:|
| comparator 리플레이(5편의 OOF comparator), 영상 199편, 이전 분열 단계를 그대로 둔 상태 | $0.6014$ |
| 그 comparator에서 잰 배포 verifier의 로컬 이득 | $+0.0053$ |
| 첫 verifier 제출(v79)의 Public 점수 | $0.935$ |
| 직전 제출본의 Public 점수 | $0.921$ |

이 표에는 이상한 점이 두 가지 있었다. 하나는 약 $2.6\times$라는 비율이다. 그동안 간선 채널 변경은 Public에서 로컬의 $0.14\times$에서 $0.9\times$ 정도로 줄어든다고 인용해 왔는데, 이 범위를 뒷받침하는 실험별 표는 없었다. 다른 하나는 로컬 base $0.60$ 부근과 Public $0.94$ 부근의 차이다. 몇 주 동안 모집단 차이로 설명했지만 확인한 적은 없었다.
두 가지 모두 로컬 OOF 그래프가 조금 비관적이어도 노트북이 만드는 그래프를 충실히 대신한다는 가정에 기대고 있었다. 이 가정이 틀렸다면 최근의 판단은 모두 제출하는 것과 다른 대상에서 확인됐다.

---

## 1. 첫 해석: verifier의 데이터가 부족하다

$0.935$에 대한 첫 해석은 verifier가 통하니 더 밀어붙이자는 것이었다.
verifier는 약 $80$개의 positive로 $+0.0557$인 oracle 상한의 10분의 1 정도를 잡고 있었고, 대회 전체에서 annotation이 달린 분열 이벤트는 $151$개였다. 그래서 첫 시도들은 데이터 부족을 겨냥했고, detector도 함께 시험했다.
여기 나오는 verifier 수치는 comparator 리플레이에서 잰 것이다. 3절에서 이 리플레이를 기준에서 뺀 뒤에도 부호는 유효하지만, 절대 수준은 쓸 수 없다.

### 1.1 verifier에 대한 네 가지 시도

| 시도 | 겨냥한 병목 | 결과 |
|---|---|---|
| 후보 생성기의 범위 넓히기 | recall: $151$개 중 $75$개에만 도달 | positive $80 \to 110$, 후보 pool은 거의 두 배; 적용 delta $+0.0053 \to +0.0021$ |
| 패치 세 장의 appearance를 보는 CNN, 배아가 섞이지 않게 학습 | feature가 기하와 트랙 정보뿐 | training loss $1.42 \to 0.075$; 두 fold 모두 held-out 실제 분열의 중앙값이 negative의 99번째 백분위수보다 낮음 |
| 공개 합성 데이터셋으로 pretrain한 ranker | positive가 너무 적음 | positive $163{,}422$개, holdout AUC $0.9988$; feature로 넣으면 소수 넷째 자리까지 같은 점수 |
| 공개 zebrafish 데이터셋의 실제 분열 $23{,}977$개로 학습한 CNN | 도메인이 다름 | 배아가 바뀌어도 일반화됨; 추가 이득은 $10^{-4}$ 미만 |

후보 pool을 넓히자 뒤 단계로 들어가는 분포가 바뀌었고, 한 pool에 맞춰 보정한 operating point가 경계의 더 나쁜 후보까지 통과시켰다.
CNN은 학습 fold당 positive가 $61$개와 $19$개뿐이어서 외울 수밖에 없었다.
합성 데이터 ranker는 자기 도메인은 풀었지만, 실제 ranker의 feature 공간에서는 진짜 분열과 가짜 분기가 구분되지 않았다.
positive 부족과 도메인 차이는 어느 쪽을 고쳐도 점수가 움직이지 않았으므로 병목이 아니었다.

### 1.2 detector에 대한 두 가지 시도

detector 쪽에서는 epoch만 늘리는 것과, 주최 측이 명시적으로 허용한 dense 외부 데이터셋을 시험했다.
레시피를 그대로 두고 epoch $200$에서 $400$까지 늘리자 embryo-out 검증 점수는 $0.8151$에서 epoch $218$에 $0.8326$까지 올랐다가 $0.7907$로 떨어졌다.
epoch $400$의 평가 결과가 epoch $200$과 바이트 단위까지 같아서 best checkpoint 저장의 resume 버그가 드러났고, 그래서 외부 데이터 fine-tuning에서는 epoch마다 같은 checkpoint에서 resume한 대조군을 짝지었다.
미리 적어 둔 게이트 $+0.005$에 대해 결과는 $+0.0466$으로, 프로젝트에서 나온 검증 이득 가운데 가장 컸다.
trainer의 검증 수치는 그래프가 아니라 detector를 설명한다. 그래서 fine-tuning 모델을 primary detector로 넣고, embryo-out 예시 영상 두 편에서 배포 파이프라인을 끝까지 돌렸다.

| 조건 (영상 2편 embryo-out 파이프라인 harness) | 점수 | node recall | 예측 노드 수, dense 영상 |
|---|---:|---:|---:|
| 대조군 쌍 | $0.8028$ | $0.9662$ | $64{,}477$ |
| fine-tuning 모델을 primary로 | $0.7756$ | $0.9625$ | $66{,}122$ |

end-to-end 지표는 $-0.0272$ 움직였다. fine-tuning 모델은 객체를 더 많이 찾았지만 위치를 조금 더 부정확하게 잡았고, 노드 매칭(반경 $7\,\mu\mathrm{m}$ 안의 최적 할당)과 ILP의 링크 비용은 둘 다 peak 위치에 따라 달라진다.
두 번째 field 빼기, threshold 옮기기, 두 detector의 역할 바꾸기는 모두 손해였다. single-detector base를 가정했을 때 약 $+0.03$으로 보이던 조건도 그 base를 실제로 돌려 보니 $-0.0082$였다.
영상 두 편의 결과로 이 fine-tuning 모델을 배포용 detector로 쓰는 안은 중단했다. 배포 파이프라인 밖에서 계산한 검증 지표는 파이프라인의 점수가 아니다.

---

## 2. hidden test의 분열 항을 재다

### 2.1 리더보드만 답할 수 있었던 이유

앞의 시도들은 hidden test에서 분열 단계가 얼마인지 답하지 못했다.
예시 영상 네 편에서 배포된 분열 단계는 그래프를 $45$번 수정했지만 간선 점수 변화는 정확히 0이었다. 수정 $45$개가 모두 annotation이 듬성듬성 달린 영역 밖에 있었다. 이런 단계는 로컬 채점으로는 보이지 않고 hidden test에서만 채점된다.

### 2.2 분해 제출

분해 제출(decomposition probe)은 정확히 한 단계만 다른 제출 두 개다. 두 점수의 차이가 로컬 대용 지표를 거치지 않고 hidden test에서 그 단계의 값을 매긴다.
2026-09-01에 제출 두 번을 여기에 썼다. v80은 배포 노트북에서 분열 단계의 operating point만 sweep으로 고른 값으로 바꿨다. v81은 첫 verifier 제출의 노트북(모델과 테이블 포함)에서 verifier의 *apply* 단계(고른 분기를 그래프에 써 넣는 함수)만 입력을 그대로 돌려주는 함수로 바꿨다.
모델은 그대로 학습되므로 뒤 단계는 모두 같은 입력을 받고 수정만 빠진다. *apply*를 무력화하면 v79와 v80의 설정 상수는 아무 역할도 하지 않으므로, v81 하나를 둘 모두와 비교할 수 있다.

| arm (Public split) | 내용 | Public |
|---|---|---:|
| v79 | 첫 verifier 제출 | $0.935$ |
| v80 | sweep으로 고른 operating point, verifier 작동 | $0.937$ |
| v81 | v79 노트북에서 apply만 무력화 | $0.903$ |

두 점수를 빼면 $\Delta J_{\mathrm{edge}}^{\mathrm{adjusted}} + 0.1\,\Delta J_{\mathrm{division}}$이 v79 기준으로 $+0.032$, v80 기준으로 $+0.034$다.
분열 가중치 $0.1$로 나누면 hidden 분열 Jaccard는 $0.32$에서 $0.34$ 부근이고, 같은 단계의 로컬 OOF 분열 Jaccard는 $0.062$였다.

v81의 그래프에는 분기가 하나도 없었으므로, 이 차이는 v79에서의 hidden 분열 항 전체다. verifier가 이전 분열 단계보다 더 얻은 이득은 v79의 $+0.014$였다.
이 차이에는 분열 단계가 간선 쪽에 남기는 작은 흔적(예시 영상 네 편에서 false positive 하나)도 섞여 있고, 리더보드의 반올림 때문에 $+0.032$는 실제로 $+0.032 \pm 0.001$이다. hidden 분열 Jaccard로는 약 $0.31$에서 $0.33$ 사이로, 여전히 로컬 수준의 다섯 배 정도다.

sweep으로 고른 v80의 로컬 이득 $+0.0007$은 두 배아 모두에서 음수가 아니었지만 단독 변경에 요구하는 $+0.001$에 못 미쳤고, $0.937$과 $0.935$는 리더보드 해상도에서 같은 점수다.
측정 결과는 로컬 검증 쪽을 가리켰고, 이 방법이 C13이 됐다. hidden test에서만 보이는 항은 설계한 분해 제출로 잰다.

---

## 3. 단서: 로컬 검증은 다른 파이프라인을 리플레이하고 있었다

### 3.1 다섯 배 차이에 대한 두 가지 해석

다섯 배 차이는 두 가지로 해석할 수 있었다. hidden test에 채점되는 분열이 더 많거나, 로컬 검증이 다른 파이프라인을 재고 있었거나.
직접 확인할 수 있는 것은 두 번째뿐이었고, 맞다면 프로젝트의 모든 로컬 수치가 영향을 받는다.

여기서 학습하는 구성 요소는 모두, 어떤 파이프라인을 학습 영상에 돌려 만든 후보 테이블로 학습한 selector다. 한 파이프라인이 만드는 후보 그래프 전체를 그 파이프라인의 *환경*(universe)이라고 부르겠다.
문제가 되는 경우는 selector $\phi^{*}$를 로컬 검증이 리플레이하는 파이프라인의 후보 $\mathcal{C}(\mathcal{U}_{\mathrm{fit}})$로 튜닝하고, 노트북이 실제로 돌리는 파이프라인의 후보 $\mathcal{C}(\mathcal{U}_{\mathrm{run}})$에 적용하는데, $\mathcal{U}_{\mathrm{fit}} \ne \mathcal{U}_{\mathrm{run}}$인 경우다.
이것은 leak이 아니다. 두 파이프라인 모두 fold마다 배아가 겹치지 않게 깨끗하게 리플레이할 수 있으므로, 어느 한쪽 안에서는 이 차이가 보이지 않는다.

### 3.2 제출 파이프라인 리플레이

제출 파이프라인 리플레이(deployed-stack replay)는 노트북의 파이프라인을 학습 영상 199편 전체에 그대로 돌린다. checkpoint를 배아 단위로 나눠 두어 각 영상은 그 배아를 본 적 없는 모델이 처리하고, 결과는 공식 scorer로 채점한다.
probe 결과가 나오자 2026-09-02에 이것부터 돌렸다.

| 리플레이한 파이프라인 (영상 199편, embryo-out, 공식 scorer) | 점수 | 최종 그래프의 node recall |
|---|---:|---:|
| comparator 리플레이, 분기 제거: 로컬 검증이 리플레이하던 파이프라인으로, 최근의 로컬 선택은 모두 여기서 이루어짐 | $0.6005$ | $0.8870$ |
| 제출 파이프라인(deployed stack): 실제로 제출한 파이프라인, 노트북 자체의 출력, 같은 영상, 같은 scorer | $0.7499$ | $0.9255$ |

![같은 영상 199편에서 잰 두 점수: comparator 리플레이 0.6005와 제출 파이프라인 0.7499, 차이 0.149]({{ site.baseurl }}/assets/img/posts/2026-09-06-biohub-working-note-6/fig-01-two-instruments.png)
_그림 1. 같은 학습 영상 199편을 같은 scorer로 잰 두 측정 도구. 8월의 선택은 점수가 더 낮게 나오는 comparator 리플레이에서 이루어졌고, 노트북이 실제로 돌리는 것은 제출 파이프라인이다._

지난 몇 주 동안의 verifier threshold, 삭제 규칙, association 설정은 모두 recall이 더 낮은 comparator 리플레이의 그래프에서 골랐다.
로컬 base $0.60$ 부근과 Public $0.94$ 부근 사이의 오래된 차이도 일부는 처음부터 모집단 차이가 아니었다. 나머지는 이번 주에 확인하지 않았다.

이 리플레이로 3편부터 동시에 쓰이던 기준선 일곱 개도 정리됐다. 0절의 $0.6014$는 이전 분기를 남겨 둔 같은 comparator이고, 4편의 research 리플레이(base $0.74$ 부근)는 다른 그래프 계열이다. 이제부터 절대 점수는 제출 파이프라인 리플레이를 기준으로 쓴다.

여기서 노트북이 원래 만든 분기를 모두 접은 base는 $0.7489$이고, 아래의 분열 수치는 모두 이 base 위에서 잰 것이다.
제출 파이프라인이 만드는 후보에 label oracle을 적용하면 점수는 $0.8007$, 분열 Jaccard는 $0.5097$이다(TP $79$개, FP $4$개, 놓친 분열 $72$개).
놓친 $72$개에는 후보 행이 아예 없다. ranking을 따지기 전에 분열 상한의 절반 정도가 후보 생성기 밖에 있다. 원인은 이번 주에 재지 않았다.

### 3.3 리플레이로도 설명되지 않은 것

제출 파이프라인의 그래프에서 배포된 verifier는 TP $2$개, FP $3$개를 냈다. 로컬 분열 Jaccard로 $0.013$ 부근이고, comparator 리플레이의 $0.062$보다 낮다.
리플레이하는 파이프라인을 바로잡자 hidden $0.32$와의 거리가 오히려 벌어졌으므로, 모집단 차이라는 해석은 열려 있다.
대신 고칠 수 있는 문제가 하나 드러났다. verifier는 예전 파이프라인의 후보로 학습됐는데, 제출 파이프라인의 후보에 적용되고 있었다.
여기서 C12가 나온다. 로컬 검증은 실제로 제출하는 파이프라인에서 한다.

---

## 4. 제출 파이프라인 위에서 다시 학습한 verifier와 v83

학습 대상이 어긋났다면, 제출 파이프라인 자체의 후보로 학습한 verifier가 예전 후보로 학습한 verifier보다 그 파이프라인의 그래프에서 더 잘해야 한다.
시험에서는 그 그래프를 고정하고 verifier를 학습하는 후보만 바꿨다.

| 학습 방식 (제출 파이프라인 리플레이, 영상 199편 OOF, embryo-disjoint, base $0.7489$) | base 대비 delta | 분열 TP / FP |
|---|---:|---:|
| 배포된 verifier, comparator 리플레이 후보로 학습 | $+0.0013$ | $2 / 3$ |
| 제출 파이프라인 후보로만 refit | $+0.0030$ | $5 / 13$ |
| 두 테이블의 합집합으로 refit | $+0.0065$ | $17 / 90$ |

예측대로였다.
배포된 selector는 얻을 수 있는 것의 5분의 1만 잡았고, 큰 배아에서는 영상 $128$편 전체에서 분기를 하나도 만들지 않았다. 학습도 제대로 됐고 배아 단위로 분리돼 있었지만, 파이프라인이 더 이상 만들지 않는 분포를 겨냥하고 있었다.

합집합 refit을 골랐다. sweep 결과가 plateau였고, peak에서 두 배아 모두 양수였다($+0.0096$, $+0.0060$).
런타임으로 옮기면서 배아별 threshold를, 미리 정한 규칙으로 고른 flat threshold 하나로 바꿨다.
커널은 예시 영상 네 편의 hold-in 게이트를 통과했고, 분열만 바꾸는 변경답게 간선 채널은 움직이지 않았다(간선 TP/FP/놓친 수 $2028/155/99$, 이전 $2027/155/100$).

v83은 refit의 sanity check였고, 결과를 어떻게 읽을지 제출 전에 적어 두었다.

```text
>= 0.940       refit 효과가 유지된다. 새 base로 삼는다
0.937 - 0.939  중립. refit 효과를 구분할 수 없다
< 0.937        flat threshold가 손해였다. 배아별 threshold로 되돌린다
```

결과는 $0.944$로, 구간의 기준인 v80의 $0.937$보다 $+0.007$ 높았고 refit을 base로 삼는 구간 안이었다. 문제가 발견되지 않았으므로 refit이 base가 됐다.

로컬 $+0.0065$ 대비 Public $+0.007$(v80 대비)이면 비율은 약 $1.1\times$이고, 5절에서 바로잡은 $+0.0046$을 쓰면 $1.5\times$에 가깝다. 앞의 $2.6\times$는 comparator 리플레이에서 나온 값이고, 어느 비율이든 분자가 반올림된 점 하나다.
이 비율들에서는 부호만 가져와 실험 순서를 정하는 데 썼다. 분열 변경의 효과는 hidden test에서 줄어들지 않았다.

---

## 5. 제출 코드를 그대로 검증 도구로 쓰다

2026-09-03에는 같은 질문을 한 단계 아래에서 던졌다. 커널 안의 제출 런타임 모듈과 로컬에서 돌린 같은 분열 단계 레시피를 비교하는 parity check였다. 예시 영상 네 편에서 커널은 분열을 각각 $50$, $23$, $3$, $50$개 적용했고, 로컬 레시피는 하나도 적용하지 않았다.

원인은 flag 하나였다. 라벨은 annotation 영역에서만 나오므로, 로컬에서 분열 테이블을 다시 만들 때마다 annotation이 달린 source 주변에서만 후보를 생성했다.
예시 영상 한 편에서 이렇게 만들면 행이 $7{,}519$개다. annotation 위치를 모르는 제출 런타임은 $358{,}000$개를 만든다.
따라서 이 단계의 threshold와 budget을 로컬에서 sweep할 때마다 커널이 떠안는 false positive 부담이 빠져 있었고, 로컬 최적점은 모두 느슨한 쪽으로 치우쳤다.

해결은 C12를 따랐다. 제출 런타임 모듈 자체를 선택 도구로 쓰고, 다시 만든 테이블은 학습 데이터로만 쓴다.
이렇게 다시 돌리자 배포된 operating point가 이미 최적이었다. 분기가 없는 base $0.7489$ 대비 $0.7535$로 $+0.0046$이었고, 두 배아 모두 양수였다. threshold나 cap을 어떻게 바꿔도 새 operating point에 대해 미리 적어 둔 규칙을 넘지 못했다.

같은 sweep에서 평가지표의 구조적 성질이 하나 드러났다. 예측한 분열의 부모 위치에서 매칭 거리 안에 정답 노드가 없으면, 그 분열은 TP도 FP도 아니다.
전체 학습 데이터로 학습한 in-sample 실행은 분열을 $4{,}662$개 적용했지만 FP는 $71$개, TP는 $42$개였다. embryo-out 조건에서는 영상당 cap을 네 배 범위로 바꿔도 점수가 $0.7536$, $0.7535$, $0.7535$였다. budget을 늘리면 추가로 고른 분열은 평가지표가 아무것도 평가하지 않는 곳에 떨어진다.
손실은 scorer가 보는 annotation 달린 부모에서 ranking을 잘못 매기는 데서 생기고, 여기서 도달 가능한 분열 $79$개 중 $8$개만 잡았다. 로컬에서 이 채널의 한계는 ranking이다. hidden test가 false positive에 매기는 비용은 이번 주에 재지 않았다.

---

## 6. 예전 개선책 세 가지를 제출 파이프라인에서 다시 재다

### 6.1 joint lineage action과 fold 구성

부모와 자식에 대한 결정을 한꺼번에 점수 매기는 조합(joint lineage action)은 배포하지 않은 이득 가운데 가장 자주 언급되던 것이다. 제출 파이프라인 리플레이에서 $+0.0187$이었고, 두 배아 모두 양수였다.
그런데 fold를 점검해 보니 head가 fold 네 개로 학습돼 있었고, fold마다 두 배아의 영상이 섞여 있었다. 4편에서 뜯어본 바로 그 결함이다.

| head 구성 (제출 파이프라인 리플레이, 영상 199편 OOF) | 전체 | 71편 배아 | 128편 배아 |
|---|---:|---:|---:|
| 두 배아가 섞인 fold | $+0.0187$ | $+0.0152$ | $+0.0191$ |
| 배아가 겹치지 않는 fold | $+0.0004$ | $+0.0150$ | $-0.0021$ |

128편 배아로 학습한 head는 다른 배아에서도 통하고, 71편 배아로 학습한 head는 통하지 않는다.
세 번째 배아가 없으니 영상 수가 적어서인지 일반화를 못 해서인지 구분할 수 없다.
두 배아 모두에서 음수가 아니어야 한다는 규칙에 따라 이 계열은 중단했다.

### 6.2 sub-voxel 좌표와 대회 규정

커널은 짧은 트랙 구간을 따라 노드 위치를 smoothing한 다음 정수 voxel 좌표로 저장한다.
제출 파이프라인 리플레이에서 smoothing한 위치 $475$만 개를 정수로 양자화하면 $-0.0050$ 손해이고, 두 배아 모두 손해였다($-0.0005$, $-0.0057$).
좌표를 소수 셋째 자리까지 쓰는 커널도 문제없이 돌았지만 제출하지 않았다. 대회 Evaluation 페이지가 centroid 좌표를 정수 voxel로 명시하므로, 이 손실은 규정을 지키는 제출이 모두 치르는 비용이다.

그 커널의 hold-in 확인은 반대 방향으로 움직였다($2028/155/99$ 대비 $2024/157/103$). in-sample detector는 peak를 annotation이 달린 voxel 위에 찍으므로 반올림하면 정답 위치로 다시 붙고, embryo-out detector는 노이즈가 커서 sub-voxel 위치에서 이득을 본다.
hold-in의 편향은 판단의 방향을 뒤집을 수 있다.

철회할 결과가 하나 있다. 2026-09-01에 interior line-fit smoothing을 대회 기간 중 가장 큰 간선 채널 로컬 이득으로 기록했고, 자체 hold-in 게이트에서 기각했다. 이틀 뒤 코드를 점검하다가 커널이 처음부터 같은 smoothing을 돌리고 있었다는 것을 발견했다. 09-01의 수치는 실행 환경에 따라 효과가 달라지는 개선책이 아니라, 같은 smoothing을 한 번 더 적용한 효과였다.

### 6.3 파이프라인이 바뀌자 부호가 바뀐 삭제 규칙

마지막으로 남은 가벼운 후처리 규칙은 연결된 간선의 확률이 모두 낮은 노드를 지운다.
comparator 리플레이에서는 $+0.0009$였지만 제출 파이프라인 리플레이에서는 같은 설정으로 $-0.0106$이었고, 가장 좋은 변형은 말 그대로 아무것도 하지 않는 설정이었다.
제출 파이프라인에서는 두 detector의 점수를 다른 방식으로 합쳐 간선 확률을 만들고, 여러 단계에서 확률이 정확히 0인 간선을 추가한다. 그래서 "연결된 간선의 확률이 모두 낮다"는 조건에 걸리는 노드는 어려운 영역의 실제 세포다.

---

## 7. 새 검증에서 다시 본 detector tail

### 7.1 남은 여지는 어디에 있나

제출 파이프라인 리플레이에서 영상별 adjusted 간선 Jaccard와 node recall의 상관은 두 배아에서 $0.82$, $0.79$다. 각 배아에서 가장 나쁜 decile의 영상만 그 배아의 중앙값까지 끌어올려도 $+0.0277$, $+0.0233$이 오른다. 영상의 약 $10\%$가 남은 구조적 여지를 사실상 전부 쥐고 있다.
가장 나쁜 영상 12편과 중앙값 근처 12편에서 놓친 세포의 비율은 각각 $36.5\%$와 $4.6\%$였다.

| 가장 나쁜 영상 12편, 3분위 | 낮음 / 중간 / 높음 |
|---|---|
| 정답 위치의 intensity 기준 놓친 비율 | $0.579$ / $0.324$ / $0.192$ |
| 주변 예측 노드 밀도 기준 놓친 비율 | $0.653$ / $0.272$ / $0.109$ |

놓친 세포는 장면이 성긴 곳에 몰려 있고, 밝기에 따라 정도가 갈린다.
가장 나쁜 12편에서는 놓친 세포의 $48\%$가 detector의 field에서 아예 보이지 않는다(정답 위치의 주변 최댓값 logit이 0 미만). 학습 단계의 성질이다. 중앙값 12편에서는 $62\%$가 threshold를 넘었지만 이웃 peak에 흡수됐다. 추론 단계의 성질이다.
다른 배아의 tail 영상은 detector 자체의 recall이 $0.987$이고, 세포를 그 뒤의 association과 solver 단계에서 잃는다.

### 7.2 처방과 그 효과

trainer에는 intensity augmentation이 없었고, 밝기에 따라 정도가 갈리는 실패는 domain shift의 모양이다.
augmentation을 넣었고(gamma, global gain, 부드러운 regional dimming field, additive haze floor, voxel별 노이즈), 진행 조건은 후보가 나오기 전에 적어 두었다.
epoch 10에서 큰 배아의 가장 나쁜 영상 7편의 raw node recall은 $0.7123$에서 $0.8884$로 올랐고, 놓친 세포 중 보이지 않는 비율은 $0.706$에서 $0.198$로 줄었다.
tail에서 추정 세포당 peak 수도 $1.701$에서 $2.10$으로 늘었다.

### 7.3 파이프라인 안에서 재기

이어서 이 detector를 배포 파이프라인에 넣고 공식 평가지표와 embryo-out으로 쟀다. 대상은 큰 배아의 영상 $15$편, 가장 나쁜 7편과 중앙값 근처 8편이다.

| 조합 (영상 15편 embryo-out 측정 subset) | delta, 가장 나쁜 7편 | delta, 중앙값 8편 | 음수가 나온 중앙값 영상 |
|---|---:|---:|---:|
| primary detector, 배포 threshold | $+0.1007$ | $-0.0198$ | $5/8$ |
| primary detector, 엄격한 threshold | $+0.0583$ | $-0.0169$ | $5/8$ |
| detection 전용 세 번째 field, 낮은 가중치 | $+0.0124$ | $-0.0073$ | $5/8$ |
| 새 peak union, 엄격한 threshold | $+0.0417$ | $-0.0306$ | $8/8$ |
| 빈 곳 채우기 union, 성긴 영역만 | $+0.0448$ | $-0.0175$ | $6/8$ |

![detector 조합 다섯 가지를 가장 나쁜 영상 7편에서의 이득과 중앙값 근처 영상 8편에서의 손실로 비교한 그림]({{ site.baseurl }}/assets/img/posts/2026-09-06-biohub-working-note-6/fig-02-tail-versus-typical.png)
_그림 2. 모든 조합이 가장 나쁜 영상 7편에서는 이득을, 중앙값 근처 8편에서는 손실을 냈다. tail 비중을 $0.10$ 부근으로 두고 가중하면 모든 조합이 전체로는 음수로 추정된다. 이 추정은 측정값이 아니다._

같은 모양을 보인 조건이 네 개 더 있었다. 모든 조건에서 augmentation을 적용한 첫 번째 split 모델과 배포된 두 번째 split 모델을 짝지었다(augmentation을 적용한 쌍은 학습하지 않았다).
가장 나쁜 영상에서 공식 adjusted 간선 Jaccard는 $0.248$에서 $0.487$로 올랐지만, 판정은 보통 영상에서 갈린다. tail 비중을 $\pi \approx 0.10$으로 두고 $\mathbb{E}[\Delta S] = \pi\,\Delta_{\mathrm{tail}} + (1-\pi)\,\Delta_{\mathrm{typical}}$를 계산하면 모든 조건이 약 $-0.005$에서 약 $-0.025$ 사이로 나온다. 측정한 subset이 각 층을 대표하고 hidden test의 배아에도 비슷한 tail이 있다는 가정 아래서다.

**추가한 객체가 개수 경계를 넘을 때.** 평가지표는 영상마다 간선 Jaccard에 $1-0.1\,r_i$를 곱한다. $r_i$는 함께 주어지는 대략적인 세포 수 추정치 대비 예측 노드의 상대 초과량이다. 추정치 바로 아래의 영상은 작은 보너스를, 바로 위의 영상은 페널티를 받는다.
추정 세포 수가 $5{,}257$개인 중앙값 영상 하나에서 빈 곳 채우기 조건은 노드 수를 $5{,}047$에서 $5{,}439$로 늘렸고, 점수는 $0.7735$에서 $0.7050$으로 떨어졌다. node recall은 $0.9915$에서 $0.9957$로 올랐을 뿐이다. 노드 $392$개를 더해 annotation 달린 세포 스무 개 정도를 더 얻었다.

**노드 수가 그대로여도 peak가 움직일 때.** 가장 약하게 섞은 조건에서도 한 영상은 노드가 $+0.2\%$ 늘었을 뿐인데 점수가 $0.767$에서 $0.722$로 떨어졌다. detection field에 무엇이든 섞으면 배포된 peak가 voxel 하나보다 짧은 거리만큼 움직이고, 경계에 걸린 $7\,\mu\mathrm{m}$ 매칭이 깨진다.
extraction threshold를 올리자 tail 이득은 절반으로 줄었고 중앙값 영상의 손실은 거의 그대로였다. 위치 정확도 손실이라면 예상되는 결과다.
2026-09-04의 대조 실험에서 평범한 세 번째 seed를 detection 전용 역할로 넣자 같은 중앙값 8편에서 $-0.0063$이었고($8$편 중 $6$편 음수), augmentation 모델의 $-0.0073$과 비슷했다. 이 조건에서 보통 영상의 손실은 대부분 augmentation이 아니라 세 번째 field를 섞은 데서 나온다.

### 7.4 실험을 중단한 이유

subgroup만 고치는 처방은 test 시점에 해당 영상을 그 처방으로 보내는 신호(router)가 있어야 쓸 수 있다.
살펴본 통계량 두 가지, 프레임 contrast와 새 peak 비율은 두 집단을 가르지 못했다. 더 넓은 router 탐색은 하지 않았고, 이 판정의 가장 약한 부분이다.
router가 없으면 측정한 조합은 모두 tail 이득을 더 큰 보통 영상 손실과 맞바꾸므로 이 실험은 중단했다. tail 진단, 6분이면 끝나는 embryo-out detector 평가, 그리고 이 모델 계열에서 intensity augmentation이 어두운 영역의 recall을 높인다는 사실은 남겨 두었다.

---

## 8. 판단 기록

적용하던 규칙은 학습 영상 199편에 대한 2-fold embryo-disjoint OOF 리플레이로, 두 배아 모두 음수가 아니어야 했다. 이번 주에 그 대상이 comparator 리플레이에서 제출 파이프라인 리플레이와 제출 런타임 모듈로 옮겨 갔다.
제출은 세 번 했고, 모두 결과를 읽는 방법을 미리 적어 두었다. v80과 v81은 로컬 검증으로는 볼 수 없는 항을 쟀고, v83은 로컬에서 고른 refit의 sanity check였다.

| 결정 | 당시의 이유 | 결과 | 바뀐 것 |
|---|---|---|---|
| 병목을 하나씩 정해 겨냥한 시도 여섯 가지 (1절) | verifier가 약 $80$개의 positive로 상한의 10분의 1만 잡고 있었음 | verifier 이득 없음; detector는 trainer 기준 $+0.0466$, 파이프라인 기준 $-0.0272$ | 데이터 양과 도메인은 병목이 아님; 여섯 가지 모두 중단 |
| v80/v81 분해 제출 (09-01) | 이 단계는 annotation 밖에서 작동해 hidden test에서만 채점됨 | hidden 분열 항 $+0.032$, Jaccard $0.32$ 부근 대 로컬 $0.062$; v80은 동점 | C13; 차이를 보면 문제는 로컬 측정 도구 쪽 |
| 제출 파이프라인 리플레이 (09-02) | 다섯 배 차이; 확인한 적 없는 $0.60$ 대 $0.94$ 차이 | 같은 영상, 같은 scorer에서 $0.7499$ 대 $0.6005$ | C12; 절대 점수의 기준을 하나로(C5) |
| 제출 파이프라인 후보로 refit, v83을 그 sanity check로 (09-03) | 다른 파이프라인의 후보로 학습한 배포 verifier가 TP $2$개, FP $3$개 | 로컬 $+0.0065$, 두 배아 모두 양수; Public $0.944$, "새 base로 삼는다" 구간 | refit이 base가 됨 |
| 제출 런타임을 선택 도구로 (09-03) | 커널은 분열을 수십 개 적용하는데 로컬 레시피는 하나도 적용하지 않음 | 배포된 operating point가 이미 최적, $+0.0046$; cap은 점수에 영향 없음 | 다시 만든 테이블은 학습 데이터로만 씀 |
| 예전 개선책 세 가지 다시 재기 | 셋 다 예전 측정에서 양수였음 | embryo-disjoint에서 $+0.0004$; 정수 좌표가 규정; $-0.0106$ | 셋 다 제출 없이 중단 |
| detector tail 실험 (09-04) | 영상의 약 $10\%$가 남은 여지를 쥐고 있음 | tail recall $0.7123 \to 0.8884$; 모든 조합이 중앙값 영상에서 손해 | router가 없어 중단; 분열 후보를 직접 라벨링하기 시작 |

### 이 기간이 끝났을 때의 선택 기준

| 조항 | 내용 | 도입 |
|---|---|---|
| C1 | 모든 그래프 수정은 OOF로 평가한다. 학습·보정·평가는 서로 겹치지 않는 영상에서 하고, 그래프 전체를 공식 평가지표로 채점한다 | 2편 |
| C2 | 게이트는 결과를 보기 전에 정해 둔다 | 3편(07-15) |
| C3 | 규칙은 실제로 적용될 모집단에서 보정한다 | 3편 |
| C4 | 구성 요소는 자체 정확도가 아니라, 파이프라인을 그대로 재현해 만든 그래프로 평가한다 | 3편 |
| C5 | 절대 점수는 같은 기준 환경 안에서만 비교하고, 환경이 다르면 변화량만 비교한다 — 이 편에서 정리(기준은 제출 파이프라인 리플레이) | 3편 |
| C6 | 후보는 hidden test에서 제한 시간 안에 실행을 마쳐야 한다 | 3편 |
| C7 | fold는 배아 단위로 나눈다(embryo-out) | 4편 |
| C8 | 제출 모델이 학습한 영상에서 잰 수치(hold-in)는 일반화의 근거로 쓰지 않는다 | 4편 |
| C9 | Public은 기대치를 미리 적어 둔 sanity check에만 쓰고, 인접한 설정 중 하나를 고르는 데는 쓰지 않는다 | 4편(08-10) |
| C10 | 한 탐색 공간 안에서 최적화하기 전에 그 공간의 상한부터 측정한다 | 5편 |
| C11 | 게이트는 결론을 낼 수 있어야 한다 | 5편 |
| C12 **(신규)** | 로컬 검증은 실제 제출 파이프라인을 그대로 재현해서 한다 | 6편 |
| C13 **(신규)** | hidden test에서만 드러나는 항은 분해 제출(decomposition probe)로 측정한다 | 6편 |

---

## 9. 이 기간에 확인된 것

### 확인된 사실

1. v81의 그래프에 분기가 없으므로 v79 − v81 차이 $+0.032$는 hidden 분열 항 전체이고, hidden 분열 Jaccard로는 $0.32$ 부근이다(로컬 $0.062$).
2. 같은 영상 199편과 같은 scorer에서 comparator 리플레이는 $0.6005$(node recall $0.8870$), 제출 파이프라인은 $0.7499$($0.9255$)였다.
3. 제출 파이프라인이 만든 같은 그래프에서 합집합 refit은 배포된 selector의 $+0.0013$보다 큰 $+0.0065$를 냈고, 제출 런타임으로는 $+0.0046$($0.7489$에서 $0.7535$)이었으며, 두 배아 모두 양수였다.
4. 부모에서 매칭 거리 안에 정답 노드가 없는 분열 예측은 TP도 FP도 아니고, 로컬에서 이 채널의 한계는 budget이 아니라 annotation 달린 부모에서의 ranking이며(도달 가능한 $79$개 중 $8$개), hidden test에서 false positive에 매겨지는 비용은 재지 않았다.
5. 제출 파이프라인에서는 annotation이 달린 분열 이벤트 $151$개 중 $72$개에 후보가 아예 없다.
6. 외부 데이터로 학습한 detector는 trainer 검증에서 $+0.0466$, 영상 두 편의 배포 파이프라인에서 $-0.0272$였다.
7. joint lineage action의 이득은 두 배아가 섞인 fold에서 $+0.0187$, 배아가 겹치지 않는 head에서 $+0.0004$다.
8. intensity augmentation은 큰 배아의 가장 나쁜 영상 7편에서 raw recall을 $0.7123$에서 $0.8884$로 올렸고, 파이프라인에서 잰 조합은 모두 보통 영상에서 손해였다.

### 근거는 있지만 아직 확정하지 못한 판단

1. 분열 변경의 효과가 hidden test에서 줄어들지 않는다는 판단(근거는 분자가 반올림된 점 몇 개이고, 분모는 서로 다른 로컬 자 두 개에서 나왔다).
2. 작은 배아로 학습한 head가 다른 배아에서 통하지 않는 이유가 배아가 어려워서가 아니라 영상 수가 적어서라는 판단.
3. test 시점에 tail 영상과 보통 영상을 가를 수 있는 영상 단위 통계량이 없다는 판단.
4. augmentation을 적용한 detector를 쌍으로 쓰더라도, augmentation을 적용한 첫 번째 split과 비슷하게 동작한다는 판단.

### 열린 질문

1. 리플레이하는 파이프라인을 바로잡자 로컬 수준은 오히려 낮아졌는데, hidden test의 분열 수준이 어떤 로컬 측정보다 다섯 배 정도 높은 이유는 무엇인가?
2. 새 라벨이 아닌 다른 신호로 annotation이 달린 부모에서의 ranking을 개선할 수 있을까?
3. peak 위치를 보존하도록 학습한 detector라면 조합할 때 생기는 손실을 피할 수 있을까?

---

## 마무리

이번 주에 잰 hidden 분열 항은 로컬 수준의 다섯 배 정도였다. 로컬 검증은 다른 파이프라인을 리플레이하고 있었고, 같은 영상과 같은 scorer에서 제출 파이프라인보다 $0.149$ 낮았다.
이제 검증은 실제로 제출하는 파이프라인에서 돌고, 거기서 다시 학습한 verifier를 v83으로 냈다.
이 리플레이에서 분열 채널의 한계는 annotation 달린 부모에서의 ranking이고, 상한의 절반 정도는 후보 생성기가 닿지 못한다. 합집합 refit 말고 시도한 ranking 신호는 association logit과 terminal-source 열거까지 모두 0이거나 손해였다.

2026-09-04부터 분열 후보에 직접 라벨을 붙이기 시작했다. 남은 ranking 신호는 새 라벨뿐이라고 봤다. 대회 규정에 대한 내 해석에 따른 작업이고, 주최 측의 확인은 받지 않았다. 첫 판단 $69$개로 제출 파이프라인 리플레이에서 제출 런타임의 점수가 $0.7535$에서 $0.7547$로 움직였지만, 읽기에는 너무 적다.
다음 글은 제출 파이프라인을 리플레이하는 것만으로 hidden test의 조건에서 아이디어를 판단할 수 있는지를 묻는다.

시리즈:

- [1편: Lineage Graph 학습과 평가지표에 맞춘 후처리]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)
- [2편: Public 점수가 멈췄을 때 — OOF 기반 오류 분석]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/)
- [3편: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- [4편: Local Gain이 Public Board에서 보이지 않았던 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
- [5편: 한 칸씩 쌓아 올린 방식이 Local Optimum에 갇힌 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
- **6편: 로컬 검증이 제출 파이프라인과 달랐던 문제**
