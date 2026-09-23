---
title: "BioHub Cell Tracking 작업 기록 7: 같은 코드로도 검증이 어긋나는 이유"
date: 2026-09-12 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, hand-labels, label-convention, pseudo-labels, logit-alignment, deployment-regime, leakage, oof, working-note, korean]
math: true
last_modified_at: 2026-09-23
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-09-12-biohub-working-note-7/cover.png
  alt: "BioHub Cell Tracking 작업 기록 7: 같은 코드로도 검증이 어긋나는 이유"
---

<style>
/* Local to the BioHub manuscripts; labels follow each table's own headings. */
.content .table-wrapper:has(> table.biohub-table) {
  max-width: 100%;
  overflow-x: auto;
  container: biohub / inline-size;
}
.content .table-wrapper > table.biohub-table {
  table-layout: fixed;
  width: 100%;
  min-width: var(--table-min, 0);
  font-size: 0.92rem;
  line-height: 1.6;
  font-variant-numeric: tabular-nums;
}
.content table.biohub-table th,
.content table.biohub-table td {
  padding: 0.6rem 0.7rem;
  white-space: normal;
  overflow-wrap: anywhere;
  vertical-align: top;
}
.content table.biohub-table { word-break: keep-all; }
.content p { word-break: keep-all; overflow-wrap: break-word; }
.content table.biohub-table th:nth-child(1) { width: var(--c1); }
.content table.biohub-table th:nth-child(2) { width: var(--c2); }
.content table.biohub-table th:nth-child(3) { width: var(--c3); }
.content table.biohub-table th:nth-child(4) { width: var(--c4); }
.content table.biohub-table th:nth-child(5) { width: var(--c5); }
.content table.biohub-table th:nth-child(6) { width: var(--c6); }
@container biohub (max-width: 620px) {
  .content .table-wrapper > table.biohub-records {
    display: block;
    min-width: 0;
    border: 0;
  }
  .content table.biohub-records thead {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  .content table.biohub-records tbody { display: block; }
  .content table.biohub-records tr {
    display: block;
    margin-bottom: 0.9rem;
    border: 1px solid var(--tb-border-color, #9996);
    border-radius: 0.3rem;
  }
  .content table.biohub-records td {
    display: block;
    width: auto;
    border: 0;
    text-align: left !important;
    padding: 0.45rem 0.75rem;
  }
  .content table.biohub-records td:first-child {
    font-weight: 600;
    border-bottom: 1px solid var(--tb-border-color, #9996);
    padding-block: 0.65rem;
  }
  .content table.biohub-records td:last-child { padding-bottom: 0.75rem; }
  .content table.biohub-records td:not(:first-child)::before {
    display: block;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted-color, #6c757d);
    margin-bottom: 0.1rem;
  }
  .content table.biohub-records td:nth-child(2)::before { content: var(--label2); }
  .content table.biohub-records td:nth-child(3)::before { content: var(--label3); }
  .content table.biohub-records td:nth-child(4)::before { content: var(--label4); }
  .content table.biohub-records td:nth-child(5)::before { content: var(--label5); }
  .content table.biohub-records td:nth-child(6)::before { content: var(--label6); }
}

@container biohub (max-width: 575px) {
  .content table.biohub-numeric:has(th:nth-child(4))::before {
    content: "↔ 좌우로 움직이면 나머지 열을 볼 수 있습니다.";
    display: table-caption;
    text-align: left;
    font-size: 0.78rem;
    color: var(--text-muted-color, #6c757d);
    padding-bottom: 0.35rem;
  }
}
@container biohub (max-width: 703px) {
  .content table.biohub-numeric:has(th:nth-child(5))::before {
    content: "↔ 좌우로 움직이면 나머지 열을 볼 수 있습니다.";
    display: table-caption;
    text-align: left;
    font-size: 0.78rem;
    color: var(--text-muted-color, #6c757d);
    padding-bottom: 0.35rem;
  }
}
.content mjx-container {
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
}
.content mjx-container[display="true"] { padding-block: 0.25rem; }
.content details { min-width: 0; }
.content summary { cursor: pointer; }
.content code { overflow-wrap: anywhere; }
</style>

<details markdown="1">
<summary>시리즈 안내와 참고 링크</summary>

- 대회: [BioHub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
- 공식 평가지표: [RoyerLab kaggle-cell-tracking-competition metrics.md](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)
- 이전 글:
  - [작업 기록 1: Lineage Graph 학습과 평가지표에 맞춘 후처리]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)
  - [작업 기록 2: Public 점수가 멈췄을 때 — OOF 기반 오류 분석]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/)
  - [작업 기록 3: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
  - [작업 기록 4: 로컬 검증에서 발견한 세 가지 빈틈]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
  - [작업 기록 5: 고정된 그래프가 시험하지 못한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
  - [작업 기록 6: 로컬 검증이 제출 파이프라인과 달랐던 문제]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline-KR/)
- 영문판: [BioHub Cell Tracking Working Note 7: When the Same Code Was Not the Same Experiment]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce/)
- 후속 글: [BioHub Cell Tracking 작업 기록 8: 최종 제출을 고를 때 고민한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two-KR/)

</details>

<details markdown="1">
<summary>관련 공개 노트북</summary>

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

</details>

> **시리즈 소개.** BioHub는 3D 현미경 영상에서 세포의 lineage graph를 복원하는 대회다. 라벨이 있는 학습 자료는 두 배아에서 촬영한 영상 199편이며, hidden test의 29%는 Public, 나머지 71%는 Private 점수에 반영된다. hidden test는 학습에 쓰이지 않은 배아에서 나온다. 각 편은 해당 기간의 기록을 따라가며, 나중에 확인한 사실과 회고는 따로 표시했다.
{: .prompt-info }

6편에는 질문 하나가 남았다. 제출 파이프라인을 그대로 replay하면 hidden test에서의 효과도 판단할 수 있을까. 9월 5–12일에는 hand label을 더한 구성의 199편 그래프 점수가 $$+0.0048$$ 올랐지만 Public에서는 $$0.005$$ 떨어졌다. pseudo-label 검출기를 embryo-out primary와 섞은 구성은 $$+0.0423$$을 얻었으나, all-train 가중치로 제출한 v88은 $$0.924$$였다.

hand label은 분열 precision과 주석 기준의 차이를 살펴보게 했다. 검출기는 실제 제출 가중치로 돌렸을 때 정렬 과정에서 피크가 사라졌다. teacher의 누수를 제거한 대조 실험도 유용한 pseudo-label 구성을 만들지 못했다. 비교를 정의하는 것은 코드뿐 아니라 가중치, 후보 집단, 라벨, teacher의 학습 경로였다.

---

## 0. 출발점: 제출 파이프라인 위에 다시 만든 로컬 검증

파이프라인은 6편과 같다.
검출기가 3D 프레임의 모든 voxel에 점수를 매기고, 그 피크가 세포핵 후보가 된다. 피크를 뽑기 전에 secondary 검출기의 검출 logit 맵을 primary에 맞춰 정렬해 섞는다.
transformer가 프레임 사이의 연결에 점수를 매기고, 정수 선형 계획(ILP)이 일관된 그래프를 고른다. gradient boosting division verifier는 분열 후보 중 무엇을 추가할지 정한다. 이 division verifier는 라벨이 붙은 후보 테이블과 함께 제출되어 노트북 안에서 다시 학습한다.
커널은 실제로 제출한 Kaggle 노트북이고, hidden test는 리더보드 뒤에 있는 테스트 영상이다.

학습 영상 $$199$$편은 배아 두 개에서 나왔고, 각각 $$128$$편과 $$71$$편이다(큰 배아, 작은 배아). embryo-out은 한 배아로 학습하고 다른 배아로 채점한다는 뜻이다.
로컬 검증의 중심은 제출 파이프라인 replay(deployed-stack replay)이고, 프로젝트에서는 kernel-faithful이라고 불렀다. 실제로 제출하는 런타임 코드를 embryo-out 모델로 199편 전체에 돌리고, 공식 채점기로 채점해 배아별로 따로 보고한다.
라벨이 공개된 예시 영상 네 편은 학습 영상의 복사본이라, 제출 모델을 여기서 채점하면 hold-in이다. 손상은 잡아낼 수 있어도 후보를 고를 수는 없다.

2026-09-05 시점에 제출돼 있던 커널은 v83이었다. 제출 파이프라인 replay $$0.7535$$, 로컬 분열 TP $$8$$개와 FP $$19$$개, Public $$0.944$$였다.

---

## 1. 첫 번째 아이디어: division verifier를 위한 hand label

6편에서 분열 쪽 점수는 division verifier의 순위 매기기에서 막혀 있었다. 순위 모델은 라벨이 붙은 예시로 배우므로, 맞는 라벨을 더 주면 순위를 매길 근거도 늘어난다.

### 1.1 라벨로 로컬에서 얻은 것

라벨링은 학습 영상의 분열 후보를 하나씩 보고 yes, no, skip 중 하나로 답하는 방식이었다. 표시 없이 섞어 둔 gold 문항은 정답 주석으로 답이 이미 정해져 있어서, 내 판정이 정답 주석과 얼마나 맞는지 잴 수 있었다.
첫 $$250$$문항에서 gold $$16$$개 중 $$14$$개를 맞혔고, 양성이 $$58$$개 늘어 모두 $$232$$개가 됐다.

영상 199편의 제출 파이프라인 replay에서 division verifier 학습 자료를 비교했다.

| 학습 자료와 임계값 | 점수 | 분열 TP / FP |
| --- | ---: | ---: |
| 기존 제출 테이블, 임계값 0.75 | 0.7535 | 8 / 19 |
| 직접 붙인 양성 +58, 임계값 0.75 | 0.7573 | 19 / 71 |
| 직접 붙인 양성 +58, 임계값 0.65 | 0.7583 | 24 / 96 |
{: #biohub-table-1 .biohub-table .biohub-numeric style="--c1: 52%; --c2: 21%; --c3: 27%; --table-min: 0; --label1: '학습 자료와 임계값'; --label2: '점수'; --label3: '분열 TP / FP'" }

두 설정 모두 두 배아에서 점수가 올랐고(전체 $$+0.0038$$, $$+0.0048$$), 점수가 더 높은 임계값 $$0.65$$ 설정을 v85 커널로 제출했다.
FP는 $$19$$개에서 $$96$$개로 늘었다. 로컬에서는 TP 증가의 이득이 이를 상쇄했지만, FP의 비용이 없었던 것은 아니다(1.3절).

### 1.2 Public sanity check

Public sanity check는 로컬 판단이 처음 보는 배아에서도 유지되는지 확인하는 제출이다. v85를 읽는 기준은 점수가 나오기 전에 적어 두었다. $$0.947$$ 이상이면 라벨을 지지하는 결과, $$0.944$$~$$0.946$$이면 중립, $$0.944$$ 미만이면 직접 붙인 라벨이나 낮춘 임계값이 손해를 낸 것으로 읽는다.
결과는 $$0.939$$로 $$0.005$$ 떨어졌고, 이 제출에 미리 정한 손실 기준선 아래였다. 9월 5–6일에 정리한 $$\pm0.002$$ 동점 범위도 벗어났다(3절). 이 범위는 프로젝트의 판단 규칙이지 반올림의 수학적 해상도가 아니다.

분열 주석이 세 건뿐인 예시 영상 네 편에서 v85의 간선 집계는 v83과 같았다. hidden division precision의 손실을 의심할 단서는 됐지만, hidden metric의 어느 항이 변했는지는 알 수 없었다.

### 1.3 거짓 분열의 비용이 달라질 수 있는 이유

공식 점수는 노드 수로 보정한 간선 Jaccard에 분열 Jaccard를 가중치 $$0.1$$로 더한 값이다. 분열 항은 전체 분열 이벤트를 한데 모아 계산한다. TP는 정답 주석과 매칭된 분열, FP는 채점 규칙에 따라 잘못된 것으로 집계된 예측 분열, FN은 놓친 정답 분열이다. 매칭되지 않은 모든 예측이 FP인 것은 아니다.

$$
J_{\mathrm{div}}=\frac{TP}{TP+FP+FN}.
$$

기존 사건의 매칭은 그대로 두고, 놓친 참 분열과 채점 대상 거짓 분열만 추가한다고 하자. 이때 추가 후보의 precision이 $$J_{\mathrm{div}}/(1+J_{\mathrm{div}})$$보다 높으면 분열 항이 오른다. 실제 분기 수정은 매칭과 간선도 바꿀 수 있으므로, 이 조건만으로 전체 점수의 상승을 예측할 수는 없다. 로컬에서는 이 항의 분모 대부분이 놓친 분열(FN)이다. $$J_{\mathrm{div}}$$는 $$0.06$$~$$0.10$$ 사이였고 이벤트 $$151$$개 중 $$127$$개를 놓치고 있었다.

6편의 단계 제거 실험으로 hidden division Jaccard를 약 $$0.32$$로 추정하려면 hidden 간선 효과가 작다는 가정이 필요하다. 아래는 그 가정에서 출발해 보는 산술 예시다.

TP 32, FP 20, FN 48이면 $$J=0.32$$다. 로컬에서 추가한 분열의 precision은 $$p=16/(16+77)\approx0.172$$였다. 같은 precision으로 참 분열을 8개 더 찾는다면, 거짓 분열은 약 $$8(1-p)/p=38.5$$개 늘어난다. Jaccard는 $$40/(40+58.5+40)\approx0.289$$로 내려가며, 간선이 그대로라면 최종 점수 변화는 약 $$-0.0031$$이다. 손실이 $$-0.01$$에 가까워지려면 precision이 더 낮아야 한다. 같은 TP 8개를 얻으며 FP가 80개 늘면 $$40/(40+100+40)\approx0.222$$가 된다.

![division Jaccard에 따른 손익분기 precision과 v85의 로컬 추가 분열 precision 0.172]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-01-division-breakeven.png)
_그림 1. 기존 매칭이 바뀌지 않는다고 가정한 계산이다. 로컬 추가 분열의 precision은 로컬 Jaccard에서의 기준은 넘지만, 가정한 $$J=0.32$$의 기준에는 못 미친다._

후보 수준의 라벨에서도 낮은 precision이 관찰됐다. v85가 추가한 분열 대부분이 속한 점수 구간 $$0.65$$~$$0.85$$에서 후보가 실제 분열인 비율은 $$13$$~$$29\%$$였다.

여기서 두 가지를 바꿨다.
분열 설정을 **precision frontier**로 옮겼다. FP를 v83 수준 근처로 제한하고, 그 안에서 TP가 가장 많은 설정을 고르는 방식이다.
그리고 6편에서 잠정적으로 세운 기대, 분열 쪽 변경의 효과가 hidden test에서 줄어들지 않는다는 기대를 철회했다. 로컬 이득이 Public에서 음수로 돌아온 분열 변경은 v85가 처음이었고, 그 기대는 비율이 아니라 몇 번의 관측에 기대고 있었다.

---

## 2. 실패 원인 추적: precision부터, 그다음은 라벨 자체

### 2.1 Precision frontier에서 다시 시험한 두 번째 배치

손실이 FP 분열에서 왔다면, 더 엄격한 임계값과 적용량 설정에서는 라벨이 여전히 도움이 될 수 있다.
두 번째 배치 $$500$$문항에서 양성이 $$253$$개 나왔고, v86은 이를 임계값 $$0.90$$으로 실었다. 로컬 점수는 $$0.7591$$, 분열은 TP $$18$$개와 FP $$25$$개였다.
단순한 추정으로는 v86이 유리했다. 추가한 분기마다 해당 점수 구간에서 내가 붙인 라벨의 precision을 적용한 계산이다(v86은 $$2{,}324$$개에 $$0.825$$, v83은 $$2{,}242$$개에 $$0.41$$). 하지만 이는 내가 판단한 후보의 precision이지, 채점기가 매칭한 사건의 precision이 아니다. 어느 분기가 실제 집계에 들어가는지 모르면 공식 Jaccard를 예측할 수 없다.
Public은 $$0.943$$으로, hand label이 없는 v83과 동점이었다. v85의 손실은 대부분 사라졌고, 이득은 보이지 않았다.
v85와 v86은 둘 다 라벨 테이블과 임계값을 함께 바꿨으므로, 어느 결과로도 각 변경의 효과를 나눌 수 없다(C16).

### 2.2 내 라벨을 채점기의 기준과 대조하기

gold 문항은 라벨 자체에 질문을 던졌다. 정답 주석이 달린 분열의 약 $$15\%$$에 내가 no라고 답했으므로, 정답 주석이 분열 $$151$$개를 어디에 두는지 측정했다.
정답 주석상의 부모 프레임에서, 부모 위치 $$7\,\mu\mathrm{m}$$ 안에 검출이 정확히 하나인 경우가 $$75\%$$, 둘인 경우가 $$15\%$$, 없는 경우가 $$8\%$$였다. 정답 주석은 부모가 아직 핵 하나일 때 분열 간선을 두고, 딸세포는 한 프레임 뒤에 나타난다. 딸세포 사이 거리의 중앙값은 정답 주석이 $$10\,\mu\mathrm{m}$$, 내가 양성으로 답한 것이 $$8.7\,\mu\mathrm{m}$$였다.

내가 no로 답한 gold 분열 일곱 개에서는 모두 정답 주석의 딸세포가 우리 후보 쌍 위에 있었다.
no로 답한 문항 중 딸세포 거리가 $$9\,\mu\mathrm{m}$$ 이상인 것(두 배치에서 각각 $$99$$개, $$109$$개)을 정답 주석의 기준으로 다시 판정했다.
60개가 no에서 skip으로 바뀌었고, yes가 된 것은 없었다. 다른 세포 뒤에서 나타나는 세포, z축 방향으로 일어나는 분열 같은 경우였다. no 라벨로 들어가 있던 이 문항들은 정답 주석이 분열로 셀 수도 있는 후보를 거부하도록 division verifier를 가르쳤다.

두 배치를 합친 테이블의 로컬 점수는 두 번째 배치만 쓴 테이블보다 낮았다.
주석 기준의 차이가 v85와 v86의 Public 결과까지 설명하는지는 확인하지 못했다. 코드와 채점기는 맞췄지만, 내가 붙인 라벨의 뜻은 채점기의 사건 정의와 대조하지 않았던 것이다. 그래서 C17에 라벨 기준을 먼저 확인하도록 적었다.

---

## 3. 변경 하나만 떼어 읽기: v87과 선택 규칙

v87은 한 가지만 바꿨다. v86의 테이블과 임계값을 그대로 두고 division verifier의 런타임 특징만 바꿨다.
새 특징 세 개는 내가 라벨링할 때 보던 단서다. 한 프레임 앞에서 딸세포 위치에 가장 가까운 검출까지의 거리, 그리고 두 프레임에 걸친 그 위치의 밝기다.
로컬 점수는 $$0.7609$$, 분열은 TP $$23$$개와 FP $$40$$개였다. 결과를 읽는 기준은 돌리기 전에 적어 두었다.

v86 점수가 나오기 전에 그날 밤의 조건을 정해 두었다. v86이 $$0.945$$ 이상이면 v87을 제출하고, $$0.944$$ 이하면 제출하지 않는다.
v86은 $$0.943$$으로, 전체 테스트 중 Public에 쓰이는 $$29\%$$에서 v83과 $$0.001$$ 차이였다. 조건대로라면 동점이 실험 여부를 정하게 된다. 게다가 v87의 질문은 v86의 점수 수준과 무관했다. v86은 전체 실험에서 유일하게 한 가지만 바꾼 비교의 대조군이었다.
v86 점수를 본 뒤에 그 조건을 따르지 않기로 하고 v87을 제출했다.

v87은 $$0.946$$이었다. v86보다 $$+0.003$$으로 프로젝트에서 Public 차이를 조사하기로 한 기준선에 있고, v83과는 동점이다.
이 한 축의 차이가 실제라면 런타임 특징의 몫이다. 이 특징은 라벨 없이 돌렸을 때 로컬에서 아무 효과가 없었다(기존 테이블에서 $$0.7489$$, 분열 TP 없음).

2026-08-28에 적은 규칙은 오래 점수가 멈춘 데 대한 대응이었고, 직전 주의 선택이 모두 로컬에서 이뤄졌는데도 여전히 리더보드를 목표로 적고 있었다.
2026-09-05와 2026-09-06에 걸쳐 이 규칙을 다시 썼다.

```text
선택은 누수 없는 로컬 근거로 한다: embryo-out, kernel-faithful,
  두 배아 모두 0 이상, 분열 operating point는 precision frontier 위
Public 차이가 0.002 이내면 동점이다
Public은 약 0.003 이상 크게 움직였을 때 반증 근거로만 인용한다
결과가 나오기 전에 적어 둔 가설만 제출한다
점수를 본 뒤에 규칙을 고쳐 쓰지 않는다
```

이 중 두 줄은 v87을 제출한 밤에서 나왔다. $$0.002$$ 이내의 Public 차이를 입력으로 받는 게이트는 두지 않고, 적어 둔 게이트는 점수를 본 뒤에 고쳐 쓰지 않는다.
v87 이후 2026-09-10까지는 제출하지 않았다.

---

## 4. 제출 없이 로컬 검증으로 답한 세 가지 질문

2026-09-06에 실험 네 개로 된 계획을 세웠고, 각 실험의 게이트는 첫 수치가 나오기 전에 적었다. 그래프 단계의 상수, division verifier용 다른 부모 후보의 정보, 분열을 고려한 association head fine-tuning, 그리고 우리 track으로 학습한 검출기(5절)다.
앞의 세 실험은 replay로 직접 잴 수 있는 메커니즘을 다뤘고, 셋 다 각자의 게이트에 따라 이틀 안에 끝냈다.

**Association head는 판별 성능 기준을 넘지 못했다.**
제출된 association head들의 점수 margin으로 두 경우를 가려낸 AUC는 $$0.610$$과 $$0.683$$으로, 게이트 $$0.70$$에 못 미쳤다. 실제 제브라피시 분열 1만 개 이상으로 학습한 association head도 $$0.643$$으로 나아지지 않았다.
이 실험은 pilot을 돌리고 몇 시간 만에 중단했다.

**다른 부모 후보의 정보를 더해도 division verifier는 충분히 좋아지지 않았다.**
특징 세 개짜리 logistic 모델은 embryo-out AUC $$0.80$$으로 제대로 이어진 딸세포를 구분했다. 하지만 런타임에서 임계값 $$0.90$$으로 돌린 division verifier는 $$0.7599$$(분열 TP $$21$$, FP $$39$$)로 기존 특징의 $$0.7605$$(TP $$23$$, FP $$47$$)에 못 미쳤고, 제출 기준과는 거리가 멀었다.
당시에는 이 결과를 기하·외형 신호가 더 이상 도움이 되지 않는다는 뜻으로 읽고 upstream을 살폈다. 그러나 실제로 중단한 것은 시험한 특징과 ranker 구성이다. 분열 정보를 표현하는 모든 방법의 한계를 확인한 것은 아니었다.

**시험한 그래프 구성 상수 중 두 배아 모두에서 요구한 이득을 낸 것은 없었다.**
replay에서 환경 변수 override가 조용히 무시되던 버그를 고친 뒤, 진단용 영상 12편에서 돌린 설정 14개 중 두 배아 조건을 통과한 것은 없었다.
간선 임계값을 $$0.50$$에서 $$0.35$$로 낮추면 작은 배아에서 $$+0.0646$$, 큰 배아의 영상 세 편에서 $$-0.0117$$이었고, 추정 세포 수 대비 노드 수 비율은 $$1.056$$에서 $$1.146$$으로 올랐다. 이 임계값은 ILP 전에 간선을 잘라 내고, ILP는 연결이 끊긴 노드를 지운다. 임계값을 낮추자 작은 배아에서는 실제 세포가, 큰 배아에서는 과검출된 노드가 살아남았다.
$$0.45$$에서는 199편 전체로 $$+0.0081$$과 $$+0.0008$$이었다. 규칙은 두 배아 모두 $$+0.003$$ 이상이었으므로 이 실험을 끝냈다.

---

## 5. 두 번째 아이디어: 주석 없는 세포핵에 양성 라벨 주기

### 5.1 가설

9월의 검출기 학습 코드는 주석이 있는 핵을 양성, 나머지 voxel을 음성으로 두고 음성 손실에 $$0.01$$의 가중치를 적용했다. 주석은 추정 세포 수의 약 $$2.8\%$$였다. 대부분의 핵에는 약한 음성 목표만 있었으므로, pseudo-label로 양성 학습 신호를 더할 수 있다고 보았다.
pseudo-label은 모델이 만든 라벨이다. 우리 embryo-out track을 정답에 더하면 같은 도메인의 라벨 노드가 $$36$$배가 됐다($$476$$만 개 대 $$13$$만 $$3$$천 개). track을 만든 모델이 teacher, 그 라벨로 학습한 검출기가 student다. student가 늘린 피크는 ILP가 정리할 것으로 예상했다.

첫 수치가 나오기 전에 적은 누수 논거는, student가 채점받는 배아로는 학습하지 않는다는 것이었다.
그 옆에 주의 사항 하나와 이를 확인할 대조 실험 계획을 함께 적었다. student의 학습 배아에 붙은 track은 student가 채점받는 배아로 학습한 teacher가 만들었다. 따라서 이득의 일부는 그 배아의 정답 주석이 teacher를 거쳐 student로 흘러 들어간 것일 수 있다.

### 5.2 게이트, 그리고 고쳐 쓴 조건 두 개

| student를 primary로 사용, embryo-out | tail: 평균 adjusted-edge Δ | 일반 영상: 평균 adjusted-edge Δ |
| --- | ---: | ---: |
| epoch 10, primary로 사용, 큰 배아 영상 15편 | +0.1280 | +0.0307 |
| epoch 50, primary로 사용, 같은 영상 | +0.1358 | +0.0774 |
| 반대 방향 student, primary로 사용, 작은 배아 영상 9편 | +0.1124 | +0.1033 |
{: #biohub-table-2 .biohub-table .biohub-records style="--c1: 56%; --c2: 22%; --c3: 22%; --table-min: 0; --label1: 'student를 primary로 사용, embryo-out'; --label2: 'tail: 평균 adjusted-edge Δ'; --label3: '일반 영상: 평균 adjusted-edge Δ'" }

tail 영상은 제출 파이프라인의 점수가 가장 낮았던 영상, 일반 영상은 중앙값 근처의 영상이다. 반대 방향 student는 학습과 평가 배아를 뒤바꿔 학습했다.
첫 줄의 통과 기준은 미리 적어 둔 대로 일반 영상 $$-0.003$$ 이상, tail $$+0.03$$ 이상이었다. 이후 인용한 $$+0.031$$은 15편 primary 검출기 패널 중 일반 영상 8편의 평균 adjusted-edge 증가량 $$+0.0307$$을 반올림한 값이다. 같은 구간의 평균 전체 점수 증가량은 $$+0.0293$$이었다. 6편에서 augmentation을 더한 검출기가 손해를 봤던 일반 영상에서, 처음으로 검출기 쪽 이득이 나왔다.

미리 적어 둔 조건 두 개가 충족되지 않았고, 둘 다 고쳤다.
반대 방향 student의 규칙은 합산 노드 수 비율이 기준값보다 $$0.05$$ 넘게 오르지 않아야 한다는 것이었는데, 실제로는 $$1.027$$에서 $$1.201$$로 올랐다. 199편 결과가 나오기 전에 이 제한을 두 조건으로 바꿨다. 두 배아 모두 점수 $$+0.003$$ 이상(이 점수에는 노드 수 penalty가 이미 들어 있다), 그리고 배아마다 영상별 노드 수 비율의 중앙값 $$1.5$$ 이하.
나중에 정한 구성 규칙은 primary로 썼을 때 손해가 가장 컸던 작은 배아 영상 여섯 편의 손실을 $$0.02$$ 이하로 묶었다. 실제 손실은 $$0.0257$$이었고, 이 수치를 본 뒤에 그 조건을 면제하고 판단을 기준이 이미 적혀 있던 199편 실행에 넘겼다.
두 변경 모두 제출하는 쪽으로 게이트를 느슨하게 했다. C2가 막으려는 방향이고, 두 변경 모두 기록에 남겨 두었다.

### 5.3 199편에서 커널로

이어서 가져간 구성은 student를 *secondary* 검출기로 쓰고, primary는 replay의 embryo-out 모델로 두는 것이었다.
199편 전체에서 공식 점수는 $$+0.042302511$$ 올랐고 두 배아 모두 양수였다. 43편은 점수가 떨어졌고, 여기에는 작은 배아의 고득점 영상(기준 점수 $$0.85$$ 이상) $$28$$편 중 $$17$$편이 들어 있다. 이 구간은 어떤 제출보다도 먼저 적어 둔 확인 대상이었다.

v88 커널은 secondary 가중치를 학습 영상 전체로 학습한 all-train student로 바꿨다. checkpoint는 학습 영상 $$40$$편에서 잰 학습용 proxy로 epoch $$52$$ 무렵을 골랐다. 여기에 작은 유효성 수정 두 개를 함께 실었다. 추가된 분열 때문에 한 세포가 자식을 셋 갖는 것을 막는 guard, 그리고 정수로 바꾼 출력 좌표를 volume 안에 두는 처리다.
v88이 불러온 all-train student는 어디에서도 채점된 적이 없다. 199편의 수치는 모두 embryo-out 모델 쌍에서 나왔다.
누수 대조 실험은 아직 돌리지 않았다.
v88의 기준도 점수가 나오기 전에 적었다. $$0.943$$ 이하면 분명한 악화, $$0.949$$ 이상이면 지지로 읽는다.

---

## 6. sanity check 실패: 로컬 검증은 제출 가중치를 돌린 적이 없었다

v88은 $$0.924$$로 v87보다 $$-0.022$$였다. 악화 기준선보다 한참 아래였고, 로컬 게이트는 5.2절에서 고쳐 쓴 대로 모두 통과한 상태였다.

### 6.1 정렬 공식 하나

검출기는 voxel마다 로짓을 출력한다. 배경으로 보이면 크게 음수, 세포핵으로 보이면 양수다.
커널은 secondary의 로짓 맵을 primary에 맞춰 정렬한 뒤, 피크를 뽑기 전에 둘을 섞는다.

$$
A=\left(S-\mu_S\right)\operatorname{clip}\!\left(\frac{\sigma_P}{\sigma_S},\,0.5,\,2\right)+\mu_P,
\qquad
B=0.525\,P+0.475\,A,
$$

여기서 $$P$$는 primary의 로짓 맵, $$S$$는 secondary의 로짓 맵이고, $$\mu$$와 $$\sigma$$는 프레임 전체에서 구한다.

이 공식은 프레임 전체의 평균과 표준편차 차이를 calibration 차이로 취급한다. 하지만 촘촘한 pseudo-label로 학습하면 핵으로 예측하는 영역 자체가 늘어난다. 평균을 맞추는 과정에서 배경뿐 아니라 실제 피크도 함께 이동할 수 있다.
밝은 예시 영상 하나에서 커널의 primary(예전에 학습한 all-train 모델)는 프레임 평균이 $$-14.7$$이다.
촘촘한 pseudo-label로 학습한 student는 voxel의 $$10$$~$$16\%$$에서 반응하고(primary는 $$3\%$$), 평균이 $$-5.0$$이다. 두 맵의 프레임 평균은 약 10 차이 난다. 이 평균에는 배경과 세포 반응이 함께 들어 있으므로 배경 수준만의 차이는 아니다.
이것을 약 10만큼 끌어내리면 진짜 피크까지 임계값 아래로 내려간다(그림 2).

![한 프레임의 피크 수: primary 281, 기존 secondary와 섞으면 253, student 644, 정렬한 student 0, primary와 student를 섞으면 25]({{ site.baseurl }}/assets/img/posts/2026-09-12-biohub-working-note-7/fig-02-alignment-peaks.png)
_그림 2. 밝은 예시 영상의 한 프레임을 제출 경로 그대로 처리했을 때의 피크. 프레임 전체 통계로 정렬하자, 새 검출기를 섞은 결과에는 피크가 $$25$$개만 남았다. primary만 쓰면 $$281$$개다. 한 프레임에서 메커니즘을 보여 주는 그림이고, 재현율을 잰 것은 아니다._

### 6.2 로컬 검증이 이것을 보지 못한 이유

제출 파이프라인 replay는 제출 코드를 embryo-out 가중치로 돌리고, 이 primary들의 평균 로짓은 약 $$-6$$~$$-9$$다. 시험한 모델 조합에서는 정렬한 student가 재현율을 높였다.
로컬에서 후보를 평가하는 과정은 제출용 primary를 쓴 적이 없어서, 실제로 제출한 구성(예전 all-train primary와 student)은 한 번도 측정되지 않았다.
두 환경은 영상 하나의 세포 수부터 달랐다. 추정 세포 수가 $$32{,}795$$개로 주어진 예시 영상에서 커널은 노드 $$18{,}423$$개(비율 $$0.56$$), replay는 $$47{,}740$$개(비율 $$1.46$$)를 만들었다.
replay에서 평가한 노드 수 민감 변경은 모두, 커널이 과소검출하는 영상을 과검출하는 맵 위에서 평가됐다.

커널의 snapshot, 실행 명령, 환경을 그대로 두고 유효성 수정만 빼서 다시 돌리자 노드 몇 개 차이로 같은 출력이 나왔다. 유효성 수정 두 개를 빼도 붕괴가 재현됐으므로, 이 수정들이 있어야만 생기는 문제는 아니었다. student 단독은 오히려 과검출하고(그 영상에서 노드 $$61{,}553$$개), 둘을 섞었을 때만 무너진다.
6편에서는 로컬 검증이 제출과 다른 코드를 replay하고 있었다. 이번에는 같은 빈틈이 한 단계 아래에 있었다. kernel-faithful은 코드를 가리키는 이름이었고, 가중치까지 보장하지는 않았다.

### 6.3 커널이 실제로 낸 출력

| 예시 영상 네 편, 제출 출력 | v87 | v88 |
| --- | ---: | ---: |
| 공식 점수 | 0.889473 | 0.857810 |
| 최종 예측 노드 수 | 120,450 | 77,002 |
| 라벨 노드 재현율 | 0.994528 | 0.957592 |
| 간선 TP / FP / FN | 2027 / 156 / 100 | 1939 / 158 / 188 |
{: #biohub-table-3 .biohub-table .biohub-numeric style="--c1: 46%; --c2: 27%; --c3: 27%; --table-min: 0; --label1: '예시 영상 네 편, 제출 출력'; --label2: 'v87'; --label3: 'v88'" }

제출 전 검증은 ID, degree, 좌표가 volume 안에 있는지를 봤고 점수는 보지 않았다. 그래서 기반 커널(v87)보다 노드가 3분의 1 이상 빠진 파일이 구조 검사를 모두 통과했다.
기반 커널과 점수를 비교하는 단계는 릴리스 절차에 적혀 있었지만 자동 검증 밖에 있었고, 돌리지 않았다.

### 6.4 바뀐 것: 실제 제출 가중치로 검증하기

2026-09-10에 정한 세 규칙을 C14로 묶었다.
첫째, 제출 전 예시 영상 네 편에서 후보의 실제 출력을 기반 커널과 비교해 채점한다. 예상하지 못한 재현율이나 간선 손실이 있다면 원인을 확인한다.
둘째, 검출이나 검출기 구성을 바꾸면 두 환경에서 평가한다. embryo-out replay는 backbone 학습에서 뺀 배아에서의 효과를 본다. **kernel regime(KR)**은 실제 제출 가중치와 노트북 코드를 학습 영상에 그대로 적용해, 제출할 조합이 제대로 동작하는지 확인한다. KR은 hold-in이므로 손해를 근거로 거부할 수 있지만 일반화 성능을 근거로 선택할 수는 없다. 두 환경이 엇갈리면 제출을 멈춘다.
셋째, 실행 기록에 측정한 가중치와 제출할 가중치를 나란히 적는다.

---

## 7. Public에서 두 변경 분리하기: v89와 v90

v88 점수가 나오기 전에 골라 둔 두 번째 변경 묶음 E는, 간선 점수 모델이 읽기 전에 primary의 특징 map을 test-time view들에 걸쳐 평균한다.
199편 전체에서 E는 대조군보다 $$+0.004523282$$ 높았고 두 배아 모두 양수였지만, 손해도 있었다. 작은 배아의 고득점 영상 $$28$$편 중 $$21$$편이 떨어졌고, 분열 FP는 $$37$$개에서 $$41$$개로 늘었다.

v89는 E와 v88의 유효성 수정 두 개를 합쳤고, 계획에는 Public 결과 하나로는 이 둘을 분리할 수 없다고 미리 적었다.
결과는 $$0.943$$으로 v87보다 $$-0.003$$이었다. 예시 영상 네 편의 공식 점수는 $$+0.0028158874$$로 부호가 반대였다.
v90은 v89에서 E만 껐다. v89 점수를 본 뒤에 나머지 조건을 맞춰 만든 대조군이므로 독립적인 재현은 아니다.
v90의 예시 영상 네 편 점수는 v87과 정확히 같았고, Public도 $$0.946$$으로 v87과 같았다.

v89와 v90의 대조에서는 유효성 수정을 적용한 파이프라인에서 E의 Public 효과가 음수였다. 차이는 프로젝트에서 조사 대상으로 삼은 0.003과 같았다. v87과 v90의 동점으로는 더 작은 유효성 수정 효과를 구분할 수 없다.
이 비교와 v87에서 남긴 원칙은 같다. 제출 하나에 한 축만 바꾸거나, 조건을 맞춘 대조군을 둔다(C16).

---

## 8. Teacher 누수를 제거하고 정렬 규칙 변경을 시험하다

### 8.1 teacher를 학습 배아 안으로 옮기다

대조 실험은 5.1절의 주의 사항에 적은 경로를 없앴다. teacher는 student 자신의 학습 배아로만 학습했다(처음 적은 설계안처럼 teacher를 다른 폴드에서 가져오면 그 경로가 다시 생긴다).
pseudo-label student와 정답 주석만 쓴 모델을 같은 시드, 같은 window, 고정된 $$60$$ epoch로 학습하고, 각각을 단독으로(자체 검출과 association만 쓰고 secondary와 division verifier는 끔) 예시 영상 네 편에서 양방향으로 돌렸다.

teacher 모델의 학습 경로를 바로잡은 대조 실험이다. 예시 영상 4편에서 모델을 각각 단독으로 실행했다.

| 평가 범위 | 정답만 학습 | pseudo student | 차이 |
| --- | ---: | ---: | ---: |
| 네 편 전체 | 0.7459 | 0.7253 | -0.0206 |
| 큰 배아 영상 | 0.7413 | 0.7215 | -0.0198 |
| 작은 배아 영상 | 0.8474 | 0.8112 | -0.0362 |
{: #biohub-table-4 .biohub-table .biohub-numeric style="--c1: 40%; --c2: 20%; --c3: 20%; --c4: 20%; --table-min: 36rem; --label1: '평가 범위'; --label2: '정답만 학습'; --label3: 'pseudo student'; --label4: '차이'" }

6편의 augmentation 조합에서 봤던 것처럼, 재현율 상승과 노드 수·간선 오류 증가가 함께 나타났다. 라벨 노드 재현율은 $$0.936$$에서 $$0.971$$로 올랐지만, 최종 노드는 $$24{,}712$$개 늘었고 간선은 TP가 $$83$$개, FP가 $$122$$개 늘었다. 노드 수 보정 항에서만 $$-0.0175$$가 나왔고, 분열은 그대로였다.
2026-09-06에 student의 늘어난 피크를 ILP가 정리한다고 적었던 예상은 철회한다. 누수을 제거한 대조 실험에서는 그렇지 않았다.

대조 실험은 teacher를 바꾸고, 고정된 60 epoch의 모델을 영상 네 편에서 단독으로 실행했다. 앞선 15편 primary 패널이나 199편 secondary 혼합 구성을 그대로 다시 돌린 것은 아니므로, 이전 이득 중 누수의 몫을 분리할 수는 없다. 빠진 비교는 유망했던 구성에서 teacher 경로만 바로잡은 대조군이었다. C15는 큰 이득에 기대기 전에 그 대조 실험을 먼저 하라는 요구다.

### 8.2 다른 정렬 규칙을 시험하다

정렬 수정안은 프레임 전체 평균에 맞추던 방식을 대체했다. 먼저 실제 all-train 가중치로 예시 영상 네 편을 평가했다(C14).

| 구성 | 공식 점수 | 대조군 대비 변화 |
| --- | ---: | ---: |
| base: 제출 검출과 association | 0.8898631853 | — |
| A: 제출 검출, student association | 0.8905639836 | A−base +0.0007007983 |
| Q: A에 student를 검출에도 추가, quantile 정렬 | 0.8818274412 | Q−A -0.0087365423 |
| P: A에 student를 검출에도 추가, 확률 blend | 0.8736703501 | P−A -0.0168936334 |
{: #biohub-table-5 .biohub-table .biohub-records style="--c1: 46%; --c2: 24%; --c3: 30%; --table-min: 0; --label1: '구성'; --label2: '공식 점수'; --label3: '대조군 대비 변화'" }

Q와 P는 둘 다 작은 배아에서 양수, 큰 배아에서 음수였고, 둘 다 재현율을 잃었다.
association만 바꾼 A는 같은 영상에서 embryo-out 모델로 돌리면 $$-0.003265956$$으로 두 배아 모두 음수였다. 두 경로의 부호가 엇갈렸으므로 A도 제출하지 않았다.

---

## 마무리

v88의 실제 출력을 기반 커널과 비교해 채점하자 예시 영상에서도 정렬 실패가 드러났다. 제출 전에 빠뜨린 점검이었다. hand label 실험에는 다른 불확실성이 남았다. 관련 Public 비교에서 라벨과 임계값을 함께 바꿨기 때문이다.

시험한 pseudo 검출기 조합 중 로컬 근거와 제출 동작을 모두 만족한 것은 없었다. embryo-out과 KR의 결과가 엇갈릴 수 있다면, 남은 후보에서 최종 두 개를 어떤 기준으로 골라야 할까.

<details markdown="1">
<summary>판단 기록·기준의 변화·남은 질문</summary>

아래 표는 당시의 결정과 그 근거를 정리한 것이다. 각 조항은 근거가 뒷받침하는 범위로 읽으며, 앞선 모든 실험이 해당 조건을 충족했다는 뜻은 아니다.

## 9. 판단 기록

2026-09-06부터는 3절의 규칙을 적용했다. embryo-out, kernel-faithful 로컬 근거로 고르고, 제출은 점수가 나오기 전에 읽는 법을 적어 둔 sanity check로만 한다.
sanity check로 고른 것은 없다. v85와 v88은 적어 둔 기대와 반대로 움직였고, 로컬 검증이 재현하지 못한 조건을 드러냈다.

| 판단 | 당시의 이유 | 결과 | 바뀐 것 |
| --- | --- | --- | --- |
| v85 제출: 라벨로 학습한 첫 division verifier | 로컬 +0.0048, 두 배아 모두 상승 | 0.939, -0.005 | 낮은 로컬 Jaccard에서는 손익분기 precision도 낮음; precision frontier 도입 |
| precision frontier에서 v86 제출 | 진단한 메커니즘 시험 | 0.943, v83과 동점 | 두 축을 함께 바꾸면 읽을 수 없음(C16); 라벨러 측정(C17) |
| v86이 0.943으로 나온 뒤 v87 제출 | 한 축만 바꾼 대조; 동점에 게이트를 걸면 노이즈를 판정으로 읽게 됨 | 0.946, v86보다 +0.003 | 유일한 한 축 비교, 프로젝트의 해석 기준선; 09-06 규칙 |
| 세 실험을 각자의 게이트에서 중단 | 첫 수치 전에 적어 둔 중단 기준 | 모두 게이트 미달 | 제출 없이 세 질문에 답함 |
| v88 제출: pseudo-label student를 secondary로 | 로컬 +0.042302511, 두 배아 모두 상승; 악화 기준선을 미리 적음 | 0.924, -0.022 | KR 검증과 제출 전 점수 확인(C14) |
| v89 제출 후 대조군 v90 | E 로컬 +0.004523282; 추론 대신 대조 실험 | v90 0.946, v87과 동점 | 유효성 수정의 작은 효과는 미해결; E는 Public -0.003으로 채택하지 않음 |
| 누수 대조 실험: teacher를 학습 배아 안으로 | +0.031에 다른 배아로 학습한 teacher가 끼어 있었음 | -0.0206, 두 배아 모두 음수 | 큰 이득을 믿기 전에 누수 없는 대조 실험(C15) |
{: #biohub-table-6 .biohub-table .biohub-records style="--c1: 25%; --c2: 25%; --c3: 28%; --c4: 22%; --table-min: 36rem; --label1: '판단'; --label2: '당시의 이유'; --label3: '결과'; --label4: '바뀐 것'" }

### 이 기간이 끝났을 때의 선택 기준

| 조항 | 내용 | 도입 |
| --- | --- | --- |
| C1 | 학습·보정·평가의 의존 관계를 분리하고 그래프 전체를 채점한다. 영상 분리만으로 배아 독립성이 보장되지는 않는다 | 2편 |
| C2 | 게이트는 결과를 보기 전에 정해 둔다 | 3편(07-15) |
| C3 | 규칙은 실제로 적용될 모집단에서 보정한다 | 3편 |
| C4 | 구성 요소는 자체 정확도가 아니라, 파이프라인을 그대로 재현해 만든 그래프로 평가한다 | 3편 |
| C5 | 점수와 변화량에 대조군을 함께 기록한다. 다른 replay의 변화량으로 우열을 매기지 않는다 | 3편 |
| C6 | 후보는 hidden test에서 제한 시간 안에 실행을 마쳐야 한다 | 3편 |
| C7 | 폴드는 배아 단위로 나눈다(embryo-out) | 4편 |
| C8 | 제출 모델이 학습한 영상에서 잰 수치(hold-in)는 일반화의 근거로 쓰지 않는다 | 4편 |
| C9 | Public은 기대치를 미리 적어 둔 sanity check에만 쓰고, 인접한 설정 중 하나를 고르는 데는 쓰지 않는다 | 4편(08-10) |
| C10 | 후보의 도달 범위를 잰다. 상한이라는 말은 선언한 범위를 포괄할 때만 쓴다 | 5편 |
| C11 | 게이트는 결론을 낼 수 있어야 한다 | 5편 |
| C12 | 로컬 검증은 실제 제출 파이프라인을 그대로 재현해서 한다 | 6편 |
| C13 | 단계 제거로 전체 효과를 잰다. metric 항을 분리하려면 추가 가정을 밝힌다 | 6편 |
| C14 **(신규)** | kernel regime(실제 제출 가중치·코드)에서 검증하고, 제출 전에는 노트북 출력 자체를 이전 버전과 비교해 채점한다 | 7편 |
| C15 **(신규)** | 큰 이득을 믿기 전에 누수이 없는 대조 실험부터 돌린다 | 7편 |
| C16 **(신규)** | 한 제출에서는 한 가지만 바꾸거나, 조건을 맞춘 대조군을 함께 둔다 | 7편 |
| C17 **(신규)** | 라벨은 채점 기준을 따른다. 라벨러의 판정부터 정답과 대조한다 | 7편 |
{: #biohub-table-7 .biohub-table .biohub-records style="--c1: 14%; --c2: 72%; --c3: 14%; --table-min: 0; --label1: '조항'; --label2: '내용'; --label3: '도입'" }

---

## 10. 이 기간에 확인된 것

### 확인된 사실

1. 로컬에서 $$+0.0048$$, 두 배아 모두 상승으로 평가된 hand label division verifier가 Public에서는 $$-0.005$$였다. 낮은 로컬 분열 Jaccard에서는 양의 손익에 필요한 precision도 낮았다.
2. 주석이 있는 분열의 $$75\%$$는 부모 프레임에서 핵이 하나로 보이고, 내가 거부한 gold 분열 일곱 개는 모두 우리 후보 쌍 위에 있었다.
3. pseudo-label student는 제출 파이프라인 replay에서 $$+0.042302511$$을 얻었다. 그 all-train 버전을 실은 커널은 $$0.924$$였고, 기반 커널보다 노드가 3분의 1 이상 빠진 채로 구조 검증을 통과했다.
4. E를 끈 v90은 v87과 동점이었고, E를 켠 v89는 $$0.003$$ 낮았다.
5. teacher를 student의 학습 배아 안으로 제한하자, pseudo student는 네 편 단독 실행에서 $$0.0206$$을 잃었고 두 배아 모두 음수였다. 대안으로 시험한 두 정렬 방식도 KR 평가에서 모두 음수였다.

### 근거는 있지만 아직 확정하지 못한 판단

1. hidden test에서 분열 precision이 낮았다면 v85의 손실을 설명할 수 있다. 직접 측정한 원인은 아니다.
2. 라벨 기준의 불일치가 v85와 v86의 Public 결과를 설명한다.
3. v86에서 v87로 오른 $$+0.003$$(프로젝트의 해석 기준선)은 런타임 특징 덕분이다.
4. 유효성 수정을 적용한 파이프라인에서 E가 Public 점수를 낮췄다.
5. 다른 배아의 정답 주석이 teacher를 거쳐 이전 이득에 영향을 주었을 가능성이 있다. 그 기여량은 조건을 맞춘 비교로 측정하지 않았다.

### 열린 질문

1. pseudo 검출기의 재현율을 KR 평가에서 노드 수 비용 없이 살릴 수 있는 구성이 있을까?
2. 검출 맵을 전혀 건드리지 않는 변경이라면, embryo-out 이득 중 얼마가 KR 평가에서도 남을까?

</details>

시리즈:

- [1편: Lineage Graph 학습과 평가지표에 맞춘 후처리]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)
- [2편: Public 점수가 멈췄을 때 — OOF 기반 오류 분석]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/)
- [3편: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- [4편: 로컬 검증에서 발견한 세 가지 빈틈]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
- [5편: 고정된 그래프가 시험하지 못한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
- [6편: 로컬 검증이 제출 파이프라인과 달랐던 문제]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline-KR/)
- **7편: 같은 코드로도 검증이 어긋나는 이유**
