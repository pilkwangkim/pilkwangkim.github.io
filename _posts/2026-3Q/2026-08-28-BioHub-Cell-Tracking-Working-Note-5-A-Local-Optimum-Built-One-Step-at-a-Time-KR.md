---
title: "BioHub Cell Tracking 작업 기록 5: 고정된 그래프가 시험하지 못한 것들"
date: 2026-08-28 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, objective-design, candidate-coverage, process-debt, division-recovery, working-note, korean]
math: true
last_modified_at: 2026-09-23
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-08-28-biohub-working-note-5/cover.png
  alt: "BioHub Cell Tracking 작업 기록 5: 고정된 그래프가 시험하지 못한 것들"
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
- 영문판: [BioHub Cell Tracking Working Note 5: What a Frozen Graph Left Untested]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time/)
- 후속 글: [BioHub Cell Tracking 작업 기록 6: 로컬 검증이 제출 파이프라인과 달랐던 문제]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline-KR/)

</details>

<details markdown="1">
<summary>관련 공개 노트북</summary>

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

</details>

> **시리즈 소개.** BioHub는 3D 현미경 영상에서 세포의 lineage graph를 복원하는 대회다. 라벨이 있는 학습 자료는 두 배아에서 촬영한 영상 199편이며, hidden test의 29%는 Public, 나머지 71%는 Private 점수에 반영된다. hidden test는 학습에 쓰이지 않은 배아에서 나온다. 각 편은 해당 기간의 기록을 따라가며, 나중에 확인한 사실과 회고는 따로 표시했다.
{: .prompt-info }

> **나중에 확인한 내용 — 2026-09-23.** 4–5절은 당시 탐색 방식을 돌아보는 내용이며, Public 차이 ±0.002 이내는 9월 5–6일에 정한 기준으로 동점으로 읽었다. 로컬 comparator와 제출 노트북의 그래프 처리 단계가 달랐다는 사실도 9월에 확인했다(6편). 8월 28일의 운영 전환 당시에는 아직 확인되지 않은 문제였다.
{: .prompt-info }

4편에는 설명되지 않은 Public 결과, 새로 정한 폴드 규약, 철회한 hold-in 진단이 남았다. 8월 17–28일에는 comparator를 고정해 수정안을 비교했다. 대부분의 이득은 작았다. 지금 돌아보면, 같은 파이프라인을 계속 다듬을 만큼 대안을 충분히 비교했는지가 더 큰 질문이었다.

두 진단은 기존 간선 복원 절차가 손댈 수 있는 범위가 좁다는 점을 보여 줬다. 8월 28일 점검은 이를 “provably exhausted”, 곧 더 개선할 수 없다고 증명된 상태로 읽었다. 측정이 뒷받침하는 것보다 강한 판단이었다. 다른 모델도 대부분 우리 그래프의 부품으로만 시험했으므로, 완성된 파이프라인끼리의 비교는 남아 있었다.

---

## 0. Comparator를 고정하면 무엇이 달라지는가

### 0.1 로컬 기준 그래프를 고정한 이유

8월의 폴드 점검 이후, 이 기간은 리더보드를 선택 과정에서 빼는 원칙 아래 진행했다.

> Public 리더보드 점수는 로컬 결과가 통하는지 관찰하는 값이지, 모델 선택의 목적함수가 아니다. 채택 여부는 폴드를 깨끗하게 나눈 로컬 OOF와 정확한 그래프 replay로 결정한다.

리더보드에는 제출이 160번쯤 쌓여 있었고, 그중 상당수는 한 축씩 바꿔 보는 sweep이었다. 리더보드는 hidden test의 29%에서 계산한 점수를 반올림해 보여 주므로 화면에서 동점인 두 모델은 순서를 가릴 수 없다. 문서상의 목표는 여전히 Public $$0.970$$이었다.

### 0.2 파이프라인 하나에 구성 요소를 한 칸씩 더해 온 방식

시스템의 뼈대는 처음부터 하나였다. 1편에서 다룬 공식 learned-model 계열로, 세포 중심을 찾는 temporal UNet, 학습한 간선 확률, 그래프를 고르는 ILP(integer linear program), repair 단계로 이루어져 있다. 이후의 분열 단계와 track-fragment matcher도 주로 기존 그래프에 더했을 때의 효과로 평가했다.

08-05에는 이 설계를 모델 계열별 capability matrix로 정리했다. dual-seed TemporalUNet3D 검출기 행에는 "공통 그래프 환경이자 비교 기준으로 유지"라고 적었고, 나머지 계열은 모두 그 그래프 위의 증거로 들어갔다. 08-11의 GPU 계획은 480 GPU-hours를 "조건부 연구 포트폴리오"라고 불렀지만, 실제로는 그래프 하나에 붙일 구성 요소들의 포트폴리오였다. 계획서에는 "공통 후보 토폴로지와 물리 특징은 바꾸지 않는다. 새 시드나 계열은 그 토폴로지에 대한 점수를 append-only sidecar로 덧붙인다"고 적혀 있다. 새 구성원 $$m_n$$은 현재 활성 집합 $$A_n$$에 더했을 때의 효과로 평가했다.

$$
\Delta_n = S(A_n \cup \{m_n\}) - S(A_n).
$$

3편에서는 baseline 일곱 개를 동시에 돌렸다. 그래프를 공유한 목적은 대조군을 고정해 비교를 쉽게 하는 것이었다. 서로 다른 수정의 이득을 단순히 더할 수 있게 되거나, 학습 의존 관계까지 사라진다는 뜻은 아니다.

### 0.3 고정 기준 그래프

고정 기준 그래프의 점수는 199편에서 $$0.6013708666$$이었다. 기반 예측은 배아별로 학습을 나눴고, 후보는 모델·그래프·후처리를 고정한 상태에서 비교했다. 기준 자체가 움직여 생기는 착시를 줄인 것이다.

여기서 embryo-out은 backbone의 학습 배아를 나눴다는 뜻이다. 아래의 prefix는 각 배아를, hold-in은 모델이 학습한 영상에서 잰 결과를 가리킨다.

당장 드러난 한계는 후보의 범위였다. 수정안은 고정한 절차가 만든 후보만 쓸 수 있었는데, 그 후보들이 어떤 오류까지 고칠 수 있는지는 아직 재지 않았다.

### 0.4 이 글의 숫자가 나온 곳

| 측정 도구 | 영상 | 점수 수준 |
| --- | --- | ---: |
| 기준 그래프의 부분집합 | 대회에서 배포한 테스트 영상 네 편(학습 영상의 사본) | 약 0.64 |
| 제출 노트북 파이프라인, hold-in | 같은 네 편 | 0.8899 |
| 제출 노트북 파이프라인, 배아가 겹치지 않게 학습한 검출기 | 같은 네 편 | 약 0.81 |
| 노드 구성이 다른 환경의 replay | 177편 | 0.69 부근 |
{: #biohub-table-1 .biohub-table .biohub-records style="--c1: 43%; --c2: 37%; --c3: 20%; --table-min: 0; --label1: '측정 도구'; --label2: '영상'; --label3: '점수 수준'" }

변화량은 각 도구 안에서만 읽고, 도구끼리 절대 점수를 비교하지 않는다. 이 기간에 리더보드는 점수 세 개를 돌려주었다. 제출 환경을 옮긴 버전, 출력이 바이트 단위로 같은 실행 시간 점검용 버전, 로컬에서 유일하게 양수였던 후보다. 셋 다 $$0.921$$로, 08-05 후보보다 높은 표시 점수는 나오지 않았다.

---

## 1. 기준 그래프로 시험한 아이디어들: 0에 가까운 결과와 동점 하나

| 실험 | 측정 도구 | 변화량 | 관찰 |
| --- | --- | ---: | --- |
| 검출기 시드 추가, 2-fold 정확 replay(당시 합산 비용 약 36 GPU-hours로 추정) | 177편 replay | -0.0000325 | 폴드별 +0.0000560, -0.0003967 |
| track-fragment matcher, 한 학습 그룹의 가중치 2.0→4.0 | 199편 embryo-out | -0.0000427 | 두 배아 모두 음수 |
| track history를 입력으로 넣은 matcher | 199편 embryo-out | -0.0000451 | 사전에 정한 점검은 모두 통과, 중단 |
| 보조 중심 검출기와 gap filling 파일럿 | 199편 embryo-out | +0.0000032, -0.0003537 | 둘 다 게이트 미통과 |
| 로컬 표현(representation) 계열 세 가지 | 영상 네 편 부분집합 | 세 번 모두 정확히 0.0 | 허용된 편집 0개 |
| **고정 checkpoint에서 독립적으로 재학습한 matcher** | 199편 embryo-out | $$\mathbf{+0.0006686}$$ | 34편 개선, 9편 악화, 156편 동일 |
{: #biohub-table-2 .biohub-table .biohub-records style="--c1: 32%; --c2: 19%; --c3: 20%; --c4: 29%; --table-min: 36rem; --label1: '실험'; --label2: '측정 도구'; --label3: '변화량'; --label4: '관찰'" }

matcher는 끊긴 track의 끝을 한두 프레임 뒤의 후속 노드에 다시 잇는 모델이다. 검출기 시드 추가는 이 기간에 계산을 가장 많이 쓴 항목인데, 처음 정리할 때는 양수였던 폴드 하나의 값($$+5.6\times10^{-5}$$)만 적었다. 비용 약 36시간도 완료된 12.78시간과 당시 예상한 나머지 약 23시간을 합친 값이므로, 확정된 최종 실측 비용과 구분해야 한다.

### 1.1 하나뿐인 양의 결과를 Public에서 확인하다

독립적으로 재학습한 matcher는 matcher 전 그래프($$0.6006730$$) 대비 $$+0.0006685607$$을 얻었다. 배아별로는 $$+0.0014526$$과 $$+0.0005501$$이었고, 편집은 3,207개, 분열 거짓 양성 수는 그대로였다. 이 계열의 마지막 버전은 로컬 replay와 예측 바이트가 같아야 하는 노트북으로 제출했고, 그 그래프가 기준 그래프가 되었다.

로컬 판단이 처음 보는 배아에서도 유지되는지 확인하는 Public 제출을 sanity check라고 불렀다. 08-22에 이렇게 제출한 v78은 $$0.921$$이었다. 표시 점수가 그대로였다는 사실만으로 로컬 $$+0.0006686$$의 전이를 지지하거나 반박할 수는 없었다. 0에 가까운 결과가 이어지자, 다음에는 이 복원 절차가 손댈 수 있는 오류가 얼마나 남아 있는지 살폈다.

### 1.2 표현 계열 세 가지, 고른 편집은 0개

로컬 표현 계열 세 가지(phase correlation, 학습한 dense descriptor cost volume, census cost volume)는 노드가 고정된 편집 정책에 더 좋은 매칭 신호를 주려고 만들었다. 셋 다 사전에 정한 점검을 모두 통과했지만 허용된 편집을 하나도 고르지 않았고, 고정 그래프의 영상 네 편 점수 $$0.6391791890184004$$를 16자리까지 그대로 돌려주었다. 이 결과는 고정된 정책 아래 세 신호가 편집을 바꾸지 못했다는 뜻이다. 신호 자체가 쓸모없는지, 정책이 그 신호를 활용하지 못했는지는 가리지 못했다.

### 1.3 단순 기준선과의 비교

full-$$Z$$ Otsu thresholding과 인접 프레임 Hungarian linking을 쓴 고전적 baseline은 같은 영상 네 편, 같은 채점기에서 $$0.4626$$이었다. 학습 그래프의 $$0.6392$$보다 $$-0.1766$$ 낮았다. 시험한 고전적 방법이 더 나은 출발점이라는 설명은 배제했지만, 학습 그래프의 남은 오류에 후보가 얼마나 닿는지는 여전히 확인해야 했다.

---

## 2. 기존 복원 절차가 손댈 수 있는 오류

### 2.1 이미 고른 수정 3,178개를 정답으로 걸러 내기

08-24에는 고정 ShiftCorr replay가 이미 고른 교체 3,178개를 살폈다. 정답 간선은 줄이지 않고 거짓 간선도 늘리지 않으면서, 둘 중 하나는 개선하는 수정만 정답 라벨로 골라 다시 채점했다.

$$
S_{\mathrm{full}}=0.6013708666,
\qquad S_{\mathrm{filtered}}=0.6017363254,
$$

$$
\Delta_{\mathrm{selected\ filter}}=+0.0003654588.
$$

이득은 모두 한 배아에서 나왔다. 다른 배아의 점수는 이 필터를 적용해도 그대로였다.

### 2.2 복원한 후보 목록에서 빠진 간선을 찾기

별도의 진단은 소스에 있는 후보 생성기를 다시 실행해 간선 후보 58,427개를 복원했다. 기준 그래프가 놓친 정답 간선 38,060개 중 대체 후보가 있는 것은 183개, 그중 토폴로지 조건까지 통과할 수 있는 것은 많아야 115개였다.

![빠진 정답 간선 38,060개 중 복원한 후보 목록에서 도달 가능한 183개와 구조상 허용되는 최대 115개]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-01-edge-swap-inventory.png)
_그림 1. 현재 소스의 후보 생성기를 복원한 결과다. 노드를 고정한 채 가능한 모든 편집의 목록은 아니며, 과거 실행에서 사용한 후보 바이트가 완전히 같았다는 증명도 아니다. 분열 편집은 이 집계 밖이다._

ranker가 아무리 좋아도 후보에 없는 간선은 고를 수 없다. 이 목록은 현재 생성기를 그대로 둔 채 순위 모델만 개선해서는 효과가 작을 수 있는 이유를 보여 줬다.

### 2.3 Public 점수로 환산할 수는 없다

이전 계획은 로컬 이득의 $$0.14$$배에서 $$0.9$$배가 Public에 나타났다고 적었다. 하지만 4편의 로컬 association 이득 $$+0.0073$$은 짝지은 Public 대조군과 나중의 해석 기준으로 동점이었다. 이런 관찰로 선택된 수정의 진단값을 Public 이득으로 환산할 수는 없다.

### 2.4 당시의 결론과 지금의 정정

8월 26일 요약은 이를 oracle bound로 설명했고, 8월 28일 점검은 comparator의 수정 공간이 “provably exhausted”라고 결론 냈다. 이 판단은 탐색을 노드와 분열 쪽으로 옮기는 근거가 됐다.

하지만 지금 보면 결론이 너무 강했다. $$+0.000365$$는 이미 선택된 교체 3,178개만 걸러 낸 결과다. TP와 FP를 함께 바꾸면서도 점수를 올릴 수 있는 수정은 필터에서 제외했다. 후보 범위 진단도 복원한 생성기 하나를 대상으로 했을 뿐, 가능한 그래프 수정과 조합을 모두 조사하지 않았다. 두 절차에서 추가로 얻을 여지가 작았다는 결론은 가능하지만, 파이프라인이 최적점에 도달했다고 할 수는 없었다.

이 경험에서 얻은 교훈을 C10으로 정리했다.

```text
C10. 후보의 도달 범위를 잰다. 상한이라는 말은 선언한 범위를 포괄할 때만 쓴다.
```

---

## 3. 고정한 편집 범위 바깥에서 던진 질문

다음 비교는 고정된 간선 교체 후보 바깥의 노드와 분열 후보를 살폈다. 후속 실험의 방향은 제시했지만, 대안 파이프라인의 일반화 성능을 확인한 것은 아니다.

### 3.1 외부 모델을 우리 그래프의 부품으로 시험했을 때

08-20에는 외부에서 학습된 모델 네 개를 우리의 고정 노드와 토폴로지에 맞춰 붙여서, association이나 repair 판단만 달라지게 했다. 영상 네 편 부분집합에서 forward-acceleration lookahead는 바꿀 만한 곳을 하나도 찾지 못했고($$0.0$$), exact-coordinate 검출기 어댑터는 $$-0.0044123$$, 학습한 movement-field 모델은 $$-0.0017270$$이었다. 4-D convolution 어댑터는 $$0.324$$초 만에 fail-closed로 멈췄다. 4편의 사전학습 부모 순위 모델도 같은 식으로 끝났었다.

이 이식 실험만으로 연관 쪽에 여지가 없다고 할 수는 없었다. 각 모델에는 우리 그래프의 부품으로서 무엇을 더하는지만 물었고, 자기 노드와 링크, 분열 예측을 갖춘 end-to-end로 우리 파이프라인을 이기는지는 한 번도 묻지 않았다.

### 3.2 독립 파이프라인과 나란히 놓고 본 우리 그래프

08-26에는 로컬에서 채점할 수 있는 영상 네 편에서, 따로 만들어진 참조 파이프라인과 우리 그래프를 두 번 비교했다. 참조 파이프라인의 기본 빌드와 비교하면 참조 쪽에만 있는 노드가 53,858개, 우리 쪽에만 있는 노드가 59,382개였다. 이 추가 노드 가운데 주석된 세포를 참조 쪽은 103개 찾았고 우리는 19개 찾았다. 두 번째 빌드와 비교하면 공유 노드 106,543개 위의 간선 Jaccard가 $$0.9978$$이었고, 자식이 둘인 부모는 참조 쪽이 384개, 우리가 37개였다.

공유 노드 위에서는 두 파이프라인이 거의 같다. 차이는 어떤 노드가 존재하는지와 분열을 몇 개 내는지에서 났고, 둘 다 앞선 고정 간선 교체 실험에서는 고정해 둔 요소였다. 영상 네 편에 대한 우리 쪽 진단이고, 누군가의 점수를 잰 결과가 아니다.

---

## 4. 돌아보니 탐색에서 빠져 있던 질문

0.2절의 설계 덕분에 작은 변경끼리는 비교하기 쉬웠다. 그 대신 몇 가지 중요한 비교는 끝내 하지 못했다.

### 4.1 경로가 하나뿐이다

각 단계는 현재 그래프를 개선할 때만($$\Delta_n > 0$$) 받아들였다. 채택한 단계는 다음 후보의 대조군이 되므로, 탐색은 여러 파이프라인을 고루 비교하기보다 한 경로를 따라갔다. 대부분의 비교에서 검출기와 노드 집합을 고정한 것도 탐색을 좁혔다. 노드를 바꾸려던 시도가 전혀 없었던 것은 아니다. 다만 4.3절의 두 시도는 최종 그래프를 채점하기 전에 멈췄다. 초기 선택이 최선임을 확인한 채 유지한 것은 아니었다.

### 4.2 대안을 부품으로 평가했다

다른 계열은 그 자체로 낼 점수가 아니라 현재 그래프에 더했을 때의 효과로 평가했다. 그 효과는 계열이 그래프에 이미 있는 것과 겹치거나 충돌하면 작아지고, 다른 노드를 중심으로 만든 계열을 고정 노드에 맞추면 원래 기능의 일부가 사라질 수 있다. end-to-end로는 이기지만 부품으로는 지는 계열과 그냥 더 나쁜 계열이 똑같이 보인다.

### 4.3 질문 하나에 드는 비용

리더보드를 선택에서 빼는 원칙은 4편 기간에 만든 체계로 지켰다. 프로세스의 종료 코드(exit code)가 곧 과학적 판단은 아니므로, 실험마다 계획을 미리 적어 두고, 실행을 시작하거나 결과로 인정받으려면 게이트를 통과해야 하며, 실행 기록(receipt)은 고칠 수 없게 했다.

이 기간에는 한 축만 바꾸는 실험에도 새 파일이 16개쯤 필요했다. 두 기록 형식 사이에 필드 이름 하나가 맞지 않아 작업 331개짜리 계획이 하나도 진행되지 못했다. 한 줄짜리 수정 때문에 새 이름으로 다시 띄운 연관 실험은 전체 파이프라인을 두 번 돌려 두 구성 모두 $$0.8277207263$$을 받았다.

노드 구성을 바꾸려던 탐색 두 개는 아무것도 결정하지 못하고 멈췄다. scratch 검출기는 held-out 영상 하나에서 제안 11,594개 가운데 1,190개를 기존 노드에서 $$7\,\mu\mathrm{m}$$ 넘게 떨어진 곳에 놓았지만, 그 출력 형식을 받을 어댑터를 미리 선언해 두지 않아 멈췄다. native spot 검출기는 held-out 재현율이 $$7\,\mu\mathrm{m}$$ 기준 $$0.50$$ 이상이어야 한다는 식의 준비 기준(readiness floor)에 묶여 있었는데, $$0.179$$와 $$0.036$$에 그쳤고 채점한 영상은 없었다. 규칙상 준비 기준 미달로는 그 계열을 시작할 수도 끝낼 수도 없었다. 2026-08-28 00:02 UTC에 RTX 5090의 사용률은 0%였고, 실행들은 승인을 기다리고 있었다.

이 사례들은 실행 조건과 입력 형식 요구 때문에 예측 실험이 지연됐음을 보여 준다. 해당 요구를 간소화할 이유는 되지만, 모든 게이트의 비용이 그 효용보다 컸음을 입증하지는 않는다. 확인 절차에 시간을 들이려면 결과에 따라 다음 행동이 달라져야 한다. 통과하면 다음 단계로 가고, 떨어지면 그 방향을 끝내거나 무엇을 고칠지 알려 줘야 한다.

```text
C11. 게이트는 결론을 낼 수 있어야 한다.
```

### 4.4 선택 기준으로 할 수 있는 것과 없는 것

선택 기준은 주어진 후보를 평가한다. 부품으로만 시험한 모델이 독립된 파이프라인으로는 어떤 결과를 낼지 보여 주지는 못한다. 수정안이 목표한 오류에 닿지 못하는 문제도 선택 기준으로 해결할 수 없다.

---

## 5. 다시 한다면 무엇부터 비교할까

다시 한다면 독립된 end-to-end 파이프라인을 초반에 비교하는 데 시간을 더 쓰겠다. 순서는 다음과 같다.

1. **강한 end-to-end 계열 여러 개를 먼저 만든다.** 각 계열은 자기 그래프를 끝까지 만들고 제한 시간 안에 끝나야 한다(C6). 그래야 구성 요소가 아니라 그 자체로 제출 후보가 된다.
2. **검증 체계는 하나로 둔다.** 같은 embryo-out 폴드와 공식 평가지표를 그래프 전체에 적용하되, 간선 항과 분열 항을 따로 읽는다. 그러면 노드 구성이나 분열 예측의 차이가 항별 점수 차이로 드러난다.
3. **선택과 조합은 그 검증 체계 안에서, 그래프 전체 단위로 한다.**
4. **깊이는 그다음이다.** 이 기간에 처음부터 파이프라인 하나에 했던 단계별 다듬기는 이긴 계열에 한다.

0.2절의 설계는 그래프를 공유해서 숫자를 비교할 수 있게 했다. 공통 검증 체계는 폴드와 채점기를 공유해서 같은 일을 하고, 그래프는 서로 달라도 된다. 그렇게 하면 이 기간에 고정해 둔 그래프 차이를 직접 시험할 수 있다. 넓이에도 함정은 있다. 배아 두 개로 여러 계열 가운데 가장 좋은 것을 고르면 winner's curse가 따라오므로, 계열을 고르는 단계에도 따로 held-out 규율이 필요하다.

---

## 6. 한 달을 남기고: 전면 점검과 분열 단계

### 6.1 전면 점검과 여섯 가지 규칙

8월 28일까지 23일 동안 Public 표시 점수는 $$0.921$$을 넘지 못했다. 8월 22일 v78로 sanity check를 했지만 점수는 그대로였다.

전면 점검은 2.4절의 판단을 근거로 운영 전환을 요구했다. 예측 품질보다 comparator와 실험 절차를 유지하는 데 일이 집중됐고, 과거의 큰 이득 여러 개는 실제 제출 파이프라인에서 확인하지 않았다는 것이었다.

4.3절의 체계는 걷어 내고 여섯 가지 규칙으로 바꿨다. Public 리더보드를 목적함수로 삼고 로컬 OOF는 선택 도구로 쓴다. 관리용 코드(governance code)는 만들지 않는다. 할 일이 있는 동안 GPU는 놀리지 않는다. 새 아이디어에서 예측을 내는 첫 산출물까지 두 시간 안에 간다. 실패한 실행은 버그를 고친 뒤 같은 이름으로 다시 돌린다. 이미 결론이 난 결과는 다시 도출하지 않는다. 이 규칙과 함께 쓴 계획은 마감까지 리더보드에서 금메달권에 드는 것을 목표로 잡았고, 하루 최대 다섯 번의 제출을 거의 비용이 들지 않는 확인 수단으로 셈했다.

성과가 나지 않는 흐름을 바꾸려던 첫 규칙은 Public을 선택에서 제외한다는 기존 원칙과 충돌했다. 운영 목표는 바꿨지만, 같은 Public 자료에 반복해서 맞추게 되는 위험을 해결하지는 못했다.

### 6.2 보류해 둔 조합을 같은 날 다시 재다

마지막 발견 때문에 새 계획의 맨 위에는 4편에서 보류해 둔 joint lineage-action 조합이 올라갔다. replay 근거는 $$+0.0144$$, 이어서 $$+0.0313$$이었고, 리더보드에서는 $$+0.005$$에서 $$+0.025$$로 예상했다. 하지만 이 숫자들은 4-fold 연구용 replay(기준 점수 약 $$0.74$$)에서 나왔고, 그 4-fold 결과는 4편의 embryo-out 규약 아래에서 근거의 지위가 낮아진 상태였다. 새 규칙 아래 첫 측정은 제출 노트북을 단계적으로 재구성한 로컬 파이프라인에 이 조합을 넣어, 라벨이 있는 예시 영상 네 편에서 비교했다.

| 단계 | 조건 | 대조군 | 적용 | 변화량 |
| --- | --- | ---: | ---: | ---: |
| 1 | hold-in, 대조군이 제출 kernel과 다름 | 0.8831 | 0.8842 | +0.0011 |
| 2 | hold-in, v78 혼합 단계 재현; 일부 개수 일치 | 0.8866 | 0.8858 | -0.0008 |
| 3 | 배아가 겹치지 않게 학습한 검출기, 4-fold association head | 0.8064 | 0.8129 | +0.0065 |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 9%; --c2: 43%; --c3: 16%; --c4: 16%; --c5: 16%; --table-min: 44rem; --label1: '단계'; --label2: '조건'; --label3: '대조군'; --label4: '적용'; --label5: '변화량'" }

1단계는 제출본의 혼합 단계를 재현하지 못했다. 2단계에서는 v78의 양방향 합성과 consensus를 반영했다. 실행 기록상 영상별 개수는 거의 같았지만, 노드 수까지 정확히 같은 영상은 두 편이었다. 기준 점수도 노트북의 $$0.8899$$와 다른 $$0.8866$$이므로 출력이 완전히 같다고 확인한 것은 아니다. 이처럼 제출본에 더 가깝게 만든 replay에서 조합의 효과는 $$-0.0008$$이었다. 3단계에는 여전히 두 배아가 섞인 폴드로 학습한 association head가 들어 있었다. 이 결과로 준비한 이식을 보류했지만, 새 배아에서의 효과나 모델 계열 전체를 판정할 수는 없었다.

변경 $$L$$의 효과는 대조군 $$B$$에 따라 달라진다. 이를 $$\Delta_L(B)=S(L\circ B)-S(B)$$로 쓸 수 있다. 그사이 제출 파이프라인에는 TTA, 역방향 연결 점수의 조화 평균, consensus, 다른 간선 임계값이 들어갔다. 당시에는 이들이 예전 이득의 “약 75%를 흡수했다”고 적었다. 하지만 서로 다른 그래프와 평가 조건의 변화량으로는 그 비율을 구할 수 없다. 새로 확인한 것은 준비된 조합을 추가해도 이 네 편의 학습 예시에서는 v78에 더 가깝게 맞춘 replay가 좋아지지 않았다는 사실이다.

### 6.3 분열 복원으로 방향을 옮긴 이유

마감인 09-29까지 한 달이 남은 시점에서, 이미 제한 시간 안에 도는 파이프라인의 분열 후보를 조사하기로 했다. division Jaccard는 다음처럼 가중치 $$0.1$$로 점수에 더해진다.

$$
S = J^{\mathrm{adjusted}}_{\mathrm{edge}} + 0.1\,J_{\mathrm{division}}
$$

한 번 시도한 편집이 약 $$-0.00018$$의 손해를 낸 뒤로, 몇 주 동안 이 항은 건드리지 않고 지키기만 하는 대상이었다. 두 관찰이 이 항을 가리켰다. 3편과 4편에서 리더보드는 분열 단계를 $$0.004$$ 크기의 계단으로 읽었지만 replay는 약 $$0.001$$만 인정했다. 그리고 3.2절의 참조 파이프라인은 자식이 둘인 부모를 384개 냈고 우리는 37개였다. C10에 따라 구조와 후보 도달 범위부터 199편 embryo-out 그래프에서 쟀다. 아래에서 분기는 나가는 간선이 둘인 노드, 곧 예측된 분열이다.

| 항목 | 값 |
| --- | ---: |
| 199편 전체의 정답 분열 | 151 |
| 완전히 복원 가능(부모와 두 딸 세포가 모두 $$7\,\mu\mathrm{m}$$ 안에서 매칭) | 107 |
| 부모-딸 간선 두 개 중 적어도 하나가 이미 그래프에 있음 | 101 |
| 주석된 부모 위치의 예측 분기 | 849 |
| 그중 실제 분열 부모에 있는 것 | 2 |
{: #biohub-table-4 .biohub-table .biohub-numeric style="--c1: 67%; --c2: 33%; --table-min: 0; --label1: '항목'; --label2: '값'" }

복원 가능한 실제 딸 세포 쌍은 중앙값 $$10.7\,\mu\mathrm{m}$$ 떨어져 있었고, 집계된 거짓 분기는 $$5.4\,\mu\mathrm{m}$$였다. 현재 제출본의 규칙은 두 자매 후보가 서로 $$8.5\,\mu\mathrm{m}$$ 안에 있고 각각 부모에서 $$4.66\,\mu\mathrm{m}$$ 안에 있을 때만 분기를 허용했다. 이 자매 거리 조건은 집계된 거짓 분기의 96.3%를 통과시켰고, 실제 분열은 29.1%만 통과시켰다. 이 표본에서는 거리 조건이 복원 가능한 참 분열보다 집계된 거짓 분기를 더 자주 통과시켰다.

### 6.4 정답을 사용한 분열 진단과 학습한 division verifier

199개 그래프 전체에서 oracle 모드로 쟀다. 딸 세포를 새 부모에 다시 붙일 수 있게 하고(reparenting), 어떤 분기를 남길지는 라벨이 고르게 했다.

| 구성 | 점수 | 0.6014 대비 | 분열 세부 |
| --- | ---: | ---: | --- |
| 분기 없는 기반 | 0.6005 | -0.0009 | 현재 분열 단계 전체의 가치를 +0.0009로 매김 |
| 현재 제출본의 엄격한 분열 단계 | 0.6014 | — | 현재 기준 |
| 정답 oracle, reparenting 허용 | $$\mathbf{0.6571}$$ | $$\mathbf{+0.0557}$$ | $$J_{\mathrm{div}}=0.5577$$; TP 87, FP 5, FN 64; 분기 107개 추가, 그중 37개는 reparenting |
{: #biohub-table-5 .biohub-table .biohub-records style="--c1: 28%; --c2: 16%; --c3: 19%; --c4: 37%; --table-min: 36rem; --label1: '구성'; --label2: '점수'; --label3: '0.6014 대비'; --label4: '분열 세부'" }

![서로 다른 개입과 대조군에서 측정한 라벨 기반 진단 및 학습 모델의 이득]({{ site.baseurl }}/assets/img/posts/2026-08-28-biohub-working-note-5/fig-02-two-ceilings.png)
_그림 2. 199편에서 측정한 학습 모델의 이득과 정답 기반 진단. 각 행에 비교 기준을 표시했다. 추론 때 정답 라벨을 읽는 진단은 그대로 제출할 수 있는 방법이 아니다._

이어서 학습한 division verifier로 reparenting이 가능한 후보들의 순위를 매겼다. 배아 단위로 분리해서(prefix-pure), 각 배아에 적용하는 순위 모델은 다른 배아로만 학습했다. 하루 동안 기하 특징만 쓰는 선형 모델($$+0.0015$$, oracle의 참 양성 87개 중 6개)에서 track 특징을 넣은 gradient boosting 순위 모델($$+0.0035$$)로, 다시 appearance 특징까지 넣은 버전($$+0.0053$$, 점수 $$0.6067$$, 참 양성 11개, 거짓 양성 26개)으로 나아갔다.

같은 날 한계 두 가지도 쟀다. 시뮬레이션한 분열 Jaccard는 각 순위 모델이 학습한 배아에서 $$0.43$$과 $$0.67$$이었지만, 학습하지 않은 배아에서는 합쳐서 $$0.062$$였다. 배아 사이의 전이 문제가 드러났지만, 이 요약 수치만으로 순위·보정·후보 집단의 영향을 완전히 분리할 수는 없었다. 또 후보 생성기는 151개 분열 중 75개만 덮었다.

### 6.5 로컬 게이트와 sanity check

사전에 정한 hold-in 게이트로 분열 편집이 간선 쪽 점수를 망가뜨리지 않는지 확인했다. 후보 kernel은 이전의 영상 네 편 점수 $$0.8899$$를 영상별 간선 수까지 똑같이 재현했고, 분열 편집 45개는 모두 주석된 track에서 떨어진 곳에 있었다. 이 네 편에서의 성질이고, 분열 편집이 어디서나 안전하다는 근거는 아니다.

후보 v79는 08-28 늦게 sanity check로 리더보드에 올렸다. 이 기간 후보 가운데 로컬 근거가 가장 강했고, 이전 분열 구성에서도 Public 변화가 로컬 변화보다 컸었다. 그날 적어 둔 기대치는 하나였다. 이득은 간선 변화량처럼 줄지 않고 비슷한 크기로 리더보드에 나타난다. 3~4편의 분열 두 쌍은 로컬 $$+0.000949$$와 $$+0.0010142$$가 리더보드에서 각각 $$+0.004$$씩 움직였지만, 방향을 보여 줄 뿐 비율은 아니다. 점수가 분명히 떨어지면 이 후보는 반증된다. 이 기간이 끝날 때까지 리더보드는 답하지 않았다.

---

## 마무리

comparator를 고정하자 작은 변화도 잴 수 있게 됐지만, 모든 비교에서 검출기·노드·후보 생성기가 같았다. 결과가 0 근처에 머물렀을 때는 파이프라인이 최적점에 가까운지, 시험 범위가 좁아서 대안이 보이지 않는지 가릴 수 없었다. 두 진단이 잰 것은 이미 고른 교체의 필터와 생성기 하나의 도달 범위였고, 완성된 다른 파이프라인은 채점하지 않았다.

다시 한다면, 초반에 여러 완성된 파이프라인을 비교할 시간을 먼저 확보하겠다. fold와 scorer는 같게 두되 각 파이프라인의 노드·간선·분열 예측은 그대로 살려 비교하고, 그 뒤에 깊이 들어갈 대상을 고르겠다.

남은 한 달에 실제로 택한 후보는 reparenting을 허용한 division verifier였다. 정답 기반 진단은 $$+0.0557$$, 학습 모델은 $$+0.0053$$을 얻었고, v79를 막 제출한 상태였다. 8월 28일에는 로컬 이득이 Public에도 비슷한 크기로 나타나리라고 적었다. 다음 결과는 그 기대를 시험하게 된다.

<details markdown="1">
<summary>판단 기록·기준의 변화·남은 질문</summary>

아래 표는 당시의 결정과 그 근거를 정리한 것이다. 각 조항은 근거가 뒷받침하는 범위로 읽으며, 앞선 모든 실험이 해당 조건을 충족했다는 뜻은 아니다.

## 7. 판단 기록

08-28까지는 고정 embryo-out 기준 그래프가 선택의 목적함수였고, sanity check v78은 표시상 동점이었다. 08-28에는 성과가 나지 않는 흐름을 바꾸려고 Public을 목적함수로 바꿨고, 같은 날 로컬 근거로 고른 분열 후보 v79를 두 번째 sanity check로 보냈다. 이 운영 규칙과 C9의 충돌은 남아 있었다.

| 결정 | 당시의 이유 | 돌아온 결과 | 바뀐 것 |
| --- | --- | --- | --- |
| 08-05·08-11의 그래프 하나 설계 위에서, 고정 embryo-out 기준 그래프(0.6013708666)를 목적함수로 삼음 | 폴드 문제 이후, 조건이 움직이지 않는 비교 기준이 필요 | 0이거나 10⁻⁴ 미만의 변화량 | 이 공간은 최대 얼마를 줄 수 있나? |
| matcher 이득(+0.0006686)을 v78로 제출(08-22) | 처음 보는 배아에서의 sanity check | 0.921, 동점 | 표시상 동점만으로 로컬 이득의 전이를 판정할 수 없음 |
| 선택된 수정 필터와 후보 도달 범위 진단(08-24) | 0에 가까운 결과가 이어진 뒤, 여지가 있기는 한가? | +0.000365; 빠진 간선 38,060개 중 교체 가능한 것 115개 | C10 |
| 고정 그래프 바깥을 살핌(08-20, 08-26) | 여지는 다른 어디에 있나? | 외부 모델 이식은 음수이거나 변화 없음; 공유 노드 간선 Jaccard 0.9978; 자식이 둘인 부모 384 대 37 | 노드와 분열의 차이를 따로 시험할 필요 |
| native 검출기를 준비 기준 뒤에 둠 | 세포를 찾지 못하는 검출기에 채점 실행을 쓰지 않는다 | 재현율 0.179, 0.036; 점수도 종료도 아님 | C11 |
| 탐색 범위를 회고(09-23) | 대안을 주로 기존 그래프의 부품으로 평가 | 전체 파이프라인 대안은 미검증 | 다음에는 더 넓은 비교에 시간을 배정 |
| 전면 점검: 리더보드를 목적함수로, 보류한 +0.0313 조합을 다시 잼(08-28) | 23일 동안 Public 표시 점수가 0.921을 넘지 못함 | 여섯 가지 규칙; v78에 더 가까워졌지만 일치하지는 않은 replay에서 -0.0008 | 변경의 가치는 기반에 달려 있음 |
| 분열: 정답 기반 진단 뒤 division verifier를 만들고 v79 제출(08-28) | 기존 파이프라인 안에서 손댈 수 있는, 측정된 가장 큰 여지 | 진단 +0.0557; division verifier +0.0053; Public 대기 | 다음 sanity check를 기다림 |
{: #biohub-table-6 .biohub-table .biohub-records style="--c1: 25%; --c2: 25%; --c3: 28%; --c4: 22%; --table-min: 36rem; --label1: '결정'; --label2: '당시의 이유'; --label3: '돌아온 결과'; --label4: '바뀐 것'" }

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
| C10 **(신규)** | 후보의 도달 범위를 잰다. 상한이라는 말은 선언한 범위를 포괄할 때만 쓴다 | 5편 |
| C11 **(신규)** | 게이트는 결론을 낼 수 있어야 한다 | 5편 |
{: #biohub-table-7 .biohub-table .biohub-records style="--c1: 14%; --c2: 72%; --c3: 14%; --table-min: 0; --label1: '조항'; --label2: '내용'; --label3: '도입'" }

---

## 8. 이 기간에 확인된 것

### 확인된 사실

1. 고정 기준 그래프가 고른 편집 3,178개에 정답 필터를 적용한 가치는 $$+0.0003654588$$로 모두 한 배아에서 나왔고, 별도로 복원한 후보 목록에서 빠진 정답 간선 38,060개 가운데 구조상 허용되는 교체가 있는 것은 115개(0.30%)였다.
2. 08-05와 08-11의 설계는 검출기 하나의 그래프를 공통 환경으로 두고, 다른 계열은 모두 그 그래프에 더했을 때의 효과로 평가했다.
3. matcher($$+0.0006686$$)는 리더보드에서 동점이었고, 표현 계열 세 가지는 편집을 하나도 고르지 않았으며, 노드를 고정한 채 외부 모델을 이식한 시도는 모두 음수이거나 변화가 없었다.
4. 영상 네 편에서 독립 파이프라인은 공유 노드 위에서는 우리와 거의 같았고(간선 Jaccard $$0.9978$$), 어떤 노드가 존재하는지(주석된 세포 103개 대 우리 19개)와 자식이 둘인 부모 수(384 대 37)에서 달랐다.
5. native 검출기는 결과로도 종료로도 치지 않는 준비 기준에서 두 번 멈췄고, 보류했던 연관 조합은 v78에 더 가까워졌지만 완전히 일치하지는 않은 replay에서 $$-0.0008$$이었다.
6. 정답 분열 151개 중 107개는 완전히 복원 가능했고 주석된 부모 위치의 분기 849개 가운데 실제 분열 부모에 있는 것은 2개였으며, reparenting을 허용한 oracle은 $$+0.0557$$, 배아 단위로 분리한 division verifier는 하루 만에 $$+0.0053$$이었다.

### 근거는 있지만 아직 확정하지 못한 판단

1. 노드 구성과 분열 후보를 바꾸는 실험이 유용할 수 있다. 연관 쪽의 모든 여지를 부정하지는 않는다. 영상 네 편에서 한 비교 두 번과 분열 oracle에 기댄 판단이다.
2. 전체 파이프라인을 여러 개 비교했다면 다른 차이를 더 일찍 발견했을 가능성이 있다. 4절의 메커니즘에서 읽은 판단이고, 그런 포트폴리오는 만들지 않았다.
3. 배포 기준의 변화로 보류한 연관 이득이 줄었을 가능성이 있다. 기준과의 상호작용을 폴드 누수와 분리하지 못했으며, 흡수한 비율도 추정할 수 없다.
4. 과거의 $$0.14\times$$~$$0.9\times$$ 범위로 새 후보를 예측할 수 있는지는 확인되지 않았다. 환산 계수로 쓰지 않는다.

### 열린 질문

1. 각 후보 생성기는 어떤 오류에 도달하며, 정답을 이용한 진단 가운데 어느 것이 선언한 편집 범위의 실제 상한인가?
2. 리더보드를 목적함수로 삼으면 리더보드에 맞춰지는 것은 무엇으로 막는가? 이 규칙은 C9와 어떻게 맞물리는가?
3. 분열 이득은 08-28의 기대치대로 줄어들지 않고 리더보드에 나타나는가?
4. division verifier는 배아를 넘어 일반화되는가? 배아가 두 개이고 라벨이 있는 양성이 약 80개뿐이라, "prefix-pure"는 사실상 폴드 두 개다.
5. 어떤 강한 end-to-end 계열이 제한 시간 안에 끝나고, 간선 항과 분열 항을 따로 읽는 하나의 embryo-out 검증 체계에서 서로 어떻게 비교되는가?

</details>

시리즈:

- [1편: Lineage Graph 학습과 평가지표에 맞춘 후처리]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)
- [2편: Public 점수가 멈췄을 때 — OOF 기반 오류 분석]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/)
- [3편: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- [4편: 로컬 검증에서 발견한 세 가지 빈틈]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
- **5편: 고정된 그래프가 시험하지 못한 것들**
