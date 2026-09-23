---
title: "BioHub Cell Tracking 작업 기록 8: 최종 제출을 고를 때 고민한 것들"
date: 2026-09-18 21:00:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, lineage-reconstruction, final-selection, stage-jitter, registration, association-head, embryo-out, selection-bias, oof, working-note, korean]
math: true
last_modified_at: 2026-09-23
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-09-18-biohub-working-note-8/cover.png
  alt: "BioHub Cell Tracking 작업 기록 8: 최종 제출을 고를 때 고민한 것들"
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
  - [작업 기록 7: 같은 코드로도 검증이 어긋나는 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce-KR/)
- 영문판: [BioHub Cell Tracking Working Note 8: What Went Into Choosing the Final Two]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two/)

</details>

<details markdown="1">
<summary>관련 공개 노트북</summary>

- [Biohub Cell Tracking: Data Model, EDA, Baseline](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline)
- [Biohub Cell Tracking: Learned Graph w Gap Recovery](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery)
- [Biohub Cell Tracking: Blend Preprocessings](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings)

</details>

> **시리즈 소개.** BioHub는 3D 현미경 영상에서 세포의 lineage graph를 복원하는 대회다. 라벨이 있는 학습 자료는 두 배아에서 촬영한 영상 199편이며, hidden test의 29%는 Public, 나머지 71%는 Private 점수에 반영된다. hidden test는 학습에 쓰이지 않은 배아에서 나온다. 각 편은 해당 기간의 기록을 따라가며, 나중에 확인한 사실과 회고는 따로 표시했다.
{: .prompt-info }

7편에서는 embryo-out(EO)과 kernel regime(KR)의 결과가 엇갈릴 수 있다는 문제가 남았다. 9월 13–18일에는 최종 제출 두 개를 골라야 했다. hidden test의 나머지 $$71\%$$가 Private에 쓰이고, 두 제출 중 Private 점수가 높은 쪽이 성적이 된다. v92는 stage-jitter registration을 더했고, v93은 association head를 교체했다. v93을 남기려면 KR 게이트 실패와 예시 영상 네 편의 손실을 받아들여야 했다.

선택은 v93과 v92였다. 둘 다 registration을 유지했으므로, head 재학습이 해로운 경우에는 대비하지만 registration이 hidden 배아에 도움이 되지 않을 위험은 함께 안았다. 이후 열한 갈래의 작업에서는 더 채택할 후보가 나오지 않았다.

---

## 0. 파이프라인과 측정 도구: 어떤 질문을 무엇으로 재는가

검출기 두 개의 출력을 합쳐 프레임마다 핵을 찾는다. Transformer 기반의 **association head**가 각 핵에서 다음 프레임 핵들로 이어지는 후보 연결에 점수를 매기는데, 정방향과 역방향으로 한 번씩 호출해 두 결과를 합친다. ILP가 서로 모순 없는 연결 집합을 고르고, motion relink와 line-fit smoother를 포함한 결정론적 후처리 단계들이 그래프를 고친다. 마지막으로 학습된 **division verifier**가 점수가 $$0.90$$을 넘는 곳마다 분기를 더하며, 영상 한 편에 최대 $$50$$개까지다. 점수는 adjusted edge Jaccard에 division Jaccard의 $$0.1$$배를 더한 값이다. 배포 기준선은 v90이었다. v87에 7편의 유효성 수정 두 가지를 더한 버전이고, Public은 $$0.946$$이었다.

판단에는 평가 환경 세 가지를 썼다. 후보와 대조군은 각 환경 안에서 비교했으며, 서로 다른 환경의 점수 수준을 같은 척도로 읽지는 않았다.

| 측정 도구 | 돌리는 것 | 역할 |
| --- | --- | --- |
| EO (embryo-out) | 검출·association의 gradient 학습에서 평가 배아를 뺀다. checkpoint 선택과 후속 단계는 완전히 분리되지 않았다. 영상 199편 (44b6: 71, 6bba: 128) | 효과 판정. 선택을 결정하는 게이트 |
| KR (kernel regime) | 배포 가중치와 노트북 코드 그대로, 같은 영상 199편 | 실제 제출 파이프라인의 손해 감지. in-sample |
| 예시 영상 4편 | 노트북의 실제 출력을 로컬에서 채점 | 배포 결과가 로컬과 일치하는지 확인하는 용도로만 |
{: #biohub-table-1 .biohub-table .biohub-records style="--c1: 20%; --c2: 56%; --c3: 24%; --table-min: 0; --label1: '측정 도구'; --label2: '돌리는 것'; --label3: '역할'" }

EO는 6–7편의 제출 파이프라인에 embryo-out 가중치를 넣은 평가다. 전체 파이프라인이 독립적인 평가는 아니었다. backbone checkpoint는 평가 배아에서 골랐고, 일부 후속 단계에는 all-train 학습 자산을 썼다. 이 한계는 v90·v92·v93 모두에 해당한다.

**LB90**은 영상을 bootstrap한 90% 구간의 하한이다. 그 재표본화 방식 안에서의 안정성을 보여 주며, 처음 보는 배아 사이의 불확실성을 나타내지는 않는다.

---

## 1. 마감 직전, Public으로 아직 답할 수 있었던 것

09-06부터 Public 차이가 $$\pm 0.002$$ 이내면 동점으로 보고, 약 $$0.003$$ 이상 움직일 때만 조사 근거로 삼았다. v91은 v90에서 secondary 검출기의 특징을 test-time view에 걸쳐 평균한 버전이다. 로컬의 무효과 판정이 처음 보는 배아에서도 유지되는지 확인하는 sanity check로 09-13에 제출했다. Public은 $$0.946$$으로 v90과 동점이었다. 그날 저녁에는 Public의 역할을 한 번 더 좁혔다.

```text
검증을 통과한 후보마다 제출은 한 번만 한다.
Public 점수에서는 한 가지만 본다. 배포 기준 버전(parent)보다 0.003 이상 떨어졌는가.
점수가 떨어지면 검증을 시작할 뿐, 그것만으로 결정하지 않는다.
"점수가 X 이상이면 Public에서도 통한 것" 같은 구간은 어떤 기준에도 넣지 않는다.
배포 결정은 미리 정해 둔 로컬 게이트, 즉 EO와 KR로 한다.
최종 두 개는 내가 지정하고, Public 최고점을 선택 규칙으로 쓰지 않는다.
```

예전에 리더보드로만 답하던 질문에는 이제 로컬 측정 도구가 있었다. 작게 반올림된 차이로 후보 순서를 정하지 않기로 했다. "X 이상이면 통한 것" 같은 구간을 두면 운 좋게 나온 점수가 근거가 되고, 제출이 쌓이면 가장 운 좋은 후보가 뽑힌다. 리더보드가 여전히 잘하는 일은 v88 같은 붕괴를 잡아내는 것이었다(C19).

---

## 2. 무엇을 근거로 볼 것인가: 작은 패널이 아니라 영상 199편 전체

7편에서는 놓친 분열이 어디에 있는지 모르는 채로 분열 순위 실험들을 중단했다. 09-13에는 v87의 division verifier와 기존 raw graph·ShiftCorr 경로를 사용한 기존 분열 replay를 199편 전체에서 추적했다. 3절의 registration 평가와는 별도의 기준 그래프다. 주석이 달린 분열 $$151$$건 가운데 $$23$$건은 복원됐고, $$72$$건은 올바른 후보가 없었으며, $$56$$건은 올바른 후보가 있었지만 점수가 임계값 $$0.90$$보다 낮았다. 이 $$56$$건 중 $$46$$건에서는 가장 좋은 올바른 후보가 같은 부모의 후보들 사이에서 이미 1위였다. 나는 calibration 오류를 의심했다. 학습 라벨이 현재 그래프의 후보와 맞지 않는 것이 한 가지 가설이었다. 그렇다면 유효한 라벨로 다시 학습했을 때 임계값을 넘는 실제 분열이 늘어나야 한다.

이번에는 재학습의 효과를 분리해서 보기 위해 임계값을 고정했다. 앞선 v85는 라벨과 임계값을 함께 바꿨다가 Public에서 $$0.005$$ 떨어졌다. 임계값이 원인이었다고 확인한 것은 아니지만, 그 손실을 설명하지 못한 상태에서 다시 낮추기보다는 재학습부터 시험하기로 했다. 분열 사례가 있는 영상 4편에서는 참 분열 하나를 더 찾아 $$+0.01099$$를 얻었다. 사건 하나의 영향이 컸으므로 채택 여부는 199편 전체에서 판단했다.

| 측정값 | 영상 4편 | 영상 199편 |
| --- | ---: | ---: |
| 공식 Δ | +0.01099 | -0.00679 |
| 분열 TP/FP/FN | +1 / +2 / -1 | 23/37/128 → 22/196/129 |
{: #biohub-table-2 .biohub-table .biohub-numeric style="--c1: 34%; --c2: 28%; --c3: 38%; --table-min: 0; --label1: '측정값'; --label2: '영상 4편'; --label3: '영상 199편'" }

노드 수와 노드 재현율은 그대로였지만, 참 분열을 더 찾지 못한 채 거짓 분열만 늘었다. 이 후보와 파이프라인에서는 재학습으로 calibration 문제를 해결한다는 가설이 성립하지 않았다.

그날 오후에는 두 번째 작은 패널이 뒤집혔다. 7편에서 결론을 내지 못한 quantile alignment는 pseudo-label 검출기를 다시 배포하려는 것이었는데, 영상 2편짜리 소규모 실행 점검에서 $$+0.104$$와 $$-0.0016$$, KR 평가의 영상 29편 패널에서 $$-0.033027$$이 나왔다. 이 실험의 출발점이던 embryo-out replay의 6bba tail 중앙값 $$+0.066$$은 커널에서 부호가 반대였다. C14에 따라 KR 평가에서 잰 덕분에 노트북을 만들기 전에 이를 잡았다.

세 번째 경고는 7편에서 다룬 teacher 누수 대조 실험이었다(영상 네 편 $$-0.0206$$). 앞서 이득을 낸 구성을 그대로 다시 돌린 실험이 아니므로, 그 이득 중 얼마가 teacher 누수 때문인지는 구분할 수 없었다.

이후 채택은 미리 정한 199편 패널 전체에서 판단했다(C18). 작은 패널은 구현 가능성과 실제 개입 여부를 확인하는 데 계속 쓸 수 있었다. 다만 영상을 모두 평가해도 같은 자료를 다시 쓴다는 사실은 달라지지 않는다.

---

## 3. 첫 번째 후보: stage-jitter registration(v92)

### 3.1 정말 모든 가능성을 검토했을까

같은 날에는 모든 가능성을 검토했는지 묻고, 남아 있던 아이디어를 전부 다시 훑었다. 새로 나온 제안 $$30$$개 가운데 스물여덟 개는 기각됐고, 남은 두 개는 같은 메커니즘이었다. 당시 정리한 아이디어 목록을 훑은 것이지, 현재 제약 아래 가능한 모든 방법을 검토했다는 뜻은 아니다.

### 3.2 메커니즘

일부 프레임 사이에서 현미경 스테이지가 움직이면 세포 전체가 한꺼번에 몇 마이크로미터씩 이동한다. 이것이 **stage jitter**다. 이 데이터에서는 한 번 일어나면 $$3$$에서 $$9\,\mu\mathrm{m}$$이고, 학습 데이터의 프레임 전환 가운데 $$11.6\%$$가 $$3\,\mu\mathrm{m}$$ 이상 움직인다.

배포된 단계 가운데 두 곳이 이런 프레임을 잘못 처리한다. motion relink는 실제 후속 노드 대신 $$z$$ 방향 이웃에 연결하고, 이어서 line-fit smoother가 점프한 프레임의 좌표를 되돌린다. registration은 프레임 전환마다 검출 결과만으로 전역 이동량 하나를 추정해 relink와 line-fit에 넣는다. 이 단계는 별도 모델을 학습하지 않고 추론 중 정답 라벨도 쓰지 않는다. 검출기와 association head는 v90에 쓰던 학습 모델 그대로다. 스테이지 이동이 주요 오류 원인이라면, 점프가 잦은 영상에서 보정 이득이 클 것이라는 가설이었다. 배아가 달라도 반드시 같은 관계가 유지된다는 보장은 없었다.

### 3.3 결과 전에 적어 둔 게이트

| 게이트 | registration 적용 버전 − v90 |
| --- | --- |
| G3: KR 평가, 영상 29편 패널 | +0.0254, 심각한 손해 없음 |
| G4: KR 평가, 영상 199편 | +0.026578 (44b6 +0.023806 / 6bba +0.027151) |
| G5: embryo-out, 영상 199편 | +0.013526 (44b6 +0.012265 / 6bba +0.013666) |
| G6: 노트북 자체 | 예시 영상 4편 0.889473 → 0.961506 |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 36%; --c2: 64%; --table-min: 0; --label1: '게이트'; --label2: 'registration 적용 버전 − v90'" }

별도 모델을 학습하지 않는 이 registration은 두 환경과 두 배아 모두에서 양수였다. 제출 가중치 환경에서 $$3\,\mu\mathrm{m}$$ 이상 점프가 $$15$$회 이상인 영상의 이득 중앙값은 $$+0.039$$, 점프가 없는 영상은 $$0.000$$이었다. 다만 주의할 결과도 있었다. 배아를 제외한 평가의 이득은 학습 자료를 다시 평가한 경우의 절반 정도였다. 더 우려스러운 것은 분열 항이었다. EO에서는 $$-0.0047$$(TP $$36 \to 26$$)이었는데, KR에서는 TP가 $$6$$개 늘었다. 이 부호 차이를 설명하지 못했고, 제출 전에 그대로 기록했다.

### 3.4 Public sanity check

v92는 프로젝트에서 가장 강한 로컬 근거를 가진 후보였고, 09-13에 sanity check로 제출했다. 미리 적어 둔 기준은 Public에서 $$+0.002$$에서 $$+0.012$$를 예측했는데, 제출 뒤 점수가 나오기 전에 이 구간은 이상치만 보는 규칙으로 대체됐다.

결과는 v90의 $$0.946$$에 대해 $$0.945$$였다. 점수가 나오기 전에 바꾼 규칙으로는 이상 징후가 없었다. 반올림된 점수 하나로는 작은 이득, 드문 점프, 간선 이득과 분열 손실의 상쇄를 구분할 수 없었다.

9월 14일에는 각 배아 영상의 71%를 중복 없이 뽑는 subsampling을 10,000번 했다. 전체 199편을 대상으로 했을 때는 두 환경 모두 음의 차이가 한 번도 나오지 않았다. 하지만 $$3\,\mu\mathrm{m}$$ 이상 점프가 두 번 이하인 EO 영상 64편만 보면 $$-0.003284$$였다. 이 집단에서는 adjusted edge 항이 약 $$+0.0015$$였지만 division Jaccard가 약 $$-0.0481$$이었다. 손실은 분열 항에서 나왔다.

점프가 없는 영상 41편의 결과는 음수가 아니었다. EO에서는 $$+0.002375$$, KR에서는 $$+0.000063$$이었다. 점프가 드물다는 이유만으로 손실을 예측할 수는 없었다.

따라서 전체 패널의 재표본화만으로 점프가 드문 hidden 집단에서의 손실까지 배제할 수는 없었다. 9절에서 이 공통 위험을 다시 다룬다.

---

## 4. 두 번째 후보: association head 재학습(v93)

H1은 추론 파이프라인이 실제로 만드는 후보로 학습한 association head가 배포된 association head보다 연결을 잘하는지 물었다. 검출기는 비트 단위까지 그대로 두고 association head만 warm start로 다시 학습했다. 검출 결과는 평가지표가 매칭하는 방식 그대로 주석에 매칭했고, 실제 후속 노드를 아는 곳에서 negative 샘플을 만들었다. v93은 v92에서 association head만 이것으로 바꾼 버전이다.

embryo-out 확인은 두 부분으로 되어 있다. K4는 6bba 영상 $$128$$편을 그 영상 없이 학습한 association head로 채점하고, **K5**는 반대 방향 분할(reciprocal)을 더해 199편 전체를 채점한다. K5의 조건은 다섯 가지다. 합산 이득 $$+0.004$$ 이상, 두 배아 모두 0 이상, LB90 0 초과, 분열 TP가 기존 association head보다 세 개 넘게 줄지 않을 것, 점수가 떨어진 영상이 지나치게 많지 않을 것.

| 게이트 (결과 전에 적음) | 측정값 |
| --- | --- |
| K5: EO, 영상 199편, reciprocal | +0.010347 (44b6 +0.01341 / 6bba +0.00983); 분열 TP $$26 = 26$$, FP 38 → 31 |
| 기여 분해 (K5 이후, 설명용) | 기존 학습 규칙으로 같은 예산만큼 재학습 +0.0081; 그 재학습 대비 새 규칙 +0.0023 |
| KR: 배포 backbone, all-train association head, 영상 199편 | 합산 +0.00003 (44b6 +0.0017 / 6bba -0.00025): 중단 |
{: #biohub-table-4 .biohub-table .biohub-records style="--c1: 38%; --c2: 62%; --table-min: 0; --label1: '게이트 (결과 전에 적음)'; --label2: '측정값'" }

H1은 K5의 다섯 조건을 모두 통과했지만 KR 평가에서 중단 판정을 받았다. division Jaccard가 $$0.1879$$에서 $$0.1784$$로 떨어졌고, 기록에는 `stop_kr`가 남았다.

### 4.1 두 환경의 결과가 갈린 이유와 예외

두 환경이 갈린 이유는 기여 분해 행에서 짐작할 수 있다. EO에서 얻은 $$+0.0104$$ 가운데 $$+0.0081$$은 기존 학습 규칙 그대로라도 추론 파이프라인의 후보로 다시 학습한 데서 나왔다. hidden test는 어느 환경과도 정확히 같지 않다. 처음 보는 배아를 all-train backbone으로 처리한다.

나는 실패한 KR 조건에 대해 정확히 이 association head 하나에만 적용되는 예외를 두고 기록에 밝혔다. 이유는 네 가지였다.

1. 제출 head는 이미 199편 전체로 학습돼 있었다. 같은 자료에서 다시 학습할 때는 얻을 것이 적지만, EO에서 재학습으로 얻은 $$+0.0081$$은 처음 보는 배아에서 도움이 될 수 있다고 보았다. 이는 검증할 가설이었으며, KR 결과를 무시해도 된다는 증명은 아니었다.
2. hidden test는 처음 보는 배아여서 EO가 그 축에서 더 가깝다. 다만 평가 배아로 checkpoint를 골랐다는 한계는 남는다. EO에서는 $$+0.01035$$였고, 두 배아 모두 양수, LB90은 $$+0.0061$$이었다.
3. KR은 0 근처였고, KR이 잡아내야 할 v88 같은 붕괴가 아니었다.
4. 다른 배포 전 점검은 모두 그대로 적용했고, 중단 라벨도 기록에 남겼다.

배포 전 예시 영상 4편에서 v93은 $$0.92754$$, v92는 $$0.96151$$로 $$-0.034$$였고, 거의 전부 영상 한 편에서 실제 분열 하나를 놓치고 거짓 분열 하나가 생긴 데서 나왔다. 이 4편은 in-sample이고 분열 사례가 세 건뿐이라 예측력이 약하다. 예외는 이 결과 없이 정한 것이어서, 이 결과를 놓고 다시 결정했고 v93에 한해 유지했다.

### 4.2 Public 점수를 읽은 방식

v93은 9월 15일 제출했고, Public 결과가 나오기 전에 K5와 KR 예외를 근거로 후속 후보의 기준 버전이 됐다. Public $$0.949$$는 이상 징후 기준선 $$0.942$$보다 높았다.

v92와 head만 다른 비교에서 $$+0.004$$가 나왔고 EO와 방향이 같았다. 9월 15일 기록은 이를 C19에 따른 관찰로 남겼다. 로컬 K5 결과와 KR 예외 결정은 점수가 나오기 전의 일이었다.

---

## 5. H1 이후에 던진 질문 열한 개

H1 다음으로 실험 열한 개가 이어졌고, 각각 남은 여지가 어디에 있는지 물었다. 후속 association head는 합산 기준으로 H1보다 $$+0.002$$ 이상 좋아야 했고, 09-17부터는 앞 실험의 결과를 읽은 뒤에야 새 실험을 열었다. 통과한 후보는 없었다. 아홉 개는 결과 전에 적어 둔 기준에서 멈췄고, D-0은 결과를 본 뒤의 진단이었으며, GF0은 설계 단계까지만 갔다.

| 실험 | 질문 | 미리 적어 둔 중단 기준 | 답 |
| --- | --- | --- | --- |
| H2 | 학습을 세 배로 늘리면 나아지나? | H1보다 +0.002 | H1 대비 -0.002339 |
| S0 | 라벨을 이용한 secondary margin 진단이 기준을 넘나? | 진단 이득 ≥ +0.008 | +0.002368 |
| D-0 | H1에서 늘어난 거짓 분열은 어디서 오나? | 없음 (진단) | 노이즈 범위 안 |
| H3 | 역방향 호출용 학습 항을 넣으면 나아지나? | K4 ≥ +0.002 | +0.00096 |
| H4 | 시드 두 개를 평균하면 나아지나? | H1보다 +0.002 | +0.000716 |
| GF0 | H1에 남은 연결 오류는 주로 무엇 때문인가? | 두 배아 모두 구제 건수가 손해 건수보다 많을 것 | 대부분 위치 |
| GO1 | 학습된 위치 보정기가 도움이 되나? | 영상 16편 패널, 두 배아 모두 ≥ 0 | +0.012533, 44b6 -0.001389 |
| A1 | 다음 위치를 예측하는 모델이 placebo보다 나은가? | 모델 − placebo ≥ 0 | KR -0.0486, EO -0.0662 |
| CE1 | H1과 H2를 평균하면 나아지나? | H1보다 +0.002 | -0.000223 |
| D2 | mitosis 이미지 점수로 놓친 분열의 순위를 매길 수 있나? | 두 배아 모두 rank AUC ≥ 0.75 | 0.6922 / 0.5752 |
| GO2 | GO1이 영상 199편 전체에서도 유지되나? | K5 | +0.002100, LB90 -0.000556 |
{: #biohub-table-5 .biohub-table .biohub-records style="--c1: 10%; --c2: 35%; --c3: 27%; --c4: 28%; --table-min: 36rem; --label1: '실험'; --label2: '질문'; --label3: '미리 적어 둔 중단 기준'; --label4: '답'" }

![실험 여섯 개를 embryo-out에서 H1 association head 대비 합산 변화량으로 나타내고 각자의 기준선을 함께 표시한 그림]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-01-lanes-versus-h1.png)
_그림 1. 여섯 실험의 H1 대비 EO 합산 변화량. H3의 $$-0.0087$$은 설명용 전체 EO 결과이며, 표의 $$+0.00096$$은 중단 판정에 쓴 K4 결과로 기준점도 다르다. S0은 기존 결합 규칙 안에서 라벨을 이용한 margin 진단이다. 평가 범위와 기준선이 서로 다른 막대를 같은 검정으로 읽어서는 안 된다._

### 5.1 답들이 좁혀 준 범위

**시험한 association head 확장안에서는 채택할 후보가 없었다.**
학습을 길게 하자(H2) 큰 배아로 학습한 association head는 좋아졌지만 영상 71편짜리 배아로 학습한 association head는 나빠졌다. 시험한 CE1 조합은 H1보다 좋아지지 않았다. 두 association head의 오류가 비슷했을 가능성은 있지만, 이 비교만으로 원인을 분리하지는 못했다. H4는 H3을 중단한 뒤 09-29 마감까지 시간이 남아 있어 하나 더 연 실험이었다.

**시험한 보조 association head 결합에서는 측정된 여지가 작았다.**
S0은 정답 부모를 아는 열 중 primary가 이미 그 부모를 1위로 고른 곳에서 secondary 로짓을 높였다. 결합에는 기존 low-margin consensus 조건을 그대로 썼다. primary와 정렬한 secondary의 1위 부모가 같을 때만 섞는 규칙이다. 따라서 이득은 그 규칙 안에서 정답에 대한 확신을 강화한 결과이며, 새로운 secondary head나 1위를 바꿀 수 있는 다른 결합 방식의 상한은 아니다.

**이번에 시험한 움직임 모델은 도움이 되지 않았다.**
전체 행의 $$94.35\%$$에서 가장 가까운 검출 결과가 이미 실제 후속 노드였다. A1의 사전 학습 점검은 대부분 같은 핵을 다시 찾는 능력을 쟀고, 정작 어려운 행에서 모델은 움직임이 없다고 예측했다.

**진단은 다른 곳을 가리켰다.**
GF0은 예측 위치 대신 주석 위치를 넣어 보았다. 올바른 1순위 선택의 순증가는 6bba에서 $$+282$$, 44b6에서 $$+48$$이었고 대부분 위치에서 나왔다. 이 결과로 위치 보정기 하나를 시도하기로 했다(8절).

---

## 6. 로컬의 "아니다"도 확인할 수 있을까 (v94)

H4는 09-16에 통과 게이트에서 떨어졌다. 기준 $$+0.002$$에 대해 $$+0.000716$$이었다. 로컬의 탈락 판정에도 사각지대가 있는지 확인하려 했다. H4의 작은 이득은 분열에서 나왔는데, 분열은 예전에도 리더보드와 로컬 replay가 엇갈린 부분이다.

게이트에서 떨어졌는데도 배포용 후보를 만들었으므로 예외로 기록했고, 숫자가 나오기 전에 손해가 났을 때의 처리를 정해 두었다. KR 평가 점검이 실패하면 빌드를 멈추고 결정한다는 것이었다. 점검에서는 조건 13개 중 하나가 실패했고(H1 대비 $$0.02$$ 넘게 떨어진 영상 7편, 오른 영상 2편), 합산 $$+0.00039$$는 분열 항에서 나왔다. 진단용으로 한 번 제출했다. 게이트에서 떨어진 v94는 어떤 점수로도 채택될 수 없었지만, 어느 쪽으로든 $$0.003$$ 이상 움직였다면 로컬 기준이 무엇을 놓쳤는지 찾아봤을 것이다.

v94는 $$0.948$$, v93은 $$0.949$$로 동점이었고 로컬 판정과 맞았다. 동점이 두 버전이 같다는 증거는 아니지만, 이번 확인에서 사각지대는 나오지 않았다. 게이트 두 개에서 떨어진 v94는 최종 후보에서 뺐다.

---

## 7. 분열에 남은 여지가 있었나

09-17에 D1으로, H1 association head의 embryo-out 그래프에서 division verifier가 가진 후보 안에서 라벨을 이용해 제약을 풀면 놓친 분열에 얼마나 도달할 수 있는지 물었다.

| oracle | Δ | 분열 TP / FP |
| --- | ---: | --- |
| 재현율을 늘리는 라벨 기반 진단 | +0.024351 | TP 26 → 89, FP 31 → 78 |
{: #biohub-table-6 .biohub-table .biohub-numeric style="--c1: 44%; --c2: 23%; --c3: 33%; --table-min: 0; --label1: 'oracle'; --label2: 'Δ'; --label3: '분열 TP / FP'" }

주석이 달린 분열 $$151$$건 가운데 $$89$$건은 기존 후보로 닿을 수 있고 $$62$$건은 닿을 수 없다. 닿을 수 있는데도 놓친 $$63$$건은 모두 임계값 $$0.90$$에 막혀 있었고, 개수 상한이나 충돌 규칙에 막힌 것은 없었다. 이 진단은 도달 가능한 누락 분열을 보여 준다. FP도 31개에서 78개로 늘었으므로, 완벽한 precision이나 모든 분열 정책의 최고점을 나타내지는 않는다.

D1에서는 임계값 아래에 있던 참 분열 후보가 부모별로는 대개 1위여도 영상 전체 순위의 중앙값은 $$296$$위임을 확인했다. 2절의 재학습이 거짓 양성만 늘렸다는 결과와 함께 보면, 부모별 순위만으로는 충분하지 않았다. 임계값을 낮추기 전에 서로 다른 부모의 후보들을 더 잘 정렬해야 할 이유가 생겼다. $$0.90$$은 보수적인 선택으로 유지했다. v85가 임계값의 효과를 분리해 검증한 것은 아니었다.

D2의 이미지 기반 mitosis 점수는 학습 전에 rank AUC 기준에서 떨어졌다(5절). 이것으로 현재 학습 데이터 규모에서 순위 신호 하나를 접었고, 분열 쪽 실험 전체를 끝내지는 않았다.

D3은 닿을 수 없는 $$62$$건을 처음 실패한 단계별로 분류했다.

![주석이 달린 분열 151건을 복원 26건, 닿을 수 있지만 임계값 아래 63건, 검출 30건, 게이트 25건, 구조적 불가 7건으로 나눈 그림]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-02-division-151.png)
_그림 2. v93의 association head 학습 방식에 해당하는 폴드별 H1 모델의 그래프에서 주석이 달린 분열 151건이 각각 어디서 멈추는지 나눈 그림. 닿을 수 있는데 놓친 $$63$$건은 모두 임계값 $$0.90$$ 아래에 있다. $$+0.024351$$은 FP도 늘어나는 라벨 기반 재현율 진단의 이득이다. 배포 가능한 후보 점수나 전체 상한이 아니다. 폴드별 가중치는 v93에 실린 all-train 가중치와도 다르다._

게이트에 막힌 분열을 살리려면 부모 게이트가 $$12.14$$에서 $$20.75\,\mu\mathrm{m}$$는 되어야 하고, 그중 $$8$$건은 기존 게이트보다 필요한 반경이 $$1\,\mu\mathrm{m}$$ 이내로 컸다. 게이트를 $$13\,\mu\mathrm{m}$$로 넓히면 여덟 건 모두 닿고 산술적 상한은 약 $$+0.0044$$지만, 지금 division verifier는 닿을 수 있는 $$89$$건 중 $$26$$건만 복원하므로 같은 복원 비율을 대입한 거친 예상은 약 $$+0.001$$이었다. 새로 허용할 후보에서도 그 비율이 유지되는지는 검증하지 않았다. 관찰된 누락을 보고 게이트 값을 고르는 것은 임계값에서 이미 거부한 방식이기도 하다. 게이트는 그대로 두었다.

남은 아이디어를 다시 살폈지만, 남은 며칠 안에 만들 수 있고 커널의 실행 시간 예산에 들어가면서 양의 효과까지 측정된 다른 변경은 없었다. 전체 규모로 확인할 후보로는 위치 보정기 하나가 남았다.

---

## 8. 마지막으로 바꿔 넣어 볼 수 있었던 것: 위치 보정기를 199편 전체에서

GO1은 추론 시점에 주석 없이 예측된 노드 위치를 조정한다. 영상 16편 패널에서 $$+0.012533$$을 얻었지만 44b6이 음수여서, 두 배아 모두 0 이상이라는 패널 조건에 따라 미결로 중단됐다. 합산 수치의 절반가량은 분열 하나가 놓침에서 복원으로 바뀐 데서 나왔다.

패널의 중단 조건에는 199편 확인이 없었으므로, GO2는 예외로 기록하고 embryo-out 영상 199편 전체에서 더 엄격한 K5로 확인했다. 그 판정으로 이 실험을 끝내기로 했다.

| 측정값 | 합산 | 44b6 | 6bba |
| --- | ---: | ---: | ---: |
| GO2 − H1, EO 199편 | +0.002100 | +0.000860 | +0.002293 |
{: #biohub-table-7 .biohub-table .biohub-numeric style="--c1: 40%; --c2: 20%; --c3: 20%; --c4: 20%; --table-min: 36rem; --label1: '측정값'; --label2: '합산'; --label3: '44b6'; --label4: '6bba'" }

두 조건이 실패했고(LB90 $$-0.000556$$), 중단 판정을 받아들였다.

GO1 출력을 이미 본 $$20$$편에서는 $$+0.008900$$, 나머지 $$179$$편에서는 $$+0.001328$$로 이득이 $$6.7$$배 작았다. 유망해 보인 사례를 바탕으로 후보를 고를 때 생길 수 있는 편향을 보여 주지만, GO2까지 199편 전체를 열세 번 평가했으므로 어느 쪽도 새로운 배아에서의 효과를 독립적으로 추정한 값은 아니다.

두 번째 예외로 all-train 위치 보정기를 제출 가중치 환경에서 시험했다. 성능 조건은 이미 통과하지 못했지만, 실제 제출용 특징과 가중치로 실행 가능한지 확인하는 작은 시험이었다. 첫 영상에서 노드 $$26{,}356$$개를 모두 출력 한계인 $$7\,\mu\mathrm{m}$$까지 옮겼다. 입력 정규화가 제출용 특징과 다른 embryo-out 특징에 맞춰져 있었기 때문이다. 두 편을 계획한 시험은 첫 편에서 이 문제를 드러냈다. EO에서는 양수였지만 실제 제출 조합에서 무너졌다는 점이 v88과 같았고, 이번에는 노트북을 만들기 전에 잡았다. 더 진행할 후보가 없어 9월 18일 프로젝트 작업을 마쳤다.

---

## 9. association head 재학습의 위험을 나누는 두 제출 고르기

두 제출 중 Private 점수가 더 좋은 쪽이 성적이 되므로, 로컬 근거가 가장 강한 후보와 중요한 위험 하나를 덜어 내는 후보를 함께 두고 싶었다. 슬롯이 둘뿐이므로 둘이 함께 지는 위험도 남는다.

### 9.1 후보별 근거

로컬 수치는 모두 v90, v92, v93으로 이어지는 버전 계열에서 직전 기준 버전과 비교한 변화량이다.

표의 Public 열은 제출 당시의 이상 징후 확인 결과다. 로컬 변화량의 comparator는 둘째 열에 적었다.

| 제출·역할 | 로컬 대조군 | 로컬 근거 | Public |
| --- | --- | --- | --- |
| v93<br>주력 | v92 | EO +0.010347, LB90 +0.0061, K5 통과;<br>KR +0.00003 (중단 판정, 예외로 제출);<br>예시 영상 4편 -0.033968 | 0.949, 이상치 아님 |
| v92<br>hedge | v90 | EO +0.013526, KR +0.026578, 두 환경 모두 두 배아 양수;<br>예시 영상 4편 +0.072033;<br>전체 199편 재추출에서는 음수 없음; 점프가 적은 EO 층에서는 손실;<br>보정 단계의 추가 학습 없음 | 0.945, 이상치 아님 |
| v90<br>다른 hedge | 기준점 | 기준선. 점프가 적은 EO 층의 손실이 hidden test에서도 나타날 경우의 대안 (v92 − v90 -0.003284; 분열 항 손실) | 0.946 |
| v94<br>제외 | v93 / H1 | 관측 이득 +0.000716 (필요 이득 +0.002 미달);<br>KR 평가 조건 하나 실패 | 0.948 |
{: #biohub-table-8 .biohub-table .biohub-records style="--c1: 17%; --c2: 13%; --c3: 53%; --c4: 17%; --table-min: 36rem; --label1: '제출·역할'; --label2: '로컬 대조군'; --label3: '로컬 근거'; --label4: 'Public'" }

### 9.2 위험 축

결정 기록에는 위험 축 세 가지를 적었다. 첫째는 hidden test 배아에서 점프가 얼마나 자주 일어나는가로, v90과 v92·v93을 가른다. 둘째는 registration의 분열 항 부호가 환경에 따라 뒤집히는 문제로, v93이 v92에서 물려받는다. 셋째는 H1 association head가 hidden test에서도 통하는가로, v93을 나머지 둘과 가른다.

![위험 축 세 가지와 v90, v92, v93을 행렬로 놓고 각 위험이 어느 후보에 불리한지 표시한 그림]({{ site.baseurl }}/assets/img/posts/2026-09-18-biohub-working-note-8/fig-03-risk-axes.png)
_그림 3. 결정 기록의 위험 축별 노출 여부다. 각 조건에서 반드시 손실이 난다는 뜻은 아니다. 점프가 드물면 registration의 이득이 작아질 수 있지만, 점프가 없는 집단의 점수는 음수가 아니었다. 고른 두 후보는 registration의 두 위험을 함께 안고, head 재학습 위험은 v93에만 남는다._

v92는 기존 association head와 registration을 유지한다. 따라서 fold별 H1에서 얻은 이득이 all-train H1으로 옮겨 가지 못할 위험을 피할 수 있다. v93의 KR 결과는 중립이었고 노트북 예시 점수는 낮아졌다. 세 후보에 공통으로 남은 EO의 한계는 0절에서 설명했다.

v90을 함께 골랐다면 한쪽에서 registration도 빠지므로 위험 축 세 가지를 모두 나눌 수 있었다. 나는 두 평가 환경과 두 배아 모두에서 얻은 전체 패널의 registration 이득에 더 무게를 두고 v92를 골랐다. 그 대신 두 제출은 점프가 드문 hidden 집단과 설명되지 않은 분열 손실에 함께 노출됐다.

### 9.3 최종 선택 문서가 실제로 사용한 근거

9월 18일 문서는 근거표 아래에 “Public은 붕괴 탐지 전용”이며 이 열로 순위를 매기지 않는다고 적었다. 그런데 슬롯 A는 “v93 (사실상 자동)”이라고 부르고, 장점도 Public 최고점 $$0.949$$, K5 통과, v92와 같은 검출 결과, 후속 열한 실험에서 더 나은 채택 후보가 나오지 않았다는 순서로 나열했다. C19와의 충돌은 결정 문서 안에 이미 있었다.

문서에는 로컬 근거로 v93을 남길 이유도 있었다. 199편의 양의 EO 결과에, in-sample인 KR의 중립 결과와 예시 네 편의 손실보다 큰 비중을 두는 판단이다. 처음 보는 배아에 어느 평가가 더 가까운지 판단한 것이지만, 배아는 두 개뿐이었고 checkpoint 선택과 후속 학습의 의존 관계도 남았다. v93을 두 후보 중 하나로 남길 근거는 있었어도 자동으로 정해지는 선택은 아니었다. Private에서 v92와 비교하면 head 재학습 효과의 방향을 확인할 수 있을 것이다.

나머지 슬롯 결정에는 로컬 게이트와 후보별 위험을 적용했다. **v94는 $$0.948$$이어도 advance와 KR 게이트를 넘지 못해 제외했다.** **v92는 $$0.945$$이어도 registration을 유지하면서 head 재학습 위험을 덜기 위해 남겼다.**

### 9.4 Private이 확인할 것: 결과 전에 적어 둔 예측

이 글을 쓰는 지금 Private 결과는 알 수 없다. Private에서 확인할 것은 다음과 같다.

- **두 제출 중 하나라도 무너지는가.** 그러면 공유한 registration 위험, head의 전이, 로컬에서 고려하지 못한 hidden test 조건을 조사해야 한다. 점수만으로 원인을 가를 수는 없다.
- **v93 − v92의 부호.** EO는 약 $$+0.010$$, KR은 동점을 예측한다. $$\pm 0.002$$를 넘는 양수라면 association head의 EO 이득이 전이되었다는 해석과 맞는다. KR이 중립이었던 이유까지 증명하지는 않는다. 동점이면 두 해석을 가를 수 없고, 예외는 아무 뒷받침도 얻지 못한다. 프로젝트의 동점 범위를 벗어난 음수라면 Private 부분집합에서 association head 교체가 불리했고, v92를 남긴 선택이 도움이 됐다는 뜻이다. 원인까지 특정하는 결과는 아니다.
- **v92가 v90보다 높은가.** 두 로컬 환경 모두 전체 영상 기준으로 그렇게 예측한다. 그렇지 않다면 점프 빈도의 차이나 분열 손실 등이 가능한 설명이다. 두 제출이 공유하는 위험이다.

---

## 마무리

v93과 v92를 선택했다. H1은 embryo-out 평가에서 K5의 다섯 조건을 통과했지만 KR 예외가 필요했다. registration은 두 로컬 환경의 전체 패널에서 이득이었지만, 분열 항의 부호 차이는 설명하지 못했다. 아직 Private 결과는 모른다. 결과가 나오면 먼저 붕괴 여부, 다음으로 v93 − v92의 부호, 마지막으로 v90과의 비교를 통해 두 선택이 공유한 위험을 살펴볼 것이다.

<details markdown="1">
<summary>판단 기록·기준의 변화·남은 질문</summary>

아래 표는 당시의 결정과 그 근거를 정리한 것이다. 각 조항은 근거가 뒷받침하는 범위로 읽으며, 앞선 모든 실험이 해당 조건을 충족했다는 뜻은 아니다.

## 10. 판단 기록

09-13부터 EO로 효과를 판정하고 KR로 제출 구성의 손해를 확인했으며, v91부터 v94까지 네 번의 Public 결과에서는 이상 징후가 없었다. v93, v94, GO2, GO2의 배포 시험은 기록한 예외로 중단 기준을 넘어 진행했고, 원래 판정과 예외 사유를 함께 남겼다.

| 결정 | 당시의 이유 | 결과 | 바뀐 것 |
| --- | --- | --- | --- |
| 09-13: 리더보드를 이상치 감지용으로 좁힘 | 예전 질문에는 이제 로컬 도구가 답함; ± 0.002 차이로 후보 순서를 정하지 않음 | 네 번 확인, 이상치 없음 | C19 |
| 09-13: division verifier 재학습을 영상 199편으로 판정 | 영상 4편의 이득이 사례 하나에서 나옴 | -0.00679; FP 37 → 196 | C18 |
| 09-13: v92를 sanity check로 제출 | 그때까지 가장 강한 로컬 근거 | 0.945, 이상치 아님 | v92는 로컬 근거로 판단 |
| 09-15: `stop_kr`에 대한 예외를 밝히고 v93 제출 | KR이 새로운 배아에서의 재학습 이득에 덜 민감할 수 있음; EO +0.01035, LB90 +0.0061; KR은 붕괴가 아님 | 0.949, 이상치 아님 | K5를 근거로 v93을 parent로 |
| 09-15 ~ 09-18: 실험 열한 개 | 같은 게이트에서 H1을 넘는 아이디어가 있는지 확인 | 통과한 것 없음 | 추가로 채택할 후보 없음 |
| 09-16: v94를 진단용으로 제출 | 탈락 판정의 사각지대 확인 | 0.948, v93과 동점 | v94 제외 |
| 09-17: 예외로 GO2 시작 | 영상 16편 패널로는 결론을 낼 수 없음 | +0.002100; 처음 보는 영상 +0.001328 | 중단 판정 수용 |
| 09-18: 최종 제출 v93, v92 | v93: Public 최고점·K5 통과·검출 결과 유지; v92: head 재학습 위험에 대한 hedge | Private 미정 | C20 |
{: #biohub-table-9 .biohub-table .biohub-records style="--c1: 25%; --c2: 25%; --c3: 28%; --c4: 22%; --table-min: 36rem; --label1: '결정'; --label2: '당시의 이유'; --label3: '결과'; --label4: '바뀐 것'" }

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
| C9 | Public은 기대치를 미리 적어 둔 sanity check에만 쓰고, 인접한 설정 중 하나를 고르는 데는 쓰지 않는다; 09-13부터 C19로 좁혀짐 | 4편(08-10) |
| C10 | 후보의 도달 범위를 잰다. 상한이라는 말은 선언한 범위를 포괄할 때만 쓴다 | 5편 |
| C11 | 게이트는 결론을 낼 수 있어야 한다 | 5편 |
| C12 | 로컬 검증은 실제 제출 파이프라인을 그대로 재현해서 한다 | 6편 |
| C13 | 단계 제거로 전체 효과를 잰다. metric 항을 분리하려면 추가 가정을 밝힌다 | 6편 |
| C14 | kernel regime(실제 제출 가중치·코드)에서 검증하고, 제출 전에는 노트북 출력 자체를 이전 버전과 비교해 채점한다 | 7편 |
| C15 | 큰 이득을 믿기 전에 누수이 없는 대조 실험부터 돌린다 | 7편 |
| C16 | 한 제출에서는 한 가지만 바꾸거나, 조건을 맞춘 대조군을 함께 둔다 | 7편 |
| C17 | 라벨은 채점 기준을 따른다. 라벨러의 판정부터 정답과 대조한다 | 7편 |
| C18 **(신규)** | 작은 패널은 가능성만 확인하고, 채택은 영상 199편 전체로 결정한다 | 8편 |
| C19 **(신규)** | 선택은 embryo-out(EO)의 K5로 한다. KR은 손해를 감지하고, Public은 이상치만 감지한다 | 8편(09-13) |
| C20 **(신규)** | 최종 제출은 로컬 근거가 가장 강한 후보에, 다른 방식으로 실패하는 hedge 하나를 더한다 | 8편 |
{: #biohub-table-10 .biohub-table .biohub-records style="--c1: 14%; --c2: 72%; --c3: 14%; --table-min: 0; --label1: '조항'; --label2: '내용'; --label3: '도입'" }

---

## 11. 이 기간에 확인된 것

### 확인된 사실

1. 예전 v87/ShiftCorr replay에서 점수가 낮게 나온 참 분열 $$56$$건 중 $$46$$건은 같은 부모 안에서 이미 1위였다. 유효한 라벨로 division verifier를 다시 학습하면 영상 4편에서는 $$+0.01099$$, 199편에서는 $$-0.00679$$였다.
2. 라벨 없이 하는 stage-jitter registration은 KR 평가에서 $$+0.026578$$, EO에서 $$+0.013526$$이었고, 두 환경 모두 두 배아가 양수였다.
3. H1 association head 재학습은 EO에서 $$+0.010347$$, KR 평가에서 $$+0.00003$$이었다.
4. H1 이후 실험 열한 개 가운데 통과한 후보는 없었다.
5. H1의 embryo-out 그래프에서 주석이 달린 분열 $$151$$건은 복원 $$26$$건, $$0.90$$ 아래 $$63$$건, 앞 단계에서 놓친 것 $$30$$건, 게이트에 막힌 것 $$25$$건, 구조적으로 불가능한 것 $$7$$건으로 나뉜다. 라벨을 이용한 재현율 진단은 FP 증가와 함께 $$+0.024351$$을 얻었다.
6. GO2는 출력을 이미 본 영상 $$20$$편에서 $$+0.008900$$, 처음 보는 영상 $$179$$편에서 $$+0.001328$$로 $$6.7$$배 차이가 났다.
7. 위치 보정기는 KR 평가에서 돌려 본 영상의 모든 노드를 출력 한계까지 밀어냈다. 다른 특징으로 맞춘 normalizer와 들어맞는 결과다.

### 근거는 있지만 아직 확정하지 못한 판단

1. head 재학습은 in-sample보다 처음 보는 배아에서 더 도움이 될 수 있다. 기여 분해 비교는 이 가설을 뒷받침한다.
2. 두 제출이 공유하는 위험이 점프 축과 분열 부호 축에 있다는 판단.
3. 다른 유효 후보를 얻으려면 더 많은 시간이나 다른 탐색이 필요했을 수 있다.

### 열린 질문

1. Private에서 v93 − v92는 EO의 부호($$+0.0103$$)를 따를까, KR의 부호(약 $$0$$)를 따를까?
2. 최종 제출 중 하나라도 Private에서 무너질까?
3. hidden test 배아에서는 점프가 얼마나 자주 일어나고, v90이 v92보다 높게 끝날까?
4. 분열 순위를 개선해서 $$+0.024351$$ 진단이 드러낸 누락 분열을 복원할 신호가 있을까?
5. 같은 모집단을 열세 번 측정한 탓에, 살아남은 실험의 수치는 얼마나 부풀려졌을까?

</details>

시리즈:

- [1편: Lineage Graph 학습과 평가지표에 맞춘 후처리]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/)
- [2편: Public 점수가 멈췄을 때 — OOF 기반 오류 분석]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/)
- [3편: OOF에 기반한 판단들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-3-What-the-OOF-Machine-Refused-KR/)
- [4편: 로컬 검증에서 발견한 세 가지 빈틈]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-4-Why-the-Largest-Local-Gain-Did-Not-Show-on-the-Board-KR/)
- [5편: 고정된 그래프가 시험하지 못한 것들]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)
- [6편: 로컬 검증이 제출 파이프라인과 달랐던 문제]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline-KR/)
- [7편: 같은 코드로도 검증이 어긋나는 이유]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce-KR/)
- **8편: 최종 제출을 고를 때 고민한 것들**
