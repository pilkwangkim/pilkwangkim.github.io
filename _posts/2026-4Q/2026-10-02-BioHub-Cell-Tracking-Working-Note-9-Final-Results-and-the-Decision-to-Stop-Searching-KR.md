---
title: "BioHub Cell Tracking 작업 기록 9: 큰 계획에 비해 허술했던 시작"
date: 2026-10-02 19:00:00 +0900
last_modified_at: 2026-10-03
categories: [AI, Kaggle]
tags: [kaggle, biohub, ai-agents, cell-tracking, competition-retrospective, lineage-reconstruction, microscopy, model-portfolio, oof, research-methods, working-note]
lang: ko
slug: BioHub-Cell-Tracking-Working-Note-9-Final-Results-and-the-Decision-to-Stop-Searching-KR
math: true
pin: false
hide: false
published: true
image:
  path: /assets/img/posts/2026-10-02-biohub-working-note-9/cover.png?v=a3ef7423b8db
  alt: "BioHub Cell Tracking 작업 기록 9: 큰 계획에 비해 허술했던 시작"
topic: biohub
translation_key: biohub-09
series: biohub
series_order: 9
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

.content .biohub-figure { display: block; max-width: 100%; border: 0; }
.content picture, .content picture img { display: block; max-width: 100%; }
.content picture img { width: 100%; height: auto; }
.content mjx-container { max-width: 100%; overflow-x: auto; overflow-y: hidden; }
main h1, .content h2, .content h3, .content h4 { word-break: keep-all; overflow-wrap: break-word; }
.content table.biohub-score th:first-child, .content table.biohub-score td:first-child {
  position: sticky; left: 0; z-index: 1;
  background: var(--main-bg, #fff); white-space: nowrap;
}
.content table.biohub-score td:nth-child(3), .content table.biohub-score td:nth-child(4) {
  white-space: nowrap;
}
@media (max-width: 620px) { .content .biohub-fig-03-component-and-composition { aspect-ratio: 760 / 2070; } }
@media (max-width: 620px) { .content .biohub-fig-02-research-verdicts { aspect-ratio: 760 / 1764; } }
@media (max-width: 620px) { .content .biohub-fig-01-private-submissions { aspect-ratio: 760 / 1030; } }

</style>

[이전 글: 작업 기록 8]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two-KR/) · [English]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-9-Final-Results-and-the-Decision-to-Stop-Searching/)

> **시리즈 안내.** BioHub는 3D microscopy 영상에서 cell lineage graph를 복원하는 대회다. 학습 데이터는 두 embryo의 199개 영상이고, 학습에 없던 embryo로 구성된 hidden test의 29%가 Public, 나머지 71%가 Private에 쓰였다. 1–8편에서는 대회 중에 내가 내린 판단을 기록했다. 이 글에서는 2026년 10월 2일 확인한 최종 결과와 종료 후 공개된 해법을 바탕으로 내 연구 과정을 돌아본다.
{: .prompt-info }

BioHub를 시작할 때는 계획이 컸다. 3D microscopy 데이터를 공부하고, 쓸 만한 detection·association 모델을 학습해 성능이 훨씬 좋은 cell tracking 시스템을 만들고 싶었다. GPU를 한 달 넘게 대여했고, 아이디어를 구현하고 조사하는 데 AI agent의 도움을 받았다. 결과는 기대했던 성능에 한참 못 미쳤다.

가장 아쉬운 것은 확보한 시간과 GPU를 제대로 활용하지 못했다는 점이다. 마감까지 2–3주가 남아 있었는데도, 다음에 어떤 실험을 해야 할지 감을 잡지 못할 때가 많았다. GPU를 비워두거나, 모델을 고르고 예측을 개선하는 데 도움이 되지 않는 계산에 시간을 썼다. 작업은 계속되고 있었지만, 그 작업이 훨씬 나은 시스템으로 이어질 만한 경로는 잘 보이지 않았다.

문제는 더 일찍 시작됐다. 데이터에서 예측을 만들고 채점까지 할 수 있는 안정적인 경로를 갖추기 전에 코드 수리에 너무 오래 매달렸다. 하나의 pipeline을 계속 개선하는 동안, 대안을 실제로 개발하고 비교하는 작업을 꾸준히 이어 갈 기반은 갖추지 못했다. 모델과 calibration, graph 규칙이 서로 맞물릴수록 초기 가정 하나를 바꾸려면 다른 단계들도 고쳐야 했다. 그 관계를 알아차렸을 때는 이미 방향을 바꾸는 비용이 커져 있었다.

이번 회고에서는 이 흐름을 따라가려 한다. **처음에 제대로 갖추지 못한 기반이 어떻게 나중의 연구 방향을 바꾸기 어렵게 만들었는지, 다음에는 무엇을 더 일찍 알아차려야 할지** 살펴보고 싶다. 앞선 여덟 편에는 프로젝트가 진행되는 동안의 판단을 기록했다. 이 글에서는 그 기록을 대회 종료 후 공개된 해법과 함께 다시 읽는다. 당시 알 수 있었던 것과 나중에 알게 된 것은 구분하면서 돌아보려 한다.

## 1. 기대했던 성능과 실제 결과의 차이

10월 2일 공식 계정 조회에서 제출 175건을 확인했다. 그중 COMPLETE 상태이면서 Private 점수가 있는 제출은 162건이었다. 나는 v93과 v92를 선택했고, v93의 **0.91920**으로 **445위**를 기록했다. 처음 목표했던 성능과는 한참 거리가 있었다.

도움이 된 변경도 있었다. Private에서 v93은 v92보다, v92는 v90보다 높았다. 실제 hidden 평가로 이어진 개선을 만들어 낸 것이다. 돌아볼 것은 그런 개선을 하고도 왜 전체 시스템의 성능은 여전히 기대에 못 미쳤느냐다.

조회된 제출 중 Private 점수가 가장 높았던 것은 v94였다. 이를 선택했다면 최종 결과는 조금 좋아졌겠지만, 원하는 성능과의 큰 차이는 남았을 것이다. 정확한 비교는 부록에 두었다. 이미 만든 후보 안에서 선택으로 바꿀 수 있었던 작은 차이를 설명하는 계산이지, 훨씬 강한 시스템을 만들지 못한 이유를 설명하는 것은 아니다.

그보다 큰 차이를 이해하려면, 아이디어를 실행 가능한 실험으로 바꾼 과정부터 살펴봐야 한다.

## 2. 전체 실행을 확인하기 전에 너무 많은 것을 고치려 했다

초기에는 마주치는 버그를 전부 고치려고 agent의 도움을 받았다. 며칠 동안 수리를 반복하고 나서야 코드 전체를 온전하게 만들려는 작업에는 끝이 없겠다는 생각이 들었다. 초반 열흘이 넘는 기간 동안 이런 수리에 시간과 token을 쏟았던 것으로 기억한다. 남은 기록에서 구체적인 중단 사례는 확인할 수 있지만, 전체 token 사용량을 복원하거나 그 기간 내내 생산적인 일이 없었다고 입증할 수는 없다.

7월 1일부터 7월 10일 사이의 RunPod 로그에는 데이터와 파일 준비에 필요한 항목이 빠진 문제, 압축 해제를 돕는 코드의 누락, DataLoader의 mask shape 오류가 남아 있다. 실제 Center trainer가 launcher의 옵션을 받지 못했고, 실행 모드와 checkpoint 이름도 맞지 않았다. 단계 사이의 인터페이스가 어긋난 구체적인 문제들이었다. 원격에 옮긴 trainer가 다른 인자를 기대한다면 launcher만 고쳐서는 소용이 없었다. Trainer를 고쳤다고 inference가 의도한 checkpoint를 읽는지까지 확인된 것도 아니었다.

그 와중에도 실제 학습은 진행됐다. 6월 30일에는 이미 로컬 채점이 가능했고, TemporalUNet 학습은 7월 2일에 시작해 이후 며칠 동안에도 상당히 진행됐다. 문제는 코드 작업과 모델 학습을 함께 점검할 기준이 되는 전체 실행 경로 하나를 갖추지 못했다는 데 있었다.

### 모든 코드를 고치기 전에 단계 사이의 연결부터 확인해야 했다

Software engineering 경험이 거의 없어, 처음에는 파일 사이에 얼마나 많은 가정이 오가는지 알아채지 못했다. Tensor의 축, sampling 방식, command-line 인자, checkpoint 구조, 좌표 변환, graph 저장 형식이 모두 그런 가정이었다. 오류 하나를 고쳐도 다음 단계가 그 변경과 맞는지는 모를 수 있었다. 로컬 test가 통과하면 안심했지만, 그 test가 확인한 범위는 실제 실험에 필요한 범위보다 좁을 때가 많았다.

여기서 배운 것은 먼저 가장 작은 전체 결과를 정의해야 한다는 점이다. 이 프로젝트라면 실제 training batch를 읽어 shape와 mask를 확인하고, 의도한 checkpoint를 저장했다가 학습을 재개해 보는 것부터 시작할 수 있었다. 이어서 실제 영상 하나를 예측하고, graph를 저장한 뒤 다시 읽어 공식 scorer로 채점한다. 두 번째 영상도 실행해 첫 영상에만 우연히 맞는 경로가 아닌지 확인한다. 기록에는 이 순서의 일부가 이미 있었다. 병렬 launcher와 범용 실행 코드를 늘리기 전에 그 부분들을 하나로 연결했어야 했다.

그다음부터는 수리할 때마다 구체적인 질문을 할 수 있었다. 다음 비교를 실행하지 못하게 막는 것은 무엇인가? Agent에게 그 연결을 고치고 출력을 실제로 쓰는 단계까지 보여 달라고 요청하면 됐다. 정상적으로 진행 중인 학습은 그대로 두면서 할 수 있는 작업이었다. 새 모델에 adapter가 필요할 수는 있다. 그러나 유용한 예측 하나를 만들기도 전에 기존의 모든 편의 기능과 일반 검사를 갖출 필요는 없었다.

8월 20일 상태 기록에서는 같은 문제가 더 선명하게 보인다. 검증과 실행 관리 코드를 고치는 동안 **계획한 CPU 연구 작업 331개가 하나도 시작되지 않았다.** 한 verifier는 **57분 13.424초** 동안 약 **696 GB**를 다시 읽었다. 뒤에는 mock에 실제 process의 시작 메시지가 없거나, 파일을 옮긴 뒤 시간 정보가 예상과 달라지는 문제가 나왔다. 과거 기록의 필드는 `bytes`였는데 verifier는 `size`를 기대하기도 했다. 실제 결함들이었지만, 이 연결된 문제들을 모두 해결하는 일이 연구 작업을 시작하기 위한 선행 조건이 돼 버렸다.

이 과정을 거치며 유용한 engineering 습관도 배웠다. 어떤 코드가 출력을 만들고 어떤 코드가 그것을 쓰는지 정확히 따라가고, 실제 입력으로 단계 사이의 연결을 확인하고, 작동하는 checkpoint를 보존하는 것이다. 실행 오류를 고치는 일과 연구 질문을 바꾸는 일도 구분해야 했다. Engineering 지식이 거의 없던 내게는 가치 있는 배움이었다. 다만 배우는 방식에도 한계가 필요했다. 커져 가는 시스템을 완벽하게 만든 뒤에 연구하려 하기보다, 다음 실험을 실행하고 해석하는 데 필요한 지식부터 갖췄어야 했다.

### 로컬 점수가 무엇을 비교하는지도 확인해야 했다

실행이 된다고 충분한 것은 아니다. 평가하는 pipeline이 실제 배포 경로와 다를 수 있다. [6편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline-KR/)에서 다룬 9월 2일 수정 후, 199편의 embryo-out replay 점수는 **0.6005에서 0.7499**로 바뀌었다. 두 실행 모두 embryo-out backbone asset을 썼고, 0.7499는 notebook이 hidden input에서 직접 낸 점수가 아니었다. 새 모델의 개선이 아니라 비교 도구를 바로잡은 결과였다.

전처리, 학습된 모델의 역할, proposal 선택, graph 처리 순서, 저장한 출력이 서로 맞는지 더 일찍 확인했어야 했다. 그렇지 않으면 로컬 점수를 아무리 꼼꼼히 계산해도 실제 배포에 관해 궁금했던 것과는 다른 질문에 답할 수 있다. 내가 처음에 갖추지 못한 기반에는 두 가지가 있었다. 안정적으로 실행되는 경로, 그리고 그 경로로 무엇을 비교하고 있는지에 대한 이해였다.

## 3. 모델을 더 고르기 전에 무엇을 학습할지 정해야 했다

Baseline이 작동한 뒤에는 이를 개선하는 데 많은 노력을 썼다. 출발점으로는 합리적이었다. 하지만 그것만으로 다음 큰 개선이 어디서 나올지는 정해지지 않았다. 시스템에 어떤 정보가 빠져 있고, 어떤 모델이 그 정보를 배울지 더 분명하게 설명할 수 있어야 했다.

BioHub에서는 구체적으로 물을 수 있었다. Detector는 밀집한 volume에서 희미한 세포핵의 위치를 잘 찾는가? Representation은 가까운 세포의 identity를 구분하는가, 주로 세포와 배경을 구분하는가? Sparse annotation 때문에 표기하지 않은 실제 세포가 학습의 negative가 되는가? Division crop에는 mother와 두 daughter가 모두 들어 있는가? 뒤의 graph 처리가 앞에서 만든 유용한 정보를 지워 버릴 수 있는가?

이 질문들이 도메인 지식과 모델 설계를 연결한다. Architecture 이름을 많이 아는 것만으로 충분하지 않은 이유도 여기에 있다. 서로 다른 backbone이 같은 쉬운 과제를 배우고, 정작 중요한 오류는 함께 낼 수 있다. Temporal target이 적절한 작은 모델은 성능이 좋은 세포·배경 detector가 한 번도 배우지 않은 정보를 줄 수도 있다.

### 생물학과 microscopy는 어디까지 공부했어야 할까

예측을 시작하기 전에 발생생물학을 통달할 필요는 없었다고 생각한다. 다만 학습 예제를 정의하고, 애매한 label을 알아보고, 실패를 해석할 만큼은 이해해야 했다. 이 대회에서는 축마다 다른 voxel 간격, XY와 XZ view에서 세포가 보이는 방식, 깊이 방향의 가림, division의 시점과 모습, stage motion이 그런 지식에 포함됐다. Sparse annotation에서 표기되지 않은 세포에 대해 알 수 있는 것과 알 수 없는 것도 이해해야 했다.

다음에는 작은 오류 사례 묶음을 의도적으로 골라 그 지식을 배우고 싶다. 올바른 detection, 희미해서 놓친 세포, 밀집한 이웃 사이의 identity swap, 실제 division, false fork, 깊이 때문에 판단이 애매한 사례를 하나씩 살펴보는 것이다. 각 사례의 원본 화면, annotation, 모델 입력과 예측을 나란히 놓는다. Target을 설명하지 못하거나 불확실한 판정과 오류를 구분하지 못한다면, 모델을 설계하기 전에 관련 도메인 자료를 읽거나 annotation 기준을 확인해야 한다.

도메인 공부를 어디까지 할지도 이렇게 정할 수 있다. Target, crop, negative label, augmentation, 오류의 해석을 바꿀 수 있다면 그 부분을 더 공부한다. 그것들이 서로 맞게 정리되면 representation, loss balance, 초기화, sampling, optimization, calibration처럼 machine learning의 문제에 집중할 수 있다. 배울 수 있어야 하는 예제에서도 이 방법들이 실패한다면, 더 큰 backbone만 찾기보다 도메인에 대한 가정을 다시 살펴본다.

우승자가 나중에 쓴 글은 이 연결을 선명하게 보여 준다. 이미지를 관찰해 annotation을 늘렸을 뿐 아니라, 다른 temporal 학습 과제를 발견했다. 그 recipe를 미리 알 수는 없었다. 하지만 내가 직접 이미지를 본 경험도 모델에게 어떤 답을 요구할지에 영향을 줘야 한다는 점은 알아차릴 수 있었다.

### 초기 portfolio에서는 서로 다른 질문을 계속 개발할 기회를 확보해야 했다

믿을 만한 baseline 하나와 구체적으로 부족한 정보를 다루는 대안 하나로 시작할 수 있었다. 중요한 오류를 골라 조사한 결과로 다음 투자 방향을 정하는 것이다. Agent의 도움을 받으며 혼자 참가하는 상황에서도 감당할 수 있는 방식이다. 처음부터 모든 모델 계열·fold·seed를 학습할 필요는 없다.

당시에도 portfolio와 모델 간 상호작용을 생각했다. 7월 15일 구조 개선 roadmap은 anchor에 최적화된 후처리로 blend를 시험하면, blend에 맞는 최선의 구성을 공정하게 평가한 것이 아니라고 지적했다. 8월 3일에는 association과 division을 다르게 결합한 여러 구성을 실제로 제출했다. 원칙은 있었다. 그러나 다른 representation에 충분한 학습 시간과 그 출력에 맞는 예측 경로를 마련하는 일정으로 꾸준히 이어 가지 못했다.

이 계획과 실행 사이의 차이가 이번 회고의 중심이다. Baseline을 계속 개선하면서도 대안은 실제로 배우고 있는지, 그 출력을 쓸 단계는 준비됐는지, 어떤 측정 결과가 나오면 다음 투자를 할지 물었어야 했다. 그 질문이 없다면 큰 계획도 결국 이미 실행할 줄 아는 시스템을 조금씩 바꾸는 작업의 연속으로 끝날 수 있다.

## 4. 강한 모델은 필요한 판단을 잘하는 모델이어야 했다

대안에 시간을 더 주기 전에, 조사할 가치가 있는 학습 과제와 이름이나 지표 하나가 좋아 보일 뿐인 모델을 구분해야 했다.

| 역할 | 요구했어야 할 근거 | 그것만으로는 부족한 근거 |
|---|---|---|
| Detector | 밀도·밝기·움직임이 다른 영상에서 표기된 세포를 찾고 위치를 맞추는 성능, 그 proposal이 최종 graph에 미친 영향 | Peak 수 증가, training loss 감소, 유명한 architecture |
| Appearance 또는 association 모델 | 실제로 경쟁하는 parent나 successor의 구분, 특히 가까운 어려운 후보의 구분과 최종 link 개선 | 쉬운 pair에서 높은 AUC, 세포와 배경을 구분하도록만 학습한 feature |
| Division 모델 | 실제 운영 proposal에서 threshold 근처의 유용한 순위 판단, graph 구성 후에도 남는 근거 있는 mother–daughter 선택 | GT 중심 crop을 잘 분류한 결과만 있는 경우 |
| Ensemble 구성원 | 감당할 수 있는 전체 실행 시간 안에서 고정한 조합에 실제로 기여한 결과 | Seed가 다르다는 사실, 낮은 오류 상관관계, incumbent에 가까운 단독 점수 |
{: #biohub-table-2 .biohub-table .biohub-records style="--c1: 22%; --c2: 41%; --c3: 37%; --label2: '요구했어야 할 근거'; --label3: '그것만으로는 부족한 근거';" }
학습이 얼마나 진행됐는지도 이 판단의 일부다. 점수가 낮았다고 모델을 공정하게 시험한 것으로 보기 전에, sampler가 어떤 예제를 보여 줬는지 알아야 했다. 관련 positive와 구분하기 어려운 경쟁 후보가 있었는가? Learning curve는 계속 좋아지고 있었는가? 조건을 통제한 작은 학습 문제는 맞출 수 있었는가? Inner-development 예측에서도 실제 배포에 필요한 위치 추정이나 순위 판단을 평가해야 했다.

Epoch 수만으로는 이 질문들에 답할 수 없다. Loader가 한 sampling 주기를 어떻게 정하느냐에 따라 의미가 달라지고, 오래 돌려도 쉬운 예제만 반복해서 볼 수 있다. 전체 AUC가 높다고 드물게 나타나는 운영 threshold 근처의 proposal까지 유용하게 구분한다는 보장도 없다.

기록에는 너무 이른 판단과 실제로 낮은 성능을 확인한 결과가 모두 있다. CandidatePU G3의 GPU 학습은 약 **13.4분** 걸렸는데, 채택 가능 여부를 확인하고 정확히 replay하는 데에는 약 **1.89시간**이 걸렸다. 이 시간만으로 학습 부족을 입증할 수는 없다. 다만 첫 설계를 대규모로 replay하기 전에, 모델의 용량과 예제 노출, 학습 종료 시점을 더 저렴하게 비교할 필요가 있었다는 점은 보여 준다. Best checkpoint가 epoch 48이라는 이유만으로 CandidatePU를 학습 부족으로 본 이전 주장은 철회됐다. 해당 실행은 **fold마다 57,600 step**을 수행했다.

Spotiflow의 Exposure256 실행은 약 **599.9초**였다. Detector와 candidate coverage 조건을 충족하지 못해 held-out graph 채점까지 가지 못했고, association head도 fit하지 않았다. 이 학습 조건으로는 의도한 비교를 할 준비가 되지 않았다고 판단할 수 있었다. 충분히 개발한 모델 계열이 전체 graph 비교에서 졌다는 결과로 볼 수는 없었다.

Teacher 노출 문제를 정리한 pseudo A2 비교는 달랐다. Control과 student는 각각 자체 detection과 association을 사용했고, graph 설정은 이전에 선택한 것을 썼다. Student가 더 많은 node를 찾았는데도 배아를 바꿔가며 평가한 네 영상의 graph 점수는 control 대비 **−0.0205909080**이었다. 그 composition을 제외할 실제 근거가 있었다. 같은 설정을 자동으로 199편 전체 평가로 확대해도 새로운 질문에 답하는 것은 아니었다.

유망한 모델에는 얼마나 시간을 줬어야 할까? 미리 정한 학습 목표에 도달해 판단할 만한 결과를 얻도록 투자한 뒤, 그 결과로 다시 결정했어야 한다. 첫 실행에서는 실제 예제 노출과 처리 속도, target을 배울 수 있는지 확인한다. 유용한 정보를 배웠다면 그 출력에 맞게 inference를 구현하고, 작은 규모로 전체 pipeline을 비교한다. 배우지 못한 모델에는 원인을 진단해 학습 데이터·조건이나 목표에서 변경 하나를 정하거나, 구체적인 이유와 함께 보류한다. 충분히 개발한 composition이 졌다면 실제 시험한 범위에서 제외한다.

이 순서가 중요한 이유는 성적이 더 좋은 여러 해법이 내 공개 가중치를 출발점으로 썼기 때문이다. Detector가 쓸모없었던 것은 아니다. 어떤 identity나 event 정보가 부족한지 이해하고, 그 정보를 배우는 모델과 출력을 활용할 단계를 개발해야 했다. 여기서 프로젝트 내내 판단을 어렵게 했던 비교 문제로 이어진다.

## 5. Component를 붙이는 실험과 다른 pipeline을 개발하는 실험은 달랐다

Detector, association 모델, graph 규칙을 줄여서 A+B+C라고 표현해 왔다. 여기서 더하기 기호는 이 component들을 쓰는 pipeline이라는 뜻이다. 수를 더하는 식도, 각 component의 점수 개선을 합산할 수 있다는 뜻도 아니다. 관계를 분명히 하려고 기존 pipeline을 **P₀ = (A₀, B₀, C₀)**라고 하자. 운영 설정 θ₀는 이 pipeline에 맞춰 정한 것으로, proposal gate, confidence threshold, graph cost, repair 규칙, blend 가중치 등을 포함한다.

**D**는 추가하려는 정보나 방법 하나라고 하자. D의 학습 recipe와 새로 필요한 설정은 미리 고정한다. 기존 pipeline과 공유하는 운영 설정 θ₀를 유지하면서 P₀에 D를 붙인 구성을 **P_D**라고 하면,

$$
\Delta_D(\theta_0)=S(P_D;\theta_0)-S(P_0;\theta_0)
$$

이 비교는 기존 설정을 그대로 둔 pipeline에 D를 추가했을 때 도움이 되는지 묻는다. S는 명시한 평가 대상의 점수다. 결과가 0이거나 음수여도 유용한 판단을 할 수 있다. 기존 시스템은 시험한 방식으로 D를 추가했을 때 이득을 얻지 못했다는 것이다. D에 맞는 다른 방식으로 활용하면 도움이 될지까지 답한 것은 아니다.

예를 들어 D는 후보 세포를 늘리거나 feature scale을 바꾸고, 일반 link와 division proposal 사이의 상대적인 confidence를 바꿀 수 있다. 기존 candidate 분포로 fit한 linker는 새 후보의 순위를 잘못 매길 수 있다. 기존 score 분포에 맞춘 graph 규칙은 유용한 proposal을 억제할 수 있고, repair 단계가 뒤에서 이를 덮어쓸 수도 있다. 각각 확인해 볼 수 있는 구체적인 결합 가설이다. 실패한 component마다 무조건 기회를 한 번 더 줘야 한다는 뜻은 아니다.

이 plug-in 시험과 구분할 비교는 두 가지다. 하나는 P_D의 학습된 component들을 유지하고, 학습용 데이터에서 범위를 작게 정한 조정 θ_D를 fit하는 것이다. Score calibration이나 graph cost 설정을 조정하는 경우가 여기에 해당한다. 다른 하나는 **P₁ = (A₁, B₁, C₁)**을 별도로 개발한 pipeline으로 비교하는 것이다. 아래첨자는 같은 역할에 속하는 다른 버전을 구분하며, component 세 개를 모두 바꿔야 한다는 뜻은 아니다. 예를 들어 **P₁ = (A₀, B₁, C₁)**은 detector를 유지하고 identity 모델과 graph 규칙만 바꾼 구성이다. Association head를 다시 fit하면 B가 바뀌므로 두 번째 비교로 설명해야 한다. 이를 단순한 운영 설정의 변경 속에 숨겨서는 안 된다. Architecture가 이 역할들로 나뉘지 않는다면 억지로 기호에 맞추기보다 P₁의 구성을 직접 설명하면 된다.

별도로 개발한 pipeline이라고 자동으로 더 강한 버전이 되는 것은 아니며, 단순히 P₀에 D를 붙인 것과도 다르다. 학습 목표와 candidate 분포, 뒤의 단계에 맞춰 정한 운영 설정 θ₁을 명시해야 한다. P₀, P_D, P₁은 비교할 구성을 구분하는 이름이다. 이름만으로 어느 쪽 점수가 더 높을지는 알 수 없다.

| 고정해서 비교할 구성 | 답할 수 있는 질문 |
|---|---|
| P₀, 운영 설정 θ₀ | 현재 전체 기준 시스템은 무엇인가? |
| P_D, 기존 θ₀ 유지 | D를 plug-in으로 붙이면 도움이 되는가? |
| P_D, 학습용 데이터에서 사전에 정한 조정 θ_D | 구체적인 결합 문제를 고치면 D를 유용하게 쓸 수 있는가? |
| P₁, 자체 학습 조건과 운영 설정 명시 | 다른 전체 예측 경로에 더 투자할 가치가 있는가? |
{: #biohub-table-3 .biohub-table .biohub-records style="--c1: 35%; --c2: 65%; --label2: '답할 수 있는 질문';" }
모든 아이디어에 네 구성이 다 필요한 것은 아니다. Held-out 결과를 보기 전에 가능한 설명들을 구분할 가장 작은 비교를 정하면 된다. 조정에는 예산과 종료 시점을 정하고, 학습용 데이터나 한계를 명시한 screening 근거를 사용해야 한다. 평가한 배아를 보면서 계속 재조정한 뒤 최고 결과를 새로운 OOF라고 부르면 또 다른 선택 문제가 생긴다.

학습이 얼마나 진행됐는지도 함께 봐야 한다. 충분히 개발한 P₀가 초기 P₁을 이기면 지금 무엇을 배포할지는 알 수 있다. 그러나 P₁의 예제 노출, 학습 추세, 다른 단계와의 결합에 대한 근거 없이 그 결과만으로 추가 개발에 제한된 예산을 쓸 가치가 있는지까지 알기는 어렵다. 개발에 대한 투자 판단은 최종 채택을 위한 비교보다 앞선다. 그렇다고 P₁이 마지막에 가장 좋은 전체 기준 시스템과 경쟁하지 않아도 되는 것은 아니다.

7월 roadmap은 blending에 대해 이미 이 차이를 짚었다. 그 구분을 실험 일정에도 이어 갔어야 했다. 실제로는 기존 운영 설정에 무언가를 추가하는 작업을 계속하면서, 그 결과를 근거로 대안의 가능성까지 판단하곤 했다.

{::nomarkdown}
<a class="popup img-link biohub-figure" href="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-03-component-and-composition.png?v=ce142e053151">
<picture>
  <source media="(max-width: 620px)" srcset="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-03-component-and-composition-mobile.png?v=9061502684d4">
  <IMG class="biohub-fig-03-component-and-composition" src="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-03-component-and-composition.png?v=ce142e053151" alt="Plug-in 시험과 별도로 개발한 예측 경로" width="1800" height="1310" loading="lazy">
</picture>
</a>
{:/nomarkdown}

_그림 1. P₀는 기존 pipeline, P_D는 D를 추가한 구성, P₁은 component를 명시해 별도로 개발한 pipeline이다. 두 경로는 서로 다른 질문에 답한다. 실험 설계를 설명하는 개념도이며, 실제로 측정한 최적화 지형을 그린 것은 아니다._

## 6. 계속할지 판단하는 기준이 방향을 바꾸기 어렵게 했다

비교의 범위를 제대로 구분하지 못한 문제는 threshold와 낮은 점수를 해석할 때에도 영향을 줬다. 예측을 정하는 규칙과 다음 연구에 투자할지를 정하는 규칙을 나눠 봐야 했다.

**Candidate gate**는 어떤 action을 후보로 만들 수 있는지 정했다. 후반 division 진단에서는 12 µm parent gate 하나 때문에 표기된 event 25개가 제외됐다. Gate를 넓히면 올바른 proposal과 잘못된 proposal이 모두 늘어난다. 여전히 쓸 만한 순위 판단과 graph 규칙이 필요하다. 현재 gate에서 놓친 결과만으로 다른 candidate 분포에 맞춰 학습한 모델의 성능까지 평가한 것은 아니다.

**Verifier의 0.90 threshold**는 제안된 division 중 어떤 것이 graph를 바꿀 수 있는지 정했다. 기록된 embryo-out 평가의 event 151개 중 26개를 복원했고, 후보에는 들어왔지만 놓친 63개는 verifier 점수가 이 threshold보다 낮았다. 순위 판단과 운영 지점에 문제가 있음을 보여 줬다. 그렇다고 그 proposal을 받아들이면 graph가 좋아진다는 뜻은 아니다. False division과 일반 edge의 손상을 함께 측정해야 했다.

**연구를 다음 단계로 진행할 기준**은 더 시험하거나 배포할지를 정했다. 후반 K5는 전체 집계 개선 폭이 **+0.004** 이상이고, 두 배아 prefix의 점 추정치가 모두 0 이상이며, 다른 조건도 충족할 것을 요구했다. 각 prefix에서 +0.004씩 개선하라는 규칙은 아니었다. 로컬 H4 ensemble은 **H1 대비 +0.0007159**여서 별도의 **+0.002** 진행 기준에는 못 미쳤다. 이전 기준 모델과 비교한 K5는 통과한 상태였다.

완성된 후보를 채택할 때에는 이런 기준이 합리적일 수 있다. 아직 개발 중인 접근에 투자할지를 판단하기에는 부족하다. 학습의 매 단계마다 전체 시스템의 채택 기준을 이미 통과해야 한다면, 새로운 supervision과 출력을 활용할 단계, calibration이 필요한 접근은 유용한 정보를 배우는지 확인하기도 전에 탈락할 수 있다. 반대로 무엇을 배웠다는 사실만으로 배포할 수도 없다. 두 판단이 모두 필요했고, 순서를 맞춰야 했다.

### 낮은 점수로 실제 제외할 수 있었던 것은 무엇인가

[5편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/)에서 고정한 action 3,178개에 적용한 GT-assisted **+0.000365**는 제한된 필터를 평가한 결과였다. 없는 event를 새로 만들거나 다른 representation을 도입하거나, 서로 영향을 주는 모든 수정 조합을 열거할 수는 없었다. 다른 division 후보 공간에서 더 큰 GT-assisted 개선 여지가 있었다고 그만큼 학습으로 개선할 수 있다는 뜻도 아니다. 두 결과의 차이는 oracle의 범위가 어떤 action을 허용했는지에 따라 정해진다는 점을 보여 줬다.

원본 patch의 NCC를 학습 없이 쓰는 screen에서는 distance-only 기준의 AUC가 **0.99538784**인데 거기서 **AUC +0.03**을 요구했다. 목표가 1을 넘었다. 신호와 예제 수에 관한 다른 조건도 통과하지 못했으므로 불가능한 기준만이 문제였던 것은 아니다. 하지만 충분히 개발한 contextual appearance 모델을 평가한 실험도 아니었다. 진입 조건과 그 결과로 내린 결론을 함께 살펴봤어야 했다.

GO1/GO2는 작은 평가에서의 경고와 전체 결과가 어떻게 다른지 보여 준다. 16편 묶음에서는 전체 **+0.012533**, 한 prefix는 **−0.001389**였고, 이득의 약 절반은 division 하나를 복원한 데서 나왔다. 질문을 다시 열어 199편을 평가한 결과는 **+0.002100**이었다. 두 prefix는 모두 양수였지만 bootstrap 하한은 **−0.000556**이었고, +0.004 진행 기준에는 못 미쳤다. 크고 안정적인 개선을 놓쳤다는 결과는 아니었다. 다만 작은 묶음에서의 부호와 전체 평가에 따른 진행 판단이 서로 다른 측정이라는 점은 보여 줬다.

{::nomarkdown}
<a class="popup img-link biohub-figure" href="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-02-research-verdicts.png?v=29b04b56f94f">
<picture>
  <source media="(max-width: 620px)" srcset="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-02-research-verdicts-mobile.png?v=edceded37e12">
  <IMG class="biohub-fig-02-research-verdicts" src="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-02-research-verdicts.png?v=29b04b56f94f" alt="실험 결과에 따라 달라져야 할 다음 판단" width="1800" height="1210" loading="lazy">
</picture>
</a>
{:/nomarkdown}

_그림 2. 전체 비교에서 진 composition, 시험한 조건에서 아직 배우지 못한 모델, 미완료 실행, 특정 action에 한정된 진단을 구분해야 한다. 각 결과로 내릴 다음 판단은 다르다._

### Local minimum 비유는 내 작업 방식을 설명할 뿐, 실패의 필연성을 뜻하지 않는다

9월 17일 재검토는 남은 제약 안에서 incumbent가 사실상 도달 가능한 최선의 시스템이라고 서술했다. 나는 실제 비교로 확인한 것보다 넓은 결론을 받아들였다. 실제로 낮은 점수를 낸 결과, 제한된 oracle, 학습 부족, 구현 비용 추정을 한데 모아도 가능한 모든 representation이나 전체 대안을 평가한 것이 되지는 않는다.

Local minimum 비유에서 유용한 부분은 앞선 선택이 이후의 선택을 제약했다는 점이다. 여러 단계를 함께 조정해 둔 상태였다. 대안의 모델과 출력을 쓸 단계를 개발하는 동안은 성능이 잠시 낮아질 수도 있었다. 계속할지를 정하는 규칙 때문에 그런 중간 작업에 투자할 근거를 마련하기 어려웠다. 그러다 준비된 대안이 없다는 사실이 연구를 끝낼 이유가 됐다. 그런데 대안이 준비되지 않은 이유에는 앞서 시간을 배분한 방식도 있었다.

실패가 불가피했다거나 측정할 수 있는 장벽 바로 너머에 훨씬 좋은 해법이 있었다고 입증할 수는 없다. 확인할 수 있는 것은 방향을 바꾸는 비용이 커진 과정, 그리고 그렇게 제한된 대안을 두고 더 조사할 것이 별로 없다고 판단한 방식이다. 마감 직전에는 비용 때문에 멈추는 것이 타당할 수 있었다. 더 일찍 했어야 할 일은 구체적인 대안 하나를 계속 개발하고, 실제 시험한 범위 안에서만 결론을 내리는 것이었다.

대회 종료 후 공개된 write-up을 읽으면 그 방향 전환에 필요한 작업이 구체적으로 보인다. 다른 참가자들은 세포와 label, 오류를 관찰해 다른 학습 과제를 만들었고, 그 모델을 전체 예측으로 이어 갔다.

## 7. Write-up을 읽고 내 실험에서 다시 보게 된 것

대회가 끝난 뒤 해법들을 읽으면서, 내가 이미 부딪혔던 질문들이 눈에 들어왔다. 나도 division 모델과 contrastive temporal representation을 학습했고, 독립 detector를 시험했다. 대안 graph 가설과 division을 함께 판단하는 방법을 비교했고, 외부 microscopy 데이터를 확보했으며, candidate event 1,000개를 직접 검토했다. 여러 방향은 이미 7월 계획에 들어 있었다. 차이는 첫 실험 결과 다음에 무엇을 했는지 따라가면서 더 선명해졌다. 모델은 무엇을 배웠고, 어떤 정보가 여전히 부족했으며, 그 출력은 tracking을 어떻게 바꾸도록 설계돼 있었을까?

특히 참고할 만한 사례들은 이 질문들을 연결했다. 이미지에서 관찰한 문제를 학습 target으로 만들고, 그 target에 맞춰 supervision과 sample을 구성했다. 그렇게 얻은 예측이 graph에서 맡을 역할도 정했다. 결과가 기대에 못 미치면 이 과정의 어느 부분이 어긋났는지에 맞춰 다음 설계를 바꿨다. 나도 문제가 생긴 지점을 찾아낸 경우가 많았다. 다만 그 진단에 이어서 무엇을 개발하고 시험할지까지 꾸준히 진행하지는 못했다.

여기서 읽은 것은 각 저자가 대회 종료 후 공개한 설명이다. 내가 그 실험들을 직접 재현한 것은 아니다. 당시에는 완성된 recipe도, 어느 방법이 성공할지도 알 수 없었다. 비교할 수 있는 것은 그들이 개발 방향을 정한 이유와 내 프로젝트에 이미 나타나 있던 문제들이다.

### 7.1 먼저 loss가 모델에 무엇을 배우게 하는지 알아야 했다

[4위 저자](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744673)는 sparse annotation에 해당하는 항이 전체 loss에서 0.5%도 차지하지 못한 detector를 설명한다. Probability head의 출력이 0에 가까워졌다. Positive, background, unknown 영역을 각각 정규화해서 voxel 수가 영역별 학습 비중을 좌우하지 않도록 고쳤다. [14위 기록](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744486)에도 division-flow 학습에서 비슷한 문제가 나온다. 세포 수를 기준으로 의도한 가중치와 실제 negative voxel의 양이 맞지 않았다. 두 사례 모두 loss가 실제로 어떤 과제를 배우게 했는지부터 살펴봤다.

내게도 7월에 비슷한 신호가 있었다. Temporal embedding 모델에서 association은 계속 좋아지는데 center head는 나빠졌다. 당시 기록은 dark-background voxel이 학습을 지배하는 문제를 진단하고, region-balanced PU loss와 head별 checkpoint를 제안했다. 나는 association과 motion을 재사용하는 방향으로 진행했다. 기록을 다시 읽었지만, 이 center objective를 따로 정규화하고 자체 detector 경로까지 개발한 후속 결과는 찾지 못했다. 다른 detector를 연구하기는 했어도, 이 진단에 직접 답하는 결과는 확인하지 못한 것이다.

이 진단만으로도 작은 학습 비교를 시작할 이유는 있었다. 예제, 초기화, 학습에서 보여 주는 예제의 양을 고정하고 기존 loss와 영역별로 정규화한 loss 하나를 비교할 수 있었다. 먼저 두 방식이 아주 작은 학습 집합을 맞출 수 있는지 확인했어야 한다. 그다음 inner-development 영상에서 candidate 수를 비슷하게 맞춘 뒤 center recall, 영역별 loss 기여, head별 학습 추세를 비교했을 것이다. Loss를 바꾸면 probability scale도 달라질 수 있으므로 threshold는 학습 데이터 안에서 calibration할 필요가 있었다.

그 결과로 다음 투자를 정할 수 있었다. Center head가 회복되고 쓸 만한 inner 예측을 만들었다면 그에 맞는 association과 decoder 개발을 이어 갈 근거가 됐을 것이다. 아주 작은 집합도 맞추지 못했다면 mask, normalization, supervision을 살펴볼 차례였다. 배우기는 했지만 해로운 proposal을 지나치게 많이 만들었다면, 그 조건에서 학습한 설계를 제외할 수 있었다. 전체 loss나 `best` checkpoint 하나만 보는 것보다 분명한 판단이 가능했을 것이다. 의도한 과제를 배우지 못한 것인지, 아니면 배우기는 했지만 graph에 도움이 되지 않는 과제였는지를 구분하는 일이다.

### 7.2 영상을 직접 보면 label뿐 아니라 학습 과제도 바꿀 수 있었다

[우승자](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744801)는 detection에 상당한 시간을 쓰고 세포 약 1,800개를 직접 칠했지만, detector 개선은 기대에 못 미쳤다고 썼다. 저자가 오랜 관찰에서 더 유용하게 얻었다고 설명한 것은 다른 사실이었다. 기하학적으로 애매한 상황에서도 시간에 따른 이미지에서는 관련 세포와 임박한 division을 알아볼 수 있었다.

그 관찰에서 만든 모델이 query-conditioned 모델이었다. 따라갈 세포를 marker로 지정하면 세포의 상태와 시간에 따른 continuation 또는 daughter의 occupancy를 예측한다. Crop에 jitter를 줘 늘 중앙 세포만 따라가는 편법을 줄였고, 시간 간격을 바꾸는 augmentation으로 큰 이동도 보여 줬다. 모델이 제안한 event를 영상으로 확인해 division 학습에 쓸 event를 151개에서 515개로 늘렸으며, 이후에는 어려운 interphase 예제 2,492개를 넣었다. 두 round 모두 class balancing을 명시해 100 epoch씩 학습했고, epoch마다 crop 약 6,600개를 뽑았다. 서로 독립적인 event가 수십만 개 있었다는 뜻은 아니다. 정한 과제를 지속해서 학습한 구체적인 사례다. Learned linker는 그 예측을 사용하면서 division 확률과 daughter 선택을 구분했다.

이 설명을 읽고 내가 직접 영상을 검토한 작업도 다시 살펴봤다. 나 역시 event를 판정하는 데 시간을 썼고, 그 판정은 실제 모델과 제출에 반영됐다. 그렇다면 그 작업은 시스템에 무엇을 가르쳤을까?

#### 내가 검토한 후보 1,000개는 실제로 어떻게 반영됐나

9월 4–6일, 제안된 division을 500개씩 두 batch로 나눠 검토했다. 일반 후보는 agent가 기존 verifier 점수로 골랐고, batch 1의 점수 하한은 0.5, batch 2는 0.4였다. 각 batch에는 GT label을 아는 control 50개를 섞되 화면에서는 control 여부를 표시하지 않았다. 판정의 일관성을 확인하는 용도였으며 새 학습 예제는 아니었다. 나는 각 후보를 `y`, `n`, `s`로 표시했다. 각각 positive, negative, uncertain을 뜻한다. Control과 uncertain 판정은 추가 fitting row에서 제외했고, 기존 HistGradientBoostingClassifier division verifier를 다시 학습했다. 현재 proposal을 판단하는 tabular 모델에 supervision을 더한 작업이었다. 새로운 temporal image 모델을 학습한 것은 아니었다.

제출마다 사용한 판정의 범위는 달랐다. v85에는 batch 1에서 먼저 검토한 250개의 판정을 사용해 row 136개를 추가했다. Positive 58개, negative 78개였다. v86에는 batch 2만 사용해 positive 79개와 negative 186개, 총 265개 row를 추가했다. v87도 같은 batch-2 table을 유지했다. 이후 재검토에서는 판정 1,000개 중 486개가 uncertain으로 남았다. 여기에는 GT control에 대한 판정도 포함됐다. 이 label들은 재검토를 거친 뒤의 상태여서, v86과 v87이 이미 사용한 table과는 달랐다. 검토한 이미지 수가 배포한 모델에 추가된 학습 예제 수를 뜻하지는 않았다.

#### 로컬에서 좋아진 것과 실제 제출에서 나온 결과

Verifier threshold를 0.75로 유지하고 batch 1의 일부 label을 추가하자, paired local replay 점수가 0.7535에서 0.7573으로 바뀌었다. **+0.0038**이었다. Threshold를 0.65로 낮추면서 **+0.0010**이 더해져 0.7583이 됐다. 전체 +0.0048에는 label 추가와 operating point 변경이 함께 들어 있었다. 이 로컬 비교들은 같은 upstream pipeline을 공유했다. 그 조건에서의 개선을 측정한 것이며, 학습에 없는 집단으로의 transfer를 독립적으로 확인한 결과는 아니었다.

완료된 제출에서는 결과가 엇갈렸다.

| 버전 | Verifier 변경 | Public | Private |
|---|---|---:|---:|
| v83 | GT table; threshold 0.75 | 0.94437 | 0.91331 |
| v85 | Batch 1 일부; threshold 0.65 | 0.93996 | 0.91373 |
| v86 | Batch 2; threshold 0.90 | 0.94347 | 0.90877 |
| v87 | 같은 batch 2; cue 세 개와 imputation 추가; threshold 0.90 | 0.94662 | 0.91217 |
{: #biohub-handlabel-scores .biohub-table .biohub-numeric .biohub-score style="--c1: 12%; --c2: 50%; --c3: 19%; --c4: 19%; --label2: 'Verifier 변경'; --label3: 'Public'; --label4: 'Private'; --table-min: 42rem;" }
v85의 Public 점수는 내려갔지만, v83과의 Private 점수 차이는 **+0.00042**였다. v87의 Public 점수는 올라갔지만 v83과의 Private 점수 차이는 **−0.00114**였다. 이 비교에는 table과 threshold 변경이 함께 들어 있으므로 전체 효과를 labeling에 돌릴 수는 없다. 저장한 제출 조회 결과에는 집계 점수만 있고 hidden graph는 없다. 어떤 division을 복원하거나 망가뜨렸는지까지는 알 수 없었다. [근거 기록]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/handlabel-evidence.json)에는 실제 제출한 composition과 로컬에서 label 추가만 비교한 결과를 구분해 두었다.

v86에서 v87로의 변경은 범위가 더 좁다. Upstream 모델, batch-2 label, threshold 0.90을 유지한 채, 근처의 기존 detection과 daughter 후보 위치의 frame별 원본 이미지 밝기에 관한 cue 세 개를 추가했다. 확장된 feature schema에는 median imputation을 적용했다. 이 묶음으로 로컬 replay는 **+0.0018**, Public 점수는 **+0.00315**, Private 점수는 **+0.00340** 좋아졌다. 해당 composition에서 이 묶음이 유용했다는 근거이며, cue 각각의 효과를 분리한 것은 아니다. v86에서 낮아졌던 Private 점수를 상당 부분 회복했지만 v87은 여전히 v83보다 낮았다.

#### 직접 검토하면서 드러난 sampling과 representation의 문제

Batch 1 전체를 써도 일부만 썼을 때보다 로컬 점수가 더 좋아지지는 않았다. False positive가 늘었지만 복원한 true division은 그만큼 늘지 않았다. 이쯤에서는 같은 방식으로 label을 더 모으는 일이 덜 유망해 보였다. 다만 중요한 것은 그 이유였다.

9월 5일, batch 1 전체를 쓴 recipe의 6bba 배아 쪽을 진단했을 때, 놓친 GT division의 verifier 점수 중앙값은 **0.084**, 90th percentile은 **0.368**이었다. 둘 다 batch를 고를 때 쓴 점수 하한보다 훨씬 낮았다. 진단과 후보 선정에 쓴 fit이 달랐으므로, 이 수치로 sampling에서 빠진 event 수를 정확히 셀 수는 없다. 그래도 labeling 계획의 약점은 드러났다. 현재 ranker가 이미 그럴듯하다고 보는 후보를 주로 검토하면, 낮은 점수를 받은 별개의 true event들은 충분히 들어오지 않을 수 있었다.

기존 feature도 두 상황을 혼동했다. 실제 daughter가 이미 다른 track에 붙어 있는 경우와, 무관한 이웃 세포가 제안된 fork에 들어온 경우였다. Label을 추가하면 판단 경계를 옮길 수 있지만, refit은 여전히 그 table에 담긴 정보로 판단해야 한다. 판정 수를 늘리는 것만으로 더 풍부한 temporal identity representation을 배울 수는 없었다.

판정 자체에 더 나은 화면이 필요한 경우도 있었다. 내 판정을 거리로 설명한 해석도 바로잡아야 했다. GT-positive 사례를 거부한 이유는 거리가 아니었다. 어떤 3D 화면에서는 한 세포가 다른 세포 뒤에서 나타나 관계를 판단하기 어려웠다. 더 넓게 다시 검토하면서 60개 판정을 `n`에서 `s`로 바꿨고, `n`에서 `y`로 바꾼 것은 없었다. 이미지에서 판단하기 어렵다는 것과 확실히 division이 아니라는 판정을 구분하고, annotation 기준도 명시할 필요가 있었다.

이 결과는 recipe를 그대로 둔 채 수천 개를 더 판정하는 일을 보류할 근거가 됐다. 동시에 더 작은 검토를 다른 방식으로 할 이유도 줬다. 낮은 점수를 받은 missed event와 비슷한 false fork를 모으고, 깊이 방향을 더 잘 보여 주며, 내가 영상에서 볼 수 있는 것과 모델의 입력을 나란히 비교하는 것이다. 그런데 당시 내가 사용하던 Public 해석 규칙은 v86 점수가 충분히 낮으면 hand row가 어떤 threshold에서도 hidden 성능을 해친다는 결론까지 허용했다. 제출한 composition 하나로는 그렇게 판단할 수 없었다. 지금의 Private 결과도 그 구분이 필요했음을 보여 준다. 하지만 Private를 알기 전에도 한계는 분명했다. 부적절한 threshold, 부족한 representation, 다른 annotation 과제는 각각 따로 확인해야 할 설명이었다.

#### Daughter 모델이 여전히 답해야 했던 질문

나는 hard-negative fine-tuning과 정확한 graph 비교를 포함한 strict division 모델도 학습했다. 9월 daughter proposal 작업에서는 endpoint 12개를 각각 3,000 step씩 학습했다. Inner cross-fit readiness의 division AUROC는 0.813/0.912였지만, 두 daughter를 모두 각 GT 위치에서 3.5 µm 안에 찾은 비율은 0.31/0.36에 그쳤다. 진단은 shared target에서 두 번째 peak가 약하다는 문제를 짚었다. Tabular refit과는 별개의 개발이었고, 다른 문제를 보여 줬다. Division을 알아본다고 해서 지정한 parent의 두 daughter를 모두 찾는 것은 아니었다.

Division 학습에 쓸 event가 적다는 문제는 7월부터 보였다. 9월에는 두 번째 peak의 실패가 target을 살펴볼 구체적인 이유를 더했다. Missed fork와 false fork의 고정된 사례 묶음을 raw XY/XZ temporal 화면에서 보고, 모델의 입력과 출력을 함께 확인할 수 있었다. Crop에 두 daughter가 모두 들어 있었을까? Label의 기준은 일관됐을까? 모델이 지정한 parent 대신 가장 가까운 밝은 세포를 따라가고 있지는 않았을까? 이런 질문을 통해 직접 관찰한 것을 실제 학습 변경으로 연결할 수 있었다.

범위를 정한 training-side 비교 하나로, 기존 shared target과 query marker를 명시한 별도의 continuation/division target을 시험할 수 있었다. Label과 학습 노출을 고정하고, 모르는 사례는 unknown으로 남겨 뒀어야 한다. 일반 세포에서 오류를 지나치게 늘리지 않으면서 두 daughter를 더 잘 찾았다면, 그 정보를 사용할 linker와 더 넓은 평가를 개발할 근거가 됐을 것이다. 학습 자체가 안 됐다면 진단한 target이나 학습 예제 구성의 문제 하나를 고칠 차례였다. 배우기는 했지만 inner generalization이 유용하지 않았다면 그 설계는 끝낼 수 있었다.

우승자의 기록은 그 후속 질문을 더 구체적으로 생각하게 해 줬다. 그대로 따르면 성공할 recipe를 준 것은 아니다. 저자가 밝힌 낙관적인 pseudo-label CV와 MAE만의 효과를 분리하기 어렵다는 한계도 남아 있다. 내 실험에서 필요했던 것은 영상에서 보이는 단서, query, 학습 target, graph가 실제로 선택할 daughter를 연결하는 일이었다. Annotation 총량이나 epoch 수만으로는 그 연결을 확인할 수 없었다.

### 7.3 Identity는 실제로 경쟁하는 세포들 사이에서 배워야 했다

Parent를 지정하는 문제는 일반 link도 다시 보게 했다. Detector가 nucleus를 찾을 수 있다고 해서 가까운 다른 nucleus와 구분할 정보까지 충분히 배웠다는 뜻은 아니다. 여러 write-up이 이 차이를 명시적으로 다뤘다.

[3위 해법](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744484)은 별도의 appearance matcher를 학습하고, annotation이 없는 불확실한 세포를 조심스럽게 다뤘다. [10위 해법](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744980)은 독립적인 contrastive appearance learner와 contextual linker를 개발했다. 더 풍부한 모델을 완성하기 전에는 position-only transformer로 비용이 적게 드는 전체 시스템 비교를 먼저 했다. [6위 해법](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744582)은 자체 detection을 바탕으로 association을 학습하면서, GT와 대응시킨 뒤 위치를 흔든 center와 detector가 만든 distractor를 포함했다. [16위 기록](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744671)은 detector가 candidate 분포를 바꾸면 edge 모델도 다시 학습했다.

나도 7월에 identity contrastive learning과 hard negative를 제안했고, temporal 모델을 학습했다. Center head가 무너지는 동안 association은 좋아졌다. 그다음 확인할 것은 이 representation이 tracking에서 중요한 어려운 경쟁 상대를 구분했는지였다. 이후 readiness 점검에서 구체적인 문제가 드러났다. 한 readiness fold에서는 true successor의 **94.35%**가 가장 가까운 detection이었다. 다음 frame을 보는 것은 정당했다. 다만 이런 후보 집합에서 복원율이 높다는 사실은 쉬운 기하학적 판단으로도 상당 부분 설명될 수 있었다. 밀집하거나 애매한 이웃 세포 사이의 identity를 얼마나 구분했는지는 충분히 알려 주지 못했다.

다음 비교는 정답 parent와 틀린 parent가 모두 geometric gate를 통과한 training-side 사례에서 시작할 수 있었다. 정답 node나 link candidate 자체가 없는 사례는 따로 뒀어야 한다. 이 고정된 사례 묶음에서 현재 representation과 해당 과제에 맞춰 학습한 identity representation 하나를 비교할 수 있었다. 가까운 경쟁 세포를 포함하고, 정답 여부가 불분명한 link를 조심스럽게 다뤄야 했다. Inner 사례에서 후보들 사이의 순위를 보면 새 정보가 유용한지 알 수 있다. 그다음 그 정보를 쓰는 head를 학습하고, 저장한 graph를 작게 비교하면 시스템이 실제로 그 정보를 사용했는지까지 확인할 수 있었다.

이 순서는 실제 실패를 변명하지 않으면서 학습과 결합을 구분하는 데 도움이 됐을 것이다. Teacher-clean A2 비교에서 student는 자체 detection과 association을 사용했고, 당시 선택해 둔 graph 설정 아래에서 졌다. 그 실패를 새 detector에 옛 linker를 강제로 붙였기 때문이라고 설명할 수는 없다. 측정한 composition은 제외할 근거가 있었다. Identity에 대한 후속 질문은 별도의 구체적 이유가 있는 설계들, 그리고 거기에 투자하기 전에 필요한 근거에 관한 것이었다.

### 7.4 Label은 입력으로 답할 수 있는 질문이어야 했다

Detection, identity, daughter 선택을 나누고 나니 label 자체도 더 자세히 볼 필요가 있었다. 두 저자는 겉으로 보면 반대 방향으로 target을 바꿨다.

[14위 저자](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744486)는 division-site reader의 label을 graph-TP에서 image-site로 옮겼다. 이미지에 보이는 생물학적 event와 특정 fork를 graph에 넣었을 때의 성공은 다른 질문이었다. [17위 저자](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744930)는 semantic duplicate classifier의 AUROC는 좋았지만 graph 수정이 해로웠다고 보고했고, supervision을 수정의 metric 효과로 바꿨다. Image reader에는 이미지로 답할 수 있는 target이 필요했고, graph editor에는 실제 행동에 맞는 target이 필요했다.

이 구분은 내 division 점수의 한계도 설명해 준다. GT-centered crop에서 inner AUROC가 **0.813, 0.912**였던 endpoint들은 다른 배아의 verifier candidate pool에서 각각 **0.5752, 0.6922**를 기록했다. Candidate 분포와 평가 domain·split이 함께 바뀌었으므로, crop 중심 오차만의 영향을 분리한 결과는 아니다. 한 fold에는 **training positive가 26개**뿐이었다. 실제 적용에서는 애매한 daughter 후보와 경쟁하는 일반 edge까지 판단해야 했다. 다른 appearance 모델이 무엇을 배울 수 있는지 판단하기 전에 이런 차이들을 구분할 필요가 있었다.

9월 14일의 S4 검토에서는 일반 identity에서도 비슷한 문제가 나타났다. 표시한 source cell을 다음 frame에서 이어받은 것이 GT successor인지, 아니면 내 pipeline이 고른 다른 세포인지를 묻는 작업이었다. 나는 temporal 화면과 깊이 방향 projection을 보며 control 여부를 모르는 A/B 항목 **100개**를 검토했다. 의심되는 swap **80개**, 알려진 link를 사용하는 control **20개**였다. Control 중 **15개**에서는 GT successor를 골랐고 **다섯 개**는 uncertain으로 남겼다. 틀린 쪽을 확신해서 고른 것은 없었다. Uncertain을 불일치로 세면 일치율은 **75%**로, 미리 정한 **85%** 기준보다 낮아 진단은 inconclusive가 됐다. 나는 노란 source marker가 두 세포 사이에 놓이는 경우가 많다고 알렸고, 그런 사례를 uncertain으로 표시했다. 이 판정을 후속 모델 개발의 근거로 삼으려면 화면과 annotation 질문부터 살펴봐야 했다.

S4는 이 진단에서 멈췄다. 그 판정을 모델, fitting table, threshold에 넣지 않았으므로 판정을 실제 모델에 적용해 candidate를 만든 것도 아니고, 그에 따른 점수 변화를 평가한 것도 아니다. Refit과 실제 제출까지 이어진 9월 4–6일 division 검토와는 달랐다. 화면과 labeling 기준을 더 분명하게 만든 뒤 작게 다시 검토했다면, 의도한 identity 질문에 내가 일관되게 답할 수 있는지 확인할 수 있었을 것이다. 확신해서 잘못 고른 control은 없고 다섯 개가 불확실했다는 결과만으로 annotation을 할 수 없다거나, GT 또는 tracking 방식이 잘못됐다고 결론 낼 수는 없었다.

[12위 해법](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744501)은 이런 단계를 나눈 뒤 실제 판단을 어떻게 평가하는지 보여 준다. 놓친 division **109개 중 11개**를 복원하는 지점에서, 저자가 보고한 precision은 reader만 쓰면 **5.2%**, chooser를 함께 쓰면 **32.4%**, 전체 시스템을 쓰면 **84.6%**였다. Upstream in-sample graph를 사용한 조건에서의 측정이었다. 그래도 global AUC보다 제안된 graph 수정에 대해 훨씬 구체적으로 알려 줬다. Event를 알아보고, daughter를 고르고, fork를 채택하는 단계마다 오류율이 달랐다.

내 후속 실험에서는 실제 적용과 비슷한 training proposal pool 하나를 고정하고 세 질문을 나눴어야 한다. 이미지에 division이 보이는가, 어떤 daughter 후보가 맞는가, 경쟁하는 일반 link 대신 이 fork를 채택할 것인가를 각각 다루는 것이다. 학습이나 calibration은 학습 데이터 안에서 하고, 비교에서는 미리 정한 복원량에서의 precision과 전체 graph의 손상을 함께 봐야 했다. 각 단계가 입력에 맞는 target을 배우게 하되, 최종 기여는 완성된 시스템에서 판단할 수 있었을 것이다.

### 7.5 유용한 proposal도 다른 역할로 써야 할 수 있었다

신호를 배우는 일과 graph 수정을 채택하는 일을 구분하고 나니, 실패한 detector 교체도 다르게 보였다. 빠진 유용한 후보를 공급할 수 있는 모델이 전체 graph를 맡기기에는 좋지 않을 수 있었다.

[2위 저자](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744723)는 희미한 세포에 대한 synthetic augmentation을 강화한 detector를 개발했다. CV는 좋아졌지만 주 detector를 교체했을 때 Public은 **0.957에서 0.950**으로 내려갔다. 최종 시스템에는 보조 track 공급원으로 남겼다. 이후 recovery에서는 여러 frame에 걸쳐 이어지는지 확인하고 원래 해상도의 이미지에서 contrast를 확인했다. 추가 detector의 역할이 주 candidate 분포를 만드는 일에서 특정한 빠진 구조를 제안하는 일로 바뀐 것이다.

나도 7월 30일에 multi-hypothesis graph union과 tracklet admission을 시험했다. 넓게 추가하면 공식 점수가 나빠졌고, learned tabular selector의 작은 개선은 promotion 기준보다 낮았다. 그런데 proposal pool에는 anchor가 놓친 unique true edge가 **1,704개** 있었다. 이 결과는 모델을 더 넣는 일 전체를 거부하는 것보다 문제를 구체적으로 보여 줬다. 유용한 대안이 있었지만, 현재 selector가 안전하게 받아들이지 못하는 추가 후보들도 함께 있었다. 기록된 입력은 source, geometry, topology, persistence, probability 요약이었고 raw image 정보는 포함돼 있지 않았다.

그래서 이미지 정보를 쓰는 보조 역할을 작게 비교할 이유가 있었다. Candidate를 고정하고 주 모델 교체와 보조 사용을 비교하며, 기존 selector와 이미지 정보를 명시적으로 더한 selector를 비교할 수 있었다. 추가 node 수가 비슷한 지점에서 unique correct track, 채점에 잡히는 false edge, 정답 여부가 불확실한 추가 node, node-count 항의 기여를 측정했을 것이다. 후보 공급원에 쓸 만한 것이 없는지, 아니면 선택 규칙에 그것을 고를 정보가 부족한지를 알아보는 비교다. 새 역할로 사용해도 졌다면 그 composition을 제외했을 것이다.

2위 결과에서 그대로 복사할 상수를 찾으려는 것은 아니다. Recovery 개선은 몇몇 영상에 집중됐고, nested parameter selection을 하면 추정 개선 폭도 줄었다. 두 배아가 섞인 CV와 선택된 acquisition별 규칙도 transfer 주장의 한계였다. 참고할 것은 개발 판단이다. 주 모델 교체의 실패를 후보 공급원 자체가 쓸모없다는 뜻으로 읽기 전에, 추가 모델이 맡을 수 있는 일을 살펴보는 것이다.

### 7.6 Supervision을 늘리려면 teacher의 target도 맞아야 했다

모델이 맡을 역할을 바꾸면 무엇으로 그 역할을 학습시킬지도 다시 묻게 된다. 외부 데이터와 pseudo-label로 학습 예제를 늘릴 수는 있지만, teacher의 출력은 student가 이미지에서 배울 수 있는 것과 맞아야 했다.

6위의 teacher–student 작업은 topology를 최적화한 association을 쓰면서도 이미지 학습에는 raw detector 위치를 남겼다. GT의 비중을 크게 유지하고 pseudo 예제에는 가중치를 줬으며, 일부 target은 GT로만 제한했다. 저자는 첫 round에서 개선됐지만 단순한 두 번째 round에서는 더 좋아지지 않았고, 이후 low-contrast augmentation을 추가했다고 보고한다. Pseudo-label round를 추가할 때마다 도움이 될 것이라고 가정한 것이 아니라, 학습 과제의 특성에 맞춰 다음 변경을 골랐다.

나도 외부 데이터 fine-tuning, synthetic division transfer, teacher-clean student 비교를 실제로 마쳤다. 개선이 작거나 점수가 나빠진 결과는 시험한 recipe를 제외할 근거가 됐다. 그다음 무엇을 시험할지 정하려면 실제 target을 확인해야 했다. 어떤 위치, link, event label을 줬는지, 거기에 어떤 오류가 있는지, student가 그 답을 pixel에서 어떻게 알아낼 수 있는지를 살펴보는 것이다.

전체 graph의 topology로는 link를 정당화할 수 있어도, 조정한 좌표가 이미지를 읽기 좋은 crop 중심이 아닐 수 있다. 반대로 밝은 center를 찾았다고 parent identity가 정해지는 것은 아니다. 따라서 어떤 teacher는 association에는 유용하지만 localization에는 맞지 않을 수 있고, 반대도 가능하다. Incumbent와 일치한다는 사실만으로 student가 쓸 만한 새 신호를 배웠는지 알기는 어렵다.

긴 학습을 다시 시작하기 전에 학습용 target의 고정된 사례 묶음을 살펴볼 수 있었다. In-domain 학습 노출과 inner 비교 조건을 같게 두고, 기존 recipe와 target 또는 가중치 규칙 하나를 정확히 바꾼 recipe를 비교하는 것이다. Target이 부적절하면 고치거나 보류할 차례다. Target은 배웠는데 최종 graph에서 졌다면 그 전체 recipe를 제외하고, 미리 제시한 상호작용이 어디서 나타났는지 확인할 수 있었다. 이런 순서라면 학습 비용을 더 들이기 전에, 추가 supervision이 왜 도움이 될지 근거를 확인할 수 있었다. 6위의 후반 pseudo-label 선택도 validation에 노출됐으므로, 내 recipe를 정한 뒤에는 설정을 고정한 별도의 confirmation이 여전히 필요했다.

### 7.7 모델이 준 정보는 최종 graph에도 남아야 했다

Target이 유용하고 학습도 충분히 됐으며 맡긴 역할도 맞더라도, 마지막 단계에서 문제가 생길 수 있었다. 뒤의 graph 처리가 앞 모델의 판단을 되돌리는 경우다.

[9위 기록](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744835)은 내가 공개한 기반을 사용하면서 motion relinking이 ILP의 division 선택을 덮어쓴다는 사실을 발견했다. Final DIVCARRY는 pipeline 끝에 가까운 곳에서 조건에 맞는 fork를 복원했다. 저자가 보고한 개선은 **Public +0.011 / Private +0.013**이었다. [18위 해법](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744531)도 division critic을 전체 graph가 만들어진 뒤로 옮기고, 그 단계에서 event 정보와 일반 link가 경쟁하도록 했다.

내 7월 기록에도 비슷한 문제가 있었다. Hyperedge reward를 높이면 raw ILP fork는 늘었지만, 뒤의 graph filter를 거치면 최종 division TP/FP/FN은 같았다. Zero-arm parity도 통과했고, 당시 기록은 downstream 처리 방식이 문제라고 정확히 짚었다. 의도한 변경의 상당 부분이 채점 전에 사라졌다. 이런 null score로는 바뀐 fork를 유지하는 graph에서 그 정보가 얼마나 유용할지 알 수 없었다.

바로 다음에 할 수 있었던 일은 실제 영상 몇 편에서의 추적이었다. Event ID를 proposal, optimization, filtering, relinking, serialization까지 따라가며, 바뀐 fork를 처음 없애는 단계를 기록하는 것이다. 그다음 fork를 고려하는 downstream 처리 하나나 더 뒤의 삽입 위치 하나를 정해 변경 없는 control과 비교할 수 있었다. 일반 edge의 손상과 graph 유효성도 함께 확인해야 했다. 이렇게 하면 특정한 결합 문제를 시험할 수 있다. Filter 전체를 없애거나 upstream reward만 계속 높여서는 같은 질문이 풀리지 않는다.

3위의 개발 과정도 앞에서 **기존 θ₀를 유지한 추가 구성 P_D**와, 그 구성에 맞게 미리 정한 adaptation 또는 별도로 개발한 **P₁**을 나눈 이유를 보여 준다. Matcher만 바꾼 단계에서는 집계 점수가 거의 달라지지 않았지만, 이후 division 정보, calibration, joint graph optimization을 묶은 변경은 훨씬 크게 개선됐다. 여러 축이 함께 바뀌었으므로 그 개선을 matcher에 돌릴 수는 없다. 다만 상호작용을 의심할 근거가 있을 때 왜 component×base 비교를 미리 정해야 하는지는 보여 준다. 이 질문에 답할 곳은 개별 module의 점수보다 완성된 graph였다.

### 7.8 시간과 데이터, 코드 작업은 이 개발 과정을 뒷받침해야 했다

Target에서 최종 graph까지 사례들을 따라가면서, 내가 제공한 자원도 다르게 보게 됐다. 데이터나 계산을 늘리는 투자는 구체적인 학습 또는 결합 문제를 충분히 일찍 다룰 때 도움이 될 수 있었다.

[5위 기록](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744549)은 공개 ZebraHub microscopy로 pretraining한 뒤 대회 데이터에 fine-tuning했다고 보고한다. 나도 나중에는 외부 배아 데이터셋 **네 개**, 사용 가능한 division event **10,000개 이상**을 준비했다. Synthetic geometry 구성과 실제 microscopy fine-tuning 구성들을 학습했고, 앞에서 설명한 수작업 검토도 마쳤다. 남아 있던 질문은 그 데이터와 target으로 내 tracking 시스템에 부족한 정보를 개발했는지였다.

Pretraining 비교를 더 일찍 했다면, 같은 과제 모델에서 scratch 학습과 외부 pretraining 초기화를 비교하고 in-domain 학습 노출을 같게 둘 수 있었다. Source와 target의 실제 예제를 보며 데이터가 맞는지 판단하고, 더 빠른 수렴이나 해당 과제의 inner-development 개선을 찾았을 것이다. 외부 과제는 배웠지만 이 과제로 옮겨오지 못했다면 그 source와 objective의 조합은 더 진행하지 않거나, 구체적인 adaptation 하나를 시험할 이유가 있는지 판단할 수 있었다. Synthetic AUC가 높다는 사실만으로 필요한 시각적 tracking 구분을 배웠다고 볼 수는 없었다.

Engineering도 같은 기준으로 볼 필요가 있었다. 6위는 encoder feature cache와 더 적절한 numerical precision으로 inference 시간을 줄였다고 보고한다. Notebook 시간 제한 안에서 모델과 view를 비교할 여유가 생겼다. 실제 예측을 제한하는 조건을 줄인 작업이었다. 내 프로젝트에서는 또 다른 mock을 고치거나 범용 verifier를 확장하는 데 시간을 쓰고도 다음 연구 비교를 실행하지 못할 수 있었다. Code 작업의 완료 기준에는 그것이 가능하게 만드는 비교가 명시돼 있어야 했다.

[7위 해법](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744937)은 연구 방향의 수와 ensemble 크기를 구분하게 해 줬다. 강한 detection·motion architecture 하나와 별도의 event 판단을 개발했다. 내 portfolio에 필요했던 것은 아직 풀리지 않은 서로 다른 질문에 답할 수 있는 방법과, 그에 맞는 실제 예측 경로였다. 모델의 기여로 inference 예산을 정하고, cache와 adapter가 가능하게 만든 실험으로 engineering의 가치를 판단할 수 있어야 했다.

이 기록들이 더 좋은 결과를 약속하는 것은 아니다. 17위는 design half에서 얻은 개선 **19개 중 일곱 개**를 별도의 confirmation half에서 제외했다고 보고한다. 두 묶음 모두 같은 **두 배아**에서 나왔다. [Public 12위 / Private 95위 기록](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744912)은 모델을 추가로 학습하고 여러 단계를 바꿨지만 순위가 크게 내려갔다. 4위는 cross-embryo detector 성능이 일찍 최고점에 도달한 경우를 보고했고, 7위의 더 긴 synthetic pretraining은 Public에는 도움이 됐지만 Private 점수는 나빠졌다. 이런 사례를 보면 learning curve와 설정을 고정한 confirmation도 개발 과정의 일부여야 한다. 작업을 더 하면 반드시 좋아진다고 볼 이유는 없다.

상위권 해법을 내 실험 기록과 대조해 보니, 문제를 진단해 놓고도 다음 학습 과제로 이어 가지 못한 지점들이 보였다. 유용한 모델과 실제 실험이 있었고, 타당한 진단도 여러 번 했다. 그중 하나를 골라 다음 학습 과제로 만들고, 필요한 예제와 그 정보를 사용할 graph 경로를 마련한 뒤, 정해 둔 지점에서 그 설계를 판단했어야 한다. 그 결과 다른 쓸 만한 시스템이 생길 수도 있었고, 더 충분한 근거로 멈출 수도 있었다. 아직 방향을 바꿀 시간이 있을 때 큰 계획을 뒷받침할 실험을 더 탄탄하게 갖출 수 있었을 것이다.

## 8. 다음 연구에는 이 경험을 어떻게 적용할까

공개된 write-up을 읽으면서 초기 계획을 더 구체화할 방법이 보였다. 예측에 빠진 정보가 무엇인지 찾고, 그 정보를 학습할 모델을 고른 뒤, 필요한 예제와 학습 기회를 주고, 다음 단계가 그 출력을 활용할 수 있게 해야 한다. Portfolio는 서로 다른 가능성을 비교할 수 있을 만큼 개발을 이어 가는 장치여야 한다. 모델 목록에 이름이 많이 올라 있다고 그런 역할을 하는 것은 아니다.

| 연구 방향 | 새로 얻거나 확인할 정보 | 처음 판단할 질문 |
|---|---|---|
| 기존 temporal detector–linker | 믿을 만한 기준 예측과 주된 오류의 원인 | 기존 병목 중 어디를 개선할 가치가 있는가? |
| Identity representation 또는 함께 작동하는 다른 구성 | 가까운 경쟁 세포를 구분할 근거, 또는 localization과 candidate support가 다른 detector–linker 조합 | 필요한 과제를 학습하고 있으며, 활용할 수 있는 전체 예측을 만드는가? |
| Temporal event 모델 | 최종 graph까지 반영될 수 있는 mother–daughter 근거와 candidate support | 일반 edge에 허용하기 어려운 손상을 주지 않으면서 실제 event 후보의 순위를 매길 수 있는가? |
{: #biohub-table-4 .biohub-table .biohub-records style="--c1: 22%; --c2: 41%; --c3: 37%; --label2: '새로 얻거나 확인할 정보'; --label3: '처음 판단할 질문';" }
다음에는 baseline 하나와 개발할 시간과 예산을 따로 확보한 대안 하나로 시작하겠다. 다른 방향은 질문과 예산이 분명해졌을 때 추가할 생각이다. 대안은 서로 다른 오류를 다루거나 다른 정보를 학습해야 한다. Seed나 architecture 이름만 다르다면, 실제 기여를 확인하거나 학습 문제를 진단하는 데 도움이 될 때 의미가 있다.

### 이미 학습한 representation에서 시작할 수 있는 구체적인 실험

새 모델 이름부터 찾기보다, 저장해 둔 temporal representation의 예측을 baseline과 정답 link 옆에 놓고 살펴보고 싶다. 먼저 학습용 데이터에서 정답 parent와 그럴듯한 오답 parent가 모두 geometric gate를 통과한 사례들을 골라 고정한다. Node가 없거나 정답 candidate가 빠진 경우는 따로 둬야 한다. Identity 모델은 후보 안에 없는 세포를 선택할 수 없기 때문이다.

이 사례들에서 기존 representation이 놓친 구체적인 외형 차이가 드러난다면, 다음 후보 하나와 비교할 수 있다. 과제에 맞는 pretrained image encoder나 작은 temporal encoder가 그 후보가 될 수 있다. 학습 목표를 고정하고, 가까운 경쟁 세포를 hard negative로 뽑으며, 판단할 수 없는 link는 unknown으로 남기고 실제로 본 예제 수를 기록한다. 첫 판단은 inner-development 사례에서 필요한 대상을 구분하는지다. 이 단계부터 전체 시스템의 큰 점수 상승을 요구할 필요는 없다.

순위를 유용하게 매긴다는 근거가 나오면, 그 출력을 활용할 association head를 만들고 같은 node 후보군에서 소규모 final-graph 비교를 진행한다. 달라진 identity link와 일반 edge의 손상, 전체 실행 시간을 함께 확인해야 한다. 학습 자체가 실패했다면 데이터를 바꾸거나 예제 노출을 늘리거나 초기화 방식을 바꾸는 것 중, 근거가 있는 변경 하나를 검토한다. 학습은 됐지만 정보가 뒤의 단계에서 사라졌다면, 확인한 결합 문제를 시험한다. 충분히 개발한 구성이 비교에서 졌다면 그 방식은 중단한다. 어느 결과가 나오든 다음 행동을 더 분명히 정할 수 있다.

Division 학습에도 같은 순서를 적용할 수 있다. 직접 labeling해 보니, tabular 입력에 없는 시간상의 차이를 label만으로 제공할 수는 없었다. 다음 한 시간을 어디에 쓸지도 나눠 판단해야 한다. Annotation 화면을 개선할지, 놓친 positive 유형을 더 모을지, 다른 temporal 학습 과제를 개발할지는 각각 다른 실험이며 남겨야 할 결과도 다르다.

### Held-out 경계는 처음에 정하고, 유망한 후보에 확인 비용을 쓰기

Portfolio를 개발한 뒤 평가하자는 초기 생각에는 한 가지 조건이 필요하다. Held-out 경계는 학습 전에 정해야 한다. 그렇지 않으면 all-train teacher, checkpoint 선택, 학습한 feature가 나중의 비교에 영향을 줄 수 있다. 뒤에서 row를 걸러 내는 것만으로는 그 영향을 지울 수 없다.

미성숙한 아이디어마다 199편 전체를 돌려야 한다는 뜻은 아니다. 실제 영상 몇 편으로 실행 경로를 확인하고, 학습용 데이터에서 예제 노출과 학습 여부를 점검할 수 있다. 한계를 명시한다면 기존의 불완전한 근거도 구성을 고정한 소규모 paired screening에 쓸 수 있다. 거기서 유용한 신호를 보인 recipe에 reciprocal 평가와 전체 notebook 검증 비용을 쓰면 된다.

전체 outer-pure 비교에서는 held-out 배아를 detector·linker 학습, teacher 생성, checkpoint 선택, stacked feature, calibration, 학습한 모델 조합에서 모두 제외해야 한다. All-train asset을 함께 사용했다면 그 결과에는 해당 조건이 붙는다. 같은 배아 두 개를 보면서 여러 번 선택한 과정도 평가 이력에 남는다. 나중에 구성을 고정해 한 번 더 실행해도 독립적인 domain이 새로 생기지는 않는다.

Deployment parity도 모델을 고르는 데 영향을 줄 수 있을 만큼 일찍 확인하고 싶다. 같은 전처리, 모델 역할, graph 처리 순서에서 실제 notebook 단계들이 로컬에서 본 효과를 유지하는지 살펴봐야 한다. 신뢰할 만한 외부 시스템도 source와 asset, 실행 시간이 허용한다면 전체 구성을 비교할 수 있도록 보관하겠다. 상수 하나만 가져오는 것은 다른 실험이다. Public과 로컬 결과가 다르면 학습 예제 노출, proposal 분포, graph 단계, domain별 반응을 확인한 뒤 해석을 정해야 한다.

### 판단이 필요한 시점마다 agent에게 무엇을 물을까

Agent에게 trainer 구현, 사례 수집, graph 변화 추적을 맡길 수 있다. 다만 활동 보고서가 길다는 이유로 연구 결과가 생겼다고 판단해서는 안 된다. 다음 행동으로 이어지게 하려면 풀고 싶은 불확실성을 중심으로 요청해야 한다.

- **학습이 잘되지 않을 때:** “모델이 학습할 수 있어야 하는 training 사례, 잘 처리한 inner 사례, 놓친 사례를 하나씩 보여 줘. 각 head는 필요한 예제를 얼마나 봤고, 어떤 learning curve가 달라졌는가?”
- **후보 점수가 낮을 때:** “이 후보에서 시험한 역할은 무엇인가? 뒤의 단계들은 이 입력에 맞춰 fit됐는가? 유용하거나 해로운 변화가 어디서 생겨 최종 graph까지 어떻게 남는지 보여 줘.”
- **Label을 더 만들지 정할 때:** “다음 batch는 놓친 positive와 그에 대응하는 hard negative를 무엇으로 보완하는가? 이 화면으로 내가 판단할 수 있는가? 모델 입력으로도 그 차이를 구분할 수 있는가?”
- **계산을 시작할지 정할 때:** “어떤 관찰이 이 실험의 근거인가? 처음 유용한 판단을 할 수 있는 시점은 언제이고 비용은 얼마인가? 긍정적 결과, 차이 없음, 부정적 결과가 나왔을 때 각각 다음에 무엇을 할 것인가?”
- **중단할지 정할 때:** “완료된 비교에서 낮은 점수를 얻은 경우, 학습이 부족한 경우, 실행이 끝나지 않은 경우, 시험하지 않은 아이디어를 나눠 줘. 실행 가능한 질문 중 가장 유망한 것은 무엇이며, 남은 예산을 쓸 가치가 없다고 보는 이유는 무엇인가?”

그 답을 바탕으로 interface를 고칠지, 원인을 확인한 학습 조건 하나를 바꿀지, 고정한 recipe를 확인할지, 비교한 구성을 탈락시킬지, 비용 때문에 한 방향을 미룰지 정할 수 있다. 매번 전면 점검을 새로 하기보다 설명들을 구분하는 자료 몇 가지를 요청하겠다. 사례 화면, learning curve, candidate coverage, 달라진 graph가 그런 자료다.

구현 작업은 기준 시스템을 유지하는 일과 대안을 개발하는 일로 나눌 생각이다. 실제 판단에 영향을 주는 결과가 나오면 그 부분을 집중 검토하되, 질문과 예산, 해석에 대한 책임은 내가 맡아야 한다. 현재 내 역량으로도 가능한 역할 분담이다. 모든 trainer를 직접 쓰거나 모든 코드를 줄마다 검토할 필요는 없지만, 모델에 무엇을 가르치고 있는지, 어떤 결과가 나오면 계획을 바꿀지는 이해해야 한다.

내가 직접 배울 것도 여기서 분명해진다. Engineering에서는 데이터·모델·interface 사이의 경계를 알아보고 전체 예측 경로를 확인할 수 있어야 한다. Domain에서는 label이 모호한 경우와, 주어진 입력만으로 모델이 바로잡을 수 없는 오류를 구분해야 한다. Machine learning에서는 필요한 예제가 없는지, 학습 목표가 잘못됐는지, optimization에 실패했는지, 충분히 개발하고도 더 나쁜 시스템인지를 나눠 봐야 한다. 이 역량들은 서로 연결돼 있다. 세 분야를 따로 공부하는 것보다 실제 오류를 해결하면서 함께 익히는 편이 내게 더 도움이 될 것이다.

### 실험이 도는 동안 다음 판단을 준비하기

Adapter를 어떻게 만들지 정하지 않았거나 학습 데이터가 없고, 현재 결과를 어떻게 해석할지도 모른다면 다음 후보가 준비됐다고 할 수 없다. 실행 중에는 다음에 가능한 비교를 준비하고, 그 비교에 무엇이 필요한지 기록하겠다. 할 만한 작업이 남지 않았다면 그 구체적인 이유로 계산 자원을 줄이면 된다. 사용률을 높이려고 GPU를 채울 필요는 없다.

연구를 진행하는 하루나 이틀마다 예측, 후보에 대한 판단, 막힌 문제의 해결, 또는 바로 실행할 다음 비교 중 하나는 남겨야 한다. 유용한 CPU 진단이 그날의 가장 좋은 결과일 수도 있다. 그런 결과 없이 날이 반복된다면, 대여 시간이 더 지나가기 전에 시간 배분부터 다시 봐야 한다.

마감이 가까워지면 학습과 평가를 한 번 더 하는 데 실제로 걸리는 시간과 비용 때문에 선택지가 줄어든다. 그래서 초기 계획이 중요하다. 학습 목표와 representation, 그 출력을 활용하는 다음 단계를 만드는 데는 시간이 걸린다. 마지막에 비슷한 제출 중 무엇을 고를지 더 신중하게 결정하는 것만으로 그 개발을 대신할 수는 없다.

## 마무리

하고 싶은 일은 많았지만, 그 일들을 실험으로 이어 갈 기반은 충분히 갖추지 못했다. 유용한 것을 배웠고, 다른 참가자도 활용할 수 있는 모델을 학습했으며, 실제 개선도 만들었다. 그러나 코드를 고치고 시스템을 확장하는 데 너무 오래 썼다. 이후에는 점점 더 세밀하게 조정된 pipeline을 기준으로 대안의 가치를 판단했다. 그 흐름을 바꾸려 했을 때는 남은 개발 작업을 일정 안에 끝내기 어려워져 있었다.

다른 계획을 택했다면 얻었을 점수는 알 수 없다. 대신 다음에 무엇을 다르게 할지는 더 정확히 말할 수 있다. 믿을 만한 예측 경로 하나를 일찍 만들고, 빠진 학습 과제가 무엇인지 설명할 수 있을 때까지 오류를 살펴보고, 서로 맞물려 작동하는 대안 몇 개를 개발하며, 각 단계에서 얻은 근거에 맞춰 계속할지 판단하는 것이다. 이 프로젝트에서 다음 연구로 가져가고 싶은 실질적인 배움은 여기에 있다.

<details markdown="1">
<summary>부록: 이미 제출한 후보 안에서의 작은 차이</summary>

다음은 10월 2일 계정 조회에서 확인한 서로 관련된 제출 다섯 개다. 이 표는 전체 구성의 기록이며, component 하나만 바꾼 ablation은 아니다.

| 버전 | 주요 변경 또는 역할 | Public | Private | 최종 선택 |
|---|---|---:|---:|---|
| v89 | Feature-E 구성; 진단용 | 0.94326 | 0.91752 | 아니요 |
| v90 | Feature averaging을 끈 validity control | 0.94662 | 0.91217 | 아니요 |
| v92 | Registration과 motion-repair 경로 | 0.94569 | 0.91418 | **B** |
| v93 | v92의 association head를 H1으로 교체 | **0.94921** | **0.91920** | **A** |
| v94 | H4X two-seed head ensemble; 진단용 | 0.94811 | **0.92050** | 아니요 |
{: #biohub-table-1 .biohub-table .biohub-numeric .biohub-score style="--c1: 12%; --c2: 42%; --c3: 15%; --c4: 15%; --c5: 16%; --label2: '주요 변경 또는 역할'; --label3: 'Public'; --label4: 'Private'; --label5: '최종 선택'; --table-min: 42rem;" }
조회된 제출 중 COMPLETE 상태이고 Private 점수가 있는 162개를 S, 내가 선택한 두 제출을 P={v93,v92}라고 하자. 기록된 best-of-two 규칙에 따른 최종 점수는 0.91920이었다. 이미 제출한 후보 안에서, 결과를 알고 나서 계산할 수 있는 차이는 다음과 같다.

$$
\begin{aligned}
L_{\mathrm{select}}
&=\max_{v\in S}q(v)-\max_{v\in P}q(v)\\
&=0.92050-0.91920\\
&=0.00130.
\end{aligned}
$$

V93은 이 후보군에서 두 번째로 높았다. V92 대신 v90이나 v89를 골랐어도 best-of-two 점수는 달라지지 않는다. V94는 당시 기록한 다음 단계 진입 조건을 통과하지 못한 진단용 제출이었다. 실제 배포한 H4X 구성과 로컬에서 비교한 H4 구성도 동일하지 않았다.

이 차이는 회고를 시작하게 한 성능의 격차에 비하면 작다. Private 점수가 공개됐으므로 이 차이는 계산할 수 있다. 그러나 더 나은 초기 계획이나 supervision, 실행하지 않은 모델이 점수를 얼마나 올렸을지는 계산할 수 없다.

{::nomarkdown}
<a class="popup img-link biohub-figure" href="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-01-private-submissions.png?v=7663b97f998c">
<picture>
  <source media="(max-width: 620px)" srcset="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-01-private-submissions-mobile.png?v=5477ac31e139">
  <IMG class="biohub-fig-01-private-submissions" src="{{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/fig-01-private-submissions.png?v=7663b97f998c" alt="제출한 다섯 버전의 Private 점수" width="1800" height="1030" loading="lazy">
</picture>
</a>
{:/nomarkdown}

_그림 3. 10월 2일 공식 계정 조회에서 확인한 aggregate Private 점수다. 나는 v93과 v92를 선택했고, 조회된 점수 이력에서는 v94가 가장 높았다. 이 그림은 기존 제출 후보 안에서의 작은 차이를 보여 준다._

</details>

## 출처와 근거

제출 관련 수치는 10월 2일 공식 계정 조회에서 가져왔다. 연구 과정은 7월 첫 RunPod 로그, 7월 15일 structural roadmap, 7월 28·30일 coupled-pipeline 분석, 8월 3일 portfolio 기록, 8월 20일 실행 상태, 날짜가 남은 학습·실험 기록, 9월 17일 재검토 문서와 대조했다. 초기에 시간과 token을 많이 썼다는 설명은 내 기억이며, 구체적으로 언급한 중단 사례는 기록에서 확인했다. 전체 GPU 사용률이나 비용을 모두 복원한 것은 아니다.

앞선 과정과 필요한 정의는 [1편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-1-Learned-Lineage-Graphs-KR/), [2편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-2-OOF-Structural-Diagnostics-KR/), [5편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-5-A-Local-Optimum-Built-One-Step-at-a-Time-KR/), [6편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-6-When-Local-Validation-Ran-a-Different-Pipeline-KR/), [7편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-7-Deciding-by-Logic-and-What-Validation-Must-Reproduce-KR/), [8편]({{ site.baseurl }}/posts/BioHub-Cell-Tracking-Working-Note-8-What-Went-Into-Choosing-the-Final-Two-KR/)에 기록했다.

대회 종료 후 공개된 write-up은 각 해법을 만든 참가자들이 직접 쓴 기록이다. 구현 내용과 로컬 측정값은 작성자가 보고한 내용으로 인용했고, 함께 밝힌 검증상의 한계도 반영했다. 이들의 학습을 다시 실행하거나 비교 결과를 독립적으로 재현하지는 않았다. 당시 내 근거로도 시도할 이유가 있었던 개발 질문을 찾는 데 도움이 되지만, 실행하지 않은 방향의 결과를 보장해 주는 자료는 아니다.

사람이 직접 검토한 부분은 9월 4–6일 batch 선정, 저장한 판정, ingest 보고서, verifier 비교, 실제 notebook 출력과 9월 14일 S4 실험 카드·완료 결과를 확인했다. 링크한 labeling 기록에 측정값과 출처를 정리했다. 이 글을 쓰려고 학습을 다시 돌리지는 않았다. Aggregate Private 점수만으로 hidden graph 안의 오류 원인을 알 수도 없다.

[조회 결과·실험 근거 요약]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/evidence-summary.json)과 [이 글에 사용한 원기록·write-up]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/research-depth-evidence.json)에 근거 자료를 정리해 두었다. 표지와 본문 그림은 내가 직접 제작했으며, 외부 자료를 사용한 부분에는 출처를 달았다. 표지는 시리즈의 원본 일러스트를 다시 사용했다. 본문 그림 1·2로는 실험 설계와 판단 범위를 설명했고, 부록 그림 3에는 공식 제출 결과를 그렸다. [그림 출처·제작·검수 기록]({{ site.baseurl }}/assets/img/posts/2026-10-02-biohub-working-note-9/figure-sources.json).
