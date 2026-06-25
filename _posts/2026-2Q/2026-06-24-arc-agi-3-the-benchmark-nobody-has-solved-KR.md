---
title: "ARC-AGI-3: 아직 아무도 풀지 못한 벤치마크 (KR)"
date: 2026-06-24 21:00:00 +0900
categories: [AI, Kaggle]
tags: [arc-agi, arc-agi-3, benchmarks, reinforcement-learning, agents, world-models, agi, korean]
math: true
pin: false
---

<link rel="stylesheet" href="{{ site.baseurl }}/assets/css/arc-agi-3.css">

# ARC-AGI-3: 아직 아무도 풀지 못한 벤치마크

> **작성 시점 메모 (2026-06-25).** ARC-AGI-3는 아직 진행 중인 대회입니다. 평가 환경 자체와 리더보드 해석에 필요한 정보도 계속 갱신되고 있습니다. 아래에서 쉽게 바뀌지 않는 내용은 공식 ARC Prize 페이지, 문서, technical report를 기준으로 확인했지만, 리더보드 순위·노트북 런타임·milestone claim 규칙은 달라질 수 있습니다. 특히 점수 숫자는 "이 시점의 관측값"으로 읽어야 합니다.

대회 링크:  
[ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)

공식 벤치마크 페이지:  
[ARC Prize 2026 - ARC-AGI-3 Competition](https://arcprize.org/competitions/2026/arc-agi-3)

기술 보고서:  
[ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence](https://arxiv.org/abs/2603.24621)

<figure class="arc-agi-figure arc-agi-hero">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/arc-agi-3-banner.jpg" alt="작은 grid game 환경들이 모인 배경 위에 ARC-AGI-3 제목이 적힌 배너 이미지">
  <figcaption>ARC-AGI-3는 ARC 계열 벤치마크를 정적인 grid 변환 문제에서, 설명서 없는 interactive game 환경으로 옮겨 놓습니다.</figcaption>
</figure>

> **한 문단 요약.** ARC-AGI-3는 "reasoning"을 정답으로 출력하는 능력이 아니라, 실제 환경 안에서 *행동으로 증명해야 하는 능력*으로 다시 묻습니다. 에이전트는 설명서 없는 낯선 2D grid 게임에 들어가고, 얼마나 적은 시행착오로 규칙을 배워 이기는지로 평가됩니다. 기준은 같은 게임을 처음 본 사람의 행동 수입니다. 2026년 중반 기준으로 가장 놀라운 사실은 이것입니다. Frontier LLM도, 이 대회를 겨냥해 만든 agent도 **모두 2%에 못 미칩니다.** 이것은 측정 오류가 아닙니다. 이 벤치마크의 핵심이고, 제가 지금 Kaggle에서 이 문제를 가장 흥미롭게 보는 이유입니다.

## 왜 이 글을 쓰는가

저는 ARC-AGI-3를 꽤 진지하게 들여다보고 있습니다. 그리고 보통 글을 쓰면서 생각을 정리합니다. 이 글은 대회가 실제로 무엇인지, 점수가 어떻게 계산되는지, 리더보드 숫자를 왜 조심해서 읽어야 하는지, 현재 방법들이 어디에서 막히는지, 그리고 제가 어떤 방향으로 접근할지를 가능한 한 정확하게 정리하려는 시도입니다.

주장은 꽤 분명합니다. **리더보드의 숫자를 그대로 믿으면 안 됩니다.** 이유는 하나가 아닙니다. 점수 단위가 헷갈리기 쉽고, 오래된 scoring rule로 계산된 submission이 섞여 있으며, 한때는 public game source를 이용한 exploit에 가까운 접근도 있었습니다. 겉으로 보이는 숫자보다 실제 신호는 훨씬 낮고, 훨씬 균일합니다.

이 글의 목표는 여섯 가지입니다.

1. ARC-AGI-1, ARC-AGI-2, ARC-AGI-3가 각각 어떤 문제였고, 왜 이런 순서로 발전했는지 설명합니다.
2. ARC Prize Foundation과 Kaggle competition이 어떤 맥락에서 이 대회를 운영하는지 설명합니다.
3. ARC-AGI-3의 클리어 기준, 채점 기준, RHAE 수식, human baseline의 의미를 초심자도 이해할 수 있게 풉니다.
4. StochasticGoose, Blind Squirrel, FORGE/BFS 계열처럼 이미 시도된 전략들이 무엇을 배웠고 어디서 막혔는지 정리합니다.
5. 실제로 참여하려는 사람이 어떤 배경 지식을 공부하고, 어떤 code부터 작성해야 하는지 작업 순서로 제안합니다.
6. toy example, local validation, 실패 진단 예시처럼 손에 잡히는 사례를 넣습니다.

### 처음 읽는 사람을 위한 안내도

먼저 용어부터 맞추겠습니다. ARC-AGI-3는 이름부터 낯섭니다. 하지만 핵심은 어렵지 않습니다.

| 용어 | 쉽게 말하면 |
|---|---|
| **ARC** | 색이 칠해진 작은 grid를 사용하는 reasoning benchmark 계열입니다. 자연어 상식 문제라기보다, 보드 위의 규칙을 추론하는 문제에 가깝습니다. |
| **AGI** | 여기서는 거창한 선언이 아닙니다. "새로운 기술을 얼마나 효율적으로 습득하는가"를 측정하려는 좁은 의미로 쓰입니다. |
| **Task / game / environment** | 저마다 숨은 규칙을 가진 작은 세계입니다. 에이전트는 grid를 보고 action을 선택합니다. |
| **Level** | 한 game을 구성하는 단계입니다. 보통 뒤 level로 갈수록 앞에서 배운 mechanic이 조합됩니다. |
| **Agent** | 제출하는 프로그램입니다. 현재 frame을 보고 다음 action을 고릅니다. |
| **Action** | 위/아래/왼쪽/오른쪽 이동, interact, 좌표 클릭, undo, reset 같은 환경 입력입니다. |
| **World model** | "이 action을 하면 무엇이 바뀌는가"에 대한 agent 내부의 모델입니다. |
| **Generalization** | 공개된 25개 게임을 외우는 것이 아니라, 처음 보는 hidden game에서도 잘하는 능력입니다. |
| **Human baseline** | 같은 game을 처음 본 사람이 몇 action으로 level을 깼는지에 대한 기준값입니다. ARC-AGI-3 점수는 이 기준과 비교됩니다. |
| **Scorecard / replay** | agent가 어떤 game에서 어떤 action을 했고, 어떤 score를 받았는지 기록한 실행 결과입니다. |

가장 쉬운 비유는 이렇습니다. 설명서 없는 퍼즐 게임을 처음 열었다고 해봅시다. 키를 눌러 보고, 무엇이 움직이는지 보고, 규칙을 추측하고, 목표를 짐작한 뒤, 적은 행동 수로 클리어합니다. ARC-AGI-3는 AI에게 바로 그 일을 시킵니다.

### 어떤 배경 지식이 필요한가

처음부터 RL 논문을 잔뜩 읽고 시작할 필요는 없습니다. 오히려 게임 frame 하나도 제대로 저장하지 않은 상태에서 deep RL부터 붙잡으면 진행이 더 느려질 가능성이 큽니다. ARC-AGI-3는 연구 난도가 높은 문제이지만, 첫걸음은 꽤 구체적입니다.

시작 전에 있으면 좋은 최소 배경은 이 정도입니다.

| 배경 | 왜 필요한가 | 초심자 기준 |
|---|---|---|
| Python | starter kit가 Python 중심입니다. | 파일 하나를 수정하고, 명령을 실행하고, 에러 메시지를 읽을 수 있으면 됩니다. |
| 배열과 grid | observation이 2D integer grid입니다. | 두 grid를 비교해서 어떤 cell이 바뀌었는지 찾을 수 있으면 됩니다. |
| Search | agent는 action sequence를 시도해야 합니다. | BFS, 깊이 제한 search, visited-state set 정도면 시작할 수 있습니다. |
| Logging | interactive agent는 최종 점수만 봐서는 디버깅이 거의 불가능합니다. | frame, action, state hash, action을 고른 이유를 저장해야 합니다. |
| 기본 ML 용어 | CNN, value model, world model 같은 말이 자주 나옵니다. | 모델이 무엇을 예측하려는지만 이해해도 충분합니다. |
| Kaggle code competition | 제출 파일은 Kaggle이 실행하는 notebook입니다. | 평가 중 internet이 꺼지고, code가 self-contained여야 한다는 점을 알아야 합니다. |

반대로 첫날부터 필요하지 않은 것도 있습니다.

- 거대한 language model은 필요하지 않습니다. final evaluation에서는 hosted API를 호출할 수 없습니다.
- 멋진 neural architecture도 바로 필요하지 않습니다. 로그가 부실한 fancy model보다, 관찰 가능한 graph search가 훨씬 낫습니다.
- public 25개 game을 전부 손으로 풀 필요도 없습니다. 몇 개를 직접 해보는 것은 좋지만, 외우는 순간 hidden game에는 도움이 안 됩니다.
- Kaggle forum의 모든 논쟁을 이해할 필요도 없습니다. 공식 docs, starter kit, local game 하나로 시작하면 됩니다.

초심자에게는 이 태도가 가장 중요합니다.

> 각 game을 처음 보는 작은 기계처럼 다룬다.

agent가 처음 해야 할 일은 영리한 척하는 것이 아닙니다. 실험하고, 무엇이 바뀌었는지 기록하고, 이미 쓸모없다고 확인한 action은 반복하지 않고, 다음 실험을 조금 더 낫게 만드는 것입니다.

이 벤치마크가 교육적으로 좋은 이유도 여기에 있습니다. 디버깅을 잘할 때 거치는 과정을 그대로 요구하기 때문입니다.

```text
주의 깊게 관찰한다
작은 개입을 한다
결과를 기록한다
가설을 수정한다
다음 개입을 고른다
```

이 loop를 만들 수 있다면, hand-written heuristic만으로도 이미 ARC-AGI-3 agent의 뼈대를 만든 셈입니다.

## 1. 계보: 정적인 puzzle에서 행동하는 game으로

ARC-AGI, 즉 *Abstraction and Reasoning Corpus*는 François Chollet이 제안한 intelligence 측정 방식에서 출발합니다. 핵심은 이미 가진 skill의 크기가 아니라 **skill-acquisition efficiency**입니다. 말하자면 "처음 보는 문제에서 얼마나 적은 정보와 시도로 새 기술을 배워내는가"입니다.

이 대회의 뿌리는 2019년 Chollet의 논문 *On the Measure of Intelligence*입니다. 여기서 중요한 관점은 "AI가 이미 훈련된 지식을 얼마나 많이 저장했는가"가 아닙니다. 오히려 다음 질문에 가깝습니다.

> 낯선 문제를 만났을 때, 시스템은 얼마나 적은 경험으로 새 규칙을 배워 쓸 수 있는가?

이 관점에서는 benchmark가 단순 지식 시험이면 안 됩니다. 인터넷에 있는 사실을 많이 외운 모델이 이기면, 그것은 general intelligence라기보다 memorization일 수 있습니다. 그래서 ARC 계열은 자연어 trivia나 상식 문제 대신, 작은 grid 안에서 새 규칙을 추론하게 만듭니다. 사람에게는 비교적 자연스럽지만, 훈련 분포 바깥의 문제를 다루지 못하는 모델에게는 어렵습니다.

ARC Prize Foundation은 이 benchmark를 open competition 형태로 운영하면서 두 가지를 목표로 합니다.

- **측정**: frontier model이 정말 새로운 문제에 적응하는지 확인합니다.
- **유도**: 단순 leaderboard 경쟁이 아니라 open-source solution과 paper를 통해 연구 방향을 공개적으로 축적합니다.

Kaggle은 여기서 실행 플랫폼 역할을 합니다. 참가자는 code를 제출하고, Kaggle은 internet이 꺼진 환경에서 agent를 실행합니다. 즉, 이 대회는 "좋은 prompt를 만들어 hosted LLM을 잘 부르는 능력"이 아니라, **스스로 실행되는 agent를 만드는 능력**을 봅니다.

- **ARC-AGI-1 (2019)** — 정적(static)입니다. 몇 개의 input→output grid 예시를 보고 변환 규칙을 추론한 뒤, 빠진 output grid를 만듭니다.
- **ARC-AGI-2 (2025)** — 여전히 정적이지만 더 어렵고 조합적입니다. 여러 규칙이 함께 작동하고, 문맥에 따라 규칙이 달라집니다.
- **ARC-AGI-3 (2026)** — interactive입니다. 이제 output grid 하나를 맞히는 문제가 아닙니다. 에이전트가 직접 환경을 탐색하고, 규칙을 배우고, 목표를 추론하고, action sequence를 계획해야 합니다.

세 버전의 차이를 숫자까지 포함해서 보면 이렇습니다. 단, 아래 점수들은 서로 다른 평가 맥락에서 나온 값이라 정밀한 사과 대 사과 비교는 아닙니다. 여기서 봐야 할 것은 방향성입니다.

| 항목 | ARC-AGI-1 (2019) | ARC-AGI-2 (2025) | ARC-AGI-3 (2026) |
|---|---|---|---|
| Format | static grid puzzle | 더 어렵고 조합적인 static grid puzzle | interactive game environment |
| Instructions | input-output demo pair | input-output demo pair | 자연어 설명 없음. interaction으로 규칙을 발견해야 함 |
| Best reported AI score | o3가 semi-private에서 public compute 기준 75.7%, high-compute 기준 87.5%를 기록했고, public eval report에서는 90%를 넘었습니다. | ARC Prize 2025 Kaggle private top score 24.03% | 30-day preview winner가 12.58%를 기록했지만, full launch benchmark에서는 훨씬 낮아졌습니다. |
| Human reference | original private task에서 human tester가 각각 97-98%를 풀었고, 둘을 합치면 100%를 풀었습니다. 요약 자료에서는 예전 rough benchmark로 ~85%가 자주 쓰입니다. | public eval sample의 average human performance는 66%였습니다. 선별된 evaluation task들은 human-solvable 조건을 통과했습니다. | 포함된 environment는 사람이 100% 풀 수 있도록 calibration되었습니다. AI 점수는 human action baseline 대비 효율로 계산됩니다. |
| Scoring | task accuracy, 대체로 solved/not solved | accuracy에 cost-per-task reporting이 함께 중요해짐 | Relative Human Action Efficiency. 사람 대비 action efficiency가 핵심 |
| Dataset scale | 400 public train, 400 public eval, 100 semi-private, 100 private | 1,000 public train, 120 public eval, 120 semi-private, 120 private | 25 public demo environments, 55 semi-private, 55 fully private. 각 environment 안에 여러 level이 있음 |

특히 ARC-AGI-3의 12.58%는 조심해서 읽어야 합니다. 이 값은 **preview competition**의 hidden 3 games에서 나온 점수입니다. 현재 full competition의 일반적인 성능 수준을 뜻하지 않습니다. 같은 계열의 agent도 full launch benchmark에서는 크게 떨어졌고, 바로 그 점이 이 대회의 핵심입니다. 몇 개의 preview environment를 잘 푼다고 해서, 전체 hidden set에 일반화한다는 뜻은 아닙니다.

전환의 핵심은 이것입니다.

> 지능을 "고정된 데이터셋에서 패턴을 맞히는 능력"이 아니라, "낯선 환경에서 적응적으로 행동하는 능력"으로 본다.

이 전환 때문에 현재 시스템들이 급격하게 무너집니다.

<figure class="arc-agi-figure">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/lineage.svg" alt="ARC-AGI lineage from static puzzles to interactive games">
</figure>

작은 toy example로 보면 차이가 더 분명합니다.

정적인 ARC 문제에서는 이런 식입니다.

```text
input grid:   왼쪽에 빨간 사각형 하나
output grid:  같은 빨간 사각형이 오른쪽으로 이동
```

해야 할 일은 변환 규칙을 추론하고 output을 쓰는 것입니다. 세계 안에서 행동하지 않습니다. 답을 제출할 뿐입니다.

ARC-AGI-3에서는 비슷한 난이도의 아이디어도 이렇게 바뀝니다.

```text
frame 0:   빨간 사각형, 파란 문, 초록 버튼
action 1:  오른쪽으로 이동
frame 1:   빨간 사각형이 오른쪽으로 이동
action 2:  버튼을 누름
frame 2:   파란 문이 사라짐
```

이제 에이전트는 interaction을 통해 배워야 합니다. 이동이 가능하다는 것, 버튼이 어떤 효과를 낸다는 것, 문이 무언가를 막고 있었다는 것, 어쩌면 최종 목표가 문 뒤에 있다는 것까지 추론해야 합니다. "output grid를 완성하라"보다 훨씬 더 까다로운 문제 설정입니다.

## 2. 실제로 무엇을 측정하는가

공식 설명에서 ARC-AGI-3는 agentic intelligence를 네 가지 능력으로 나눕니다.

1. **Exploration** — 정보가 그냥 주어지지 않습니다. 행동해서 알아내야 합니다.
2. **Modeling** — 관측한 frame을 바탕으로 world model을 만들어야 합니다. 즉, 다음 상태를 예측할 수 있어야 합니다.
3. **Goal-setting** — 목표가 자연어로 주어지지 않습니다. 무엇을 향해 가야 하는지 스스로 추론해야 합니다.
4. **Planning & execution** — 추론한 목표까지 가는 action sequence를 계획하고, 중간 피드백에 따라 수정해야 합니다.

이 네 가지는 추상적인 슬로건이 아닙니다. 각각 매우 구체적인 실패 모드와 연결됩니다.

| 능력 | 머릿속에 그릴 모습 | 없으면 생기는 문제 |
|---|---|---|
| Exploration | action을 눌러 보며 어떤 object가 반응하는지 찾습니다. | 같은 무의미한 행동을 반복하거나 핵심 mechanic을 발견하지 못합니다. |
| Modeling | "버튼을 누르면 문이 열린다" 같은 cause-effect를 기억합니다. | action이 어떤 결과를 낼지 예측하지 못합니다. |
| Goal-setting | board가 암시하는 winning condition을 추론합니다. | mechanic 일부를 이해해도 어디로 가야 할지 모릅니다. |
| Planning & execution | 목표까지 짧은 action sequence를 고릅니다. | 사실을 알아도 solution path로 바꾸지 못합니다. |

많은 AI benchmark는 사실상 마지막 줄만 묻습니다. 답을 맞히는가? ARC-AGI-3는 그 앞의 과정까지 함께 묻습니다.

```text
행동한다 → 관측한다 → 모델을 수정한다 → 다시 행동한다
```

그리고 점수는 이 순환을 얼마나 효율적으로 수행하는지로 계산됩니다.

## 3. 게임은 어떻게 생겼는가

환경은 일부러 단순하게 만들어져 있습니다. 어려워야 하는 부분은 interface 조작이 아니라 reasoning이기 때문입니다.

- **Observation**: 최대 64×64 grid입니다. 각 cell은 0–15 사이의 integer color/state 값을 가집니다. 좌표 원점은 왼쪽 위 `(0,0)`입니다.
- **Actions**: `RESET`과 표준화된 `ACTION1`–`ACTION7` interface가 있습니다.
  - `RESET` — level을 다시 시작합니다.
  - `ACTION1`–`ACTION4` — 보통 위/아래/왼쪽/오른쪽 같은 단순 action입니다.
  - `ACTION5` — interact, select, rotate, attach/detach, execute 등 game마다 의미가 달라질 수 있습니다.
  - `ACTION6` — `(x, y)` 좌표 클릭입니다. 좌표는 0–63 범위입니다.
  - `ACTION7` — undo입니다.
- **Scale**: technical report 기준으로 25개 public-demo environment와 110개 hidden environment가 있습니다. hidden 110개는 55개 semi-private와 55개 fully private으로 나뉩니다.

프로그래밍에 익숙하지 않은 독자라면 grid를 작은 보드게임처럼 생각하면 됩니다.

<figure class="arc-agi-figure arc-agi-frame">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/arc-agi-3-game-frame.png" alt="색이 다른 object, player block, 진행 막대가 보이는 ARC-AGI-3 스타일 pixel-art game frame">
  <figcaption>실제 game frame을 보면 문제가 훨씬 덜 추상적으로 보입니다. agent는 색이 칠해진 cell들을 보지만, 무엇이 player이고 어느 object가 유용하며 목표가 무엇인지는 알려받지 못합니다.</figcaption>
</figure>

```text
0 0 0 0 0
0 2 0 3 0
0 0 1 0 0
0 4 0 0 0
0 0 0 0 0
```

여기서 숫자가 크다고 더 많은 양을 뜻하는 것은 아닙니다. 색이나 object type에 가깝습니다. 하지만 agent는 `1`이 player인지, `3`이 door인지, `4`가 button인지 알지 못합니다. action을 해보고 frame이 어떻게 바뀌는지를 보면서 의미를 배워야 합니다.

한 턴의 흐름은 이렇습니다.

| 단계 | 일어나는 일 |
|---|---|
| 1 | 환경이 가장 최근 frame을 보냅니다. grid, metadata, available actions, state가 포함됩니다. |
| 2 | agent가 memory를 갱신합니다. 이전 frame과 무엇이 달라졌는지 확인합니다. |
| 3 | agent가 action 하나를 고릅니다. 예를 들어 `ACTION1` 또는 좌표가 붙은 `ACTION6`입니다. |
| 4 | 환경이 action을 적용하고 새 frame을 돌려줍니다. |
| 5 | level을 클리어하거나, game over가 되거나, action budget이 끝날 때까지 반복합니다. |

어려운 점은 action 이름이 game마다 구체적으로 설명되지 않는다는 것입니다. `ACTION5`는 어떤 game에서는 "줍기"일 수 있고, 다른 game에서는 "회전"일 수 있습니다. `ACTION6`은 좌표 클릭이지만, 어느 좌표가 의미 있는지는 알려주지 않습니다. 그래서 처음 해야 할 일은 puzzle solution이 아니라 **control discovery**입니다.

### 프로그래머가 실제로 받는 것은 무엇인가

여기서부터 ARC-AGI-3는 일반적인 ML 문제와 완전히 달라집니다. 보통의 prediction 문제라면 이렇게 생각할 수 있습니다.

```python
answer = model(question)
```

하지만 ARC-AGI-3에서는 이런 식이 아닙니다. agent는 environment를 반복해서 호출하고, 돌아온 bundle을 해석하고, action 하나를 고르고, 그 action의 비용을 바로 치릅니다. API 형태로 단순화하면 대략 이런 loop입니다.

```python
# game을 시작하거나 reset한다.
bundle = post("/api/cmd/RESET", {
    "game_id": game_id,
    "card_id": card_id,
})

guid = bundle["guid"]

while bundle["state"] == "NOT_FINISHED":
    action = choose_action(bundle, memory)

    payload = {
        "game_id": game_id,
        "card_id": card_id,
        "guid": guid,
    }
    if action["id"] == 6:
        payload["x"] = action["x"]
        payload["y"] = action["y"]

    next_bundle = post(f"/api/cmd/{action['name']}", payload)
    update_memory(memory, bundle, action, next_bundle)
    bundle = next_bundle
```

돌아오는 값도 교과서적인 `(observation, reward, done)` 한 줄이 아닙니다. 실제로는 이런 bundle에 가깝습니다.

```python
{
    "game_id": "ls20-016295f7601e",
    "guid": "2fa5332c-2e55-4825-b5c5-df960d504470",
    "frame": [[0, 0, 0, ...], [0, 3, 3, ...], ...],
    "state": "NOT_FINISHED",
    "levels_completed": 0,
    "win_levels": 254,
    "action_input": {"id": 6, "data": {"x": 12, "y": 34}},
    "available_actions": [1, 2, 3, 4, 6],
    "score": 0.0,  # API/toolkit response에 포함되는 경우가 많다
}
```

이 bundle에서 유용한 state를 뽑아내는 것도 참가자가 해야 할 일입니다.

| Field | 알려주는 것 | 알려주지 않는 것 |
|---|---|---|
| `frame` | 눈에 보이는 grid | object의 의미, 목표, 물리 규칙, 클릭할 만한 좌표 |
| `available_actions` | 지금 쓸 수 있는 action id | `ACTION6`에서 유용한 `(x, y)` 좌표 |
| `state` | 진행 중인지, 이겼는지, 끝났는지 | 얼마나 이기기에 가까운지 |
| `levels_completed` | 완료한 level 수 | 어떤 mechanic을 배웠는지, 왜 진행됐는지 |
| `win_levels` | 해당 environment의 전체 level 수 | 다음 level로 가는 방법 |
| `action_input` | 방금 반환된 frame을 만든 action | 그 action이 전략적으로 좋았는지 |
| local action counter | 지금까지 몇 action을 썼는지 | 보통 친절한 `remaining_turns_to_good_score` 같은 값은 없습니다. 직접 세거나 framework wrapper에서 읽어야 합니다. |

이 마지막 줄이 초반에 은근히 막막합니다. 환경은 `state`와 progress는 알려주지만, "좋은 점수를 받으려면 앞으로 몇 턴 남았습니다" 같은 친절한 신호를 주지 않습니다. action budget을 신경 쓰려면 agent가 직접 action call 수를 세고, level이 바뀐 시점과 reset 시점을 기록하고, 완료한 level당 action 수를 따로 계산해야 합니다.

처음 디버깅할 때의 느낌은 대략 이렇습니다. agent가 `ACTION6`을 `(12, 34)`에 보냈습니다. 그런데 다음 bundle을 보니 `frame`도 그대로이고, `levels_completed`도 그대로이고, `state`도 여전히 `NOT_FINISHED`입니다. 그럼 무엇을 배운 걸까요?

```python
def update_memory(memory, prev, action, nxt):
    prev_grid = prev["frame"]
    next_grid = nxt["frame"]

    changed = prev_grid != next_grid
    progress = nxt.get("levels_completed", 0) - prev.get("levels_completed", 0)
    state_changed = nxt["state"] != prev["state"]

    key = (hash_grid(prev_grid), canonical_action(action))
    memory["outcome"][key] = {
        "changed": changed,
        "progress": progress,
        "state_changed": state_changed,
        "next_hash": hash_grid(next_grid),
    }

    if not changed and progress == 0 and not state_changed:
        memory["no_ops"].add(key)
```

이 작은 함수도 이미 일종의 learning입니다. 거창한 game theory를 배운 것은 아니지만, "이 state에서 이 action은 거의 쓸모없었다"는 사실을 배웠습니다. 다음 call에서는 그 정보를 써야 합니다.

```python
def choose_action(bundle, memory):
    grid = bundle["frame"]
    legal = bundle["available_actions"]
    state_hash = hash_grid(grid)

    candidates = []
    for action_id in legal:
        if action_id == 6:
            for x, y in candidate_clicks(grid):
                candidates.append({"id": 6, "name": "ACTION6", "x": x, "y": y})
        else:
            candidates.append({"id": action_id, "name": f"ACTION{action_id}"})

    candidates = [
        a for a in candidates
        if (state_hash, canonical_action(a)) not in memory["no_ops"]
    ]

    return rank_candidates(candidates, bundle, memory)[0]
```

작은 model을 학습시킨다면 training data도 이런 transition에서 나옵니다. 예를 들어 StochasticGoose 계열의 action learning은 처음부터 최종 goal을 예측하려 하지 않습니다. 더 작은 문제부터 잡습니다.

```python
training_example = {
    "grid_before": prev["frame"],
    "action": encode_action(action),
    "label_changed": int(prev["frame"] != nxt["frame"]),
    "label_progress": int(nxt["levels_completed"] > prev["levels_completed"]),
}
```

CNN은 `P(frame changes | grid, action)` 또는 `P(progress | grid, action)` 같은 값을 배울 수 있습니다. 이건 꽤 유용합니다. agent가 명백한 no-op에 action budget을 덜 쓰게 만들기 때문입니다. 하지만 이것만으로 game을 푸는 것은 아닙니다. 탐색 낭비를 조금 줄인 것에 가깝습니다.

### 왜 그냥 torch나 LLM에 넣으면 잘 안 되는가

처음에는 이런 생각을 하기 쉽습니다.

```python
policy = CNN()
action = policy(torch.tensor(frame))
```

또는 이런 식입니다.

```python
prompt = f"이 grid에서 다음에 뭘 해야 해?\n{frame}"
action = llm(prompt)
```

이 발상 자체가 틀린 것은 아닙니다. 하지만 실제로 해보면 아주 구체적인 벽에 부딪힙니다.

| 단순한 생각 | 실제로 막히는 이유 |
|---|---|
| 25개 public game에서 CNN policy를 학습한다 | hidden game은 다릅니다. public mechanic을 외우면 "여기서 먹힌 행동"은 배우지만, "다음 낯선 game에서 무엇을 실험해야 하는지"는 잘 못 배웁니다. |
| RL을 처음부터 돌린다 | reward가 sparse하고 action이 비쌉니다. 실패한 탐색 action도 RHAE를 깎고, Kaggle runtime은 유한합니다. hidden game마다 수백만 step을 태우는 blank-slate RL은 현실적이지 않습니다. |
| next frame prediction을 학습한다 | transition data가 충분해야 의미가 있습니다. 초반에는 어떤 action이 transition을 만드는지도 모릅니다. |
| LLM에게 grid를 보여주고 물어본다 | final Kaggle evaluation에서는 hosted API를 쓸 수 없습니다. 게다가 64×64 grid를 text로 풀면 spatial structure가 흐려지고 context도 많이 씁니다. |
| local LLM을 notebook 안에서 돌린다 | 느리고 메모리를 많이 먹습니다. 그리고 말을 잘한다고 hidden mechanic을 아는 것은 아닙니다. 결국 action으로 배워야 합니다. |
| image classification처럼 푼다 | label이 "cat/dog"처럼 고정되어 있지 않습니다. 목표는 unknown rule 아래에서 action sequence를 찾는 것입니다. |
| brute force로 action sequence를 많이 시도한다 | scorer는 모든 action을 봅니다. 사람이 20 action으로 깬 level을 500 action으로 겨우 깨면, 점수는 거의 남지 않을 수 있습니다. |

그러니 실전적인 결론은 "neural net을 쓰지 말자"도 아니고, "LLM은 쓸모없다"도 아닙니다. 핵심은 model에게 **지금 가진 데이터로 학습 가능한 작은 역할**을 맡겨야 한다는 것입니다.

| Model의 역할 | Input | Target | 왜 가능한가 |
|---|---|---|---|
| Action-change predictor | `(frame, action)` | frame이 바뀌었는가 | 매 step 후 label을 바로 얻을 수 있습니다. |
| Progress predictor | `(frame, action)` | `levels_completed`나 score가 좋아졌는가 | sparse하지만 직접 측정 가능합니다. |
| State embedding | frame | 비슷한 state를 가깝게 표현 | state de-duplication과 loop detection에 도움이 됩니다. |
| Transition model | `(frame, action)` | next frame 또는 changed cells | transition이 충분히 쌓이면 short-horizon planning에 쓸 수 있습니다. |
| Goal hypothesis scorer | frame history | 어떤 object/state가 progress와 관련 있어 보이는가 | heuristic으로 시작하고, action outcome으로 검증할 수 있습니다. |

그래서 처음 부딪히는 진짜 질문은 "어떤 transformer를 쓸까?"가 아닙니다. 훨씬 더 기본적입니다.

```text
나는 정확히 무엇을 관측했나?
무슨 action을 보냈나?
무엇이 바뀌었나?
progress가 바뀌었나?
이 transition을 전에 본 적이 있나?
이번 관측이 다음 call을 어떻게 바꿔야 하나?
```

ARC-AGI-3가 강제로 만들게 하는 것은 바로 이 작은 loop입니다. 너무 평범해 보이지만, 이 loop가 없으면 비싼 model도 배울 자료가 없습니다.

### 클리어 기준: 무엇을 하면 "깼다"고 보는가

ARC-AGI-3에서 agent는 각 game의 level을 순서대로 진행합니다. 환경은 내부적으로 대략 세 가지 상태를 가집니다.

| State | 의미 |
|---|---|
| `NOT_FINISHED` | 아직 level이 진행 중입니다. agent가 다음 action을 골라야 합니다. |
| `WIN` | 현재 level의 objective를 만족했습니다. 이 level은 완료로 기록됩니다. |
| `GAME_OVER` | action budget이 끝났거나, game이 실패 상태에 들어갔습니다. 보통 `RESET`만 의미 있는 action입니다. |

초심자가 헷갈리기 쉬운 지점은 이것입니다. ARC-AGI-3의 objective는 자연어로 주어지지 않습니다. 환경이 "초록 버튼을 눌러 파란 문을 열고 빨간 block을 목표 지점에 놓으세요"라고 말해주지 않습니다. agent는 frame 변화와 score/state 변화만 보고 어떤 조건이 `WIN`으로 이어지는지 추론해야 합니다.

따라서 "클리어"에는 두 층이 있습니다.

1. **환경상의 클리어**: game state가 `WIN`이 됩니다.
2. **점수상 의미 있는 클리어**: human baseline에 비해 너무 많은 action을 쓰지 않고 `WIN`에 도달합니다.

두 번째가 중요합니다. brute-force로 500번 헤매다가 `WIN`에 도달했다면, 사람 기준으로는 "깼다"고 말할 수 있어도 ARC-AGI-3 점수에서는 거의 의미가 없을 수 있습니다. 이 대회는 "언젠가 맞혔는가"가 아니라 "처음 보는 규칙을 얼마나 효율적으로 습득했는가"를 묻기 때문입니다.

### Public과 private split

이 대회에서 반드시 이해해야 할 것은 public/private split입니다.

- **25 public games** — source code를 볼 수 있고, 훈련·디버깅에 쓸 수 있습니다.
- **110 hidden games** — 참가자와 agent가 볼 수 없습니다. 이 중 55개가 public leaderboard 쪽, 55개가 최종 private leaderboard 쪽입니다.

즉, Kaggle에서 "public leaderboard"라고 부르는 점수도 참가자가 미리 본 public game에서 계산되는 것이 아닙니다. leaderboard는 여전히 hidden game에서 계산됩니다. 그래서 public game level을 꽤 많이 풀었는데도 competition score가 **0.00**일 수 있습니다.

<figure class="arc-agi-figure">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/split.svg" alt="ARC-AGI-3 public demo, semi-private, and fully private split">
</figure>

### starter kit가 실제로 해주는 일

초심자에게 가장 현실적인 출발점은 [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)입니다. 공식 docs의 의도는 꽤 분명합니다. local에서 Python 파일 하나를 수정하고, 실제 game environment에서 실행해 보고, 준비가 되면 Kaggle notebook submission으로 올리는 흐름입니다.

대략 이런 순서입니다.

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
cd ARC-AGI-3-Kaggle-Starter
make setup
make play-local
make submit
make status
```

초심자가 집중할 파일은 보통 이것입니다.

```text
agent/my_agent.py
```

여기에 agent의 행동 선택 규칙을 구현합니다. 나머지는 game을 로드하고, loop를 돌리고, notebook을 packaging하고, Kaggle에 제출하기 위한 기반 코드입니다. 이 구분이 중요합니다. 제출 준비 코드와 agent의 핵심 로직을 혼동하면 초반 시간을 많이 쓰게 됩니다.

처음에는 작업 모드를 세 가지로 나눠서 생각하는 것이 좋습니다.

| Mode | 용도 | 기억할 점 |
|---|---|---|
| **Local offline** | public game에서 빠르게 실험합니다. | 개발용으로 가장 좋습니다. online scorecard는 없지만 rate limit도 없습니다. |
| **Online API / scorecard** | 공유 가능한 scorecard와 replay를 남깁니다. | 분석과 공유에 좋지만 느리고 rate limit이 있습니다. |
| **Kaggle submission** | 실제 leaderboard scoring입니다. | 제출 기회가 제한적입니다. 평소 디버깅용으로 쓰면 안 됩니다. |

공식 docs도 local/offline 실행을 개발용으로 권장합니다. Kaggle submission을 쓰기 전에 local harness가 최소한 다음 질문에 답할 수 있어야 합니다.

- agent가 어떤 game을 시도했는가?
- 어떤 level을 완료했는가?
- 각 level에서 action을 몇 개 썼는가?
- frame을 실제로 바꾼 action은 얼마나 되는가?
- 같은 state를 얼마나 반복했는가?
- click을 했다면 어디를 클릭했고, 왜 그 좌표를 골랐는가?
- 개발용 public game과 holdout public game에서 행동이 어떻게 달라졌는가?

이 질문들에 답하지 못하면 leaderboard 점수만으로는 거의 배울 수 없습니다. 실패했다는 사실은 알려주지만, 왜 실패했는지는 알려주지 않기 때문입니다.

### agent가 기억해야 하는 것

무작위 agent보다 한 단계 나아가려면 가장 먼저 필요한 것은 neural network가 아니라 memory입니다. ARC-AGI-3는 의미 없는 action 반복에 매우 불리한 점수를 줍니다. 그래서 초반 agent도 최소한 다음을 기록해야 합니다.

| 기억할 것 | 예시 | 왜 도움이 되는가 |
|---|---|---|
| State hash | grid와 주요 metadata의 hash | 같은 frame을 목적 없이 다시 방문하는 것을 막습니다. |
| Action outcome | `(state_hash, action) -> changed / no-op / error / win` | 이미 아무 변화가 없던 action을 반복하지 않습니다. |
| Changed cells | action 뒤 값이 바뀐 좌표들 | control object와 반응하는 object를 찾는 단서가 됩니다. |
| Available actions | frame metadata가 알려준 action set | sampling할 action 공간을 줄입니다. |
| Level identity | 현재 game과 level 정보 | level이 바뀌면 이전 가정을 reset할 수 있습니다. |
| Short trajectory | 최근 N개의 state/action | frame이 조금 바뀌어도 loop를 감지할 수 있습니다. |

가장 단순한 no-op detector는 이전 grid와 다음 grid를 비교하는 것입니다.

```python
def grid_changed(prev_grid, next_grid):
    return prev_grid != next_grid
```

실제 코드에서는 array comparison을 제대로 써야 하고, metadata를 어디까지 포함할지도 신중히 정해야 합니다. 하지만 아이디어는 단순합니다. 이 state에서 `ACTION1`을 했는데 같은 state가 나왔다면, 같은 state에서 `ACTION1`을 계속 고르지 않도록 막는 것입니다.

이건 사소해 보이지만 random agent가 action budget을 날리는 가장 큰 이유 중 하나를 줄여 줍니다. random exploration은 "무언가를 배웠다"와 "벽에 또 부딪혔다"를 구분하지 못합니다.

## 4. Scoring: RHAE와 헷갈리기 쉬운 단위

ARC-AGI-3의 점수는 **Relative Human Action Efficiency**, 줄여서 **RHAE**입니다. 한 level $\ell$에 대한 기본 score는 다음과 같습니다.

$$
s_\ell \;=\; \left[\min\!\left(\frac{h_\ell}{a_\ell},\, 1.15\right)\right]^{2}
$$

여기서 $h_\ell$은 human baseline action count이고, $a_\ell$은 agent가 그 level을 클리어하는 데 쓴 action 수입니다.

human baseline은 대충 정한 숫자가 아닙니다. 공식 scoring methodology는 처음 보는 human player들의 action count를 기준으로 삼습니다. 평균이 아니라 upper median 계열의 기준을 쓰는 이유는, 한 명의 운 좋은 speedrun이나 한 명의 극단적으로 느린 run에 benchmark가 흔들리지 않게 하려는 것입니다. 즉, 이 기준은 "이론상 최단 경로"가 아니라 **처음 본 사람이 능숙하게 풀었을 때의 현실적인 효율**에 가깝습니다.

중요한 설계 선택이 세 가지 있습니다.

- **제곱합니다.** agent가 사람보다 2배 느리면 ratio는 0.5지만, score는 $0.5^2 = 0.25$입니다. 절반이 아니라 25%만 인정됩니다.
- **1.15 cap이 있습니다.** 사람이 10 action에 푼 level을 agent가 2 action에 exploit처럼 풀어도, level 하나가 전체 점수를 터무니없이 끌어올리지는 못합니다.
- **action cutoff가 있습니다.** human baseline이 $h$라면, agent가 대략 $5h$ action을 넘기는 순간 해당 level은 0점입니다.

이 설계는 꽤 의도적입니다.

| 설계 | 왜 필요한가 |
|---|---|
| Completion을 봅니다. | level을 실제로 끝내지 못한 agent가 높은 점수를 받으면 안 됩니다. |
| Efficiency를 봅니다. | random search로 언젠가 맞히는 것은 인간 수준의 skill acquisition이 아닙니다. |
| 제곱합니다. | 사람보다 조금 느린 것과 10배 느린 것을 강하게 구분합니다. |
| 1.15 cap을 둡니다. | 한 level에서 exploit성 초고속 solve가 전체 평균을 왜곡하지 못하게 합니다. |
| 뒤 level에 더 큰 weight를 둡니다. | 앞 level은 tutorial에 가깝고, 뒤 level은 mechanic 조합과 transfer를 묻기 때문입니다. |

game score는 level index로 가중한 평균입니다. 뒤 level일수록 더 중요하게 계산됩니다. 한 game에 $n$개 level이 있고, agent가 앞에서부터 $k$개 level을 순서대로 클리어했다면:

$$
S_g
\;=\;
\min\!\left(
\frac{\sum_{\ell=1}^{k} \ell}{\sum_{\ell=1}^{n} \ell},
\frac{\sum_{\ell=1}^{n} \ell\,s_{g,\ell}}{\sum_{\ell=1}^{n} \ell}
\right),
\qquad
S \;=\; \frac{1}{G}\sum_{g=1}^{G} S_g .
$$

첫 번째 항은 environment cap입니다. 앞 level 몇 개를 아주 빠르게 풀었다고 해도, 뒤 level을 못 푼 agent가 높은 game score를 받지 못하게 막는 장치입니다.

Python으로 쓰면 대략 이런 모양입니다.

```python
def rhae_score(games):
    """ARC-AGI-3 score를 fraction으로 반환한다. percent는 여기에 100을 곱한다."""
    game_scores = []

    for levels in games:
        weighted = 0.0
        completed_weight = 0
        total_weight = sum(range(1, len(levels) + 1))

        for i, level in enumerate(levels, start=1):
            human_actions, agent_actions, solved = level
            if solved and 0 < agent_actions <= 5 * human_actions:
                level_score = min(human_actions / agent_actions, 1.15) ** 2
                weighted += i * level_score
                completed_weight += i

        env_cap = completed_weight / total_weight
        env_score = weighted / total_weight
        game_scores.append(min(env_cap, env_score))

    return sum(game_scores) / len(game_scores)
```

<figure class="arc-agi-figure">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/rhae.svg" alt="ARC-AGI-3 RHAE scoring pipeline">
</figure>

### 숫자로 보는 예시

human baseline이 10 action인 level이 있다고 해봅시다.

| Agent action 수 | Ratio $h/a$ | 제곱 후 score | 의미 |
|---:|---:|---:|---|
| 10 | 1.00 | 1.00 | 사람과 같은 효율입니다. |
| 20 | 0.50 | 0.25 | 2배 느리면 25%만 인정됩니다. |
| 50 | 0.20 | 0.04 | 5배 느리면 4%만 인정됩니다. |
| 51 | cutoff | 0.00 | $5h$를 넘으면 0점입니다. |

여기서 직관이 한 번 꺾입니다.

> **풀었다고 충분하지 않습니다.**

랜덤 탐색으로 언젠가 클리어하는 agent는 level을 "완료"했을 수 있습니다. 하지만 사람이 10 action에 푼 것을 50 action에 풀면 4%만 인정됩니다. 이 대회가 묻는 것은 "결국 이겼나"가 아니라 "처음 본 문제를 얼마나 효율적으로 배웠나"입니다.

game cap도 중요합니다. 5-level game에서 agent가 level 1과 2만 완벽하게 풀었다고 해봅시다. 그러면 최대 game score는:

$$
\frac{1 + 2}{1 + 2 + 3 + 4 + 5}
\;=\;
\frac{3}{15}
\;=\;
20\%.
$$

뒤 level을 못 풀면 앞 level을 아무리 빨리 풀어도 game 전체 점수는 높아지지 않습니다. 뒤 level이야말로 mechanic을 정말 이해했는지를 묻는 곳이기 때문입니다.

### 단위 혼동

리더보드 점수는 **percentage**입니다. 이 부분이 정말 많이 헷갈립니다. leaderboard entry가 `0.46`이면 **0.46%**입니다. fraction으로는 0.0046입니다. 46%가 아닙니다.

이 차이는 엄청 큽니다.

```text
0.46   = 0.46%
46.0   = 46%
100.0  = 100%
```

저도 처음 자료를 읽을 때 이 단위에 한 번 걸렸습니다. "0.46 score"라고 쓰여 있으면 눈이 자동으로 46%처럼 읽습니다. 하지만 ARC-AGI-3 맥락에서는 0.46%입니다. 지금 참가자 전체가 ~2% 아래에 있다는 말은, 정말로 거의 바닥에 붙어 있다는 뜻입니다.

이렇게 읽고 나면 현재 상황이 훨씬 선명해집니다. 이것은 강한 generalizer끼리의 순위 경쟁이 아닙니다. 아직 아무도 제대로 풀지 못한 문제에서 누가 덜 못하고 있는가에 가깝습니다.

### RHAE가 개발 방식을 어떻게 바꾸는가

RHAE 때문에 ARC-AGI-3는 일반적인 win/loss game benchmark와 다르게 접근해야 합니다. local run에서 "level completed"가 찍혔다고 바로 좋은 agent가 아닙니다. 얼마나 비싼 completion이었는지를 같이 봐야 합니다.

human baseline이 20 action인 level을 두 agent가 모두 풀었다고 해봅시다.

| Agent | 사용한 action 수 | Level score | 해석 |
|---|---:|---:|---|
| A | 24 | $(20/24)^2 = 0.694$ | 조금 느리지만 의미 있는 클리어입니다. |
| B | 90 | $(20/90)^2 = 0.049$ | 풀긴 했지만 action budget을 거의 태운 결과입니다. |

completion count만 보면 A와 B는 같아 보입니다. RHAE에서는 전혀 다릅니다. 그래서 local report에는 최소한 다음이 들어가야 합니다.

- **completion count**
- **actions per completed level**
- **estimated RHAE per completed level**
- **no-op action rate**
- **repeated-state rate**
- **first state-changing action step**
- **first score-changing action step**
- **late-level reach rate**

마지막 metric도 중요합니다. game score는 뒤 level에 더 큰 weight를 둡니다. level 1을 반복해서 잘하는 것과 level 3, 4까지 도달하는 것은 다른 신호입니다. 실전에서는 public tutorial level 하나를 과하게 최적화한 agent보다, 초반 score는 낮아도 여러 game에서 더 깊은 level까지 가는 agent가 더 흥미로울 수 있습니다.

### 작은 local validation protocol

public game은 leaderboard game이 아닙니다. 그래도 투명하게 볼 수 있는 유일한 개발 set입니다. 작은 dataset처럼 다루는 편이 좋습니다.

```text
public games
→ development split
→ holdout split
→ final sanity split
```

단순한 split은 이렇게 잡을 수 있습니다.

| Split | 용도 | 규칙 |
|---|---|---|
| Development | heuristic과 model을 고칩니다. | log를 마음껏 보고 수정해도 됩니다. |
| Holdout | candidate들을 비교합니다. | 매 결과를 본 뒤 바로 tuning하지 않습니다. |
| Final sanity | accidental overfit을 확인합니다. | 실제 submission 전에만 봅니다. |

public game이 25개뿐이라 통계적으로 강한 split은 아닙니다. 그래도 모든 public game을 매번 tuning에 쓰는 것보다는 훨씬 낫습니다. development game에서는 좋아졌는데 holdout game에서 무너지면, Kaggle submission을 쓰기 전에 봐야 할 경고를 본 것입니다.

초심자용 report는 이렇게 단순해도 됩니다.

```text
agent: graph_noop_v03
games: 18 development, 7 holdout

development:
  completed_levels: 14
  median_actions_per_completed_level: 37
  no_op_rate: 0.31
  repeated_state_rate: 0.18

holdout:
  completed_levels: 3
  median_actions_per_completed_level: 92
  no_op_rate: 0.62
  repeated_state_rate: 0.44
```

이건 leaderboard에 올릴 만한 agent는 아닙니다. 하지만 디버깅 가능한 agent입니다. development game에서 배운 public game 전용 prior가 holdout으로 옮겨가지 않았고, holdout 실패의 주된 원인이 no-op과 loop라는 점을 알려줍니다. 단순한 `score = 0.00`보다 훨씬 쓸모 있는 신호입니다.

candidate를 비교할 때도 단일 metric 하나만 보지 않는 편이 좋습니다. 초반 local objective는 이런 식으로 잡을 수 있습니다.

$$
J_{\text{local}}
=
0.35 \cdot \text{completed\_level\_rate}
+ 0.25 \cdot \widehat{\text{RHAE}}
+ 0.20 \cdot \text{late\_level\_reach}
- 0.10 \cdot \text{no\_op\_rate}
- 0.10 \cdot \text{repeat\_state\_rate}.
$$

이 수식이 정답이라는 뜻은 아닙니다. 중요한 것은 어떤 agent를 고를 때 "많이 풀었나"만 보지 말고, "효율적으로 풀었나", "뒤 level까지 갔나", "action을 낭비하지 않았나"를 같이 보자는 것입니다. 나중에는 game별, level별로 weight를 다르게 줄 수 있지만, 처음에는 이런 단순한 composite score만으로도 후보 비교가 훨씬 안정됩니다.

## 5. 대회 구조: 규칙, 상금, 마감

ARC-AGI-3는 **ARC Prize 2026**의 핵심 track 중 하나입니다. Kaggle code competition으로 열렸고, 일반적인 prediction competition과는 조금 다릅니다.

- **평가 중 internet이 꺼집니다.** GPT, Claude, Gemini 같은 hosted model API를 호출하는 방식은 final Kaggle submission으로 쓸 수 없습니다.
- **상금을 받으려면 open-source가 필요합니다.** 정확한 milestone claim 절차는 주최 측 clarification을 봐야 하지만, prize-eligible solution은 공개되어야 합니다.
- **Notebook-only submission입니다.** Kaggle overview 기준으로 CPU/GPU notebook runtime은 최대 9시간입니다. ARC Prize docs는 local starter kit도 제공합니다.
- **Timeline**: 2026년 3월 25일 시작, 6월 30일과 9월 30일 milestone, 11월 2일 final submission, 11월 8일 paper due, 12월 4일 결과 발표입니다.
- **ARC-AGI-3 prize pool**: 총 $850K입니다. 100% agent를 위한 $700K Grand Prize, 확정 상금인 $75K Top Score Award, $75K milestone prize가 있습니다.

확정 상금은 다음처럼 나뉩니다.

| Pool | Distribution |
|---|---|
| Top Score Award | $40K / $15K / $10K / $5K / $5K |
| Milestone #1 | $25K / $10K / $2.5K |
| Milestone #2 | $25K / $10K / $2.5K |

Kaggle code competition이 처음이라면 핵심 차이는 이것입니다.

| 일반 prediction competition | ARC-AGI-3 |
|---|---|
| 보통 prediction CSV를 제출합니다. | code를 제출하고 Kaggle이 실행합니다. |
| test data는 고정된 row입니다. | evaluation은 interactive game loop입니다. |
| 모델은 offline에서 학습하고 숫자만 출력해도 됩니다. | agent가 반복적으로 행동하고 피드백을 받아야 합니다. |
| internet은 보통 training 이후 중요하지 않습니다. | internet이 꺼져 있어 hosted API를 쓸 수 없습니다. |

즉, 외부 LLM API를 감싼 harness는 연구용으로는 흥미로울 수 있지만, Kaggle final submission으로는 그대로 쓸 수 없습니다. 제출용 agent에는 필요한 구성요소가 모두 포함되어 있어야 합니다. 예를 들면 rule, heuristic, 작은 model, learned prior, search procedure, local memory 같은 것들입니다.

## 6. 현재 상황: 겉치레를 걷어내고 보면

현재 상황을 정직하게 요약하면 이렇습니다.

- **Frontier LLM: sub-1%.** Technical report의 release 당시 semi-private evaluation에서 Anthropic Opus 4.6 (Max)는 0.50%, Google Gemini 3.1 Pro Preview는 0.40%, OpenAI GPT 5.4 (High)는 0.20%, xAI Grok-4.20은 0.10%였습니다.
- **Purpose-built preview agent도 버티지 못했습니다.** 30-day preview winner였던 **StochasticGoose**는 hidden preview 3 games에서 12.58%를 받았지만, full launch benchmark에서는 0.25%로 떨어졌습니다.
- **Kaggle 참가자 전체도 sub-2%입니다.** 2026년 6월 말 public high-score 기준으로는 대략 1.2%대입니다. 숫자는 계속 변할 수 있지만 큰 그림은 그대로입니다. 아직 human efficiency에 가까운 시스템은 없습니다.

그래도 preview-era 결과표는 여전히 볼 가치가 있습니다. 어떤 전략 계열이 처음으로 신호를 냈는지 보여주기 때문입니다. 다만 이 표는 현재 Kaggle leaderboard가 아니라, **full benchmark 이전의 방법론 지도**로 읽어야 합니다.

| 기준 | Team / agent | Approach | Score | 완료한 level 수 |
|---|---|---|---:|---:|
| preview 1위 | StochasticGoose (Tufa Labs) | CNN + sparse-RL action learning | 12.58% | 18 |
| preview 2위 | Blind Squirrel | state-graph exploration + ResNet18 value model | 6.71% | 13 |
| 주목할 만한 preview method | Explore It Till You Solve It | training-free frame graph exploration | 3.64% | 12 |
| Frontier LLM reference | technical report의 best frontier LLM agents | LLM 기반 agentic prompting / control | <1% | 같은 표에서 직접 보고되지는 않았지만, level 수로 보면 몇 개 수준에 그칩니다. |
| Human reference | Human players | 탐색, 모델링, 목표 추론, 계획을 결합한 인간의 문제 풀이 | 100% | calibration된 environment 전체 |

이 표의 결론은 "CNN이 LLM보다 영원히 낫다"도 아니고, "graph search만 있으면 된다"도 아닙니다. 더 좁고 실용적인 결론은 이것입니다. **screenshot을 보고 말로 추론만 하는 방식보다, action을 써서 environment dynamics를 발견하는 방식이 훨씬 강합니다.** 하지만 preview에서 성과를 냈던 방식도 full hidden benchmark로 가면 일반화가 크게 흔들립니다.

<figure class="arc-agi-figure">
  <img src="{{ site.baseurl }}/assets/img/arc-agi-3/score-distribution.svg" alt="ARC-AGI-3 score distribution showing current systems below two percent">
</figure>

### 왜 현재 시스템이 어려워하는가

Frontier LLM은 language로 표현되는 문제와 training distribution 안에 있는 지식에 강합니다. ARC-AGI-3는 이 두 편안함을 모두 제거합니다.

| 일반 LLM benchmark | ARC-AGI-3 |
|---|---|
| 문제 지시가 자연어로 주어집니다. | 목표가 주어지지 않습니다. |
| 모델은 한 번 답합니다. | agent는 수십~수백 번 행동해야 합니다. |
| 입력은 주로 symbolic text입니다. | 입력은 변화하는 grid world입니다. |
| memorized pattern에 어느 정도 기댈 수 있습니다. | private game은 unseen이고 public demo와 다르게 설계됩니다. |
| 중간의 틀린 생각은 비용이 없습니다. | 잘못된 action은 action budget을 소모합니다. |

마지막 줄이 특히 가혹합니다. 일반 benchmark에서는 중간 reasoning이 틀려도 최종 답만 맞으면 됩니다. ARC-AGI-3에서는 틀린 exploratory move도 action입니다. 그리고 action은 점수 계산에서 가장 중요한 비용 단위입니다. agent는 배우기 위해 action을 써야 하지만, 너무 많이 쓰면 점수가 무너집니다.

예를 들어 사람이 12 action에 푼 level을 agent가 120 action에 풀었다면:

$$
\left(\frac{12}{120}\right)^2 = 0.01.
$$

그 level에서 인정되는 점수는 1%뿐입니다. brute-force로 이긴 것도 점수 관점에서는 실패와 크게 다르지 않을 수 있습니다.

### 왜 leaderboard가 "거짓말"처럼 보이는가

모두 sub-2%라면 왜 notebook에 "0.4" 같은 점수가 보일까요? 이유는 세 가지입니다.

1. **Unit confusion** — `0.4`는 0.4%입니다.
2. **Stale score** — Kaggle은 metric이 바뀌어도 오래된 submission을 자동으로 재채점하지 않습니다. 예전 rule로 계산된 점수가 섞여 있을 수 있습니다.
3. **Patched exploit** — 한동안 public game source를 disk에서 찾아 실제 simulator에 대해 exhaustive search를 돌리는 white-box trick이 가능했습니다. private leaderboard에는 맞지 않고, 해당 path는 이후 막힌 것으로 알려져 있습니다.

따라서 public notebook score를 기준점으로 삼는 것은 위험합니다. 중요한 것은 "이 notebook에 몇 점이라고 적혀 있나"가 아니라, **unseen game에서 같은 원리가 살아남는가**입니다.

## 7. 접근법들: 각각 무엇을 배우고 어디서 막히는가

먼저 용어를 번역해 둡니다.

| 용어 | 여기서의 의미 |
|---|---|
| **CNN** | image/grid의 local spatial pattern을 잘 보는 neural network입니다. |
| **Sparse reward** | 보상이 드물게 주어지는 상황입니다. 보통 level을 클리어하거나 milestone에 도달했을 때만 의미 있는 피드백이 생깁니다. |
| **State graph** | agent가 본 frame들을 node로, action을 edge로 만든 graph입니다. |
| **Value model** | 어떤 state/action이 성공에 가까운지 추정하는 model입니다. |
| **World model** | 작은 learned simulator입니다. "state S에서 action A를 하면 next state가 무엇인가"를 예측합니다. |
| **Intrinsic exploration** | 외부 reward만이 아니라 novelty, uncertainty, information gain으로 탐색하는 방식입니다. |
| **Planning** | action을 바로 실행하기 전에 가능한 future sequence를 탐색하는 것입니다. |
| **BFS / A\*** | 가능한 action sequence를 systematic하게 탐색하는 search algorithm입니다. BFS는 짧은 경로부터 넓게 보고, A\*는 heuristic으로 유망한 경로를 먼저 봅니다. |
| **Beam search** | 모든 후보를 보지 않고, 매 depth마다 상위 몇 개 후보만 남기는 search입니다. |
| **White-box / black-box** | game source나 내부 field를 볼 수 있으면 white-box, 오직 observation/action API만 쓰면 black-box에 가깝습니다. hidden evaluation에서는 black-box 가정이 더 안전합니다. |

중요한 구분은 **reactive** system과 **model-based** system입니다. reactive system은 "지금 어떤 action이 좋아 보이는가"를 묻습니다. model-based system은 "이 action을 하고, 그 다음 action을 하면 무엇이 일어날까"를 묻습니다. 사람이 새로운 game을 배울 때는 보통 두 번째 방식에 많이 의존합니다.

**(a) CNN + sparse-RL action prediction (StochasticGoose 계열).** 작은 CNN이 어떤 action이 frame을 바꾸는지 예측하게 하고, 변화가 있을 것 같은 action을 더 자주 시도합니다. click 좌표도 64×64 head로 예측할 수 있습니다.  
*장점:* preview winner로 검증된 계열이고, self-contained이며, sample-efficient합니다.  
*한계:* 본질적으로 reactive합니다. action이 "변화를 만든다"는 것과 "목표에 도움이 된다"는 것은 다릅니다.

**(b) State graph + value model (Blind Squirrel / just-explore).** 이미 본 state를 graph로 만들고, loop나 no-op action을 줄입니다. 점수가 좋아지는 path를 발견하면 그 path를 거꾸로 labeling해서 value model을 학습할 수 있습니다.  
*장점:* 방문한 상태에 대해서는 해석 가능하고, loop를 피하는 데 도움이 됩니다.  
*한계:* 이미 가본 frontier 안에서만 강합니다. 아직 가보지 않은 목표 상태를 상상하는 힘은 약합니다.

**(c) Learned world model + intrinsic exploration.** $(s, a, s')$ 경험으로 transition model을 배우고, 그 model 위에서 planning을 합니다. exploration은 random flailing이 아니라 uncertainty나 disagreement가 큰 곳을 향하게 합니다.  
*장점:* benchmark가 실제로 요구하는 modeling, goal inference, planning을 직접 겨냥합니다.  
*한계:* 어렵습니다. sparse reward에서 goal을 추론해야 하고, test-time learning이 불안정할 수 있으며, compute budget도 있습니다.

**(d) Frontier LLM harness.** 강한 language model을 감싸서 reasoning, memory, planning을 시키는 방식입니다.  
*장점:* public game에서는 강력한 reasoning scaffold를 만들 수 있습니다.  
*한계:* Kaggle evaluation에서 internet이 꺼지므로 hosted API 의존 방식은 final submission으로 부적합합니다. 또한 public game overfitting 위험이 큽니다.

**(e) Public-game source 기반 BFS / FORGE 계열.** local `Public_Examples` 폴더의 notebook들에서 가장 눈에 띄는 계열입니다. 예를 들어 `0-35-forge-v16-trigger-aware-bfs.ipynb`, `forge-arc-agi-3-agent.ipynb`, `arc-agi-3-hybrid-solver-bfs-cnn-heuristics.ipynb`, `ash-s-arc-agi-3-agent.ipynb`는 모두 public game source를 최대한 활용해 search를 강하게 돌리는 흐름을 보입니다. 핵심은 game class를 직접 실행해 보고, 실제로 효과가 있는 action을 찾아내고, BFS/A\*/beam search로 solution path를 찾는 것입니다. 여기에 hidden field나 transient field를 hash에 넣거나 빼면서 state 구분을 더 정교하게 만듭니다.  
*장점:* public game에서는 매우 강합니다. 어떤 action이 실제로 frame을 바꾸는지 미리 scan하고, click 좌표도 `_get_valid_actions()` 같은 내부 helper를 통해 좁힐 수 있으면 search가 훨씬 쉬워집니다.  
*한계:* 이 방식은 public source와 내부 구현에 기대는 부분이 많습니다. hidden/private game에서도 같은 방식의 source introspection이 가능하다고 가정하면 위험합니다. 따라서 이런 notebook은 "final solution"이라기보다, 어떤 search 장치가 도움이 되는지 배우는 참고 자료로 보는 편이 안전합니다.

초심자에게 더 중요한 교훈은 "이 중 하나를 복사하라"가 아닙니다. 각 접근이 무엇을 가르쳐 주고, 어디서 실패하는지를 보는 것입니다.

| 접근 | 먼저 가르쳐 주는 것 | 먼저 실패하는 것 |
|---|---|---|
| Action-prediction RL | 어떤 action이 의미 있는 변화를 만드는가 | 그 변화가 왜 중요한가 |
| State graph search | 같은 state를 반복하지 않는 법 | 아직 안 가본 state를 상상하는 법 |
| World-model planning | 행동 전에 예측하고 계획하는 법 | 빠르게 신뢰할 만한 model을 만드는 법 |
| LLM harness | language reasoning과 memory scaffold | Kaggle eligibility와 unseen generalization |
| Public-source BFS/FORGE | search, action scan, state hashing의 힘 | hidden game에서 source/introspection 의존이 깨지는 문제 |

제 생각에는 여러 요소를 조합해야 할 가능성이 큽니다. loop를 피하기 위한 graph memory, action 낭비를 줄이는 action prior, transition을 예측하는 world model, goal이 보이기 시작했을 때 짧게라도 planning하는 search가 함께 필요해 보입니다.

### Kaggle public example에서 보이는 패턴

제가 로컬에서 확인한 `/Users/pilkwang/Documents/VSDocs/Kaggle/ARC_AGI_3/Public_Examples` 폴더에는 public notebook 네 개가 있었습니다.

| Notebook | 전략 | 배울 점 | 조심할 점 |
|---|---|---|---|
| `0-35-forge-v16-trigger-aware-bfs.ipynb` | trigger-aware BFS, hidden field probing, A\* fallback | state hash에 무엇을 넣고 뺄지가 search 성능을 크게 바꿉니다. | 내부 field probing은 hidden evaluation에서 그대로 통한다고 가정하면 위험합니다. |
| `forge-arc-agi-3-agent.ipynb` | FORGE v18, public source 기반 BFS, action dedup, CLTI demo injection | public game에서 solution replay와 cross-level transfer를 어떻게 활용하는지 볼 수 있습니다. | public source exploit에 가까운 부분과 일반화 가능한 부분을 분리해야 합니다. |
| `arc-agi-3-hybrid-solver-bfs-cnn-heuristics.ipynb` | BFS + A\* + beam + CNN fallback + novelty-guided exploration | search와 learned prior를 섞는 실전적 형태입니다. | 복잡도가 높아서, 어떤 구성요소가 실제로 기여하는지 분해 실험이 필요합니다. |
| `ash-s-arc-agi-3-agent.ipynb` | BFS solver 개선, demo analysis, transient field 제거, CNN fallback | frame extraction, reset 횟수, epsilon reset 같은 작은 실행 버그가 점수를 크게 바꿀 수 있습니다. | 좋은 engineering과 overfit 사이의 경계가 얇습니다. |

이 notebook들이 알려주는 것은 분명합니다.

- public game에서는 **search가 매우 강력**합니다.
- search가 강해지려면 action space를 줄여야 합니다. 특히 click 좌표를 줄이는 것이 중요합니다.
- state hash가 틀리면 search가 쉽게 무너집니다. 너무 적게 넣으면 다른 state를 같은 state로 보고, 너무 많이 넣으면 transient counter 때문에 같은 state를 계속 다른 state로 봅니다.
- BFS만으로 부족하면 A\*, beam search, heuristic, CNN fallback을 붙입니다.
- level 0에서 찾은 demonstration을 level 1 이후의 prior로 쓰려는 시도가 있습니다.

하지만 이걸 그대로 베끼면 안 됩니다. public notebook의 강한 점수는 종종 public game source, 내부 helper, 구현 세부사항에 접근할 수 있다는 사실에서 나옵니다. ARC-AGI-3의 진짜 목표는 unseen game generalization입니다. 따라서 제 관점에서는 public examples를 이렇게 읽는 것이 맞습니다.

```text
복사할 것:
  logging discipline
  action scan 아이디어
  no-op/action dedup
  state hash 설계
  BFS/A*/beam의 역할 분담
  search 실패 시 fallback 설계

조심할 것:
  public source 직접 사용
  hidden field introspection
  game마다 직접 박아 넣은 trigger
  leaderboard score만 보고 구성요소를 믿는 것
```

즉, public examples는 "정답"이라기보다 **어떤 문제가 실제로 agent를 실패하게 만드는지 보여주는 해부도**에 가깝습니다.

### 초심자를 위한 세 번의 구현 패스

처음 시작하는 사람에게 바로 "world model을 만들자"고 말하면 너무 큽니다. 저는 세 번의 구체적인 구현 패스로 나누겠습니다.

#### Pass 1: random을 덜 낭비하게 만들기

먼저 가능한 action만 고르는 random agent에서 시작합니다. 그다음 가장 분명한 낭비를 제거합니다.

- available action 안에서만 sampling합니다.
- 같은 state에서 이미 no-op으로 확인된 action은 다시 고르지 않습니다.
- 바로 직전 state로 돌아가는 2-step loop를 피합니다.
- coordinate click은 전체 pixel에서 균등 무작위로 찍지 말고, object처럼 보이는 cell 근처에서 고릅니다.
- reset은 state가 막혔거나 game over일 때만 씁니다.

목표는 똑똑한 agent가 아닙니다. random exploration이 쓸 만한 데이터를 남기게 만드는 것입니다.

click action의 후보 좌표도 처음에는 아주 단순하게 만들 수 있습니다.

```python
def candidate_clicks(grid):
    background = most_common_color(grid)
    cells = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != background:
                cells.append((x, y))
    return cells
```

이 방법은 자주 틀릴 겁니다. 그래도 64×64 전체 board에서 아무 pixel이나 찍는 것보다는 대개 낫습니다. background를 계속 클릭하는 것만 줄여도 action budget이 꽤 절약됩니다.

작은 예시를 보겠습니다.

```text
0 0 0 0 0
0 1 0 2 0
0 0 0 0 0
0 3 0 0 0
0 0 0 0 0
```

여기서 `0`이 background라고 추정되면 click 후보는 `(1,1)`, `(3,1)`, `(1,3)` 세 곳입니다. 전체 25개 cell을 균등하게 클릭하면 대부분 background를 누르게 됩니다. 하지만 non-background cell만 후보로 두면 적어도 object를 건드릴 가능성이 커집니다.

물론 이 heuristic도 틀릴 수 있습니다. 어떤 game에서는 background처럼 보이는 cell이 버튼일 수 있고, object 주변 빈칸을 클릭해야 할 수도 있습니다. 그래서 click 후보는 하나가 아니라 여러 family로 나누는 것이 좋습니다.

| 후보 family | 예시 | 왜 필요한가 |
|---|---|---|
| object cell | 색이 background와 다른 cell | 가장 단순한 object interaction 후보입니다. |
| object border | object 주변 1-cell neighborhood | object 옆 빈칸을 눌러야 하는 game에 필요합니다. |
| changed cell | 직전 action 후 바뀐 cell | 이미 반응을 보인 위치를 다시 조사합니다. |
| centroid | 같은 색 object의 중심점 | 큰 object를 한 번에 대표하는 좌표입니다. |
| sparse grid | 일정 간격으로 찍은 보조 좌표 | object detector가 틀렸을 때 최소한의 탐색을 보장합니다. |

초반 agent는 이 후보들을 모두 만들고, no-op 결과를 보면서 점점 줄여가면 됩니다.

#### Pass 2: state graph 만들기

no-op reduction이 있으면, 이제 관측한 transition을 전부 저장합니다.

```text
state_hash --ACTION1--> next_state_hash
state_hash --ACTION5--> next_state_hash
state_hash --ACTION6(x=12,y=7)--> next_state_hash
```

그러면 이런 질문을 할 수 있습니다.

- 이 state에서 아무 변화도 만들지 않는 action은 무엇인가?
- 어떤 action이 아직 보지 못한 state로 이어지는가?
- 어떤 action이 score나 game state를 바꾸는가?
- 어떤 state를 계속 반복 방문하고 있는가?
- agent가 graph를 넓히고 있는가, 아니면 같은 부분 안에서 돌고 있는가?

state graph는 첫 번째 world map입니다. 아직 learned model은 아니지만, raw experience를 구조화된 데이터로 바꿔 줍니다. 이게 있어야 그 다음 learning도 의미가 생깁니다.

state hash는 생각보다 중요합니다. 예를 들어 grid만 hash하면 다음 두 state가 같아 보일 수 있습니다.

```text
grid는 같음
하지만 내부 counter는 2/3

grid는 같음
하지만 내부 counter는 3/3, 다음 action에서 문이 열림
```

반대로 action count처럼 매번 변하는 transient field까지 hash에 넣으면, 사실상 같은 state가 매번 다른 state로 보입니다. 그러면 visited-state cache가 무력화됩니다.

그래서 public examples의 FORGE 계열이 hidden field probing, transient field detection, trigger-aware hashing에 신경을 쓰는 것입니다. 이건 꼼수가 아니라, interactive search에서는 **state representation이 곧 search space**라는 사실을 보여줍니다.

#### Pass 3: action priority 붙이기

graph가 생겼다면 이제 action을 균등 무작위로 고르지 않습니다. 점수를 매깁니다.

| Signal | 의미 |
|---|---|
| `+ novelty` | 아직 보지 못한 state로 갈 것 같은 action을 선호합니다. |
| `+ change_prior` | 비슷한 state에서 frame을 바꾼 action을 선호합니다. |
| `+ object_focus` | background가 아닌 object와 관련된 action을 선호합니다. |
| `+ score_delta` | score나 level state를 개선한 transition을 선호합니다. |
| `- no_op` | 이미 아무 변화가 없던 action의 점수를 낮춥니다. |
| `- loop_risk` | 최근 state로 돌아갈 가능성이 큰 action의 점수를 낮춥니다. |

처음에는 hand-written priority function이어도 됩니다.

```python
score = (
    2.0 * novelty
  + 1.0 * change_prior
  + 0.5 * object_focus
  + 3.0 * score_delta
  - 2.0 * no_op
  - 1.5 * loop_risk
)
```

이것이 AGI는 아닙니다. 하지만 최소한 일관된 agent입니다. 행동하고, 관측하고, 기억하고, 배운 것에 따라 다음 행동을 바꿉니다. 여기까지 와야 neural action prior나 learned transition model을 붙일 자리가 생깁니다.

### 초반에 꼭 진단해야 할 실패 모드

대부분의 agent는 지루한 방식으로 실패합니다. 다행히 지루한 실패는 측정하기 쉽습니다.

| 증상 | 가능한 원인 | 첫 번째 수정 |
|---|---|---|
| action은 많은데 frame 변화가 거의 없음 | unavailable 또는 irrelevant action을 sampling함 | available-action metadata와 no-op cache를 씁니다. |
| frame은 바뀌지만 score/level progress가 없음 | action prior는 motion을 찾지만 goal을 못 찾음 | goal hypothesis와 score-change tracking을 추가합니다. |
| 같은 frame 근처를 반복함 | loop detector가 없음 | recent-state window를 두고 되돌아가는 action의 점수를 낮춥니다. |
| public game 하나에서는 좋은데 다른 game에서 무너짐 | public-game overfit | public game을 split하고 holdout에서 평가합니다. |
| click action이 많지만 의미가 없음 | 좌표를 균등 무작위로 고름 | non-background cell, changed cell, object center 근처를 클릭합니다. |
| level은 풀지만 score가 낮음 | exploratory action을 너무 많이 씀 | actions per completed level을 보고 brute-force rollout을 줄입니다. |
| runtime이 터짐 | search width가 너무 큼 | candidate action 수, planning depth, per-turn model update를 제한합니다. |

거대한 architecture diagram보다 이런 표가 초반에는 더 유용합니다. 무엇을 log해야 하고, 다음에 무엇을 고쳐야 하는지 알려주기 때문입니다.

## 8. Meta-game: scoring의 함정과 남은 질문

이 부분은 forum을 읽어야만 알게 되는 실전 함정입니다.

- **`env.make()`를 여러 번 호출하면 위험할 수 있습니다.** 특히 parallelization 과정에서 process마다 scorecard를 열면 점수가 0이 되는 식의 문제가 생길 수 있습니다. main thread에서 한 번만 여는 방식을 지키는 것이 중요합니다.
- **`MAX_ACTIONS = ∞` 같은 설정은 위험합니다.** recording이 커져 Kaggle storage quota를 터뜨릴 수 있습니다. logic error가 아니라 storage failure로 실패할 수 있습니다.
- **human baseline이 agent에게 보이는가?** `environment_info.baseline_actions` 같은 정보가 test game에서 의도적으로 노출되는지, 실제로 활용할 수 있는지는 2026년 6월 기준으로 조심스럽게 다뤄야 합니다. 확정된 사실처럼 쓰면 안 됩니다.
- **open-source 공개 딜레마가 있습니다.** milestone prize를 받으려면 공개해야 하지만, 공개 직후 복사 제출이 쏟아지면 원 작성자의 ranking이 흔들릴 수 있습니다. 이 timing rule은 주최 측 clarification을 확인해 따라야 합니다.
- **variance가 있습니다.** 같은 코드도 run-to-run 점수가 달라질 수 있습니다. 특히 낮은 점수대에서는 작은 차이가 순위를 크게 흔듭니다.

작지만 중요한 실전 정보도 있습니다.

| 항목 | 의미 |
|---|---|
| Local offline | 빠릅니다. development와 분해 실험에 씁니다. online scorecard/replay는 없지만 iteration이 빠릅니다. |
| Online API | scorecard와 replay를 남길 수 있습니다. 다만 rate limit과 API key 관리가 있습니다. |
| Kaggle submission | 실제 leaderboard scoring입니다. 제출 횟수와 runtime이 귀하므로 마지막 확인용으로 써야 합니다. |
| Scorecard | 여러 game run의 결과를 묶어 보는 기록입니다. 실패 원인을 추적하려면 scorecard보다 더 자세한 local log가 필요합니다. |
| Replay | action sequence와 frame 변화를 다시 볼 수 있는 자료입니다. 사람이 실패를 이해하는 데 매우 중요합니다. |
| Public notebook score | unit confusion, stale metric, exploit history 때문에 그대로 믿으면 안 됩니다. |
| Private leaderboard | 최종 standing의 핵심입니다. public game overfit은 여기서 드러납니다. |

초심자가 가장 많이 하는 실수는 Kaggle submission을 local debugger처럼 쓰는 것입니다. 이 대회에서는 반대로 해야 합니다. local/offline에서 수백 번 망하고, 그 망한 이유를 log로 설명할 수 있을 때만 Kaggle submission을 써야 합니다.

## 9. 나라면 어떻게 접근할 것인가

제 접근은 이렇습니다.

1. **Kaggle submission loop로만 실험하지 않습니다.** 먼저 local evaluation harness를 만듭니다.
2. **25 public games를 train/holdout처럼 나눕니다.** 예를 들어 18개는 개발용, 7개는 holdout으로 둡니다. public set 안에서도 overfit을 감지해야 합니다.
3. **leaderboard 숫자가 아니라 공개된 방법론을 기준으로 삼습니다.** StochasticGoose나 graph-explore baseline을 재현해 보고, 같은 local protocol에서 비교해야 합니다.
4. **작은 기초 산출물부터 만듭니다.** replay log, state cache, no-op detector, state graph가 neural model보다 먼저입니다.
5. **world model은 나중에 붙입니다.** 관측/로그/metric이 없으면 model이 좋아졌는지 알 수 없습니다.
6. **milestone open-source release를 관찰합니다.** 상위 팀이 공개하는 code는 전략의 현실적인 방향을 알려줄 가능성이 큽니다.

starter kit에서 실제로 수정해야 하는 부분은 많지 않습니다. 보통 중요한 코드는 `agent/my_agent.py` 안의 `MyAgent` class입니다.

```python
class MyAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        """agent가 playthrough를 멈추고 싶으면 True를 반환한다."""
        return False

    def choose_action(self, frames, latest_frame) -> GameAction:
        """frame history를 보고 다음 legal action 하나를 반환한다."""
        state = parse_grid(latest_frame)
        model = update_world_model(frames)
        goal = infer_goal(state, model)
        return plan_next_action(state, model, goal)
```

이 stub 하나가 문제 전체를 압축해서 보여 줍니다.

```text
perception → memory/model update → goal inference → planning → action
```

처음 code를 열면 막막할 수 있으니, 저는 `choose_action()`을 네 개의 작은 함수로 나누는 것을 권합니다.

```python
def choose_action(self, frames, latest_frame):
    state = self.parse_state(latest_frame)
    self.update_memory(frames, state)
    candidates = self.make_candidates(state)
    return self.select_action(state, candidates)
```

각 함수의 첫 버전은 아주 단순해도 됩니다.

| 함수 | 첫 버전에서 할 일 | 나중에 붙일 것 |
|---|---|---|
| `parse_state` | grid, available actions, game state를 꺼냅니다. | object segmentation, changed-cell extraction |
| `update_memory` | state hash와 직전 action outcome을 저장합니다. | state graph, cross-level memory |
| `make_candidates` | available actions와 click 후보를 만듭니다. | action scan, object-focused click, learned action prior |
| `select_action` | no-op과 loop를 피하고 점수 높은 action을 고릅니다. | short-horizon planning, world-model rollout |

처음 공부할 것도 이 순서로 맞추면 좋습니다.

1. **Python data handling**: list, dict, dataclass, JSONL log.
2. **NumPy grid operation**: equality, changed-cell mask, connected component.
3. **Graph/search basics**: BFS, visited set, shortest path, depth limit.
4. **Kaggle notebook runtime**: internet off, file path, package import, submission output.
5. **Small ML model**: CNN이나 value model은 로그와 graph가 생긴 뒤에 붙입니다.

처음부터 "world model"이라는 큰 단어를 붙잡지 말고, 다음 질문에 답하는 함수부터 만들면 됩니다.

```text
이 action은 frame을 바꾸는가?
바꾼다면 어떤 cell이 바뀌는가?
같은 state에서 이미 해본 action인가?
이 action은 최근 loop로 돌아가는가?
이 action은 score나 level state를 바꾼 적이 있는가?
```

이 질문들이 code로 답해지기 시작하면, 그 다음이 model입니다.

### 실제 구축 순서

처음 시작한다면 저는 neural network부터 만들지 않을 겁니다. 아래 순서로 쌓겠습니다.

| 단계 | 만들 것 | 이유 |
|---|---|---|
| 0 | random starter를 local에서 실행 | environment, submission path, logging이 되는지 확인합니다. |
| 1 | 모든 frame/action/result를 replay log로 저장 | 보지 못하는 것은 개선할 수 없습니다. |
| 2 | no-op action detector | 아무 변화 없는 action 반복을 줄입니다. |
| 3 | visited-state cache | 같은 state를 빙빙 도는 것을 막습니다. |
| 4 | simple state graph | agent가 발견한 세계 지도를 만듭니다. |
| 5 | action prior | 비슷한 state에서 frame을 바꾼 action을 우선합니다. |
| 6 | local goal guess | 도달, 정렬, 제거, 색 matching 같은 후보 goal을 추적합니다. |
| 7 | short-horizon planning | 몇 step 앞을 state graph나 model 위에서 탐색합니다. |
| 8 | learned transition prediction | 아직 시도하지 않은 action의 next frame을 예측합니다. |
| 9 | cross-level memory | 같은 game 안의 앞 level mechanic을 뒤 level에 재사용합니다. |

이 순서가 중요합니다. replay logging 없는 learned world model은 black box입니다. visited-state cache 없는 planner는 loop를 다시 발견하느라 시간을 씁니다. level-change detector 없는 cross-level memory는 이전 board에서만 맞았던 가정을 다음 board에 그대로 가져갈 수 있습니다.

처음 볼 local metric도 단순해야 합니다.

| Metric | 알려주는 것 |
|---|---|
| unique state 수 | exploration이 넓어지는가, loop인가 |
| no-op action rate | action budget을 얼마나 낭비하는가 |
| first useful action step | 의미 있는 변화를 발견하기까지 얼마나 걸리는가 |
| completed level 수 | 가장 거친 성공 신호 |
| actions per completed level | 클리어가 점수로 의미 있을 만큼 효율적인가 |
| cross-level reuse | level 1에서 배운 것이 level 2에 도움이 되는가 |

이 metric들이 보이기 전에는 ML을 얹어도 개선인지 착시인지 알기 어렵습니다.

### 첫 일주일 계획

처음부터 제대로 된 baseline까지 가는 것을 목표로 한다면, 저는 첫 일주일을 이렇게 잡겠습니다.

| Day | 목표 | 산출물 |
|---|---|---|
| 1 | starter를 local에서 실행하고, 변경 없는 baseline을 한 번 제출합니다. | 환경과 submission path가 동작한다는 확인. |
| 2 | frame, action, state hash, outcome을 replay log로 저장합니다. | `runs/YYYYMMDD/*.jsonl` 같은 log 파일. |
| 3 | no-op detection과 repeated-state detection을 구현합니다. | random보다 덜 낭비하는 agent. |
| 4 | 최소 state graph를 만들고, game 하나에 대해 시각화합니다. | state와 transition이 보이는 graph dump. |
| 5 | object-focused click candidate와 available-action filtering을 추가합니다. | click-heavy game에서 낮아진 no-op rate. |
| 6 | public game을 development/holdout으로 나누고 둘 다 실행합니다. | 작은 comparison report. |
| 7 | simple action-priority function을 붙이고 baseline과 비교합니다. | 첫 candidate와 실패 기록. |

첫 주의 목표는 leaderboard score가 아닙니다. 관측과 기록 체계를 만드는 것입니다. 주말에는 실패 run 하나를 열고 이렇게 말할 수 있어야 합니다.

```text
이 agent가 막힌 이유:
- ACTION5가 70%의 state에서 아무 변화도 만들지 않았다.
- coordinate click 대부분이 background였다.
- 같은 11개 state를 140 action 동안 반복했다.
- level 2에 한 번도 도달하지 못했다.
```

이 정도면 진전입니다. 실패에 이름이 붙으면, 다음 수정도 정해집니다.

### 초반에 피하고 싶은 것들

저라면 초반에는 다음을 피하겠습니다.

- **leaderboard 제출 결과로 직접 튜닝하지 않습니다.** 피드백이 너무 느리고 드뭅니다.
- **public game mechanic을 코드에 박아 넣지 않습니다.** local에서는 좋아 보여도 hidden game에서는 무너집니다.
- **깨끗한 log 없이 비싼 neural model부터 붙이지 않습니다.** 그러면 model이 왜 실패하는지 보이지 않습니다.
- **completion count만 최적화하지 않습니다.** RHAE에서는 느린 win이 거의 쓸모없을 수 있습니다.
- **runtime 제한 없이 search 폭만 넓히지 않습니다.** timeout이 나는 훌륭한 plan은 0점입니다.
- **실패한 trajectory를 버리지 않습니다.** 실패 run에는 어떤 action이 쓸모없는지 알려주는 데이터가 들어 있습니다.

원칙은 간단합니다. 모든 실험은 결과물을 남겨야 합니다. replay, metric table, changed-state summary, failure label 중 하나는 있어야 합니다. leaderboard 숫자 하나만 남는 실험은 배운 것이 너무 적습니다.

### "world model + intrinsic exploration"이 실제로 뜻하는 것

말은 거창하지만 첫 버전은 작게 시작할 수 있습니다.

```python
memory = StateGraph()
model = TransitionModel()

while not done:
    state = observe()
    candidates = legal_actions(state)

    scored = []
    for action in candidates:
        predicted = model.predict(state, action)
        novelty = memory.novelty(predicted)
        uncertainty = model.uncertainty(state, action)
        goal_value = goal_heuristic(predicted)
        scored.append((novelty + uncertainty + goal_value, action))

    action = max(scored)[1]
    next_state = step(action)
    memory.add(state, action, next_state)
    model.update(state, action, next_state)
```

이것이 곧 final solution이라는 뜻은 아닙니다. 다만 제가 신뢰하는 방향은 대략 이쪽입니다. 관측을 기록하고, action이 무엇을 하는지 배우고, 목표가 불분명할 때는 novelty와 uncertainty를 따라가며, 환경이 조금 읽히기 시작하면 goal-directed planning으로 전환합니다.

## 10. 이 대회에서 실제로 얻는 것

- **상금** — milestone pool, top score pool, 그리고 현재로서는 매우 어려운 grand prize가 있습니다.
- **Paper Prize track** — 점수뿐 아니라 방법론을 설명하는 paper도 별도 보상을 받을 수 있습니다.
- **Kaggle medal/ranking** — portfolio 관점에서 의미가 있을 수 있습니다.
- **진짜 open problem에 기여** — 전체 성능이 거의 0 근처에 있으므로, 새 아이디어가 보일 공간이 있습니다.
- **RL, world models, program induction 사이의 포지셔닝** — 학습 주제로도 매우 좋습니다.

## 11. 제 생각

저는 리더보드가 이 문제의 본질을 꽤 흐린다고 봅니다. 특히 unit confusion 때문에 더 그렇습니다. 실제 상황은 오히려 단순합니다.

> launch-era purpose-built preview winner도 full benchmark에서는 0.25% 근처로 떨어졌고, 2026년 6월 말 Kaggle 참가자 전체도 여전히 2% 아래입니다.

이 말은 우울하기보다 흥미롭습니다. 문제는 아직 열려 있고, 진짜 다른 접근이 의미를 가질 여지가 있습니다.

제가 더 가능성이 있다고 보는 방향은 **learned test-time world model + intrinsic-motivation exploration + lightweight planning + explicit goal inference**입니다. reactive action-prediction과 pure frontier search는 좋은 baseline이지만, benchmark가 명시한 네 능력 — exploration, modeling, goal-setting, planning — 을 모두 직접 겨냥하지는 못합니다.

이 대회의 질문은 "어떤 trick이 지금 leaderboard를 조금 올리는가"가 아닙니다. 더 근본적으로는:

> 처음 보는 작은 세계 안에서, 시스템이 어떻게 실험하고, 기억하고, 모델을 만들고, 목표를 세우고, 계획하는가?

그 질문이 좋아서 이 문제를 붙잡고 있습니다. 구축 과정도 계속 정리해 볼 생각입니다.

---

### Resources

- [ARC Prize 2026 - ARC-AGI-3 on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [What is ARC-AGI?](https://arcprize.org/arc-agi)
- [ARC-AGI-1](https://arcprize.org/arc-agi/1)
- [ARC-AGI-2 dataset repository](https://github.com/arcprize/ARC-AGI-2)
- [OpenAI o3 ARC-AGI-1 report](https://arcprize.org/blog/oai-o3-pub-breakthrough)
- [ARC Prize 2025 technical report](https://arxiv.org/abs/2601.10904)
- [ARC-AGI-3 information page](https://arcprize.org/arc-agi/3)
- [Official ARC-AGI-3 competition page](https://arcprize.org/competitions/2026/arc-agi-3)
- [ARC Prize 2026 overview and key dates](https://arcprize.org/competitions/2026)
- [ARC-AGI-3 technical report](https://arxiv.org/abs/2603.24621)
- [ARC-AGI-3 docs](https://docs.arcprize.org/arc-prize-2026), especially [Games](https://docs.arcprize.org/games), [Actions](https://docs.arcprize.org/actions), [Scoring methodology](https://docs.arcprize.org/methodology), [Local vs Online](https://docs.arcprize.org/local-vs-online), and [Scorecards](https://docs.arcprize.org/scorecards)
- [ARC-AGI-3 Kaggle Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter)
- [ARC-AGI-3 Preview: 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [StochasticGoose preview solution](https://github.com/DriesSmit/ARC3-solution)
- [ARC-AGI Community Leaderboard](https://github.com/arcprize/ARC-AGI-Community-Leaderboard)
- 초심자용 보조 자료: [DataCamp ARC-AGI-3 overview](https://www.datacamp.com/blog/arc-agi-3), [Mark Barney ARC-AGI-3 explainer](https://arc.markbarney.net/arc3), [TokenCost evaluation-cost writeup](https://tokencost.app/blog/arc-agi-3-benchmark-cost)
- Community tooling과 game guide는 빠르게 바뀌므로, historical leaderboard tracker, simplified submission notebook, offline recording viewer, game mechanic wiki는 Kaggle discussion forum에서 최신 링크를 확인하는 편이 안전합니다.
