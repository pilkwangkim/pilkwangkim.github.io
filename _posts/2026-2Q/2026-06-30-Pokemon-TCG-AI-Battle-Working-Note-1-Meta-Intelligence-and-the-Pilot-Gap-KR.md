---
title: "Pokémon TCG AI Battle Working Note (1편): 메타 분석과 파일럿 격차"
date: 2026-06-30 21:30:00 +0900
categories: [AI, Kaggle]
tags: [kaggle, pokemon-tcg, game-ai, reinforcement-learning, behavior-cloning, meta-analysis, benchmark, working-note, korean]
math: true
pin: false
---

# Pokémon TCG AI Battle Working Note (1편): 메타 분석과 파일럿 격차

> **Working note 기준일: 2026-06-30.**  
> 이 글은 Pokémon TCG AI Battle 대회를 다루는 주간 시리즈의 첫 번째 기록이다. 목표는 단순히 제출 점수를 올리는 데서 끝나지 않는다. 공개 대전 로그를 모아 매일 바뀌는 메타를 읽고, 덱의 강함과 파일럿의 강함을 분리하고, 실제 필드에 맞춘 벤치마크를 만들고, 나아가 행동 복제(behavior cloning)와 강화학습으로 규칙 기반 에이전트의 약점을 보완하는 반복 가능한 연구 루프를 만드는 것이 목표다.

대회 링크:  
[The Pokémon Company - PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)

---

## 1. 이 대회가 실제로 요구하는 것

처음 보면 이 대회는 평범한 Kaggle 에이전트 대회처럼 보인다. 에이전트를 작성해서 제출하고, skill rating이 오르내리는 것을 지켜보면 된다. 하지만 실제로 풀어야 하는 문제는 그보다 훨씬 복잡하다.

에이전트는 시뮬레이터에서 Pokémon Trading Card Game을 플레이한다. 매 턴 observation을 받는데, 여기에는 현재 보드 상태, 게임 로그, 상대 손패처럼 볼 수 없는 정보의 경계, 그리고 지금 고를 수 있는 합법적인 선택지 목록이 들어 있다. 에이전트는 그 선택지 중 무엇을 고를지 번호로 반환한다. 시뮬레이터가 이미 합법적인 행동만 보여주기 때문에, 이 문제는 "합법 수를 처음부터 만들어내라"가 아니다. 핵심은 다음에 가깝다.

$$
a_t \in \text{LegalOptions}(o_t)
$$

조금 더 구체적으로 쓰면:

$$
\pi(i \mid o_t, \text{option}_1, \ldots, \text{option}_k)
$$

각 선택지는 규칙상 가능한 행동이다. 하지만 가능하다고 해서 모두 좋은 수는 아니다.

따라서 이 대회는 세 층의 문제가 섞여 있다.

| 계층 | 질문 | 중요한 이유 |
|---|---|---|
| 덱 구성 | 어떤 60장 덱을 가져갈 것인가? | 약한 덱은 좋은 플레이만으로 살리기 어렵다. |
| 파일럿 정책 | 그 덱을 매 순간의 선택지 목록에서 어떻게 굴릴 것인가? | 강한 덱도 파일럿이 나쁘면 낮은 점수를 받는다. |
| 메타 적응 | 오늘 어떤 상대가 많고, 어떤 전략이 뜨고 있는가? | 어제의 필드를 이기던 덱이 오늘은 틀린 답일 수 있다. |

첫 주의 가장 큰 교훈은 이 세 층을 하나의 점수로 뭉개면 안 된다는 것이었다. 공개 로그에서 강한 deck hash를 찾을 수는 있다. 하지만 내가 그 덱을 제대로 굴리지 못한다면 아직 제출물이라고 부를 수 없다. 반대로 로컬 벤치마크에서 강한 에이전트라도, 실제 라이브 필드에서 중요한 상대가 벤치마크 패널에 빠져 있으면 쉽게 무너질 수 있다.

그래서 이 프로젝트에서는 강함을 다음처럼 본다.

$$
\text{Strength} = f(\text{deck}, \text{pilot}, \text{field}, \text{seat}, \text{variance})
$$

마지막 항도 중요하다. 공개 인터페이스만으로는 engine의 randomness를 완전히 seed-control하기 어렵다. common-random-number 방식의 paired evaluation이 가능하면 이상적이겠지만, 현재 경로에서는 어렵다. 그래서 현실적인 대체로 seat balancing, 공통 상대 패널, Wilson interval, holdout run, local-to-live calibration을 사용한다.

---

## 2. 왜 working note 파이프라인이 필요한가

초기 접근은 평범했다. 공개 노트북을 읽고, 그럴듯한 규칙 기반 덱/파일럿을 만들고, 제출하고, 점수를 확인한다. 이 방식도 어느 정도는 유용했다. 하지만 Kaggle 게임 대회에서 흔히 만나는 벽에 금방 닿았다.

1. 로컬 상대 파일럿이 약하면 후보가 실제보다 좋아 보인다.
2. noisy한 variant를 많이 만들고 그중 최고만 고르면 winner's curse가 생긴다.
3. 어제의 주류 덱에는 강하지만 오늘 떠오르는 덱에는 약할 수 있다.
4. 로그에서 덱은 찾았지만 파일럿이 준비되지 않은 경우가 많다.
5. 높은 live score의 원인이 덱 리스트가 아니라 파일럿일 수 있다.

그래서 워크플로우는 "제출 파일을 조금씩 튜닝한다"에서 "메타를 읽고 검증하는 시스템을 만든다"로 바뀌었다.

현재 일일 루프는 다음과 같다.

```text
Kaggle 리플레이 로그
-> compact extraction table
-> archetype / exact-deck 집계
-> matchup matrix와 strategy profile
-> 필드 가중 benchmark panel
-> pilot library와 fidelity check
-> 후보 deck / pilot search
-> holdout과 local-to-live gap 진단
-> known reference를 이길 때만 제출
```

마지막 줄이 특히 중요하다. 후보는 그럴듯해 보인다는 이유만으로 제출되어서는 안 된다. 이미 알고 있는 기준(reference) 세트보다 확실한 우위가 있거나, 그렇지 않다면 "아직 검증된 edge가 없다"고 말해야 한다.

---

## 3. 데이터 계층: 6월 30일까지 처리한 것

6월 30일 기준 local store는 5일치 리플레이 로그를 처리했다. 2026-06-30 batch만 해도 이미 꽤 큰 규모였다.

| 항목 | 2026-06-30 값 |
|---|---:|
| Raw replay JSON files | 5,734 episodes |
| Raw bytes scanned | 21.47 GB |
| Player deck rows | 11,468 |
| Winner decklists captured | 133 |
| Strategy profiles | 17 archetypes |
| Exact decklists exported for reuse | 55 ready decks |
| Candidate decision rows extracted | 327,937 |
| Subsampled decision rows kept | 82,182 |

이 정도 규모가 되면 저장 공간 문제가 바로 현실이 된다. daily log가 하루 20-22 GB라면 raw JSON을 전부 영구 보관하는 방식은 300 GB 정도의 프로젝트 용량 안에서 오래 버티기 어렵다. 현재 정책은 다음과 같다.

- 아직 필요한 raw daily log만 보관한다.
- raw mirror는 `zstd`로 압축한다.
- 영구 분석 소스는 compact Parquet / CSV table로 남긴다.
- 모든 파생 산출물은 날짜 기준으로 나눈다.
- benchmark run과 field selection은 반드시 어느 날짜의 로그에서 왔는지 연결해 둔다.

이것은 단지 디스크 정리 문제가 아니다. 재현성을 위한 장치이기도 하다. 나중의 주간 리포트는 "이 덱은 20260630 필드 가중 패널을 이겼기 때문에 선택되었고, 그 weight는 이 리플레이 로그에서 왔다"고 말할 수 있어야 한다.

---

## 4. 첫 번째 메타 스토리: Archaludon 하락, Alakazam 상승, Marnie 등장

6월 29일에서 6월 30일로 넘어오며 가장 눈에 띈 변화는 Archaludon이 많은 필드에서 Alakazam이 많은 필드로 중심이 이동했다는 점이다.

| Archetype | 2026-06-29 점유율 | 2026-06-30 점유율 | 변화량 | 2026-06-30 score rate |
|---|---:|---:|---:|---:|
| Alakazam / Dunsparce | 0.2124 | 0.3306 | +0.1182 | 0.5186 |
| Archaludon | 0.4067 | 0.2892 | -0.1176 | 0.4801 |
| Starmie | 0.0710 | 0.0787 | +0.0076 | 0.4501 |
| Dragapult | 0.0659 | 0.0687 | +0.0028 | 0.4365 |
| Marnie's Impidimp / Munkidori | 0.0126 | 0.0412 | +0.0286 | 0.6469 |
| Cynthia's Power Weight / Cynthia's Gabite | 0.0048 | 0.0367 | +0.0319 | 0.4988 |
| Crustle | 0.0045 | 0.0169 | +0.0124 | 0.5412 |
| Okidogi / Solrock | 0.0079 | 0.0149 | +0.0070 | 0.6140 |

핵심은 단순히 "Alakazam이 최고다"가 아니다. 상황은 그보다 더 미묘했다.

- Alakazam은 가장 큰 점유율을 차지하는 archetype이 되었지만, 점수율은 근소하게 50%를 넘는 정도였다.
- Archaludon은 점유율이 줄고 field score rate도 50% 아래였지만, 튜닝된 Archaludon 에이전트는 여전히 로컬 벤치마크에서 중요했다.
- Marnie's Impidimp / Munkidori는 아직 share가 작았지만, score rate와 matchup matrix 기준으로 가장 중요한 상승 위협이었다.
- Okidogi / Solrock과 Crustle도 아직 작지만 더 이상 무시할 수 없는 수준이다.

그래서 단순 점유율만 보는 것은 위험하다. 인기 archetype은 많은 팀이 미숙한 파일럿으로 플레이하기 때문에 score rate가 낮게 나올 수 있다. 반대로 점유율이 작은 archetype이라도 두 개의 대형 덱을 모두 잘 잡는다면, 실제 압력은 그쪽에서 나올 수 있다.

---

## 5. Share보다 matchup이 변화를 더 잘 설명한다

가장 유용한 표는 필드 점유율 표가 아니라 매치업 표였다.

6월 30일 기준:

| Matchup | Games | First archetype win rate |
|---|---:|---:|
| Alakazam / Dunsparce vs Archaludon | 1,026 | 0.6637 |
| Marnie's Impidimp / Munkidori vs Archaludon | 148 | 0.6014 |
| Marnie's Impidimp / Munkidori vs Alakazam / Dunsparce | 134 | 0.6791 |
| Archaludon vs Starmie | 261 | 0.7471 |
| Archaludon vs Dragapult | 251 | 0.6534 |
| Crustle vs Alakazam / Dunsparce | 62 | 0.6774 |
| Okidogi / Solrock vs Alakazam / Dunsparce | 59 | 0.6780 |

이 표를 보면 필드가 왜 움직였는지 조금씩 보인다.

Alakazam의 상승은 합리적이다. 필드의 Archaludon을 매우 잘 잡기 때문이다. Archaludon이 필드의 30-40%를 차지한다면 Alakazam은 늘어날 이유가 충분하다.

하지만 그 다음 층이 더 흥미롭다. Marnie's Impidimp / Munkidori는 관측된 matrix에서 두 개의 큰 덱을 모두 이긴다. Archaludon 상대로 약 60%, Alakazam 상대로 약 68%다. 이것은 "대형 덱 둘을 동시에 겨냥하는 2차 메타 답안"에 가까운 모양이다.

필드 가중 매치업 추정도 같은 이야기를 한다.

| Archetype | 점유율 | 필드 가중 score | 커버된 필드 weight |
|---|---:|---:|---:|
| Marnie's Impidimp / Munkidori | 0.0412 | 0.6623 | 0.9554 |
| Cubchoo / Gravity Gemstone | 0.0094 | 0.6520 | 0.7545 |
| Meowth ex / Mega Kangaskhan ex | 0.0098 | 0.6198 | 0.8403 |
| Okidogi / Solrock | 0.0149 | 0.6073 | 0.8883 |
| Team Rocket Spidops | 0.0106 | 0.5969 | 0.7671 |
| Crustle | 0.0169 | 0.5660 | 0.8309 |
| Alakazam / Dunsparce | 0.3306 | 0.5207 | 0.9980 |
| Archaludon | 0.2892 | 0.4720 | 0.9987 |

이 표는 프로젝트 초기에 나온 가장 쓸모 있는 산출물 중 하나였다. 6월 30일에 가장 강해 보인 archetype은 가장 흔한 archetype이 아니었다. 새로 떠오르는 Marnie/Munkidori 계열이었다.

하지만 여기에는 중요한 단서가 있다. 덱을 발견하는 것과, 그 덱을 제출 가능한 에이전트로 만드는 것은 전혀 다른 일이다.

---

## 6. 덱의 강함과 파일럿의 강함은 별개다

0630 log에서는 여러 exact deck을 찾을 수 있었다.

| Archetype | Exact deck sample | Games | Score rate | Wilson low |
|---|---:|---:|---:|---:|
| Marnie's Impidimp / Munkidori | rank 2 hash | 137 | 0.6788 | 0.5967 |
| Marnie's Impidimp / Munkidori | rank 1 hash | 246 | 0.6626 | 0.6014 |
| Okidogi / Solrock | rank 1 hash | 141 | 0.6454 | 0.5635 |
| Alakazam / Dunsparce | rank 1 hash | 287 | 0.6272 | 0.5699 |
| Archaludon | rank 1 hash | 144 | 0.6042 | 0.5226 |
| Crustle | rank 1 hash | 193 | 0.5440 | 0.4736 |

이 정보는 유용하지만, 동시에 현재 시스템의 가장 큰 gap도 드러낸다.

Archaludon 같은 덱은 규칙 기반 파일럿으로도 어느 정도 잘 굴릴 수 있다. 게임 플랜이 비교적 명확하기 때문이다. 진화하고, 필요한 금속 자원을 붙이고, 빠르게 공격하고, tempo를 유지한다. Alakazam은 더 어렵다. draw/ability sequencing, 벤치 관리, late payoff timing이 더 많이 필요하다. Marnie/Munkidori는 그보다 더 어렵다. damage pressure가 나오기 전까지 setup이 많이 필요한 콤보 패턴을 제대로 세워야 하기 때문이다.

그래서 이 프로젝트는 다음 네 단계를 분리한다.

```text
deck discovery: 로그에서 강한 덱을 찾는다
pilot acquisition: 그 덱을 제대로 굴릴 파일럿을 만든다
benchmark validation: 실제 simulator benchmark로 검증한다
submission packaging: 제출 가능한 형태로 패키징한다
```

강한 exact deck은 FieldStore에 향후 후보로 들어갈 수 있다. 하지만 파일럿이 준비되지 않았다면 제출 후보로는 막아야 한다.

이 구분은 초기에 겪은 불편한 실패도 설명한다. 어떤 로컬 벤치마크는 deck 자체는 잘 골랐지만 live score가 약했다. deck이 반드시 틀렸다는 뜻은 아니다. 그 deck을 이기게 굴리는 파일럿이 약했을 수 있다.

---

## 7. 전략 프로파일: 덱들이 실제로 하는 일

strategy profiler는 카드 이름이 아니라 행동으로 deck을 설명하려고 한다.

| Archetype | Class | First attack turn | Attack cadence | Prize rate | Abilities / turn | Draw-search / turn | Main attack concentration |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alakazam / Dunsparce | combo | 4.049 | 0.4840 | 0.3056 | 1.0336 | 4.7817 | 0.9115 |
| Archaludon | aggro | 2.773 | 0.6601 | 0.3149 | 0.2473 | 5.3600 | 0.6724 |
| Starmie | aggro | 2.719 | 0.6564 | 0.4599 | 0.2275 | 4.1409 | 0.6621 |
| Dragapult | midrange | 2.968 | 0.6250 | 0.4023 | 0.5759 | 8.1486 | 0.7005 |
| Marnie's Impidimp / Munkidori | combo | 4.734 | 0.4962 | 0.3943 | 0.8169 | 7.6289 | 0.9929 |
| Okidogi / Solrock | midrange | 3.971 | 0.5448 | 0.3349 | 0.1089 | 5.7270 | 0.7438 |
| Crustle | combo / wall | 9.546 | 0.3059 | 0.0050 | 0.0908 | 2.1338 | 0.9433 |

행동 기준으로 읽으면 다음과 같다.

- **Archaludon**은 가장 깔끔한 tempo deck이다. first attack이 빠르고, attack cadence가 높고, ability complexity가 낮다. 규칙 기반 파일럿과 잘 맞는다.
- **Alakazam / Dunsparce**는 ability/draw combo deck이다. first attack은 느리고, Alakazam payoff attack에 집중되어 있으며, sequencing 선택지가 많다.
- **Marnie/Munkidori**는 setup이 많이 필요한 combo deck이다. first attack이 늦고, first attack 전 setup action이 많으며, payoff는 거의 Marnie's Grimmsnarl ex `Shadow Bullet`에 집중된다.
- **Crustle**은 prize race deck이 아니다. wall / collapse pattern에 가깝다. prize rate가 거의 0에 가까워서, prize tempo만 보는 benchmark는 이 덱을 오해하기 쉽다.
- **Okidogi/Solrock**은 midrange support deck이다. 단순한 attacker pile이 아니라 support engine이 중요하다.

이런 profile은 파일럿 search의 목표를 구체화해 준다. "Marnie를 더 잘 플레이하자"가 아니라 다음처럼 쓸 수 있다.

```text
파일럿이 Impidimp -> Morgrem / Grimmsnarl setup을 완성하는가?
너무 일찍 낮은 value attack을 하지 않고 support package를 보존하는가?
field pilot과 비슷한 payoff attack concentration에 도달하는가?
setup 중 가치가 낮은 선택지를 피하는가?
```

여기서 behavior cloning이 중요해진다.

---

## 8. Benchmarking: 실제로 무엇을 테스트했는가

로컬 벤치마크는 상대 패널과 파일럿이 충분히 좋을 때만 의미가 있다. 그래서 benchmark는 다음 원칙을 따른다.

- mock이 아니라 실제 CABT simulator game을 돌린다.
- seat와 turn order가 중요하므로 seat balancing을 한다.
- 후보 비교가 noisy해지지 않도록 공통 상대 패널을 쓴다.
- 작은 matchup slice를 과신하지 않도록 Wilson interval을 본다.
- daily meta에서 나온 field weight를 적용한다.
- 약한 proxy pilot이 local score를 부풀리지 않도록 pilot status label을 둔다.

첫 0630 relevance run은 Archaludon-family 후보를 선호했다.

| Candidate | Games | Raw score rate | Wilson low |
|---|---:|---:|---:|
| `flex_archaludon_0021` | 128 | 0.7500 | 0.6684 |
| `public_archaludon_75wr` reference | 112 | 0.7500 | 0.6624 |
| `flex_archaludon_0018` | 128 | 0.7422 | 0.6601 |
| `archaludon_75wr_vs_starmie` | 128 | 0.7109 | 0.6272 |
| `flex_alakazam_dunsparce_0000_seed` | 112 | 0.6786 | 0.5874 |

필드 가중 기준으로 보면 같은 run은 다음과 같았다.

| Candidate | 필드 가중 score | Edge vs field |
|---|---:|---:|
| `flex_archaludon_0021` | 0.6929 | +0.2322 |
| `flex_alakazam_dunsparce_0000_seed` | 0.6490 | +0.0896 |
| `public_archaludon_75wr` reference | 0.6213 | +0.1860 |
| `flex_archaludon_0018` | 0.5905 | +0.1298 |

여기서 멈췄다면 결론은 간단했을 것이다. Archaludon을 밀면 된다. 하지만 더 큰 validation을 돌리자 해석이 바뀌었다.

GPS=16 검증에서는 다음과 같았다.

| Candidate | Games | Raw score rate | Wilson low |
|---|---:|---:|---:|
| `public_archaludon_75wr` reference | 224 | 0.7634 | 0.7036 |
| `flex_alakazam_dunsparce_0000_seed` | 224 | 0.7589 | 0.6989 |
| `flex_archaludon_0021` | 256 | 0.6914 | 0.6323 |
| `flex_archaludon_0018` | 256 | 0.6758 | 0.6162 |
| `archaludon_75wr_vs_starmie` | 256 | 0.6289 | 0.5682 |

필드 가중 기준으로는:

| Candidate | 필드 가중 score | Edge vs field |
|---|---:|---:|
| `flex_alakazam_dunsparce_0000_seed` | 0.7430 | +0.1837 |
| `public_archaludon_75wr` reference | 0.6600 | +0.2246 |
| `flex_archaludon_0021` | 0.5674 | +0.1067 |
| `flex_archaludon_0018` | 0.5223 | +0.0616 |

여기서 첫 번째 큰 방법론적 교훈이 나온다.

> 작은 로컬 벤치마크는 우연히 좋은 panel에서 맞는 family를 고를 수 있다. 반대로 맞는 후보를 과대평가할 수도 있다. 두 번째 validation run은 선택 사항이 아니다. 결론을 바꿀 수 있다.

현재 해석은 "Alakazam이 해결됐다"가 아니다.

- Archaludon은 여전히 실용적인 강한 family다. 파일럿이 비교적 쉽고 Alakazam/Starmie-like panel에 압박을 줄 수 있다.
- `flex_alakazam_dunsparce_0000_seed`는 더 큰 validation에서 의외로 강했으므로 진지한 B 후보 트랙으로 되살려야 한다.
- `public_archaludon_75wr`은 좋은 reference/oracle이다. 하지만 공개 노트북을 그대로 복사하는 것이 목표는 아니다. 유용한 것은 behavior comparison과 controlled derivative testing이다.
- Marnie/Munkidori deck family는 아마 가장 가치 있는 field discovery지만, 아직 pilot-gated 상태다.

---

## 9. Local-live gap

이 프로젝트는 불편하지만 유용한 실패를 겪었다. 로컬에서는 그럴듯해 보였던 후보가 live에서는 기대만큼 점수를 내지 못했다.

원인은 하나가 아니다. local-live gap은 여러 곳에서 생길 수 있다.

| 원인 | 증상 | 해결 |
|---|---|---|
| 약한 opponent pilot | proxy가 field deck을 잘못 플레이해서 local win rate가 부풀려진다. | opponent pilot audit / repair |
| meta coverage 부족 | live에서 후보를 이기는 덱이 benchmark panel에 없다. | log 기반 field panel refresh |
| 후보 pilot이 field average를 따라 하기만 함 | deck은 괜찮지만 pilot이 average pilot을 이기지 못한다. | fidelity가 아니라 beat-field supremacy 최적화 |
| Winner's curse | 많은 variant 중 최고를 골라 live에서 회귀한다. | holdout validation과 shrinkage |
| Meta drift | 오늘 live field가 어제 log field와 다르다. | daily weighting과 trend report |
| Runtime / packaging 차이 | local smoke는 통과하지만 live에서 다르게 동작한다. | self-game diagnostics와 package validation |

이 때문에 프로젝트 정책이 바뀌었다.

```text
그럴듯해 보인다는 이유만으로 "meta snapshot" 후보를 공개하거나 제출하지 않는다.
새로 갱신한 known reference보다 우위가 있거나, 없으면 검증된 edge가 없다고 말한다.
```

이것은 보수적이기 위한 보수성이 아니다. leaderboard를 validation set처럼 써버리지 않기 위한 장치다.

---

## 10. 프로젝트 히스토리 원장

이 시리즈는 나중의 Working Note / report를 위한 기록이기도 하다. 그래서 최종 결론만 남기면 부족하다. 실패한 방향, 정책이 바뀐 이유, benchmark가 틀렸던 이유도 함께 남아야 한다.

6월 30일 snapshot까지의 프로젝트 원장은 다음과 같다.

| Phase | 무슨 일이 있었나 | 무엇을 배웠나 | 남는 artifact |
|---|---|---|---|
| Competition orientation | competition page, simulator note, sample deck, public example, local guide를 읽었다. | 제출 단위는 deck만이 아니다. deck, callable pilot, packaging discipline이 합쳐져야 한다. | Guide notes, 초기 `AGENTS.md`, sample deck inventory |
| First baseline era | Lucario-style rule-based notebook을 만들고, `main.py`, deck CSV, `cg` package inclusion, notebook cell visibility, standalone behavior를 디버깅했다. | 깨끗한 notebook은 제출 가능하지만, meta context 없는 rule-only pilot은 약하다. packaging mistake는 strategy failure처럼 보일 수 있다. | Lucario baseline notebooks, submission-format checks |
| Public example reading | 고득점 public notebook들을 비교하고, mainline code로 복사하지 않고 아이디어만 추출했다. | public notebook은 reference, opponent, behavior oracle로 쓰는 편이 가장 좋다. 그대로 복제하면 연구 루프가 죽는다. | Public example inventory, derivative-candidate notes |
| Two-agent portfolio turn | 하나의 "best" agent가 아니라 두 개의 active submission을 portfolio로 생각하기 시작했다. | 목표는 stable average 하나가 아니라 portfolio upside와 서로 다른 failure mode다. | Two-submission specs, portfolio benchmark tables |
| Local engine breakthrough | `kaggle_environments.make("cabt")`와 packaged CABT engine으로 full local game이 돈다는 것을 확인했다. | mock benchmark는 필요 없지만, engine이 reliable RNG seeding을 공개하지 않는다. 따라서 CRN-style paired comparison은 어렵다. | Seat-balanced benchmark runner, engine diagnostics |
| MetaStore / FieldStore buildout | raw replay를 매번 읽는 대신 date-partitioned compact store로 옮겼다. | daily log는 너무 크다. 영구 계층은 compact하고 queryable해야 하며 source date와 연결되어야 한다. | `Archive/MMDD`, `MetaStore/YYYYMMDD`, `FieldStore/YYYYMMDD` |
| Deck discovery layer | exact deck hash, winner decklist, field share, matchup matrix를 추출했다. | deck은 발견됐지만 내 agent가 아직 플레이하지 못하는 상태가 있을 수 있다. deck discovery와 pilot acquisition은 별도의 gate다. | Ready decks, exact deck summaries, matchup matrices |
| Strategy profiler | first attack turn, attack cadence, draw/search rate, ability rate, attack concentration, prize rate 같은 behavior feature를 추가했다. | archetype 이름만으로는 부족하다. deck의 mechanism은 behavior space에서 설명되어야 한다. | Strategy profile reports |
| Pilot audit and repair | pilot fidelity, weak-proxy detection, local-live gap diagnosis, beat-field pilot target, holdout / shrinkage를 명세화했다. | 평균 field pilot을 충실히 따라 하는 것은 opponent로는 유용하지만, 제출 pilot은 field를 따라 하는 것이 아니라 이겨야 한다. | Pilot audit specs, gap diagnosis policy |
| Meta snapshot lessons | public meta snapshot notebook을 만들고, snapshot recommendation은 known reference를 이기거나 edge가 없다고 명시해야 한다는 정책을 강화했다. | 그럴듯한 candidate 공개로는 부족하다. note는 evidence, uncertainty, compromise decision을 분리해야 한다. | Meta snapshot notebooks, audit manifests |
| June 30 weekly staging | 5,734개 episode를 처리했고, Alakazam 상승, Archaludon 하락, Marnie/Munkidori 등장을 확인했다. | 가장 좋은 deck discovery가 가장 쉬운 current submission은 아닐 수 있다. pilot acquisition이 우선이다. | Daily meta report, deep analysis, benchmark staging report |

live ladder도 유용한 피드백을 줬다. 초기에 로컬에서는 그럴듯해 보였던 후보들이 600-800대에 머문 반면, 이후 Archaludon tempo challenger는 대략 high-900 band까지 올라갔다. Kaggle rating은 noisy하고 시간에 따라 변하므로 이 숫자를 최종 진실로 보지는 않는다. 하지만 workflow를 수정하기에는 충분한 신호다. 로컬 벤치마크는 known live reference와 calibration되어야 하고, "괜찮아 보인다"는 promotion criterion이 될 수 없다.

---

## 11. 보고서로 재조립 가능한 산출물 지도

나중에 통합 보고서를 만들 때 중요한 것은 단일 notebook이 아니다. 중요한 것은 evidence chain이다. 어떤 결론이 어떤 로그, 어떤 벤치마크, 어떤 정책 변경에서 나왔는지 이어져 있어야 한다.

| 산출물 계열 | 무엇을 기록하나 | 보고서에서 답하는 질문 |
|---|---|---|
| `Archive/MMDD` | 하루치 official replay raw log | ladder에서 실제로 무슨 일이 있었나? |
| `Archive/raw_zst/YYYYMMDD` | 압축 raw mirror | 저장 공간 한도 안에서 raw evidence를 보존할 수 있나? |
| `MetaStore/YYYYMMDD/daily_meta_report.md` | daily field share, exact deck hash, common matchup, strategy snapshot | 오늘 무엇이 바뀌었나? |
| `MetaStore/meta_trend_report.md` | cross-day share와 score movement | 어떤 archetype이 오르고 내리는가? |
| `FieldStore/YYYYMMDD` | ready decklist, field weight, selection report | 어떤 deck을 테스트해야 하고 panel weight는 어떻게 잡아야 하나? |
| Strategy profile tables | archetype별 behavior summary | 카드 리스트를 넘어, 그 deck은 왜 이기는가? |
| `pokemon_benchmark_runs/...` | local simulator result, weighted score, validation panel, portfolio report | runnable agent가 실제 current field proxy를 이기는가? |
| Pilot library / pilot audit outputs | fidelity, weak-proxy label, missing archetype coverage, behavior divergence | 이 benchmark opponent나 submitted pilot을 믿어도 되는가? |
| Decision extraction / BC datasets | replay에서 나온 observation-option-action rows | 발견된 deck의 실제 behavior를 learned pilot이 모방할 수 있는가? |
| Meta snapshot notebooks | 공개용 summary와 optional submission package | evidence trail을 잃지 않고 무엇을 공유할 수 있는가? |
| `AGENTS.md` and specs | 운영 정책과 promotion gate | workflow가 왜 그런 결정을 내렸는가? |

앞으로의 note는 evidence level도 분리해서 써야 한다.

| Level | Evidence type | 어떻게 써야 하나 |
|---|---|---|
| L0 | anecdotal live score 또는 단일 notebook claim | 힌트로는 유용하지만 이것만으로는 부족하다. |
| L1 | field census / deck frequency | meta weighting에는 좋지만 submission strength는 아니다. |
| L2 | log에서 나온 matchup matrix | deck discovery와 counter-map reasoning에 좋다. |
| L3 | validated pilot이 포함된 runnable local benchmark | candidate screening에 좋다. |
| L4 | holdout / larger validation with shrinkage | serious promotion 전 필요하다. |
| L5 | known reference 대비 live calibration | local benchmark를 장기적으로 신뢰하려면 필요하다. |

초기에 빠져 있던 부분이 바로 이것이다. working note는 "A 후보가 좋아 보였다"에서 멈추면 안 된다. 그 주장이 어떤 evidence level에 의해 지지되는지, 무엇이 아직 covered되지 않았는지, 나중에 누군가 재현하거나 반박하려면 어떤 artifact를 보면 되는지까지 남겨야 한다.

---

## 12. RL과 behavior cloning: 왜 imitation부터 시작하는가

Pokémon TCG는 reinforcement learning 문제로 보기에 매우 매력적이다. 하지만 pure self-play에서 시작하는 것은 비싸고 불안정하다.

- action space는 매 턴 달라지는 합법 선택지 목록이다.
- hidden information과 stochastic draw 때문에 credit assignment가 noisy하다.
- 많은 합법 수는 카드 문맥을 이해해야만 왜 나쁜지 알 수 있다.
- deck-specific pilot은 단순한 win/loss feedback보다 sequencing이 중요하다.
- simulator는 local game을 돌리기엔 충분히 빠르지만 blind exploration에 낭비할 만큼 무한히 빠르지는 않다.

그래서 첫 RL 방향은 replay log 기반 behavior cloning이다.

supervised learning object는 다음과 같다.

$$
\pi_\theta(i \mid o_t, c_i)
$$

여기서 \(c_i\)는 simulator의 `select.option` menu에 있는 합법 선택지다. masked softmax는 합법 선택지만 score한다.

$$
P(i \mid o_t) =
\frac{\exp(s_\theta(o_t, c_i))}
{\sum_{j \in \text{legal}(o_t)} \exp(s_\theta(o_t, c_j))}
$$

label은 replay에서 실제로 선택된 선택지다. 중요한 engineering detail이 하나 있다. replay schema에서는 step \(i\)의 observation이 step \(i+1\)에 나타나는 action과 연결된다. 이 alignment를 틀리면 조용히 엉뚱한 모델을 학습하게 된다.

초기 BC playbook은 이미 다음을 다뤘다.

- replay schema audit
- winner-only decision extraction
- game-level split
- legal-option candidate feature scaffold
- masked legal-option scorer
- minimal behavior-cloning loop

초기 smoke test는 의도적으로 작게 돌렸다. 약 30 games, 2,120 decision records 정도로 pipeline이 연결되는지만 확인했다. 이 결과를 성능 주장으로 해석하면 안 된다.

BC의 실제 용도는 "내일부터 neural policy로 agent를 갈아엎는다"가 아니다. 지금 당장의 목표는 더 작고 실용적이다.

```text
replay behavior로 발견된 deck의 pilot을 개선한다.
behavior profile로 우리 pilot과 field pilot을 비교한다.
learned scorer를 rule-based agent 안의 prior 또는 fallback ranker로 쓴다.
local candidate가 live에서 약한 이유를 policy divergence로 찾는다.
```

요약하면 RL은 마법 버튼이 아니다. 파일럿 격차를 닫기 위한 도구다.

---

## 13. 6월 30일 기준 덱 해석

현재 전략 지형은 다음과 같다.

### Alakazam / Dunsparce

6월 30일 가장 흔한 archetype이다. 필드의 Archaludon을 잘 잡기 때문에 상승한 이유가 명확하다. best exact deck sample은 287 games에서 약 0.627 score rate를 보였다. `flex_alakazam_dunsparce_0000_seed` benchmark 결과는 이 계열을 두 번째 submission track으로 다시 검토하게 만들었다.

위험은 있다. 과거 Alakazam build들이 live에서 약했다. 이는 이 덱이 파일럿 품질과 로컬 proxy 품질에 민감하다는 뜻이다.

### Archaludon

6월 30일 raw field 기준 최고 archetype은 아니었다. 하지만 여전히 가장 실용적인 rule-based candidate family다. 빠르게 공격하고, tempo plan이 명확하며, 로컬 파일럿이 brittle combo decision 없이 실행할 수 있다. public Archaludon-like notebook들이 강했기 때문에 reference로도 중요하다.

위험은 필드의 Alakazam이 필드의 Archaludon을 강하게 잡는다는 점이다. 따라서 Archaludon pilot은 average field Archaludon pilot보다 좋아야 한다.

### Marnie's Impidimp / Munkidori

가장 중요한 discovery다. 의미 있는 share, non-tiny archetype 중 가장 좋은 field-weighted signal, 그리고 Alakazam과 Archaludon 둘 다 상대로 강한 관측 matchup을 갖고 있다.

위험은 pilot availability다. setup이 많이 필요한 combo family이므로 naive pilot은 edge를 쉽게 망칠 수 있다.

### Crustle

wall/collapse stress deck이다. 일반적인 prize-race strategy처럼 보이지 않는다. 당장 submission candidate가 아니더라도 가치가 있다. 우리 agent가 non-tempo game plan을 처리할 수 있는지 드러내기 때문이다.

### Okidogi / Solrock

작지만 강한 midrange family다. log상으로는 Alakazam/Archaludon panel의 blind spot으로 특히 중요하다.

---

## 14. 다음에 해야 할 일

다음 주는 random card swap search가 되어서는 안 된다. 구조화된 research loop가 필요하다.

1. **Marnie/Munkidori pilot acquisition**  
   winner decision을 추출하고, pilot profile을 만든다. exact rank-1/rank-2 deck은 trace-level sanity check를 통과한 뒤에만 benchmark한다.

2. **Alakazam pilot audit**  
   `flex_alakazam_dunsparce_0000_seed`, 과거 약했던 Alakazam submission, 0630 rank-1 exact deck을 비교한다. 차이가 deck composition인지, pilot priority인지, benchmark proxy인지 구분한다.

3. **Archaludon robustness validation**  
   `0021`, `0018`, public-Archaludon-like variant를 더 큰 holdout에 넣는다. 첫 GPS=8 signal이 stable하다고 가정하지 않는다.

4. **Stress opponent expansion**  
   Crustle, Okidogi/Solrock, Dragapult, Cynthia/Gabite pilot을 충분히 쓸 만한 수준으로 확보한다. 나쁜 stress pilot은 없는 것보다 나쁘다. benchmark가 틀린 교훈을 주기 때문이다.

5. **Local-live gap accounting**  
   모든 live submission에 대해 local predicted win rate, field-weighted matchup expectation, 실제 rating movement를 비교한다. 목적은 단순한 scoring이 아니라 simulator workflow를 calibrate하는 것이다.

6. **BC as pilot repair**  
   behavior cloning은 먼저 deck-specific pilot imitation에 사용한다. supervised pipeline이 유용한 prior를 만들 수 있을 정도로 강해진 뒤에 RL/self-play를 고려한다.

---

## 15. 결론

첫 주가 지나며 이 대회를 보는 관점이 바뀌었다.

이것은 단순한 "최고의 덱 찾기" 대회가 아니다. 세 층의 게임이다.

```text
deck discovery: 어떤 덱이 강한가
pilot quality: 그 덱을 얼마나 잘 굴리는가
meta timing: 지금 그 덱이 맞는 타이밍인가
```

6월 30일 로그는 이 점을 특히 잘 보여줬다. Alakazam은 Archaludon을 잘 잡기 때문에 가장 큰 점유율을 차지하는 덱이 되었다. Marnie/Munkidori는 Alakazam과 Archaludon 둘 다 잡을 가능성을 보였기 때문에 새로운 위협으로 떠올랐다. Archaludon은 archetype 자체가 raw field leader가 아니어도, 깨끗하고 강한 파일럿이 average field Archaludon보다 낫다면 여전히 local candidate로 유효했다.

실용적인 교훈은 evidence에 정직해야 한다는 것이다.

- 로그는 강한 덱을 찾아낼 수 있다.
- behavior profile은 그 덱이 왜 이기는지 설명할 수 있다.
- benchmark는 실제로 실행 가능한 agent를 테스트한다.
- pilot fidelity가 benchmark의 의미를 결정한다.
- live score는 전체 loop를 calibration한다.

다음 글의 방향은 meta discovery에서 pilot acquisition으로 넘어가는 것이다. 특히 로그는 이미 강하다고 말하지만 local agent가 아직 잘 플레이하지 못하는 emerging deck들이 핵심이다.
