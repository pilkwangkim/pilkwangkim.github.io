---
title: "Verifiable Quantum Circuit Compilation: 세 가지 병목으로 시작하는 회로 최적화"
description: "qubit과 phase, uncomputation부터 invsqrt·Reed–Solomon decoder·분자 Hamiltonian까지. 세 가지 병목을 예제와 그림으로 이해하고, 공식 baseline의 검증 결과와 비용을 재현한다."
date: 2026-10-23 10:00:00 +0900
categories: [Physics, Quantum Computing]
tags: [kaggle, quantum-circuit-compilation, quantum-computing, coding-agents, verification, optimization, working-note]
math: true
mermaid: true
image:
  path: /assets/img/posts/quantum-circuit-compilation/cover-1.webp
  alt: "복잡한 회로와 간결한 회로가 같은 검증 지점으로 이어지는 개념도"
  width: 1672
  height: 941
  fit: contain
  hide_caption: true
pin: false
published: true
topic: quantum-circuit-compilation
lang: ko
translation_key: verifiable-quantum-circuit-compilation-getting-started
series: quantum-circuit-compilation
series_order: 0
---

[Read in English]({{ site.baseurl }}/posts/Verifiable-Quantum-Circuit-Compilation-Getting-Started/)

## 들어가며: 반복되는 계산 하나가 전체 비용을 바꾼다

양자 알고리즘을 소개하는 글에는 Fourier transform, phase estimation, 분자 시뮬레이션 같은 큰 주제가 자주 나온다. 그런데 실제 회로를 만들다 보면 훨씬 작은 계산에서 막힌다. 정수를 더하고 곱하는 데 qubit이 몇 개 필요한가? 표에서 숫자 하나를 읽으려면 gate를 얼마나 써야 하는가? 계산에 쓴 임시 공간은 어떻게 비울 것인가? 이런 계산이 수없이 반복되기 때문에, 하나를 어떻게 구현하느냐가 전체 비용을 좌우한다.

[Google Quantum AI — Verifiable Quantum Circuit Compilation][competition]은 세 양자 알고리즘에서 비용이 많이 드는 **subroutine을 더 적은 자원으로 구현하는 대회**다. 고전 프로그램에서 병목 함수를 최적화하는 일과 비슷하다. 다만 연산 횟수뿐 아니라 계산에 동시에 필요한 logical qubit 수도 줄여야 한다.

어떤 subroutine을 1,000번 호출하고, 한 번에 비용이 큰 gate 8,000개를 쓴다고 해 보자. 이 부분에만 800만 개가 필요하다. 같은 동작을 6,000개로 구현하면 600만 개로 줄어든다. 반복 호출의 효과를 보여 주기 위한 가상 예제이며, 대회의 실제 호출 횟수나 개선 결과는 아니다. qubit은 다음 호출에서 재사용할 수 있으므로 gate처럼 호출 횟수만큼 더하지 않는다. 대신 한 시점에 얼마나 많은 qubit이 필요한지를 따로 센다.

| 전체 응용 | 이 대회에서 최적화할 부분 | 왜 자주 필요한가 |
|---|---|---|
| Coulomb 상호작용 시뮬레이션 | `invsqrt`: inverse square root에 비례하는 phase | 입자쌍 사이 상호작용을 적용할 때 반복된다. |
| Decoded Quantum Interferometry | `dqi`: Reed–Solomon decoding과 phase 보정 | 최적화 알고리즘 안에서 오류 패턴을 지워야 한다. |
| 분자의 전자 구조 시뮬레이션 | `dfthcsos`: factorization, 각도 조회, 회전 구성 | Hamiltonian을 인코딩하는 연산을 반복해 사용한다. |

세 분야의 이름을 보면 배워야 할 것이 너무 많아 보일 수 있다. 하지만 처음부터 전체 알고리즘을 공부할 필요는 없다. 양자 회로가 정보를 어떻게 바꾸는지 먼저 이해하고, 한 subroutine의 입력과 출력, 임시 공간, 허용오차를 따라가면 된다. 전체 물리 시뮬레이터나 조합 최적화 알고리즘을 처음부터 만드는 대회는 아니다. [공식 배경 설명][domain]

이 글도 그 순서로 설명한다. 1–4절에서는 qubit, gate, phase, reversible computation과 검증 방식을 다룬다. 5–7절에서는 세 병목을 작은 예제로 이해한 뒤 공식 baseline 코드와 연결한다. 8절에서는 비용식을 비교하고, 9절부터는 공식 파일을 받아 로컬에서 실행하는 방법과 실제 검증 결과를 정리한다. Python의 비트 연산과 간단한 행렬·함수 표기를 읽을 수 있으면 따라갈 수 있도록, 양자 계산 용어는 나오는 곳에서 설명한다.

**어디까지 확인했나.** 이 글은 2026년 10월 23일에 확인한 자료를 바탕으로 썼다. 공식 원본을 내려받아 로컬 환경을 구성했고, `invsqrt` baseline 회로의 전체 로컬 검증을 마쳤다. 작은 수학 예제와 그림은 설명을 위한 것이며 최적화 실험 결과는 아니다. 다른 두 과제의 전체 로컬 실행과 후보 회로 비교, Kaggle 제출은 아직 하지 않았다. 표지는 회로 최적화와 검증을 표현한 개념도다.

## 1. qubit, amplitude, phase부터 이해하기

### 1.1 각 비트열에는 amplitude가 있다

고전 비트는 0 또는 1이다. qubit에서도 측정으로 구별할 수 있는 두 기본 상태를 `|0⟩`, `|1⟩`로 쓴다. 이를 **computational basis**라고 한다. 세로줄과 꺾쇠는 양자 상태를 나타내는 표기다. qubit 하나의 pure state는 다음처럼 쓸 수 있다.

$$
\lvert \psi\rangle=\alpha\lvert 0\rangle+\beta\lvert 1\rangle,
\qquad |\alpha|^2+|\beta|^2=1.
$$

α와 β는 **amplitude**라는 복소수다. computational basis에서 측정하면 0이 나올 확률은 ∣α∣², 1이 나올 확률은 ∣β∣²다. amplitude 자체가 확률은 아니다. 음수나 복소수가 될 수 있고, 서로 더해지거나 상쇄될 수도 있다. 이 점이 단순히 0과 1을 무작위로 고르는 프로그램과 다르다.

n개의 qubit으로는 n자리 비트열을 나타낸다. 예를 들어 `|101⟩`은 qubit 세 개의 basis state 하나다. 일반적인 양자 상태는 여러 비트열과 각각의 amplitude를 함께 써서 나타낸다.

$$
\lvert \psi\rangle=\sum_{z=0}^{2^n-1}\alpha_z\lvert z\rangle,
\qquad \sum_z|\alpha_z|^2=1.
$$

이런 상태를 **superposition**이라고 한다. 비트열 2ⁿ개를 표현할 수 있다고 해서 2ⁿ개의 답을 한 번에 읽을 수 있는 것은 아니다. 측정하면 그중 하나만 나온다. 양자 알고리즘은 amplitude를 조절해서 유용한 결과가 나올 확률을 높인다. 표기와 기본 연산은 [IBM의 양자 정보 강의][ibm-states]에서도 살펴볼 수 있다.

여러 qubit을 묶어 하나의 값으로 다루는 단위를 **register**라고 한다. `invsqrt`의 `x` register는 qubit 22개로 정수를 저장한다. little-endian은 q₀을 가장 낮은 자리로 읽는다는 뜻이다. q₀=1, q₁=0, q₂=1이면 값은 1+0×2+1×4=5다. 회로도에서 선을 그린 순서와 숫자의 자리값은 따로 확인해야 한다.

### 1.2 확률이 같은 두 상태도 다르게 동작한다

다음 두 상태를 비교해 보자.

$$
\lvert +\rangle=\frac{\lvert 0\rangle+\lvert 1\rangle}{\sqrt{2}},
\qquad
\lvert -\rangle=\frac{\lvert 0\rangle-\lvert 1\rangle}{\sqrt{2}}.
$$

둘 다 지금 측정하면 0과 1이 각각 절반의 확률로 나온다. 차이는 두 번째 항의 **relative phase**다. `|−⟩`에 붙은 마이너스는 amplitude에 −1=eⁱπ를 곱한 것으로, 복소평면에서 반 바퀴 회전한 셈이다. 이후 연산에 따라 이 차이가 측정 확률의 차이로 나타날 수 있다.

이 차이를 드러내는 연산이 **Hadamard gate H**다. H는 다음과 같이 작동한다.

$$
H\lvert 0\rangle=\lvert +\rangle,\quad H\lvert 1\rangle=\lvert -\rangle,
\qquad H\lvert +\rangle=\lvert 0\rangle,\quad H\lvert -\rangle=\lvert 1\rangle.
$$

0에서 시작해 H를 두 번 적용하면 다시 0이 된다. 그런데 두 H 사이에 Z를 넣어 상대 부호를 뒤집으면 마지막에는 1이 된다. phase의 차이가 **interference**를 거쳐 측정값의 차이로 바뀐 것이다.

![H, Z, H를 순서대로 적용할 때 진폭의 부호와 측정 확률이 바뀌는 네 단계](/assets/img/posts/quantum-circuit-compilation/fig-01-interference.svg){: width="1152" height="605" }

*그림 1. 파란색은 양의 실수 amplitude, 주황색은 음의 실수 amplitude다. 첫 H 뒤와 Z 뒤의 측정 확률은 같지만, 마지막 H를 적용하면 결과가 달라진다. 원리를 설명하는 qubit 하나의 예제이며, 제출 회로에 H를 자유롭게 쓸 수 있다는 뜻은 아니다.*

직접 계산해 보면 마지막 H 뒤에서 `|0⟩`의 amplitude는 (1−1)/2=0, `|1⟩`의 amplitude는 (1+1)/2=1이다. amplitude끼리 상쇄되거나 더해지는 모습을 이렇게 작은 예에서도 볼 수 있다.

상태 전체에 같은 eⁱφ를 곱하면 **global phase**만 바뀌므로 그 상태의 측정 확률은 그대로다. 대회에서 맞춰야 하는 것은 입력 x에 따라 달라지는 phase다. superposition의 각 항에 서로 다른 phase가 붙으면 relative phase가 바뀌고, 이후 알고리즘의 결과에도 영향을 준다. [phase와 interference 예제][ibm-phase]

### 1.3 양자 회로는 상태를 순서대로 바꾸는 프로그램이다

회로도에서는 보통 가로선 하나가 qubit 하나를 나타내고, 왼쪽에서 오른쪽으로 gate를 적용한다. gate는 한 qubit 또는 여러 qubit에 작용하는 연산이다. 이를 행렬로 쓰면, gate를 적용하는 일은 상태 벡터에 행렬을 곱하는 것과 같다.

$$
X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
H=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}.
$$

X는 0과 1을 바꾼다. Z는 `|1⟩` 항의 amplitude에 −1을 곱한다. H는 두 basis state의 amplitude를 섞는다. X와 Z 모두 그림에서는 작은 상자로 보이지만, 비트값을 바꾸는 일과 phase를 바꾸는 일은 구별해야 한다.

| 연산 | computational basis에서 하는 일 | 이 대회에서 필요한 관점 |
|---|---|---|
| X | 0 ↔ 1 | 비트값을 바꾼다. |
| CX 또는 CNOT | b ← b XOR a | 입력을 보존하는 조건부 비트 연산이다. |
| CCX 또는 Toffoli | c ← c XOR (a AND b) | reversible AND와 산술의 기본 연산이다. |
| Z, CZ, CCZ | 조건이 맞는 항의 부호 반전 | 비트값이 같아도 phase는 바뀐다. |
| Z_POW(q,t) | q=1인 항에 exp(iπt)를 곱함 | 숫자를 phase로 바꾸는 데 쓴다. |
| H | 두 basis state의 amplitude를 섞음 | 원리를 설명하는 데 필요하지만, H를 단독 명령으로 쓰는 것은 kickmix에서 허용하지 않는다. |

`Z_POW`의 t는 반 바퀴 단위다. 한 바퀴를 1로 세는 turns와는 2배 차이가 있다.

## 2. 고전 프로그램을 그대로 양자 회로로 옮길 수 없는 이유

### 2.1 AND를 계산해도 입력은 보존해야 한다

AND 결과가 0이라는 것만으로는 원래 입력이 00, 01, 10 중 무엇이었는지 알 수 없다. 여러 입력이 같은 출력으로 합쳐지기 때문이다. 측정이 없는 양자 gate는 **reversible**, 즉 원래 상태로 되돌릴 수 있어야 한다. 따라서 입력 정보를 이런 식으로 버리면 안 된다. 이런 양자 gate는 수학적으로 **unitary** 행렬로 나타낸다.

기본적인 해결책은 입력을 그대로 두고, 별도 target bit에 계산 결과를 XOR하는 것이다.

$$
U_f:\lvert x\rangle\lvert z\rangle\longmapsto\lvert x\rangle\lvert z\oplus f(x)\rangle.
$$

z가 0이면 결과가 f(x)다. 다시 같은 XOR을 적용하면 z로 돌아간다. `CCX(a,b,c)`는 f(a,b)=a AND b인 경우다.

| a | b | 처음 c | CCX 뒤 c | CCX를 한 번 더 적용한 c |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 | 0 |

덧셈과 곱셈도 이런 reversible 연산으로 만든다. 고전 코드에서는 `a *= b`로 이전 a를 덮어쓸 수 있다. 하지만 reversible 회로에서는 어떤 값을 보존하고 어떤 값을 복원할지 미리 정해야 한다. 이 과정에서 carry, partial product, 비교 결과를 저장할 임시 qubit이 필요하다. [reversible 고전 계산을 만드는 방법][ibm-reversible]

### 2.2 scratch에 정보가 남으면 interference가 달라진다

계산을 돕는 보조 qubit을 **ancilla**라고 하고, 임시 작업 공간을 scratch라고 부르기도 한다. 보통 계산을 마치면 이 공간을 0으로 돌려놓아야 한다. 단순히 메모리를 깔끔하게 관리하려는 목적만은 아니다.

첫 qubit이 `|+⟩`이고, CX로 그 비트값을 두 번째 보조 qubit에 복사한다고 해 보자.

$$
\frac{\lvert 0\rangle+\lvert 1\rangle}{\sqrt{2}}\lvert 0\rangle
\longmapsto
\frac{\lvert 00\rangle+\lvert 11\rangle}{\sqrt{2}}.
$$

CX가 복사한 것은 computational basis의 비트값이다. 결과는 독립적인 `|+⟩` 두 개가 아니므로, 임의의 양자 상태를 복제한 것은 아니다. 대신 두 qubit 사이에 **entanglement**가 생겼다. 보조 qubit에도 첫 qubit의 비트값을 구별할 수 있는 정보가 남은 것이다. 보조 qubit을 무시하면 첫 qubit은 원래 `|+⟩`처럼 간섭하지 않는다. 원래 상태에서는 H를 적용한 뒤 0만 나와야 하지만, 이 경우에는 0과 1이 반반씩 나온다.

입력 하나를 넣어 `x`가 그대로 나오는지만 확인하면 이 문제를 놓칠 수 있다. 그래서 검증기는 **입력 복원, scratch 초기화, phase**를 함께 검사한다. 파일이 작아지거나 gate 수가 줄어도 이 조건을 어기면 사용할 수 없는 회로다.

### 2.3 compute → phase → uncompute

`invsqrt`처럼 f(x)에 비례하는 phase가 필요할 때는 다음 순서로 구현할 수 있다.

$$
\begin{aligned}
\lvert x\rangle\lvert 0\rangle
&\xrightarrow{U_f}\lvert x\rangle\lvert f(x)\rangle\\
&\xrightarrow{\mathrm{phase}}e^{2\pi i f(x)}\lvert x\rangle\lvert f(x)\rangle\\
&\xrightarrow{U_f^\dagger}e^{2\pi i f(x)}\lvert x\rangle\lvert 0\rangle.
\end{aligned}
$$

실제 회로에서 f(x)는 정해진 비트 수로 표현한 근삿값이다. 마지막 단계인 **uncomputation**은 계산에 쓴 register를 원래대로 돌려놓는다. 앞에서 입력별로 붙인 phase는 그대로 남는다. 이전 gate를 거꾸로 적용하되 각 gate의 역연산을 써야 한다. X·CX·CCX는 같은 연산을 한 번 더 하면 원래대로 돌아가지만, 일반적인 회전은 각도 부호를 바꿔야 한다.

![계산과 위상 적용 뒤 작업 공간을 복원하는 경로와 AND 임시값의 측정 기반 정리](/assets/img/posts/quantum-circuit-compilation/fig-02-uncomputation.svg){: width="1200" height="1010" }

*그림 2. 위쪽은 phase를 남기고 scratch를 비우는 일반적인 구조다. 아래쪽은 임시 AND 결과를 지우는 두 가지 방법이다. ●는 control, ⊕는 target bit 반전, 점선은 고전 측정 비트다. HMR을 쓰는 경로에는 측정 결과에 따른 phase 보정이 필요하다. 어떤 계산 결과든 비용 없이 지울 수 있다는 뜻은 아니다.*

### 2.4 measurement-based uncomputation은 어떻게 Toffoli를 아낄까

임시 AND 결과 t=a AND b를 CCX로 지우려면 Toffoli 하나를 더 써야 한다. 다른 방법은 t를 X basis에서 측정한 뒤 0으로 돌려놓는 `HMR`을 쓰는 것이다. 측정 결과 m에 따라 다음 부호가 생긴다.

$$
\lvert a,b\rangle\lvert ab\rangle
\xrightarrow{\mathrm{HMR}}
(-1)^{mab}\lvert a,b\rangle\lvert 0\rangle.
$$

m=0이면 추가 보정이 필요 없다. m=1일 때 CZ(a,b)를 적용하면 $(-1)^{ab}$가 한 번 더 곱해져 불필요한 부호가 사라진다. 이 비용 모델에서는 조건부 CZ에 Toffoli 비용을 부과하지 않는다. 이것이 **measurement-based uncomputation(MBU)**의 간단한 예다.

측정이나 보정을 빼면 원하는 회로와 다른 동작을 한다. 정보가 남은 qubit을 `R`로 강제로 0으로 만드는 것도 해결책이 아니다. `R`은 이미 0으로 복원한 공간을 해제하는 명령이다. 정보를 잘못 버리면 시뮬레이터에서 무작위 phase가 생긴다. 정확한 명령 의미는 [kickmix 문서][kickmix-docs]를 따른다.

**어떤 값을 언제 지울까?** 어떤 값은 계산을 거꾸로 하는 것보다 측정하고 보정하는 편이 싸다. 반대로 보정 자체에 비용이 많이 드는 값도 있다. 여러 중간값을 동시에 보관하면 필요한 qubit 수가 늘고, 일찍 지운 뒤 다시 계산하면 gate 수가 늘 수 있다. 회로 전체를 보면서 이 둘 사이의 적절한 선택을 찾는 것이 reversible 회로 최적화다.

## 3. 회로를 어떻게 구현하느냐에 따라 비용이 달라진다

### 3.1 같은 수학식도 여러 회로로 구현할 수 있다

“1/√x를 계산한다”는 설명만으로는 회로를 만들 수 없다. 소수점 아래 몇 비트까지 계산할지, 어떤 근사식과 곱셈 방법을 쓸지, 임시값은 언제 지울지 정해야 한다. 같은 함수를 계산해도 이런 선택에 따라 회로의 크기와 비용이 달라진다.

| 구현 단계 | invsqrt를 예로 들면 | 정해야 할 것 |
|---|---|---|
| 필요한 기능 | 입력 x의 inverse square root에 비례하는 phase 적용 | 대회의 입출력 조건에 맞춘다. |
| 수치 알고리즘 | normalization·근사식·Newton 보정 | 근사 차수, 구간 수, 보정 횟수 |
| 정수 구현 | fixed-point 곱셈·덧셈·shift | 비트 정밀도, 하위 비트 생략, register 재사용 |
| reversible gate | CX·CCX·phase·MBU | 덧셈 회로, AND uncomputation, scratch 재사용 시점 |
| 실제 하드웨어 | logical gate를 물리 연산으로 구현 | 이 대회에서 직접 실행하거나 최적화하지는 않는다. |

이 대회에서 **compilation**은 소스 코드를 기계어로 번역하는 작업보다 범위가 넓다. 허용오차를 지키면서 수치 알고리즘과 reversible 회로를 함께 설계한다. 회로를 생성하는 고전 프로그램이 빨라지는 것과, 생성된 회로의 공식 비용이 내려가는 것은 서로 다른 개선이다.

### 3.2 logical qubit 수와 비용이 큰 gate를 센다

대회의 Q는 **logical qubit** 수다. 실제 장비에서 오류를 막으며 logical qubit 하나를 유지하려면 여러 physical qubit이 필요할 수 있다. 따라서 Q=260을 물리 소자 260개라고 읽으면 안 된다. 장비 규모를 추정하려면 물리적 오류율과 quantum error correction 방식도 정해야 한다.

gate마다 비용도 다르다. 이 대회에서는 Toffoli 같은 non-Clifford 연산의 비용이 크다. `invsqrt`는 T gate 4개를 Toffoli 하나로 환산하며, 임의 각도 회전에는 요구 정밀도에 따른 rotation synthesis 비용을 더한다. Clifford는 X·CX·Z·H 등이 속하는 gate 집합이고, T와 Toffoli는 여기에 속하지 않는다. T는 $\lvert 1\rangle$ 항에 $e^{i\pi/4}$를 곱하는 phase gate다. 공식 비용식에서 Clifford를 무료로 취급하더라도, 실제 장비에서 시간이 전혀 들지 않는다는 뜻은 아니다.

`invsqrt`는 logical qubit 수 Q와 gate 환산 비용 C를 곱해 평가한다. 이 비용식으로 두 자원의 관계를 살펴보자.

$$
\mathrm{cost}=Q C,\qquad
\frac{\mathrm{cost}_{\mathrm{new}}}{\mathrm{cost}_{\mathrm{old}}}
=\frac{Q_{\mathrm{new}}}{Q_{\mathrm{old}}}
\frac{C_{\mathrm{new}}}{C_{\mathrm{old}}}.
$$

qubit을 10% 줄이는 대신 gate 환산 비용이 5% 늘었다면, 전체 비용은 0.9×1.05=0.945가 되어 5.5% 줄어든다. 반대로 gate를 10% 줄이려고 qubit을 20% 늘리면 0.9×1.2=1.08로 비용이 커진다. 둘 다 비용식을 설명하기 위한 가상 예제다. 실제 후보 회로는 먼저 정확성 검증을 통과해야 한다.

## 4. 이 회로는 왜 고전 CPU로 검증할 수 있을까

### 4.1 basis input 하나의 비트값과 phase를 추적한다

일반적인 n-qubit 상태를 그대로 저장하려면 최대 2ⁿ개의 amplitude가 필요하다. 하지만 `kickmix`에서 허용하는 연산은 reversible Boolean gate, diagonal phase, MBU 등으로 제한된다. basis input 하나를 넣었을 때는 비트열 하나와 phase, 측정 기록을 따라가면 된다. 예를 들어 CX는 저장한 비트에 XOR를 적용하고, Z는 조건에 맞을 때 phase의 부호를 바꾼다.

앞에서 interference를 설명할 때 쓴 H처럼, basis input 하나를 여러 basis state의 superposition으로 만드는 명령은 제출 회로에 쓸 수 없다. HMR은 측정과 초기화 과정을 추적하는 방법까지 정해진 별도 명령이다. 전체 양자 알고리즘을 고전 컴퓨터로 쉽게 계산한다는 뜻은 아니다. 알고리즘 안의 특정 subroutine만 이런 형태로 분리했기 때문에 가능한 검증이다.

측정이 없고 scratch가 모두 0으로 돌아오는 phase 회로를 생각해 보자. 모든 basis input에서 동작이 맞다면, 선형성에 따라 superposition에서도 다음과 같이 동작한다.

$$
\sum_x\alpha_x\lvert x\rangle\lvert 0\rangle
\longmapsto
\sum_x\alpha_x e^{2\pi i f(x)}\lvert x\rangle\lvert 0\rangle.
$$

실제 회로에는 근사 오차와 측정 결과에 따른 분기도 있다. [동반 논문][paper]은 입력별 가중치와 각 측정 분기의 구조에 관한 가정 아래, 표본의 실패율을 양자 상태 오차와 연결한다. 따라서 이 검증을 통과했다고 해서 임의의 양자 회로에 대한 formal proof를 얻은 것은 아니다.

### 4.2 shot마다 무엇을 확인할까

입력을 준비하고 회로를 한 번 시뮬레이션하는 단위를 **shot**이라고 한다. 검증기는 shot마다 다음 항목을 확인한다.

| 검사 | 놓치면 생기는 문제 |
|---|---|
| 입출력 register가 요구 조건을 지키는가 | phase만 맞고 x나 syndrome은 바뀌어 있을 수 있다. |
| scratch가 0으로 돌아왔는가 | 중간값에 정보가 남아 interference가 달라질 수 있다. |
| 누적 phase가 허용오차 이내인가 | 비트값은 맞아도 필요한 phase가 틀릴 수 있다. |

`invsqrt`는 모든 x를 64번씩 검사하지만, 가능한 모든 측정 결과를 열거하지는 않는다. DQI는 공식 분포에서 뽑은 오류 패턴을 검사한다. 화학 과제에서는 주소별 ANGLE 출력과 uncomputation 등을 확인하고, 수치 자료를 함께 사용해 전체 비용을 계산한다. 따라서 세 과제에서 PASS로 확인한 범위는 서로 다르다.

### 4.3 회로 해시, 비공개 salt, 로컬 재현

검증에 쓸 난수는 제출 내용의 SHA-256과 salt로 결정한다. `invsqrt`는 canonical binary 회로의 해시를 쓰고, 화학 과제는 factorization과 tree 등의 자료도 반영한다. Kaggle의 Public과 Private에서는 서로 다른 비공개 salt를 사용한다. 참가자가 공개된 몇 가지 입력에만 맞춰 회로를 만드는 것을 어렵게 하려는 장치다. 세부 구현은 [공식 메트릭][metric]에서 확인할 수 있다.

같은 회로와 salt로 다시 실행하면 같은 의사난수 표본이 나온다. 반복 실행만으로 독립적인 검증 사례가 늘어나지는 않는다. 회로를 바꾸면 해시도 바뀌므로 같은 salt에서도 다른 난수열을 쓴다. 조건부 gate의 실행 횟수나 실패율이 측정 결과에 따라 달라진다면, 비용 역시 달라질 수 있다.

이 글의 **전체 로컬 검증**은 기록해 둔 원본·설정·환경에서 공식 과제의 전체 검사량을 끝냈다는 뜻이다. Kaggle에서도 같은 결과가 나온다거나, 가능한 모든 측정 분기를 증명했다는 의미는 아니다. 이후 후보 회로를 비교할 때도 이 구분을 유지한다.


## 5. invsqrt: 거리의 역수를 phase로 표현하기

### 5.1 핵융합에서 왜 제곱근의 역수가 나올까

빠른 하전 입자가 물질을 통과하면 주변 전자와 상호작용하며 운동 에너지를 잃는다.
단위 거리당 잃는 에너지를 **stopping power**라고 한다. 핵융합 연료 안에서 입자가
어디에 얼마나 에너지를 전달하는지 알려면, 뜨겁고 조밀한 물질 속 전자들의 운동을
계산해야 한다. 이 대회에서는 물리 시뮬레이션 전체를 구현하지 않는다. 그 안에서
반복해서 쓰는 작은 산술 회로 하나를 최적화한다. 물리적 배경은
[Rubin 등의 원 논문](https://arxiv.org/abs/2308.12352)에서 출발한다.

두 전자의 위치를 $\mathbf r_i,\mathbf r_j$라 하면 Coulomb 상호작용의 크기는
두 전자 사이 거리에 반비례한다. 단위와 물리 상수를 생략하면 다음과 같다.

$$
V_{ij}\propto \frac{1}{\lVert\mathbf r_i-\mathbf r_j\rVert}
=\frac{1}{\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2}}.
$$

따라서 거리의 제곱을 정수 $x$로 나타내면 계산해야 할 함수는 $1/\sqrt{x}$가 된다.
물리 시스템의 시간 변화를 작은 단계로 나누어 근사할 때, 위치에 따른 potential
energy의 차이는 **phase의 차이**로 반영된다. 일반적인 시간 진화 식은
$e^{-iV\Delta t/\hbar}$이며, 대회에서는 단위·부호·배율을 따로 정해 둔다.

전자 218개에서 고를 수 있는 서로 다른 쌍은 $\binom{218}{2}=23{,}653$개다.
매 시간 단계마다 같은 연산을 수만 번 반복한다면, 한 번 계산할 때 줄인 gate 수가
전체 계산에서는 큰 차이를 만든다. 단순해 보이는 함수 하나가 과학 계산의 병목이
되는 이유다. 이 배경 설명은 [공식 Domain Context](https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/overview/domain-context)에 따른다.

### 5.2 계산 결과를 phase에 반영한다

고전 프로그램이라면 `return 1 / sqrt(x)`라고 쓰고 근삿값을 반환하면 된다.
이 회로가 해야 할 일은 조금 다르다. 입력 숫자는 보존하면서 그 숫자에 해당하는
amplitude에 phase factor를 곱해야 한다. 계산에 사용한 보조 qubit도 마지막에는
모두 0으로 돌아와야 한다.

$$
\lvert x\rangle\lvert 0\cdots0\rangle
\longmapsto
e^{2\pi i/\sqrt{x}}\lvert x\rangle\lvert 0\cdots0\rangle.
$$

`|x⟩`는 22개의 qubit으로 나타낸 정수다. 여러 정수가 superposition 상태로
들어오면 각 항의 phase가 입력값에 따라 달라진다. 예를 들어 4와 16의
superposition이라면 각각 $1/2$바퀴와 $1/4$바퀴만큼 phase가 바뀐다.

$$
\frac{\lvert 4\rangle+\lvert 16\rangle}{\sqrt2}
\longmapsto
\frac{-\lvert 4\rangle+i\lvert 16\rangle}{\sqrt2}.
$$

이 단계 직후에 측정하면 4와 16이 나올 확률은 여전히 같다. 달라진 것은 두 항
사이의 **relative phase**다. 뒤에 이어지는 양자 연산에서 이 차이가 interference로
드러난다. 따라서 출력 비트가 맞더라도 phase가 틀리면 검증을 통과할 수 없다. 입력
하나만 놓고 보면 global phase처럼 보여도, superposition에서는 입력별 phase를
정확히 맞춰야 한다.

baseline 회로는 근삿값을 작업용 register에 계산한 뒤, 각 비트를 이용해 phase를
적용한다. 그런 다음 산술 계산을 되돌려 작업용 register를 처음 상태로 복원한다. 긴 회로도
**compute → phase → uncompute**의 세 단계로 나누어 보면 이해하기 쉽다.

$$
\lvert x\rangle\lvert 0\rangle
\xrightarrow{\text{compute}} \lvert x\rangle\lvert \widetilde f(x)\rangle
\xrightarrow{\text{phase}} e^{2\pi i\widetilde f(x)}\lvert x\rangle\lvert \widetilde f(x)\rangle
\xrightarrow{\text{uncompute}} e^{2\pi i\widetilde f(x)}\lvert x\rangle\lvert 0\rangle.
$$

공식 입출력 조건은 다음과 같다.

| 조건 | 공식 인스턴스 |
|---|---|
| 입력 x | 1…3,133,452 |
| 입력 register | 서로 다른 22개 qubit으로 이루어진 little-endian `x` 하나; 다른 이름의 register는 허용하지 않음 |
| 목표 phase | β=1에서 1/√x turns; 한 바퀴를 주기로 비교 |
| 허용오차 | 2⁻²⁴/√x turns |
| 계산 종료 시 상태 | x를 원래 값으로 복원하고, 나머지 모든 qubit을 0으로 만듦 |
| 검사량 | 모든 x에 64 shots, 합계 200,540,928; 한 번이라도 실패하면 탈락 |

### 5.3 소수점의 위치를 고정하면 숫자도 비트열이 된다

reversible 회로에서는 Python의 부동소수점 연산을 그대로 호출할 수 없다.
baseline 코드는 소수점 위치를 미리 정하는 **fixed-point** 방식을 쓴다. 소수부에
$F$비트를 쓰기로 했다면, 저장된 정수 $Y$를 $Y/2^F$로 해석한다. 예를 들어
소수부가 4비트일 때 `0011`은 정수 3을 저장하지만, 그 값은 $3/16=0.1875$로 읽는다.

비트 수를 늘리면 더 정확해지는 대신, 곱셈에서 처리해야 할 partial product와
보조 qubit도 늘어난다. 그래서 중간값마다 필요한 정밀도를 따로 정한다. 최종 결과에
남기지 않을 낮은 자리의 partial product를 처음부터 생략하는 **column-truncated
multiplication**도 쓴다. 이렇게 계산 중간에 값을 잘라 내면, 정확한 실수 곱셈을
모두 마친 뒤 마지막에 한 번 반올림했을 때와 결과가 달라질 수 있다.

공식 회로 생성 코드의 `Precision`은 중간 단계마다 정밀도를 따로 지정한다. 아래 표에는 최종 phase word의 비트 폭도 함께 정리했다.

| 단계의 값 | 소수부 비트 수 | 하는 일 |
|---|---:|---|
| 다항식 입력 차이 `delta` | 14 | 작은 구간 안에서 입력이 어디에 있는지 나타낸다 |
| 계수와 Horner 중간값, 초기값 $y_0$ | 17 | 적은 비용으로 초기 근삿값을 구한다 |
| $y_0^2$ | 28 | Newton 보정에 쓸 제곱을 계산한다 |
| $My_0^2$, $3-My_0^2$ | 28, 정수부 2 추가 | 보정 배수를 계산한다 |
| 최종 곱 $p$ | 29, 정수부 1 추가 | 보정한 값을 저장한다 |
| phase에 사용할 $Y$ | 총 40 | normalization 배율과 마지막 $1/2$까지 반영한다 |

### 5.4 첫 단계: 큰 입력 범위를 `[1, 4)`로 줄인다

입력 범위는 $1\le x\le3{,}133{,}452$이다. 입력에 따라 결과값의 크기도 크게 달라진다. 이 범위를 그대로 다루려면 작은 결과값에도 필요한 정밀도를 확보하도록 곱셈의 비트 폭을 넉넉히 잡아야 한다. baseline 회로는 먼저 **normalization**으로 계산 범위를 좁힌다. 입력 앞의
`00` 쌍을 세어 $s$에 저장하고, 그 수만큼 두 비트씩 왼쪽으로 shift한다. 이때
register에 저장된 정수와, 소수점 위치를 적용해 읽은 실수 값을 구분해야 한다.

$$
x_{\rm norm}=x\,2^{2s},\qquad
M=\frac{x_{\rm norm}}{2^{20}}\in[1,4),\qquad
\frac1{\sqrt{x}}=\frac{2^{s-10}}{\sqrt M}.
$$

**직접 계산해 보자: $x=64$.** 22비트로 나타내면 유효한 7비트 앞에 0이 15개
붙는다. 따라서 `00` 쌍은 7개이고 $s=7$이다. 왼쪽으로 14비트 shift하면
$x_{\rm norm}=64\times2^{14}=2^{20}$이므로 $M=1$이 된다. 마지막에 배율을
반영하면 $2^{7-10}/\sqrt1=1/8$을 얻는다. 여기서는 normalization의 관계를
손으로 정확히 계산했다. 실제 baseline 회로에서 truncation을 거친 근삿값을 측정한
결과와는 구분해야 한다.

![invsqrt phase 계산과 x=64의 이상적인 배율](/assets/img/posts/quantum-circuit-compilation/fig-03-invsqrt-pipeline.svg){: width="1200" height="820" }

*그림 3. x=64를 s=7, M=1로 normalize한 뒤 배율을 반영하면 목표 phase는 1/8 turn이다. 40비트 출력 word의 이상적인 목표는 $2^{37}$이다. 이 그림은 계산 원리를 보여 준다. 실제 baseline 회로의 하위 비트를 잘라 낸 계수로 이 값이 정확히 나왔다는 뜻은 아니다.*

$s$도 입력에 따라 달라지므로 회로 안에 저장해야 한다. baseline 코드는 이를 위해
10개의 flag qubit을 쓴다. 여기서 7은 일반적인 이진수 `0111`로 저장하지 않는다.
flag 7개를 켜 두는 **unary encoding**을 사용하며, 각 flag가 뒤의 shift를
한 번씩 제어한다.

### 5.5 두 번째 단계: 초기 근삿값을 구한 뒤 한 번 보정한다

Normalization을 마치면 입력이 `[1, 4)` 안에 들어온다. 이를 `[1, 1.5)`,
`[1.5, 2)`, `[2, 3)`, `[3, 4)`의 네 구간으로 나누고, 각 구간에서 사용할 초기
근삿값을 구한다. 상위 비트로 구간을 고른 뒤 해당하는 3차 다항식의 계수를 작업용
register에 넣는다. 뒤의 두 구간에서는 앞의 구간용 계수를 $1/\sqrt2$ 관계에
맞게 조정해 재사용한다.

선택한 구간의 왼쪽 끝을 $a\in\{1,1.5,2,3\}$라 두면 $\delta=M-a$다. 즉, 구간 시작점에서 입력까지의 차이만 계산하면 된다. 실제 회로에서는 δ의 소수부를 14비트까지만 사용한다.

다항식이 $c_0-c_1\delta+c_2\delta^2-c_3\delta^3$이면 다음처럼 괄호를 묶어
계산한다. **Horner's method**를 쓰면 $\delta^2$, $\delta^3$를 따로 계산하지
않고 곱셈 세 번으로 다항식 값을 구할 수 있다.

$$
y_0=c_0-\delta\bigl(c_1-\delta(c_2-c_3\delta)\bigr)
\approx\frac1{\sqrt M}.
$$

초기 근삿값이 처음부터 최종 정밀도를 만족할 필요는 없다. 다음 **Newton update**를
한 번 적용하면 오차를 크게 줄일 수 있다.

$$
y_{\rm next}=\frac{y_0}{2}(3-My_0^2).
$$

한 번만 보정해도 효과가 큰 이유를 식으로 확인해 보자. Truncation과 반올림이
없는 이상적인 계산에서 $y_0=(1+e_0)/\sqrt M$를 대입하면, 보정 후 상대오차는
다음과 같다.

$$
e_1=-\frac32e_0^2-\frac12e_0^3.
$$

처음 상대오차가 $10^{-4}$라면 보정 후 크기는 약 $1.5\times10^{-8}$이다.
오차가 작을 때는 대략 그 제곱에 비례해 줄어든다. 다만 위 식에는 fixed-point
truncation에서 생기는 오차가 빠져 있다. 실제 회로에서는 다항식 근사, 계수를 정해진 비트 수로 표현하는 과정과 각 곱셈의 truncation에서 생기는 오차를 모두 포함해 검사해야 한다.

**공식 코드에서는 배율을 어디에 반영하는지 확인해야 한다.** 회로 생성 코드는 위 식의
$1/2$를 따로 곱하지 않고 $p\approx y_0(3-My_0^2)\approx2/\sqrt M$를 계산한다.
소수부 29비트인 정수 `p`를 $s$만큼 shift한 결과가 $Y$이며, 이를 $2^{40}$으로
나누어 읽는다. 식의 $1/2$를 출력의 소수점 위치에 반영한 셈이다. $x=64$, $M=1$에서
이상적인 실수 목표를 $p_*=2$라 쓰면, 최종 출력 목표는 $Y_*=2^{29}p_*2^7=2^{37}$이며
$Y_*/2^{40}=1/8$이다. 실제 p는 소수부 29비트와 정수부 1비트, 총 30비트이므로
2 미만의 값만 표현할 수 있다. 따라서 이 손계산이 실제 register에 경계값 2를
정확히 저장한다는 뜻은 아니다. 실제로는 truncation을 거친 값이 허용오차 안에
드는지 검증한다.

### 5.6 세 번째 단계: 각 비트에 phase를 적용한 뒤 uncompute한다

$Y=\sum_{j=0}^{39}b_j2^j$이면 목표 phase는
$Y/2^{40}=\sum_jb_j2^{j-40}$바퀴다. 각 비트 $b_j=1$에 대해
$2^{j-40}$바퀴씩 phase를 바꾸면, 이를 모두 더한 값이 목표 phase가 된다. 이렇게
큰 phase 연산 하나를 비트별 작은 phase 연산으로 나눈다. 실제 `Z_POW`의 인자는
반 바퀴 단위이므로 코드에서는 $2^{j-39}$를 전달한다. `Z_POW q t`는
$\lvert 1\rangle$ 성분에 $e^{i\pi t}$를 곱한다. 따라서 일반적으로 $2^{j-F}$바퀴를
적용하려면 $t=2^{j-F+1}$을 써야 한다. Turn과 Z_POW 인자의 단위를 혼동하면
각도가 두 배 어긋난다.

Phase를 적용한 뒤에는 shift, Newton 계산, 다항식, 구간 선택, 입력 normalization을
역순으로 되돌린다. $x$는 원래 정수로 복원되고, $Y$를 비롯한 scratch는 모두
0이 된다. 앞에서 적용한 phase는 그대로 남는다. 중간값을 그냥 버리는 것으로는
이 uncomputation을 대신할 수 없다.

### 5.7 어디에 비용이 드는가

공식 배포 파일 `invsqrt.kmb`에서 각 단계의 **Toffoli count**를 집계하면 아래와
같다. baseline 회로의 비용이 어디서 발생하는지 보여 주는 정적 분석 결과다.

| 단계 | Compute | Uncompute | 주요 내용 |
|---|---:|---:|---|
| Normalization | 120 | 110 | 앞에 붙은 0 판정과 controlled shift |
| 구간 선택 | 3 | 0 | 선택 flag를 지울 때 MBU 사용 |
| Horner 곱셈 3회 | 1,071 | 1,071 | 초기 근사를 만드는 비용 |
| $y_0^2$ | 442 | 442 | 대칭을 이용한 제곱 |
| $M y_0^2$ | 1,043 | 1,043 | 큰 산술 병목 |
| $3-z$ | 28 | 28 | 보정항을 만드는 뺄셈 |
| $y_0(3-z)$ | 867 | 867 | 마지막 보정 곱셈 |
| Denormalization | 300 | 300 | 배율을 복원하는 controlled shift |
| 합계 | 3,874 | 3,861 | 총 7,735 |

![공식 invsqrt baseline 회로의 compute·uncompute 단계별 Toffoli count](/assets/img/posts/quantum-circuit-compilation/fig-04-invsqrt-cost.svg){: width="1152" height="768" }

*그림 4. baseline 회로에는 Toffoli 계열 연산이 7,735개 있다. 개선한 후보를 실행한 결과가 아니라 원본 회로의 정적 분석 결과다. 최종 비용에는 rotation synthesis 비용이 추가된다.*

곱셈·제곱·뺄셈에서 compute와 uncompute를 합쳐 6,902개, 약 89%를 쓴다.
Python 함수 호출 수보다 산술 구조와 정밀도를 먼저 살펴봐야 하는 이유다.
Phase rotation을 합성하는 비용도 추가되며, 기존 로컬 결과에서는 이를 Toffoli로
환산했을 때 약 253개에 해당했다.

Uncompute에 비용의 절반 가까이 든다고 해서 측정만으로 이를 전부 없앨 수 있는
것은 아니다. 임시 AND를 지우는 `del_and`에는 이미 **measurement-based
uncomputation**이 들어 있다. `_add_row`는 양쪽 방향 모두 partial product용
AND를 만든 뒤 덧셈 또는 뺄셈을 하고, 그 AND를 측정해 지운다. 덧셈의 carry와
이전 accumulator 값까지 복원해야 하는 계산을 임시 AND 하나처럼 비용 없이
지울 수는 없다.

### 5.8 정밀도는 통과 조건인 동시에 비용 변수다

모든 유효한 $x$에서 phase 오차가 $2^{-24}/\sqrt{x}$바퀴 이하여야 한다.
입력이 커져 목표 phase가 작아지면 허용 절대오차도 함께 작아진다. 작은 입력 몇
개에서 맞았다는 것만으로는 부족하다. 공식 검사는 모든 입력을 64번씩, 총
200,540,928번 실행하면서 입력 복원, scratch 0, phase를 함께 확인한다.

작은 각도의 phase gate를 fault-tolerant gate로 구현하려면 **rotation synthesis**가
필요하고, 여기에도 오차와 비용이 발생한다. 검증기는 먼저 산술 오차를 확인한 뒤,
허용오차에서 남은 최소 여유를 synthesis 오차에 배정한다. 따라서 비트 하나를 줄여
Toffoli count가 감소하더라도, 산술 오차가 커져 여유가 줄면 rotation synthesis는
더 비싸질 수 있다. 산술 계산에서 허용오차를 모두 써 버리면 phase 검사 자체를
통과해도 synthesis에 남길 여유가 없어 거부된다. **검증을 통과한 회로의 최종 비용**을
비교해야 하는 이유다.

### 5.9 무엇부터 바꾸어 볼까

| 접근 | 무엇이 달라지나 | 먼저 확인할 것 |
|---|---|---|
| 작업용 qubit 재사용 | 동시에 필요하지 않은 값을 같은 logical wire에 저장 | 이전 값이 0으로 돌아오는 시점과 입력 보존 여부 |
| 중간 정밀도 조정 | 필요 없는 하위 partial product와 carry를 생략 | 구간 경계와 큰 $x$에서의 최악오차, rotation synthesis에 남는 여유 |
| 초기 근사 재설계 | 구간 수·다항식 차수·lookup 크기를 조정 | 초기값 계산에서 아낀 비용보다 Newton 계산의 비트 폭 증가로 드는 비용이 크지 않은가 |
| Shift 구조 변경 | unary flag로 제어하는 shift를 다른 구조로 구현 | 제어 bit 계산과 uncompute 비용까지 포함해도 이득인가 |
| 산술 구현 변경 | 곱셈·제곱·덧셈의 reversible 구현을 변경 | carry, overflow, signed/unsigned, truncation 규칙의 일치 여부 |

이 표는 시도해 볼 만한 방향을 정리한 것이며, 아직 개선 효과를 확인한 항목은 없다.
첫 시도에서는 정밀도·다항식·wire 배치를 한꺼번에 바꾸지 말고, 한 가지씩 바꿔
원래 회로와 같은 검증 조건에서 비교하는 편이 좋다. 그래야 비용이 줄거나 검증에
실패했을 때 원인을 찾을 수 있다.

공식 코드는 `nr_reference_circuit.py`에서 `Precision` → `_compute_y1` → `_phase_by_bits` → `build_nr_circuit` 순서로 읽으면 이해하기 쉽다. `_compute_y1` 안의 `_normalize`, `_horner`, `_square`, `_multiply_m`, `_correct`, `_multiply_y0`, `_denormalize`가 앞에서 설명한 단계에 해당한다. `fixed_point_model`은 별도의 수치 모델이다. 이 모델로 계산한 오차와 실제로 생성한 회로의 검증 결과는 구분해서 읽어야 한다.

## 6. DQI: 조합 최적화에 Reed–Solomon decoder가 필요한 이유

세 과제 가운데 `dqi`는 알고리즘이 풀려는 문제와 우리가 최적화할 회로가 가장 달라 보인다. 최종 목표는 조건에 잘 맞는 다항식을 찾는 조합 최적화다. 하지만 이 대회에서 제출하는 것은 다항식을 직접 찾는 프로그램이 아니라, **superposition을 유지하면서 오류 패턴의 흔적을 지우는 quantum subroutine**이다. 둘이 어떻게 연결되는지 알아야 통신 분야의 Reed–Solomon decoder에서 어떤 아이디어를 가져올 수 있는지, 무엇을 새로 설계해야 하는지 판단할 수 있다.

### 6.1 출발점: finite field에서 조건에 잘 맞는 다항식 찾기

보통 곡선을 맞출 때는 관측점에 가까이 지나는 실수 다항식을 찾는다. **Optimal Polynomial Intersection(OPI)**에서는 finite field의 각 평가점마다 허용되는 값의 집합이 주어진다. 계수 벡터 $x$로 다항식 $p_x$를 만들고, 다항식의 값이 허용 집합에 들어가는 평가점의 수를 최대한 늘리는 문제다.

$$
p_x(z)=\sum_{k=0}^{n-1}x_kz^k,
\qquad
f(x)=\left|\{j:p_x(\alpha_j)\in F_j\}\right|.
$$

여기서 $F_j$는 평가점 $\alpha_j$에서 허용되는 값들의 집합이다. 후보 다항식 하나를 받으면 모든 평가점에서 $f(x)$를 계산해 점수를 확인할 수 있다. 하지만 점수를 계산할 수 있다는 것과 좋은 후보를 쉽게 찾을 수 있다는 것은 별개다. DQI는 좋은 후보가 더 자주 측정되도록 확률분포를 만드는 양자 알고리즘이다. 논문에서 말하는 quantum advantage는 특정 문제와 비교 알고리즘, 자원 가정에 따른 주장이다. 대회 참가 자체가 quantum advantage를 입증하는 것은 아니다. [OPI와 DQI 원 논문](https://www.nature.com/articles/s41586-025-09527-5)

DQI가 만들려는 최종 상태를 개념적으로 쓰면 다음과 같다.

$$
\lvert \psi\rangle\propto\sum_x P(f(x))\lvert x\rangle,
\qquad
\Pr[x]\propto|P(f(x))|^2.
$$

$P$는 점수가 좋은 후보에 더 큰 amplitude를 주도록 선택하는 다항식이다. 그렇다면 이 상태를 어떻게 만들 수 있을까? DQI는 Fourier transform을 이용해 원래 문제를 희소한 오류 패턴과 그 **syndrome**을 다루는 문제로 바꾼다. syndrome은 오류 벡터의 정보를 짧게 압축한 대수적 요약이라고 생각하면 된다.

### 6.2 오류를 복원하는 decoder로 register를 비운다

syndrome을 계산하는 선형변환을 $B^T$로 표시하면, 핵심 과정을 다음처럼 쓸 수 있다.

$$
\sum_e a_e\lvert e\rangle\lvert B^Te\rangle
\ \longrightarrow\
\lvert 0\rangle\sum_e a_e\lvert B^Te\rangle.
$$

왼쪽에서는 오류 패턴 $e$와 syndrome $B^Te$가 서로 얽혀 있다. 오른쪽처럼 오류 register를 0으로 되돌려야, 남은 syndrome에 Fourier transform을 적용해 원하는 interference를 얻을 수 있다. 오류 패턴을 측정해서 알아내면 superposition이 달라진다. 대신 syndrome으로부터 오류를 reversible하게 복원하고, 이를 원래 오류 register와 XOR한다. 그러면 $\lvert e\oplus e\rangle=\lvert 0\rangle$가 된다. **이 복원에 error-correcting decoder가 필요하다.**

오류 개수를 제한하면 syndrome만으로 원래 오류를 유일하게 복원할 수 있다. 대회의 Reed–Solomon code는 길이 4,095, 차원 4,025, 최소거리 71이다. 따라서 최대 35개의 **symbol error**를 고칠 수 있다. 35는 뒤집힌 비트 수가 아니다. symbol 하나는 12비트로 표현하며, 0이 아닌 symbol이 최대 35개라는 뜻이다.

실제 DQI는 qubit을 아끼기 위해 오류 symbol을 하나씩 만들고, X basis로 측정해 지우는 방법을 쓴다. **Measurement-based uncomputation(MBU)**이다. 이 방법은 workspace를 줄여 주지만, 측정 결과에 따라 superposition의 각 성분에 부호가 남는다. decoder는 숨겨진 오류 패턴을 복원해 이 부호를 상쇄해야 한다. [Reversible decoder와 MBU 구성](https://arxiv.org/abs/2510.10967)

이번 대회의 decoder는 오류값을 출력하는 대신, 측정 결과에 맞는 phase correction을 적용한다. 다음 두 절에서는 이때 쓰는 field 연산과 실제 입출력 형식을 살펴보자.

### 6.3 GF(2¹²)를 12비트로 표현해도 정수 연산과는 다르다

공식 회로는 원소가 4,096개인 finite field $\mathrm{GF}(2^{12})$를 사용한다. 원소 하나는 12비트로 표현하지만, 정수를 곱한 뒤 4,096으로 나눈 나머지를 구하는 방식과는 다르다.

비트열 $(a_0,\ldots,a_{11})$을 다항식 $a_0+a_1u+\cdots+a_{11}u^{11}$로 읽는다. 덧셈은 각 계수를 mod 2로 더하는 연산, 즉 XOR다. 곱셈은 두 다항식을 곱한 다음 정해진 irreducible polynomial로 나눈 나머지를 구한다. 공식 비트 순서는 little-endian이며, 사용하는 polynomial은 다음과 같다.

$$
u^{12}+u^6+u^4+u+1.
$$

| 연산 | 비트 수준에서 하는 일 | 회로 구현 |
|---|---|---|
| 두 field 원소의 덧셈 | 비트별 XOR | CNOT으로 구현 가능 |
| 고정된 field 상수와의 곱셈 | GF(2) 위의 선형변환 | CNOT network로 구현 가능 |
| qubit으로 표현한 두 field 원소의 곱셈 | 두 입력값 모두에 의존하는 쌍선형 계산 | Toffoli 필요 |
| 0이 아닌 원소의 inverse | $a^{-1}$ 계산 | 여러 번의 곱셈과 임시 공간 필요 |

같은 곱셈이라도 한쪽 값을 compile 시점에 알고 있는지에 따라 비용이 크게 달라진다. CPU에서 빠른 decoder가 양자 회로에서도 저렴하다고 볼 수 없는 첫 번째 이유다.

### 6.4 실제 입출력: 회로에 오류 벡터를 주지는 않는다

오류 벡터를 $e=(e_0,\ldots,e_{4094})$, field의 primitive element를 $\alpha$라고 하자. verifier는 오류 벡터 대신 70개의 syndrome을 입력한다.

$$
S_i=\sum_{j=0}^{4094}e_j\alpha^{(i+1)j},
\qquad i=0,\ldots,69.
$$

여기서 덧셈은 field 연산이므로 비트 표현으로는 XOR다. $S_0,\ldots,S_{69}$에는 각각 12개의 qubit, 총 840개의 qubit이 필요하다. 각 위치에는 12개의 **고전 비트**로 된 측정 mask $M_j$도 주어진다. 공식 register 이름 `E0`…`E4094`에 들어 있는 것이 이 mask다. 이름이 E라고 해서 숨겨진 오류 $e_j$를 담고 있는 것은 아니다.

회로가 해야 하는 일은 다음과 같다.

$$
U_M\lvert S(e)\rangle\lvert 0\rangle_W
=(-1)^{b(e,M)}\lvert S(e)\rangle\lvert 0\rangle_W,
\quad
b(e,M)=\left(\sum_j\operatorname{popcount}(e_j\mathbin{\&}M_j)\right)\bmod2.
$$

`&`는 비트 AND, `popcount`는 1인 비트의 개수다. 이 식은 field 곱셈이 아니라, mask가 선택한 오류 비트들의 **parity**를 계산한다. 선택한 비트 중 켜진 비트의 개수가 홀수이면 −1, 짝수이면 +1을 곱한다. syndrome은 원래 값으로, 추가 qubit은 모두 0으로 돌아와야 한다. 최종 출력에 오류 벡터를 남길 필요는 없다. 따라서 오류값 전체를 보관하지 않고 필요한 phase만 계산하는 설계를 생각할 수 있다.

고정된 basis 입력 하나만 보면 전체 부호를 바꾸는 것처럼 보인다. 하지만 $e$에 따라 부호가 달라지므로, superposition에서는 성분 사이의 relative phase가 된다. 이 차이가 다음 Fourier transform 이후의 측정 확률을 바꾼다.

입력 register는 서로 겹치지 않는 qubit 12개로 된 register `S0`…`S69`와 고전 비트 12개로 된 register `E0`…`E4094`여야 한다. 다른 이름의 register를 추가할 수 없고, `Z_POW`도 쓸 수 없다. 정확한 입출력 조건은 공식 데이터의 `verifiers/dqi/verify_submission_dqi.py`를 따른다.

### 6.5 GF(4) 예제로 입력부터 phase까지 따라가기

다음은 **설명용 RS(3,1)의 1-symbol error 예제**다. 공식 GF(2¹²)·4,095개 symbol 입력으로 실행한 결과가 아니다. 원소를 $\{0,1,\alpha,\alpha+1\}$로 두고 $\alpha^2=\alpha+1$을 사용한다.

| 원소 | 다항식 | 비트 표현(상위→하위) |
|---|---|---|
| 0 | 0 | `00` |
| 1 | 1 | `01` |
| $\alpha$ | $u$ | `10` |
| $\alpha^2$ | $u+1$ | `11` |

$\alpha^3=1$이며 $\alpha+\alpha=0$이다. 위치 1에 값이 $\alpha^2$인 오류 하나만 있다고 하자.

$$
e=(0,\alpha^2,0),\qquad
S_0=\alpha^2\alpha=1,\qquad
S_1=\alpha^2\alpha^2=\alpha.
$$

회로가 받는 입력은 $e$가 아니라 $(S_0,S_1)=(1,\alpha)$다. 오류가 하나라는 조건에서는 $S_1/S_0=\alpha^j$로 위치 $j=1$을 알아낼 수 있다. 이어서 $e_1=S_0/\alpha=\alpha^2$로 오류값을 복원한다. 오류가 여러 개면 이 비율 하나로는 풀 수 없어서, 뒤에서 설명할 다항식 알고리즘이 필요하다.

mask를 $M_1=\alpha$, 즉 `10`으로 정하면 다음과 같다.

$$
e_1\mathbin{\&}M_1=\texttt{11}\mathbin{\&}\texttt{10}=\texttt{10},
\quad \operatorname{popcount}=1,
\quad \text{요구 위상}=-1.
$$

이 예제에서 회로가 해야 할 일은 $\lvert 1,\alpha\rangle\lvert 0\rangle_W\mapsto-\lvert 1,\alpha\rangle\lvert 0\rangle_W$다. $M_1=\texttt{11}$이면 선택한 비트의 parity가 짝수이므로 부호는 +1이다. 오류가 있는 위치만 찾아서는 부족하고, 오류 **값을 이루는 각 비트**까지 알아야 하는 이유다.

![GF4 예제의 오류 복원과 bit mask에 따른 부호 계산](/assets/img/posts/quantum-circuit-compilation/fig-05-dqi-decoding.svg){: width="1200" height="815" }

*그림 5. 설명용 GF(4) 예제. syndrome (1,α)로 위치 1의 오류 α²를 복원한 뒤, mask로 선택한 비트들의 parity에 따라 −1 phase를 적용한다. 공식 크기의 decoder를 실행한 결과는 아니다.*

### 6.6 baseline 회로: EEA → Chien → Forney → uncomputation

오류가 여러 개일 때는 syndrome을 하나의 다항식으로 묶는다.

$$
S(z)=S_0+S_1z+\cdots+S_{69}z^{69}.
$$

오류가 있는 위치의 집합을 $J$라고 하자. Error locator polynomial $\sigma$와 error evaluator polynomial $\Omega$는 다음 관계를 만족한다.

$$
\sigma(z)=\prod_{j\in J}(1-\alpha^jz),
\qquad
\sigma(z)S(z)\equiv\Omega(z)\pmod{z^{70}}.
$$

characteristic 2인 field에서는 −와 +가 같다. $\sigma$의 root로 오류 위치를 찾고, $\Omega$로 해당 위치의 오류값을 복원한다. 위 합동식은 곱셈 결과에서 $z^{70}$ 이상의 항을 버린다는 뜻이다.

| 단계 | 계산할 것 | 공식 회로 생성기의 동작 |
|---|---|---|
| Key equation 풀기 | $\sigma,\Omega$ | Proos–Zalka 방식으로 동기화한 **extended Euclidean algorithm(EEA)**, 기본 222단계 |
| Chien search | 오류가 있는 위치 | 모든 4,095개 위치에서 $\sigma(\alpha^{-j})=0$인지 평가 |
| Forney 계산과 phase 적용 | 오류값과 mask가 정하는 부호 | $\Omega$, $\sigma'$의 inverse, mask로 phase를 적용한 뒤 임시값 정리 |
| Uncomputation | 계산 과정에 남은 정보 정리 | EEA와 초기 정렬을 거꾸로 실행해 원래 syndrome과 0으로 초기화된 workspace 복원 |

다항식을 normalize하는 한 가지 관례에서는 오류 위치의 값이 $e_j=\Omega(\alpha^{-j})/\sigma'(\alpha^{-j})$가 된다. 공식 구현은 EEA 배열의 표현에 필요한 지수 보정을 위치별 상수 곱에 포함한다. 따라서 이 공식을 그대로 옮기기 전에, 입력 syndrome의 지수가 어디서 시작하는지와 다항식의 항을 어떤 순서로 저장하는지를 함께 맞춰야 한다. GF(4) 예제에서는 $\sigma(z)=1+\alpha z$, $\Omega(z)=1$, $\sigma'(z)=\alpha$이고, root $z=\alpha^2$에서 $1/\alpha=\alpha^2$를 복원한다.

공식 생성기의 `forney_step`은 각 위치에서 inverse 계산용 workspace를 만든다. 오류가 있는 위치인지 나타내는 조건과 $\Omega$를 결합한 뒤 `gf2_phase_by_product(..., mask=...)`를 호출한다. 4,095개의 오류값을 배열에 담아 끝까지 보관하지 않는다. 즉, **오류값 전체를 저장하지 않고 phase만 계산하는 아이디어는 baseline 회로에도 이미 일부 들어 있다.** 개선안을 내놓으려면 여기서 어떤 계산이나 저장 공간을 더 줄일 수 있는지 보여 줘야 한다.

### 6.7 CPU용 decoder를 그대로 쓸 수 없는 이유

CPU에서 실행하는 decoder는 배열을 덮어쓰고, 이전 나머지를 버리고, 입력 데이터에 따라 반복문을 끝낼 수 있다. 하지만 양자 회로의 입력은 서로 다른 syndrome의 superposition일 수 있다. 같은 동작을 그대로 옮기면 필요한 정보가 사라지거나, 어느 계산 경로를 거쳤는지에 관한 정보가 남을 수 있다. reversible 회로에서는 이전 값을 복원하는 데 필요한 정보, 분기를 제어하는 연산, 곱셈의 중간값을 지우는 과정까지 고려해야 한다.

따라서 CPU 실행시간만으로 알고리즘을 고를 수는 없다. 이 대회에서는 계산과 uncomputation을 모두 포함한 Toffoli count, qubit 수, 공식 입력 분포에서의 실패율이 중요하다. 예를 들어 workspace를 줄이려고 중간값을 재계산하면 gate 수는 늘어난다. 그래도 비용식의 $Q^2$ 때문에 전체 비용은 낮아질 수 있다.

공식 문헌에 나온 기준값은 **qubit 1,893개, Toffoli 5,725,461개**다. 이 수치는 로컬에서 DQI baseline 회로의 전체 검증을 재현한 결과가 아니다. 이번 세션에서 전체 재현을 마친 과제는 `invsqrt`뿐이다.

### 6.8 실패를 허용하면 그만큼 비용이 늘어난다

공식 verifier는 102,400회의 shot을 실행한다. 각 shot마다 오류 개수 $w\in\{0,\ldots,35\}$, 서로 다른 오류 위치, 0이 아닌 오류값, 측정 mask를 뽑는다. 오류 개수는 균등하게 뽑지 않는다. DQI 상태에 맞춰 36×36 tridiagonal matrix의 최대 eigenvalue에 대응하는 정규화된 eigenvector $v$를 구하고, $\Pr[w]=v_w^2$를 사용한다.

$$
A_{w,w}=\frac{w(q-2r)}{\sqrt{r(q-r)}},\qquad
A_{w-1,w}=A_{w,w-1}=\sqrt{w(m-w+1)},
\quad(q,m,r)=(4096,4095,2016).
$$

오류 개수 $w$가 정해지면, 그 조건에 맞춰 오류 위치와 0이 아닌 값을 균등하게 뽑는다. $w$의 기대값은 약 31.05다. 따라서 오류가 적은 입력만 잘 처리하는 테스트로는 공식 입력 분포에서도 잘 작동한다고 말할 수 없다.

syndrome이 바뀌었거나 workspace에 값이 남았으면 값 검사 실패로 센다. 값 검사는 통과했지만 부호가 틀리면 phase 검사 실패다. 두 경우를 중복해서 세지는 않는다. 두 실패 수를 합친 값을 $k$, shot 수를 $N$이라 할 때, 비용식에는 관측된 비율 $k/N$ 대신 단측 Clopper–Pearson 상한 $\epsilon_+$를 넣는다.

$$
\mathrm{cost}_{\mathrm{DQI}}=R(\epsilon_+)Q^2T,
\qquad
R(\epsilon)=\frac{\ln2}{-\ln(\tfrac12+2\sqrt\epsilon)}.
$$

신뢰수준은 $1-2^{-128}$이고, 통과하려면 $\epsilon_+<1/16$이어야 한다. $T$는 uncomputation까지 포함해 **각 shot에서 실제 실행된** Toffoli 계열 gate 수의 평균이다. qubit 수 $Q$에는 입력에 쓰는 840개의 qubit과 workspace가 포함된다. 소스에 적힌 전체 gate 수와 조건부 실행을 반영한 평균은 구별해야 한다.

실패가 한 번도 없어도 $\epsilon_+=1-(2^{-128})^{1/102400}\approx0.00086606$, $R\approx1.19126$이다. 유한한 표본에서 실패를 관측하지 못했다고 실제 오류 확률이 정확히 0이라고 할 수는 없기 때문이다. 이 숫자는 공식에 값을 넣어 계산한 결과이며, 새 회로를 실행해 얻은 측정값이 아니다.

| 가정한 변화 | 비용비 | 해석 |
|---|---:|---|
| qubit 10% 감소, Toffoli 동일 | $0.9^2=0.81$ | 비용 19% 감소 |
| qubit 10% 감소, Toffoli 20% 증가 | $0.9^2\times1.2=0.972$ | 비용 2.8% 감소 |
| qubit 10% 감소, Toffoli 30% 증가 | $0.9^2\times1.3=1.053$ | 비용 5.3% 증가 |
| Toffoli 20% 감소, 실패 penalty $R$ 30% 증가 | $0.8\times1.3=1.04$ | 비용 4% 증가 |

표는 설계 방향을 비교하기 위해 가정한 계산이다. 실패를 허용해 회로를 짧게 만들었다면, 늘어난 $R$까지 포함해 전체 비용이 낮아졌는지 확인해야 한다.

### 6.9 처음에는 어디부터 바꿔 볼까

| 접근 | 확인할 가설 | 유지해야 할 조건 | 확인할 결과 |
|---|---|---|---|
| qubit 사용 기간과 재사용 | 동시에 필요하지 않은 workspace를 공유할 수 있는가 | syndrome 입력과 각 단계의 초기화·정리 조건 | $Q$ 감소, 추가 계산을 반영한 $Q^2T$ |
| GF 곱셈과 inverse | 같은 field 연산에 필요한 Toffoli와 workspace를 줄일 수 있는가 | irreducible polynomial, bit basis, 0 입력 처리 방식 | 연산과 uncomputation을 합한 비용 |
| EEA 저장과 addressing | 다항식·나머지·pointer를 저장하는 비용을 줄일 수 있는가 | key equation과 reversible 복원에 필요한 정보 | 전체 decoder의 $Q$, $T$ |
| Chien·Forney | 위치별 inverse와 phase 계산을 공유하거나 다른 방식으로 바꿀 수 있는가 | 4,095개 위치와 mask 조건 | 반복 구간의 총비용, workspace가 0으로 돌아오는지 |
| 다른 decoding algorithm | Berlekamp–Massey 등의 reversible 구현이 더 저렴한가 | 같은 syndrome·field·mask·입력 분포 | 완성된 회로의 공식 비용과 실패율 상한 |

처음부터 전부 바꾸면 어디서 잘못됐는지 찾기 어렵다. field 표현인지, 위치 순서인지, key equation인지, phase mask인지, uncomputation인지 따로 확인해야 하기 때문이다. 작은 field의 손계산 예제로 수학부터 확인하고, 연산 하나를 reversible하게 구현한 뒤, 같은 입출력 조건을 유지하며 공식 크기로 확장하는 순서가 좋다. 작은 예제에서 잘 작동했다는 것만으로 공식 검증을 통과했다고 볼 수는 없다.

코드를 읽을 때는 먼저 `build_reference_circuit`의 입력 구성과 verifier의 syndrome 지수 관례를 확인한다. 이어서 `dqi_reference_circuit.py`의 `ProosAndZalkaDecoder`에서 초기화와 EEA의 실행·역실행 순서를 살펴본다. 마지막으로 `forney_step`이 inverse를 계산하고, mask에 따른 phase를 적용한 뒤 중간값을 정리하는 과정을 읽으면 된다.

## 7. dfthcsos: 분자 에너지 계산에 쓰이는 데이터 로딩 줄이기

### 7.1 분자 에너지를 구하는 데 왜 테이블 조회가 병목일까

화학 과제가 궁극적으로 다루는 문제는 **분자의 바닥상태 에너지**를 추정하는 것이다. 전자 수가 정해져 있을 때 가능한 가장 낮은 에너지를 구한다는 뜻이다. 참가자가 분자 전체를 양자 시뮬레이션하거나 에너지를 직접 예측하는 것은 아니다. 에너지 계산에 반복해서 쓰이는 **Hamiltonian의 압축 표현과 데이터 로딩 회로**를 개선하는 것이 이번 과제다.

Hamiltonian H는 전자가 어떻게 움직이고 서로 영향을 주는지를 나타내는 연산자다. H의 eigenvector는 에너지가 정해진 상태를, eigenvalue는 그 상태의 에너지를 나타낸다. 모든 상태를 담은 거대한 행렬을 직접 만드는 대신, orbital 사이의 상호작용을 나타내는 계수로 H를 표현한다.

여기서 **orbital**은 한 전자의 상태를 표현할 때 쓰는 basis function이다. 특정 위치에 있는 작은 상자나 원자 하나를 뜻하지 않는다. 전자도 서로 구별되는 공처럼 하나씩 번호를 붙여 다루지 않는다. 대신 허용된 상태마다 전자가 얼마나 들어 있는지를 나타내는 점유수로 표현한다. 이 과제는 이러한 **second quantization** 표현을 사용한다.

공식 인스턴스 `FeMoCo76o113e`는 질소 고정 효소의 활성 중심인 FeMoCo를 다루는 **76개 spatial orbital, 113개 전자** 문제다. spatial orbital 하나에는 두 spin mode가 대응한다. 따라서 76을 전자 수나 원자 수, 제출 회로의 qubit 수로 이해하면 안 된다. `ANGLE` 회로의 119 qubit과 전체 비용 모델의 730 qubit도 서로 다른 범위의 자원을 센 값이다.

| 표기 | 이 글에서 뜻하는 것 | 대회에서 주어지는 것 |
|---|---|---|
| N | spatial orbital 수 | 76 |
| η | 전자 수 | 113 |
| h⁽¹⁾ₚq | 한 전자 항의 계수 | FCIDUMP에서 읽는 기준 계수와 정해진 변환 |
| h⁽²⁾ₚqrs | 두 전자 상호작용의 계수 | orbital 번호 네 개로 인덱싱하는 텐서 |
| H | 전체 전자계의 Hamiltonian | 위 계수들이 정의하는 연산자 |

한 전자 항은 전자 하나의 운동과 외부 환경의 영향을, 두 전자 항은 전자 사이의 상호작용을 나타낸다고 생각하면 된다. 전자 생성·소멸 연산자의 순서와 계수를 정의하는 방식은 여러 가지다. 공식 코드도 정해진 정의에 맞추어 계수를 변환한다. 처음에는 **두 전자 계수에 p,q,r,s라는 네 개의 인덱스가 붙는다는 점**부터 이해하면 충분하다.

각 인덱스가 N개의 값을 가지면 테이블 크기는 단순하게 세어도 N⁴에 비례한다. N=76이면 76⁴=33,362,176개다. 대칭성을 이용해 중복을 줄일 수는 있다. 그래도 모든 계수를 양자 회로의 개별 분기로 구현하기에는 비용이 크다. 그래서 먼저 고전 컴퓨터에서 계수의 구조를 압축하고, 양자 회로에서는 그 압축된 표현을 읽어 쓰는 방식을 택한다.

### 7.2 계수 압축과 회로 비용을 함께 살펴보기

간단한 행렬부터 보자.

$$
A=\begin{pmatrix}1&2\\2&4\end{pmatrix}
=\begin{pmatrix}1\\2\end{pmatrix}
\begin{pmatrix}1&2\end{pmatrix}
=5uu^{\mathsf T},\qquad
u=\frac{1}{\sqrt5}\begin{pmatrix}1\\2\end{pmatrix}.
$$

네 원소를 각각 저장하지 않아도 벡터 u와 가중치 5만 있으면 행렬 전체를 복원할 수 있다. 실제 계수 텐서는 훨씬 크고, 이 예처럼 작은 rank로 정확히 표현되지 않을 수 있다. 그래도 비슷한 형태의 항을 몇 개 더하면 허용 오차 안에서 원래 계수를 근사할 수 있다. **factorization**은 이런 구조를 찾는 작업이다. 위 식은 DFTHC의 실제 최적화 결과가 아니라, factorization이 무엇인지 보여 주기 위한 2×2 예다.

공식 `dfthcsos`에서는 **double factorization과 tensor hypercontraction을 연결한 DFTHC 구조**를 사용한다. 참가자는 그 구조 안에서 크기와 수치를 바꿀 수 있다. 검증기는 다음 식으로 계수를 재구성한다.

$$
X^{(\rho c)}_{pq}
=\sum_{b=1}^{B}w_b^{(\rho c)}u_{b,p}^{(\rho)}u_{b,q}^{(\rho)},
\qquad \|u_b^{(\rho)}\|_2=1.
$$

이어서 X의 outer product를 합쳐 네 인덱스 텐서를 복원한다.

$$
G_{pqrs}=\sum_{\rho=1}^{R}\sum_{c=1}^{C}
X^{(\rho c)}_{pq}X^{(\rho c)}_{rs}.
$$

기준 두 전자 적분과 비교하기 전에는 G에서 **공식 기준에 고정된 symmetric shift를 되돌려야 한다.** 따라서 위 식의 G를 원본 적분과 곧바로 같은 값으로 생각하면 안 된다. 벡터 u, 가중치 w, identity offset W, 고정 shift는 각각 역할이 다르다.

| 바꿀 수 있는 선택 | 뜻 | 값을 줄이거나 바꿀 때 고려할 점 |
|---|---|---|
| R | 큰 factorization 성분의 수 | 성분 수를 줄이면 원래 계수를 충분히 표현하지 못할 수 있다 |
| B | 성분별 한 입자 벡터의 수 | 로딩 주소는 줄지만 재구성 오차가 늘 수 있다 |
| C | 성분별 계수 사본의 수 | state preparation 비용과 계수를 조절할 자유도가 함께 바뀐다 |
| u, w, W | 실제 연속 수치 | 같은 오차에서도 normalization과 회로 비용이 달라진다 |

세 과제 중 **수치 최적화 경험을 가장 직접적으로 활용할 수 있는 부분**이다. 다만 재구성 오차만 작게 만드는 것이 목표는 아니다. 같은 정확도라면 주소가 적고, state preparation이 간단하며, normalization에 따른 비용이 낮은 표현이 유리하다. factorization을 고를 때부터 최종 회로 비용까지 함께 확인해야 하는 이유다.

### 7.3 block encoding: Hamiltonian을 gate처럼 실행하려면

양자 회로의 가역 연산은 unitary여야 한다. 일반적인 Hamiltonian H는 그렇지 않다. 어떤 상태에 H를 곱하면 그 상태의 길이가 달라질 수 있다. 따라서 H를 그대로 gate 하나처럼 실행할 수는 없다.

**block encoding**은 더 큰 unitary U의 일부 block에 정규화한 연산자를 넣는 방법이다. 보조 register를 0으로 준비했을 때 다음 관계를 만족하도록 U를 구성한다.

$$
(\langle0\rvert _{\mathrm{anc}}\otimes I)\,
U\,
(\lvert 0\rangle_{\mathrm{anc}}\otimes I)
=\frac{\widetilde H}{\alpha}.
$$

H̃는 사용할 알고리즘에 맞추어 상수항을 옮기는 등의 처리를 거친 Hamiltonian이다. α는 H̃/α가 unitary의 일부 block으로 들어갈 수 있도록 정하는 normalization factor다. 이 식대로 회로를 실행한다고 에너지가 한 번에 출력되는 것은 아니다. 에너지 추정 알고리즘이 U와 그 역연산을 반복해서 호출할 수 있도록 H를 구현한 것이다.

데이터 조회 비용이 중요한 이유도 이 반복 호출에 있다. 조회 한 번의 비용을 조금만 줄여도 전체 에너지 추정 과정에서는 절약 효과가 누적된다. normalization factor가 커지면 같은 정밀도로 에너지를 추정하는 데 더 많은 호출이 필요하다. 그래서 대회는 **block encoding 한 번의 비용 × normalization에 따라 정해지는 배수**를 함께 평가한다. 이 값을 실제 장비의 초 단위 실행시간으로 바꾸려면 별도의 분석이 필요하다.

DFTHC의 **sum of squares(SOS)**는 energy shift를 적용한 근사 Hamiltonian $H_{\mathrm{DFTHC}}-E_{\mathrm{SOS}}I$의 구조를 말한다. 고정된 전자 수를 다루는 공간에서 이 연산자가 positive semidefinite가 되도록 한다. 원래 분자의 음수 에너지가 모두 양수가 된다는 뜻은 아니다. 이러한 SOS 구조와 에너지 기준을 이용해 spectrum amplification으로 비용을 줄인다. 처음에는 세부 회로를 모두 유도하기보다, **근사 오차가 같아도 factorization이 만드는 normalization에 따라 전체 알고리즘 비용이 달라진다**는 점을 이해하면 된다.

공식 검증기가 실제 다전자 Hamiltonian을 대각화해 바닥상태를 구하는 것은 아니다. 고정된 에너지 상한 E_upper와 factorization에서 계산한 E_SOS로 다음 값을 구한다.

$$
E_{\mathrm{gap}}=E_{\mathrm{upper}}-E_{\mathrm{SOS}},\qquad
\lambda_{\mathrm{eff}}
=\sqrt{E_{\mathrm{gap}}(2\Lambda-E_{\mathrm{gap}})}.
$$

**LCU(linear combination of unitaries)**는 연산자를 여러 unitary의 선형 결합으로 표현하는 방식이다. 그 계수의 크기에서 얻는 raw normalization factor가 Λ다. 위 식은 raw Λ와 SOS 기준의 차이로 최종 비용에 곱할 λ_eff를 계산한다. `E_gap`은 여기서 쓰는 **SOS 기준과의 차이**다. 바닥상태와 첫 들뜬상태 사이의 물리적인 excitation gap을 가리키지 않는다. `lambda_eff` 역시 원본 분자의 에너지나 오차가 아니라, 점수에 곱하는 effective normalization factor다.

![factorization과 ANGLE 조회가 전체 block encoding 비용에 반영되는 구조](/assets/img/posts/quantum-circuit-compilation/fig-06-chemistry-block-encoding.svg){: width="1200" height="850" }

*그림 6. H를 더 큰 unitary의 일부 block에 넣는다. factorization과 회전 트리는 ANGLE 조회 비용에 영향을 준다. 전체 점수에는 state preparation과 Givens rotation 비용도 따로 포함된다. ANGLE 회로 하나의 qubit 수와 전체 block에서 쓰는 qubit 수를 구분해야 한다.*

### 7.4 QROM: 주소가 superposition일 때 표를 읽는 방법

일반적인 Python 코드에서 `table[a]`는 a번째 행을 읽는다. 양자 알고리즘에서는 주소 a가 superposition일 수 있다. 주소를 측정해 하나를 고르면 그 superposition이 사라진다. 따라서 회로는 주소를 보존하면서, 각 주소에 해당하는 데이터를 **XOR 연산으로 가역적으로 읽어야 한다.**

$$
U_d:\quad \lvert a\rangle\lvert z\rangle
\longmapsto \lvert a\rangle\lvert z\oplus d_a\rangle.
$$

이것이 **QROM(quantum read-only memory)** 조회다. 별도의 대형 양자 메모리 장치를 쓴다는 뜻은 아니다. 고전 컴퓨터에서 이미 알고 있는 표를 gate 패턴으로 구현한다. 예를 들어 d₀=01, d₁=10인 작은 표를 읽으면, 선형성에 따라 다음과 같이 동작한다.

$$
\frac{\lvert 0\rangle+\lvert 1\rangle}{\sqrt2}\lvert 00\rangle
\longmapsto
\frac{\lvert 0\rangle\lvert 01\rangle+\lvert 1\rangle\lvert 10\rangle}{\sqrt2}.
$$

오른쪽 상태에서는 각 주소에 그 주소의 데이터가 연결되어 있다. 값을 사용한 뒤 같은 XOR 조회를 다시 적용하면 데이터 register를 0으로 돌려놓을 수 있다. 대회에서는 measurement-based uncomputation도 허용하지만, 이때는 phase correction까지 정확해야 한다.

양자 회로에서 표를 조회하려면 주소를 선택하는 가역 논리와, 데이터를 출력 register에 기록하는 gate가 필요하다. 배열 인덱싱 한 줄에도 회로 비용이 드는 셈이다. 주소가 많고 출력 word가 길수록 분기 구조와 workspace를 어떻게 구성하느냐가 중요해진다.

### 7.5 벡터 대신 회전 각도 표를 읽는 이유

벡터 u의 성분을 양자 amplitude로 직접 써 넣는 일도 간단하지 않다. 공식 방식은 u를 만드는 **Givens rotation의 각도**를 읽어 순서대로 적용한다. Givens rotation은 벡터의 두 성분만 섞는 평면 회전이다. 검증기는 다음 2×2 연산을 이용해 각도 표가 나타내는 벡터를 고전 컴퓨터에서 복원한다.

$$
\begin{pmatrix}u_p'\\u_q'\end{pmatrix}
=\begin{pmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{pmatrix}
\begin{pmatrix}u_p\\u_q\end{pmatrix}.
$$

성분이 3개인 작은 예를 보자. u=(1,0,0)에서 시작해 트리의 간선 0→1과 0→2에 차례로 π/4 회전을 적용한다.

$$
(1,0,0)
\xrightarrow{(0,1),\,\pi/4}
(1/\sqrt2,1/\sqrt2,0)
\xrightarrow{(0,2),\,\pi/4}
(1/2,1/\sqrt2,1/2).
$$

각 단계에서 길이가 보존되므로 최종 벡터도 단위 벡터다. 검증기의 벡터 복원 과정을 설명하기 위한 예이며, 세 orbital 분자의 실제 전자 상태를 계산한 결과는 아니다. **같은 각도라도 회전 트리의 순서에 따라 다른 벡터가 나올 수 있다.** 따라서 트리도 제출 자료에 포함해야 한다.

N=76에서는 벡터 하나를 표현하는 데 트리 간선 75개와 각도 75개가 필요하다. 각도는 실수 그대로 저장하지 않고 b_rot 비트 정수 word로 표현한다.

$$
\theta=\pi\frac{\mathrm{word}}{2^{b_{\mathrm{rot}}-1}},
\qquad 0\leq\mathrm{word}<2^{b_{\mathrm{rot}}}.
$$

즉 [0,2π) 범위를 유한한 격자로 나누는 것이다. 12비트이면 격자 간격은 2π/4096라디안이다. 비트 수를 줄이면 각도 register와 회전 비용을 아낄 수 있지만, 복원한 벡터와 계수 텐서는 원래 값에서 더 멀어질 수 있다. 높은 정밀도에서 잘 맞춘 factorization도 각도를 반올림한 뒤에는 검증을 통과하지 못할 수 있다. **검증기는 회로가 실제로 읽어 낸, quantization을 거친 각도를 평가한다.**

![superposition 주소의 QROM 조회와 두 Givens rotation으로 단위 벡터를 만드는 예](/assets/img/posts/quantum-circuit-compilation/fig-07-qrom-givens.svg){: width="1200" height="880" }

*그림 7. 위쪽은 주소를 측정하지 않고 각 주소에 맞는 데이터를 읽는 QROM이다. 아래쪽은 두 성분씩 회전시켜 단위 벡터를 만드는 예다. 두 예 모두 설명을 위한 것이며 FeMoCo의 수치 결과가 아니다.*

### 7.6 각도 75개를 한꺼번에 저장하지 않아도 된다

각도를 J개씩 묶어 읽으면 같은 word register를 재사용할 수 있다. 기준 설정은 b_rot=12, J=8이다. 데이터 word에만 96 qubit을 쓰고, 각도 75개를 배치 10개로 나누어 전달한다. 벡터 주소는 N+RB=76+65×60=3,976개이므로 주소 register에는 12 qubit이 필요하다.

회로 `ANGLE_k`가 배치마다 기존 값을 0으로 지우고 새 값을 쓰는 것은 아니다. 공식 회로 규격은 **이웃한 배치의 XOR 차이**를 읽도록 정한다. 한 주소에 해당하는 배치를 B₀,B₁,…라고 쓰면 다음과 같다.

$$
\delta_0=B_0,\qquad \delta_k=B_k\oplus B_{k-1}.
$$

word가 B₀일 때 δ₁을 XOR하면 B₁이 된다. 덕분에 같은 register를 다음 배치에서도 이어 쓸 수 있다. 마지막 배치는 실제 각도 3개와 0 패딩 5개로 8칸을 채운다. 역순 조회에서는 이 과정을 거꾸로 실행해 register를 정리해야 한다. J를 키우면 조회 횟수는 줄지만, 동시에 저장해야 할 각도 word가 늘어 128 qubit 제한에 가까워진다.

### 7.7 제출 파일과 검증기가 실제로 사용하는 값

| 부분 | 참가자가 제공하는 것 | 검증 방식 |
|---|---|---|
| factorization 수치 | R,B,C를 정하는 배열 shape와 가중치 | 기준 적분과의 재구성 오차 및 normalization 계산 |
| 회전 표현 | `tree_edges`, word 정밀도와 배치 크기 | 회로가 읽은 각도로 단위 벡터 복원 |
| 각도 로딩 | `angle_k_0`, …, `angle_k_(K−1)` 회로 | 모든 주소에서 정·역방향을 64 shots씩 검사 |
| 전체의 나머지 | 전체 회로를 직접 제출하지는 않음 | state preparation과 Givens rotation을 공식 비용식으로 산정 |

`submission.npz`의 `U`는 배열 shape를 선언하는 데 쓰인다. **실제 벡터는 ANGLE 회로가 읽어 낸 각도에서 복원한다.** U 배열에 좋은 벡터를 넣어도, 회로에서 읽는 각도가 다르면 그 각도에 해당하는 벡터로 평가받는다. 제출한 `H1_I`와 `Hsym_lower` 값은 사용하지 않고, 공식 기준의 symmetric shift를 적용한다.

각 ANGLE 회로는 주소를 보존하고 scratch qubit을 0으로 돌려놓으며, 원하지 않은 ±1 phase를 남기지 않아야 한다. 반복 shot에서 서로 다른 word가 나오거나 패딩·역조회·phase 검사를 통과하지 못하면 거부된다. 실제 Givens rotation 자체를 `Z_POW`로 제출하는 과제도 아니다. **제출 회로는 각도를 이진 word로 읽는 Boolean oracle**이어야 하며, `Z_POW`는 금지된다.

### 7.8 전체 비용에서 어디를 줄일 수 있을까

공식 과제 비용은 다음과 같다. 여기서 T는 T gate 수가 아니라 **Toffoli 수**다.

$$
C_{\mathrm{chem}}=\lambda_{\mathrm{eff}}\,T_{\mathrm{total}}\,Q_{\mathrm{total}}.
$$

$$
T_{\mathrm{total}}=
T_{\mathrm{ANGLE,fwd+rev}}+T_{\mathrm{outer}}+T_{\mathrm{inner}}
+2(N-1)(2b_{\mathrm{rot}}-1).
$$

$$
Q_{\mathrm{total}}=
Q_{\mathrm{outer,persist}}+Q_{\mathrm{inner,persist}}+Jb_{\mathrm{rot}}
+\max(Q_{\mathrm{outer,temp}},Q_{\mathrm{inner,temp}},Q_{\mathrm{ANGLE,temp}}).
$$

state preparation과 각도 조회는 순차 실행하므로 temporary workspace를 공유할 수 있다. 따라서 필요한 임시 공간은 각 단계의 합이 아니라 **최댓값**으로 계산한다. 반면 각도 word는 회전을 적용하는 동안 유지해야 하므로 persistent workspace로 센다. ANGLE 회로 하나의 qubit 수를 줄여도 전체 Q가 같은 폭으로 줄어들지는 않을 수 있다.

| 공식 README의 기준값 — 로컬 실행 측정 아님 | 값 |
|---|---:|
| R,B,C / b_rot / J / 배치 수 K | 65,60,60 / 12 / 8 / 10 |
| ANGLE 회로 하나 | 119 qubit, 3,974 Toffoli |
| ANGLE 전체 정·역 조회 | 79,480 Toffoli |
| outer state preparation / inner state preparation / Givens | 846 / 9,461 / 3,450 Toffoli |
| 전체 T_total | 93,237 |
| 전체 Q_total | 730 = persistent 186 + 공유 temporary 544 |
| raw Λ / E_gap | 약 308.45594 / 3.96548 Ha |
| λ_eff / 과제 비용 | 약 49.30137 / 3.3556×10⁹ |

기준 설정에서는 ANGLE 조회가 전체 Toffoli의 약 85%를 차지한다. 반면 전체 qubit 중 상당수는 state preparation에 필요한 공유 temporary workspace 544개다. 조회 회로의 Toffoli를 줄이는 방법과, factorization 크기·정밀도를 바꾸어 state preparation 공간을 줄이는 방법은 효과가 나는 지점이 다르다. 조회에 scratch qubit을 더 쓰더라도 공유 공간 544개를 넘지 않고 ANGLE의 128개 제한을 지킨다면, 전체 Q는 늘지 않을 수 있다. 실제로 가능한지는 회로 배치와 공식 비용 모델로 확인해야 한다.

검증 통과 조건도 확인해야 한다. `error_1 ≤ 20 mHa`, `error_2 ≤ 20 mHa`, 각 ANGLE 회로 `≤128` qubit, `Λ>49.0`, `E_gap>3.914898 Ha`, 그리고 `0<E_gap<2Λ`를 만족해야 한다. Λ와 E_gap의 하한에서는 **등호가 허용되지 않는다.** 두 오차는 각각 한 전자 계수와 두 전자 계수 텐서의 재구성을 평가하는 Frobenius 잔차다. **이 검사를 통과했다고 실제 다전자 바닥상태 에너지 오차가 20 mHa 이하로 보장되는 것은 아니다.**

### 7.9 처음 개선한다면 어떤 순서로 접근할까

| 접근 | 먼저 고정할 것 | 바꾸는 것 | 통과 후 비교할 것 |
|---|---|---|---|
| 조회 회로 개선 | factorization·트리·각도 word | 주소 분기, 공유 논리, QROM 구성 | 같은 word를 읽을 때의 ANGLE Toffoli와 temp |
| 배치 크기 탐색 | factorization·트리·정밀도 | J와 배치 수 | 128 qubit 제한, 전체 T와 Q |
| 회전 트리·반올림 개선 | 고정밀 factorization | 트리와 각도 quantization | 복원 후 error_1·error_2·λ_eff |
| factorization 자체 개선 | 공식 적분·shift·최종 평가식 | R,B,C와 u,w,W | 오차 제약 아래 λ_eff×T_total×Q_total |

처음에는 기준 각도 표를 유지한 채 QROM 생성기부터 살펴보면 문제를 나누어 이해하기 쉽다. 각도 표까지 동시에 바꾸면 조회 회로의 오류인지, factorization의 근사 오차인지, 반올림의 영향인지 구별하기 어려워진다. 수치 최적화에 익숙하다면 고정밀 factorization부터 개선해도 된다. 이 경우에도 최종 회로에서 사용할 정밀도로 바꾼 뒤 재구성 오차를 다시 확인해야 한다.

scratch를 더 쓰면서 여러 주소를 묶어 처리하는 QROM, 반복되는 출력 비트의 논리를 공유하는 방법, 배치·트리·반올림 방식을 함께 고르는 방법 등을 연구할 수 있다. 어느 방법이 유리한지는 공식 제한을 통과한 회로끼리 실제 비용을 비교해야 알 수 있다. 여기서 설명한 예시와 접근 순서는 연구의 출발점이며, 측정된 개선 결과는 아니다.

코드는 `SCORING_README.md`의 worked example → `example_submission/angle_k_reference_circuit.py`의 배치별 XOR 표 → `verify.py`의 `evaluate_words` → `scoreinfra/dfthc_math.py`의 벡터·계수 복원 순으로 읽으면 좋다. 그러면 회로가 word를 정확히 읽는지 확인하는 과정과, 그 word가 Hamiltonian을 얼마나 잘 근사하는지 평가하는 과정을 나누어 볼 수 있다.

## 8. 회로 비용은 낮을수록, 리더보드 점수는 높을수록 좋다

세 과제의 비용식을 한곳에 모으면 다음과 같다.

| 과제 | 낮출 비용 | 비교에서 함께 고정할 것 |
|---|---|---|
| invsqrt | Q×C, C=Toffoli+T gate/4 | 전체 x, phase 오차, synthesis에 남겨 둘 오차 여유 |
| dqi | R(ε₊)×Q²×T | finite field, syndrome, mask, 공식 분포 |
| dfthcsos | λ_eff×T_total×Q_total | 재구성 오차와 전체 block encoding 자원 |

뒤의 두 과제에서 T는 T gate가 아니라 Toffoli 수다. invsqrt의 임의 각도 회전 한 개에는 synthesis 오차 ε에 따라 $0.56\log_2(1/\epsilon)+5.3$개의 T gate 비용을 부과한다. C는 이 synthesis 비용까지 포함한 값이다.

각 과제의 비용은 낮을수록 좋다. 최종 리더보드 점수는 각 과제에서 baseline 비용을 제출 회로의 비용으로 나눈 뒤, 세 비율의 기하평균을 낸 값이다. 따라서 높을수록 좋다.

$$
\mathrm{Score}
=\left(\prod_{b=1}^{3}
\frac{\mathrm{cost}^{\mathrm{ref}}_b}{\mathrm{cost}_b}
\right)^{1/3}.
$$

| 가상의 변화 | 세 과제의 개선 배수 | 전체 점수 |
|---|---|---:|
| 기준 제출과 같음 | 1, 1, 1 | 1.000 |
| 한 과제의 비용만 절반 | 2, 1, 1 | 약 1.260 |
| 한 과제의 비용만 1/4 | 4, 1, 1 | 약 1.587 |
| 세 과제 모두 비용 절반 | 2, 2, 2 | 2.000 |

이 표는 점수식을 설명하기 위한 예제이며 실제 제출 결과는 아니다. 한 과제부터 개선해도 전체 점수를 높일 수 있지만, 제출 패키지에는 세 과제가 모두 있어야 한다. 아직 바꾸지 않은 과제는 공식 baseline 파일을 그대로 쓸 수 있다.

JSON의 필드 이름에도 주의해야 한다. `invsqrt` 검증기가 쓰는 `score`는 **해당 과제의 비용 Q×C**이므로 작을수록 좋다. 세 과제를 합친 Kaggle 리더보드 점수와는 다른 값이다.

## 9. 처음 받을 파일과 설치할 환경

공식 데이터에는 `train.csv`, `test.csv` 같은 학습·평가 데이터가 없다. 처음 내려받은 배포본에는 34개 파일이 있었고, 압축을 푼 원본 크기는 합계 약 1.04 GB였다. 주요 파일은 다음과 같다.

```text
official/
├── sample_submission/
│   ├── invsqrt.kmb
│   ├── dqi.kmb
│   └── dfthcsos/
│       ├── angle_k_0.kmb ... angle_k_9.kmb
│       └── submission.npz
├── verifiers/
│   ├── invsqrt/
│   ├── dqi/
│   └── dfthcsos/
├── kickmix/
│   └── kickmix.so
└── verifiable_circuit_compilation.pdf
```

`.kmx`는 텍스트 회로 파일이고 `.kmb`는 binary 회로 파일이다. 둘 다 제출할 수 있다. Python의 `kickmix.Circuit`과 `CircuitBuilder`로 읽고 생성한다. 숫자 배열과 산술 연산이 gate로 바뀌는 과정을 이해하려면, 처음부터 회로 파일을 직접 고치기보다 **baseline 생성 코드와 builder API**를 같이 살펴보는 편이 좋다.

### Kaggle에서 시작하는 경로

설치부터 시작하기 부담스럽다면 공식 [starter notebook][starter-notebook]과 [데이터][data]를 먼저 살펴보자. Kaggle 배포본의 `kickmix/`를 Python 검색 경로에 추가하면 제공 라이브러리를 불러올 수 있다. 데이터가 연결된 실제 경로는 notebook의 Input 패널에서 확인한다. OS나 Python ABI가 다르면 같은 `.so`를 로컬에 복사해도 동작하지 않는다.

2026년 10월 23일 데이터 안내에는 `kickmix`를 PyPI에 배포할 예정이라고 되어 있었다. 따라서 아직 `pip install kickmix`만으로 어디서나 설치할 수 있다고 안내하기는 어렵다. 대회가 진행되면서 패키지가 바뀔 수 있으니, 나중에 시작한다면 공식 데이터와 메트릭의 최신 버전을 다시 확인해야 한다.

### 로컬에서 사용한 경로

이번에는 macOS arm64에서 Python 3.12.13, NumPy 2.2.6, Apple C++20 toolchain을 사용했다. `kickmix`는 공식 Qualtran 저장소의 다음 commit에서 **독립된 하위 패키지만** 빌드했다.

```text
638777fac4379bc62e2a1e4a52cf1a1753895d48
```

같은 구성으로 시작하려면 아래 명령을 참고할 수 있다. `uv`, Git, 해당 플랫폼의 C++20 컴파일러가 설치되어 있다고 가정한다. 패키지와 버전은 이번 재현에 사용한 것과 같지만, 아래 명령을 새 디렉터리에 그대로 붙여 넣는 과정을 별도 환경에서 다시 시험하지는 않았다.

```sh
mkdir quantum-circuit-start
cd quantum-circuit-start
uv venv --python 3.12.13
uv pip install --python .venv/bin/python 'numpy==2.2.6' \
  'kickmix @ git+https://github.com/quantumlib/Qualtran.git@638777fac4379bc62e2a1e4a52cf1a1753895d48#subdirectory=kickmix'
.venv/bin/python -I -c 'import numpy, kickmix; print(numpy.__version__); print(kickmix.__file__)'
```

이후 명령도 모두 `quantum-circuit-start/`에서 실행한다. 다운로드에는 [Kaggle CLI][kaggle-cli]의 인증 설정과 대회 규칙 동의가 필요하다. API 토큰은 글이나 저장소에 넣지 않는다. CLI가 이미 준비되어 있다면 다음처럼 원본 데이터와 별도의 메트릭 노트북을 받을 수 있다.

```sh
mkdir -p downloads official metric runs
kaggle competitions download google-verifiable-quantum-circuit-compilation -p downloads
.venv/bin/python -m zipfile -e \
  downloads/google-verifiable-quantum-circuit-compilation.zip official
kaggle kernels pull metric/google-quantum-ai-metric -p metric
```

공식 baseline 회로와 검증기는 원본 그대로 보관한다. 후보 회로는 별도 디렉터리에 만들고, 원본·후보·검증기 파일의 SHA-256을 기록해 두자. 메트릭 notebook은 과제별 검증 외에도 회로를 읽을 수 있는지, 허용 gate와 register 규칙을 지키는지, 제출 파일 구조가 맞는지 검사한다. **과제별 검증 스크립트 하나에서 PASS가 나왔다고 전체 제출 검사가 끝난 것은 아니다.**

로컬에서 빌드한 라이브러리와 Kaggle이 제공하는 Linux binary가 같은 구현인지는 아직 확인하지 못했다. 제공된 binary를 소스 코드의 어느 commit에서 빌드했는지도 알려져 있지 않다. 아래 결과는 공식 검증기 원본과 기록한 로컬 라이브러리로 얻은 결과다. Kaggle 환경에서도 똑같이 동작한다는 증거로 볼 수는 없다.

## 10. invsqrt baseline 회로를 끝까지 실행하기

처음에는 입력 일부만 검사해서 파일 경로와 라이브러리 로딩이 맞는지 확인한다. 아래 smoke 명령은 **22비트 원본 회로를 그대로 두고 x의 검사 범위만 줄인다.** 실행마다 새 디렉터리를 만들고, 같은 이름을 재사용하지 않는다.

```sh
mkdir runs/invsqrt-smoke-001 && \
.venv/bin/python -I official/verifiers/invsqrt/verify_submission_invsqrt.py \
  official/sample_submission/invsqrt.kmb \
  --n-p 10 --beta 1 --precision-bits 24 --eta 218 \
  --max-x 64 --shots-per-x 64 \
  --batch-size 256 --max-workers 1 \
  --salt quantum-circuit-local-v1 \
  > runs/invsqrt-smoke-001/result.json \
  2> runs/invsqrt-smoke-001/stderr.log
```

총 4,096 shots를 실행한다. 설치와 입출력 형식을 확인하는 단계이며, 대회 전체 조건의 정확성이나 최종 비용을 재현한 결과는 아니다. 특히 rotation synthesis 비용은 검사한 범위에서 남은 최소 오차 여유에 따라 달라진다.

전체 검증에서는 x의 상한을 3,133,452로 되돌린다. 아래 인자는 실제로 완료한 로컬 실행과 같다. 파일 경로만 이 글의 입문용 폴더 구조에 맞췄다.

```sh
mkdir runs/invsqrt-full-001 && \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -I official/verifiers/invsqrt/verify_submission_invsqrt.py \
  official/sample_submission/invsqrt.kmb \
  --n-p 10 --beta 1 --precision-bits 24 --eta 218 \
  --max-x 3133452 --shots-per-x 64 \
  --batch-size 256 --max-workers 4 \
  --salt quantum-circuit-local-v1 \
  > runs/invsqrt-full-001/result.json \
  2> runs/invsqrt-full-001/stderr.log
```

직접 실행할 때는 timeout과 시스템 자원을 함께 관리해야 한다. 위 명령은 공식 스크립트를 직접 실행하므로 자체 timeout이 없다. 이번 로컬 harness에서는 부모 프로세스가 최대 1,800초로 제한하고, 시작·종료 시각과 로그·파일 해시를 실행 기록에 남겼다. 이 글을 쓰면서 전체 검증을 다시 실행하지는 않았다.

출력을 파일로 보냈으므로 실행 중 터미널이 조용할 수 있다. 명령이 끝나면 결과를 다음처럼 연다.

```sh
.venv/bin/python -m json.tool runs/invsqrt-full-001/result.json
```

프로세스가 정상 종료했는지 확인한 뒤, `status`가 `PASS`이고 `verified_shots`가 200,540,928인지 살펴본다. `num_qubits`와 `score`는 아래 표와 비교하면 된다. `FAIL`이라면 JSON의 `error`와 `stderr.log`를 읽는다. 출력이 비어 있거나 실행이 중단됐다면 완료된 결과로 취급하지 않는다. 이 JSON에는 실행 시간이 없으므로, 아래 199.921초는 부모 프로세스에서 따로 측정해 실행 기록에 저장한 값이다.

### 실제로 확인한 결과

원본 `sample_submission/invsqrt.kmb`를 수정하지 않고 전체 로컬 검증을 실행한 결과다. 실행 시간은 한 번 측정한 값이다.

| 측정 항목 | 결과 |
|---|---:|
| 검증 상태 | PASS |
| 검사한 shots | 200,540,928 |
| qubit 수 Q | 260 |
| shot당 Toffoli 수 | 7,735 |
| 직접 집계한 T gate | 1 |
| 합성 비용을 계산하는 임의 각도 회전 | 37 |
| rotation synthesis를 포함한 Toffoli 환산 C | 7,987.593206597238 |
| 과제 비용 Q×C | 2,076,774.233715282 |
| 검증 프로세스 wall time | 199.921초, 약 3분 20초 |
| 검증 worker 수 | 4 |

JSON의 `t_count: 1`만 보면 회전 비용이 거의 없다고 오해하기 쉽다. 37개 회전의 synthesis 비용은 최종 `toffoli_equivalent`에 포함되어 있다. phase 오차를 허용오차로 나눈 비율의 최댓값은 약 0.476912였고, 해당 x는 2,588,411이었다. 기록한 실행에서 이 정도 여유가 있었다는 뜻이다. 그 여유를 모두 줄여도 안전하다는 보장은 아니다.

재현 대상을 구분하기 위한 원본 회로 SHA-256은 다음과 같다.

```text
e2bf8eb64350a4460a271fc7f1d591ef125b1e276377e9a0f64b96ab3ca321e1
```

측정 시간에는 Python 프로세스 시작, 회로 로딩, 검증이 포함된다. 실행 전에 계산한 파일 해시 시간은 제외했다. **고전 컴퓨터에서 검증기를 돌린 시간**이며, 양자 장비에서 회로를 실행하는 시간이 아니다. 공식 안내의 전체 제출 채점 약 45분과도 작업 범위와 환경이 다르다.

현재 확인한 결과는 “기록한 환경과 salt에서 원본 `invsqrt` 회로가 공식 과제 검증을 통과했고, 비용을 재현했다”는 것이다. 세 과제를 합친 점수나 Kaggle Public/Private 성적은 아직 측정하지 않았다.

## 11. 실험을 이어가기 위한 최소한의 구조

처음부터 복잡한 관리 시스템을 만들 필요는 없다. 다만 다음 회로와 비교할 때 무엇이 달라졌는지 설명할 자료는 필요하다. 이번 프로젝트에서는 공식 원본, 후보 생성 코드, 실행 결과를 나누어 보관하고, 같은 정보를 여러 문서에서 따로 관리하지 않도록 했다.

```text
project/
├── AGENTS.md / CLAUDE.md        # 공통 진입 지침과 도구별 연결
├── TODO.md / DECISIONS.md      # 다음 작업 / 근거가 남는 결정
├── docs/                      # 목표, 정책, 자료 안내
├── .agents/skills/             # 반복해서 쓰는 실행 절차
├── .agent_runtime/exchange/    # 날짜별 검토 요청과 응답
├── data/official/<snapshot>/   # 수정하지 않은 공식 원본과 manifest
├── candidates/                # 필요해질 때 추가할 생성 코드·후보
└── runs/<run-id>/              # 한 실행의 결과·로그·실행 기록
```

이 구조는 프로젝트를 관리하는 방식을 보여 주는 예다. 공개 배포 패키지의 설치 명세는 아니다. `candidates/` 같은 디렉터리는 실제 실험이 필요할 때 추가한다. 문서를 많이 만드는 것 자체가 연구 성과는 아니므로, baseline 하나를 재현한 뒤 다음 비교에 필요한 만큼만 늘려 가려 한다.

실행마다 남길 내용은 단순하다. **어떤 원본과 후보를 어떤 코드·환경·난수·자원 한도로 실행했는가? 끝까지 통과했는가? 비용과 시간은 얼마인가?** 실패하거나 중단된 시도도 함께 보관한다. 중간까지만 실행해서 비용이 낮아 보이는 결과를, 검증을 마친 후보와 혼동하지 않기 위해서다.

Claude와 Codex의 검토는 날짜별 exchange 문서에 남긴다. 제안과 검토 의견은 그곳에서 주고받고, 확정한 결정은 `DECISIONS.md`, 현재 할 일은 `TODO.md`, 측정값은 실행 기록에 둔다. 대화가 길어져도 다음 작업을 정할 때 무엇을 읽어야 하는지 분명하게 하려는 구성이다.

## 12. 제출 형식과 지금 확인할 일정

제출 ZIP을 열었을 때 아래 파일과 디렉터리가 바로 보여야 한다. 형식을 설명하기 위한 예이며, 이 글을 준비하면서 ZIP을 업로드하거나 Kaggle에 제출하지는 않았다.

```text
submission.zip
├── invsqrt.kmb                 # 또는 .kmx
├── dqi.kmb                     # 또는 .kmx
└── dfthcsos/
    ├── angle_k_0.kmb
    ├── angle_k_1.kmb
    ├── ...                    # 0부터 빠짐없이 연속된 번호
    └── submission.npz
```

`submission.npz`에는 `U`, `W`, `H2_I`, `H1_I`, `Hsym_lower`, `tree_edges`가 들어간다. 회로 파일을 만들었다고 제출 준비가 끝나는 것은 아니다. 전체 메트릭의 사전 검사, 과제별 검증, ZIP 경로와 필수 파일까지 확인해야 한다. 자세한 파일 규칙은 [공식 평가 페이지][evaluation]에 정리되어 있다.

다음은 조사일에 확인한 일정이다. 각 마감 시각은 해당 날짜 23:59 UTC, 한국 시각으로 다음 날 08:59다.

| UTC 날짜 | 일정 |
|---|---|
| 2026-10-22 | 대회 시작 |
| 2026-11-20 | 중간 마일스톤 상 기준 시각 |
| 2027-01-04 | Longest Leader 집계 종료 및 해당 공개 자료 마감 |
| 2027-01-13 | 참가 규칙 동의·팀 병합 마감 |
| 2027-01-20 | 최종 제출 마감 |

총상금은 150,000달러다. 팀은 최대 5명, 하루 제출은 최대 5회이며 최종 선택은 2개다. 중간·선두 유지·과제별 추가 상에는 자료 공개 등 별도 조건이 있으므로 [상금 안내][prizes]와 [규칙][rules]를 확인해야 한다. 대회 초반에는 일정이나 배포 패키지가 바뀔 수 있으니 실제 참가·제출 전에는 공식 안내를 다시 읽어야 한다.

처음에는 공식 원본을 받고 `invsqrt` baseline을 실행해 PASS와 비용을 확인하는 것부터 시작하면 된다. 그다음 한 가지 변경을 정해 같은 검증 조건에서 비교한다. 이 프로젝트도 지금 그 단계에 있다. 원본의 비용은 재현했고, 어떤 계산과 임시 공간을 줄일지 검토하고 있다. 다음 글에서는 실제로 만들고 검증한 후보 회로가 생기면, 무엇을 바꿨고 결과가 어땠는지 다루려 한다.

## 참고 자료

- [Kaggle 대회][competition] · [평가][evaluation] · [데이터][data] · [도메인 설명][domain] · [일정][timeline] · [규칙][rules]
- [공식 Google Quantum AI Metric][metric] — 제출 로딩, 사전 검사, 세 과제 점수 결합의 구현.
- [Khattar 등, *Leveraging verifiable structure in quantum circuits to find more efficient compilations with agentic AI*][paper] — 대회 동반 논문.
- [Qualtran의 kickmix 문서][kickmix-docs] · [이번 로컬 빌드의 소스 커밋][kickmix-pin]
- [DFTHC starter kit][starter-kit] · [starter notebook][starter-notebook]
- [Gu 등, *Discovering state-of-the-art quantum circuits with large language models*][gu-paper] · [구현 저장소][uncompute]
- [Kaggle CLI][kaggle-cli] — 다운로드와 인증 설정.

[competition]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation
[evaluation]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/overview/evaluation
[data]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/data
[domain]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/overview/domain-context
[timeline]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/overview/timeline
[prizes]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/overview/prizes
[rules]: https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation/rules
[metric]: https://www.kaggle.com/code/metric/google-quantum-ai-metric
[paper]: https://arxiv.org/abs/2610.23832
[kickmix-docs]: https://github.com/quantumlib/Qualtran/tree/main/docs/kickmix
[kickmix-pin]: https://github.com/quantumlib/Qualtran/tree/638777fac4379bc62e2a1e4a52cf1a1753895d48/kickmix
[starter-kit]: https://www.kaggle.com/datasets/sohier/verifiable-circuit-compilation-starter-kit
[starter-notebook]: https://www.kaggle.com/code/sohier/verifiable-circuit-compilation-starter
[gu-paper]: https://arxiv.org/abs/2610.22367
[uncompute]: https://github.com/andigu/uncompute
[kaggle-cli]: https://github.com/Kaggle/kaggle-cli

[ibm-states]: https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/single-systems/introduction
[ibm-phase]: https://quantum.cloud.ibm.com/learning/en/courses/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring/phase-estimation-procedure
[ibm-reversible]: https://quantum.cloud.ibm.com/learning/en/courses/fundamentals-of-quantum-algorithms/quantum-algorithmic-foundations/simulating-classical-computations
