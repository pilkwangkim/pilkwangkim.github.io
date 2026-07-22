---
title: "닫힌 양자계에서 열린 양자 수송으로: NEGF를 처음부터 유도하기"
date: 2026-07-22 09:00:00 +0900
categories: [Physics, Quantum Transport]
tags: [physics, quantum-transport, negf, green-functions, tight-binding, open-quantum-systems, self-energy, landauer, resonant-tunneling, python, kaggle, korean]
math: true
pin: false
---

# 닫힌 양자계에서 열린 양자 수송으로: NEGF를 처음부터 유도하기

- 실행 가능한 노트북: [From Closed to Open: Quantum Transport with NEGF, From Scratch](https://www.kaggle.com/code/pilkwang/from-closed-to-open-quantum-transport-with-negf)
- 소스 노트북: [negf_from_scratch.ipynb](https://github.com/pilkwangkim/Physics/blob/master/negf_from_scratch.ipynb)
- 영문판: [From Closed to Open Quantum Transport with NEGF: A Derivation from Scratch]({{ site.baseurl }}/posts/From-Closed-to-Open-Quantum-Transport-with-NEGF/)

유한한 Hamiltonian을 대각화하면 energy level과 eigenstate를 얻을 수 있다. 하지만 그것만으로는 steady-state current를 계산할 수 없다. 전류가 흐르는 device는 열린계다. 한쪽의 거대한 reservoir에서 전자가 들어와 유한한 device 영역을 지난 뒤 반대편 reservoir로 빠져나간다. 계산 영역 끝에 세운 가짜 벽에서 튕겨 되돌아와서는 안 된다.

그래서 이 글의 중심 질문은 다음과 같다.

```text
무한히 긴 lead가 달린 열린 수송 문제를,
임의의 absorbing boundary condition 없이 어떻게 유한 행렬로 정확히 표현할까?
```

nonequilibrium Green's function (NEGF)에서는 lead의 **자유도**를 정확히 소거한다. 소거한 자유도의 영향은 사라지는 것이 아니라, 에너지에 따라 달라지는 self-energy에 모두 담긴다.

이 단계의 의미를 제대로 이해하면 transmission, local density of states (LDOS), nonequilibrium occupation, 전류 보존, resonant tunneling, negative differential resistance (NDR)가 따로 떨어진 공식이 아니라 하나의 계산 흐름으로 연결된다.

이 글에서는 노트북보다 설명을 더 천천히 풀어 간다. wavefunction, Hamiltonian, eigenvalue problem, tight-binding band 정도는 안다고 가정하지만, Green's function이나 양자 수송을 미리 알 필요는 없다.

> **용어 표기 안내.** 문장은 자연스러운 한국어로 쓰되, NEGF 문헌에서 표준적으로 쓰이는 Hamiltonian, Green's function, self-energy, lead, device, reservoir, contact, transmission, broadening, discrete spectrum, continuum, wavefunction 같은 전문용어는 원문 검색과 교재 대조가 쉽도록 영어로 적는다.

계산 전 과정을 눈으로 따라갈 수 있도록, 단순한 모형 하나를 끝까지 사용한다.

- 유효질량 전자를 균일 격자에 놓은 1차원 모형
- nearest-neighbor hopping과 spin degeneracy 2
- coherent·elastic single-particle transport
- 균일한 semi-infinite lead
- Poisson equation의 self-consistent solution 대신 외부에서 정한 bias profile

가정을 이렇게 단순화하면 수식을 끝까지 눈으로 따라갈 수 있고, 이 모형이 설명할 수 있는 범위도 분명해진다.

---

## 0. 먼저 전체 계산 지도를 보자

세부 유도에 들어가기 전에 전체 흐름부터 잡아 두면 길을 잃기 어렵다.

```text
continuum Schrödinger equation
        ↓ finite difference
유한 tight-binding Hamiltonian H_D
        ↓ semi-infinite lead를 붙이고 정확히 소거
retarded self-energy Σ_L^R(E), Σ_R^R(E)
        ↓
open-device Green's function G^R(E)
        ↓                         ↓
spectral function A_L, A_R          transmission T(E)
        ↓                         ↓
reservoir 점유 f_L, f_R            Landauer 전류
        ↓
electron correlation function G^n, 밀도, 결합 전류
```

처음부터 세 가지 대상을 구분해야 한다.

1. **lead**는 phase coherence를 유지하는 semi-infinite system이다. lead Hamiltonian이 propagating mode를 정하고, retarded solution에는 outgoing-wave boundary condition이 적용된다.
2. **reservoir**의 chemical potential과 온도가 incident states의 occupation을 정한다.
3. 실제 **contact**는 두 역할을 함께 하지만, 수식에서는 서로 다른 항으로 나타난다. retarded self-energy는 전파와 이탈을, Fermi function은 점유를 기술한다.

또 하나, $E+i\eta$에 들어가는 작은 양수 $\eta$는 retarded boundary value를 선택하는 수치적 장치다. resonance의 물리적 lifetime을 뜻하지 않는다. 실제 contact broadening은 self-energy의 anti-Hermitian part에서 나온다.

### 0.1 기호마다 어떤 물리를 나타내는가

같은 기호가 계속 나오므로, 기호마다 물리적 역할을 질문 형식으로 정리해 두자.

- $H_D$: lead를 연결하기 전, 유한 device의 single-particle dynamics를 무엇이 정하는가?
- $g_\alpha^R$: 연결되지 않은 **isolated lead** $\alpha$는 주어진 에너지에서 어떻게 응답하는가?
- $\Sigma_\alpha^R$: 소거한 lead의 영향이 device 경계에 어떤 에너지 의존적 feedback으로 남는가?
- $G^R$: lead와 연결된 **open device**는 source에 어떻게 응답하고 진폭을 전파하는가?
- $\Gamma_\alpha$: 그 에너지에서 contact $\alpha$를 통해 진폭이 얼마나 잘 빠져나갈 수 있는가?
- $A_\alpha$: contact $\alpha$에서 들어오는 scattering state는 device 안에서 어떤 spectral weight를 갖는가?
- $f_\alpha$: reservoir $\alpha$는 그 incident state를 얼마나 채우는가?
- $G^n$: 그 결과 실제로 점유된 spectral weight는 어디에 있는가?
- $T(E)$: flux로 정규화한 incident channel 가운데 얼마가 반대편 lead에 도달하는가?
- $I$: 양쪽 incident-state occupation의 차이를 에너지에 대해 적분하면 net flow가 얼마인가?

대소문자에 따라서도 뜻이 달라진다. $\Gamma_\alpha$는 device subspace에 작용하는 행렬이고, $\gamma_\alpha$는 이 single-channel model에서 그 행렬의 0이 아닌 유일한 끝점 원소다. $T(E)$는 언제나 transmission을 뜻한다. reservoir 온도는 $T_\alpha$ 또는 에너지 척도 $k_BT_\alpha$로 적는다.

---

## 1. Schrödinger 방정식을 tight-binding 행렬로 옮기기

1차원 effective-mass equation에서 시작하자.

$$
-\frac{\hbar^2}{2m^*}\frac{d^2\psi}{dx^2}+U(x)\psi(x)=E\psi(x).
$$

$x_j=ja$에 격자점을 두면 2차 미분의 중앙차분은 다음과 같다.

$$
\left.\frac{d^2\psi}{dx^2}\right\rvert_{x_j}
\approx
\frac{\psi_{j+1}-2\psi_j+\psi_{j-1}}{a^2}
$$

이제 양의 운동에너지 척도

$$
t_0\equiv\frac{\hbar^2}{2m^*a^2}
$$

를 정의하면 각 격자점의 방정식은

$$
-t_0\psi_{j-1}+(2t_0+U_j)\psi_j-t_0\psi_{j+1}=E\psi_j
\tag{1}
$$

이 된다. 대수적 형태는 nearest-neighbor tight-binding model과 같지만, 여기서는 원자 궤도 사이의 경험적 hopping을 가정한 것이 아니다. continuum kinetic-energy operator를 finite difference로 옮기면 이 형태가 나온다.

사이트 기저에서 행렬 원소는 다음과 같다.

$$
H_{jj}=2t_0+U_j,
\qquad
H_{j,j+1}=H_{j+1,j}=-t_0
$$

여기서 $U_j$와 $H_{jj}$를 혼동하면 안 된다. $U_j$가 물리적인 potential energy이고, Hamiltonian 대각의 $2t_0$는 discrete kinetic-energy operator에서 생긴 균일한 항이다.

노트북의 $m^*=0.067m_0$, $a=1\ \mathrm{nm}$를 넣으면 다음 값을 얻는다.

$$
t_0=0.5687\ \mathrm{eV},
\qquad
4t_0=2.2746\ \mathrm{eV}
$$

```python
def device_hamiltonian(N, t0, U):
    """N개 격자점으로 이루어진 finite-difference Hamiltonian."""
    H = np.zeros((N, N), dtype=complex)
    for j in range(N):
        H[j, j] = 2.0 * t0 + U[j]
    for j in range(N - 1):
        H[j, j + 1] = -t0
        H[j + 1, j] = -t0
    return H
```

배열을 복소수형으로 만들었다고 해서 closed-system Hamiltonian이 non-Hermitian이 되는 것은 아니다. 뒤에서 complex self-energy를 더할 때 자료형을 다시 바꾸지 않으려는 선택일 뿐이다.

격자 간격은 단순한 그림 해상도가 아니라 물리 근사의 일부다. $t_0\propto a^{-2}$이므로 $a$를 바꾸면 격자 bandwidth와 $U/t_0$ 같은 모든 무차원 비율이 함께 바뀐다.

수렴성을 확인할 때는 device의 **실제 길이와 장벽의 실제 폭**을 고정하고, 더 촘촘한 격자에서 $U_j$를 새로 만든 뒤 관심 있는 관측량의 값이 더 이상 변하지 않는지 확인해야 한다. 사이트 수를 그대로 둔 채 $a$만 절반으로 줄이면 같은 device를 더 정확히 푸는 것이 아니라 길이가 절반인 다른 device를 푸는 셈이다.

### 1.1 lattice band와 continuum limit

$U_j=0$인 균일 무한 체인에 Bloch wave

$$
\psi_j=e^{ikja}
$$

를 넣으면 다음 dispersion relation을 얻는다.

$$
E(k)=2t_0-t_0e^{ika}-t_0e^{-ika}
=2t_0(1-\cos ka)
\tag{2}
$$

band bottom은 0, band top은 $4t_0$다. $\lvert ka\rvert\ll1$에서는

$$
1-\cos ka\simeq\frac{(ka)^2}{2}
$$

이므로

$$
E(k)\simeq t_0(ka)^2=\frac{\hbar^2k^2}{2m^*}
$$

가 되어 continuum limit의 parabolic dispersion과 일치한다.

Brillouin zone 끝에서 곡선이 포물선에서 벗어나는 것은 실제 물질 band의 nonparabolicity를 예측한 것이 아니다. 유한 격자에서 생긴 dispersion error다. 관심 에너지 범위에서 parabolic approximation이 충분히 잘 맞는지 $a$를 줄여 가며 확인해야 한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-01-lattice-dispersion.png" alt="finite-difference lattice의 dispersion relation과 hard-wall 유한 체인의 discrete spectrum" width="94%">
</p>

*그림 1.* lattice dispersion은 $k=0$ 부근에서 continuum의 포물선과 일치하고 $4t_0$의 유한한 bandwidth를 갖는다. hard-wall 유한 체인에서는 boundary condition이 허용하는 discrete wavevector에서만 이 band의 에너지를 취한다.

---

## 2. 닫힌 체인에서 알 수 있는 것과 알 수 없는 것

$N$개의 물리적 격자점 바깥에 두 개의 ghost sites를 두고

$$
\psi_0=\psi_{N+1}=0
$$

인, 즉 wavefunction이 경계에서 0이 되는 hard-wall boundary condition을 가정하자. 허용되는 mode와 에너지는 각각 다음과 같다.

$$
k_na=\frac{n\pi}{N+1},
\qquad
\psi_n(j)=\sqrt{\frac{2}{N+1}}
\sin\!\left(\frac{n\pi j}{N+1}\right)
$$

$$
E_n=2t_0\left[1-\cos\!\left(\frac{n\pi}{N+1}\right)\right]
$$

노트북에서 행렬을 직접 대각화해 얻은 eigenvalue와 이 식의 차이는 최대 $1.33\times10^{-15}\ \mathrm{eV}$다. eigenvector residual도 $\lVert H\psi_n-E_n\psi_n\rVert<5.5\times10^{-16}\ \mathrm{eV}$로 작다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-02-closed-chain-eigenstates.png" alt="hard-wall 유한 tight-binding 체인의 처음 다섯 eigenstate를 각 eigenenergy 주변에 표시한 그림" width="90%">
</p>

*그림 2.* 세로 방향의 물결은 eigenfunction을 보기 쉽게 각 에너지 위치에 겹쳐 그린 것이다. 에너지 불확정성이나 확률밀도를 뜻하지 않는다. 두 hard walls 사이의 실제 potential은 $U_j=0$이다.

이 닫힌계 문제는 Hamiltonian이 제대로 만들어졌는지 검증하기에 아주 좋다. 그러나 아직 수송 실험은 아니다.

- 왼쪽 reservoir가 공급하는 right-moving incident state의 점유가 없다.
- 오른쪽 reservoir가 공급하는 left-moving incident state의 점유도 없다.
- 확률이 빠져나갈 곳이 없다.
- spectrum은 discrete stationary states로만 이루어진다.

실수이고 시간반전 대칭인 이 1차원 체인에서는 hard-wall eigenstate를 실수로 고를 수 있고, nearest-neighbor bond의 확률 전류는 0이다. 닫힌 체인을 더 길게 만들어도 수송 문제는 해결되지 않는다. 인공 벽을 멀리 옮겨 반사가 돌아오는 시간을 늦추고 standing-wave level spacing을 줄일 뿐이다.

수송을 계산하려면 행렬 크기가 아니라 **boundary condition**을 바꿔야 한다.

---

## 3. Green's function은 에너지별 소스-응답 연산자다

먼저 Schrödinger 방정식을 source에 대한 선형응답 문제로 다시 써 보자.

$$
\big[(E+i\eta)I-H\big]\lvert\psi\rangle=\lvert s\rangle,
\qquad \eta>0.
$$

$\lvert s\rangle$는 실제 reservoir가 전자를 채우는 과정이 아니라, 응답을 정의하기 위한 수학적 탐침이다. 해와 retarded Green's function은 다음과 같다.

$$
\lvert\psi\rangle=G^R(E)\lvert s\rangle,
\qquad
G^R(E)=\big[(E+i\eta)I-H\big]^{-1}
\tag{3}
$$

따라서 $G^R_{ab}$는 $b$에 unit source를 놓았을 때 $a$에서 나타나는 amplitude response다. 역행렬을 취하기 전 operator의 단위가 에너지이므로 $G^R$의 단위는 $\mathrm{eV}^{-1}$다.

closed-system Hamiltonian의 eigenstate를 쓰면 다음 spectral representation을 얻는다.

$$
G^R(E)=\sum_n\frac{\lvert n\rangle\langle n\rvert}{E-E_n+i\eta}
\tag{4}
$$

$E$가 $E_n$에 가까워질수록 해당 항의 응답이 커진다. spectral function은 다음과 같이 정의한다.

$$
A(E)=i(G^R-G^A),
\qquad G^A=G^{R\dagger}
$$

여기서 $G^A$는 advanced Green's function이다. spectral function은 이 pole structure가 실수 에너지축에서 어떤 spectral weight로 나타나는지 보여 준다. 스핀 하나당 LDOS는 다음과 같다.

$$
\rho_j(E)=\frac{A_{jj}(E)}{2\pi}
$$

이 spectral representation에는 eigenvalue 목록만으로는 알 수 없는 정보도 담겨 있다. 각 eigenstate에 대응하는 projector가 그대로 남기 때문이다.

$$
P_n=\lvert n\rangle\langle n\rvert
$$

projector에는 각 mode의 공간 분포가 그대로 담겨 있고, 식 (4)의 분모는 그 mode가 **어느 에너지에서** 크게 응답하는지를 정한다. 따라서 Green's function을 알면 eigenvalue뿐 아니라 원하는 probe energy에서의 공간 응답도 계산할 수 있다.

lead를 연결해 open system이 되어도 이 analytic structure 자체는 유지된다. 다만 closed-system level의 real-axis pole은 보통 complex-energy plane으로 이동한다. 실수부는 resonance 위치를, 음의 허수부는 contact를 통한 escape-induced decay를 정한다.

여기서 pole과 peak는 구분할 필요가 있다. pole은 $G^R$를 complex-energy plane으로 analytic continuation했을 때의 특이점이다. 반면 peak는 유한한 해상도의 실수 에너지 scan에서 눈에 보이는 봉우리다.

resonance 사이의 간격이 충분히 크고 self-energy가 그 주변에서 천천히 변한다면 pole과 peak를 거의 같은 뜻으로 써도 된다. 하지만 resonance가 겹치거나 band threshold가 가깝거나 self-energy가 빠르게 변하면 peak 위치와 complex pole의 실수부가 단순히 일치하지 않을 수 있다.

고립된 유한계에서 정확한 $\eta\to0^+$ spectrum은 delta peak들의 합이다. 그림을 그리기 위해 유한한 $\eta$를 쓰면 peak가 폭을 가진 것처럼 보이지만, 그 폭은 physical lifetime이 아니다.

왜 $E-i\eta$가 아니라 $E+i\eta$일까? 시간영역에서는 source를 가한 뒤에만 응답하는 해를 고른다. lead에서는 device에서 멀어지는 outgoing wave를 선택하고, 무한대에서 device 쪽으로 들어오는 incoming wave는 포함하지 않는다. 이것이 retarded Green's function의 boundary condition이다.

---

## 4. lead 자유도를 정확히 소거해 열린계를 만든다

본격적인 block matrix 유도에 앞서, 같은 대수를 두 개의 추상적인 block으로 먼저 살펴보자.

$$
a_D\psi_D-\tau\psi_L=s_D,
\qquad
-\tau^\dagger\psi_D+a_L\psi_L=0.
$$

둘째 식에서 $\psi_L=a_L^{-1}\tau^\dagger\psi_D$를 구해 첫째 식에 대입하면 다음 effective equation이 된다.

$$
\big(a_D-\tau a_L^{-1}\tau^\dagger\big)\psi_D=s_D
$$

$\psi_L$를 식에서 없앴다고 해서 lead의 영향까지 사라지는 것은 아니다. lead가 device에 미치는 영향은 $\tau a_L^{-1}\tau^\dagger$ 항으로 정확히 남는다. Schur complement는 이 대입을 block matrix에 적용한 것이다.

single-particle basis를 유한한 device subspace $D$와 lead subspace $L$로 나누자.

$$
H_{\mathrm{tot}}=
\begin{pmatrix}
H_D & \tau\\
\tau^\dagger & H_L
\end{pmatrix},
\qquad
\lvert\psi\rangle=
\begin{pmatrix}
\lvert\psi_D\rangle\\
\lvert\psi_L\rangle
\end{pmatrix}.
\tag{5}
$$

이 글의 규약은 다음과 같다.

- $\tau=H_{DL}$이며, $\tau\psi_L$는 lead 진폭이 device 방정식에 결합되는 항이다.
- $\tau^\dagger=H_{LD}$이며, $\tau^\dagger\psi_D$는 device 진폭이 lead 방정식에 결합되는 항이다.
- $g_L^R=[(E+i\eta)I_L-H_L]^{-1}$는 contact에 연결하기 전 **isolated lead**의 Green's function이다.

소문자 $g_L^R$를 이미 device와 연결된 전체 Green's function의 $G_{LL}^R$ block과 혼동하면 안 된다. 후자는 device를 거치는 왕복까지 포함한다. self-energy는 최종적으로 device space에 작용해야 하므로, 행렬 크기도 다음처럼 $N_D\times N_D$가 된다.

$$
(N_D\!\times\!N_L)\,
(N_L\!\times\!N_L)\,
(N_L\!\times\!N_D)
\longrightarrow N_D\!\times\!N_D
$$

단위도 $(\mathrm{eV})(\mathrm{eV}^{-1})(\mathrm{eV})=\mathrm{eV}$이므로, $\Sigma_L^R$는 $H_D$에 더할 수 있는 device 크기의 에너지 행렬이다.

device 쪽에만 수학적 소스를 가하면 block 방정식은 다음과 같다.

$$
\begin{aligned}
\big[(E+i\eta)I_D-H_D\big]\psi_D-\tau\psi_L&=s_D,\\
-\tau^\dagger\psi_D+
\big[(E+i\eta)I_L-H_L\big]\psi_L&=0.
\end{aligned}
\tag{6}
$$

둘째 줄은 근사 없이 풀 수 있다.

$$
\psi_L=g_L^R\tau^\dagger\psi_D.
\tag{7}
$$

이를 첫째 줄에 넣으면 다음 식을 얻는다.

$$
\left[(E+i\eta)I_D-H_D-
\underbrace{\tau g_L^R\tau^\dagger}_{\Sigma_L^R(E)}
\right]\psi_D=s_D
\tag{8}
$$

이때 lead self-energy는 다음과 같이 정의된다.

$$
\boxed{\Sigma_L^R(E)=\tau g_L^R(E)\tau^\dagger}
\tag{9}
$$

행렬곱은 오른쪽부터 읽어야 한다. device 경계의 진폭이 $\tau^\dagger$를 통해 $D\to L$로 나가고, $g_L^R$로 lead 안을 전파한 뒤, $\tau$를 통해 $L\to D$로 돌아온다. 즉 self-energy 한 항에는 $D\to L$ coupling, lead 내부 전파, $L\to D$ coupling으로 이어지는 한 차례의 round trip이 모두 들어 있다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-03-exact-lead-elimination.png" alt="lead를 정확히 소거하고 contact device 사이트에 energy-dependent feedback을 남기는 과정과 surface 자기유사성" width="96%">
</p>

*그림 3.* (a)는 device와 lead 자유도를 모두 표시한 원래 문제다. (b)에는 device 자유도만 남아 있지만, $\Sigma_L^R$가 lead의 영향을 빠짐없이 담고 있다. (c)는 균일한 semi-infinite lead에서 같은 구조가 반복되는 recursion relation이 왜 성립하는지 보여 준다.

여기서 “정확하다”가 무엇을 뜻하는지 분명히 해 두자.

- 전체 resolvent, 즉 energy-domain Green's function의 **device block**과 식 (8)의 역행렬이 같다는 뜻이다.
- 전체 Hermitian Hamiltonian이 더 작은 non-Hermitian Hamiltonian과 같다는 뜻은 아니다.
- 앞에서 정한 single-particle Hamiltonian과 subspace 분할에 대해서는 이 block elimination에 근사가 없다.
- 소거한 lead도 자체 동역학을 가지므로 결과는 에너지에 의존한다.
- open channel에서는 남겨 둔 device subspace 밖으로 진폭이 빠져나갈 수 있으므로 effective operator가 non-Hermitian일 수 있다.

같은 식은 device와 lead 사이의 repeated round trip으로도 이해할 수 있다. 연결 전 device만의 resolvent를

$$
g_D^R=[(E+i0^+)I_D-H_D]^{-1}
$$

라고 쓰자. 전체 lead self-energy를

$$
\Sigma^R\equiv\Sigma_L^R+\Sigma_R^R
$$

로 정의하면 $\Sigma^R$에는 device에 연결한 모든 lead의 feedback이 포함된다. Dyson equation은 다음과 같다.

$$
G^R=g_D^R+g_D^R\Sigma^RG^R
$$

이를 formal series로 전개하면

$$
G^R
=g_D^R
+g_D^R\Sigma^Rg_D^R
+g_D^R\Sigma^Rg_D^R\Sigma^Rg_D^R+\cdots
$$

$\Sigma^R$ 하나는 device에서 lead로 나갔다가 돌아오는 한 차례의 round trip을 나타낸다. 식 (10)은 이 과정이 몇 번이고 반복되는 경우를 모두 합한다. 따라서 lead를 “소거한다”는 말은 전자가 lead를 한 번만 방문하도록 제한한다는 뜻이 아니다. lead 자유도를 행렬에 직접 남겨 두지 않고도, 그 linear response가 반복해서 작용하는 모든 order를 합한다는 뜻이다.

양쪽 lead를 소거한 뒤 최종적으로 역행렬을 구해야 할 finite matrix는 다음과 같다.

$$
G^R(E)=
\left[
(E+i0^+)I_D-H_D-\Sigma_L^R(E)-\Sigma_R^R(E)
\right]^{-1}
\tag{10}
$$

노트북에서는 유한 lead를 붙인 전체 역행렬의 device block과 이 Schur complement를 직접 비교한다. 최대 차이는 $3.48\times10^{-15}\ \mathrm{eV}^{-1}$다.

<details markdown="1">
<summary>코드: 유한 lead로 Schur complement를 직접 검증하기</summary>

```python
g_L = np.linalg.inv(z * np.eye(N_L) - H_L)
Sigma_L = tau @ g_L @ tau.conj().T

G_D_schur = np.linalg.inv(
    z * np.eye(N_D) - H_D - Sigma_L
)

H_full = np.block([
    [H_D,          tau],
    [tau.conj().T, H_L],
])
G_full = np.linalg.inv(z * np.eye(N_D + N_L) - H_full)
G_D_projected = G_full[:N_D, :N_D]

print(np.max(np.abs(G_D_projected - G_D_schur)))
```

</details>

---

## 5. 무한 lead가 surface Green's function 하나로 줄어드는 이유

single-orbital contact에서는 device의 경계 사이트 $\lvert d_0\rangle$와 lead surface site $\lvert\ell_0\rangle$만 맞닿는다.

$$
\tau=-t_0\lvert d_0\rangle\langle\ell_0\rvert.
$$

이 식을 self-energy에 넣으면 다음 device-space operator를 얻는다.

$$
\Sigma_L^R
=t_0^2\lvert d_0\rangle
\underbrace{\langle\ell_0\rvert g_L^R\lvert\ell_0\rangle}_{g_s^R(E)}
\langle d_0\rvert
$$

무한 행렬 $g_L^R$ 전체가 필요할 것처럼 보였지만, contact matrix $\tau$가 양쪽에서 surface subspace만 골라 내기 때문에 surface-to-surface 원소

$$
g_s^R(E)=\langle\ell_0\rvert g_L^R(E)\lvert\ell_0\rangle
$$

하나만 남는다.

lead가 site energy $\varepsilon$, nearest-neighbor hopping $-t_0$를 가진 균일한 semi-infinite chain이라고 하자. 이 finite-difference wire에서는 $\varepsilon=2t_0+U_{\mathrm{lead}}$다. 따라서 깨끗한 lead $U_{\mathrm{lead}}=0$의 band는 1절과 마찬가지로 $[0,4t_0]$가 된다.

surface site $\ell_0$를 떼어 내면 $\ell_1$에서 시작하는 나머지 chain이 남는다. site index만 하나 밀렸을 뿐 원래 lead와 똑같으므로, 새 surface에서 본 Green's function도 $g_s^R$다.

surface site에서 나머지 chain을 한 번 소거하면 다음 self-consistency equation을 얻는다.

$$
g_s^R(E)=
\frac{1}{E+i0^+-\varepsilon-t_0^2g_s^R(E)}
\tag{11}
$$

같은 식을 continued fraction으로 쓰면 self-similarity가 더 잘 보인다.

$$
g_s^R=
\cfrac{1}{E^+-\varepsilon-
\cfrac{t_0^2}{E^+-\varepsilon-
\cfrac{t_0^2}{\ddots}}}.
$$

식 (11)을 정리하면 다음 quadratic equation이 된다.

$$
t_0^2(g_s^R)^2-
(E+i0^+-\varepsilon)g_s^R+1=0
$$

이 quadratic equation에는 다음 두 root가 있다.

$$
g_\pm(E)=
\frac{E+i0^+-\varepsilon
\ \pm\ \sqrt{(E+i0^+-\varepsilon)^2-4t_0^2}}
{2t_0^2}
\tag{12}
$$

그러나 부호를 아무렇게나 고를 수는 없다. 둘 가운데 retarded condition을 만족하는 root만 physical solution이며, 이를 $g_s^R$로 쓴다.

1. complex energy의 upper half-plane에서 analytic이며, real-axis boundary에서 $\operatorname{Im}g_s^R\le0$이다.
2. $\lvert E\rvert\to\infty$에서 $g_s^R\sim1/E$다.
3. band 밖에서는 lead 내부로 갈수록 작아지는 decaying solution을 고르고, 무한대로 갈수록 커지는 growing solution을 버린다.

band 안에서

$$
E=\varepsilon-2t_0\cos ka,
\qquad 0<ka<\pi
$$

로 쓸 때, 현재의 hopping 부호 규약에 맞는 retarded surface Green's function은 다음과 같다.

$$
g_s^R(E)=-\frac{e^{ika}}{t_0}
\tag{13}
$$

여기서는 각 lead의 국소 사이트 좌표 $n=0,1,\ldots$가 device에서 **멀어지는 방향**으로 증가한다고 잡는다. 따라서 식 (13)은 양쪽 lead에서 모두 outgoing wave를 뜻한다. 전체 좌표로 보면 오른쪽 lead에서는 $+k$, 왼쪽 lead에서는 $-k$에 해당한다.

$q=-t_0g_s^R$를 정의하면 $\eta\to0^+$ limit에서 $q=e^{ika}$가 복소평면 단위원의 위쪽 반원을 따라간다. 같은 physical branch를 band 밖으로 이어 가면 $\lvert q\rvert<1$이 되어 lead 안쪽으로 감쇠한다. 다른 root는 $q^{-1}$이며 band 밖에서 공간적으로 커진다.

```python
def surface_green_scalar(E, t0, eps_lead, eta=1e-9):
    b = (E + 1j * eta) - eps_lead
    disc = np.sqrt(b * b - 4.0 * t0 * t0 + 0j)
    roots = ((b + disc) / (2.0 * t0**2),
             (b - disc) / (2.0 * t0**2))

    # propagating band 안에서는 Im(g_s)<0인 retarded root를 고른다.
    for g in roots:
        if g.imag < 0:
            return g

    # eta=0인 band 밖에서는 감쇠하면서 1/E로 가는 root를 고른다.
    return min(roots, key=lambda g: abs(t0 * g))

def lead_self_energy_scalar(E, t0, eps_lead, eta=1e-9):
    """matched endpoint의 self-energy: sigma^R = t0^2 g_s^R."""
    return t0**2 * surface_green_scalar(E, t0, eps_lead, eta)
```

contact hopping까지 $\tau=-t_0$로 맞추면 surface self-energy는 다음처럼 real shift와 broadening으로 나뉜다.

$$
\sigma^R=t_0^2g_s^R=-t_0e^{ika}
=\underbrace{-t_0\cos ka}_{\lambda(E)}
-\frac{i}{2}\underbrace{2t_0\sin ka}_{\gamma(E)}
\tag{14}
$$

실수부 $\lambda=\operatorname{Re}\sigma^R$는 resonance 위치를 옮긴다. 양의 contact broadening

$$
\gamma=-2\operatorname{Im}\sigma^R
$$

는 propagating lead channel을 통한 escape coupling을 나타낸다. 행렬에서는 broadening matrix를 다음과 같이 정의한다.

$$
\Gamma_\alpha
=i\left(\Sigma_\alpha^R-\Sigma_\alpha^A\right)
\succeq0
\tag{15}
$$

여기서 $\Sigma_\alpha^A=\Sigma_\alpha^{R\dagger}$다. 행렬에서 필요한 “허수부”는 다음 anti-Hermitian part다.

$$
\operatorname{Im}_{H}X=\frac{X-X^\dagger}{2i},
\qquad
\Gamma_\alpha=-2\operatorname{Im}_{H}\Sigma_\alpha^R
$$

이는 일반 행렬의 각 원소에 스칼라 허수부를 취하는 것과 같지 않다. 이 글의 단일 스칼라 끝점 $\sigma^R$에서는 두 정의가 일치한다.

임의의 contact hopping $t_c$에 대해서는 다음 관계를 항상 쓸 수 있다.

$$
\gamma=2\pi\lvert t_c\rvert^2\rho_s,
\qquad
\rho_s=-\frac{1}{\pi}\operatorname{Im}g_s^R
$$

여기서 $\rho_s$는 bulk가 아니라 **surface DOS**다. $\gamma=\hbar\lvert v\rvert/a$라는 추가 관계는 이 글의 matched contact, 즉 $t_c=t_0$에서만 성립한다.

1차원에서는 특히 “surface”라는 말이 중요하다. 식 (13)에서 band 안의 끝점 spectral density는 다음과 같다.

$$
\rho_s(E)=\frac{\sin ka}{\pi t_0}
$$

이 값은 양쪽 band edge에서 0으로 간다. 반면 translationally invariant infinite chain의 사이트당 bulk DOS는

$$
\rho_{\mathrm{bulk}}(E)
=\frac{1}{2\pi t_0\lvert\sin ka\rvert}
$$

라서 band edge에서 발산한다. 서로 모순이 아니다. bulk DOS는 $k$-state가 에너지축에 얼마나 조밀하게 모이는지를 나타낸다. 반면 surface DOS에는 각 mode가 끝점 orbital에 갖는 weight까지 포함된다. 끝점이 있는 semi-infinite chain의 band-edge mode는 경계에서 진폭이 작아지고, 그 boundary weight가 bulk의 van Hove divergence를 상쇄한다.

scalar matched contact에서는 group velocity

$$
v(k)=\frac{1}{\hbar}\frac{dE}{dk}
=\frac{2t_0a}{\hbar}\sin ka
$$

를 이용해 $\gamma=2t_0\sin ka=\hbar\lvert v\rvert/a$를 곧바로 얻는다. surface coupling을 band edge에서 0으로 만드는 바로 그 $\sin ka$가 group velocity도 0으로 만든다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-04-surface-green-function.png" alt="닫힌 체인과 열린 체인의 spectral response, contact self-energy의 실수부와 허수부, 복소평면에서의 retarded surface root" width="98%">
</p>

*그림 4.* 체인을 열면 hard-wall discrete levels가 lead band 안의 continuum으로 바뀐다. self-energy의 실수부는 level을 이동시키고, broadening은 propagating band 밖에서 0이 된다. 복소 $q$ 그림은 retarded root 선택을 기하학적으로 보여 준다.

band 밖에서 broadening이 0이라고 해서 contact의 영향이 전부 사라지는 것은 아니다. propagating escape channel은 닫히지만, real self-energy와 evanescent boundary response는 남을 수 있다.

양쪽 lead continuum과 완전히 분리된 진짜 bound state가 있다면 그 점유는 두 contact의 Fermi function만으로 자동 결정되지 않는다.

---

## 6. open-device Green's function에서 transmission까지

이제 수송 문제에 필요한 식이 모두 준비됐다. 각 에너지에서는 다음 순서로 계산한다.

1. 왼쪽과 오른쪽 lead의 surface Green's function을 구한다.
2. 각 self-energy를 해당 contact의 device orbital에 넣는다.
3. $\Gamma_L$, $\Gamma_R$를 만든다.
4. 유한한 open-device operator를 invert해 $G^R$를 구한다.
5. spectral function과 transmission을 계산한다.

device의 전체 spectral weight를 나타내는 spectral function은 다음과 같이 정의한다.

$$
A(E)=i(G^R-G^A)
\tag{16}
$$

contact와 결합해 생긴 continuum에 대해 $\eta\to0^+$ limit을 취하면

$$
A=A_L+A_R,
\qquad
A_\alpha=G^R\Gamma_\alpha G^A
\tag{17}
$$

로 나눌 수 있다. $A_L$은 왼쪽 contact에서 들어오는 scattering state가 device 안에서 갖는 spectral weight이고, $A_R$은 오른쪽 contact에 대한 같은 양이다. 아직 Fermi function은 들어가지 않았다. 이 식들은 scattering state가 device의 어느 위치에 나타날 수 있는지만 알려 준다. 그 state가 **실제로 얼마나 점유되는가**는 아직 정하지 않는다.

위 식에는 중요한 단서가 있다. 유한한 수치 $\eta$에서는 다음 항등식이 정확하다.

$$
i(G^R-G^A)
=G^R\big(\Gamma_L+\Gamma_R+2\eta I\big)G^A
$$

$2\eta I$는 수치적 조절항이지 세 번째 contact가 아니다. 또한 $\Gamma_L=\Gamma_R=0$인 진짜 bound state에는 두 reservoir만으로 점유를 정할 수 없는 문제가 남는다.

### 6.1 Caroli 식의 trace가 transmission이 되는 이유

coherent transmission은 다음 Caroli formula로 계산한다.

$$
\boxed{
T(E)=\operatorname{Tr}
\left[\Gamma_LG^R\Gamma_RG^A\right]
}
\tag{18}
$$

절댓값 제곱 구조를 드러내기 위해

$$
X=\Gamma_L^{1/2}G^R\Gamma_R^{1/2}
$$

라고 두면 trace의 순환성으로 다음 nonnegative form을 얻는다.

$$
T=\operatorname{Tr}(XX^\dagger)\ge0
$$

따라서 $T$는 contact weight까지 포함한 channel 사이 전파 진폭의 절댓값 제곱을 모두 더한 값이다. 이상적인 lead에 연결된 passive coherent device에서는 각 transmission eigenvalue가 0 이상 1 이하이다.

식 (18)의 행렬 순서는 한 contact에 결합해 device를 지나 반대 contact에 도달하는 과정과 그 conjugate를 곱한다는 뜻이다. trace는 두 contact에 연결된 모든 channel의 기여를 더한다.

현재의 single-channel 체인에서는

$$
\Gamma_L=\gamma_L\lvert1\rangle\langle1\rvert,
\qquad
\Gamma_R=\gamma_R\lvert N\rangle\langle N\rvert
$$

이므로 식 (18)이

$$
T(E)=\gamma_L\gamma_R\lvert G^R_{1N}\rvert^2
\tag{19}
$$

로 줄어든다. 두 $\gamma$는 lead flux에 맞게 incoming·outgoing channel을 정규화하고, $G^R_{1N}$은 두 끝점 사이의 propagation amplitude다. $G^R$의 단위가 $\mathrm{eV}^{-1}$이고 $\gamma$의 단위가 eV이므로 최종 $T$는 무차원이다.

```python
def retarded_greens_function(E, H, Sigma_L, Sigma_R, eta=1e-9):
    I = np.eye(H.shape[0], dtype=complex)
    return np.linalg.inv(
        (E + 1j * eta) * I - H - Sigma_L - Sigma_R
    )

def broadening(Sigma):
    return 1j * (Sigma - Sigma.conj().T)

def transmission(Gamma_L, GR, Gamma_R):
    GA = GR.conj().T
    return np.trace(Gamma_L @ GR @ Gamma_R @ GA).real
```

### 6.2 공간 potential과 Hamiltonian은 같은 정보가 아니다

transmission을 보기 전에 공간에 주어진 potential profile과 실제로 역행렬을 계산하는 Hamiltonian을 함께 그려 보자. 노트북에서는 깨끗한 device와 높이 $V_b=0.60\ \mathrm{eV}$, 폭 6개 사이트인 단일 장벽을 비교한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-05-potential-and-hamiltonian.png" alt="단일 장벽의 potential energy 분포와 그에 대응하는 삼중대각 수송 Hamiltonian" width="94%">
</p>

*그림 5.* 장벽은 선택한 대각 원소만 $2t_0$에서 $2t_0+V_b$로 올린다. 비대각 hopping $-t_0$는 그대로다. 주황색 $E_{\mathrm{probe}}=0.400\ \mathrm{eV}$는 장벽 내부 local band의 lower edge보다 낮으며, 뒤에서 evanescent tunneling을 공간적으로 볼 때 사용한다.

potential energy가 $U$로 일정한 국소 구간의 dispersion relation은 다음과 같다.

$$
E=U+2t_0(1-\cos ka)
$$

실수 $k$가 존재하는 에너지 범위는 다음과 같다.

$$
U\le E\le U+4t_0
$$

따라서 장벽 안에서 $E<V_b$이면 wavevector $k$가 복소수가 되면서 evanescent wave가 나타난다. 그렇다고 wavefunction이 장벽 안에서 0이 되는 것은 아니다. 유한한 evanescent region의 양쪽 끝에서는 propagating wave와 이어질 수 있다.

$k=i\kappa$로 두면 $E<U$인 sub-barrier regime에서 다음 관계를 얻는다.

$$
\cosh(\kappa a)=1+\frac{U-E}{2t_0}
$$

유한 장벽 안의 정확한 해는 decaying component 하나가 아니라 양쪽 끝의 boundary conditions에 따라 정해지는 growing·decaying evanescent components의 조합이다. 따라서 partial LDOS envelope는 사이트마다 완벽하게 단조로울 필요가 없다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-06-clean-wire-and-barrier-transport.png" alt="matched wire와 단일 장벽의 transmission 및 왼쪽 contact 기원 partial LDOS" width="94%">
</p>

*그림 6.* device와 lead가 완전히 같은 깨끗한 wire에는 반사를 만들 경계가 없으므로 open band 내부에서 $T(E)=1$이다. 장벽은 반사와 에너지 의존적 transmission, 공간적으로 달라지는 **왼쪽 contact 기원 partial LDOS** $[A_L]_{jj}/(2\pi)$를 만든다. 이는 $f_L$을 곱하기 전의 available spectral weight이며, 전체 occupied density가 아니다.

깨끗한 matched wire는 가장 먼저 통과해야 할 sanity check다. 계산이 맞다면 다음 값이 나와야 한다.

$$
\mathcal G_0=\frac{2e^2}{h}=77.4809\ \mu\mathrm{S},
\qquad
R_0=12.9064\ \mathrm{k}\Omega
$$

노트북은 band 안에서 택한 sample energies에 대해 $0.999999997\le T\le1.000000000$을 얻는다. self-energy 부호나 contact 끝점 index가 틀리면 이 검증부터 무너지는 경우가 많다. 정확한 band edge에서는 $\gamma\to0$인 특이한 limit을 따로 생각해야 한다.

conductance quantum은 transmission을 정규화하려고 임의로 끼워 넣는 상수가 아니다. electrochemical potential 차이가 $\Delta\mu=e\,\Delta V$로 작다면

$$
f_L(E)-f_R(E)
\simeq-\frac{\partial f}{\partial E}\,\Delta\mu
$$

로 선형화할 수 있다. zero-temperature limit에서는 $-\partial f/\partial E$가 $\delta(E-E_F)$가 된다. 7.1절의 식 (27)에서 나오는 Landauer current를 이 limit에서 선형화하면 다음 conductance를 얻는다.

$$
\mathcal G
=\left.\frac{dI}{dV}\right\rvert_{V=0}
=\frac{2e^2}{h}T(E_F)
$$

$2e^2/h$는 spin-degenerate single channel의 자연스러운 conductance scale이다. 깨끗한 matched wire가 그 값에 도달하는 까닭은 transmission을 편의상 normalization했기 때문이 아니라 실제 scattering probability가 1이기 때문이다.

### 6.3 장벽 tunneling을 공간에서 보면

rank-one인 왼쪽 contact에 대해 device에 projection한 contact-normalized amplitude를

$$
\lvert\psi_L(E)\rangle
=G^R(E)\sqrt{\gamma_L(E)}\lvert1\rangle
\tag{20}
$$

로 정의하자. 이는 unit norm으로 정규화한 bound-state wavefunction이 아니다. 단위는 $\mathrm{eV}^{-1/2}$이고

$$
A_L=\lvert\psi_L\rangle\langle\psi_L\rvert
$$

를 정확히 만족한다. 따라서

$$
\rho_{L,j}(E)=\frac{\lvert\psi_{L,j}(E)\rvert^2}{2\pi}
$$

는 스핀 하나당 왼쪽 contact 기원의 partial LDOS다. $j\to j+1$ bond를 지나는 directed spectral flux는 다음과 같이 정의한다.

$$
\mathcal J_{j\to j+1}^{(L)}(E)
=-2\operatorname{Im}
\left[H_{j,j+1}\psi_{L,j+1}\psi_{L,j}^*\right]
\tag{21}
$$

현재처럼 reciprocity를 갖는 single-channel model에서는 coherent steady state에서 이 값이 모든 내부 bond에서 같고 transmission과 일치한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-07-sub-barrier-scattering-state.png" alt="유한 장벽을 tunneling하는 scattering state의 partial LDOS와 보존되는 spectral flux" width="94%">
</p>

*그림 7.* $E_{\mathrm{probe}}=0.400\ \mathrm{eV}$에서 partial LDOS는 장벽을 지나며 크게 작아지지만, 오른쪽에도 0이 아닌 transmitted component가 남는다. spatial weight는 크게 달라져도 spectral flux는 $T=2.8572\times10^{-3}$으로 일정하다.

“입자가 장벽을 뚫고 간다”는 표현을 오해하지 않으려면 이 그림처럼 보아야 한다. stationary scattering state는 구조 전체에서 boundary condition을 만족하는 하나의 해다. 고전적으로 금지된 유한 구간에서는 진폭이 감쇠하지만, 이웃 site의 복소 진폭 사이에 남은 phase relation 덕분에 작지만 0이 아닌 conserved flux가 유지된다. equilibrium에서는 오른쪽에서 주입된 같은 크기의 flux가 이를 상쇄한다. 두 방향의 점유가 달라져야 실제 전류가 생긴다.

---

## 7. nonequilibrium에서는 occupation을 따로 계산해야 한다

$G^R$, $A_L$, $A_R$, $T$는 open-system spectrum과 propagation을 설명한다. Hamiltonian과 contact 구조가 고정되어 있다면 reservoir의 점유만 달리해도 이 양들은 바뀌지 않는다.

reservoir $\alpha$의 occupation은 Fermi function으로 다음과 같이 정한다.

$$
f_\alpha(E)=
\frac{1}{1+\exp[(E-\mu_\alpha)/(k_BT_\alpha)]}
\tag{22}
$$

equilibrium에서는 $f_L=f_R=f$라서 하나의 Fermi function이 contact continuum을 채운다. nonequilibrium device 내부에는 전체를 대표하는 하나의 Fermi function이 없다.

in-scattering self-energy는 다음과 같다.

$$
\Sigma^{\mathrm{in}}(E)
=\Gamma_L(E)f_L(E)+\Gamma_R(E)f_R(E)
\tag{23}
$$

이에 대응하는 electron correlation function은 다음과 같다.

$$
\begin{aligned}
G^n(E)
&=G^R\Sigma^{\mathrm{in}}G^A\\
&=A_Lf_L+A_Rf_R.
\end{aligned}
\tag{24}
$$

이 식은 random source의 covariance와 같은 형태로 보면 이해하기 쉽다. linear response가 $\lvert\psi\rangle=G^R\lvert s\rangle$이고, 서로 incoherent한 injection sources의 correlation이 $\langle ss^\dagger\rangle=\Sigma^{\mathrm{in}}$라면 다음 식이 나온다.

$$
\langle\psi\psi^\dagger\rangle
=G^R\Sigma^{\mathrm{in}}G^A
$$

NEGF에서는 이 관계를 Keldysh equation이라고 부른다.

이 글에서는

$$
G^n=-iG^<
$$

규약을 쓴다. 다른 책의 $G^<$ 식을 가져올 때 이 규약을 확인하지 않으면 밀도와 전류 부호가 쉽게 틀어진다.

세 양의 역할은 다음처럼 기억하면 된다.

```text
A_alpha(E): contact alpha가 공급할 수 있는 상태가 device 어디에 있는가
f_alpha(E): 그 contact가 그 상태를 얼마나 채우는가
G^n(E):     그 결과 실제로 점유된 spectral weight는 얼마인가
```

```python
def in_scattering(Gamma_L, Gamma_R, f_L, f_R):
    return Gamma_L * f_L + Gamma_R * f_R

def electron_correlation(GR, Sigma_in):
    return GR @ Sigma_in @ GR.conj().T

def bond_current_integrand(H, Gn, i):
    return -2.0 * np.imag(H[i, i + 1] * Gn[i + 1, i])
```

spin degeneracy 2를 포함한 site occupation은 다음과 같다.

$$
n_j=2\int\frac{dE}{2\pi}[G^n(E)]_{jj}
\tag{25}
$$

nearest-neighbor bond를 흐르는 current는 다음과 같다.

$$
I_{j\to j+1}
=-\frac{2e}{h}\int dE\,
2\operatorname{Im}
\left[H_{j,j+1}G^n_{j+1,j}(E)\right]
\tag{26}
$$

왜 대각 점유가 아니라 비대각 원소가 등장할까? 사이트 $j$의 입자수 연산자는 $\hat n_j=c_j^\dagger c_j$다. hopping Hamiltonian에 Heisenberg equation을 적용하면 lattice continuity equation

$$
\frac{d\langle\hat n_j\rangle}{dt}
+\mathcal J_{j\to j+1}
-\mathcal J_{j-1\to j}=0
$$

을 얻는다. $\hat n_j$와 commute하지 않는 항은 $j$를 이웃과 연결하는 hopping뿐이고, 그 expectation value는 $\langle c_j^\dagger c_{j+1}\rangle$ 같은 coherence다. 현재 규약에서는 이 coherence가 $G^n_{j+1,j}$에 들어 있다. site 사이의 phase difference는 그 허수부에 담기며, 바로 이 성분이 단순한 local occupation과 실제 flow를 구분한다.

그래서 $G^n$의 대각은 입자 점유를, 첫 비대각은 bond를 가로지르는 흐름을 준다. 각 site의 전하량은 이 occupation에 $-e$를 곱해 얻는다.

여기서 $\mathcal J$는 bond를 지나는 particle flux다. 여기에 전하와 부호 규약을 적용하면 식 (26)의 electrical current가 된다. 정확한 $\eta\to0^+$ coherent steady state에서는 내부 어느 사이트에도 입자가 계속 쌓이지 않는다. 따라서 continuity equation의 accumulation term은 0이 되고, 모든 내부 bond의 적분 전류가 같아야 한다.

다만 유한한 수치 $\eta$는 약한 distributed absorber처럼 작용하므로, 뒤의 수치 검증에서 보이는 정도의 작은 공간 편차는 생길 수 있다. 그 조절항의 효과보다 큰 편차가 보이면 에너지 해상도, index와 부호, 빠진 source·sink 항, 수치 오차를 확인해야 한다.

여기에는 서로 다른 두 개의 2가 있다. $2e/h$의 2는 spin degeneracy다. integrand 안의 2는 hopping 항과 그 Hermitian conjugate를 continuity equation에서 합칠 때 생긴다. 둘 다 spin 계수라고 생각하면 전류를 두 배로 잘못 계산한다.

### 7.1 electron correlation function에서 Landauer 식 유도하기

왼쪽 terminal current는 injection과 escape의 차이이며 다음과 같이 쓴다. 아래 식의 첫 항은 왼쪽 contact가 주입하는 occupied spectrum이고, 둘째 항은 이미 점유된 device spectrum이 그 contact로 빠져나가는 양이다.

$$
I_L=\frac{2e}{h}\int dE\,
\operatorname{Tr}
\left[\Sigma_L^{\mathrm{in}}(A_L+A_R)-\Gamma_LG^n\right]
$$

$\Sigma_L^{\mathrm{in}}=\Gamma_Lf_L$과 $G^n=A_Lf_L+A_Rf_R$를 넣으면 $A_Lf_L$ 항이 정확히 상쇄되고 다음 식만 남는다.

$$
I=\frac{2e}{h}\int dE\,
T(E)[f_L(E)-f_R(E)]
\tag{27}
$$

이것이 Landauer current다. transmission만으로는 전류가 아니다. transmission에 양쪽 incident state의 occupation 차이를 곱한 뒤 에너지에 대해 적분해야 한다.

노트북의 에너지 격자는 eV를 쓰고 $h$는 줄·초 단위이므로 수치 계수에는

$$
\frac{2e}{h}J_{\mathrm{per\,eV}},
\qquad
J_{\mathrm{per\,eV}}=1.602176634\times10^{-19}\ \mathrm{J/eV}
$$

가 들어간다. $e>0$는 기본전하의 크기다. 노트북은 $f_L>f_R$일 때 왼쪽에서 오른쪽으로 움직이는 **전자 흐름**을 양의 $I$로 잡는다. 부호 있는 전하의 관습적 전류 방향은 전자 전하가 $-e$이므로 반대다.

---

## 8. 예제 A: spectrum은 고정하고 두 reservoir의 occupation만 바꾸기

첫 resonant-tunneling 예제는 propagation과 occupation을 가장 깨끗하게 분리한다. 높이 $0.40\ \mathrm{eV}$인 두 장벽 사이에 16개 site로 이루어진 well이 있다. device Hamiltonian, 두 lead band, 모든 retarded self-energy는 고정한 채 $\mu_L$과 $\mu_R$만 하나의 resonance 양쪽에 놓는다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-08-example-a-double-barrier.png" alt="spectrum을 고정한 resonant-tunneling 예제의 이중 장벽 potential energy 분포" width="88%">
</p>

*그림 8.* 초록색 선은 $H_{jj}$가 아니라 $U_j$다. 실제 Hamiltonian 대각은 여기에 $2t_0$를 더한 값이다. open boundary condition은 device 내부에 임의의 complex potential을 넣는 방식이 아니라, device 양 끝의 self-energy로 구현된다.

### 8.1 닫힌 우물의 bound level이 open-system resonance로 바뀌는 과정

두 장벽이 무한히 높다면 우물에는 discrete bound levels가 있다. 장벽이 유한해지면 그 mode들이 왼쪽과 오른쪽 continuum에 결합한다.

isolated resonance $r$에 대응하는 closed-well orbital을 device 영역에서 normalize하고, 이를 $\lvert\phi_r\rangle$라고 하자. contact self-energy가 resonance width 안에서 천천히 변한다면 contact별 partial broadening을

$$
\Gamma_{\alpha,r}\simeq
\langle\phi_r\rvert\Gamma_\alpha(E_r)\lvert\phi_r\rangle,
\qquad \alpha=L,R
$$

로 근사할 수 있다. 같은 isolated-resonance approximation에서 analytic continuation한 device Green's function의 pole은 대략 다음과 같이 주어진다.

$$
\mathcal E_r\simeq E_r-\frac{i}{2}\Gamma_r,
\qquad
\Gamma_r\simeq\Gamma_{L,r}+\Gamma_{R,r}
$$

$E_r$는 contact self-energy의 real part 때문에 이동한 resonance energy다. 이 complex pole은 전체 device와 lead를 합친 Hermitian Hamiltonian에 새로 생긴 complex eigenvalue가 아니다. device에 projection한 open-system response를 complex energy로 analytic continuation했을 때 나타나는 pole이다.

self-energy가 빠르게 변하거나 resonance가 겹치면 이 단순한 projection approximation만으로는 충분하지 않다.

고립된 한 resonance 부근에서는 다음 Breit-Wigner form을 얻는다.

$$
T(E)\approx
\frac{\Gamma_{L,r}\Gamma_{R,r}}
{(E-E_r)^2+(\Gamma_r/2)^2}
\tag{28}
$$

resonance 중심에서 transmission의 최대값은

$$
T(E_r)=
\frac{4\Gamma_{L,r}\Gamma_{R,r}}
{(\Gamma_{L,r}+\Gamma_{R,r})^2}
$$

이므로 좌우 coupling이 대칭이면 각 장벽이 off-resonant energy에서 매우 불투명하더라도 $T(E_r)=1$에 도달할 수 있다. 우물 안의 다중 반사가 위상을 맞추는 Fabry-Pérot형 간섭이다.

resonance lifetime은 대략 다음과 같다.

$$
t_{\mathrm{life},r}\simeq\frac{\hbar}{\Gamma_r}
$$

$\Gamma_r$는 energy linewidth이고 escape rate는 $\Gamma_r/\hbar$다. 수명을 $t_{\mathrm{life},r}$로 쓴 것은 contact coupling matrix $\tau$와 기호가 겹치는 것을 피하기 위해서다.

노트북의 energy scan에서 서로 분리되어 보이는 resonance peak의 위치를 수치적으로 정밀화하면 다음 값을 얻는다.

$$
E_r\approx
0.0639,\ 0.1413,\ 0.2453,\ 0.3713\ \mathrm{eV}
$$

$0.5146\ \mathrm{eV}$ 부근에도 분리된 peak가 하나 더 보인다. 그중

$$
E_{\mathrm{res}}=0.1413\ \mathrm{eV},
\qquad
\mu_L=0.201\ \mathrm{eV},
\qquad
\mu_R=0.081\ \mathrm{eV},
\qquad
k_BT=0.005\ \mathrm{eV}=5\ \mathrm{meV}
$$

를 고른다. 마지막 값은 약 $58\ \mathrm{K}$에 해당하는 thermal-energy scale이다. 위 값들은 현재 energy scan에서 분명히 분리되는 peak의 위치만 정밀화한 결과다. 이보다 훨씬 좁은 다른 peak까지 모두 찾았다는 뜻은 아니다.

### 8.2 available spectrum과 occupied spectrum

우물 영역 $W$에 두 partial spectral function을 projection하면 다음과 같다.

$$
\rho_{W,L}(E)=\frac{1}{2\pi}\sum_{j\in W}[A_L(E)]_{jj},
\qquad
\rho_{W,R}(E)=\frac{1}{2\pi}\sum_{j\in W}[A_R(E)]_{jj}
$$

이 두 함수는 각각 왼쪽과 오른쪽 contact에서 우물로 들어올 수 있는 **available spectral states**를 나타낸다. 실제 occupied spectrum은 다음과 같다.

$$
f_L\rho_{W,L}+f_R\rho_{W,R}
$$

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-09-example-a-spectral-bookkeeping.png" alt="contact별로 나눈 available well spectrum과 실제 occupied well spectrum" width="94%">
</p>

*그림 9.* 양쪽 contact 모두 같은 고정 resonance에 state를 주입할 수 있다. chemical potential은 spectrum의 모양을 바꾸지 않고, 두 reservoir가 각 state를 얼마나 채우는지만 바꾼다. $G^n=A_Lf_L+A_Rf_R$의 의미를 가장 직접적으로 보여 준다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-10-example-a-transport-summary.png" alt="고정 spectrum resonant-tunneling 예제의 transmission, LDOS, contact별 점유, spectral current" width="96%">
</p>

*그림 10.* (a)는 고정된 transmission spectrum과 주된 occupation window를, (b)는 LDOS에 나타난 quasi-bound well mode를 보여 준다. (c)는 contact별 occupied spectral weight를 구분해 보여 준다. (d)는 전류가 transmission peak의 높이가 아니라 $dI/dE$의 **면적**으로 정해진다는 점을 보여 준다.

그 면적을 제대로 구하는 일도 물리 계산의 일부다. 균일한 energy grid는 linewidth가 매우 좁은 resonance를 놓치거나, 1차원 band edge에서 나타나는, 적분값은 유한하지만 매우 급하게 변하는 구조를 충분히 sampling하지 못할 수 있다.

노트북에서는 emitter band edge 부근의 geometric grid와 나머지 occupied-energy range의 촘촘한 linear grid를 이어 붙인다. 또 transmission의 각 peak candidate 주변을 별도로 탐색해 위치를 정밀화한다. plotting grid에서 가장 높은 점을 그대로 resonance energy로 읽지는 않는다.

비균일 격자 $E_0<\cdots<E_{M-1}$에서 사다리꼴 적분을 $\int dE\,F(E)\simeq\sum_iw_iF(E_i)$로 쓰면 가중치는 다음과 같다.

$$
w_0=\frac{E_1-E_0}{2},\qquad
w_i=\frac{E_{i+1}-E_{i-1}}{2},\qquad
w_{M-1}=\frac{E_{M-1}-E_{M-2}}{2}
$$

이를 코드에 명시하면 간격이 서로 다른 sample을 실수로 uniform grid의 값처럼 더하는 일을 막을 수 있다.

```python
E = np.unique(np.concatenate((
    np.geomspace(1e-10, 1e-3, 400, endpoint=False),
    np.linspace(1e-3, mu_L + 0.05, 3000),
)))
w = np.empty_like(E)
w[0], w[-1] = 0.5 * (E[1] - E[0]), 0.5 * (E[-1] - E[-2])
w[1:-1] = 0.5 * (E[2:] - E[:-2])

n_per_spin += w[iE] * np.diag(Gn).real / (2.0 * np.pi)
n_site = 2.0 * n_per_spin                 # spin degeneracy는 한 번만 적용
I = current_prefactor_A_per_eV * np.sum(w * T_E * (f_L - f_R))
```

여기서 `current_prefactor_A_per_eV`에는 spin 계수와 eV–joule 변환 계수가 이미 들어 있다. 여기에 2를 다시 곱하면 안 된다. 밀도, bond current, terminal current는 모두 같은 energy grid와 quadrature weight로 계산해야 한다. 그래야 세 결과의 차이를 적분 방식 때문이 아니라 수치 검증 오차로 해석할 수 있다.

서로 독립적인 두 전류 계산은

$$
I_{\mathrm{Landauer}}=263.0891\ \mathrm{nA}
$$

를 주고, 평균 bond current는 $263.0900\ \mathrm{nA}$, 공간 상대 편차는 $5.30\times10^{-5}$다. contact별 occupation을 더한 $n=n_L+n_R$ 관계도 상대오차 $3.0\times10^{-15}$ 이내에서 성립한다. 별도의 검증으로 occupied well spectrum의 적분과 우물 site occupation의 합을 비교하면, 이 spin-degenerate model에서 둘 다 $4.882464787$ 전자가 나온다.

예제 A의 핵심은 다음 두 줄로 요약할 수 있다.

```text
f_L과 f_R을 바꾸면 점유와 전류가 바뀐다.
H_D와 lead self-energy가 고정되어 있다면 G^R 자체는 바뀌지 않는다.
```

---

## 9. 예제 B: 전압에 따라 움직이는 resonance와 NDR

실제 전압은 점유 차이만 만드는 것이 아니라 single-particle potential을 기울이고 contact band도 이동시킬 수 있다. 예제 B에서는 가정한 선형 voltage drop $\phi$에 대응하는 electron potential energy $U=-e\phi$를 외부에서 정한 profile로 넣어 이 효과를 모사한다. Poisson equation을 self-consistent하게 푼 결과는 아니다.

예제 A와는 다른 90-site device를 사용한다. 그 안에는 높이 $0.45\ \mathrm{eV}$인 double barrier와 8-site well이 있다.

zero bias에서 서로 분리되어 보이는 resonance와 이 예제의 operating point는 다음과 같다.

$$
E_r(0)\approx0.0504,\ 0.1946,\ 0.4109\ \mathrm{eV}
$$

$$
E_F=E_1(0)-0.015\ \mathrm{eV}=0.0354\ \mathrm{eV},
\qquad
k_BT=0.003\ \mathrm{eV}\quad(T\approx35\ \mathrm{K})
$$

따라서 가장 낮은 resonance는 처음에 emitter Fermi level보다 $15\ \mathrm{meV}$ 위에 있다. 전압이 변하면 transport에 유리한 alignment에 들어왔다가 다시 벗어날 수 있다.

전자의 potential energy 이동은 다음과 같이 둔다.

$$
U_j^{\mathrm{bias}}(V)
=-eV\frac{j}{N-1}
$$

배열의 에너지 단위가 eV이므로 실제 수치 구현은 다음과 같다.

$$
U_j^{\mathrm{bias}}[\mathrm{eV}]
=-V[\mathrm{V}]\frac{j}{N-1}
\tag{29}
$$

eV 배열에 SI 기본전하를 다시 곱하면 단위가 틀어진다. 물리적인 에너지 식은 $\varepsilon_R(V)=\varepsilon_L-eV$, $\mu_R(V)=E_F-eV$다. 에너지에는 eV, $V$에는 volt 단위의 수치를 쓰면 다음과 같다.

$$
\varepsilon_R[\mathrm{eV}]=\varepsilon_L[\mathrm{eV}]-V[\mathrm{V}],
\qquad
\mu_R[\mathrm{eV}]=E_F[\mathrm{eV}]-V[\mathrm{V}],
\qquad
\mu_L[\mathrm{eV}]=E_F[\mathrm{eV}]
\tag{30}
$$

따라서 $\mu_R$과 이동한 오른쪽 lead band edge 사이의 상대 거리는 그대로다. 이는 bias를 중복해서 적용한 것이 아니다. $\varepsilon_R$은 collector에서 허용되는 band 전체를 이동시키고, $\mu_R$은 그 band의 occupation을 정하는 reservoir 기준을 이동시킨다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-11-example-b-bias-profile.png" alt="zero bias와 finite bias에서의 이중 장벽 potential energy 분포와 이동한 contact 기준" width="88%">
</p>

*그림 11.* 이 선형 경사는 Poisson equation으로 구한 결과가 아니라 처음부터 가정한 input profile이다. well과 barrier를 기울이는 동시에 collector의 energy reference도 아래로 이동시킨다.

### 9.1 삼중대각 구조를 쓰면 에너지·전압 한 점이 $O(N)$이다

voltage–energy map을 만들려면 $T(E,V)$를 매우 많이 계산해야 한다. dense matrix inversion은 $O(N^3)$이지만

$$
\mathcal A(E,V)
=(E+i0^+)I-H_D(V)-\Sigma_L^R-\Sigma_R^R
$$

는 삼중대각 행렬이다. 끝점 contact에 필요한 것은 $G^R_{1N}$ 하나뿐이다.

$$
\mathcal A\mathbf x=\mathbf e_N
$$

을 풀면 $x_1=G^R_{1N}$이므로 tridiagonal linear solver를 사용하면 $O(N)$ 연산으로 계산할 수 있다.

<details markdown="1">
<summary>코드: 삼중대각 선형계로 끝점 transmission 계산하기</summary>

```python
from scipy.linalg import solve_banded

def transmission_fast(E, N, t0, U, eps_L, eps_R, eta=1e-9):
    sigma_L = lead_self_energy_scalar(E, t0, eps_L, eta)
    sigma_R = lead_self_energy_scalar(E, t0, eps_R, eta)

    diag = ((E + 1j * eta) - (2.0 * t0 + U)).astype(complex)
    diag[0]  -= sigma_L
    diag[-1] -= sigma_R

    # H의 비대각이 -t0이므로 A의 비대각은 +t0이다.
    ab = np.zeros((3, N), dtype=complex)
    ab[0, 1:] = t0
    ab[1, :] = diag
    ab[2, :-1] = t0

    rhs = np.zeros(N, dtype=complex)
    rhs[-1] = 1.0
    last_column = solve_banded((1, 1), ab, rhs)

    gamma_L = -2.0 * sigma_L.imag
    gamma_R = -2.0 * sigma_R.imag
    return (gamma_L * gamma_R * abs(last_column[0])**2).real
```

</details>

$E=0.09\ \mathrm{eV}$에서 빠른 계산과 전체 밀집행렬 계산의 transmission 차이는 $4.61\times10^{-19}$에 불과하다.

### 9.2 전압을 높여도 전류가 줄어들 수 있는 이유

각 bias에서 전류는 여전히 다음 Landauer form으로 계산한다.

$$
I(V)=\frac{2e}{h}\int dE\,
T(E,V)\big[f_L(E)-f_R(E)\big]
\tag{31}
$$

$V$가 커지면 $f_L-f_R$가 0이 아닌 occupation window가 넓어져 전류를 늘리는 쪽으로 작용한다. 그러나 동시에 resonance-transmission ridge의 위치와 모양도 변한다. constant prefactor를 제외한 energy-resolved current integrand는 다음과 같다.

$$
\text{current integrand}
=T(E,V)[f_L(E)-f_R(E)]
$$

전류는 이 함수의 한 점 값이 아니라 에너지에 대한 적분값이다. 온도가 0일 때 에너지 적분에 실제로 기여할 수 있는 구간은 다음 세 구간의 교집합이다.

$$
[\mu_R,\mu_L]
\cap[E_{c,L},E_{c,L}+4t_0]
\cap[E_{c,R},E_{c,R}+4t_0]
$$

첫 구간은 두 reservoir의 occupation window이고, 나머지는 두 contact의 propagating band다. 아무리 높은 resonance peak라도 셋 가운데 하나를 벗어나면 steady-state DC current에 기여하지 못한다. 유한 온도에서는 occupation window의 경계가 부드러워지지만, 세 구간이 겹쳐야 한다는 조건은 바뀌지 않는다.

이 교집합을 보면 resonance가 occupation window를 벗어난 것인지, emitter band를 벗어난 것인지, collector band를 벗어난 것인지도 나누어 생각할 수 있다.

가장 낮은 resonance가 occupation window 안으로 들어와 emitter의 propagating state와 align될 때 전류가 커진다. bias를 더 올리면 resonance alignment가 다시 어긋나고, 고정된 emitter band edge 부근에서 emitter가 공급할 수 있는 propagating state도 줄어든다. 이 예제에서는 $E_{c,L}=0$이므로 band edge 부근에서

$$
\gamma_L(E)\simeq2\sqrt{t_0E}
$$

처럼 작아진다. 일반적으로는 $E$ 대신 $E-E_{c,L}$가 들어간다. 따라서 resonance가 완전히 band 아래로 내려가기 전부터 injection이 약해지고, 전압이 더 커졌는데도 current integrand의 적분 면적은 오히려 줄어들 수 있다.

그 구간에서는 다음 NDR condition이 성립한다.

$$
\frac{dI}{dV}<0
$$

이를 negative differential resistance (NDR)라 부른다. $I/V$ 자체가 음수라는 뜻은 아니다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-12-example-b-ndr-summary.png" alt="NDR 예제의 전류-전압 곡선, differential conductance, resonance-transmission map, peak와 valley의 spectral current" width="96%">
</p>

*그림 12.* 네 개의 main panel은 모두 가장 낮은 resonance가 만드는 첫 번째 resonance cycle을 보여 준다. 첫 current peak는 $0.085\ \mathrm{V}$, 뒤따르는 valley는 $0.149\ \mathrm{V}$이며 그 사이가 그림에서 강조한 첫 NDR 구간이다. 전체 sweep의 inset에는 약 $0.4\ \mathrm{V}$ 부근에서 훨씬 큰 두 번째 resonance cycle도 보이지만, 첫 cycle의 PVCR을 정의할 때는 사용하지 않았다.

정량 결과는 다음과 같다.

$$
I_{\mathrm{peak}}=57.3122\ \mathrm{nA},
\qquad
I_{\mathrm{valley}}=0.6519\ \mathrm{nA},
\qquad
\mathrm{PVCR}
=\frac{I_{\mathrm{peak}}}{I_{\mathrm{valley}}}
=87.92
$$

이 수치는 transmission map에서 pixel을 읽어 얻은 값이 아니다. heatmap은 resonance ridge의 움직임을 보여 주기 위한 그림이라 비교적 성긴 격자를 쓴다.

정량 전류에는 $dE=0.05\ \mathrm{meV}$의 에너지 간격을 쓰고, 첫 cycle 부근에는 $1\ \mathrm{mV}$의 전압 간격을 추가한다. 각 전압마다 device potential ramp, collector band reference, collector chemical potential을 해당 $V$에 맞춰 모두 갱신한다.

```python
dE = 5.0e-5                              # eV = 0.05 meV
E_current = np.arange(0.0, E_max + 0.5*dE, dE)
V_current = np.unique(np.round(
    np.r_[V_coarse, np.arange(0.04, 0.2001, 0.001)], 12
))
current = np.empty(V_current.size)

for iV, V in enumerate(V_current):
    U_V = U0 + electron_potential_ramp_eV(N, V)
    T_E = np.fromiter(
        (transmission_fast(E, N, t0, U_V, eps_L, eps_L - V)
         for E in E_current),
        dtype=float, count=E_current.size,
    )
    f_diff = fermi(E_current, E_F, kT) - fermi(E_current, E_F - V, kT)
    current[iV] = current_prefactor_A_per_eV * np.trapezoid(
        T_E * f_diff, E_current
    )
```

그림이 매끈해 보인다고 해서 linewidth가 좁은 resonance 아래 면적까지 수렴했다는 뜻은 아니다. 그래서 그림용 grid와 적분용 grid를 분리해야 한다. 노트북에서는 수렴이 확인된 current array를 미분하고 peak-prominence 조건으로 작은 수치 요동을 걸러 낸 뒤 첫 peak와 그 뒤 valley를 정한다.

(d)는 peak와 valley에 똑같은 절대 $dI/dE$ scale을 사용한다. 녹색 valley 곡선이 거의 보이지 않는 까닭은 실제 적분 면적이 훨씬 작기 때문이다. 두 곡선을 따로 정규화해서 생긴 차이가 아니다.

이 모형에서 NDR이 나타나는 까닭은 resonance alignment가 흐트러지고 emitter band edge 부근에서 emitter가 공급할 수 있는 propagating state가 줄어들기 때문이다. 여기서는 외부에서 정한 선형 potential ramp와 1차원 single band를 사용했으므로, 이 결과를 모든 resonant-tunneling diode에 공통된 microscopic mechanism으로 일반화해서는 안 된다.

실제 device에서는 전하 재분포, self-consistent electric field, scattering, transverse mode, nonparabolic band, hysteresis가 중요할 수 있다.

---

## 10. 수치 검증도 유도의 일부다

NEGF 식은 짧아서 코드가 오류 없이 실행되더라도 물리는 틀릴 수 있다. 그래서 이 노트북에서는 identity check와 limiting-case test를 부록으로 미루지 않고 계산 과정에 포함했다.

- **closed-chain spectrum:** analytic expression과 diagonalization 결과가 $1.33\times10^{-15}\ \mathrm{eV}$까지 일치한다.
- **closed-chain eigenstate:** $\lVert H\psi_n-E_n\psi_n\rVert<5.5\times10^{-16}\ \mathrm{eV}$다.
- **정확한 lead 소거:** Schur complement와 전체 역행렬의 device block 차이가 $3.48\times10^{-15}\ \mathrm{eV}^{-1}$다.
- **surface recursion:** quadratic-equation residual이 $2.22\times10^{-16}$이다.
- **retarded root:** $\operatorname{Im}g_s^R<0$이고 $q$의 기준점이 $1,i,-1$이며 band 밖에서 감쇠한다.
- **matched wire:** propagating band 안에서 $T(E)=1$이다.
- **single barrier:** $\operatorname{Tr}(\Gamma_RA_L)$, endpoint transmission formula, bond flux가 일치한다.
- **spectral identity:** 유한 $\eta$의 $2\eta G^RG^A$ 항까지 포함하면 상대오차 $1.77\times10^{-16}$이다.
- **equilibrium:** contact continuum에서 $G^n=f(A_L+A_R)$가 상대오차 $7.26\times10^{-17}$로 맞는다.
- **terminal current:** terminal formula가 Landauer integrand로 줄어드는 상대오차가 $1.67\times10^{-16}$이다.
- **예제 A:** bond current가 보고한 유한-$\eta$ 편차 안에서 거의 일정하고 독립적인 Landauer integral과 일치한다.
- **fast solver:** $O(N)$ endpoint calculation이 dense-matrix trace와 일치한다.
- **NDR 적분:** peak와 valley의 $dI/dE$ 면적이 표시한 전류를 그대로 재현한다.

각 검증은 서로 다른 오류를 잡는다. matched-wire 검증이 실패하면 contact matching이나 부호를 의심해야 한다. Schur 검증이 실패하면 block 순서와 Hermitian conjugate를 확인해야 한다. bond마다 전류가 달라지면 $G^n$의 index, bond 방향, 에너지 해상도, spin 계수 중복을 살펴봐야 한다. “코드가 끝까지 돌았다”는 하나의 검증으로 이들을 대신할 수 없다.

---

## 11. 이 단순한 모형의 한계

앞에서 유도한 lead elimination과 surface Green's function 식은 지금의 single-particle lattice model에 대해 근사 없이 성립한다. 다만 뒤의 해석과 예제에는 isolated-resonance approximation이나 외부에서 정한 bias profile처럼, 본문에서 따로 밝힌 근사와 modeling assumption이 들어간다. 이 모형이 담을 수 있는 물리에도 분명한 한계가 있다.

- 명시적인 electron-electron·electron-phonon self-energy를 포함하지 않는 single-particle description이며, mean-field 수준에 가깝다.
- phase coherence가 유지되는 elastic transport만 다룬다.
- lead는 균일하고 single-channel이며 semi-infinite다.
- 유효질량은 일정하고 격자는 1차원이다.
- spin은 degeneracy factor 2로만 들어가며 spin-orbit coupling이나 magnetic structure는 다루지 않는다.
- 예제 B는 NEGF–Poisson을 self-consistent하게 풀지 않고 선형 potential-energy drop을 가정한다.
- 두 lead continuum에서 분리된 진짜 bound state의 점유를 정하려면 별도의 초기 상태 준비 방식이나 relaxation mechanism이 필요하다.

NEGF의 틀 자체는 이보다 훨씬 넓다. multi-orbital device에서는 single-orbital contact에서 scalar였던 surface self-energy가 matrix가 되고, scattering은 추가 retarded·in-scattering self-energy로 들어간다. self-consistent electric field를 구할 때는 $G^n$에서 얻은 density로 $H_D$를 갱신한다. 모형을 이렇게 확장해도 핵심 구조는 그대로다.

```text
retarded self-energy는 열린 spectrum과 이탈을 정한다.
in-scattering self-energy는 그 spectrum의 점유를 정한다.
Green's function은 두 정보를 유한한 device 안으로 전파한다.
```

이 계산에서 가장 중요하게 기억할 점은 이것이다. NEGF는 서로 무관한 공식을 한데 모은 방법이 아니다. boundary condition, available state, reservoir occupation, conserved flow를 각각 구분해 계산한 뒤, 마지막에 observable을 만들 때 한데 묶는 체계다.

---

## 12. 계산을 직접 재현하려면

모든 그림과 수치 검증을 재현할 수 있는 노트북은 다음 두 링크에서 볼 수 있다.

- [Kaggle에서 노트북 실행하기](https://www.kaggle.com/code/pilkwang/from-closed-to-open-quantum-transport-with-negf)
- [GitHub에서 소스 노트북 읽기](https://github.com/pilkwangkim/Physics/blob/master/negf_from_scratch.ipynb)

먼저 이 글에서 수식의 흐름을 익힌 뒤, 각 절에 맞춰 노트북 cell을 실행하면서 수치 결과를 확인해 보길 권한다. 그다음에는 격자 간격, 장벽 모양, contact hopping, reservoir chemical potential, bias profile 가운데 하나만 바꿔 보자. 앞에서 사용한 검증 항목을 그대로 적용하면 새로운 물리 효과와 단순한 규약 오류를 훨씬 쉽게 구분할 수 있다.
