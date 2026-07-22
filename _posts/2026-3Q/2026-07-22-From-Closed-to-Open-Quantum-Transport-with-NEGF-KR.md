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

유한한 Hamiltonian을 대각화하면 energy level과 eigenstate는 구할 수 있다. 그러나 이것만으로는 steady-state current를 계산할 수 없다. 수송 device는 열린계이기 때문이다. 전자는 한쪽 reservoir에서 들어와 device를 지난 뒤 다른 reservoir로 빠져나가야 하며, 계산 영역의 끝에서 반사되어 돌아오면 안 된다.

이 글에서는 다음 문제를 다룬다.

```text
무한히 긴 lead가 달린 열린 수송 문제를,
임의의 absorbing boundary condition 없이 어떻게 유한 행렬로 정확히 표현할까?
```

nonequilibrium Green's function (NEGF)에서는 lead의 **자유도**를 식에서 근사 없이 소거한다. 그렇다고 lead의 영향까지 없애는 것은 아니다. 그 영향은 energy-dependent self-energy로 바뀌어 device 방정식에 남는다.

이 글에서는 이 과정에서 transmission, local density of states (LDOS), nonequilibrium occupation, 전류 보존, resonant tunneling, negative differential resistance (NDR)가 어떻게 차례로 이어지는지 설명한다.

노트북에서 생략하거나 간략히 제시한 유도는 이 글에서 한 단계씩 풀어 쓴다. wavefunction, Hamiltonian, eigenvalue problem, tight-binding band의 기본 개념만 알면 읽을 수 있다. Green's function이나 양자 수송을 미리 공부해 둘 필요는 없다.

계산 과정을 처음부터 끝까지 직접 확인할 수 있도록 단순한 모형 하나만 사용한다.

- 유효질량 전자를 균일 격자에 놓은 1차원 모형
- nearest-neighbor hopping과 spin degeneracy 2
- coherent·elastic single-particle transport
- 균일한 semi-infinite lead
- Poisson equation을 self-consistent하게 푼 결과 대신 외부에서 정한 bias profile

가정을 단순화했기 때문에 각 식의 유도 과정을 직접 확인할 수 있다. 다만 이 모형으로 설명할 수 있는 현상은 제한적이다.

---

## 0. 전체 계산 순서

세부 식을 유도하기 전에 전체 계산 순서를 정리한다.

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

먼저 lead, reservoir, contact를 구분한다.

1. **lead**는 phase coherence가 유지되는 semi-infinite 양자계다. lead Hamiltonian으로 propagating mode를 구하고, retarded solution에는 outgoing-wave boundary condition을 적용한다.
2. **reservoir**는 chemical potential과 온도를 통해 incident state를 얼마나 채울지 정한다.
3. 실제 **contact**는 두 기능을 함께 맡는다. NEGF에서는 전파와 이탈을 retarded self-energy로 계산하고, occupation은 Fermi function으로 따로 계산한다.

여기서 $E+i\eta$의 작은 양수 $\eta$는 retarded boundary value를 선택하기 위해 넣은 수치적 조절항이다. resonance의 물리적 lifetime과는 관계가 없다. contact가 만드는 실제 broadening은 self-energy의 anti-Hermitian part로 정해진다.

### 0.1 앞으로 사용할 기호

뒤에서 반복해서 사용할 기호와 각각의 역할은 다음과 같다.

- $H_D$: lead를 연결하기 전 유한 device의 single-particle dynamics를 정하는 Hamiltonian
- $g_\alpha^R$: device와 연결하지 않은 **isolated lead** $\alpha$의 Green's function
- $\Sigma_\alpha^R$: lead $\alpha$를 소거한 뒤 device 경계에 남는 energy-dependent self-energy
- $G^R$: lead를 연결한 **open device**의 retarded Green's function
- $\Gamma_\alpha$: contact $\alpha$를 통한 escape coupling의 세기
- $A_\alpha$: contact $\alpha$에서 들어오는 scattering state가 device 안에 만드는 spectral weight
- $f_\alpha$: reservoir $\alpha$가 incident state를 채우는 비율
- $G^n$: 실제로 점유된 spectral weight를 나타내는 electron correlation function
- $T(E)$: incident flux 가운데 반대편 lead까지 전달되는 비율
- $I$: 양쪽 reservoir의 occupation 차이를 에너지에 대해 적분해 얻는 순전류

대소문자도 구분해야 한다. $\Gamma_\alpha$는 device subspace에 작용하는 행렬이다. 이 single-channel model에서는 0이 아닌 원소가 끝점에 하나뿐이며, 그 값을 $\gamma_\alpha$로 쓴다. $T(E)$는 언제나 transmission을 뜻한다. reservoir의 온도는 $T_\alpha$, 그에 해당하는 에너지 척도는 $k_BT_\alpha$로 적는다.

---

## 1. Schrödinger 방정식을 tight-binding 행렬로 옮기기

먼저 1차원 effective-mass Schrödinger equation을 쓰자.

$$
-\frac{\hbar^2}{2m^*}\frac{d^2\psi}{dx^2}+U(x)\psi(x)=E\psi(x).
$$

$x_j=ja$에 격자점을 놓고 2차 미분을 central finite difference로 바꾸면 다음 식을 얻는다.

$$
\left.\frac{d^2\psi}{dx^2}\right\rvert_{x_j}
\approx
\frac{\psi_{j+1}-2\psi_j+\psi_{j-1}}{a^2}
$$

이제 양의 운동에너지 척도

$$
t_0\equiv\frac{\hbar^2}{2m^*a^2}
$$

를 정의하면 각 격자점에서

$$
-t_0\psi_{j-1}+(2t_0+U_j)\psi_j-t_0\psi_{j+1}=E\psi_j
\tag{1}
$$

를 얻는다. 형태만 보면 nearest-neighbor tight-binding model과 같다. 그러나 여기서 $t_0$는 경험적으로 넣은 hopping parameter가 아니다. continuum kinetic-energy operator를 finite difference로 바꾸면 nearest-neighbor coupling이 자연스럽게 생긴다.

site basis에서 Hamiltonian의 matrix element는 다음과 같다.

$$
H_{jj}=2t_0+U_j,
\qquad
H_{j,j+1}=H_{j+1,j}=-t_0
$$

여기서 $U_j$와 $H_{jj}$는 서로 다르다. 물리적인 potential energy는 $U_j$이고, Hamiltonian의 diagonal element에는 discrete kinetic-energy operator에서 생기는 $2t_0$가 함께 들어간다.

노트북에서 사용한 $m^*=0.067m_0$, $a=1\ \mathrm{nm}$를 대입하면 다음 값을 얻는다.

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

배열의 자료형을 complex로 지정해도 closed-system Hamiltonian은 여전히 Hermitian이다. 뒤에서 complex self-energy를 더할 때 자료형을 다시 바꾸지 않도록 처음부터 복소수 배열로 만든 것이다.

격자 간격 $a$는 단순한 수치 해상도가 아니라 모형을 정하는 물리적 parameter다. $t_0\propto a^{-2}$이므로 $a$를 바꾸면 lattice bandwidth와 $U/t_0$ 같은 무차원 비도 모두 달라진다.

수렴성을 확인하려면 device의 **실제 길이와 장벽의 실제 폭**을 고정해야 한다. 그런 다음 더 촘촘한 격자에서 $U_j$를 다시 만들고, 관심 있는 관측량이 더 이상 변하지 않는지 확인한다. site 수를 그대로 둔 채 $a$만 절반으로 줄이면 정확도만 높아지는 것이 아니라 device의 실제 길이도 절반으로 줄어든다.

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

band의 아래끝은 0이고 위끝은 $4t_0$다. $\lvert ka\rvert\ll1$에서는

$$
1-\cos ka\simeq\frac{(ka)^2}{2}
$$

이므로

$$
E(k)\simeq t_0(ka)^2=\frac{\hbar^2k^2}{2m^*}
$$

가 되어 continuum limit의 parabolic dispersion과 일치한다.

Brillouin zone 끝에서 lattice dispersion이 포물선과 달라지는 이유는 실제 물질의 nonparabolicity 때문이 아니다. finite-difference lattice가 만드는 dispersion error다. 따라서 관심 에너지 범위에서 parabolic approximation이 충분히 정확한지 $a$를 줄여 가며 확인해야 한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-01-lattice-dispersion.png" alt="finite-difference lattice의 dispersion relation과 hard-wall 유한 체인의 discrete spectrum" width="94%">
</p>

*그림 1.* $k=0$ 부근에서는 lattice dispersion과 continuum의 포물선이 일치한다. 그러나 lattice band의 폭은 $4t_0$로 유한하다. hard-wall 유한 chain에서는 boundary condition을 만족하는 discrete wavevector만 허용된다.

---

## 2. 닫힌 chain에서 알 수 있는 것과 없는 것

$N$개의 물리적 격자점 바깥에 ghost site를 하나씩 두고

$$
\psi_0=\psi_{N+1}=0
$$

라는 hard-wall boundary condition을 적용하자. 즉 wavefunction은 양쪽 경계에서 0이다. 이때 허용되는 mode와 energy level은 다음과 같다.

$$
k_na=\frac{n\pi}{N+1},
\qquad
\psi_n(j)=\sqrt{\frac{2}{N+1}}
\sin\!\left(\frac{n\pi j}{N+1}\right)
$$

$$
E_n=2t_0\left[1-\cos\!\left(\frac{n\pi}{N+1}\right)\right]
$$

노트북에서 Hamiltonian을 직접 대각화한 결과도 analytic solution과 일치한다. analytic eigenvalue와 numerical eigenvalue의 최대 차이는 $1.33\times10^{-15}\ \mathrm{eV}$이고, eigenvector residual은 $\lVert H\psi_n-E_n\psi_n\rVert<5.5\times10^{-16}\ \mathrm{eV}$다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-02-closed-chain-eigenstates.png" alt="hard-wall 유한 tight-binding 체인의 처음 다섯 eigenstate를 각 eigenenergy 주변에 표시한 그림" width="90%">
</p>

*그림 2.* 각 eigenfunction은 해당 eigenenergy를 기준으로 위아래로 옮겨 그렸다. 곡선의 세로 변위는 energy uncertainty나 probability density를 뜻하지 않는다. 두 hard-wall boundary 사이의 실제 potential은 $U_j=0$이다.

이 계산으로 Hamiltonian 구현은 검증할 수 있다. 그러나 reservoir와 open boundary가 없으므로 아직 수송 문제라고 할 수는 없다.

- 왼쪽 reservoir가 공급할 right-moving incident state의 occupation을 정하지 않았다.
- 오른쪽 reservoir가 공급할 left-moving incident state의 occupation도 정하지 않았다.
- 양쪽 끝에는 확률이 빠져나갈 open boundary가 없다.
- spectrum은 discrete stationary states로만 이루어진다.

이 Hamiltonian은 실수이고 time-reversal symmetry를 가지므로 hard-wall eigenstate도 실수로 선택할 수 있다. 그러면 nearest-neighbor bond를 지나는 probability current는 0이다. chain의 길이를 늘려도 boundary condition은 그대로다. 반사가 돌아오는 시간이 길어지고 standing-wave level spacing이 줄어들 뿐이다.

따라서 수송을 계산하려면 행렬을 키우는 대신 **boundary condition**을 바꿔야 한다.

---

## 3. Green's function은 source에 대한 응답을 나타낸다

Schrödinger equation에 source term을 넣으면 다음과 같은 linear equation이 된다.

$$
\big[(E+i\eta)I-H\big]\lvert\psi\rangle=\lvert s\rangle,
\qquad \eta>0.
$$

$\lvert s\rangle$는 응답을 정의하기 위해 도입한 수학적 source다. reservoir가 실제로 state를 채우는 과정을 뜻하지는 않는다. 이 방정식의 해는 retarded Green's function을 사용해

$$
\lvert\psi\rangle=G^R(E)\lvert s\rangle,
\qquad
G^R(E)=\big[(E+i\eta)I-H\big]^{-1}
\tag{3}
$$

로 쓸 수 있다. $G^R_{ab}$는 $b$에 unit source를 놓았을 때 $a$에 생기는 진폭을 나타낸다. 역행렬을 취하기 전 operator가 에너지 단위를 가지므로 $G^R$의 단위는 $\mathrm{eV}^{-1}$다.

closed-system Hamiltonian의 eigenstate를 쓰면 다음 spectral representation을 얻는다.

$$
G^R(E)=\sum_n\frac{\lvert n\rangle\langle n\rvert}{E-E_n+i\eta}
\tag{4}
$$

$E$가 $E_n$에 가까워지면 해당 eigenstate의 항이 커진다. spectral function은

$$
A(E)=i(G^R-G^A),
\qquad G^A=G^{R\dagger}
$$

으로 정의하며, $G^A$는 advanced Green's function이다. spectral function은 complex-energy plane의 pole structure를 실수 에너지축 위의 spectral weight로 나타낸다. spin 하나당 LDOS는

$$
\rho_j(E)=\frac{A_{jj}(E)}{2\pi}
$$

이다. eigenvalue만 나열하면 각 state의 공간 분포는 알 수 없다. 반면 spectral representation에는 eigenstate마다 다음 projector가 함께 들어 있다.

$$
P_n=\lvert n\rangle\langle n\rvert
$$

$P_n$으로 mode $n$의 공간 형태를 알 수 있고, 식 (4)의 분모에서는 그 mode의 응답이 **어느 에너지에서** 커지는지 알 수 있다. 따라서 Green's function을 이용하면 eigenvalue뿐 아니라 원하는 probe energy에서의 공간 응답도 계산할 수 있다.

lead를 연결해도 analytic structure 자체는 유지된다. 다만 closed-system level의 real-axis pole은 대개 complex-energy plane으로 이동한다. pole의 실수부는 resonance energy를 정하고, 음의 허수부는 contact로 빠져나가는 decay rate와 연결된다.

pole과 peak는 같은 개념이 아니다. pole은 $G^R$를 complex-energy plane으로 analytic continuation했을 때 나타나는 singularity이고, peak는 유한한 해상도로 실수 에너지축을 scan했을 때 관찰되는 극댓값이다.

resonance가 서로 충분히 떨어져 있고 그 주변에서 self-energy가 천천히 변한다면 peak 위치는 complex pole의 실수부와 거의 일치한다. resonance가 겹치거나 band threshold가 가깝거나 self-energy가 빠르게 변할 때는 두 값이 달라질 수 있다.

고립된 유한계의 spectrum은 $\eta\to0^+$ 극한에서 delta peak의 합이 된다. 그림에서는 유한한 $\eta$를 쓰기 때문에 peak에 폭이 생기지만, 이는 수치적으로 넣은 폭일 뿐 물리적 lifetime과는 관계가 없다.

$E-i\eta$가 아니라 $E+i\eta$를 쓰는 이유도 boundary condition에 있다. 시간영역에서 retarded solution은 source를 가한 뒤의 응답만 남긴다. lead에서는 이 조건에 따라 device에서 멀어지는 outgoing wave를 고르고, 무한대에서 들어오는 incoming wave는 제외한다.

---

## 4. lead 자유도를 정확히 소거해 열린계를 만든다

큰 block matrix를 다루기 전에 두 block만 있는 경우로 elimination의 구조를 확인한다.

$$
a_D\psi_D-\tau\psi_L=s_D,
\qquad
-\tau^\dagger\psi_D+a_L\psi_L=0.
$$

둘째 식에서 $\psi_L=a_L^{-1}\tau^\dagger\psi_D$를 구해 첫째 식에 대입하면

$$
\big(a_D-\tau a_L^{-1}\tau^\dagger\big)\psi_D=s_D
$$

라는 effective equation이 나온다. $\psi_L$ 자체는 식에서 사라졌지만, lead의 영향은 $\tau a_L^{-1}\tau^\dagger$ 항으로 남는다. 이 대입을 block matrix 전체에 적용한 것이 Schur complement다.

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

소문자 $g_L^R$는 device를 연결하기 전 isolated lead의 Green's function이다. 이미 device가 연결된 전체 Green's function의 $G_{LL}^R$ block과는 다르다. $G_{LL}^R$에는 device를 거쳐 lead로 돌아오는 과정까지 포함된다. 한편 self-energy는 device space에 작용하므로 행렬 크기는 다음과 같이 $N_D\times N_D$가 되어야 한다.

$$
(N_D\!\times\!N_L)\,
(N_L\!\times\!N_L)\,
(N_L\!\times\!N_D)
\longrightarrow N_D\!\times\!N_D
$$

단위도 $(\mathrm{eV})(\mathrm{eV}^{-1})(\mathrm{eV})=\mathrm{eV}$로 맞는다. 따라서 $\Sigma_L^R$는 device space에서 정의되며 $H_D$와 마찬가지로 에너지 단위를 갖는 행렬이다.

device에만 수학적 source를 두면 block equation은

$$
\begin{aligned}
\big[(E+i\eta)I_D-H_D\big]\psi_D-\tau\psi_L&=s_D,\\
-\tau^\dagger\psi_D+
\big[(E+i\eta)I_L-H_L\big]\psi_L&=0.
\end{aligned}
\tag{6}
$$

이다. 둘째 줄을 $\psi_L$에 대해 풀면 다음 식을 얻는다.

$$
\psi_L=g_L^R\tau^\dagger\psi_D.
\tag{7}
$$

이를 첫째 줄에 대입하면

$$
\left[(E+i\eta)I_D-H_D-
\underbrace{\tau g_L^R\tau^\dagger}_{\Sigma_L^R(E)}
\right]\psi_D=s_D
\tag{8}
$$

가 된다. 따라서 lead self-energy를

$$
\boxed{\Sigma_L^R(E)=\tau g_L^R(E)\tau^\dagger}
\tag{9}
$$

로 정의한다. 행렬곱은 오른쪽부터 읽는다. 먼저 $\tau^\dagger$가 device 경계의 진폭을 lead로 보낸다. 이어서 $g_L^R$가 lead 안의 전파를 계산하고, 마지막으로 $\tau$가 진폭을 device로 되돌린다. 따라서 self-energy 한 항이 device와 lead 사이의 한 차례 왕복을 나타낸다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-03-exact-lead-elimination.png" alt="lead를 정확히 소거해 contact site에 energy-dependent self-energy를 남기는 과정과 surface의 자기유사성" width="96%">
</p>

*그림 3.* (a)는 device와 lead의 자유도를 모두 포함한 행렬이다. (b)는 lead 자유도를 소거하고 그 영향을 $\Sigma_L^R$로 표현한 식이다. (c)에서는 surface site를 하나 떼어 내도 같은 semi-infinite chain이 남으며, 이 자기유사성에서 surface recursion relation이 나온다.

여기서 exact elimination은 다음을 뜻한다.

- 식 (8)의 역행렬은 전체 energy-domain Green's function에서 **device block**만 뽑은 것과 정확히 같다.
- 그렇다고 전체 Hermitian Hamiltonian 자체가 더 작은 non-Hermitian Hamiltonian으로 바뀌는 것은 아니다.
- 앞에서 정한 single-particle Hamiltonian과 subspace 분할만 놓고 보면 block elimination 과정에는 근사가 없다.
- 소거된 lead도 자체 동역학을 가지므로 self-energy는 에너지에 따라 달라진다.
- open channel로 진폭이 빠져나갈 수 있으므로 device의 effective operator는 non-Hermitian이 될 수 있다.

같은 결과를 device와 lead 사이를 여러 번 왕복하는 과정으로도 해석할 수 있다. lead를 연결하기 전 device의 resolvent를

$$
g_D^R=[(E+i0^+)I_D-H_D]^{-1}
$$

라고 하고, 모든 lead의 self-energy를

$$
\Sigma^R\equiv\Sigma_L^R+\Sigma_R^R
$$

로 묶자. Dyson equation은

$$
G^R=g_D^R+g_D^R\Sigma^RG^R
$$

이다. 이를 급수로 전개하면 다음과 같다.

$$
G^R
=g_D^R
+g_D^R\Sigma^Rg_D^R
+g_D^R\Sigma^Rg_D^R\Sigma^Rg_D^R+\cdots
$$

$\Sigma^R$가 한 번 더 들어갈 때마다 진폭이 device에서 lead로 나갔다 돌아오는 왕복도 한 번 추가된다. 따라서 lead를 소거한다는 말은 lead 방문 횟수를 제한한다는 뜻이 아니다. lead의 좌표를 행렬에 직접 저장하지 않은 채 모든 왕복 과정을 합한다는 뜻이다.

양쪽 lead를 모두 소거하고 나면 실제로 역행렬을 구할 대상은 다음 device 크기의 행렬뿐이다.

$$
G^R(E)=
\left[
(E+i0^+)I_D-H_D-\Sigma_L^R(E)-\Sigma_R^R(E)
\right]^{-1}
\tag{10}
$$

노트북에서는 유한 lead까지 포함한 전체 행렬의 역행렬을 직접 구한 뒤, 그 device block을 Schur complement 결과와 비교한다. 두 값의 최대 차이는 $3.48\times10^{-15}\ \mathrm{eV}^{-1}$다.

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

## 5. 무한한 lead를 surface Green's function 하나로 나타낼 수 있는 이유

single-orbital contact에서는 device 끝의 site $\lvert d_0\rangle$와 lead 표면의 site $\lvert\ell_0\rangle$만 서로 연결된다.

$$
\tau=-t_0\lvert d_0\rangle\langle\ell_0\rvert.
$$

이를 self-energy 식에 대입하면 device에 작용하는 다음 연산자가 나온다.

$$
\Sigma_L^R
=t_0^2\lvert d_0\rangle
\underbrace{\langle\ell_0\rvert g_L^R\lvert\ell_0\rangle}_{g_s^R(E)}
\langle d_0\rvert
$$

처음에는 무한 행렬 $g_L^R$를 통째로 알아야 할 것처럼 보인다. 하지만 $\tau$는 device와 lead의 표면 site만 연결하므로, $\tau g_L^R\tau^\dagger$에 실제로 들어가는 값은

$$
g_s^R(E)=\langle\ell_0\rvert g_L^R(E)\lvert\ell_0\rangle
$$

하나뿐이다. lead 안쪽의 모든 자유도는 이 하나의 surface-to-surface matrix element에 이미 반영되어 있다.

이제 lead를 site energy가 $\varepsilon$이고 nearest-neighbor hopping이 $-t_0$인 균일한 semi-infinite chain으로 두자. 지금 쓰는 finite-difference Hamiltonian에서는 $\varepsilon=2t_0+U_{\mathrm{lead}}$다. 따라서 $U_{\mathrm{lead}}=0$인 깨끗한 lead의 band는 1절에서 구한 것처럼 $[0,4t_0]$이다.

lead를 surface site $\ell_0$와 $\ell_1$에서 시작하는 나머지 chain으로 나누자. 나머지 부분은 site 번호만 하나 옮겨졌을 뿐 원래 lead와 완전히 같은 semi-infinite chain이다. 그러므로 $\ell_1$에서 본 surface Green's function 역시 $g_s^R$다.

$\ell_1$부터 이어지는 부분을 정확히 소거하면 $\ell_0$에는 $t_0^2g_s^R$라는 self-energy가 생긴다. 따라서 $g_s^R$는 다음 self-consistency equation을 만족한다.

$$
g_s^R(E)=
\frac{1}{E+i0^+-\varepsilon-t_0^2g_s^R(E)}
\tag{11}
$$

이 관계를 계속 대입하면 다음 continued fraction이 된다. 각 denominator 뒤에 같은 구조가 반복됨을 바로 확인할 수 있다.

$$
g_s^R=
\cfrac{1}{E^+-\varepsilon-
\cfrac{t_0^2}{E^+-\varepsilon-
\cfrac{t_0^2}{\ddots}}}.
$$

식 (11)을 정리하면 $g_s^R$에 대한 quadratic equation이 나온다.

$$
t_0^2(g_s^R)^2-
(E+i0^+-\varepsilon)g_s^R+1=0
$$

그 두 root는 다음과 같다.

$$
g_\pm(E)=
\frac{E+i0^+-\varepsilon
\ \pm\ \sqrt{(E+i0^+-\varepsilon)^2-4t_0^2}}
{2t_0^2}
\tag{12}
$$

quadratic equation에는 두 root가 있지만, retarded boundary condition을 만족하는 root는 하나뿐이다. $g_s^R$로 선택할 branch는 다음 세 조건을 모두 만족해야 한다.

1. complex-energy plane의 upper half-plane에서 analytic이고, 그쪽에서 real axis로 접근했을 때 $\operatorname{Im}g_s^R\le0$이어야 한다.
2. $\lvert E\rvert\to\infty$에서는 resolvent의 일반적인 점근형인 $g_s^R\sim1/E$로 가야 한다.
3. band 밖에서는 lead 안쪽으로 갈수록 감쇠하는 해를 택해야 한다. 멀어질수록 커지는 해는 물리적인 semi-infinite lead의 boundary condition을 만족하지 않는다.

band 안의 에너지를

$$
E=\varepsilon-2t_0\cos ka,
\qquad 0<ka<\pi
$$

로 쓰면, 지금의 hopping 부호 규약에서 retarded surface Green's function은 다음과 같다.

$$
g_s^R(E)=-\frac{e^{ika}}{t_0}
\tag{13}
$$

각 lead에서는 국소 site 번호 $n=0,1,\ldots$가 device에서 **멀어지는 쪽**으로 증가한다고 정했다. 이 좌표계에서 식 (13)은 양쪽 lead 모두에서 device를 떠나는 outgoing wave를 나타낸다. 하나의 전체 좌표계로 바꾸어 보면 오른쪽 lead에서는 $+k$, 왼쪽 lead에서는 $-k$인 wave다.

$q=-t_0g_s^R$로 놓으면 $\eta\to0^+$에서 $q=e^{ika}$이고, 에너지가 band를 가로지를 때 $q$는 복소평면의 단위원 위쪽 반원을 따라 움직인다. 이 branch를 band 밖까지 연속적으로 연장하면 $\lvert q\rvert<1$이 되어 lead 안쪽으로 감쇠한다. 반대쪽 root는 $q^{-1}$이므로 band 밖에서 거리가 멀어질수록 커진다.

```python
def surface_green_scalar(E, t0, eps_lead, eta=1e-9):
    b = (E + 1j * eta) - eps_lead
    disc = np.sqrt(b * b - 4.0 * t0 * t0 + 0j)
    roots = ((b + disc) / (2.0 * t0**2),
             (b - disc) / (2.0 * t0**2))

    # propagating band 안에서는 Im(g_s)<0인 retarded root를 선택한다.
    for g in roots:
        if g.imag < 0:
            return g

    # eta=0인 band 밖에서는 공간적으로 감쇠하고 1/E로 가는 root를 선택한다.
    return min(roots, key=lambda g: abs(t0 * g))

def lead_self_energy_scalar(E, t0, eps_lead, eta=1e-9):
    """matched contact 끝점의 self-energy: sigma^R = t0^2 g_s^R."""
    return t0**2 * surface_green_scalar(E, t0, eps_lead, eta)
```

contact hopping도 $\tau=-t_0$로 맞추면 surface self-energy를 real shift와 broadening으로 나누어 쓸 수 있다.

$$
\sigma^R=t_0^2g_s^R=-t_0e^{ika}
=\underbrace{-t_0\cos ka}_{\lambda(E)}
-\frac{i}{2}\underbrace{2t_0\sin ka}_{\gamma(E)}
\tag{14}
$$

실수부 $\lambda=\operatorname{Re}\sigma^R$는 resonance energy를 이동시킨다. contact broadening은

$$
\gamma=-2\operatorname{Im}\sigma^R
$$

로 정의하며, propagating lead channel을 통해 device의 입자가 빠져나갈 수 있는 coupling의 세기를 나타낸다. 여러 orbital이 contact에 연결되는 경우에는 broadening matrix를 다음과 같이 정의한다.

$$
\Gamma_\alpha
=i\left(\Sigma_\alpha^R-\Sigma_\alpha^A\right)
\succeq0
\tag{15}
$$

여기서 $\Sigma_\alpha^A=\Sigma_\alpha^{R\dagger}$다. 이때 행렬의 “허수부”는 원소마다 스칼라 허수부를 취한 값이 아니라 다음 anti-Hermitian part를 뜻한다.

$$
\operatorname{Im}_{H}X=\frac{X-X^\dagger}{2i},
\qquad
\Gamma_\alpha=-2\operatorname{Im}_{H}\Sigma_\alpha^R
$$

일반적인 complex matrix에서는 두 연산이 서로 다르다. 다만 이 글처럼 contact마다 스칼라 $\sigma^R$ 하나만 있는 경우에는 같은 값이 된다.

contact hopping이 임의의 $t_c$일 때에는 다음 관계가 성립한다.

$$
\gamma=2\pi\lvert t_c\rvert^2\rho_s,
\qquad
\rho_s=-\frac{1}{\pi}\operatorname{Im}g_s^R
$$

여기서 $\rho_s$는 **surface DOS**이며 bulk DOS와는 다르다. $\gamma=\hbar\lvert v\rvert/a$까지 쓸 수 있는 것은 이 글에서처럼 $t_c=t_0$인 matched contact에 한정된다.

1차원에서는 이 둘의 차이가 특히 뚜렷하다. 식 (13)으로 구한 lead 끝점의 spectral density는 band 안에서

$$
\rho_s(E)=\frac{\sin ka}{\pi t_0}
$$

이고, 양쪽 band edge에서 0으로 간다. 반면 translationally invariant infinite chain의 site당 bulk DOS는

$$
\rho_{\mathrm{bulk}}(E)
=\frac{1}{2\pi t_0\lvert\sin ka\rvert}
$$

이므로 band edge에서 발산한다. 두 결과는 서로 모순되지 않는다. bulk DOS는 $k$-state들이 에너지축에 얼마나 빽빽하게 놓이는지를 센다. surface DOS에는 여기에 더해 각 mode가 끝점 orbital에서 갖는 weight가 곱해진다. semi-infinite chain의 band-edge mode는 끝점에서 진폭이 작기 때문에, 이 boundary weight가 bulk DOS의 van Hove divergence를 상쇄한다.

scalar matched contact에서는 group velocity

$$
v(k)=\frac{1}{\hbar}\frac{dE}{dk}
=\frac{2t_0a}{\hbar}\sin ka
$$

를 이용하면 $\gamma=2t_0\sin ka=\hbar\lvert v\rvert/a$다. $\sin ka$가 band edge에서 0이 되므로 surface coupling과 group velocity도 함께 0으로 간다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-04-surface-green-function.png" alt="닫힌 체인과 열린 체인의 spectral response, contact self-energy의 실수부와 허수부, 복소평면에서의 retarded surface root" width="98%">
</p>

*그림 4.* hard-wall로 닫혀 있던 chain을 lead에 연결하면 discrete levels가 lead band 안의 continuum과 결합한다. self-energy의 실수부는 level의 위치를 바꾸고, broadening은 propagating band 밖에서 0이 된다. 복소 $q$ 평면에서는 어느 root가 retarded boundary condition을 만족하는지 확인할 수 있다.

band 밖에서 broadening이 0이 되더라도 contact의 영향까지 모두 없어지는 것은 아니다. propagating escape channel은 닫히지만 real self-energy와 evanescent response는 남을 수 있다.

또한 양쪽 lead continuum과 완전히 분리된 bound state가 있다면, 그 state의 occupation은 두 contact의 Fermi function만으로 정해지지 않는다.

---

## 6. open-device Green's function으로 transmission 구하기

각 에너지에서 다음 순서로 수송량을 계산한다.

1. 왼쪽과 오른쪽 lead의 surface Green's function을 구한다.
2. 그 값으로 각 contact의 self-energy를 만들고, 해당 device orbital에 더한다.
3. self-energy에서 $\Gamma_L$과 $\Gamma_R$를 계산한다.
4. 유한한 device 행렬의 역행렬을 구해 $G^R$를 얻는다.
5. $G^R$에서 spectral function과 transmission을 계산한다.

device에서 허용되는 전체 spectral weight는 spectral function

$$
A(E)=i(G^R-G^A)
\tag{16}
$$

으로 나타낸다. contact와 결합한 continuum에서 $\eta\to0^+$ limit을 취하면 이를

$$
A=A_L+A_R,
\qquad
A_\alpha=G^R\Gamma_\alpha G^A
\tag{17}
$$

처럼 contact별로 나눌 수 있다. $A_L$은 왼쪽 contact에서 입사한 scattering state가 device 안에서 만드는 spectral weight이고, $A_R$은 오른쪽에서 입사한 state에 해당한다. 여기에는 아직 Fermi function이 들어 있지 않다. 즉 $A_L$과 $A_R$에는 state의 공간 분포만 있고, **실제 occupation**에 관한 정보는 없다.

수치 계산에서 $\eta$를 유한하게 두었다면 식 (17)에 한 항이 더 생긴다. 이때 정확한 항등식은

$$
i(G^R-G^A)
=G^R\big(\Gamma_L+\Gamma_R+2\eta I\big)G^A
$$

이다. $2\eta I$는 역행렬을 안정적으로 계산하기 위해 넣은 조절항이며, 세 번째 contact를 뜻하지 않는다. 또한 $\Gamma_L=\Gamma_R=0$인 bound state가 있다면 두 reservoir의 정보만으로는 그 occupation을 정할 수 없다.

### 6.1 Caroli 식이 transmission을 나타내는 이유

coherent transmission은 Caroli formula

$$
\boxed{
T(E)=\operatorname{Tr}
\left[\Gamma_LG^R\Gamma_RG^A\right]
}
\tag{18}
$$

로 계산한다. transmission이 amplitude의 절댓값 제곱이라는 점은

$$
X=\Gamma_L^{1/2}G^R\Gamma_R^{1/2}
$$

로 놓자. trace의 순환성을 이용하면

$$
T=\operatorname{Tr}(XX^\dagger)\ge0
$$

를 얻는다. 따라서 $T$는 한쪽 contact에서 다른 쪽 contact로 전달되는 진폭의 절댓값 제곱을 모든 channel에 대해 더한 값이며, contact coupling의 세기도 함께 포함한다. 이상적인 lead에 연결된 passive coherent device에서는 각 transmission eigenvalue가 0 이상 1 이하이다.

식 (18)은 한 contact에서 반대편 contact로 전달되는 amplitude $G^R$와 그 Hermitian conjugate $G^A$의 곱이다. trace는 가능한 입사 channel의 기여를 모두 합한다.

현재의 single-channel 체인에서는

$$
\Gamma_L=\gamma_L\lvert1\rangle\langle1\rvert,
\qquad
\Gamma_R=\gamma_R\lvert N\rangle\langle N\rvert
$$

이므로 식 (18)은

$$
T(E)=\gamma_L\gamma_R\lvert G^R_{1N}\rvert^2
\tag{19}
$$

로 간단해진다. $G^R_{1N}$은 device의 두 끝점 사이를 전달하는 amplitude이고, $\gamma_L$과 $\gamma_R$은 양쪽 channel을 lead flux에 맞게 정규화한다. $G^R$의 단위는 $\mathrm{eV}^{-1}$, $\gamma$의 단위는 eV이므로 $T$에는 단위가 없다.

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

### 6.2 potential profile과 Hamiltonian을 함께 읽는 법

먼저 입력으로 준 potential profile과 역행렬에 실제로 들어가는 Hamiltonian을 나란히 비교한다. 노트북은 깨끗한 device와 높이 $V_b=0.60\ \mathrm{eV}$, 폭 6개 site인 단일 장벽을 사용한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-05-potential-and-hamiltonian.png" alt="단일 장벽의 potential energy 분포와 그에 대응하는 삼중대각 수송 Hamiltonian" width="94%">
</p>

*그림 5.* potential barrier를 넣으면 해당 구간의 대각 원소가 $2t_0$에서 $2t_0+V_b$로 올라가고, 비대각 hopping $-t_0$는 바뀌지 않는다. 주황색으로 표시한 $E_{\mathrm{probe}}=0.400\ \mathrm{eV}$는 장벽 구간의 local band lower edge보다 낮다. 뒤에서 이 에너지의 evanescent tunneling을 살펴본다.

potential energy가 $U$로 일정한 구간에서는 dispersion relation이

$$
E=U+2t_0(1-\cos ka)
$$

이고, $k$가 실수인 에너지 범위는

$$
U\le E\le U+4t_0
$$

이다. 따라서 장벽 안에서 $E<V_b$이면 wavevector $k$가 복소수가 되고 evanescent wave가 생긴다. 그렇다고 장벽 안의 wavefunction이 0이 되는 것은 아니다. 장벽의 폭이 유한하면 양쪽 경계에서 evanescent solution과 바깥의 propagating wave를 이어 붙일 수 있다.

$E<U$인 sub-barrier 영역에서 $k=i\kappa$로 놓으면

$$
\cosh(\kappa a)=1+\frac{U-E}{2t_0}
$$

가 된다. 유한 장벽 안의 정확한 해에는 감쇠하는 성분과 증가하는 성분이 모두 들어가며, 두 성분의 비율은 양쪽 경계에서 wavefunction을 맞추어 정한다. 따라서 partial LDOS의 envelope가 모든 site에서 정확히 단조 감소할 필요는 없다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-06-clean-wire-and-barrier-transport.png" alt="matched wire와 단일 장벽의 transmission 및 왼쪽 contact 기원 partial LDOS" width="94%">
</p>

*그림 6.* device와 lead의 Hamiltonian이 완전히 같으면 반사를 일으킬 경계가 없으므로 open band 안에서 $T(E)=1$이다. 장벽을 넣으면 반사가 생겨 transmission이 에너지에 따라 달라지고, **왼쪽 contact에서 입사한 state의 partial LDOS** $[A_L]_{jj}/(2\pi)$도 위치에 따라 변한다. 이 값은 $f_L$을 곱하기 전의 available spectral weight이며, 실제로 점유된 전체 density는 아니다.

깨끗한 matched wire는 구현을 점검하는 가장 기본적인 시험이다. 이 경우 한 channel의 conductance와 resistance는

$$
\mathcal G_0=\frac{2e^2}{h}=77.4809\ \mu\mathrm{S},
\qquad
R_0=12.9064\ \mathrm{k}\Omega
$$

이다. 노트북에서는 band 안의 여러 energy에서 $0.999999997\le T\le1.000000000$을 얻었다. self-energy의 부호나 contact 끝점의 index를 잘못 잡으면 이 값부터 1에서 벗어난다. 정확한 band edge에서는 $\gamma\to0$이므로 그 극한을 따로 다루어야 한다.

conductance quantum은 transmission의 normalization 상수가 아니라 Landauer 식의 linear-response limit에서 나온다. electrochemical potential 차이 $\Delta\mu=e\,\Delta V$가 작을 때 Fermi function의 차이는

$$
f_L(E)-f_R(E)
\simeq-\frac{\partial f}{\partial E}\,\Delta\mu
$$

로 선형화된다. zero-temperature limit에서는 $-\partial f/\partial E$가 $\delta(E-E_F)$가 된다. 이를 7.1절에서 유도할 Landauer 식에 넣으면

$$
\mathcal G
=\left.\frac{dI}{dV}\right\rvert_{V=0}
=\frac{2e^2}{h}T(E_F)
$$

를 얻는다. 따라서 $2e^2/h$는 spin-degenerate single channel이 가질 수 있는 conductance의 자연스러운 단위다. 깨끗한 matched wire가 이 값에 도달하는 이유는 실제 scattering probability가 1이기 때문이지, $T$를 임의로 정규화했기 때문이 아니다.

### 6.3 장벽 안의 tunneling state와 flux

왼쪽 contact가 rank one일 때, 그 contact에서 device로 입사한 state의 amplitude를

$$
\lvert\psi_L(E)\rangle
=G^R(E)\sqrt{\gamma_L(E)}\lvert1\rangle
\tag{20}
$$

로 정의하자. 이 amplitude는 contact flux에 맞춰 normalize한 값이다. norm이 1인 bound-state wavefunction과는 다르며 단위도 $\mathrm{eV}^{-1/2}$이다. 이 정의를 사용하면

$$
A_L=\lvert\psi_L\rangle\langle\psi_L\rvert
$$

가 정확히 성립한다. 따라서

$$
\rho_{L,j}(E)=\frac{\lvert\psi_{L,j}(E)\rvert^2}{2\pi}
$$

는 spin 하나에 대해 왼쪽 contact에서 입사한 state가 만드는 partial LDOS다. 같은 state가 $j\to j+1$ bond를 지나는 방향성 spectral flux는

$$
\mathcal J_{j\to j+1}^{(L)}(E)
=-2\operatorname{Im}
\left[H_{j,j+1}\psi_{L,j+1}\psi_{L,j}^*\right]
\tag{21}
$$

로 정의한다. 지금처럼 reciprocity가 성립하는 single-channel model의 coherent steady state에서는 이 값이 모든 내부 bond에서 같으며 transmission과도 일치한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-07-sub-barrier-scattering-state.png" alt="유한 장벽을 tunneling하는 scattering state의 partial LDOS와 보존되는 spectral flux" width="94%">
</p>

*그림 7.* $E_{\mathrm{probe}}=0.400\ \mathrm{eV}$에서 partial LDOS는 장벽을 지나는 동안 크게 줄어들지만, 장벽 오른쪽에도 작은 transmitted component가 남는다. 위치에 따른 spectral weight는 크게 달라져도 spectral flux는 모든 bond에서 $T=2.8572\times10^{-3}$으로 같다.

흔히 “입자가 장벽을 뚫고 간다”고 말하지만, 실제 계산은 시간에 따라 움직이는 작은 입자를 추적하지 않는다. stationary scattering state 하나가 구조 전체의 boundary condition을 동시에 만족한다. 고전적으로 허용되지 않는 유한 구간에서는 진폭이 작아지지만, 이웃 site의 complex amplitude 사이에는 flux를 만드는 phase 차이가 남는다. 그 결과 작더라도 0이 아닌 flux가 장벽 전체에서 보존된다. equilibrium에서는 오른쪽에서 입사한 state가 반대 방향으로 같은 양의 flux를 만들어 서로 상쇄한다. 실제 전류가 흐르려면 두 방향에서 입사하는 state의 occupation이 달라야 한다.

---

## 7. nonequilibrium에서는 spectrum과 occupation을 따로 계산한다

$G^R$, $A_L$, $A_R$, $T$는 Hamiltonian과 contact로 정해지는 available state와 propagation을 나타낸다. Hamiltonian과 contact가 그대로라면 reservoir의 occupation만 바꾸어도 이 네 양은 변하지 않는다.

reservoir $\alpha$가 energy $E$의 state를 채울 확률은 Fermi function

$$
f_\alpha(E)=
\frac{1}{1+\exp[(E-\mu_\alpha)/(k_BT_\alpha)]}
\tag{22}
$$

으로 정한다. equilibrium에서는 $f_L=f_R=f$이므로 양쪽 contact continuum이 같은 분포로 채워진다. nonequilibrium에서는 왼쪽과 오른쪽에서 들어오는 state의 occupation이 서로 다르기 때문에, device 전체를 하나의 Fermi function으로 나타낼 수 없다.

각 contact가 device로 공급하는 occupied spectral weight를 합치면 in-scattering self-energy가 된다.

$$
\Sigma^{\mathrm{in}}(E)
=\Gamma_L(E)f_L(E)+\Gamma_R(E)f_R(E)
\tag{23}
$$

이 source가 device 안에 만드는 electron correlation function은

$$
\begin{aligned}
G^n(E)
&=G^R\Sigma^{\mathrm{in}}G^A\\
&=A_Lf_L+A_Rf_R.
\end{aligned}
\tag{24}
$$

이다. 식 (24)는 선형계의 source correlation과 response correlation 사이 관계와 같은 구조다. source $\lvert s\rangle$에 대한 응답이 $\lvert\psi\rangle=G^R\lvert s\rangle$이고, 서로 incoherent한 injection source의 correlation이 $\langle ss^\dagger\rangle=\Sigma^{\mathrm{in}}$라면 응답의 correlation은

$$
\langle\psi\psi^\dagger\rangle
=G^R\Sigma^{\mathrm{in}}G^A
$$

가 된다. NEGF에서는 이를 Keldysh equation이라고 한다.

이 글에서 쓰는 convention은

$$
G^n=-iG^<
$$

이다. 문헌마다 $G^<$와 $G^n$의 convention이 다를 수 있으므로, 다른 식을 가져올 때에는 먼저 정의를 확인해야 한다. 그렇지 않으면 density나 current의 부호가 바뀔 수 있다.

세 양의 차이를 정리하면 다음과 같다.

```text
A_alpha(E): contact alpha가 공급할 수 있는 state의 device 내 공간 분포
f_alpha(E): reservoir alpha가 그 state를 채우는 비율
G^n(E):     실제로 점유된 spectral weight
```

```python
def in_scattering(Gamma_L, Gamma_R, f_L, f_R):
    return Gamma_L * f_L + Gamma_R * f_R

def electron_correlation(GR, Sigma_in):
    return GR @ Sigma_in @ GR.conj().T

def bond_current_integrand(H, Gn, i):
    return -2.0 * np.imag(H[i, i + 1] * Gn[i + 1, i])
```

spin degeneracy 2까지 포함하면 site $j$의 occupation은

$$
n_j=2\int\frac{dE}{2\pi}[G^n(E)]_{jj}
\tag{25}
$$

이고, nearest-neighbor bond를 지나는 전류는

$$
I_{j\to j+1}
=-\frac{2e}{h}\int dE\,
2\operatorname{Im}
\left[H_{j,j+1}G^n_{j+1,j}(E)\right]
\tag{26}
$$

으로 계산한다. 전류 식에 대각 원소가 아니라 비대각 원소가 들어가는 이유는 continuity equation에서 바로 확인할 수 있다. site $j$의 입자수 연산자는 $\hat n_j=c_j^\dagger c_j$다. hopping Hamiltonian에 Heisenberg equation을 적용하면

$$
\frac{d\langle\hat n_j\rangle}{dt}
+\mathcal J_{j\to j+1}
-\mathcal J_{j-1\to j}=0
$$

이라는 lattice continuity equation을 얻는다. $\hat n_j$와 commute하지 않는 것은 site $j$를 이웃과 연결하는 hopping 항뿐이다. 이 항의 expectation value에는 $\langle c_j^\dagger c_{j+1}\rangle$ 같은 coherence가 나타나며, 지금의 convention에서는 $G^n_{j+1,j}$가 그 정보를 갖는다. 두 site 사이의 위상차가 이 비대각 원소의 허수부에 나타나고, 그 값이 실제로 어느 방향으로 입자가 흐르는지를 정한다.

따라서 $G^n$의 대각 원소에서는 각 site의 occupation을, 첫 번째 비대각 원소에서는 bond를 지나는 흐름을 구할 수 있다. site의 전하량이 필요하면 occupation에 전자 전하 $-e$를 곱한다.

여기서 $\mathcal J$는 bond를 지나는 particle flux다. 여기에 전자의 전하와 방향 convention을 적용한 것이 식 (26)의 전기 전류다. $\eta\to0^+$인 coherent steady state에서는 device 안의 어느 site에도 입자가 계속 쌓일 수 없다. 따라서 continuity equation의 축적항은 0이고, 에너지에 대해 적분한 전류는 모든 내부 bond에서 같아야 한다.

다만 수치적으로 유한한 $\eta$를 쓰면 device 전체에 아주 약한 absorber를 둔 것과 같은 효과가 생긴다. 이 때문에 뒤의 검증에서 보이는 정도의 작은 위치별 편차는 남을 수 있다. 편차가 그보다 크다면 energy grid의 해상도, index와 부호, 빠뜨린 source 또는 sink 항, 수치 오차를 차례로 확인해야 한다.

식 (26)에는 서로 다른 이유로 생긴 2가 두 번 나온다. $2e/h$의 2는 spin degeneracy에서 온다. integrand 안의 2는 continuity equation에서 hopping 항과 Hermitian conjugate 항을 합칠 때 생긴다. 두 값을 모두 spin 계수로 해석하면 전류를 실제보다 두 배 크게 계산하게 된다.

### 7.1 $G^n$에서 Landauer 식을 얻는 과정

왼쪽 terminal을 지나는 순전류는 왼쪽 contact가 device에 주입하는 양에서 device가 같은 contact로 내보내는 양을 뺀 값이다. 이를 식으로 쓰면

$$
I_L=\frac{2e}{h}\int dE\,
\operatorname{Tr}
\left[\Sigma_L^{\mathrm{in}}(A_L+A_R)-\Gamma_LG^n\right]
$$

이다. trace 안의 첫 항은 왼쪽 contact가 주입하는 occupied spectrum이고, 둘째 항은 이미 점유된 device spectrum 가운데 왼쪽 contact로 빠져나가는 부분이다. 여기에 $\Sigma_L^{\mathrm{in}}=\Gamma_Lf_L$과 $G^n=A_Lf_L+A_Rf_R$를 대입하면 $A_Lf_L$에 해당하는 두 항이 정확히 상쇄되어

$$
I=\frac{2e}{h}\int dE\,
T(E)[f_L(E)-f_R(E)]
\tag{27}
$$

만 남는다. 따라서 식 (27)의 Landauer current를 얻는다. $T(E)$만으로는 전류를 정할 수 없다. 양쪽에서 입사하는 state의 occupation 차이 $f_L-f_R$를 곱하고 에너지에 대해 적분해야 한다.

노트북의 energy grid에는 eV를 쓰지만 $h$는 joule·second 단위다. 따라서 실제 수치 계산의 prefactor에는

$$
\frac{2e}{h}J_{\mathrm{per\,eV}},
\qquad
J_{\mathrm{per\,eV}}=1.602176634\times10^{-19}\ \mathrm{J/eV}
$$

가 들어간다. 여기서 $e>0$는 기본전하의 크기다. 노트북에서는 $f_L>f_R$일 때 왼쪽에서 오른쪽으로 향하는 **전자 흐름**을 양의 $I$로 정했다. 전자의 전하가 $-e$이므로, 부호 있는 전하를 기준으로 한 conventional current의 방향은 이와 반대다.

---

## 8. 예제 A: Hamiltonian은 그대로 두고 reservoir occupation만 바꾸기

예제 A에서는 retarded 계산에 들어가는 Hamiltonian과 self-energy를 고정한 채 reservoir occupation만 바꾼다. 이를 통해 spectrum을 정하는 계산과 실제 occupation을 정하는 계산을 분리해서 볼 수 있다. device에는 높이 $0.40\ \mathrm{eV}$인 두 장벽이 있고, 그 사이의 well은 16개 site로 이루어져 있다. device Hamiltonian과 두 lead band, retarded self-energy는 바꾸지 않고 $\mu_L$과 $\mu_R$만 하나의 resonance 양쪽에 둔다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-08-example-a-double-barrier.png" alt="spectrum을 고정한 resonant-tunneling 예제의 이중 장벽 potential energy 분포" width="88%">
</p>

*그림 8.* 초록색 선은 potential energy $U_j$이며 $H_{jj}$ 자체가 아니다. Hamiltonian의 대각 원소는 $2t_0+U_j$다. open boundary condition은 device 내부에 임의의 complex potential을 더해서 만드는 것이 아니라, 양 끝의 contact self-energy로 구현한다.

### 8.1 닫힌 well의 bound level이 resonance가 되는 과정

두 장벽의 높이가 무한하면 well 안에는 discrete bound levels가 생긴다. 장벽이 유한하면 각 mode의 wavefunction이 장벽 바깥까지 이어지고, 그 결과 양쪽 lead continuum과 결합한다.

isolated resonance $r$에 대응하는 closed-well orbital을 $\lvert\phi_r\rangle$라 하고 device 영역에서 정규화하자. contact self-energy가 resonance width 안에서 천천히 변한다면, contact별 partial broadening은

$$
\Gamma_{\alpha,r}\simeq
\langle\phi_r\rvert\Gamma_\alpha(E_r)\lvert\phi_r\rangle,
\qquad \alpha=L,R
$$

로 근사할 수 있다. 같은 isolated-resonance approximation을 적용하면, device Green's function을 complex energy로 analytic continuation했을 때의 pole은 대략

$$
\mathcal E_r\simeq E_r-\frac{i}{2}\Gamma_r,
\qquad
\Gamma_r\simeq\Gamma_{L,r}+\Gamma_{R,r}
$$

가 된다. $E_r$는 contact self-energy의 실수부까지 반영된 resonance energy다. 이 complex pole을 device와 lead를 모두 포함한 Hermitian Hamiltonian의 eigenvalue로 해석하면 안 된다. 전체 Hamiltonian의 eigenvalue는 여전히 실수다. complex pole은 device에 projection한 Green's function을 analytic continuation했을 때 생기는 singularity다.

self-energy가 resonance 부근에서 빠르게 변하거나 여러 resonance가 겹치면 이 단순한 projection approximation은 더 이상 정확하지 않다.

서로 잘 분리된 resonance 하나의 근방에서는 transmission이 Breit-Wigner form

$$
T(E)\approx
\frac{\Gamma_{L,r}\Gamma_{R,r}}
{(E-E_r)^2+(\Gamma_r/2)^2}
\tag{28}
$$

으로 근사된다. resonance 중심에서의 transmission은

$$
T(E_r)=
\frac{4\Gamma_{L,r}\Gamma_{R,r}}
{(\Gamma_{L,r}+\Gamma_{R,r})^2}
$$

이다. 좌우 coupling이 같으면 각 장벽의 off-resonant transmission이 매우 작더라도 $T(E_r)=1$이 될 수 있다. well 안에서 여러 번 반사된 amplitude가 같은 위상으로 더해지는 Fabry-Pérot형 간섭 때문이다.

같은 근사에서 resonance lifetime은

$$
t_{\mathrm{life},r}\simeq\frac{\hbar}{\Gamma_r}
$$

이다. $\Gamma_r$는 energy linewidth이며, escape rate는 $\Gamma_r/\hbar$다. lifetime 기호로 $t_{\mathrm{life},r}$를 쓴 이유는 contact coupling matrix $\tau$와 혼동하지 않기 위해서다.

노트북의 energy scan에서 서로 분리된 resonance peak를 찾고 그 위치를 정밀하게 계산한 결과는 다음과 같다.

$$
E_r\approx
0.0639,\ 0.1413,\ 0.2453,\ 0.3713\ \mathrm{eV}
$$

$0.5146\ \mathrm{eV}$ 부근에도 별도의 peak가 있다. 이 가운데 두 번째 resonance를 택해

$$
E_{\mathrm{res}}=0.1413\ \mathrm{eV},
\qquad
\mu_L=0.201\ \mathrm{eV},
\qquad
\mu_R=0.081\ \mathrm{eV},
\qquad
k_BT=0.005\ \mathrm{eV}=5\ \mathrm{meV}
$$

로 둔다. $k_BT=5\ \mathrm{meV}$는 약 $58\ \mathrm{K}$에 해당한다. 위 resonance 목록에는 현재 energy scan에서 서로 분리해 확인할 수 있는 peak만 포함했다. 이보다 훨씬 좁은 peak가 하나도 없다는 뜻은 아니다.

### 8.2 available spectrum과 실제 occupation

두 partial spectral function을 well 영역 $W$에 projection하면

$$
\rho_{W,L}(E)=\frac{1}{2\pi}\sum_{j\in W}[A_L(E)]_{jj},
\qquad
\rho_{W,R}(E)=\frac{1}{2\pi}\sum_{j\in W}[A_R(E)]_{jj}
$$

를 얻는다. $\rho_{W,L}$과 $\rho_{W,R}$은 각각 왼쪽과 오른쪽 contact에서 입사한 state가 well 안에 갖는 **available spectral weight**다. 각 항에 해당 reservoir의 Fermi function을 곱하면 실제 occupied spectrum은 다음과 같다.

$$
f_L\rho_{W,L}+f_R\rho_{W,R}
$$

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-09-example-a-spectral-bookkeeping.png" alt="contact별로 나눈 available well spectrum과 실제 occupied well spectrum" width="94%">
</p>

*그림 9.* 양쪽 contact 모두 같은 resonance에 state를 공급할 수 있다. 이 예제에서 chemical potential을 바꾸어도 resonance의 모양은 변하지 않는다. 달라지는 것은 두 reservoir가 그 resonance를 채우는 비율이며, 두 기여를 합한 결과가 $G^n=A_Lf_L+A_Rf_R$다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-10-example-a-transport-summary.png" alt="고정 spectrum resonant-tunneling 예제의 transmission, LDOS, contact별 점유, spectral current" width="96%">
</p>

*그림 10.* (a)에는 고정된 transmission spectrum과 주된 occupation window가, (b)에는 LDOS의 quasi-bound well mode가 나타나 있다. (c)는 occupied spectral weight를 contact별로 나눈 결과다. (d)에서 전류를 정하는 양은 transmission peak의 높이가 아니라 $dI/dE$를 에너지에 대해 적분한 **면적**이다.

이 면적을 정확히 구하려면 energy grid도 충분한 해상도를 가져야 한다. uniform grid는 linewidth가 매우 좁은 resonance peak 자체를 건너뛸 수 있다. 1차원 band edge처럼 적분값은 유한하지만 함수가 급격히 변하는 구간도 제대로 잡지 못할 수 있다.

노트북에서는 emitter band edge 근처에 geometric grid를 놓고, 나머지 occupied-energy 구간에는 촘촘한 linear grid를 쓴다. 두 grid를 합친 뒤 각 transmission peak 후보의 주변을 따로 탐색해 정확한 위치를 구한다. 따라서 plotting grid에서 가장 높은 grid point를 resonance energy로 그대로 사용하지 않는다.

비균일 grid $E_0<\cdots<E_{M-1}$에서 사다리꼴 적분을 $\int dE\,F(E)\simeq\sum_iw_iF(E_i)$로 쓰면 각 점의 weight는

$$
w_0=\frac{E_1-E_0}{2},\qquad
w_i=\frac{E_{i+1}-E_{i-1}}{2},\qquad
w_{M-1}=\frac{E_{M-1}-E_{M-2}}{2}
$$

로 정해진다. 비균일 grid에서는 각 함수값을 단순히 더하면 안 된다. 코드에 $w_i$를 명시하면 서로 다른 간격이 적분에 올바르게 반영된다.

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

`current_prefactor_A_per_eV`에는 spin factor와 eV–joule 변환 계수가 이미 포함되어 있으므로 2를 다시 곱하면 안 된다. density, bond current, terminal current에는 모두 같은 energy grid와 quadrature weight를 사용한다. 그래야 서로 다른 적분 규칙 때문에 생긴 차이를 물리적 불일치로 오해하지 않는다.

Landauer 식으로 계산한 전류는

$$
I_{\mathrm{Landauer}}=263.0891\ \mathrm{nA}
$$

이다. 이와 독립적으로 구한 평균 bond current는 $263.0900\ \mathrm{nA}$이며, bond 사이의 상대 편차는 $5.30\times10^{-5}$다. contact별 occupation을 더한 $n=n_L+n_R$도 상대오차 $3.0\times10^{-15}$ 이내에서 맞는다. occupied well spectrum을 적분한 값과 well site의 occupation을 합한 값은 모두 $4.882464787$ 전자다.

예제 A의 결과를 두 줄로 정리하면 다음과 같다.

```text
f_L과 f_R을 바꾸면 점유와 전류가 바뀐다.
H_D와 lead self-energy가 고정되어 있다면 G^R 자체는 바뀌지 않는다.
```

---

## 9. 예제 B: bias에 따라 움직이는 resonance와 NDR

bias는 두 reservoir의 occupation만 갈라놓는 것이 아니다. device의 single-particle potential을 기울이고 contact band의 에너지 기준도 바꿀 수 있다. 예제 B에서는 선형 voltage drop $\phi$를 가정하고, 그에 따른 electron potential energy $U=-e\phi$를 Hamiltonian에 직접 넣는다. 이 profile은 Poisson equation을 self-consistent하게 풀어 얻은 결과가 아니다.

여기서는 90-site device를 사용한다. 높이 $0.45\ \mathrm{eV}$인 double barrier 사이에 8-site well을 두었다.

zero bias에서 확인되는 resonance energy는 다음과 같다.

$$
E_r(0)\approx0.0504,\ 0.1946,\ 0.4109\ \mathrm{eV}
$$

operating point는 다음과 같이 정했다.

$$
E_F=E_1(0)-0.015\ \mathrm{eV}=0.0354\ \mathrm{eV},
\qquad
k_BT=0.003\ \mathrm{eV}\quad(T\approx35\ \mathrm{K})
$$

가장 낮은 resonance는 처음에 emitter Fermi level보다 $15\ \mathrm{meV}$ 높다. bias가 변하면 이 resonance가 transport window와 정렬됐다가 다시 어긋날 수 있다.

device를 따라 선형으로 변하는 전자의 potential energy는

$$
U_j^{\mathrm{bias}}(V)
=-eV\frac{j}{N-1}
$$

로 둔다. 배열의 에너지 단위가 eV이면 코드에 들어가는 수치는

$$
U_j^{\mathrm{bias}}[\mathrm{eV}]
=-V[\mathrm{V}]\frac{j}{N-1}
\tag{29}
$$

이다. $e\times1\ \mathrm{V}=1\ \mathrm{eV}$이므로 eV 단위 배열에 SI 단위의 기본전하를 다시 곱하면 안 된다. 물리적인 관계는 $\varepsilon_R(V)=\varepsilon_L-eV$, $\mu_R(V)=E_F-eV$다. 에너지를 eV로, bias를 volt의 수치로 쓰면

$$
\varepsilon_R[\mathrm{eV}]=\varepsilon_L[\mathrm{eV}]-V[\mathrm{V}],
\qquad
\mu_R[\mathrm{eV}]=E_F[\mathrm{eV}]-V[\mathrm{V}],
\qquad
\mu_L[\mathrm{eV}]=E_F[\mathrm{eV}]
\tag{30}
$$

가 된다. $\mu_R$과 오른쪽 lead의 band edge가 같은 양만큼 내려가므로 둘 사이의 거리는 변하지 않는다. 그렇다고 bias를 두 번 적용한 것은 아니다. $\varepsilon_R$은 collector band의 energy range를 옮기고, $\mu_R$은 그 band를 어디까지 채울지를 정한다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-11-example-b-bias-profile.png" alt="zero bias와 finite bias에서의 이중 장벽 potential energy 분포와 이동한 contact 기준" width="88%">
</p>

*그림 11.* 선형 경사는 Poisson solution이 아니라 계산에 직접 넣은 bias profile이다. 같은 profile이 well과 barrier의 potential을 기울이고, collector의 에너지 기준도 아래로 옮긴다.

### 9.1 삼중대각 구조를 이용한 $O(N)$ 계산

$T(E,V)$ map을 만들려면 여러 energy와 bias에서 Green's function을 반복해서 계산해야 한다. dense matrix의 역행렬을 매번 구하면 계산량이 $O(N^3)$이다. 하지만

$$
\mathcal A(E,V)
=(E+i0^+)I-H_D(V)-\Sigma_L^R-\Sigma_R^R
$$

는 tridiagonal matrix이고, 끝점 contact의 transmission에는 $G^R_{1N}$ 하나만 필요하다.

$$
\mathcal A\mathbf x=\mathbf e_N
$$

을 풀면 $x_1=G^R_{1N}$이다. 따라서 tridiagonal solver를 사용하면 각 $(E,V)$ 조합의 계산량을 $O(N)$으로 줄일 수 있다.

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

$E=0.09\ \mathrm{eV}$에서 이 방법과 전체 dense matrix를 사용해 구한 transmission의 차이는 $4.61\times10^{-19}$다.

### 9.2 bias를 높였는데 current가 줄어드는 이유

각 bias의 current는 Landauer 식

$$
I(V)=\frac{2e}{h}\int dE\,
T(E,V)\big[f_L(E)-f_R(E)\big]
\tag{31}
$$

으로 계산한다. $V$가 커지면 $f_L-f_R$가 0이 아닌 occupation window도 넓어진다. occupation만 생각하면 current는 계속 증가해야 할 것처럼 보인다. 그러나 $T(E,V)$의 resonance ridge도 bias에 따라 이동하고 모양이 바뀐다. 상수 prefactor를 제외한 energy-resolved current integrand는

$$
\text{current integrand}
=T(E,V)[f_L(E)-f_R(E)]
$$

이다. current는 이 함수의 특정 energy에서의 값이 아니라 전체 면적으로 정해진다. zero temperature에서 적분에 기여하려면 energy가 다음 세 구간에 모두 들어 있어야 한다.

$$
[\mu_R,\mu_L]
\cap[E_{c,L},E_{c,L}+4t_0]
\cap[E_{c,R},E_{c,R}+4t_0]
$$

첫 번째는 두 reservoir 사이의 occupation window이고, 나머지 두 구간은 각 contact의 propagating band다. resonance peak가 아무리 높아도 이 교집합 밖에 있으면 steady-state DC current에 기여하지 않는다. 온도가 유한하면 occupation window의 경계는 부드러워지지만, 양쪽 lead에 propagating state가 있어야 한다는 조건은 그대로다.

세 구간을 따로 표시하면 current가 줄어든 원인도 구분할 수 있다. resonance가 occupation window를 벗어났는지, emitter band 또는 collector band와 겹치지 않는지를 각각 확인하면 된다.

가장 낮은 resonance가 occupation window 안에서 emitter의 propagating state와 정렬되면 current가 커진다. bias를 더 높이면 alignment가 다시 어긋난다. 동시에 resonance가 고정된 emitter band edge에 가까워지면서 injection도 약해진다. 이 예제에서는 $E_{c,L}=0$이므로 band edge 근처에서

$$
\gamma_L(E)\simeq2\sqrt{t_0E}
$$

처럼 $\gamma_L$이 작아진다. 일반적인 band edge에서는 $E$ 대신 $E-E_{c,L}$를 쓴다. 따라서 resonance가 band 아래로 완전히 내려가기 전부터 injection이 약해질 수 있다. occupation window는 넓어졌는데도 $T(E,V)[f_L-f_R]$의 적분값은 오히려 감소하는 것이다.

current가 감소하는 bias 구간에서는

$$
\frac{dI}{dV}<0
$$

이고, 이를 negative differential resistance (NDR)라고 한다. NDR은 $I/V$가 음수라는 뜻이 아니라 differential conductance $dI/dV$가 음수라는 뜻이다.

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-12-example-b-ndr-summary.png" alt="NDR 예제의 전류-전압 곡선, differential conductance, resonance-transmission map, peak와 valley의 spectral current" width="96%">
</p>

*그림 12.* 네 개의 main panel은 모두 가장 낮은 resonance가 만드는 첫 cycle을 확대해 분석한다. current는 $0.085\ \mathrm{V}$에서 peak에 도달하고, $0.149\ \mathrm{V}$에서 valley에 도달한다. 그 사이에서는 $dI/dV<0$이다. inset의 전체 sweep에는 약 $0.4\ \mathrm{V}$에서 훨씬 큰 두 번째 cycle도 나타난다. 여기서 보고한 PVCR은 첫 cycle의 peak와 valley로 계산했다.

첫 cycle의 정량 결과는

$$
I_{\mathrm{peak}}=57.3122\ \mathrm{nA},
\qquad
I_{\mathrm{valley}}=0.6519\ \mathrm{nA},
\qquad
\mathrm{PVCR}
=\frac{I_{\mathrm{peak}}}{I_{\mathrm{valley}}}
=87.92
$$

이다. 이 값은 transmission map의 pixel을 읽어서 구한 것이 아니다. heatmap은 resonance ridge의 움직임을 확인하기 위한 그림이므로 비교적 성긴 grid를 사용한다.

정량적인 current를 계산할 때에는 energy 간격을 $dE=0.05\ \mathrm{meV}$로 줄이고, 첫 cycle 근처에서는 bias 간격도 $1\ \mathrm{mV}$로 줄인다. 각 bias point마다 device potential, collector band reference, collector chemical potential을 새로 계산한다.

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

heatmap이 매끄럽다는 사실만으로 linewidth가 좁은 resonance의 적분 면적까지 수렴했다고 판단할 수는 없다. 그래서 plotting grid와 integration grid를 따로 둔다. 노트북에서는 수렴을 확인한 current data를 미분하고, peak-prominence 조건으로 작은 수치 요동을 제거한 뒤 첫 peak와 그 뒤의 valley를 찾는다.

(d)의 peak와 valley 곡선에는 같은 절대 $dI/dE$ 축 범위를 적용했다. 녹색 valley 곡선이 거의 보이지 않는 것은 실제 적분 면적이 훨씬 작기 때문이다. 두 곡선을 서로 다르게 정규화해서 생긴 차이가 아니다.

이 예제의 NDR에는 두 효과가 함께 작용한다. bias가 커지면서 resonance alignment가 어긋나고, emitter band edge 근처에서는 propagating state의 공급도 줄어든다. 다만 이는 외부에서 정한 선형 potential ramp와 1차원 single band를 사용한 현재 모형의 결과다. 모든 resonant-tunneling diode에서 똑같은 microscopic mechanism이 작동한다고 일반화할 수는 없다.

실제 device에서는 charge redistribution, self-consistent electric field, scattering, transverse mode, nonparabolic band, hysteresis도 중요할 수 있다.

---

## 10. 수치 검증으로 구현 확인하기

NEGF 구현에서는 부호나 index를 잘못 써도 코드가 오류 없이 끝까지 실행될 수 있다. 실행이 끝났다는 사실만으로는 물리적으로 옳은 결과인지 알 수 없다. 그래서 노트북은 각 계산 단계에서 항등식과 limiting case를 직접 확인한다.

- **closed-chain spectrum:** analytic solution과 diagonalization 결과가 $1.33\times10^{-15}\ \mathrm{eV}$까지 일치한다.
- **closed-chain eigenstate:** $\lVert H\psi_n-E_n\psi_n\rVert<5.5\times10^{-16}\ \mathrm{eV}$다.
- **exact lead elimination:** Schur complement와 전체 역행렬의 device block 차이가 $3.48\times10^{-15}\ \mathrm{eV}^{-1}$다.
- **surface recursion:** quadratic-equation residual이 $2.22\times10^{-16}$이다.
- **retarded root:** $\operatorname{Im}g_s^R<0$이고 $q$의 기준점이 $1,i,-1$이며 band 밖에서 감쇠한다.
- **matched wire:** propagating band 안에서 $T(E)=1$이다.
- **single barrier:** $\operatorname{Tr}(\Gamma_RA_L)$, endpoint transmission formula, bond flux가 일치한다.
- **spectral identity:** 유한 $\eta$의 $2\eta G^RG^A$ 항까지 포함하면 상대오차 $1.77\times10^{-16}$이다.
- **equilibrium:** contact continuum에서 $G^n=f(A_L+A_R)$가 상대오차 $7.26\times10^{-17}$로 맞는다.
- **terminal current:** terminal formula가 Landauer integrand로 줄어드는 상대오차가 $1.67\times10^{-16}$이다.
- **예제 A:** bond별 current는 유한한 $\eta$로 예상되는 편차 안에서 거의 일정하며, 독립적으로 계산한 Landauer integral과 일치한다.
- **fast solver:** $O(N)$ endpoint calculation이 dense-matrix trace와 일치한다.
- **NDR integration:** peak와 valley의 $dI/dE$ 면적을 적분하면 표시한 current가 그대로 나온다.

검증마다 찾아낼 수 있는 오류가 다르다. matched-wire test가 실패하면 contact matching과 self-energy의 부호부터 확인한다. Schur test가 실패하면 block 순서와 Hermitian conjugate를 점검한다. bond마다 current가 다르면 $G^n$의 index, bond 방향, energy resolution, spin factor의 중복 여부를 살펴봐야 한다. 어느 한 검증만 통과했다고 해서 나머지 오류까지 없다고 볼 수는 없다.

---

## 11. 현재 모형에서 포함하지 않은 물리

앞에서 유도한 lead elimination과 균일한 lead의 surface Green's function은 지금 정한 single-particle lattice model 안에서 근사 없이 성립한다. 반면 isolated-resonance approximation과 외부에서 정한 bias profile은 이후의 해석과 예제에서 추가한 가정이다. 근사 없이 성립하는 결과와 모형화 과정에서 추가한 가정을 구분해야 한다. 또한 현재 모형은 다음 물리를 포함하지 않는다.

- electron-electron 또는 electron-phonon interaction을 나타내는 self-energy를 포함하지 않는다. 따라서 single-particle 또는 mean-field 수준의 description에 가깝다.
- phase coherence가 유지되는 elastic transport만 다룬다.
- lead는 균일하고 single-channel이며 semi-infinite다.
- 유효질량은 일정하고 격자는 1차원이다.
- spin은 degeneracy factor 2로만 반영하며 spin-orbit coupling이나 magnetic structure는 다루지 않는다.
- 예제 B는 NEGF–Poisson을 self-consistent하게 풀지 않고 선형 potential-energy drop을 가정한다.
- 두 lead continuum과 분리된 bound state의 occupation을 정하려면 초기 상태를 따로 지정하거나 relaxation mechanism을 추가해야 한다.

이 제한들은 NEGF formalism 자체의 한계가 아니다. multi-orbital contact에서는 surface self-energy가 scalar가 아니라 matrix가 된다. scattering을 포함하려면 retarded self-energy와 in-scattering self-energy를 추가한다. electric field까지 self-consistent하게 구할 때에는 $G^n$에서 density를 계산하고, 그 density로 $H_D$를 갱신한다. 세부 모형이 달라져도 retarded propagation과 occupation을 분리하는 계산 구조는 그대로 유지된다.

NEGF 계산에서는 boundary condition과 available spectrum을 retarded self-energy와 $G^R$로 다루고, reservoir occupation은 in-scattering self-energy와 $G^n$로 다룬다. 이 둘을 처음부터 섞지 않고 각각 계산한 다음, density와 current 같은 observable을 구할 때 결합한다.

---

## 12. 노트북으로 직접 계산하기

본문의 그림과 수치 검증은 아래 노트북에서 모두 재현할 수 있다.

- [Kaggle에서 노트북 실행하기](https://www.kaggle.com/code/pilkwang/from-closed-to-open-quantum-transport-with-negf)
- [GitHub에서 소스 노트북 읽기](https://github.com/pilkwangkim/Physics/blob/master/negf_from_scratch.ipynb)

본문에 나온 순서대로 notebook cell을 실행하면 각 식과 수치 결과를 함께 확인할 수 있다. 그다음에는 격자 간격, 장벽 모양, contact hopping, reservoir chemical potential, bias profile 가운데 하나씩 바꾸고 같은 검증을 반복하면 된다. 이렇게 하면 parameter를 바꾸어 생긴 물리적 변화와 부호·단위·index 오류를 구분할 수 있다.
