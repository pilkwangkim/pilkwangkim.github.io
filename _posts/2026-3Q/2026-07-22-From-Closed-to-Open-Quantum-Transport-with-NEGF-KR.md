---
title: "Hamiltonian에서 전류까지: NEGF의 원리와 계산"
date: 2026-07-22 09:00:00 +0900
categories: [Physics, Quantum Transport]
tags: [quantum-transport, physics, green-functions, kaggle, landauer, negf, open-quantum-systems, python, resonant-tunneling, self-energy, tight-binding]
math: true
interactive: negf-barrier
pin: false
topic: quantum-transport
lang: ko
translation_key: quantum-transport-negf
image:
  path: /assets/img/posts/2026-07-22-negf-from-scratch/cover-exact-lead-elimination.png
  alt: "Semi-infinite lead의 변수를 소거하고 그 영향을 device 경계의 energy-dependent self-energy로 나타내는 과정."
  hide_caption: true
  fit: contain
  width: 1400
  height: 1040
---

- 실행 가능한 노트북: [From Closed to Open: Quantum Transport with NEGF, From Scratch](https://www.kaggle.com/code/pilkwang/from-closed-to-open-quantum-transport-with-negf)
- 소스 노트북: [negf_from_scratch.ipynb](https://github.com/pilkwangkim/Physics/blob/master/negf_from_scratch.ipynb)
- 영문판: [From Closed to Open Quantum Transport with NEGF: A Derivation from Scratch]({{ site.baseurl }}/posts/From-Closed-to-Open-Quantum-Transport-with-NEGF/)

고립된 device의 유한한 Hamiltonian을 대각화하면 eigenstate는 구할 수 있다. 하지만 contact가 어떤 state를 채우는지, device를 떠나는 wave가 어디로 가는지는 알 수 없다. 양쪽에 reservoir를 연결하면 전자가 계속 들어오고, device를 통과한 전자는 계산 영역의 끝에서 인위적으로 반사되지 않고 빠져나간다.

> 무한히 긴 lead에 연결된 device를 계산할 때, 임의의 absorbing boundary condition을 넣지 않고도 유한한 행렬만으로 정확히 풀 수 있을까?

**Nonequilibrium Green's function(NEGF)**에서는 lead의 각 site에 대한 amplitude를 device 방정식에 대입해 소거한다. Lead 안의 amplitude를 미지수로 직접 계산하지 않아도 된다는 뜻이지, lead의 영향을 무시한다는 뜻은 아니다. Device의 retarded response에 미치는 영향은 energy-dependent self-energy로 남으며, reservoir가 주입하는 전자 분포는 따로 지정한다. 주어진 single-particle Hamiltonian에 대해 이 소거 과정은 정확하다.

model의 조건은 다음과 같다.

- 일정한 effective mass를 갖는 전자를 균일 격자에서 계산하는 1차원 model
- nearest-neighbor hopping과 spin degeneracy 2
- coherent·elastic single-particle transport
- 균일한 semi-infinite lead
- Poisson equation을 self-consistent하게 푼 결과 대신 외부에서 정한 bias profile

**Single-particle**은 정해진 potential 안에서 각 전자를 독립적으로 다룬다는 뜻이지, device에 전자가 하나만 있다는 뜻은 아니다. **Elastic**은 시간이 지나도 변하지 않는 device에서 전자가 통과하거나 반사될 때 에너지가 보존된다는 뜻이다. **Coherent**는 여러 wave 성분 사이의 phase 관계가 유지된다는 뜻이다. 따라서 여러 번 반사된 wave도 서로 interference를 일으킬 수 있다. 두 spin state에는 같은 방정식을 적용하며, 전자수나 전류를 합산할 때 2를 곱해 두 spin의 기여를 포함한다.

여기서는 potential을 미리 정해 계산한다. Self-consistent 계산을 하려면 전자 density를 구한 뒤 Poisson equation을 풀어 electrostatic potential을 계산하고, 그 결과로 Hamiltonian을 갱신하는 과정을 반복해야 한다. Lead elimination을 사용한다고 해서 voltage drop을 특정한 모양으로 정해야 하는 것은 아니다.

---

## 0. 전체 계산 순서

<div class="negf-workflow-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-00-negf-workflow.svg" alt="NEGF 계산 흐름도. device Hamiltonian과 lead self-energy로 retarded Green's function과 spectral function 및 transmission을 구하고, 별도로 주어진 reservoir의 Fermi function을 결합해 occupation, density, bond current와 Landauer current를 계산한다." width="2280" height="2360">
</div>
<div class="negf-workflow-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-00-negf-workflow-mobile.svg" alt="NEGF 계산 흐름도의 세로 배치. device와 lead 정보를 결합해 spectrum과 transmission을 계산하고, 각 reservoir의 occupation을 반영해 density와 current를 구한다." width="1000" height="3856">
</div>

*계산 흐름도.* Device Hamiltonian과 lead self-energy로 spectrum과 transmission을 계산한다. Reservoir의 chemical potential과 온도로 각 state의 occupation을 정한다. Contact별 $A_\alpha$에 $f_\alpha$를 곱해 더하면 $G^n$을 얻고, 여기서 density와 bond current를 구한다. $T(E)$에 $f_L-f_R$를 곱해 적분하면 Landauer current를 얻는다. $\alpha=L,R$이며 그림의 식들은 coherent contact continuum에서 $\eta\to0^+$ 극한을 취한 관계다.

Lead와 reservoir는 역할이 다르다.

1. **Lead**는 phase coherence가 유지되는 semi-infinite quantum system이다. Lead Hamiltonian으로 propagating mode를 구하고, retarded solution에는 outgoing-wave boundary condition을 적용한다.
2. **Reservoir**는 thermal equilibrium을 유지하는 큰 전자 집단이다. State는 chemical potential보다 낮은 에너지에서 대체로 채워지고 높은 에너지에서 대체로 비어 있으며, 온도가 높아질수록 그 사이에서 occupation이 완만하게 바뀐다. Device에서 나온 전자는 멀리 떨어진 reservoir에서 equilibrium에 도달하고, device로 입사하는 전자 분포는 일정하게 유지된다고 가정한다.
3. 실제 **contact**는 두 기능을 함께 맡는다. NEGF에서는 retarded self-energy로 전자가 lead를 따라 전파하고 device를 빠져나가는 효과를 반영한다. Reservoir에서 들어오는 state의 occupation은 Fermi function으로 따로 지정한다.

주어진 에너지에서 **channel**은 특정 방향으로 진행하는 독립적인 lead mode 하나를 뜻한다. 지금의 single-orbital chain에서는 band 안의 각 에너지에 대해 각 lead에 spin별로 incoming channel이 하나씩 있다. **Scattering state**는 한 mode가 입사할 때 생기는 reflected wave와 transmitted wave를 모두 포함한 stationary state다. 한곳에 갇힌 bound state와는 다르다.

$E+i\eta$의 작은 양수 $\eta$는 retarded response에 필요한 outgoing boundary condition을 선택하기 위해 넣는다. $\eta$는 resonance의 물리적 lifetime이 아니다. Contact로 빠져나가는 효과는 complex self-energy에 반영된다.

<div id="01-앞으로-사용할-기호" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

### 0.1 기호와 물리량

- $H_D$: lead를 연결하지 않은 유한 device의 single-particle Hamiltonian
- $g_\alpha^R$: device와 연결하지 않은 **isolated lead** $\alpha$의 Green's function
- $\Sigma_\alpha^R$: lead $\alpha$를 소거한 뒤 device 경계에 남는 energy-dependent self-energy
- $G^R$: lead를 연결한 **open device**의 retarded Green's function
- $\Gamma_\alpha$: contact $\alpha$를 통한 escape coupling의 세기를 나타내는 행렬
- $A_\alpha$: contact $\alpha$에서 들어오는 scattering state가 device 안에 만드는 spectral weight
- $f_\alpha$: reservoir $\alpha$가 incident state를 채우는 비율
- $G^n$: 실제로 채워진 spectral weight를 나타내는 electron correlation function
- $T(E)$: incident flux 가운데 반대편 lead까지 전달되는 비율
- $I$: transmission에 양쪽 reservoir의 occupation 차이를 곱하고 에너지에 대해 적분해 구하는 net current

대소문자도 구분해야 한다. $\Gamma_\alpha$는 device subspace에 작용하는 행렬이다. 이 single-channel model에서는 0이 아닌 원소가 끝점에 하나뿐이며, 그 값을 $\gamma_\alpha$로 쓴다. $T(E)$는 언제나 transmission을 뜻한다. reservoir의 온도는 $T_\alpha$, 그에 해당하는 에너지 척도는 $k_BT_\alpha$로 적는다.

---

<div id="1-schrödinger-방정식을-tight-binding-행렬로-옮기기" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

## 1. Schrödinger equation을 tight-binding 행렬로 쓰기

먼저 1차원 effective-mass Schrödinger equation을 쓰자.

$$
-\frac{\hbar^2}{2m^*}\frac{d^2\psi}{dx^2}+U(x)\psi(x)=E\psi(x).
$$

여기서 $U$는 potential **energy**다. Band minimum 부근에서 dispersion을 parabolic하게 근사할 때, free-electron mass 대신 일정한 effective mass $m^*$를 사용한다. 아래에서 $m_0$는 free-electron mass를 뜻한다.

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

를 얻는다. 식의 형태는 nearest-neighbor tight-binding model과 같다. 다만 여기서 $t_0$는 경험적으로 정한 hopping parameter가 아니다. Continuum kinetic-energy operator를 finite difference로 근사하면서 이웃 site 사이의 coupling으로 나타난 값이다.

site basis에서 Hamiltonian의 matrix element는 다음과 같다.

$$
H_{jj}=2t_0+U_j,
\qquad
H_{j,j+1}=H_{j+1,j}=-t_0
$$

Site state는 orthonormal하다고 둔다. 즉 $\langle i\vert j\rangle=\delta_{ij}$이며, $\delta_{ij}$는 $i=j$이면 1, 다르면 0이다. 이 basis에서 $U_j=U(x_j)$이고 $\psi_j=\langle j\vert\psi\rangle$다. 공간 적분으로 normalize한 continuum wavefunction과는 다음 관계로 연결된다.

$$
\begin{gathered}
\psi_j\simeq\sqrt{a}\,\psi(x_j),
\\
\sum_j\lvert\psi_j\rvert^2\simeq\int dx\,\lvert\psi(x)\rvert^2
\end{gathered}
$$

상수 $\sqrt a$는 식 (1)의 양변에서 상쇄되지만 density를 해석할 때에는 중요하다. Site occupation은 격자 cell 하나에 들어 있는 전자수를 뜻한다. 단위 길이당 전자수와는 구분해야 한다.

여기서 $U_j$와 $H_{jj}$를 구분해야 한다. Potential energy는 $U_j$이고, Hamiltonian의 diagonal element에는 kinetic-energy operator를 discretize하면서 생기는 $2t_0$도 들어간다.

노트북에서 사용한 $m^*=0.067m_0$, $a=1\ \mathrm{nm}$를 대입하면 다음 값을 얻는다.

$$
t_0=0.5687\ \mathrm{eV},
\qquad
4t_0=2.2746\ \mathrm{eV}
$$

코드에서는 NumPy를 `np`로 import해 사용한다. 각 site의 방정식에 들어가는 계수들을 Hamiltonian의 해당 행에 넣는다.

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

배열의 dtype을 complex로 지정해도 이 closed-system Hamiltonian은 여전히 Hermitian이다. 나중에 complex self-energy를 더할 때 dtype을 바꾸지 않아도 되도록 complex array로 만든 것뿐이다.

격자 간격 $a$를 바꾸면 finite-difference Hamiltonian도 달라진다. $t_0\propto a^{-2}$이므로 $a$에 따라 lattice bandwidth와 $U/t_0$ 같은 dimensionless ratio가 바뀐다. 여기서 $a$는 continuum model을 근사하기 위한 격자 간격이며, 재료 자체의 lattice constant를 뜻하지 않는다.

Grid convergence를 확인하려면 device의 **실제 길이와 barrier의 실제 폭**을 고정해야 한다. 더 촘촘한 격자에 맞춰 $U_j$를 다시 만들고, 관심 있는 observable의 값이 충분히 수렴하는지 확인한다. Site 수를 그대로 둔 채 $a$만 절반으로 줄이면 격자가 촘촘해질 뿐 아니라 device의 실제 길이도 절반으로 줄어든다.

### 1.1 lattice band와 continuum limit

$U_j=0$인 균일 무한 chain에 Bloch wave

$$
\psi_j=e^{ikja}
$$

를 넣으면 다음 dispersion relation을 얻는다.

$$
E(k)=2t_0-t_0e^{ika}-t_0e^{-ika}
=2t_0(1-\cos ka)
\tag{2}
$$

이 band에서 가능한 에너지는 0부터 $4t_0$까지다. $\lvert ka\rvert\ll1$에서는

$$
1-\cos ka\simeq\frac{(ka)^2}{2}
$$

이므로

$$
E(k)\simeq t_0(ka)^2=\frac{\hbar^2k^2}{2m^*}
$$

가 되어 continuum limit의 parabolic dispersion과 일치한다.

이 dispersion은 $k$가 $2\pi/a$만큼 바뀌면 반복된다. 따라서 $-\pi/a\le k<\pi/a$처럼 중복 없이 한 주기를 나타내는 구간을 **Brillouin zone**이라고 한다. Lattice dispersion이 continuum의 포물선과 일치하는 것은 $\lvert ka\rvert\ll1$일 때다. Brillouin zone 끝에서 포물선과 달라지는 것은 실제 물질의 nonparabolicity가 아니라 finite-difference lattice의 dispersion error다. 관심 에너지 범위에서 이 오차가 충분히 작아지는지 $a$를 줄여 가며 확인해야 한다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-01-lattice-dispersion.svg" alt="finite-difference lattice의 dispersion relation과 hard-wall 유한 chain의 discrete spectrum" width="2160" height="1320">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-01-lattice-dispersion-mobile.svg" alt="finite-difference lattice의 dispersion relation과 hard-wall 유한 chain의 discrete spectrum" width="1000" height="2280">
</div>

*그림 1.* $k=0$ 부근에서는 lattice dispersion과 continuum의 포물선이 일치한다. 그러나 lattice band의 폭은 $4t_0$로 유한하다. hard-wall 유한 chain에서는 boundary condition을 만족하는 discrete wavevector만 허용된다.

---

## 2. 닫힌 chain에서 알 수 있는 것과 없는 것

$U_j=0$인 균일 chain의 물리적 격자점에 $j=1,\ldots,N$이라는 번호를 붙이자. 그 바깥에 ghost site를 하나씩 두고

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

여기서 $n=1,\ldots,N$이다. 노트북에서 Hamiltonian을 직접 대각화한 결과도 analytic solution과 일치한다. analytic eigenvalue와 numerical eigenvalue의 최대 차이는 $1.33\times10^{-15}\ \mathrm{eV}$이고, eigenvector residual은 $\lVert H\psi_n-E_n\psi_n\rVert<5.5\times10^{-16}\ \mathrm{eV}$다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-02-closed-chain-eigenstates.svg" alt="hard-wall 유한 tight-binding chain의 처음 다섯 eigenstate를 각 eigenenergy 주변에 표시한 그림" width="2160" height="1840">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-02-closed-chain-eigenstates-mobile.svg" alt="hard-wall 유한 tight-binding chain의 처음 다섯 eigenstate를 각 eigenenergy 주변에 표시한 그림" width="1000" height="2160">
</div>

*그림 2.* 각 eigenfunction을 해당 eigenenergy 높이로 옮겨 그렸다. 곡선이 기준 에너지에서 위아래로 벗어나는 정도는 energy uncertainty나 probability density를 뜻하지 않는다. 두 hard-wall boundary 사이의 실제 potential은 $U_j=0$이다.

이 계산으로 Hamiltonian 구현은 검증할 수 있다. 그러나 reservoir와 open boundary가 없으므로 아직 transport 문제라고 할 수는 없다.

- 왼쪽 reservoir가 공급할 right-moving incident state의 occupation을 정하지 않았다.
- 오른쪽 reservoir가 공급할 left-moving incident state의 occupation도 정하지 않았다.
- 양쪽 끝에는 확률이 빠져나갈 open boundary가 없다.
- spectrum은 discrete stationary states로만 이루어진다.

이 Hamiltonian은 실수이고 time-reversal symmetry를 가지므로 hard-wall eigenstate도 실수로 선택할 수 있다. 이 eigenstate에서는 이웃 site 사이의 probability current가 0이다. Chain을 길게 만들어도 boundary condition은 바뀌지 않는다. 끝에서 반사된 wave가 돌아오는 데 더 오래 걸리고 standing-wave level spacing도 줄어들지만, 그렇다고 reservoir와 전자를 주고받는 open system이 되지는 않는다.

따라서 transport를 계산하려면 단순히 행렬을 키우는 것이 아니라 **boundary condition**을 바꿔야 한다.

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

로 쓸 수 있다. $G^R_{ab}$는 $b$에 단위 크기의 source를 가했을 때 $a$에서 얻는 amplitude다. 역행렬을 취하기 전 operator의 단위가 에너지이므로 $G^R$의 단위는 $\mathrm{eV}^{-1}$다.

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

으로 정의하며, $G^A$는 advanced Green's function이다. Spectral function에서 각 에너지에 해당하는 state의 weight를 읽을 수 있다. Spin 하나당 local density of states(LDOS)는

$$
\rho_j(E)=\frac{A_{jj}(E)}{2\pi}
$$

이다. LDOS는 **site $j$에서 에너지 단위 구간마다 state의 weight가 얼마나 되는지** 나타낸다. 실제로 전자가 얼마나 들어 있는지를 나타내는 양은 아니다. 식 (4)를 대입하면 다음과 같다.

$$
\begin{gathered}
A_{jj}(E)=
\sum_n\lvert\psi_n(j)\rvert^2
\frac{2\eta}{(E-E_n)^2+\eta^2},
\\
\psi_n(j)=\langle j\vert n\rangle
\end{gathered}
$$

Lorentzian kernel $2\eta/[(E-E_n)^2+\eta^2]$의 전체 면적은 $2\pi$다. 따라서 $\rho_j$의 $n$번째 항을 모든 에너지에 대해 적분하면 eigenstate가 site $j$에 갖는 probability weight $\lvert\psi_n(j)\rvert^2$를 얻는다. Peak들이 충분히 떨어져 있다면 peak 하나의 면적이 이 weight에 해당한다. 모든 항을 합해 적분하면 각 site에서 spin별로 state 하나에 해당하는 weight를 얻는다. $N$개 site의 기여를 더하면 $N$개 state가 된다. 이처럼 적분값으로 state 수를 셀 수 있도록 $1/(2\pi)$를 곱한 것이다. 실제 전자 분포를 구하려면 여기에 occupation을 따로 반영해야 한다.

Eigenvalue만 나열하면 각 state의 공간 분포는 알 수 없다. 반면 spectral representation에는 eigenstate마다 다음 projector가 함께 들어 있다.

$$
P_n=\lvert n\rangle\langle n\rvert
$$

$P_n$에는 mode $n$의 site별 amplitude 정보가 들어 있다. 식 (4)에서 해당 mode의 분모가 작아지면 그 mode가 응답에 크게 기여한다. 따라서 Green's function으로 eigenvalue뿐 아니라, 원하는 probe energy에서 각 site가 source에 어떻게 응답하는지도 계산할 수 있다.

닫힌계의 level이 lead의 propagating state와 coupling하면 대체로 폭이 있는 resonance가 생긴다. 이 폭은 수치적으로 넣은 것이 아니라 contact로 빠져나가는 효과에서 생긴다. Device와 lead를 모두 포함한 Hamiltonian은 여전히 Hermitian이다. Decay를 설명할 때 쓰는 complex energy는 전체 Hamiltonian의 complex eigenvalue가 아니라 device 영역의 응답을 기술하는 값이다.

<details markdown="1">
<summary>Pole과 peak, complex resonance energy의 차이</summary>

Pole은 응답을 complex energy의 함수로 보았을 때 발산하는 에너지를 뜻한다. Device에 projection한 Green's function을 lead continuum을 지나 complex plane으로 연장하면 real axis 아래쪽에 resonance pole이 나타날 수 있다. 그 실수부는 resonance 위치를 정하고 음의 허수부는 decay와 관련된다.

Pole과 peak는 같은 개념이 아니다. Pole은 $G^R$의 analytic continuation으로 정의하지만, peak는 유한한 해상도로 실수 에너지축을 scan했을 때 관찰되는 극댓값이다.

Resonance가 서로 충분히 떨어져 있고 그 주변에서 self-energy가 천천히 변한다면 peak 위치는 complex pole의 실수부와 거의 일치한다. Resonance가 겹치거나 band threshold가 가깝거나 self-energy가 빠르게 변할 때는 두 값이 달라질 수 있다.

</details>

고립된 유한계의 spectrum은 $\eta\to0^+$ 극한에서 delta peak의 합이 된다. 그림에서는 유한한 $\eta$를 쓰기 때문에 peak에 폭이 생기지만, 이는 수치적으로 넣은 폭일 뿐 물리적 lifetime과는 관계가 없다.

$E-i\eta$ 대신 $E+i\eta$를 쓰는 것은 **retarded** response를 구하기 위해서다. Source를 가한 뒤에만 응답이 나타나고, source를 가하기 전에는 나타나지 않아야 한다는 조건이다. Lead에서는 이 조건에 따라 source에서 멀어지는 outgoing wave를 선택한다. 실제 transport 문제에 incident wave가 없다는 뜻은 아니다. Incident wave는 따로 지정하고 $G^R$로 그 injection에 대한 응답을 구한다.

<details markdown="1">
<summary>에너지에 양의 imaginary part를 더하는 이유</summary>

시간에 따른 phase를 $e^{-iEt/\hbar}$로 쓰면 resolvent를 다음과 같이 나타낼 수 있다.

$$
G^R(E)=-\frac{i}{\hbar}\int_0^\infty dt\,
e^{i(E+i\eta)t/\hbar}e^{-iHt/\hbar}
$$

Source를 가한 시점부터 그 이후의 시간만 적분하며, $e^{-\eta t/\hbar}$ 때문에 적분이 수렴한다. $H$의 eigenstate에서 이 적분을 계산하면 식 (4)에 들어간 $1/(E-E_n+i\eta)$를 얻는다. $i0^+$는 이 boundary condition을 선택한 뒤 $\eta$를 **양수 쪽에서** 0으로 보낸다는 뜻이다.

</details>

---

<div id="4-lead-자유도를-정확히-소거해-열린계를-만든다" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

## 4. Lead를 소거해도 그 영향은 self-energy로 남는다

Device와 lead의 amplitude를 각각 한 묶음의 미지수로 두자. 각 부분에 작용하는 계수 행렬을 $a_D,a_L$, coupling 행렬을 $\tau$, device에 가한 source를 $s_D$로 쓰면 다음 두 식을 얻는다.

$$
a_D\psi_D-\tau\psi_L=s_D,
\qquad
-\tau^\dagger\psi_D+a_L\psi_L=0.
$$

둘째 식에서 $\psi_L=a_L^{-1}\tau^\dagger\psi_D$를 구해 첫째 식에 대입하면

$$
\big(a_D-\tau a_L^{-1}\tau^\dagger\big)\psi_D=s_D
$$

라는 effective equation을 얻는다. 이제 $\psi_L$을 미지수로 따로 구할 필요는 없지만, lead의 영향은 $\tau a_L^{-1}\tau^\dagger$ 항에 남아 있다. Block matrix에서도 같은 방식으로 한쪽 변수를 소거할 수 있다. 이때 남는 행렬을 Schur complement라고 한다.

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

Coupling 행렬과 isolated-lead Green's function은 다음과 같이 정의한다.

- $\tau=H_{DL}$은 lead의 amplitude에 곱해지는 coupling 행렬이며, 그 결과인 $\tau\psi_L$가 device 방정식에 들어간다.
- $\tau^\dagger=H_{LD}$는 device의 amplitude에 곱해지는 coupling 행렬이며, 그 결과인 $\tau^\dagger\psi_D$가 lead 방정식에 들어간다.
- $g_L^R=[(E+i\eta)I_L-H_L]^{-1}$는 device에 연결하기 전 **isolated lead**의 Green's function이다.

이 $g_L^R$는 device가 연결된 전체 Green's function의 $G_{LL}^R$ block과 구분해야 한다. $G_{LL}^R$에는 device를 거쳐 lead로 돌아오는 과정도 포함된다. Self-energy는 device space에 작용하므로 행렬 크기가 $N_D\times N_D$여야 한다.

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

이다. 앞에서 쓴 계수 행렬은 $a_D=(E+i\eta)I_D-H_D$, $a_L=(E+i\eta)I_L-H_L$다. $I_D,I_L$는 각 subspace의 identity matrix다. $\eta$를 양수로 두면 $E$가 isolated Hermitian lead의 spectrum 안에 있더라도 resolvent를 계산할 수 있다.

둘째 줄을 $\psi_L$에 대해 풀면 다음 식을 얻는다.

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

로 정의한다. 행렬곱은 오른쪽부터 읽는다. Device amplitude에 $\tau^\dagger$를 적용한 $\tau^\dagger\psi_D$는 lead 방정식의 source term이다. 여기에 $g_L^R$를 곱하면 lead amplitude를 얻고, 다시 $\tau$를 곱하면 device 방정식에 들어갈 항을 얻는다. “Device에서 나가 lead를 거쳐 돌아온다”는 설명은 이 operator 관계를 뜻한다. 빠져나간 전자가 실제로 모두 되돌아온다는 뜻은 아니다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-03-exact-lead-elimination.svg" alt="Lead의 변수를 소거해 contact site의 self-energy를 구하는 과정과 surface site를 떼어 내도 반복되는 semi-infinite chain의 구조" width="2160" height="2920">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-03-exact-lead-elimination-mobile.svg" alt="Lead의 변수를 소거해 contact site의 self-energy를 구하는 과정과 surface site를 떼어 내도 반복되는 semi-infinite chain의 구조" width="1000" height="3140">
</div>

*그림 3.* (a)에서는 device와 lead의 amplitude를 모두 미지수로 포함한다. (b)에서는 lead의 amplitude를 소거하고 그 영향을 $\Sigma_L^R$로 나타낸다. (c)에서 surface site 하나를 떼어 내면 원래와 같은 semi-infinite chain이 남는다. 이 구조가 반복되므로 surface Green's function의 recursion relation을 세울 수 있다.

식 (8)의 대괄호 안을 역행렬로 바꾸면 전체 resolvent의 **device block**을 정확히 얻는다. 주어진 single-particle Hamiltonian을 device와 lead로 나눴을 때, lead를 소거하는 과정에는 근사가 없다. 다만 이렇게 얻은 작은 행렬이 전체 Hamiltonian과 같은 것은 아니다. Lead의 응답 때문에 에너지에 따라 달라지고, outgoing wave로 확률이 빠져나갈 수 있으면 실수 에너지에서도 non-Hermitian이 될 수 있다.

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

이다. 이를 formal series로 전개하면 다음과 같다.

$$
G^R
=g_D^R
+g_D^R\Sigma^Rg_D^R
+g_D^R\Sigma^Rg_D^R\Sigma^Rg_D^R+\cdots
$$

$\Sigma^R$가 한 번 더 들어갈 때마다 device에서 lead로 나갔다 돌아오는 과정이 한 번 추가된다. Lead amplitude를 미지수에서 제외해도 이 모든 과정은 계산에 포함된다.

양쪽 lead를 모두 소거하고 나면 실제로 역행렬을 구할 대상은 다음 device 크기의 행렬뿐이다.

$$
G^R(E)=
\left[
(E+i0^+)I_D-H_D-\Sigma_L^R(E)-\Sigma_R^R(E)
\right]^{-1}
\tag{10}
$$

같은 $G^R$로 lead의 injection에 대한 응답도 구할 수 있다. Lead에 source $s_L$를 두면 둘째 줄의 해는 $\psi_L=g_L^R(\tau^\dagger\psi_D+s_L)$가 되고, device에 작용하는 source는 $s_D+\tau g_L^Rs_L$가 된다. 식 (10)의 응답 operator는 그대로다. Lead의 incident wave를 정하면 그에 맞는 injection source가 device 경계에 추가된다. 이때도 $G^R$를 incoming Green's function으로 바꾸지는 않는다. 이 source–response 관계를 single-particle Schrödinger equation에서 유도하는 과정은 [Camsari, Chowdhury와 Datta의 2–3절](https://arxiv.org/html/2008.01275v2)에도 정리되어 있다.

노트북에서는 유한 lead까지 포함한 전체 행렬의 역행렬을 직접 구한 뒤, 그 device block을 Schur complement 결과와 비교한다. 두 값의 최대 차이는 $3.48\times10^{-15}\ \mathrm{eV}^{-1}$다.

<details markdown="1">
<summary>코드: 유한 lead로 Schur complement를 직접 검증하기</summary>

여기서 $z=E+i\eta$이고 $N_D,N_L$는 각각 device와 검증용 lead의 orbital 수다. 이 finite-lead 계산으로 확인하는 것은 block elimination의 대수 관계다. Lead 끝에서 반사되어 돌아오는 wave가 없다는 성질을 검증하는 것은 아니다.

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

<div id="5-무한한-lead를-surface-greens-function-하나로-나타낼-수-있는-이유" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

## 5. Surface Green's function으로 무한 lead의 영향 계산하기

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

하나뿐이다. Lead 안쪽의 degrees of freedom은 모두 이 surface-to-surface matrix element에 반영되어 있다.

이제 lead를 site energy가 $\varepsilon$이고 nearest-neighbor hopping이 $-t_0$인 균일한 semi-infinite chain으로 두자. 지금 쓰는 finite-difference Hamiltonian에서는 $\varepsilon=2t_0+U_{\mathrm{lead}}$다. 따라서 $U_{\mathrm{lead}}=0$인 깨끗한 lead의 band는 1절에서 구한 것처럼 $[0,4t_0]$이다.

lead를 surface site $\ell_0$와 $\ell_1$에서 시작하는 나머지 chain으로 나누자. 나머지 부분은 site 번호만 하나 옮겨졌을 뿐 원래 lead와 완전히 같은 semi-infinite chain이다. 그러므로 $\ell_1$에서 본 surface Green's function 역시 $g_s^R$다.

$\ell_1$부터 이어지는 부분을 정확히 소거하면 $\ell_0$에는 $t_0^2g_s^R$라는 self-energy가 생긴다. 따라서 $g_s^R$는 다음 self-consistency equation을 만족한다.

$$
g_s^R(E)=
\frac{1}{E+i0^+-\varepsilon-t_0^2g_s^R(E)}
\tag{11}
$$

이 관계를 계속 대입하면 다음 continued fraction을 얻는다. 분모 안에 같은 형태의 분수가 반복된다.

$$
g_s^R=
\cfrac{1}{E^+-\varepsilon-
\cfrac{t_0^2}{E^+-\varepsilon-
\cfrac{t_0^2}{\ddots}}}.
$$

여기서 $E^+=E+i0^+$다. **Semi-infinite**는 한쪽 끝은 있지만 반대쪽으로는 무한히 이어진다는 뜻이다. 따라서 끝점 하나를 제거해도 원래와 같은 semi-infinite chain을 다시 얻는다. Site가 하나 줄어든 finite chain과는 다르다.

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

1. Complex-energy plane의 upper half-plane에서 analytic이어야 한다. 이 영역에서 real axis로 접근한 극한에서는 $\operatorname{Im}g_s^R\le0$이어야 한다.
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

각 lead의 site 번호 $n=0,1,\ldots$는 device와 맞닿은 곳을 0으로 두고, device에서 **멀어질수록 증가하도록** 정의한다. 이 좌표계에서 식 (13)은 양쪽 lead 모두에서 device를 떠나는 outgoing wave를 나타낸다. Device와 두 lead를 하나의 좌표계로 나타내면 오른쪽 lead에서는 $+k$, 왼쪽 lead에서는 $-k$인 wave다.

$q=-t_0g_s^R$로 놓으면 $\eta\to0^+$에서 $q=e^{ika}$이다. 에너지를 band의 아래쪽 끝에서 위쪽 끝까지 높이면 $q$의 값은 복소평면의 단위원 위쪽 반원을 따라 변한다. 이 branch를 band 밖까지 연속적으로 연장하면 $\lvert q\rvert<1$이 되어 해가 lead 안쪽으로 갈수록 감쇠한다. 다른 root에 해당하는 값은 $q^{-1}$이므로, 그 root로 만든 해는 band 밖에서 lead 안쪽으로 갈수록 커진다.

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

$\succeq0$는 positive semidefinite를 뜻한다. 즉 임의의 device vector $u$에 대해 $u^\dagger\Gamma_\alpha u\ge0$여야 한다. 이상적인 passive lead는 device의 amplitude를 밖으로 전달할 수 있지만 증폭시키지는 않는다. $\Gamma_\alpha$의 단위는 에너지이므로 그 자체가 escape rate는 아니다. 특정 mode의 linewidth를 구한 뒤 $\hbar$로 나누면 escape rate를 얻는다.

여기서 $\Sigma_\alpha^A=\Sigma_\alpha^{R\dagger}$다. 행렬의 “허수부”는 원소별로 scalar imaginary part를 취하는 대신 다음과 같이 정의한다. 이 정의로 얻은 행렬 자체는 Hermitian이다.

$$
\operatorname{Im}_{H}X=\frac{X-X^\dagger}{2i},
\qquad
\Gamma_\alpha=-2\operatorname{Im}_{H}\Sigma_\alpha^R
$$

일반적인 complex matrix에서는 두 연산이 서로 다르다. 다만 이 글처럼 contact마다 scalar $\sigma^R$ 하나만 있는 경우에는 같은 값을 얻는다.

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

이므로 band edge에서 발산한다. 두 결과는 서로 모순되지 않는다. Bulk DOS는 단위 에너지 구간에 들어가는 $k$-state 수로 정해진다. Surface DOS를 구할 때는 각 mode가 끝점 orbital에서 갖는 weight도 반영해야 한다. Semi-infinite chain의 band-edge mode는 끝점에서 amplitude가 작다. 이 boundary weight를 반영하면 bulk DOS의 van Hove divergence가 상쇄된다.

Scalar matched contact에서 wavevector $k$ 근처에 분포하는 wavepacket을 생각하자. 그 중심이 이동하는 속도인 **group velocity**는

$$
v(k)=\frac{1}{\hbar}\frac{dE}{dk}
=\frac{2t_0a}{\hbar}\sin ka
$$

다. 이를 이용하면 $\gamma=2t_0\sin ka=\hbar\lvert v\rvert/a$를 얻는다. $\sin ka$가 band edge에서 0이 되므로 surface coupling과 group velocity도 함께 0으로 간다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-04-surface-green-function.svg" alt="닫힌 chain과 열린 chain의 spectral response, contact self-energy의 실수부와 허수부, 복소평면에서의 retarded surface root" width="2160" height="3900">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-04-surface-green-function-mobile.svg" alt="닫힌 chain과 열린 chain의 spectral response, contact self-energy의 실수부와 허수부, 복소평면에서의 retarded surface root" width="1000" height="4040">
</div>

*그림 4.* (a) 같은 균일 chain이라도 hard-wall로 닫으면 discrete level이, lead에 맞춰 연결하면 continuum이 나타난다. 닫힌 chain의 peak 폭은 그림에 나타내려고 넣은 값이지 escape lifetime에서 생긴 폭이 아니다. (b) 음영으로 표시한 band 안에서는 propagating mode로 빠져나갈 수 있어 broadening이 생긴다. Band 밖에서도 self-energy의 실수부는 남을 수 있다. (c) 실선 branch는 band 안에서 outgoing wave를, 밖에서 $\lvert q\rvert<1$인 감쇠 해를 준다. 점선의 reciprocal root를 고르면 입사하거나 커지는 해가 나온다.

Band 밖에서 broadening이 0이 되어도 contact의 영향까지 모두 없어지는 것은 아니다. 이 에너지에서는 전자가 propagating wave로 lead를 따라 빠져나갈 수 없다. 그래도 real self-energy와 evanescent response를 통한 contact의 영향은 남을 수 있다.

또한 양쪽 lead continuum과 완전히 분리된 bound state가 있다면, 그 state의 occupation은 두 contact의 Fermi function만으로 정해지지 않는다.

---

## 6. open-device Green's function으로 transmission 구하기

각 에너지에서 spectrum과 transmission을 구하는 순서는 다음과 같다.

1. 왼쪽과 오른쪽 lead의 surface Green's function을 구한다.
2. 그 값으로 각 contact의 self-energy를 구하고, device의 접점 orbital에 해당하는 행렬 원소에 넣는다.
3. self-energy에서 $\Gamma_L$과 $\Gamma_R$를 계산한다.
4. 유한한 device 행렬의 역행렬을 구해 $G^R$를 얻는다.
5. $G^R$에서 spectral function과 transmission을 계산한다.

Device의 전체 spectral weight는 spectral function

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

이 spectral decomposition은 다음 inverse-matrix identity에서 직접 나온다.

$$
G^R-G^A
=G^R\big[(G^A)^{-1}-(G^R)^{-1}\big]G^A
$$

양변에 i를 곱하고 식 (10)을 대입하면 앞의 spectral-function 항등식을 얻는다. 그 broadening 항을 왼쪽과 오른쪽 contact의 기여로 나누면 식 (17)이 된다. 따라서 $A_L,A_R$는 전체 spectral weight를 injection contact에 따라 나눈 결과다.

### 6.1 Caroli 식이 transmission을 나타내는 이유

coherent transmission은 Caroli formula

$$
\boxed{
T(E)=\operatorname{Tr}
\left[\Gamma_LG^R\Gamma_RG^A\right]
}
\tag{18}
$$

로 계산한다. Trace는 행렬의 diagonal element를 모두 더한 값이다. Transmission이 amplitude와 어떻게 연결되는지 보려면 다음 행렬을 정의하면 된다.

$$
X=\Gamma_L^{1/2}G^R\Gamma_R^{1/2}
$$

$\Gamma_\alpha$의 positive matrix square root는 eigenvector를 그대로 두고, 0 이상인 각 eigenvalue를 그 제곱근으로 바꾼 행렬이다. 행렬곱의 순서를 순환해서 바꾸어도 trace가 같다는 성질을 이용하면

$$
T=\operatorname{Tr}(XX^\dagger)\ge0
$$

를 얻는다. 따라서 $T$는 contact coupling을 반영한 propagation amplitude의 절댓값 제곱을 channel에 대해 더한 값이다. 이 계산은 $T\ge0$를 보이지만, probability의 상한까지 보이는 것은 아니다. 상한을 정하려면 flux conservation과 incident mode의 flux normalization이 추가로 필요하다. Incoming channel이 여러 개이면 전체 $T$는 각 channel의 transmission probability를 더한 값이므로 1보다 클 수 있다. 여기서 다루는 single-channel $T$는 1을 넘을 수 없다.

식 (18)에서 $G^R$는 device 안에서 amplitude가 전달되는 응답을 나타내고, $G^A$는 그 Hermitian conjugate다. 양쪽 contact의 broadening을 함께 곱하면 lead의 incident·outgoing flux에 맞춘 transmission을 얻는다. Trace는 가능한 입사 channel의 기여를 모두 더한다.

현재의 single-channel chain에서는

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

로 간단해진다. $G^R_{1N}$은 device의 두 끝점 사이에서 amplitude가 어떻게 전달되는지 나타낸다. 여기에 $\gamma_L$과 $\gamma_R$을 곱해 양쪽 lead의 flux normalization을 반영한다. $G^R$의 단위는 $\mathrm{eV}^{-1}$, $\gamma$의 단위는 eV이므로 $T$에는 단위가 없다.

<details markdown="1">
<summary>Single channel에서 transmission이 1을 넘지 않는 이유</summary>

Hopping이 실수이고 device가 Hermitian이며 양쪽 lead에 propagating state가 있는 $\eta\to0^+$ 극한을 생각하자. 경계에서 wavefunction을 맞추면 flux-normalized reflection·transmission amplitude를 다음과 같이 쓸 수 있다. Channel phase를 어떻게 정하느냐에 따라 amplitude의 phase는 달라질 수 있다.

$$
\begin{gathered}
r_L=1-i\gamma_LG^R_{11},
\\
t_{RL}=-i\sqrt{\gamma_R\gamma_L}\,G^R_{N1}
\end{gathered}
$$

이는 [Fisher–Lee의 Green's function–scattering matrix 관계](https://doi.org/10.1103/PhysRevB.23.6851)를 single channel에 적용한 식이다. 실수 hopping에서는 reciprocity에 의해 $G^R_{N1}=G^R_{1N}$이다. 식 (17)의 $11$ 원소는

$$
A_{11}=\gamma_L\lvert G^R_{11}\rvert^2+\gamma_R\lvert G^R_{1N}\rvert^2
$$

이므로

$$
\begin{aligned}
\lvert r_L\rvert^2
&=1-\gamma_L A_{11}+\gamma_L^2\lvert G^R_{11}\rvert^2\\
&=1-\gamma_L\gamma_R\lvert G^R_{1N}\rvert^2\\
&=1-T.
\end{aligned}
$$

를 얻는다. Outgoing flux는 reflection과 transmission의 합이므로 $0\le T\le1$이다. Absorber나 active medium이 있거나 outgoing channel을 빠뜨렸다면 이 논증을 그대로 적용할 수 없다.

</details>

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

입력으로 준 potential profile과 Green's function을 구할 때 사용하는 Hamiltonian을 비교하자. 노트북에서는 barrier가 없는 device와, 높이 $V_b=0.60\ \mathrm{eV}$인 barrier를 6개 site에 걸쳐 놓은 device를 계산한다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-05-potential-and-hamiltonian.svg" alt="Single barrier의 potential energy 분포와 그에 대응하는 tridiagonal Hamiltonian" width="2160" height="1500">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-05-potential-and-hamiltonian-mobile.svg" alt="Single barrier의 potential energy 분포와 그에 대응하는 tridiagonal Hamiltonian" width="1000" height="2520">
</div>

*그림 5.* potential barrier를 넣으면 해당 구간의 diagonal element가 $2t_0$에서 $2t_0+V_b$로 올라가고, off-diagonal hopping $-t_0$는 바뀌지 않는다. 파란색 dash-dot 선으로 표시한 $E_{\mathrm{probe}}=0.400\ \mathrm{eV}$는 barrier 구간의 local band lower edge보다 낮다. 이 에너지에서는 barrier 안에 evanescent wave가 생긴다.

potential energy가 $U$로 일정한 구간에서는 dispersion relation이

$$
E=U+2t_0(1-\cos ka)
$$

이고, $k$가 실수인 에너지 범위는

$$
U\le E\le U+4t_0
$$

이다. 따라서 barrier 안에서 $E<V_b$이면 wavevector $k$가 복소수가 되고 evanescent wave가 생긴다. 그렇다고 barrier 안의 wavefunction이 0이 되는 것은 아니다. barrier의 폭이 유한하면 양쪽 경계에서 evanescent solution과 바깥의 propagating wave를 이어 붙일 수 있다.

$E<U$인 sub-barrier 영역에서 $k=i\kappa$로 놓으면

$$
\cosh(\kappa a)=1+\frac{U-E}{2t_0}
$$

가 된다. 유한 barrier 안의 정확한 해에는 감쇠하는 성분과 증가하는 성분이 모두 들어가며, 두 성분의 비율은 양쪽 경계에서 wavefunction을 맞추어 정한다. 따라서 partial LDOS의 envelope가 모든 site에서 정확히 단조 감소할 필요는 없다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-06-clean-wire-and-barrier-transport.svg" alt="Matched wire와 single barrier의 transmission, 왼쪽 contact에서 입사한 state의 partial LDOS" width="2160" height="1500">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-06-clean-wire-and-barrier-transport-mobile.svg" alt="Matched wire와 single barrier의 transmission, 왼쪽 contact에서 입사한 state의 partial LDOS" width="1000" height="2520">
</div>

*그림 6.* Device와 lead의 Hamiltonian이 완전히 같으면 반사를 일으킬 경계가 없으므로 band 내부에서 $T(E)=1$이다. Barrier를 넣으면 반사가 생겨 transmission이 에너지에 따라 달라지고, **왼쪽 contact에서 입사한 state의 partial LDOS** $[A_L]_{jj}/(2\pi)$도 위치에 따라 변한다. 이 값은 $f_L$을 곱하기 전의 available spectral weight이며, occupation까지 반영한 전체 electron density는 아니다.

Barrier가 없는 matched wire에서 transmission이 1이 나오는지 확인하면 contact 구현을 점검할 수 있다. 노트북에서는 band 안의 여러 energy에서 $0.999999997\le T\le1.000000000$을 얻었다. Self-energy의 부호나 contact 끝점의 index를 잘못 잡으면 이 값부터 1에서 벗어난다. 정확한 band edge에서는 group velocity와 $\gamma$가 0으로 가므로 그 극한을 따로 다루어야 한다.

<div id="63-장벽-안의-tunneling-state와-flux" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

### 6.3 barrier 안의 tunneling state와 flux

왼쪽 contact가 device의 site 하나에만 연결된 현재 model에서는 그 contact의 broadening matrix가 rank one이다. 이 contact에서 device로 입사한 state의 amplitude를

$$
\lvert\psi_L(E)\rangle
=G^R(E)\sqrt{\gamma_L(E)}\lvert1\rangle
\tag{20}
$$

로 정의하자. Device 안에 있을 확률을 1로 맞추는 대신 입사 flux를 1로 맞춘다. Matched contact에서 propagating wave가 존재하는 band 내부의 에너지를 택하고 $\eta\to0^+$로 둔다. 왼쪽 lead에서 device로부터 멀어지는 방향으로 site 번호 $n$을 붙이면 incident wave는 $a_{\rm in}e^{-ikna}$다. 입사 spectral flux의 크기는 $\gamma_L\lvert a_{\rm in}\rvert^2$이므로 $\lvert a_{\rm in}\rvert=1/\sqrt{\gamma_L}$로 두면 된다. 경계에서 wavefunction을 맞추면 이 입사 성분이 $\gamma_L\lvert a_{\rm in}\rvert=\sqrt{\gamma_L}$ 크기의 source로 들어간다. Source 전체에 붙는 phase는 아래의 물리량에 영향을 주지 않는다.

이렇게 구한 amplitude의 단위는 $\mathrm{eV}^{-1/2}$이며, partial spectral function은 다음처럼 쓸 수 있다.

$$
A_L=\lvert\psi_L\rangle\langle\psi_L\rvert
$$

따라서

$$
\rho_{L,j}(E)=\frac{\lvert\psi_{L,j}(E)\rvert^2}{2\pi}
$$

는 왼쪽 contact에서 입사한 state의 spin 하나당 partial LDOS다. 같은 state에 대해 $j\to j+1$ 방향으로 bond를 지나는 spectral flux를 다음과 같이 정의한다.

$$
\mathcal J_{j\to j+1}^{(L)}(E)
=-2\operatorname{Im}
\left[H_{j,j+1}\psi_{L,j+1}\psi_{L,j}^*\right]
\tag{21}
$$

$\eta\to0^+$인 coherent steady state에서는 이 값이 모든 내부 bond에서 같다. 오른쪽 contact에서는 $\gamma_R\lvert\psi_{L,N}\rvert^2=\gamma_L\gamma_R\lvert G^R_{N1}\rvert^2$이며, reciprocity가 성립하는 현재 model에서는 $T$와 같다. 즉 real-space wave로 계산한 flux와 식 (19)의 transmission이 일치한다.

여기서 $\psi_L$의 단위가 $\mathrm{eV}^{-1/2}$이므로 $\mathcal J^{(L)}(E)$는 **dimensionless**다. Spin 하나를 기준으로 에너지 단위 구간당 particle-number flux를 나타내는 양은 $\mathcal J^{(L)}(E)/h$다. 여기에 $f_L(E)$를 곱해 에너지에 대해 적분하면 왼쪽에서 입사한 전자들이 만드는 particle flow를 얻는다. LDOS와 current는 서로 다른 양이다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-07-sub-barrier-scattering-state.svg" alt="유한 barrier를 tunneling하는 scattering state의 partial LDOS와 보존되는 spectral flux" width="2160" height="1500">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-07-sub-barrier-scattering-state-mobile.svg" alt="유한 barrier를 tunneling하는 scattering state의 partial LDOS와 보존되는 spectral flux" width="1000" height="2520">
</div>

*그림 7.* $E_{\mathrm{probe}}=0.400\ \mathrm{eV}$에서 partial LDOS는 barrier 구간에서 크게 줄어들지만, barrier 오른쪽에도 작은 transmitted component가 남는다. Spectral weight는 위치에 따라 크게 달라지지만 spectral flux는 모든 bond에서 $T=2.8572\times10^{-3}$으로 같다.

Barrier 안에 감쇠하는 exponential 하나만 있다면, 전체에 어떤 phase를 곱해도 flux는 0이다. 폭이 유한할 때는 양쪽 경계에서 wavefunction을 맞추면서 감쇠하는 성분과 커지는 성분이 함께 생긴다. $\psi_{L,j}=B_-e^{-\kappa ja}+B_+e^{\kappa ja}$를 식 (21)에 대입하면, barrier 내부의 bond에서

$$
\mathcal J^{(L)}
=4t_0\sinh(\kappa a)\operatorname{Im}(B_+ B_-^*)
$$

를 얻는다. 각 성분의 절댓값 제곱이 들어 있는 항은 실수라서 flux에 기여하지 않는다. 두 성분의 상대 phase가 들어 있는 cross term만 남고, 이 항에서는 위치에 따른 exponential이 서로 상쇄된다. 그래서 그림 7의 partial LDOS는 barrier 안에서 크게 줄어들어도 flux는 일정하다.

Equilibrium에서는 오른쪽에서 입사한 state가 반대 방향으로 같은 크기의 flux를 만들므로 두 기여가 상쇄된다. 전류가 흐르려면 두 방향에서 입사하는 state의 occupation이 달라야 한다.

---

## 7. nonequilibrium에서는 spectrum과 occupation을 따로 계산한다

$G^R$, $A_L$, $A_R$, $T$는 어떤 state가 존재하고 전자가 어떻게 전파하는지를 나타내며, Hamiltonian과 contact로 정해진다. Hamiltonian과 contact를 그대로 둔 채 reservoir의 occupation만 바꾸면 이 네 양은 변하지 않는다.

전자에는 Pauli principle이 적용되므로, spin까지 지정한 single-particle state 하나에는 전자가 최대 하나만 들어갈 수 있다. Reservoir $\alpha$에서 energy $E$의 incident state가 채워져 있을 확률은 **Fermi function**

$$
f_\alpha(E)=
\frac{1}{1+\exp[(E-\mu_\alpha)/(k_BT_\alpha)]}
\tag{22}
$$

으로 정한다. $\mu_\alpha$는 reservoir의 chemical potential이다. Electric potential까지 포함할 때에는 더 정확히 electrochemical potential이라고 부른다. $k_B$는 Boltzmann constant다. Zero temperature에서는 $\mu_\alpha$ 아래의 state는 채워지고 위의 state는 비어 있다. 온도가 0보다 높으면 $f_\alpha(\mu_\alpha)=1/2$이며, $k_BT_\alpha$ 정도의 에너지 범위에 걸쳐 occupation이 완만하게 바뀐다. Chemical potential이 모든 전자의 에너지라는 뜻은 아니다.

Equilibrium에서는 $f_L=f_R=f$이므로 양쪽 contact continuum이 같은 분포로 채워진다. Nonequilibrium에서는 일반적으로 device 전체를 하나의 Fermi function으로 나타낼 수 없다.

두 reservoir에서 들어오는 state 사이에는 고정된 상대 phase가 없다. 따라서 amplitude를 더한 뒤 제곱하는 대신, 각 **outer product**에 occupation을 곱해 더해야 한다. 식 (20)의 left-incident state와 같은 방식으로 정의한 right-incident state에 대해

$$
G^n
=f_L\lvert\psi_L\rangle\langle\psi_L\rvert
+f_R\lvert\psi_R\rangle\langle\psi_R\rvert
$$

가 된다. 각 incident wave는 device에서 반사된 성분과 여전히 coherent interference를 일으킨다. 평균을 취하면 서로 독립적인 incident source 사이의 cross term만 사라진다. 두 wave를 Green's function으로 표현하면 in-scattering self-energy를 얻는다.

$$
\Sigma^{\mathrm{in}}(E)
=\Gamma_L(E)f_L(E)+\Gamma_R(E)f_R(E)
\tag{23}
$$

이 injection으로 채워진 spectrum을 나타내는 electron correlation function은 다음과 같다.

$$
\begin{aligned}
G^n(E)
&=G^R\Sigma^{\mathrm{in}}G^A\\
&=A_Lf_L+A_Rf_R.
\end{aligned}
\tag{24}
$$

여기서 **in-scattering**은 contact에서 전자가 들어오는 injection을 뜻한다. Device 내부에 phonon collision을 추가했다는 뜻은 아니다. 같은 $\Gamma_\alpha$가 contact를 통한 escape와 incident mode의 spectral strength를 함께 나타낸다. Incident state가 실제로 채워져 있을 확률은 $f_\alpha$로 정한다.

$G^n(E)/(2\pi)$는 spin 하나를 기준으로 **occupied one-particle density matrix를 에너지별로 나타낸 양**이다. 이를 에너지에 대해 적분한 행렬을

$$
C_{ij}=\int\frac{dE}{2\pi}\,G^n_{ij}(E)
$$

로 쓰자. Pure state 하나의 density matrix는 $\psi_i\psi_j^*$다. 여러 state가 독립적으로 채워져 있다면 각 outer product에 occupation을 곱해 더한다. 따라서 $C_{jj}$는 site $j$에서 spin 하나당 평균 occupation이고, off-diagonal element에는 site 사이의 phase 정보가 남는다. 입자가 정확히 하나인 density matrix는 trace가 1이지만, 여기서는 $\operatorname{Tr}C$가 해당 spin의 평균 전자수다. Scattering state에서 density를 구하는 이 방법은 [Waintal 등의 7.1절](https://arxiv.org/html/2407.16257v2)에도 정리되어 있다.

<details markdown="1">
<summary>Orbital 하나로 보는 예: 같은 resonance도 occupation은 달라질 수 있다</summary>

Device에 에너지가 $\epsilon_d$인 orbital 하나만 있다고 하자. Resonance 부근에서 두 contact의 self-energy를 상수로 근사한다.

$$
\begin{gathered}
\sigma_\alpha^R=\lambda_\alpha-i\gamma_\alpha/2,
\\
\tilde\epsilon_d=\epsilon_d+\lambda_L+\lambda_R,
\\
\gamma=\gamma_L+\gamma_R>0
\end{gathered}
$$

Orbital이 하나뿐이므로 모든 행렬 계산을 scalar 계산으로 줄일 수 있다.

$$
\begin{gathered}
G^R=\frac{1}{E-\tilde\epsilon_d+i\gamma/2},
\\
A=\frac{\gamma}{(E-\tilde\epsilon_d)^2+(\gamma/2)^2}
\end{gathered}
$$

Occupation을 반영한 correlation function은

$$
\begin{gathered}
G^n=A\,f_{\mathrm{eff}},
\\
f_{\mathrm{eff}}=
\frac{\gamma_Lf_L+\gamma_Rf_R}{\gamma_L+\gamma_R}
\end{gathered}
$$

이다. 좌우 coupling이 같고 좁은 peak 전체에서 $f_L\simeq1,\ f_R\simeq0$이면 $G^n\simeq A/2$다. 즉 spin별 평균 occupation이 절반 정도이며, 두 spin을 합하면 평균적으로 전자가 약 하나 들어 있다. Equilibrium에서는 $f_L=f_R$다. 두 reservoir의 공통 chemical potential이 이 peak보다 충분히 높으면 orbital은 spin별로 거의 완전히 채워지고 net current는 0이다. 어느 경우에도 $A$의 resonance 위치나 모양을 바꿀 필요는 없다.

Self-energy를 상수로 둔 것은 contact band가 충분히 넓을 때 resonance 근방에서 사용하는 근사다. 유한한 band의 edge까지 정확하게 기술하는 식은 아니다.

</details>

<details markdown="1">
<summary>Source correlation과 lesser Green's function 표기</summary>

식 (20)의 spectral normalization을 사용하자. Source $\lvert s\rangle$에 대한 응답이 $\lvert\psi\rangle=G^R\lvert s\rangle$이고 source correlation이 $\langle ss^\dagger\rangle=\Sigma^{\mathrm{in}}$이면, 평균한 outer product는

$$
\langle\psi\psi^\dagger\rangle
=G^R\Sigma^{\mathrm{in}}G^A
$$

가 된다. 여기서 평균은 서로 독립적으로 채워진 incident source에 대해 취한다. 이렇게 평균하더라도 device 안의 coherent propagation이 고전적인 입자 궤적으로 바뀌는 것은 아니다. 이 식은 [Camsari, Chowdhury와 Datta의 3절](https://arxiv.org/html/2008.01275v2)에 나오는 contact-only steady-state Keldysh relation이다.

통상적인 **lesser Green's function** $G^<$는 서로 다른 두 시간의 electron correlation을 나타낸다. Stationary state에서 이를 에너지로 표현하면 occupied spectrum을 얻는다. 여기서 쓰는 convention은

$$
G^n=-iG^<
$$

이다. 이는 표기를 바꾸는 관계이며 occupation에 새로운 가정을 추가하는 식이 아니다.

</details>

세 양의 차이를 정리하면 다음과 같다.

```text
A_alpha(E): contact alpha가 공급할 수 있는 state의 device 내 공간 분포
f_alpha(E): reservoir alpha가 그 state를 채우는 비율
G^n(E):     occupation을 반영한 spectral weight
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

이다. 즉 $n_j=2C_{jj}$는 격자 cell 하나에 들어 있는 평균 전자수다. 이 1차원 model에서 $n_j/a$는 단위 길이당 전자수인 line density이며, 3차원 volume density로 바꾸려면 transverse geometry가 필요하다.

Site 번호가 증가하는 방향의 electron flow를 양의 전류로 정한다. Nearest-neighbor bond의 전류는

$$
I_{j\to j+1}
=-\frac{2e}{h}\int dE\,
2\operatorname{Im}
\left[H_{j,j+1}G^n_{j+1,j}(E)\right]
\tag{26}
$$

으로 계산한다. 왜 diagonal이 아니라 off-diagonal element가 들어갈까? Normalize한 wavefunction에 대해 $i\hbar\dot\psi_j=\sum_kH_{jk}\psi_k$를 사용하여 $\lvert\psi_j\rvert^2$를 시간으로 미분해 보자. 실수인 on-site 항은 상쇄되고, 양옆 site와 연결된 hopping 항만 site $j$의 probability를 바꾼다. 각 state의 occupation까지 반영하면 spin 하나당 particle-number flow는

$$
\begin{gathered}
J^{\mathrm{part}}_{j\to j+1}
=-\frac{2}{\hbar}\operatorname{Im}
\left[H_{j,j+1}C_{j+1,j}\right],\\
\frac{dC_{jj}}{dt}
+J^{\mathrm{part}}_{j\to j+1}
-J^{\mathrm{part}}_{j-1\to j}=0.
\end{gathered}
$$

가 된다. $J^{\mathrm{part}}$의 단위는 초당 입자수다. 식 (21)의 dimensionless spectral flux $\mathcal J(E)$와 구분해야 한다. 여기에 $C=\int G^n\,dE/(2\pi)$를 대입하고 $h=2\pi\hbar$를 사용한다. 두 spin의 기여를 포함하고, 앞에서 정한 electron-flow 방향의 전류로 환산하기 위해 $2e$를 곱하면 식 (26)을 얻는다.

Hopping이 실수 $-t_0$일 때 real standing wave는 허수부가 0이므로 흐름도 0이다. 반면 right-moving wave $\psi_j\propto e^{ikja}$, $0<ka<\pi$는 $\sin ka$에 비례하는 양의 흐름을 만든다. Diagonal population의 크기만으로는 current를 알 수 없다. 이웃 site 사이의 phase 관계가 중요하다. Site의 전하량이 필요하면 $n_j$에 전자 전하 $-e$를 곱한다.

<details markdown="1">
<summary>같은 occupation matrix를 operator로 표현하면</summary>

Spin 하나를 지정했을 때 $c_j^\dagger$는 site $j$에 전자를 하나 추가하는 operator이고 $c_j$는 하나 제거하는 operator다. Occupation operator는 $\hat n_j=c_j^\dagger c_j$이며, $C_{ij}=\langle c_j^\dagger c_i\rangle$다. $\hat n_j$에 Heisenberg equation을 적용해도 같은 continuity equation을 얻는다. Site $j$와 연결된 hopping 항만 commutator에 기여하기 때문이다. 이는 같은 관계를 다른 방식으로 유도한 것이며, model에 새로운 가정을 추가한 것은 아니다.

</details>

$\eta\to0^+$인 coherent steady state에서는 device 안의 어느 site에도 입자가 계속 쌓이지 않는다. 따라서 continuity equation의 시간 변화 항은 0이고, 에너지에 대해 적분한 전류는 모든 내부 bond에서 같아야 한다.

다만 수치적으로 유한한 $\eta$를 쓰면 device 전체에 아주 약한 absorber를 둔 것과 같은 효과가 생긴다. 이 때문에 뒤의 검증에서 보이는 정도의 작은 위치별 편차는 남을 수 있다. 편차가 그보다 크다면 energy grid의 해상도, index와 부호, 빠뜨린 source 또는 sink 항, 수치 오차를 차례로 확인해야 한다.

식 (26)에는 서로 다른 이유로 생긴 2가 두 번 나온다. $2e/h$의 2는 spin degeneracy에서 온다. integrand 안의 2는 continuity equation에서 hopping 항과 Hermitian conjugate 항을 합칠 때 생긴다. 두 값을 모두 spin 계수로 해석하면 전류를 실제보다 두 배 크게 계산하게 된다.

### 7.1 $G^n$에서 Landauer 식을 얻는 과정

왼쪽 terminal을 지나는 순전류는 왼쪽 contact가 device에 주입하는 양에서 device가 같은 contact로 내보내는 양을 뺀 값이다. 이를 식으로 쓰면

$$
I_L=\frac{2e}{h}\int dE\,
\operatorname{Tr}
\left[\Sigma_L^{\mathrm{in}}(A_L+A_R)-\Gamma_LG^n\right]
$$

이다. trace 안의 첫 항은 왼쪽 contact가 주입하는 occupied spectrum이고, 둘째 항은 이미 채워진 device spectrum 가운데 왼쪽 contact로 빠져나가는 부분이다. 여기에 $\Sigma_L^{\mathrm{in}}=\Gamma_Lf_L$과 $G^n=A_Lf_L+A_Rf_R$를 대입하면 $A_Lf_L$에 해당하는 두 항이 정확히 상쇄되어

$$
I=\frac{2e}{h}\int dE\,
T(E)[f_L(E)-f_R(E)]
\tag{27}
$$

만 남는다. 이것이 Landauer current다. $T(E)$만으로는 전류를 정할 수 없다. 양쪽에서 입사하는 state의 occupation 차이 $f_L-f_R$를 곱하고 에너지에 대해 적분해야 한다.

노트북의 energy grid에는 eV를 쓰지만 $h$는 joule·second 단위다. 따라서 실제 수치 계산의 prefactor에는

$$
\frac{2e}{h}J_{\mathrm{per\,eV}},
\qquad
J_{\mathrm{per\,eV}}=1.602176634\times10^{-19}\ \mathrm{J/eV}
$$

가 들어간다. 여기서 $e>0$는 기본전하의 크기다. 노트북에서는 $f_L>f_R$일 때 왼쪽에서 오른쪽으로 향하는 **전자 흐름**을 양의 $I$로 정했다. 전자의 전하가 $-e$이므로 conventional current는 전자가 이동하는 방향과 반대다.

계수 $1/h$는 lead의 incident mode 수를 세어서도 얻을 수 있다. 길이 $L$인 긴 1차원 영역에서 wavefunction을 normalize하면, 허용되는 right-moving wavevector의 간격은 $2\pi/L$이다. 채워진 mode 하나가 단면을 통과하는 rate는 $v/L$이므로, DOS에 이 rate를 곱하면

$$
\frac{L}{2\pi}\frac{dk}{dE}\frac{v}{L}
=\frac{1}{2\pi\hbar}=\frac1h
$$

를 얻는다. 여기서 $v=(1/\hbar)dE/dk$를 사용했다. 느린 mode일수록 DOS는 크지만 단면을 통과하는 rate는 작아서, 두 효과가 정확히 상쇄된다. 이 incident flux에 transmission을 곱하고 반대쪽 reservoir의 기여를 뺀다. 여기에 두 spin과 전하의 크기를 반영하면 prefactor $2e/h$가 나온다.

**Conductance**는 작은 voltage 변화에 대한 current 변화의 비율 $\mathcal G=dI/dV$다. Transmission probability인 $T$와는 다른 양이다. 두 reservoir의 온도가 같고 electrochemical potential 차이 $\Delta\mu=e\,\Delta V$가 작으면 Fermi function의 차이는

$$
f_L(E)-f_R(E)
\simeq-\frac{\partial f}{\partial E}\,\Delta\mu
$$

로 선형화된다. Zero temperature에서는 $f$가 equilibrium Fermi energy $E_F=\mu$에서 바뀌는 step function이므로, $-\partial f/\partial E$가 $\delta(E-E_F)$가 된다. Delta function의 면적은 1이며, 적분하면 함수의 $E_F$에서의 값만 남는다. 따라서 식 (27)을 linearize하면

$$
\mathcal G
=\left.\frac{dI}{dV}\right\rvert_{V=0}
=\frac{2e^2}{h}T(E_F)
$$

를 얻는다. $E_F$가 band 내부에 있는 matched wire에서는 $T(E_F)=1$이므로

$$
\mathcal G_0=\frac{2e^2}{h}=77.4809\ \mu\mathrm{S},
\qquad
R_0=12.9064\ \mathrm{k}\Omega
$$

이다. $R_0$는 $\mathcal G_0$의 역수이며 contact까지 포함한 two-terminal resistance다. 전자가 반사 없이 통과하는 device 내부에 dissipative collision이 있다는 뜻은 아니다. Conductance quantum은 occupation 차이와 mode flux에서 나온다. 계산한 transmission을 임의의 상수로 나누어 맞춘 값이 아니다.

---

## 8. 예제 A: Hamiltonian은 그대로 두고 reservoir occupation만 바꾸기

Device에는 높이 $0.40\ \mathrm{eV}$인 두 barrier가 있고, 그 사이의 well은 16개 site로 이루어져 있다. Device Hamiltonian과 두 lead band, retarded self-energy는 고정하고 $\mu_L$과 $\mu_R$만 하나의 resonance 양쪽에 둔다. Reservoir의 chemical potential만 바꾸므로 available spectrum은 그대로지만 occupation과 전류는 달라진다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-08-example-a-double-barrier.svg" alt="spectrum을 고정한 resonant-tunneling 예제의 이중 barrier potential energy 분포" width="2160" height="800">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-08-example-a-double-barrier-mobile.svg" alt="spectrum을 고정한 resonant-tunneling 예제의 이중 barrier potential energy 분포" width="1000" height="1100">
</div>

*그림 8.* 그림의 선은 potential energy $U_j$이며 $H_{jj}$ 자체가 아니다. Hamiltonian의 diagonal element는 $2t_0+U_j$다. open boundary condition은 device 내부에 임의의 complex potential을 더해서 만드는 것이 아니라, 양 끝의 contact self-energy로 구현한다.

{% include interactive/negf-barrier.html %}

전체 transmission spectrum과 첫 peak를 확대해 그린 그래프를 비교해 보자. Barrier를 높이면 peak 높이가 거의 1로 유지되어도 폭은 줄어들 수 있다. Well을 넓히면 peak 위치도 달라진다. 확대 그래프는 각 곡선의 peak 위치를 원점으로 맞춰 폭을 비교하므로, peak가 이동한 거리는 전체 spectrum에서 확인한다. 이 계산기에서는 device의 Hamiltonian을 바꿔 비교하며, 아래 예제에서 사용하는 reservoir occupation은 바꾸지 않는다.

### 8.1 닫힌 well의 bound level이 resonance가 되는 과정

두 barrier의 높이가 무한하면 well 안에는 discrete bound levels가 생긴다. barrier가 유한하면 각 mode의 wavefunction이 barrier 바깥까지 이어지고, 그 결과 양쪽 lead continuum과 결합한다.

다른 resonance와 충분히 떨어져 있는 resonance $r$의 mode를 $\lvert\phi_r\rangle$라 하자. 이 mode는 device 영역에서 norm이 1이 되도록 normalize한다. 주로 well 안에 분포하지만, 유한한 barrier를 지나 contact 끝점까지 tail이 이어진다. Contact self-energy가 resonance width에 해당하는 에너지 구간에서 천천히 변한다면, contact별 partial broadening은

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

가 된다. $E_r$는 contact self-energy의 실수부까지 반영한 resonance energy다. 이 complex pole은 device와 lead를 모두 포함한 Hermitian Hamiltonian의 eigenvalue가 아니다. 전체 Hamiltonian의 eigenvalue는 여전히 실수다. Complex pole은 device 영역으로 projection한 Green's function을 analytic continuation했을 때 생기는 singularity다.

self-energy가 resonance 부근에서 빠르게 변하거나 여러 resonance가 겹치면 이 단순한 projection approximation은 더 이상 정확하지 않다.

서로 잘 분리된 resonance 하나의 근방에서는 transmission이 Breit-Wigner form

$$
T(E)\approx
\frac{\Gamma_{L,r}\Gamma_{R,r}}
{(E-E_r)^2+(\Gamma_r/2)^2}
\tag{28}
$$

으로 근사된다. Green's function에서 mode 하나만 남기면

$$
G^R(E)\simeq
\frac{\lvert\phi_r\rangle\langle\phi_r\rvert}
{E-E_r+i\Gamma_r/2}
$$

가 된다. 이를 Caroli trace에 대입하면 분자의 contact projection이 각각 $\Gamma_{L,r}$와 $\Gamma_{R,r}$가 되어 식 (28)을 얻는다. 이 근사를 적용하려면 인접한 resonance와 nonresonant transmission background의 기여를 무시할 수 있어야 한다.

Resonance 중심에서의 transmission은

$$
T(E_r)=
\frac{4\Gamma_{L,r}\Gamma_{R,r}}
{(\Gamma_{L,r}+\Gamma_{R,r})^2}
$$

이다. 좌우 coupling이 같으면 각 barrier의 off-resonant transmission이 매우 작더라도 $T(E_r)=1$이 될 수 있다. Well 안에서 여러 번 반사된 amplitude가 constructive interference를 일으키기 때문이다. Fabry-Pérot cavity에서 빛이 여러 번 반사되며 resonance를 만드는 원리와 같다.

전자가 이 mode에 남아 있을 확률이 감소하는 시간 척도를 lifetime이라고 한다. 같은 근사에서 lifetime은

$$
t_{\mathrm{life},r}\simeq\frac{\hbar}{\Gamma_r}
$$

이다. $\Gamma_r$는 energy linewidth이며, escape rate는 $\Gamma_r/\hbar$다. Lifetime은 $t_{\mathrm{life},r}$로 쓰며, contact coupling matrix $\tau$와 구별한다. Peak 높이의 절반에서 잰 전체 energy 폭을 **full width at half maximum(FWHM)**이라고 한다.

Amplitude와 probability의 decay rate가 두 배 차이 나는 이유는 time evolution에서 확인할 수 있다.

$$
\begin{gathered}
e^{-i(E_r-i\Gamma_r/2)t/\hbar}
\\
=e^{-iE_rt/\hbar}e^{-\Gamma_rt/(2\hbar)}
\end{gathered}
$$

이 amplitude의 절댓값을 제곱하면 **probability**는 $e^{-\Gamma_rt/\hbar}$로 감소한다. 같은 Lorentzian 근사에서 half-height point는 $E_r\pm\Gamma_r/2$이므로 FWHM은 $\Gamma_r$다. 좌우 coupling이 대칭이면 coupling이 약해져도 resonance peak의 높이는 1을 유지할 수 있다. 하지만 폭과 적분 면적은 줄어든다. 따라서 transmission peak가 높다고 해서 전류까지 큰 것은 아니다.

위 계산기에서 resonance peak를 찾고 위치를 정밀하게 계산한 결과는 다음과 같다.

$$
\begin{aligned}
E_r\approx{}&0.01613,\ 0.06387,\ 0.14131,\\
            &0.24526,\ 0.37135,\ 0.51460\ \mathrm{eV}
\end{aligned}
$$

이 목록은 위 계산기의 refinement로 확인한 $0.001$–$0.600\ \mathrm{eV}$ 구간의 resonance다. 원래 노트북의 peak 검출 목록에서는 $0.01613\ \mathrm{eV}$ 부근의 좁은 첫 resonance를 놓쳤다. 여기서는 $0.14131\ \mathrm{eV}$ 부근의 resonance를 택해

$$
E_{\mathrm{res}}=0.1413\ \mathrm{eV},
\qquad
\mu_L=0.201\ \mathrm{eV},
\qquad
\mu_R=0.081\ \mathrm{eV},
\qquad
k_BT=0.005\ \mathrm{eV}=5\ \mathrm{meV}
$$

로 둔다. 표시한 에너지는 반올림한 값이며, 실제 계산에서는 refinement로 구한 $E_{\mathrm{res}}=0.14131276097\ \mathrm{eV}$와 $\mu_L=E_{\mathrm{res}}+0.06\ \mathrm{eV}$, $\mu_R=E_{\mathrm{res}}-0.06\ \mathrm{eV}$를 사용한다. $k_BT=5\ \mathrm{meV}$는 약 $58\ \mathrm{K}$에 해당한다.

<div id="82-available-spectrum과-실제-occupation" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

### 8.2 Available spectrum 중 실제로 채워지는 부분

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

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-09-example-a-spectral-bookkeeping.svg" alt="contact별로 나눈 available well spectrum과 실제 occupied well spectrum" width="1944" height="1301">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-09-example-a-spectral-bookkeeping-mobile.svg" alt="contact별로 나눈 available well spectrum과 실제 occupied well spectrum" width="919" height="2270">
</div>

*그림 9.* 양쪽 contact 모두 같은 resonance에 state를 공급할 수 있다. 이 예제에서 chemical potential을 바꾸어도 resonance의 모양은 변하지 않는다. 달라지는 것은 두 reservoir가 그 resonance를 채우는 비율이며, 두 기여를 합한 결과가 $G^n=A_Lf_L+A_Rf_R$다. 세로축은 spin 하나당 spectral density이며, occupied spectrum을 적분한 뒤 spin degeneracy 2를 곱하면 well의 전체 occupation이 된다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-10-example-a-transport-summary.svg" alt="고정 spectrum resonant-tunneling 예제의 transmission, LDOS, contact별 occupation, spectral current" width="1944" height="2183">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-10-example-a-transport-summary-mobile.svg" alt="고정 spectrum resonant-tunneling 예제의 transmission, LDOS, contact별 occupation, spectral current" width="906" height="4221">
</div>

*그림 10.* (a)에는 가장 낮은 에너지의 좁은 resonance까지 확인할 수 있도록 세밀하게 계산한 transmission spectrum과 주요 occupation window를 표시했다. (b)의 LDOS에서는 well 안에 주로 분포하는 quasi-bound mode를 확인할 수 있다. (c)는 에너지에 대해 적분한 site occupation을 두 contact의 기여로 나눈 결과다. (d)에서 전류를 정하는 양은 transmission peak의 높이가 아니라 $dI/dE$를 적분한 **면적**이다. (c)와 (d)는 아래의 resonance-local grid와 같은 적분 설정으로 다시 계산했다.

이 면적을 정확히 적분하려면 좁은 resonance 부근에 energy point를 충분히 촘촘하게 배치해야 한다. Uniform grid에서는 linewidth보다 점 사이의 간격이 크면 peak 자체를 건너뛸 수 있다. 1차원 band edge처럼 적분값은 유한하지만 함수가 급격히 변하는 구간도 충분히 세분화해야 한다.

<details markdown="1">
<summary>좁은 resonance를 정확히 적분하는 energy grid와 코드</summary>

원래 노트북은 emitter band edge 근처의 geometric grid와 occupied-energy 구간의 linear grid를 합친다. 하지만 peak의 위치를 정밀하게 찾는 것과 peak의 면적을 정확히 적분하는 것은 별개의 문제다. 첫 resonance의 FWHM은 약 $61.3\ \mu\mathrm{eV}$인데, 기존 3,000-point linear grid의 간격은 약 $83.5\ \mu\mathrm{eV}$다. 이 grid로는 well occupation이 수렴하지 않는다.

아래 재계산에서는 전체 구간의 grid에 각 resonance의 linewidth를 기준으로 한 local grid를 추가한다. 각 peak의 양쪽 $20\,\mathrm{FWHM}$ 범위에 점을 더 놓고, 그 안의 간격은 $\mathrm{FWHM}/(100r)$로 정한다. Fermi tail도 포함하도록 적분 상한을 $E_{\max}=\mu_L+18k_BT$까지 늘린다. 수렴 검증에서는 grid 해상도를 정하는 $r$, 적분 상한, 수치적 $\eta$를 각각 바꾸어 본다.

간격이 일정하지 않은 grid $E_0<\cdots<E_{M-1}$에서 trapezoidal rule을 $\int dE\,F(E)\simeq\sum_iw_iF(E_i)$로 쓰면 각 점의 weight는

$$
w_0=\frac{E_1-E_0}{2},\qquad
w_i=\frac{E_{i+1}-E_{i-1}}{2},\qquad
w_{M-1}=\frac{E_{M-1}-E_{M-2}}{2}
$$

로 정해진다. 점 사이의 간격이 다르므로 각 함수값을 단순히 더해서는 안 된다. 코드에서 $w_i$를 곱하면 각 점이 대표하는 energy 구간의 폭이 적분에 반영된다.

```python
E_res = 0.141312760965677
mu_L, mu_R, kT = E_res + 0.06, E_res - 0.06, 0.005

# (peak center, FWHM), eV 단위. 적분 구간 안의 네 resonance를 모두 분해한다.
peak_data = [
    (0.016126494083, 6.12837580566e-5),
    (0.063869428566, 5.35092219254e-4),
    (0.141312760935, 2.09218053569e-3),
    (0.245260414529, 6.10631008345e-3),
]
r = 16                       # r=8과 비교해 적분값의 수렴을 확인
E_max = mu_L + 18 * kT        # occupation window 밖의 Fermi tail도 포함
parts = [
    np.geomspace(1e-10, 1e-3, 400 * r, endpoint=False),
    np.linspace(1e-3, E_max, 3000 * r + 1),
]
for center, width in peak_data:
    local = center + width * np.linspace(-20, 20, 4000 * r + 1)
    parts += [local[(local > 1e-10) & (local < E_max)], np.array([center])]
E = np.unique(np.concatenate(parts))

# 비균일 grid의 각 점이 대표하는 에너지 구간을 weight에 반영한다.
w = np.empty_like(E)
w[0], w[-1] = 0.5 * (E[1] - E[0]), 0.5 * (E[-1] - E[-2])
w[1:-1] = 0.5 * (E[2:] - E[:-2])

# 각 에너지에서 Gn을 구하는 기존 loop 안의 density 누적 항:
n_per_spin += w[iE] * np.diag(Gn).real / (2.0 * np.pi)

# loop가 끝난 뒤: T_E는 같은 E grid에서 계산한 transmission 배열이다.
n_site = 2.0 * n_per_spin                 # spin degeneracy는 한 번만 적용
f_L_E = 1.0 / (1.0 + np.exp((E - mu_L) / kT))
f_R_E = 1.0 / (1.0 + np.exp((E - mu_R) / kT))
I = current_prefactor_A_per_eV * np.sum(w * T_E * (f_L_E - f_R_E))
```

`current_prefactor_A_per_eV`에는 spin factor와 eV–joule 변환 계수가 이미 포함되어 있으므로 2를 다시 곱하면 안 된다. density, bond current, terminal current에는 모두 같은 energy grid와 quadrature weight를 사용한다. 그래야 서로 다른 적분 규칙 때문에 생긴 차이를 물리적 불일치로 오해하지 않는다.

</details>

$\eta=10^{-9}\ \mathrm{eV}$에서 이 grid로 다시 계산한 Landauer current는

$$
I_{\mathrm{Landauer}}\approx263.0911\ \mathrm{nA}
$$

이다. 평균 bond current는 $263.0919\ \mathrm{nA}$, bond 사이의 상대 편차는 $5.30\times10^{-5}$다. Occupied well spectrum을 적분하거나 well site의 occupation을 더해 구한 평균 전자수는 모두 **약 $4.87203$**이며, 두 계산의 차이는 $10^{-14}$ 전자보다 작다. 원래의 coarse grid에서 얻은 평균 전자수 약 $4.88246$은 적분이 수렴하기 전의 값이다. 전류가 기존 값보다 조금 커진 것은 적분 상한을 $\mu_L+10k_BT$에서 $\mu_L+18k_BT$로 늘려 남아 있던 Fermi tail까지 포함했기 때문이다.

같은 grid를 사용한 두 계산이 일치하는지 확인하는 것과 grid 자체의 수렴을 확인하는 것은 구분해야 한다. $r=8$에서 $16$으로 늘리면 well occupation은 $3.88\times10^{-7}$ 전자만큼 변한다. 적분 상한을 $\mu_L+18k_BT$에서 $\mu_L+22k_BT$로 늘렸을 때의 변화는 $1.82\times10^{-8}$ 전자다. 두 경우 모두 Landauer current의 변화는 $10^{-7}\ \mathrm{nA}$보다 작다.

수치적으로 넣은 $\eta$의 영향도 따로 확인할 수 있다. 같은 refined grid에서 $\eta\to0^+$ 극한을 취하면 well occupation은 약 $4.87211$ 전자, current는 약 $263.0914\ \mathrm{nA}$로 수렴한다. 이때 bond current와 Landauer current는 수치 정밀도 안에서 일치한다. 유한한 $\eta$에서 남았던 작은 차이는 앞서 설명한 수치적 absorption 때문이다.

```text
f_L과 f_R을 바꾸면 occupation과 전류가 바뀐다.
H_D와 lead self-energy가 고정되어 있다면 G^R 자체는 바뀌지 않는다.
```

---

## 9. 예제 B: bias에 따라 움직이는 resonance와 NDR

Bias를 주면 두 reservoir의 occupation뿐 아니라 device의 potential energy와 contact band도 달라질 수 있다. 예제 B에서는 electrostatic potential $\phi$가 device를 따라 선형으로 변한다고 가정한다. 그에 따른 electron potential energy $U=-e\phi$를 Hamiltonian에 직접 넣는다. 이 profile은 Poisson equation을 self-consistent하게 풀어 얻은 결과가 아니다.

이 예제의 양의 bias에서는 왼쪽 contact가 주로 전자를 공급하므로 **emitter**, 오른쪽 contact는 **collector**라고 부른다. Contact의 역할을 구분하는 이름이며 device 안에 새로운 영역을 추가한 것은 아니다.

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

가장 낮은 resonance는 처음에 emitter Fermi level보다 $15\ \mathrm{meV}$ 높다. Bias에 따라 resonance가 이동하면서 전류에 기여할 수 있는 energy 구간에 들어왔다가 다시 벗어날 수 있다.

이 bias profile을 코드의 배열로 만들 때에는 앞의 endpoint 식에서 쓴 $1,\ldots,N$ 대신 Python의 zero-based site 번호 $j=0,\ldots,N-1$을 사용한다. 왼쪽 끝에는 추가 potential energy가 없고 오른쪽 끝은 $-eV$만큼 내려간다. 선형 profile은

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

가 된다. $\mu_R$과 오른쪽 lead의 band edge가 같은 양만큼 내려가므로 둘 사이의 에너지 차이는 변하지 않는다. Bias를 두 번 적용한 것이 아니다. $\varepsilon_R$은 collector band의 에너지 범위를 정하고, $\mu_R$은 그 band를 어디까지 채울지 정한다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-11-example-b-bias-profile.svg" alt="zero bias와 finite bias에서의 이중 barrier potential energy 분포와 이동한 contact 기준" width="1944" height="832">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-11-example-b-bias-profile-mobile.svg" alt="zero bias와 finite bias에서의 이중 barrier potential energy 분포와 이동한 contact 기준" width="900" height="1139">
</div>

*그림 11.* 선형 경사는 Poisson equation을 풀어 얻은 결과가 아니라 계산에 직접 넣은 bias profile이다. 이 profile에 따라 well과 barrier의 potential energy에 선형 경사가 생기고, collector의 에너지 기준도 내려간다.

<div id="91-삼중대각-구조를-이용한-on-계산" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

### 9.1 Tridiagonal matrix를 이용한 $O(N)$ 계산

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
<summary>코드: tridiagonal linear system으로 endpoint transmission 구하기</summary>

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

으로 계산한다. 이 예제에서는 $V$를 높일 때 $f_L$과 온도는 그대로 두고 $\mu_R$만 낮춘다. $T$가 고정되어 있다면 전류는 늘거나 일정해질 뿐, 줄어들 수는 없다. 각 에너지에서 $f_R$이 작아지고 $T\ge0$이기 때문이다. Bias에 따라 Hamiltonian이 바뀌는 경우에는 전류의 기울기에 두 항이 생긴다.

$$
\begin{aligned}
\frac{dI}{dV} &= S_{\rm occ}+S_T,\\
S_{\rm occ}&=\frac{2e}{h}\int dE\,T\,\partial_V(f_L-f_R),\\
S_T&=\frac{2e}{h}\int dE\,(\partial_V T)(f_L-f_R).
\end{aligned}
$$

그림 12(b)의 점선은 각 bias의 transmission을 사용해 구한 occupation 항 $S_{\rm occ}$다. 다른 device의 전류를 미분한 곡선이 아니다. Occupation 항은 음수가 되지 않는다. Transmission 변화로 전류가 줄어드는 효과가 이 증가분을 넘을 때에만 전체 기울기가 음수가 된다. 상수 prefactor를 빼면 전류 적분에 들어가는 양은

$$
\text{current integrand}
=T(E,V)[f_L(E)-f_R(E)]
$$

이다. Current는 이 함수의 특정 에너지에서의 값이 아니라 에너지에 대해 적분한 면적으로 정해진다. Lead $\alpha$의 propagating band bottom을 $E_{c,\alpha}=\varepsilon_\alpha-2t_0$로 쓰자. On-site matrix element $\varepsilon_\alpha$와 달리, 이 값은 해당 lead에 propagating mode가 존재할 수 있는 최저 에너지다. Zero temperature이고 $\mu_L\ge\mu_R$인 $\eta\to0^+$ 극한에서, 전류에 기여하려면 에너지가 다음 세 구간에 모두 속해야 한다.

$$
[\mu_R,\mu_L]
\cap[E_{c,L},E_{c,L}+4t_0]
\cap[E_{c,R},E_{c,R}+4t_0]
$$

첫 번째는 두 reservoir 사이의 occupation window이고, 나머지 두 구간은 각 contact의 propagating band다. Zero temperature에서는 이 교집합 밖의 에너지에서 current integrand가 0이다. Peak 중심이 교집합 밖에 있더라도 tail 일부가 교집합 안에 있으면 그 부분은 전류에 기여할 수 있다. 온도가 0보다 높으면 occupation은 window의 경계에서 완만하게 바뀌지만, 양쪽 lead에 propagating state가 있어야 한다는 조건은 그대로다.

세 구간을 따로 표시하면 current가 줄어든 원인도 구분할 수 있다. resonance가 occupation window를 벗어났는지, emitter band 또는 collector band와 겹치지 않는지를 각각 확인하면 된다.

가장 낮은 resonance가 occupation window 안에 있고, 같은 에너지의 propagating state가 emitter에도 있으면 current가 커질 수 있다. Bias를 더 높이면 resonance가 전류에 기여할 수 있는 구간에서 벗어난다. 또한 resonance가 고정된 emitter band edge에 가까워지면 contact broadening이 작아져 injection도 약해진다. 이 예제에서는 $E_{c,L}=0$이므로 band edge 근처에서

$$
\gamma_L(E)\simeq2\sqrt{t_0E}
$$

처럼 $\gamma_L$이 작아진다. Emitter band의 아래쪽 끝이 0이 아니면 $E$ 대신 $E-E_{c,L}$를 쓴다. 따라서 resonance가 band 아래로 완전히 내려가기 전부터 injection이 약해질 수 있다. Occupation window는 넓어졌는데도 $T(E,V)[f_L-f_R]$의 적분값은 오히려 감소하는 것이다.

current가 감소하는 bias 구간에서는

$$
\frac{dI}{dV}<0
$$

이고, 이를 negative differential resistance (NDR)라고 한다. NDR은 $I/V$가 음수라는 뜻이 아니라 differential conductance $dI/dV$가 음수라는 뜻이다.

<div class="negf-figure-desktop">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-12-example-b-ndr-summary.svg" alt="NDR 예제의 전류-전압 곡선, differential conductance, resonance-transmission map, peak와 valley의 spectral current" width="1944" height="3050">
</div>
<div class="negf-figure-mobile">
  <img src="{{ site.baseurl }}/assets/img/posts/2026-07-22-negf-from-scratch/fig-12-example-b-ndr-summary-mobile.svg" alt="NDR 예제의 전류-전압 곡선, differential conductance, resonance-transmission map, peak와 valley의 spectral current" width="928" height="5031">
</div>

*그림 12.* (a) $0.085\ \mathrm{V}$의 첫 전류 peak와 $0.149\ \mathrm{V}$의 valley로 첫 구간의 peak-to-valley current ratio(PVCR)를 구한다. (b) Occupation 항은 음수가 아니지만 전체 기울기는 음수가 된다. (c) 첫 transmission ridge가 고정된 emitter band edge에 가까워진다. (d) 두 곡선을 같은 축에 그렸으며, 색을 채운 면적이 peak와 valley의 전류다. (e) 약 $0.4\ \mathrm{V}$ 부근의 NDR은 다른 resonance에서 생긴다.

아래에 명시한 energy·voltage grid에서 첫 번째 peak와 그 뒤 valley의 값은

$$
I_{\mathrm{peak}}=57.3122\ \mathrm{nA},
\qquad
I_{\mathrm{valley}}=0.6519\ \mathrm{nA},
\qquad
\mathrm{PVCR}
=\frac{I_{\mathrm{peak}}}{I_{\mathrm{valley}}}
=87.92
$$

이다. 이 값은 아래의 촘촘한 integration grid에서 직접 계산했다. 반면 heatmap은 resonance ridge가 이동하는 모습을 보여 주기 위해 비교적 성긴 grid로 그렸다.

정량적인 current를 구할 때에는 energy 간격을 $dE=0.05\ \mathrm{meV}$로 줄이고, 첫 peak와 valley 부근에서는 bias 간격도 $1\ \mathrm{mV}$로 줄인다. 각 bias point마다 device potential, collector band reference, collector chemical potential을 새로 계산한다.

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

Heatmap이 매끄럽다는 사실만으로 linewidth가 좁은 resonance의 적분 면적까지 수렴했다고 판단할 수는 없다. 그래서 plotting grid와 integration grid를 따로 둔다. 노트북에서는 더 촘촘한 integration grid로 계산한 current data를 미분하고, peak-prominence 조건으로 작은 수치 요동을 제거한 뒤 첫 peak와 그 뒤의 valley를 찾는다.

(d)에서는 peak와 valley의 두 곡선을 같은 $dI/dE$ 축 범위에 그렸다. 녹색 valley 곡선이 거의 보이지 않는 것은 실제 적분 면적이 훨씬 작기 때문이다. 각 곡선을 따로 normalize해서 생긴 차이가 아니다.

이 예제에서는 bias가 커지면서 resonance가 전류에 기여할 수 있는 energy 구간에서 벗어나고, emitter band edge 부근에서는 contact broadening이 작아져 injection도 약해진다. 두 효과가 함께 NDR에 기여한다. 이는 외부에서 정한 선형 potential ramp와 1차원 single band를 사용한 현재 model의 결과다. 모든 resonant-tunneling diode에서 같은 microscopic mechanism으로 NDR이 생긴다고 일반화할 수는 없다.

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
- **equilibrium:** contact continuum에서 $G^n=f(A_L+A_R)$를 확인한 상대오차는 $7.26\times10^{-17}$다.
- **terminal current:** terminal formula가 Landauer integrand로 줄어드는 상대오차가 $1.67\times10^{-16}$이다.
- **예제 A:** bond별 current는 유한한 $\eta$로 예상되는 편차 안에서 거의 일정하며, 독립적으로 계산한 Landauer integral과 일치한다.
- **fast solver:** $O(N)$ endpoint calculation이 dense-matrix trace와 일치한다.
- **NDR integration:** peak와 valley의 $dI/dE$ 면적을 적분하면 표시한 current가 그대로 나온다.

검증마다 찾아낼 수 있는 오류가 다르다. matched-wire test가 실패하면 contact matching과 self-energy의 부호부터 확인한다. Schur test가 실패하면 block 순서와 Hermitian conjugate를 점검한다. bond마다 current가 다르면 $G^n$의 index, bond 방향, energy resolution, spin factor의 중복 여부를 살펴봐야 한다. 어느 한 검증만 통과했다고 해서 나머지 오류까지 없다고 볼 수는 없다.

---

<div id="11-현재-모형에서-포함하지-않은-물리" style="scroll-margin-top: 5rem" aria-hidden="true"></div>

## 11. 현재 model에 포함하지 않은 물리

Lead elimination과 균일 lead의 surface Green's function은 지금 정한 single-particle lattice Hamiltonian에 대해 정확하다. 반면 continuum discretization, isolated-resonance projection, 외부에서 정한 bias profile은 근사이며, 앞의 정확한 대수 관계와 구분해야 한다. 각각의 수렴 조건이나 적용 조건도 따로 확인해야 한다. 현재 model의 범위와 가정은 다음과 같다.

- electron-electron 또는 electron-phonon interaction을 나타내는 self-energy를 포함하지 않는다. Single-particle 또는 mean-field 수준에서 기술한다.
- phase coherence가 유지되는 elastic transport만 다룬다.
- lead는 균일하고 single-channel이며 semi-infinite다.
- effective mass는 일정하고 lattice는 1차원이다.
- spin은 degeneracy factor 2로만 반영하며 spin-orbit coupling이나 magnetic structure는 다루지 않는다.
- 예제 B는 NEGF–Poisson을 self-consistent하게 풀지 않고 선형 potential-energy drop을 가정한다.
- 두 lead continuum과 분리된 bound state의 occupation을 정하려면 초기 상태를 따로 지정하거나 relaxation mechanism을 추가해야 한다.

이 제한들은 NEGF formalism 자체의 한계가 아니다. multi-orbital contact에서는 surface self-energy가 scalar가 아니라 matrix가 된다. scattering을 포함하려면 retarded self-energy와 in-scattering self-energy를 추가한다. electric field까지 self-consistent하게 구할 때에는 $G^n$에서 density를 계산하고, 그 density로 $H_D$를 갱신한다. 세부 model이 달라져도 retarded propagation과 occupation을 분리하는 계산 구조는 그대로 유지된다.

어떤 model을 쓸지는 전류를 좌우하는 물리에 따라 달라진다. Barrier 사이의 phase interference가 중요하면 subband별 occupation만으로는 부족하다. 반대로 transverse confinement는 중요하지만 길이 방향의 phase를 버려도 된다면, [MS-BTE]({{ site.baseurl }}/posts/Multi-Subband-Boltzmann-Transport-From-Scratch-KR/)로 subband별 occupation과 subband 사이의 transition을 계산할 수 있다. 공간적으로 균일하고 equilibrium에 가까운 상태에서 momentum relaxation이 주된 효과라면, [Drude model]({{ site.baseurl }}/posts/Drude-Model-From-Scratch-KR/)은 전체 분포 대신 drift velocity를 구한다. 이 두 근사만으로는 예제 A의 좁은 coherent resonance를 재현할 수 없다.

---

## 12. 노트북으로 직접 계산하기

기본 NEGF 구현과 두 transport 예제는 아래 노트북에서 실행할 수 있다. 이 글의 예제 A 수치는 8.2절에 제시한 refined energy grid로 다시 계산했으므로, 원래 노트북의 coarse-grid 출력과는 차이가 있다.

- [Kaggle에서 노트북 실행하기](https://www.kaggle.com/code/pilkwang/from-closed-to-open-quantum-transport-with-negf)
- [GitHub에서 소스 노트북 읽기](https://github.com/pilkwangkim/Physics/blob/master/negf_from_scratch.ipynb)

본문에 나온 순서대로 notebook cell을 실행하면 각 식과 수치 결과를 함께 확인할 수 있다. 그다음에는 격자 간격, barrier 모양, contact hopping, reservoir chemical potential, bias profile 가운데 하나씩 바꾸고 같은 검증을 반복하면 된다. 이렇게 하면 parameter를 바꾸어 생긴 물리적 변화와 부호·단위·index 오류를 구분할 수 있다.

### 참고문헌

- K. Y. Camsari, S. Chowdhury와 S. Datta, [*The Non-Equilibrium Green Function (NEGF) Method*](https://arxiv.org/html/2008.01275v2), 2–4절: single-particle equation에서 contact self-energy, source correlation, current를 유도한다.
- D. S. Fisher와 P. A. Lee, [*Relation between conductivity and transmission matrix*](https://doi.org/10.1103/PhysRevB.23.6851), *Physical Review B* **23**, 6851 (1981): Green's function과 flux-normalized scattering amplitude의 관계를 다룬다.
- X. Waintal 등, [*Computational quantum transport: a scattering approach perspective*](https://arxiv.org/html/2407.16257v2), 5·7·8절: lead elimination, occupied scattering state, NEGF와 scattering formulation의 관계를 설명한다.
