---
title: "Channel constriction과 quantum transport: 인용 관계로 읽는 연구 흐름"
description: "Channel access geometry, mode coupling, underlap doping을 다룬 논문들의 인용 관계. Channel constriction이 전류에 미치는 영향을 모델과 비교 조건에 따라 읽는다."
date: 2026-11-13 17:15:00 +0900
categories: [Physics, Quantum Transport]
tags: [quantum-transport, bibliometrics, nanosheet, negf, semiconductor]
topic: quantum-transport
lang: ko
translation_key: quantum-access-resistance-literature-map
interactive: quantum-access-literature
math: true
comments: false
pin: false
published: true
---

[Read in English]({{ site.baseurl }}/posts/Quantum-Access-Resistance-Literature-Map/)

넓은 source에서 좁은 channel로 전자가 들어갈 때는 channel 안에 별도의 scattering mechanism이 없어도 electron wave가 반사될 수 있다. 두 영역의 confinement 조건이 다르면 전자에 허용되는 transverse quantum state, 즉 **mode**가 달라지고, 두 영역의 mode가 서로 연결되는 방식도 바뀌기 때문이다. 실제 transistor에서는 doping, electrostatic potential, contact, collision도 전류에 영향을 준다. **Quantum access resistance**는 reservoir에서 channel로 전자를 주입할 때 생기는 quantum transport의 제약과 관련된 저항이다. 그 크기는 이 요인들이 함께 작용한 결과에 따라 달라진다.

지도에는 기준이 되는 **논문 6편**을 포함해 **논문 {{ site.data.quantum_access_literature.nodes.size }}편**과 **인용 관계 {{ site.data.quantum_access_literature.edges.size }}개**가 담겨 있다. 연도별로 따라가면 연구자들이 어떤 질문을 던졌고 어떤 계산 방법을 사용했는지 볼 수 있다. 이 지도는 선별한 문헌의 관계를 보여 주므로, 연도별 논문 수를 분야 전체의 성장 추세로 해석할 수는 없다.

{% include interactive/quantum-access-literature.html %}

## 인용 관계와 공통 참고문헌

**인용 관계**에는 방향이 있다. A에서 B로 향하는 화살표는 A가 B를 인용했다는 뜻이다. A가 B의 방법을 사용했는지, 결과를 비교했는지, 해석에 이견을 제시했는지는 인용한 문맥을 읽어야 알 수 있다.

**공통 선행연구** 화면에서는 두 논문이 어떤 참고문헌을 공유하는지 볼 수 있다. A와 B가 모두 R을 인용하면 **A → R ← B**로 연결된다. 이를 **bibliographic coupling**이라고 한다. A가 B를 인용하거나 B가 A를 인용하지 않아도 이런 관계가 생긴다. 예를 들어 Venugopal2004와 Laux2004는 DOI로 확인한 참고문헌 4편을 공유한다. QDAME2002, real space와 mode space를 비교한 Venugopal2002, Svizhenko2002, Datta2000이다. 두 논문이 어떤 방법과 개념을 공유하는지는 알 수 있지만, 물리적 해석과 결론까지 모두 같다는 뜻은 아니다.

공통 참고문헌 수는 출판사가 등록한 참고문헌 중 DOI가 정확히 일치하는 항목을 세었다. DOI가 없는 항목은 제외했으므로, 0은 등록된 기록에서 공통 DOI를 찾지 못했다는 뜻이다. 지도에서 논문을 선택하면 연구 질문, 계산 방법, 결과, 설계상 의미와 함께 어떤 자료까지 확인했는지 볼 수 있다. 연구 흐름에서 항목을 선택하면 해당 시기의 논문으로 이동한다.

## Channel constriction에서 달라지는 것

Reservoir에서 좁은 channel로 이어지는 junction에서는 여러 변화가 함께 일어날 수 있다. propagating mode 수가 줄고, 단면이 급격히 바뀌는 곳에서 반사가 생기거나 transverse mode들이 섞인다. Confinement energy도 달라지며, doping과 electrostatic potential 분포에 따라 electrostatic barrier가 생길 수 있다. Collision은 junction으로 들어오는 carrier의 분포를 다시 바꾼다. 이를 모두 “constriction barrier”라고 부르면, 각 논문이 어떤 효과를 비교해 분리했는지 구분하기 어렵다.

충분히 낮은 온도에서 coherent·elastic한 two-terminal transport를 생각하자. Linear response 영역의 conductance는 Landauer 식으로

$$
G=\frac{2q^2}{h}\sum_n T_n(E_F)
$$

주어진다. $q>0$는 기본전하의 크기, $h$는 Planck 상수다. $T_n$은 0과 1 사이의 transmission eigenvalue이며, 앞의 2는 spin degeneracy를 가정한 값이다. 그 밖의 전도 channel은 합에 포함하고, 해당하는 경우 valley channel도 센다. Mode 수가 전달 가능한 conductance의 크기를 정한다면, transmission은 그중 실제로 얼마나 전달되는지를 정한다. 유한한 bias에서는 $E_F$ 한 점의 전자만으로 전류를 계산할 수 없고, 일정한 에너지 구간의 기여를 함께 봐야 한다. [van Wees et al.](https://doi.org/10.1103/PhysRevLett.60.848), [Datta](https://doi.org/10.1006/spmi.2000.0920)

위치에 따른 subband edge를 에너지 기여별로 나누면 개략적으로

$$
U_n(x)=E_{c0}(x)-q\phi(x)+E_{\mathrm{conf},n}(x)
$$

처럼 쓸 수 있다. 진행 방향의 기준을 정하면 $E_{c0}$는 band-edge offset, $\phi$는 electrostatic potential, $E_{\mathrm{conf},n}$은 그 기준에서 잰 transverse confinement energy다. 단면이 좁아지면 confinement energy가 올라갈 수 있다. Doping이 낮아지면 self-consistent potential이 달라질 수 있다. 두 변화 모두 carrier가 넘어야 할 에너지 장벽에 영향을 준다.

이 식은 **diagonal energy term의 구성**을 나타내며, coupled-mode 모델 전체를 담고 있지는 않다. transverse potential이 바뀌면 mode의 모양도 달라지고, 이 mode들을 연결하는 과정에서 off-diagonal coupling이 생긴다. 따라서 1차원 장벽의 모양만으로 모든 mode conversion을 설명할 수는 없다. Mode coupling을 포함한 모델에서도 electrostatics를 올바르게 풀어야 한다. [Polizzi and Ben Abdallah](https://doi.org/10.1016/j.jcp.2004.07.003), [Luisier et al.](https://doi.org/10.1063/1.2244522)

Electrostatic potential $\phi$를 같게 맞춰도 단면이 달라지면 confinement energy는 달라질 수 있다. 따라서 같은 potential을 썼다는 조건이 Hamiltonian 전체가 같다는 뜻은 아니다. 이런 형상 비교에는 confinement의 변화도 남으므로, off-diagonal mode conversion만 분리한 결과로 해석해서는 안 된다. **Mode mismatch**라는 용어도 문헌에 따라 mode 수의 불일치, junction의 전체 반사, transverse mode 사이의 변환을 가리킬 수 있으므로 문맥을 확인해야 한다.

## 출발점이 된 논문 6편

P1–P6이 지도의 출발점인 논문 6편이다. 온라인 공개일과 권호 발행일이 다를 때는 학술지 권호의 연도를 사용했다.

| 논문 | 연도 | 전체 제목과 DOI |
|:--|:--|:--|
| P1 · Hung et al. | 2025 | **On-Current Degradation in Ultra-Scaled Nanosheet FETs With S/D Underlap Doping** · [10.1109/LED.2025.3614369](https://doi.org/10.1109/LED.2025.3614369) |
| P2 · Venugopal et al. | 2004 | **Quantum mechanical analysis of channel access geometry and series resistance in nanoscale transistors** · [10.1063/1.1631754](https://doi.org/10.1063/1.1631754) |
| P3 · Laux et al. | 2004 | **Analysis of quantum ballistic electron transport in ultrasmall silicon devices including space-charge and geometric effects** · [10.1063/1.1695597](https://doi.org/10.1063/1.1695597) |
| P4 · Ahn and Hong | 2023 | **Geometric Scattering Describing Mode-Coupling Effects in Non-Uniform Cross-Sections for Non-Equilibrium Green’s Function and Multi Subband Boltzmann Transport Equation Solvers** · [10.1109/IEDM45741.2023.10413710](https://doi.org/10.1109/IEDM45741.2023.10413710) |
| P5 · Luisier et al. | 2006 | **Quantum transport in two- and three-dimensional nanoscale transistors: Coupled mode effects in the nonequilibrium Green’s function formalism** · [10.1063/1.2244522](https://doi.org/10.1063/1.2244522) |
| P6 · Pourghaderi et al. | 2026 | **On Effectiveness of Constriction Barrier in Nanotransistors** · [10.1109/LED.2026.3681884](https://doi.org/10.1109/LED.2026.3681884) |

## Point contact에서 transistor의 access region까지

1988년 van Wees et al.과 Wharam et al.은 point-contact 실험에서 횡방향 subband에 따라 conductance와 저항이 계단식으로 변하는 것을 보였다. 짧은 ballistic channel을 극저온에서 측정한 결과다. Szafer와 Stone의 1989년 이론, Nonoyama et al.의 1992년 wide–narrow 계산은 junction 모양과 quantum state의 연결 방식이 transmission에 미치는 영향을 다뤘다. 이 연구들이 남긴 핵심 질문은 서로 다른 영역의 quantum state를 어떻게 연결하느냐는 것이다. 이 결과에서 상온 FET에 공통으로 적용할 전류 손실값을 얻을 수는 없다. [Szafer and Stone](https://doi.org/10.1103/PhysRevLett.62.300), [Nonoyama et al.](https://doi.org/10.1063/1.106974)

전자가 드나드는 open device를 계산하려면 파동을 주입하고 흡수하는 reservoir도 모델에 포함해야 한다. Lent와 Kirkner의 quantum transmitting boundary method, QDAME 연구, real-space NEGF–Poisson 계산은 이를 서로 다른 수치 방법으로 구현한다. Datta2000은 quantum transport와 electrostatics를 함께 풀어야 한다고 강조한다. Venugopal2002는 계산 차원을 줄인 decoupled mode-space 모델을 real-space 결과와 비교했다. 이런 차원 축소가 유효한 범위는 따로 확인해야 하며, 모든 형상에 같은 근사를 적용할 수 있는 것은 아니다. [Lent and Kirkner](https://doi.org/10.1063/1.345156), [Venugopal et al.](https://doi.org/10.1063/1.1503165)

2004년의 두 논문은 channel access geometry를 서로 다른 조건에서 분석한다. Laux와 동료들은 phonon·impurity collision이 없는 **ballistic transport**를 self-consistent하게 계산해 straight, taper, dog-bone 형상을 비교한다. Venugopal과 동료들은 Poisson 방정식과 결합한 **nonequilibrium Green’s function(NEGF)** 계산에 근사적인 scattering 모델을 넣는다. Scattering은 넓은 reservoir의 전자가 channel의 몇 안 되는 mode로 들어가는 방식을 바꾸며, 형상에 따른 전류 차이를 줄일 수 있다. 두 논문은 모두 2002년 QDAME 방법론 논문을 인용한다. [Laux et al.](https://research.ibm.com/publications/analysis-of-quantum-ballistic-electron-transport-in-ultrasmall-silicon-devices-including-space-charge-and-geometric-effects), [Venugopal et al.](https://docs.lib.purdue.edu/nanodocs/188/)

여기서 형상 간 전류 차이가 줄었다고 해서 소자의 총전류가 늘었다고 볼 수는 없다. Scattering에 의한 mode 사이 재분배가 injection에 유리하게 작용하더라도, 추가된 scattering 자체는 별도의 수송 손실을 만들 수 있다.

Luisier et al.은 이런 비교에 앞서 coupled-mode 계산이 갖춰야 할 수치적 조건을 다룬다. 단면이 급격히 바뀌는 곳에서 전하와 전류를 복원하려면 mode 사이의 correlation을 빠뜨리지 않아야 한다. 필요한 항을 누락하면 charge spike가 생기거나 전류 보존이 깨질 수 있다. 해당 항을 포함한 ballistic 계산은 real-space 결과와 일치한다. 따라서 계산에 포함하는 mode 수를 늘려 수렴을 확인하고 복원식을 점검하는 것은 수치적 정확성을 검증하는 일이다. 여기에 새로운 물리적 scattering mechanism을 추가하는 것은 아니다. [저자 공개 원문](https://iis-people.ee.ethz.ch/~schenk/JApplPhys_100_043713.pdf)

개별 dopant의 위치와 roughness까지 영향을 주면 형상만으로 소자를 설명하기가 더 어려워진다. Berrada et al.(2015)은 access region의 형상과 개별 dopant의 위치가 quasi-localized state를 통한 injection에 영향을 준다는 점을 보인다. 평균 doping 농도가 같아도 개별 소자의 상태까지 같아지는 것은 아니다. Ahn과 Hong(2023)은 단면이 일정하지 않을 때의 mode-coupling 효과를 geometric scattering으로 표현해 NEGF와 multi-subband Boltzmann transport equation(MS-BTE) solver에 반영한다. 여기서 geometric scattering은 형상 변화에 따른 mode mixing을 나타낸다. Phonon·impurity collision으로 생기는 scattering과는 원인을 구분해서 해석해야 한다. [Berrada et al.](https://doi.org/10.1063/1.4933392), [Ahn and Hong](https://doi.org/10.1109/IEDM45741.2023.10413710)

## 비교 조건이 중요한 이유

Stacked nanosheet gate-all-around FET에서는 넓은 bulk source/drain을 얇은 sheet에 연결하는 구조가 실제 설계 문제가 된다. Kim et al.(2025)은 이 구조를 모델링해 doping이 낮은 constriction에서 강한 반사가 생길 수 있음을 보인다. 결과에는 confinement energy의 변화뿐 아니라 scattering, doping, contact depletion도 영향을 준다. Source/drain 전체에 균일한 고농도 doping을 가정한 비교와는 injection 조건이 다르다. 이전 조건에서 작았던 효과도 injection 환경이 달라지면 커질 수 있다. [공개 원문](https://www.nature.com/articles/s44172-025-00435-0)

이 연구에서 계산한 구조에서는 contact를 깊게 하거나 extension을 짧게 할수록 injection이 좋아지는 것은 아니다. 모델에 사용한 constriction에서는 적절한 contact 깊이와 충분한 extension 길이가 유리할 수 있다. Contact 면적을 넓히고 저항 경로를 줄인다는 고전적인 직관만으로는 이런 경향을 설명하기 어렵다. Mode가 junction을 통과하는 조건과 self-consistent potential이 형성되는 과정도 함께 봐야 한다. [Kim et al., Discussion](https://www.nature.com/articles/s44172-025-00435-0)

Hung et al.은 doping이 부족한 source/drain extension에 생기는 secondary barrier와 linear regime에서의 비정상적인 전류 포화를 분석한다. Subband BTE로 전류를 계산하면서 potential profile의 역할을 함께 다룬다. Underlap에 생기는 장벽과 geometry에만 기인한 mode mismatch는 모두 on-current를 낮출 수 있지만, 원인은 구분해야 한다. [공개 원문](https://repositum.tuwien.at/bitstream/20.500.12708/227257/1/Hung-2025-IEEE%20Electron%20Device%20Letters-vor.pdf)

Pourghaderi et al.은 **electrostatic potential을 같게 맞춘 뒤** transmission을 비교해 mode mismatch의 영향을 살펴본다. **공개 초록**에 따르면 **nominal overlap junction**에서는 transmission에 미치는 영향이 매우 작다. 본문을 확보하지 못했으므로 이 결과는 초록에 명시된 비교 조건 안에서 해석해야 한다. 모든 doping profile이나 junction 설계에서 constriction 효과가 사라진다는 근거로 확대할 수는 없다. [공개 초록](https://www.researchgate.net/publication/403634150_On_effectiveness_of_constriction_barrier_in_nanotransistors)

어떤 성능을 비교하느냐에 따라서도 결론이 달라진다. Pech와 Schulz(2025)는 time-dependent coupled-mode density-matrix 계산법을 제시하고 NEGF 결과와 비교한다. Ballistic 조건의 예제에서 channel을 좁힌 소자와 균일한 channel을 가진 소자는 steady-state 전류가 다르다. 하지만 operating point를 맞추면 amplifier의 거동은 비슷하다. 이 RF 결과가 뒷받침하는 범위는 해당 operating point와 amplifier 지표의 비교이며, 모든 geometry 효과가 작다는 뜻은 아니다. [Pech and Schulz](https://pubs.aip.org/jap/article/138/12/125705/3364353/Coupled-mode-effects-in-the-stationary-and)

Hung et al.의 2026년 p-type nanosheet 연구는 부족한 extension doping, stress relaxation, inner-spacer interface trap이 injection에 미치는 영향을 함께 다룬다. 연구 대상 소자가 injection-limited regime에 들어가면 이 효과들의 영향이 더 커진다. 특정 소자에서 여러 mechanism이 결합한 결과이므로, 모든 전류 손실을 coherent geometric reflection만으로 설명할 수는 없다. [저자 소속 기관의 설명](https://www.globaltcad.com/performance-degradation-mechanisms-in-p-type-nanosheet-fets-role-of-source-draindoping-stress-and-inner-spacer-interface-traps/)

## 비교 조건에 따라 답할 수 있는 질문

같은 물리량을 비교하는 것처럼 보여도 무엇을 같게 두고 무엇을 바꿨는지에 따라 답할 수 있는 질문이 달라진다.

| 비교 | 맞추거나 명시할 조건 | 비교로 확인할 수 있는 것 |
|:--|:--|:--|
| Coupled mode space와 real space | 같은 Hamiltonian, grid, boundary condition과 충분한 mode 수 | 계산 차이가 수치적 차원 축소에서 생기는가 |
| 같은 potential에서 constricted·uniform 형상 비교 | Electrostatic potential과 injection 조건 | Confinement 변화를 포함한 geometry·mode mismatch가 transmission에 얼마나 기여하는가 |
| Self-consistent 계산에서 constricted·uniform 형상 비교 | Bias, 물질·doping 가정; potential은 형상에 따라 달라지도록 둠 | Geometry와 electrostatics가 함께 전류를 어떻게 바꾸는가 |
| Ballistic 계산과 scattering을 포함한 계산 | Geometry, band model, bias, dopant | Relaxation이 injection과 geometry 의존성을 어떻게 바꾸는가 |
| Doping이나 contact를 바꾸는 계산 | 어떤 profile·contact 가정을 바꾸는지 명시 | Depletion이나 injection 조건 때문에 병목이 생기는가 |
| Operating point를 맞춘 RF 비교 | Bias를 맞추는 기준, 입력 신호, 회로 성능 지표 | DC 전류의 차이가 선택한 amplifier 지표에서도 나타나는가 |

소자를 설계할 때는 transverse mode가 어떻게 연결되는지와 injection 영역의 potential이 어떻게 형성되는지를 함께 봐야 한다. Source/drain 부피를 늘리거나, extension을 줄이거나, doping gradient를 바꾸면 고전적인 저항의 단면적·길이 관계만으로 설명할 수 없는 변화가 생긴다. 목표 bias에서 mode coupling, scattering, contact를 적절히 모델링한 뒤 설계 변경의 효과를 판단해야 한다. 각 문헌의 결과도 이런 조건 안에서 비교해야 하며, 하나의 constriction 손실값을 모든 transistor에 적용할 수는 없다.

## 데이터와 근거

데이터 수집일은 **2026년 11월 13일**이다. 제목, 식별자, 날짜, 피인용 수는 [OpenAlex](https://api.openalex.org/works)와 [Crossref](https://api.crossref.org/works)에서 가져왔다. 각 인용선에는 출판사가 등록한 reference key와 함께 DOI 또는 서지정보가 일치한다는 근거를 남겼다. OpenAlex에서도 같은 인용 관계가 확인되는지 기록했다. 논문별 상세 정보에서는 초록만 확인했는지, 저자 manuscript나 출판 원문까지 읽었는지를 구분했다.

등록·색인된 참고문헌에는 누락이 있을 수 있다. 따라서 지도에 화살표가 없어도 실제 인용이 없다고 단정할 수는 없다. 피인용 수는 수집일에 조회한 값이다. 논문의 연도는 온라인 선공개일과 권호 발행일이 뒤섞이지 않도록 권호 기준으로 정리했다. [지도 데이터]({{ site.baseurl }}/assets/data/quantum-access-literature.json)에서 각 인용선의 reference key와 근거 URL을 확인할 수 있다.

## 전체 문헌

<details markdown="1">
<summary>발행연도순으로 정리한 전체 {{ site.data.quantum_access_literature.nodes.size }}편</summary>

{% assign literature_papers = site.data.quantum_access_literature.nodes | sort: "year" %}
{% for paper in literature_papers %}
- **{{ paper.year }} · {{ paper.label }}** — {{ paper.title }}. [DOI: {{ paper.doi }}]({{ paper.url }}).
{% endfor %}

</details>
