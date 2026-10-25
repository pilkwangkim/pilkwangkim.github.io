---
title: "갈루아 이론: 5차방정식의 근의 공식에 대하여"
description: "사칙연산과 제곱근·세제곱근 같은 연산으로 모든 5차방정식을 풀 수 있을까? 근 사이의 관계를 보존하는 변환의 구조가 그 가능성을 결정한다."
date: 2026-09-25 09:00:00 +0900
categories: [Algebra]
tags: [algebra, mathematics, abstract-algebra, field-theory, galois-theory, group-theory, python]
math: true
interactive_labs: [galois]
pin: false
topic: algebra
lang: ko
translation_key: galois-theory-from-scratch
image:
  path: /assets/img/posts/2026-10-25-galois-from-scratch/cover-galois.png
  alt: "5차방정식의 근을 바꾸는 변환 60개와 이들을 연결한 그래프. 옆의 식은 이 변환 구조 때문에 제곱근·세제곱근 같은 연산만으로 근을 표현할 수 없다는 결론과 관련된다."
  width: 1600
  height: 900
  fit: contain
  hide_caption: true
---

[Read in English]({{ site.baseurl }}/posts/Galois-Theory-From-Scratch/) · [직접 실행하는 Python notebook]({{ site.baseurl }}/assets/notebooks/galois-from-scratch-ko.ipynb) · [Permutation group Python 코드]({{ site.baseurl }}/assets/code/galois-reference.py)

$a\ne0$인 2차방정식 $ax^2+bx+c=0$의 근의 공식은 익숙하다.

$$
x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}.
\tag{1}
$$

위 공식의 $\sqrt{b^2-4ac}$는 제곱근을 나타낸다. $\sqrt[3]2$ 같은 세제곱근이나 더 높은 정수 차수의 근을 나타내는 표현도 같은 종류이며, 이런 표현을 **radical**이라고 한다. 이런 근을 구하는 연산과 사칙연산을 유한 번 조합해 만든 식이 **radical expression**이다. 예를 들어 $1+\sqrt2$나 $\sqrt{2+\sqrt3}$가 여기에 해당한다.

3차와 4차방정식에도 사칙연산과 radical로 근을 나타내는 공식이 있다. 하지만 5차에서는 계산이 복잡해지는 것만이 문제가 아니다. **이 연산들을 유한 번 사용해서는 근을 나타낼 수 없는 방정식이 실제로 존재한다.**

그렇다고 근이 없거나 컴퓨터로 근삿값을 구할 수 없다는 뜻은 아니다. **원하는 정확도까지 근삿값을 구하는 것**과 **정해진 연산만으로 정확한 값을 표현하는 것**은 서로 다른 문제다.

근을 radical로 표현할 수 있는지는 **근들을 어떤 방식으로 서로 바꿀 수 있는가**와 관련이 있다. 여기서 아무렇게나 바꾸어도 된다는 뜻은 아니다. 예를 들어 $x^2-2=0$의 두 근 $\sqrt2$와 $-\sqrt2$를 맞바꾸어도 제곱값은 여전히 $2$이고, 두 근의 합은 $0$, 곱은 $-2$다. 이처럼 계수로부터 얻는 덧셈·곱셈 관계를 모두 유지하는 변환들을 모은 것이 **Galois group**이다. 이런 변환들을 반복해서 적용했을 때 어떤 구조가 되는지가 radical로 풀 수 있는지를 결정한다.

## 1. 근의 공식이 존재한다는 것은 무슨 뜻일까?

먼저 사칙연산을 어디까지 허용할지 정해야 한다. **Field**는 $0$과 $1$을 포함하고, 익숙한 사칙연산 규칙이 성립하며, 계산 결과도 그 안에 남는 수의 집합이다. 단, $0$으로 나누는 연산은 제외한다. 유리수 전체는 field다. 예를 들어 $1/2+1/3=5/6$을 계산해도 다시 유리수가 나온다. 계산을 시작할 때 사용할 field를 **base field** $F$라고 한다. 계수가 유리수인 방정식에서는 보통 $F=\mathbb Q$, 즉 유리수 전체의 field에서 출발한다.

**Polynomial**은 다항식을 뜻한다. 변수의 음이 아닌 정수 거듭제곱에 계수를 곱한 항들을 유한 개 더한 식이다. 예를 들어 $3x^2-2x+1$의 계수는 $3,-2,1$이고, 계수가 $0$이 아닌 가장 높은 거듭제곱의 지수가 $2$이므로 **degree**, 즉 차수는 $2$다. **$F$ 위의 radical expression**은 다음 연산을 유한 번 적용해서 만든 표현이다.

- $F$에 들어 있는 수를 사용한다.
- 앞서 만든 수를 서로 더하거나 빼거나 곱하거나 나눈다. 나눌 때 분모는 $0$이 아니어야 한다.
- 앞서 만든 수 $a$의 $m$-th root, 즉 $b^m=a$를 만족하는 수 $b$를 하나 고른다. $m$은 양의 정수다. 예를 들어 $a=2$의 square root로 $\sqrt2$를 고를 수 있다.

중간값이 복소수여도 되고 radical이 여러 겹이어도 된다. 단, 연산은 유한 번만 사용해야 한다. 답과 같은 수를 상수로 마음대로 추가할 수 있다면 문제가 달라진다. 그래서 어느 base field에서 출발하는지를 분명히 해야 한다.

예를 들어

$$
\begin{gathered}
\sqrt{2+\sqrt3}\\
\text{and}\\
\sqrt[3]{\frac{1+\sqrt5}{2}}
\end{gathered}
\tag{2}
$$

는 모두 $\mathbb Q$ 위의 radical expression이다. $1.4405003973$ 같은 유한소수도 유리수다. 하지만 어떤 무리수인 근의 근삿값으로 썼다면, 그 근과 정확히 같은 수는 아니다. 유리수 근삿값을 계속 구해 어떤 값에 수렴시킬 수 있다고 해서, 그 값을 유한한 radical expression으로 나타낸 것은 아니다.

모든 근을 이 방식으로 나타낼 수 있는 polynomial은 **$F$ 위에서 solvable by radicals**라고 한다. **Quintic**은 5차 polynomial이다. **Abel–Ruffini theorem**은 모든 5차방정식에 통하는 radical 형태의 근의 공식이 없다는 정리다. 특정 5차방정식의 공식이나 수치 알고리즘, 수렴하는 무한급수, 다른 종류의 함수를 연산에 추가해 만든 표현까지 불가능하다는 뜻은 아니다. 실제로 $x^5-2=0$은 radical로 풀린다.

Radical을 구하면 기존에 없던 수가 추가될 수 있다. 유리수만으로 시작할 때 $\sqrt3$는 새로운 수다. 일단 $\sqrt3$를 추가하면 사칙연산만으로 $2+\sqrt3$이나 $1/(2+\sqrt3)=2-\sqrt3$도 만들 수 있다. 여기에 $\sqrt{2+\sqrt3}$를 추가하면 표현할 수 있는 수가 더 늘어날 수 있다. 이렇게 새 수 하나뿐 아니라, 그 수를 이용한 사칙연산으로 얻는 수까지 함께 다루는 것이 field extension이다.

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-01-formula.svg" alt="제곱함수의 그래프와 직선 y=2가 만나는 두 점. 각 점의 가로좌표가 square root의 두 선택에 해당한다." width="960" height="470">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-01-formula-mobile.svg" alt="제곱함수의 그래프와 직선 y=2가 만나는 두 점. 각 점의 가로좌표가 square root의 두 선택에 해당한다. 모바일용 배치." width="480" height="510">
</div>
*Figure 1. $-\sqrt2$와 $\sqrt2$는 제곱하면 둘 다 $2$다. 포물선 $y=x^2$와 직선 $y=2$의 두 교점이 square root를 구할 때의 두 선택을 보여 준다.*

## 2. Vector space로 보는 field extension
{: id="2-field-extension을-vector-space로-이해하기" }

유리수 전체 $\mathbb Q$, 실수 전체 $\mathbb R$, 복소수 전체 $\mathbb C$는 모두 field다. 반면 정수 전체는 field가 아니다. $1$을 $2$로 나누면 정수가 아니기 때문이다.

어떤 field 안에 포함되어 있으면서 같은 사칙연산을 쓰는 field를 **subfield**라고 한다. 예를 들어 $\mathbb Q$는 $\mathbb R$의 subfield다. **Field extension** $K/F$는 바로 이런 관계, 즉 $F\subseteq K$이고 $F$가 $K$의 subfield라는 뜻이다. Slash는 나눗셈 기호가 아니다. $K$의 원소에는 $F$의 수를 scalar로 곱할 수 있다. 따라서 $K$는 $F$ 위의 vector space이기도 하다.

여기서 중요한 것은 **scalar field**, 즉 어떤 수까지 scalar로 쓸 수 있느냐는 점이다. $\mathbb Q$ 위의 vector space에서는 vector 자체가 무리수여도 linear combination의 계수는 반드시 유리수다. 그래서 수 $1$과 $\sqrt2$를 두 개의 basis vector로 쓸 수 있다. $(-\sqrt2)\cdot1+1\cdot\sqrt2=0$이라는 식을 써도 이 둘이 dependent하다는 근거는 되지 않는다. 첫 번째 계수 $-\sqrt2$가 유리수가 아니어서 여기서는 scalar로 쓸 수 없기 때문이다.

$F(\alpha)$는 $F$와 수 $\alpha$를 모두 포함하는 가장 작은 field를 뜻한다. $F$에 $\alpha$ 하나만 더한 집합이 아니라, 이들을 사칙연산으로 조합해서 얻는 수까지 모두 포함한다. 이렇게 수를 추가하는 것을 **adjoining**이라고 한다. $\alpha=\sqrt2$인 경우에는 다음과 같다.

$$
\mathbb Q(\alpha)
=\{a+b\alpha:a,b\in\mathbb Q\}.
\tag{3}
$$

왜 좌표가 두 개면 충분할까? $\alpha^2=2$이므로 $\alpha^3=2\alpha$, $\alpha^4=4$처럼 높은 거듭제곱도 같은 꼴로 정리할 수 있다. 곱셈을 해도 좌표가 더 필요하지 않은 이유다. $0$이 아닌 수의 **multiplicative inverse**는 그 수에 곱하면 $1$이 되는 수, 즉 역수다. $(3+2\alpha)(3-2\alpha)=9-8=1$이므로, $3+2\alpha$의 inverse인 $3-2\alpha$도 이 집합에 들어 있다. 따라서 이 수로 나누기 위해 새로운 형태의 수를 추가할 필요는 없다.

<details markdown="1">
<summary>덧셈·곱셈·나눗셈을 해도 같은 field 안에 남는지 확인하기</summary>

어떤 연산을 해도 결과가 같은 집합에 남으면 그 연산에 대해 **closed**하다고 한다. $u=a+b\alpha$, $v=c+d\alpha$라고 쓰면 덧셈과 곱셈은 다음과 같다.

$$
\begin{aligned}
u+v&=(a+c)+(b+d)\alpha,\\
uv&=(ac+2bd)\\
  &\quad+(ad+bc)\alpha.
\end{aligned}
\tag{4}
$$

나눗셈을 해도 같은 형태가 유지된다.

$$
\begin{gathered}
\frac1{a+b\alpha}
=\frac{a-b\alpha}{a^2-2b^2},\\
a+b\alpha\ne0.
\end{gathered}
\tag{5}
$$

유리수 $a,b$에 대해 분모가 $0$이 되는 것은 $a=b=0$일 때뿐이다. 그렇지 않으면 $a/b$가 $2$의 제곱근인 유리수가 되어 모순이다. 따라서 (3)의 집합 안에서 사칙연산을 모두 할 수 있다.

</details>

또한 $1,\alpha$는 $\mathbb Q$ 위에서 linearly independent하다. $a+b\alpha=0$인데 $b\ne0$이라면 $\alpha=-a/b$가 유리수여야 하기 때문이다. 따라서 extension degree는 다음과 같다.

$$
[\mathbb Q(\sqrt2):\mathbb Q]=2.
\tag{6}
$$

대괄호로 나타낸 **extension degree**는 주어진 base field 위에서의 vector-space dimension이다. 이 dimension이 유한하면 **finite extension**이라고 한다. 원소의 개수가 유한하다는 뜻은 아니다. $\mathbb Q(\sqrt2)$에는 원소가 무한히 많지만, $\mathbb Q$ 위의 basis vector는 두 개면 된다. Degree가 $2$인 extension을 **quadratic extension**이라고 하므로, $\mathbb Q(\sqrt2)$는 $\mathbb Q$ 위의 quadratic field다.

반면 $\mathbb Q(\sqrt2)$의 모든 원소를 scalar로 쓸 수 있다면 basis는 $1$ 하나면 된다. 각 원소가 곧 scalar이므로 이를 $1$에 곱해 표현할 수 있기 때문이다. 같은 집합도 어느 field의 수를 scalar로 쓰느냐에 따라 dimension이 $2$가 되기도 하고 $1$이 되기도 한다.

$\alpha$를 곱하는 연산은 $\mathbb Q$-linear map이다. 행렬의 첫 번째 열은 $\alpha\cdot1=\alpha$의 좌표 $(0,1)$이고, 두 번째 열은 $\alpha\cdot\alpha=2$의 좌표 $(2,0)$이다. 따라서 basis를 $(1,\alpha)$ 순서로 쓰면 다음 행렬을 얻는다.

$$
\begin{gathered}
M_\alpha=\begin{pmatrix}0&2\\1&0\end{pmatrix},\\
M_\alpha\binom ab=\binom{2b}a,\\
M_\alpha^2=2I.
\end{gathered}
\tag{7}
$$

이 행렬이 작용하는 vector의 좌표는 유리수다. Eigenvalue인 $\pm\sqrt2$를 쓰려면 scalar field를 확장해야 한다. 행렬의 성분이 유리수라고 해서 eigenvalue까지 유리수여야 하는 것은 아니다.

$\alpha$를 $-\alpha$로 바꾸는 변환은 앞의 multiplication matrix가 나타내는 연산과 다르다. 이 변환은 모든 유리수를 그대로 두고 $3+2\alpha$를 $3-2\alpha$로 보낸다. 이것이 **algebraic conjugation**이며, $\sigma(a+b\alpha)=a-b\alpha$라고 쓰자. 일반적인 **complex conjugation**은 실수 $a,b$에 대해 $a+bi$를 $a-bi$로 보낸다. 여기서 $i$는 $i^2=-1$인 수다. 따라서 실수인 $3+2\alpha$와 $3-2\alpha$는 둘 다 그대로 둔다. 여기서 쓰는 algebraic conjugation은 이 두 실수를 서로 바꾼다는 점에서 다르다. $(-\alpha)^2=2$이므로 부호를 바꿔도 원래의 곱셈 관계는 유지된다. 좌표로 쓰면 $(a,b)\mapsto(a,-b)$이며, $\alpha$를 곱하는 연산이 아니다.

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-02-quadratic-field.svg" alt="유리수 좌표에서 unit square와 vector가 sqrt2 곱셈 및 algebraic conjugation에 따라 바뀌는 모습." width="960" height="515">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-02-quadratic-field-mobile.svg" alt="유리수 좌표에서 unit square와 vector가 sqrt2 곱셈 및 algebraic conjugation에 따라 바뀌는 모습. 모바일용 배치." width="480" height="920">
</div>
*Figure 2. $\mathbb Q$ 위의 basis는 $(1,\alpha)$이며 $\alpha=\sqrt2$다. 점선 unit square와 $u=(1,1)$에 각 변환을 적용한 결과를 음영 영역과 색으로 표시한 vector로 나타냈다. 곱셈을 적용하면 $M_{\alpha}u=(2,1)$, algebraic conjugation을 적용하면 $\sigma u=(1,-1)$이다. 축은 Re/Im이 아니라 각 basis vector에 곱하는 계수다.*

수와 그 conjugate를 곱하면 제곱근 항이 사라진다. 이 quadratic field에서는 그 곱을 **field norm**이라고 하며, $N(a+b\sqrt2)=a^2-2b^2$다. 예를 들어 $N(1+\sqrt2)=(1+\sqrt2)(1-\sqrt2)=-1$이다. 이름에 norm이 들어 있지만 vector의 길이가 아니므로 음수가 될 수도 있다. (5)는 conjugate를 이 유리수로 나누어 inverse를 구하는 식이다.

<details markdown="1">
<summary>Python: 유리수 좌표로 사칙연산과 algebraic conjugation 구현하기</summary>

계수를 모두 유리수 그대로 저장하는 코드다. 좌표쌍 $(a,b)$는 $a+b\sqrt2$를 나타낸다. `qmul`의 $2bd$는 $bd(\sqrt2)^2$를 정리한 항이고, `qconj`는 두 번째 좌표의 부호를 바꾼다. `qinv`는 (5)를 구현한다. 마지막 assertion은 먼저 곱한 뒤 conjugation한 값과, 각각 conjugation한 뒤 곱한 값이 같은지 확인한다.

```python
from fractions import Fraction

def qadd(u, v):
    return tuple(Fraction(a) + Fraction(b) for a, b in zip(u, v))

def qmul(u, v):
    a, b = map(Fraction, u)
    c, d = map(Fraction, v)
    # Reduce sqrt(2)**2 to the rational number 2.
    return a*c + 2*b*d, a*d + b*c

def qconj(u):
    a, b = map(Fraction, u)
    return a, -b

def qinv(u):
    a, b = map(Fraction, u)
    norm = a*a - 2*b*b
    if norm == 0:
        raise ZeroDivisionError("Zero has no multiplicative inverse")
    return a/norm, -b/norm

assert qmul((1, 1), (1, -1)) == (-1, 0)
assert qmul((3, 2), qinv((3, 2))) == (1, 0)
assert qconj(qmul((1, 2), (3, 4))) == qmul(qconj((1, 2)), qconj((3, 4)))
print("Exact arithmetic in Q(sqrt(2)): passed")
```

</details>

{% include interactive/article-lab.html id="galois-quadratic" %}

두 수의 계수를 각각 바꾸어 보고, 먼저 곱한 뒤 conjugation한 결과와 각각 conjugation한 뒤 곱한 결과를 비교해 보자. Conjugation은 $\sqrt2$의 계수 부호만 바꾸지만, 곱셈도 보존한다. 일반적인 invertible matrix에는 없는 성질이다.

## 3. Minimal polynomial의 차수와 extension degree

$\alpha=\sqrt2$의 거듭제곱을 차례로 쓰면 $1,\alpha,2,2\alpha,4,\ldots$다. 앞의 두 수는 $\mathbb Q$ 위에서 independent하지만, 세 번째는 $\alpha^2-2\cdot1=0$이라는 관계를 만족한다. 이렇게 처음 dependence가 생기는 지점에서 좌표가 두 개면 충분하다는 것을 알 수 있다. $\alpha^4-4=0$도 맞는 식이지만, 이런 관계가 있다고 해서 dimension이 $4$가 되는 것은 아니다.

원소 $\alpha$가 $F$의 수를 계수로 쓰는 $0$이 아닌 polynomial의 근이라면, **$F$ 위에서 algebraic**이라고 한다. 예를 들어 $\sqrt2$는 $x^2-2=0$을 만족하므로 $\mathbb Q$ 위에서 algebraic하다. 계수가 $F$에 속하는 polynomial 전체를 $F[x]$라고 쓴다. Polynomial equation은 $1,\alpha,\alpha^2,\ldots$ 사이의 linear dependence를 나타낸다.

가장 높은 차수의 항에 붙은 계수가 **leading coefficient**다. 이 계수가 $1$인 polynomial을 **monic**이라고 한다. 예를 들어 $2x^2-4$를 leading coefficient로 나누면 monic polynomial $x^2-2$가 된다. $\alpha$를 대입하면 $0$이 되는 polynomial 중에서, monic이고 차수가 가장 작은 polynomial은 하나뿐이다. 이것이 $\alpha$의 **minimal polynomial** $m_\alpha(x)$다. $\alpha=\sqrt2$라면 $x^2-2$가 minimal polynomial이다. $x^4-4$도 $\alpha$를 근으로 갖지만 차수가 더 높으므로 minimal polynomial이 아니다.

양의 차수인 polynomial을 $F[x]$ 안에서 양의 차수인 두 polynomial의 곱으로 쓸 수 없다면 **$F$ 위에서 irreducible**하다고 한다. 그런 곱으로 쓸 수 있으면 **reducible**하다고 한다. 어느 field에서 계산하는지도 중요하다. $x^2-2$는 $\mathbb Q$ 위에서는 irreducible하지만, $\mathbb Q(\sqrt2)$ 위에서는 $(x-\sqrt2)(x+\sqrt2)$로 나뉜다. Minimal polynomial은 항상 irreducible하다. 만약 나눌 수 있다면 차수가 더 낮은 인수 하나도 $\alpha$를 근으로 가져야 하기 때문이다.

$\deg m_\alpha=d$일 때 다음 관계가 성립한다.

$$
\begin{gathered}
\text{Basis of }F(\alpha)/F:\\
1,\alpha,\ldots,\alpha^{d-1},\\
[F(\alpha):F]=d.
\end{gathered}
\tag{8}
$$

처음 dependence가 생기기 전까지의 independent한 거듭제곱 개수가 곧 degree다. **Polynomial division**은 나누는 polynomial의 적절한 배수를 빼서, 나머지의 차수가 그 polynomial보다 작아질 때까지 정리하는 계산이다. 예를 들어 $x^3=x(x^2-2)+2x$다. 여기에 $x=\alpha=\sqrt2$를 대입하면 $x^2-2$가 들어 있는 항은 사라지고 $\alpha^3=2\alpha$만 남는다. 같은 방식으로 $m_\alpha$로 나누면 높은 거듭제곱도 (8)의 $d$개 basis vector로 정리된다.

Inverse도 같은 좌표로 표현할 수 있다. 예를 들어 $\alpha=\sqrt2$라면 $(\alpha+1)(\alpha-1)=1$이므로 $(\alpha+1)^{-1}=\alpha-1$이다.

<details markdown="1">
<summary>같은 basis로 높은 거듭제곱과 분수까지 표현할 수 있는 이유</summary>

$m_\alpha$로 polynomial division을 하면 높은 거듭제곱도 이 $d$개 항의 linear combination으로 바꿀 수 있다. 반대로 이 basis 후보들의 nonzero linear combination이 $0$이라면, $\alpha$를 근으로 갖는 차수 $d$ 미만의 polynomial을 만들 수 있다. 이는 minimality에 모순이다. Irreducibility도 같은 논리로 확인할 수 있다. $m_\alpha=uv$이고 두 인수의 차수가 더 낮다면, field 안에서 $u(\alpha)v(\alpha)=0$이므로 인수 중 하나는 $\alpha$에서 $0$이어야 한다.

**Rational expression**은 $1/(\alpha+1)$처럼 polynomial 표현끼리 나눈 식을 뜻한다. 식의 값이 반드시 유리수라는 뜻은 아니다. 이런 분수를 같은 basis로 표현하려면, $q(\alpha)\ne0$일 때 $1/q(\alpha)$를 같은 좌표로 나타낼 수 있으면 된다.

두 polynomial에 양의 차수인 공통인수가 없으면 **relatively prime**이라고 한다. $m_\alpha$가 irreducible하므로 $q$와 $m_\alpha$는 relatively prime이다. 양의 차수인 공통인수가 있다면 $m_\alpha$ 자체가 $q$를 나누어야 하고, 그러면 $q(\alpha)=0$이 되어 모순이기 때문이다.

**Polynomial Euclidean algorithm**은 나머지를 구하는 계산을 반복한다. 첫 polynomial을 둘째로 나누고, 둘째를 방금 얻은 나머지로 나눈다. 이 과정을 나머지가 $0$이 될 때까지 반복한다. 나머지의 차수가 계속 줄어드니 계산은 반드시 끝난다. 마지막으로 얻은 $0$이 아닌 나머지가 공통인수 중 차수가 가장 높은 것이며, relatively prime인 경우에는 $0$이 아닌 상수다. 나머지를 구한 식을 거꾸로 대입하고 그 상수로 나누면 $u q+v m_\alpha=1$인 $u,v\in F[x]$를 얻는다. 여기에 $\alpha$를 대입하면 $u(\alpha)q(\alpha)=1$이므로 $q(\alpha)^{-1}=u(\alpha)$다. Inverse를 나타낼 때도 좌표를 더 쓸 필요가 없다.

예를 들어 $x^2-2$를 $x+1$로 나누면 $x^2-2=(x-1)(x+1)-1$이다. 이 나머지 식 하나를 정리하면 $(x-1)(x+1)-(x^2-2)=1$을 얻는다. 여기에 $x=\sqrt2$를 대입하면 마지막 항이 사라진다. 그러면 $\sqrt2+1$의 inverse가 $\sqrt2-1$이라는 사실을 바로 얻는다.

</details>

$\sqrt[3]2$의 minimal polynomial은 $x^3-2$다. **Cubic**, 즉 3차 polynomial이 유리수 위에서 reducible하다면 일차인수가 있으므로 유리수인 근이 있어야 한다. **Rational-root test**는 계수를 이용해 유리수인 근의 후보를 추리는 방법이다. 여기서 후보는 $1,-1,2,-2$뿐인데, 어느 것도 근이 아니다. 따라서 $x^3-2$는 irreducible하고, 실수만 포함하는 field $\mathbb Q(\sqrt[3]2)$의 basis는 $(1,\sqrt[3]2,\sqrt[3]4)$이며 degree는 $3$이다.

<details markdown="1">
<summary>Cubic의 irreducibility를 rational-root test로 확인하기</summary>

유리수인 근이 있다면 기약분수 $p/q$로 쓰자. 여기서 $p,q$는 정수이고 $q>0$이며, 두 수에는 공통인 소인수가 없다. 대입하면 $p^3=2q^3$이다. $q$를 나누는 소수가 있다면 $p$도 나누어야 하므로 모순이다. 따라서 $q=1$이며, 이제 $p^3=2$이므로 $p$는 $2$의 약수다. 이것이 이 monic polynomial에 대한 rational-root test다. 유리수인 근이 있다면 상수항을 나누는 정수여야 한다.

후보 $1,-1,2,-2$를 대입하면 값이 각각 $-1,-3,6,-10$이므로 모두 근이 아니다. Cubic을 양의 차수인 두 인수의 곱으로 쓰려면 일차인수가 반드시 하나는 있어야 한다. 따라서 유리수인 근이 없다는 사실로부터 $\mathbb Q$ 위의 irreducibility를 확인할 수 있다.

</details>

### Field를 연달아 확장하면 dimension은 곱해진다

$E=\mathbb Q(\sqrt2)$를 만든 다음 $K=E(i)=\mathbb Q(\sqrt2,i)$로 확장해 보자. 첫 번째 확장의 basis는 $(1,\sqrt2)$다. $E$의 모든 원소는 실수이므로 $i$는 이 field에 들어 있지 않다. $i$를 추가하면 $E$ 위에서 basis가 $(1,i)$인 extension을 얻는다. $K$의 원소는 $x+yi$로 쓸 수 있고, 이때 $x,y\in E$다. 여기에 $x=a+b\sqrt2$, $y=c+d\sqrt2$를 대입하면 $a+b\sqrt2+ci+d\,i\sqrt2$가 된다. 좌표 두 개를 각각 다시 두 좌표로 풀어 쓴 셈이다. 따라서 $\mathbb Q$ 위의 basis는 $(1,\sqrt2,i,i\sqrt2)$다.

$F\subseteq E\subseteq K$처럼 field가 차례로 포함된 구조를 **tower**라고 한다. 각 단계의 extension degree가 유한하면 같은 방식으로 좌표를 전개할 수 있다. 이렇게 얻는 관계가 **tower law**다.

$$
[K:F]=[K:E]\,[E:F].
\tag{9}
$$

위 예에서는 $[K:\mathbb Q]=2\cdot2=4$다. 각 확장의 basis vector 개수를 곱한 결과다. 식에 radical 기호가 몇 개 들어 있는지를 세는 것이 아니다.

<details markdown="1">
<summary>각 단계의 basis를 곱하면 전체 extension의 basis가 되는 이유</summary>

$E/F$의 basis를 $e_1,\ldots,e_r$, $K/E$의 basis를 $u_1,\ldots,u_s$라고 하자. $K$의 원소를 먼저 $\sum_j c_j u_j$로 쓰면 $c_j\in E$다. 각 계수를 다시 $c_j=\sum_i a_{ij}e_i$로 전개하면 $a_{ij}\in F$이므로, $rs$개의 곱 $e_i u_j$로 $K$ 전체를 span할 수 있다.

$\sum_{i,j}a_{ij}e_i u_j=0$이라고 하자. $u_j$들이 $E$ 위에서 independent하므로 각 $j$에 대해 $\sum_i a_{ij}e_i=0$이다. 다시 $e_i$들이 $F$ 위에서 independent하므로 모든 $a_{ij}=0$이다. 따라서 곱으로 만든 vector들도 independent하다. 앞의 예에서 $1,\sqrt2,i,i\sqrt2$ 중 어느 하나도 나머지의 유리계수 linear combination으로 쓸 수 없는 이유가 여기에 있다.

</details>

## 4. 근 하나를 추가하는 것과 splitting field를 만드는 것은 다르다

복소수까지 허용하면 $n$차 polynomial에는 근이 $n$개 있다. 같은 근에 해당하는 인수가 여러 번 나오면 그 횟수만큼 센다. 근이 존재하고 개수가 정해진다는 사실과, 그 근을 radical로 표현할 수 있는지는 별개의 문제다.

$x^2-2$에서는 $\sqrt2$만 추가해도 $-\sqrt2$를 함께 표현할 수 있다. 하지만 $x^3-2$에서는 실근 $\alpha=\sqrt[3]2$를 추가해도 나머지 근을 얻을 수 없다. 세 근을 써 보면 다음과 같다.

$$
\begin{gathered}
\alpha,\quad\alpha\omega,\quad\alpha\omega^2,\\
\omega=\frac{-1+i\sqrt3}{2},\\
\omega^3=1,\qquad\omega\ne1.
\end{gathered}
\tag{10}
$$

**$m$-th root of unity**는 $z^m=1$을 만족하는 수다. 이때 $1,\zeta_m,\ldots,\zeta_m^{m-1}$가 모두 달라서 이 방정식의 근을 빠짐없이 얻을 수 있다면, 그 수를 **primitive root of unity** $\zeta_m$라고 한다. 위의 $\omega$가 primitive cube root of unity다.

예를 들어 fourth roots of unity는 $1,i,-1,-i$다. $i$를 거듭제곱하면 네 수가 모두 나오므로 $i$는 primitive하다. 반면 $-1$도 $(-1)^4=1$을 만족하지만, 계속 거듭제곱해도 $1$과 $-1$만 번갈아 나오므로 primitive하지 않다. 근 하나에 이 root of unity들을 곱하면 radical을 구할 때의 다른 선택도 얻을 수 있다. $\beta^m=a$이고 $a\ne0$이면 $m$개의 근은 $\beta\zeta_m^k$, $k=0,\ldots,m-1$이다.

<details markdown="1">
<summary>Roots of unity가 원 위에 일정한 간격으로 놓이는 이유</summary>

**Complex plane**에서는 $a+bi$를 점 $(a,b)$로 나타낸다. Fourth roots of unity인 $1,i,-1,-i$는 각각 $(1,0),(0,1),(-1,0),(0,-1)$에 놓인다. 각 수에 $i$를 곱하면 점이 사분의 일 바퀴씩 회전하므로, 네 번 곱하면 다시 $1$로 돌아온다.

일반적으로 $e^{i\theta}$는 unit circle 위의 점 $\cos\theta+i\sin\theta$를 나타낸다. Sine과 cosine의 덧셈공식에 따라 이런 복소수끼리 곱하면 각도가 더해진다. 따라서 $\zeta_m=e^{2\pi i/m}$를 고르면 한 바퀴의 $1/m$만큼 회전하는 수를 얻는다. $1$에서 시작해 차례로 거듭제곱하면 같은 간격의 서로 다른 $m$개 점을 거쳐 다시 $1$로 돌아온다. $m=3$이면 $\omega=-1/2+i\sqrt3/2$이며, (10)의 수와 같다.

</details>

$\mathbb Q(\alpha)$의 원소는 모두 실수이므로 뒤의 두 근은 이 field에 들어 있지 않다. **Splitting field**는 base field와 모든 근을 포함하는 가장 작은 field다. 이 예의 splitting field는 다음과 같다.

$$
L=\mathbb Q(\alpha,\omega).
\tag{11}
$$

첫 번째 확장의 degree는 $3$이고 두 번째는 $2$다. $\omega$는 $x^2+x+1=0$을 만족하지만 실수 field $\mathbb Q(\alpha)$에는 들어 있지 않기 때문이다. 따라서 $[L:\mathbb Q]=6$이다.

두 예의 차이는 **normality**에 있다. Base field 위의 irreducible polynomial이 extension 안에 근 하나를 가지면, 나머지 근도 모두 그 안에 있어야 한다는 조건이다. 이 조건이 성립하는 extension을 **normal**하다고 한다. 특정 방정식 하나만이 아니라, 이런 polynomial 모두에 대한 조건이다. $\mathbb Q(\sqrt2)$의 무리수 원소 $a+b\sqrt2$의 minimal polynomial은 $(x-a)^2-2b^2$이며, 다른 근 $a-b\sqrt2$도 같은 field 안에 있다. 유리수 원소의 minimal polynomial은 1차이므로 다른 근이 빠질 일이 없다. 따라서 이 extension은 normal하다. 반면 cubic 예에서는 $x^3-2$ 하나만 확인해도 조건이 깨진다. $\mathbb Q(\sqrt[3]2)$에 근 하나는 있지만 나머지 두 근이 빠져 있기 때문이다.

같은 근이 인수로 여러 번 등장하면 **repeated root**, 즉 중근이라고 한다. 예를 들어 $(x-1)^2$에서 $1$은 두 번 나타나는 근이다. 각 원소의 minimal polynomial에 repeated root가 없는 extension을 **separable**하다고 한다. Normal이면서 separable한 finite extension이 **Galois extension**이다. $\mathbb Q$ 위에서는 separability가 자동으로 성립한다. 예를 들어 minimal polynomial인 $x^2-2$와 $x^3-2$는 각각 서로 다른 근만 갖는다. 이는 $\mathbb Q$가 **characteristic zero**인 field이기 때문이다. $1$을 몇 번 더해도 $0$이 되지 않는다는 뜻이다. 이 성질 때문에 irreducible polynomial에 repeated root가 생기지 않는 이유는 아래 derivative 계산으로 확인할 수 있다. 따라서 앞의 splitting field들은 Galois extension이다. 모든 근을 모아야 하는 이유가 normality 조건에 있다.

<details markdown="1">
<summary>Normal·separable의 정확한 정의와 Q 위에서 separability가 성립하는 이유</summary>

모든 원소가 base field 위에서 algebraic인 extension을 **algebraic extension**이라고 한다. Finite extension은 모두 이 조건을 만족한다. Dimension이 유한한 vector space에서는 $1,\alpha,\alpha^2,\ldots$가 끝까지 independent할 수 없으므로, 반드시 polynomial relation이 생기기 때문이다. 두 조건의 정확한 정의는 다음과 같다.

- Finite algebraic extension $L/F$가 **normal**하다는 것은, $F$ 위의 irreducible polynomial이 $L$에 근 하나라도 가지면 그 polynomial의 모든 근이 $L$에 들어 있다는 뜻이다. 즉 $L$ 안에서 일차인수의 곱으로 완전히 분해된다.
- **Separable**하다는 것은 각 원소의 minimal polynomial이 repeated root를 갖지 않는다는 뜻이다.

이 두 조건을 모두 만족하는 finite extension이 **Galois extension**이다. $1$을 한 번 이상 더했을 때 그 합이 $0$이 되는 경우가 없으면 **characteristic zero**인 field라고 한다. $\mathbb Q$에서 이런 합은 우리가 아는 양의 정수이므로 characteristic이 zero다. 이런 field에서는 모든 irreducible polynomial이 separable하다.

이 사실은 **polynomial derivative**로 확인할 수 있다. 각 항에 $x^k\mapsto kx^{k-1}$ 규칙을 적용하고 상수항은 $0$으로 바꾸어 구한다. Characteristic zero에서 양의 차수인 irreducible polynomial의 derivative는 $0$이 아니고 차수가 더 낮다. 따라서 두 polynomial은 양의 차수인 공통인수를 가질 수 없다. 그런데 원래 polynomial에 $(x-r)^2$라는 인수가 있으면 derivative에도 $x-r$이 남는다. 두 polynomial이 공통 근을 가지므로 그 근의 minimal polynomial이 둘 다 나누어야 하는데, 이는 공통인수가 없다는 사실과 모순이다. 따라서 repeated root는 없다.

예를 들어 $x^2-2$와 그 derivative $2x$는 공통 근이 없다. 공통 근이라면 $0$이면서 동시에 $2$의 제곱근이어야 하기 때문이다. $(x^2-2)^2$에는 repeated root가 있지만, $\sqrt2$의 minimal polynomial은 여전히 $x^2-2$다. 이 extension의 separability는 같은 방정식을 제곱해 썼는지가 아니라 minimal polynomial의 성질로 결정된다.

</details>

유리계수 polynomial의 splitting field는 finite Galois extension이다. Polynomial에 중복된 인수가 있어도 서로 다른 irreducible factor만 남기면 된다. 이렇게 해도 splitting field는 달라지지 않는다.

## 5. Automorphism: 근을 바꾸어도 사칙연산은 그대로
{: id="5-automorphism은-근의-목록뿐-아니라-사칙연산을-보존한다" }

**Bijection**은 모든 출력을 정확히 한 번씩 얻는 함수다. 서로 다른 두 입력이 같은 출력으로 합쳐지지 않고, 빠지는 출력도 없으므로 변환을 되돌릴 수 있다. $\mathbb Q(\sqrt2)$의 conjugation이 한 예다. 같은 부호 변환을 한 번 더 적용하면 원래 수로 돌아온다.

$L$의 **$F$-automorphism**은 $F$의 모든 원소를 그대로 두는 bijection $\sigma:L\to L$다. 즉 $c\in F$이면 $\sigma(c)=c$이며, 다음과 같이 사칙연산을 보존해야 한다.

$$
\begin{aligned}
\sigma(u+v)&=\sigma(u)+\sigma(v),\\
\sigma(uv)&=\sigma(u)\sigma(v),\\
\sigma(1)&=1.
\end{aligned}
\tag{12}
$$

위 식은 덧셈·곱셈과 $1$을 보존한다는 뜻이다. 뺄셈과 나눗셈도 따라온다. 더해서 $0$이 되는 관계와 곱해서 $1$이 되는 관계가 유지되기 때문이다. 따라서 automorphism으로 근의 값을 바꾸더라도, 그 수들이 만족하는 사칙연산 관계는 달라지지 않는다.

$c\in F$에 대해 $\sigma(cu)=c\sigma(u)$이므로 자동으로 $F$-linear하다. 하지만 linearity만으로는 충분하지 않다. $\mathbb Q(\sqrt2)$에서 $a+b\sqrt2\mapsto a+2b\sqrt2$는 invertible linear map이고 $1$도 고정한다. 그러나 $\sqrt2$를 $2\sqrt2$로 보내면 그 제곱은 $2$가 아니라 $8$이다. 곱셈을 보존하지 못하므로 field automorphism은 아니다.

Automorphism이 근을 같은 polynomial의 근으로 보내는 이유도 알 수 있다. $f(x)=\sum c_jx^j$의 계수가 $F$에 속하고 $f(\alpha)=0$이라면 다음 등식이 성립한다.

$$
\begin{aligned}
0&=\sigma(f(\alpha))\\
 &=\sum c_j\sigma(\alpha)^j\\
 &=f(\sigma(\alpha)).
\end{aligned}
\tag{13}
$$

계수는 그대로 두면서 근의 값은 바꿀 수 있다. 같은 논리로 근들이 만족하는 공통 방정식 하나뿐 아니라 **근들 사이에 성립하는 모든 $F$-계수 polynomial relation**도 보존된다.

$\mathbb Q(\sqrt2)/\mathbb Q$에서 $\sqrt2$를 보낼 수 있는 값은 $\sqrt2$와 $-\sqrt2$뿐이다. 어느 쪽을 골라도 field 전체의 automorphism으로 확장할 수 있다. 모든 원소를 그대로 두는 변환이 **identity**이며 $\operatorname{id}$ 또는 $e$라고 쓴다. 나머지 하나가 다음 conjugation이다. $\sigma^2$는 같은 변환을 두 번 적용한다는 뜻이다.

$$
\begin{gathered}
\sigma(a+b\sqrt2)=a-b\sqrt2,\\
\sigma^2=\operatorname{id}.
\end{gathered}
\tag{14}
$$

Basis $(1,\sqrt2)$에서 이 변환의 행렬은 $\operatorname{diag}(1,-1)$이다. (7)의 multiplication matrix와 달리 $1$을 고정하고 곱셈을 보존한다.

예를 들어 $\sigma(3+2\sqrt2)=3-2\sqrt2$이고, $\sigma$를 한 번 더 적용하면 다시 $3+2\sqrt2$가 된다. 부호를 두 번 바꾸면 원래 값으로 돌아오므로, 두 번 합성한 변환은 identity다.

### Automorphism들을 모으면 왜 group이 될까?

**Group**은 두 원소를 결합하는 연산이 정해진 집합이다. 여기서 원소는 함수이고 결합 연산은 함수의 합성이다. 합성 결과가 다시 같은 집합에 속해야 하고(**closure**), 세 함수를 합성할 때 괄호를 어디에 두든 결과가 같아야 한다(**associativity**). Identity가 있어야 하며, 각 원소에는 그 변환을 되돌리는 inverse가 있어야 한다. 두 함수의 적용 순서를 바꿔도 같은 결과가 나와야 한다는 조건은 없다.

Automorphism은 이 조건들을 만족한다. 사칙연산을 보존하는 두 bijection을 합성해도 사칙연산이 보존되고, 각 bijection의 inverse도 사칙연산을 보존하기 때문이다. Quadratic 예에서는 부호 변환을 두 번 적용하면 identity가 되므로, 부호 변환 자체가 자신의 inverse다.

$L/F$가 Galois일 때 이 group을

$$
\operatorname{Gal}(L/F)=\operatorname{Aut}_F(L)
\tag{15}
$$

라고 쓴다. $\operatorname{Aut}_F(L)$는 앞서 정의한 $F$-automorphism 전체를 뜻한다. 합성은 오른쪽부터 적용한다. $\sigma\tau$는 “$\tau$를 적용한 뒤 $\sigma$를 적용한다”는 뜻이다.

Quadratic field에서 identity를 $e$라고 쓰면 group 전체는 $\lbrace e,\sigma\rbrace $다. 이들의 합성은 $e\sigma=\sigma e=\sigma$, $\sigma\sigma=e$다. Inverse도 같은 집합에 있다. $e^{-1}=e$이고 $\sigma^{-1}=\sigma$다. Associativity는 함수 합성 자체의 성질이다. 제곱근에 따로 조건을 붙여야 성립하는 것이 아니다.

$L$이 $F$ 위의 polynomial의 splitting field이고, 그 polynomial에 서로 다른 근이 $n$개 있다고 하자. 이 근 **전체**를 생각해야 한다. **Permutation**은 각 label을 정확히 한 번씩 사용하는 일대일 재배열이다. 첫 label의 출력을 고르는 방법이 $n$가지, 다음은 $n-1$가지이고 마지막은 $1$가지이므로 모두 $n!$가지다. Label이 세 개라면 $3!=6$가지다.

Base field의 수와 이 근들을 사칙연산으로 조합하면 $L$의 모든 원소를 표현할 수 있다. (13)에 따라 automorphism은 근들을 이 집합 안에서 서로 바꾼다. 각 근을 어디로 보낼지 정하면, 사칙연산을 보존해야 하므로 나머지 원소의 image도 모두 정해진다. $n$개 label의 모든 permutation으로 이루어진 group이 **symmetric group** $S_n$이다. 허용되는 근의 permutation들은 그 안에서 **subgroup**을 이룬다. 같은 합성 연산을 사용하면서 그 자체로 group이 되는 부분집합이라는 뜻이다. $x^2-2$의 두 근은 두 permutation이 모두 허용되므로 $S_2$ 전체를 얻는다. 근이 더 많을 때는 근들 사이의 관계 때문에 일부 permutation이 허용되지 않을 수 있다.

**Cycle notation**은 label이 한 바퀴 도는 순서를 적는 방법이다. $(123)$은 $1\mapsto2\mapsto3\mapsto1$로 보내고, 표시하지 않은 label은 그대로 둔다는 뜻이다. Label이 두 개인 cycle $(12)$는 두 label을 맞바꾼다. $(12)(34)$처럼 label이 겹치지 않는 **disjoint cycle**은 서로 다른 label에 작용하므로 어느 쪽을 먼저 적용해도 같다.

한 원소의 정수 거듭제곱들로 이루어진 group을 **cyclic group**이라고 하고, 그 원소를 **generator**라고 한다. 음의 거듭제곱은 inverse를 반복 적용한다는 뜻이다. 원소가 유한 개뿐인 **finite group**에서는 generator를 반복 적용하다 보면 identity로 돌아오며, 그 과정에서 group의 모든 원소를 얻는다. Quadratic group $\lbrace e,\sigma\rbrace $는 $\sigma^2=e$이므로 cyclic하다. Cyclic group은 group 전체를 어떻게 만드는지에 대한 말이지, 각 permutation을 하나의 cycle로 표현한다는 뜻이 아니다.

두 연산의 순서를 바꾸어도 결과가 같으면 **commute**한다고 한다. Cycle이 겹치면 순서에 따라 결과가 달라질 수 있다. $(12)(23)=(123)$이지만 $(23)(12)=(132)$다. 첫 번째 곱에서 $1$은 $(23)$을 적용해도 그대로이고, 그다음 $(12)$를 적용하면 $2$가 된다. 순서를 뒤집으면 $1$이 먼저 $2$가 되고 이어서 $3$이 된다. 따라서 이 두 연산은 commute하지 않는다.

<details markdown="1">
<summary>세 label에 직접 적용해 합성 순서 비교하기</summary>

$(12)(23)$의 두 변환을 순서대로 적용하면 $1\mapsto1\mapsto2$, $2\mapsto3\mapsto3$, $3\mapsto2\mapsto1$이므로 결과는 $(123)$이다. $(23)(12)$에서는 $1\mapsto2\mapsto3$, $2\mapsto1\mapsto1$, $3\mapsto3\mapsto2$이므로 $(132)$다. 각 식에서 첫 번째 화살표는 오른쪽 permutation을 적용한 결과다.

Cycle과 cyclic group의 차이는 $g=(12)(34)$에서도 볼 수 있다. 이 permutation에는 disjoint cycle이 두 개 있지만 $g^2=e$이므로, 두 원소로 이루어진 group $\lbrace e,g\rbrace $는 $g$를 generator로 하는 cyclic group이다.

</details>

## 6. 네 근의 permutation 중 네 개만 허용되는 이유
{: id="6-근은-네-개인데-허용되는-permutation도-네-개뿐인-예" }

다음 polynomial을 보자.

$$
\begin{gathered}
f(x)=(x^2-2)(x^2-3)\\
=x^4-5x^2+6,\\
L=\mathbb Q(\sqrt2,\sqrt3).
\end{gathered}
\tag{16}
$$

이 polynomial은 degree가 $4$인 **quartic**이다. 네 근은 $\sqrt2,-\sqrt2,\sqrt3,-\sqrt3$이다. **Symmetric expression**은 입력의 순서를 바꾸어도 값이 같은 식이다. 모든 근의 합과 곱이 그 예다. 어떤 permutation이든 이런 식의 값은 보존한다. 그렇다고 모든 permutation이 automorphism이 되는 것은 아니다. 각 근에 대한 관계를 포함해 base field 위의 사칙연산 관계를 모두 보존해야 하기 때문이다.

예를 들어 $\sqrt2$와 $\sqrt3$를 바꾸면 $(\sqrt2)^2=2$라는 관계가 $(\sqrt3)^2=2$로 바뀌어 성립하지 않는다. 그렇다고 $2$를 바꿀 수는 없다. $\mathbb Q$의 원소는 고정해야 하므로 $2$를 $3$으로 보낼 수 없기 때문이다.

허용되는 선택은 두 제곱근의 부호를 각각 고르는 것뿐이다.

$$
\begin{gathered}
\sqrt2\mapsto\varepsilon\sqrt2,\\
\sqrt3\mapsto\delta\sqrt3,\\
\varepsilon,\delta\in\{1,-1\}.
\end{gathered}
\tag{17}
$$

이 선택은 정확히 네 가지이며, field의 모든 원소에 다음과 같이 작용한다.

$$
\begin{gathered}
a+b\sqrt2+c\sqrt3+d\sqrt6\\
\longmapsto
a+\varepsilon b\sqrt2\\
+\delta c\sqrt3+\varepsilon\delta d\sqrt6.
\end{gathered}
\tag{18}
$$

마지막 항의 부호도 곱셈 관계를 따라야 한다. $\sqrt6=\sqrt2\sqrt3$이므로 $\sqrt6$의 부호만 따로 고를 수는 없다.

(18)의 $1$, $\sqrt2$, $\sqrt3$, $\sqrt6$는 $\mathbb Q$ 위의 basis이므로 이 field의 degree는 $4$다. $\sqrt3$를 유리수 $a,b$를 써서 $a+b\sqrt2$로 나타낼 수 없으므로, 이 수를 추가하면 degree가 다시 $2$배가 된다. 따라서 tower law에 의해 $[L:\mathbb Q]=2\cdot2$다.

<details markdown="1">
<summary>sqrt(3)가 Q(sqrt(2)) 안에 들어 있지 않은 이유</summary>

유리수 $a,b$로 $\sqrt3=a+b\sqrt2$라고 쓸 수 있다고 가정하자. 제곱하면 $3=a^2+2b^2+2ab\sqrt2$다. $1,\sqrt2$가 independent하므로 $ab=0$이어야 한다. 따라서 $a^2=3$이거나 $b^2=3/2$여야 하지만, 둘 다 유리수의 제곱은 아니다. 유리수를 분수로 써서 제곱하면 각 소수에 대해 분자와 분모의 지수 차이가 짝수가 된다. 그런데 $3$과 $3/2$에서 소수 $3$의 지수는 홀수다. 따라서 $\sqrt3\notin\mathbb Q(\sqrt2)$이며, $x^2-3$은 이 field 위에서도 irreducible하다. 그래서 두 번째 확장의 degree도 $2$다.

</details>

$\sqrt2$의 부호를 바꾸는 변환을 $\sigma$, $\sqrt3$의 부호를 바꾸는 변환을 $\tau$라고 하면 다음 관계가 성립한다.

$$
\begin{gathered}
G=\{e,\sigma,\tau,\sigma\tau\},\\
\sigma^2=\tau^2=e,\\
\sigma\tau=\tau\sigma.
\end{gathered}
\tag{19}
$$

$C_m$은 원소가 $m$개인 cyclic group이므로, $C_2$는 부호 선택 하나에 해당한다. (19)의 group을 **Klein four-group** $V_4$라고 하며 $C_2\times C_2$라고도 쓴다. 여기서 $\times$는 **direct product**를 뜻한다. 두 부호 선택을 하나의 쌍으로 묶고, 합성할 때는 각 좌표의 부호를 따로 곱한다. 예를 들어 $(-1,+1)(+1,-1)=(-1,-1)$이다. $\sqrt3$의 부호만 바꾼 뒤 $\sqrt2$의 부호만 바꾸면 두 부호가 모두 바뀐다는 뜻이다. 순서를 뒤집어도 결과는 같다.

어떤 두 원소든 합성 순서를 바꾸어도 결과가 같은 group을 **abelian**이라고 한다. 순서에 따라 결과가 달라지는 원소 쌍이 하나라도 있으면 **nonabelian**이다. 이 group은 독립적인 두 부호 변환이 commute하므로 abelian이다. 반면 앞 절의 겹치는 cycle 예에서 보았듯이, permutation으로 이루어진 group이라고 해서 모두 abelian인 것은 아니다.

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-03-permutations.svg" alt="Real axis에 실제 값대로 표시한 근들의 sign flip과, 제곱값이 서로 다른 근을 바꾸려는 교환의 확대도." width="960" height="480">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-03-permutations-mobile.svg" alt="Real axis에 실제 값대로 표시한 근들의 sign flip과, 제곱값이 서로 다른 근을 바꾸려는 교환의 확대도. 모바일용 배치." width="480" height="800">
</div>
*Figure 3. (a)의 real axis에는 근을 실제 값에 맞춰 표시했다. 각 sign flip은 제곱값이 같은 두 근을 서로 바꾼다. (b)는 가로축을 별도로 확대한 그림이다. 제곱값이 $2$와 $3$인 근을 서로 바꾸려는 허용되지 않는 교환을 보여 준다.*

{% include interactive/article-lab.html id="galois-permutations" %}

$24$개의 permutation을 하나씩 살펴보면서 제곱 관계와 서로 음수인 두 근의 관계가 유지되는지 확인해 보자. 이 예에서는 이 조건들만 확인하면 된다. 조건을 통과하는 permutation은 모두 (17)의 부호 선택이고, (18)에 따라 field 전체의 automorphism으로 확장할 수 있기 때문이다.

## 7. Galois correspondence: field와 subgroup의 대응
{: id="7-galois-correspondence-고정하는-수가-많을수록-허용되는-변환은-줄어든다" }

**Subgroup** $H\le G$는 identity를 포함하며, 그 원소들을 합성하거나 inverse를 취해도 다시 같은 집합의 원소가 되는 부분집합이다. 여기서 $\le$는 수의 크기를 비교하는 부등호가 아니라 “subgroup이다”라는 뜻이다. $G=\operatorname{Gal}(L/F)$는 $F$를 고정하며 $L$의 사칙연산을 보존하는 map들의 group으로 두자. 앞의 quartic에서 $\lbrace e,\sigma\rbrace $는 $\sigma^2=e$이므로 subgroup이다. 반면 $\lbrace e,\sigma,\tau\rbrace $는 subgroup이 아니다. $\sigma$와 $\tau$를 합성한 $\sigma\tau$가 빠져 있기 때문이다. 주어진 subgroup의 모든 변환이 그대로 두는 수들의 집합을 **fixed field**라고 한다.

$$
\begin{gathered}
L^H=\{u\in L:\sigma(u)=u\\
\text{for every }\sigma\in H\}.
\end{gathered}
\tag{20}
$$

위첨자 $H$는 어떤 변환들을 검사하는지 표시할 뿐, 거듭제곱을 뜻하지 않는다. 그 변환들이 하나도 빠짐없이 $u$를 고정해야 $u\in L^H$다. 이 집합도 field를 이룬다. 모든 변환이 $u,v$를 고정하면 $u+v$, $u-v$, $uv$, 그리고 $v\ne0$일 때의 $u/v$도 고정하기 때문이다.

반대로 base field $F$를 포함하면서 $L$ 안에 들어 있는 field $E$를 **intermediate field**라고 한다. $\mathbb Q\subseteq\mathbb Q(\sqrt3)\subseteq\mathbb Q(\sqrt2,\sqrt3)$가 그 예다. $E$의 모든 수를 그대로 두는 map들을 모으면 다음 subgroup을 얻는다.

$$
\begin{gathered}
\operatorname{Gal}(L/E)\\
=\{\sigma\in G:\sigma(u)=u\\
\text{for every }u\in E\}.
\end{gathered}
\tag{21}
$$

여기서 “field를 고정한다”는 것은 각 원소를 그대로 둔다는 뜻이다. Field 전체를 자기 자신으로 보내는 것만으로는 충분하지 않다. 예를 들어 $\sigma$는 $\mathbb Q(\sqrt2)$를 같은 집합으로 보내지만, $\sqrt2$ 자체는 $-\sqrt2$로 바꾼다. 따라서 이 field를 **pointwise**로 고정하지는 않는다. 반면 유리수 $a,c$로 나타낸 $a+c\sqrt3$는 모두 그대로 둔다.

앞의 quartic 예에서 유리수 좌표 $(a,b,c,d)$는 $a+b\sqrt2+c\sqrt3+d\sqrt6$를 나타낸다. $\sigma$를 적용하면 좌표가 $(a,-b,c,-d)$로 바뀐다. 원래 vector와 같으려면 $b=-b$, $d=-d$여야 하므로 $b=d=0$이다. 따라서 $a+c\sqrt3$ 꼴의 수들만 남고,

$$
L^{\langle\sigma\rangle}=\mathbb Q(\sqrt3).
\tag{22}
$$

이다. $\langle\sigma\rangle=\lbrace e,\sigma\rbrace $는 $\sigma$가 생성하는 subgroup이다. 마찬가지로

$$
\begin{aligned}
L^{\langle\tau\rangle}&=\mathbb Q(\sqrt2),\\
L^{\langle\sigma\tau\rangle}&=\mathbb Q(\sqrt6),\\
L^G&=\mathbb Q,\qquad L^{\{e\}}=L.
\end{aligned}
\tag{23}
$$

을 얻는다. $\sigma\tau$가 $\sqrt6$를 고정하는 이유는 두 부호가 함께 바뀌기 때문이다. $(-\sqrt2)(-\sqrt3)=\sqrt6$이지만, $\sqrt2$와 $\sqrt3$ 각각은 바뀐다. 곱을 고정한다고 각 인수까지 고정하는 것은 아니다.

<details markdown="1">
<summary>Linear system으로 fixed field 계산하기</summary>

이는 여러 변환의 공통 eigenspace를 구하는 계산이다. 선택한 각 변환에 대해 $\sigma-I$를 만들면, 고정되는 vector들은 이 kernel들의 교집합에 속한다. Basis를 $(1,\sqrt2,\sqrt3,\sqrt6)$ 순서로 쓰면 $\sigma=\operatorname{diag}(1,-1,1,-1)$이고, $\sigma-I=\operatorname{diag}(0,-2,0,-2)$다. 따라서 kernel의 좌표는 $(a,0,c,0)$이며, 이에 해당하는 수는 $a+c\sqrt3$다.

$\sigma$와 $\tau$가 모두 고정하는 vector를 찾으면 첫 번째 조건에서 $b=d=0$, 두 번째 조건에서 $c=d=0$을 얻는다. 결국 유리수 좌표 $a$만 남으므로 $L^G=\mathbb Q$다.

</details>

### Finite Galois extension의 fundamental theorem

Finite Galois extension $L/F$에서 다음 두 대응을 생각하자.

$$
\begin{gathered}
E\longmapsto\operatorname{Gal}(L/E),\\
H\longmapsto L^H.
\end{gathered}
\tag{24}
$$

두 대응은 intermediate field와 subgroup을 일대일로 연결한다. Field에서 출발해 이를 고정하는 subgroup을 구하고, 그 subgroup의 fixed field를 다시 구하면 처음 field로 돌아온다. Subgroup에서 출발해도 마찬가지다. 포함관계는 반대로 대응한다. $E_1\subseteq E_2$이면 $E_2$의 모든 원소를 고정하는 조건에는 $E_1$을 고정하는 조건도 포함되므로 $\operatorname{Gal}(L/E_2)\subseteq\operatorname{Gal}(L/E_1)$이다. 크기에 대해서는 다음 식들이 성립한다.

$$
\begin{aligned}
[L:L^H]&=\lvert H\rvert,\\
[L^H:F]&=\frac{\lvert G\rvert}{\lvert H\rvert},\\
[L:F]&=\lvert G\rvert.
\end{aligned}
\tag{25}
$$

**Order** $\lvert H\rvert$는 $H$에 속한 변환의 수이고, $[L:E]$는 vector space의 dimension이다. 이 정리는 서로 다른 두 종류의 수가 같다는 사실을 말한다. $H=\lbrace e,\sigma\rbrace $에는 전체 변환 $4$개 중 $2$개가 들어 있고, 그 fixed field $\mathbb Q(\sqrt3)$의 $\mathbb Q$ 위 degree는 $4/2=2$다. Extension $L/\mathbb Q(\sqrt3)$의 degree도 $2$이므로 tower law의 $4=2\cdot2$를 다시 확인할 수 있다. 이 비율 $\lvert G\rvert/\lvert H\rvert$를 subgroup의 **index**라 하고 $[G:H]$로 쓴다. 여기서는 $2$다. Basis vector가 아니라, 같은 수의 map으로 이루어진 묶음이 몇 개인지 세는 수다. [Milne, §3](https://www.jmilne.org/math/CourseNotes/FT.pdf)

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-04-correspondence.svg" alt="각 vertex에 field와 이를 고정하는 subgroup을 함께 표시한 Hasse diagram. 두 포함관계의 방향은 서로 반대다." width="960" height="590">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-04-correspondence-mobile.svg" alt="각 vertex에 field와 이를 고정하는 subgroup을 함께 표시한 Hasse diagram. 두 포함관계의 방향은 서로 반대다. 모바일 배치." width="480" height="710">
</div>
*Figure 4. 각 vertex의 파란 label은 field $E$, 청록 label은 이를 고정하는 subgroup $\operatorname{Gal}(L/E)$다. 선은 바로 이웃한 포함관계를 연결하며, 선을 여러 개 따라가도 포함관계가 성립한다. Field가 커지는 방향은 위쪽이고, subgroup이 커지는 방향은 아래쪽이다. 이렇게 포함관계를 그린 것을 Hasse diagram이라고 한다.*

{% include interactive/article-lab.html id="galois-correspondence" %}

Generator를 골라 어떤 subgroup이 생성되고 어떤 좌표가 고정되는지 확인해 보자. $\sigma$와 $\tau$를 선택하면 $\sigma\tau$도 포함된다. 두 변환이 생성하는 subgroup에는 그 합성도 있어야 하기 때문이다.

<details markdown="1">
<summary>Galois correspondence가 일대일 대응인 이유</summary>

Finite extension에 관한 두 사실로 증명할 수 있다.

**$F$-embedding**은 사칙연산을 보존하고 $F$를 고정하되, 원래보다 큰 field로 보낼 수도 있는 map이다. 서로 다른 두 입력을 같은 값으로 보내지 않는 **injective** map이어야 한다. 예를 들어 $a+b\sqrt2\mapsto a-b\sqrt2$는 $\mathbb Q$를 고정하면서 $\mathbb Q(\sqrt2)$를 $\mathbb C$ 안으로 보내는 embedding이다. 두 image가 같다고 놓고 빼면 $1,\sqrt2$의 independence에 의해 입력의 두 계수가 각각 같아야 한다. 이 map의 image는 $\mathbb Q(\sqrt2)$일 뿐 $\mathbb C$ 전체는 아니다. Embedding이 도착 field를 빈틈없이 채울 필요는 없다.

$\alpha=\sqrt[3]2$에 대해 $\mathbb Q(\alpha)$에서 $\mathbb C$로 가는 embedding은 세 개다. $\alpha$를 $\alpha$, $\alpha\omega$, $\alpha\omega^2$ 중 하나로 보낸 뒤, $\alpha$의 모든 유리식에 같은 값을 대입하면 된다. 원래의 실수 field를 벗어나지 않는 것은 첫 번째뿐이다. 반면 automorphism은 주어진 field에서 자기 자신으로 가는 invertible map이어야 한다.

**Algebraic closure**는 $F$ 위에서 algebraic한 원소들로 이루어지며, 그 안의 원소를 계수로 하는 모든 nonconstant polynomial이 완전히 분해되는 field다. 필요한 algebraic root들을 한곳에서 다룰 수 있게 해 준다. $\mathbb Q$의 경우에는 복소수 중 algebraic한 수 전체가 이에 해당하며, $\sqrt2$, $i$, 앞의 cubic의 근들이 모두 들어 있다.

첫째, degree가 $d$인 finite separable extension에는 base field를 고정하며 algebraic closure로 가는 embedding이 정확히 $d$개 있다. 수 하나로 생성한 extension $F(\alpha)$에서는 $\alpha$의 image가 그 minimal polynomial의 근이어야 하므로 $d$가지 선택이 있다. Extension을 연달아 만들면 선택의 수도 dimension처럼 곱해진다. Separability 때문에 각 선택은 서로 다른 embedding을 주며, characteristic zero에서는 separability가 자동으로 성립한다. Splitting field는 normal하므로 embedding이 같은 field 안으로 값을 보내며, 따라서 automorphism이 된다.

둘째, 유한한 group $H$가 field $L$에 automorphism으로 작용하면 $[L:L^H]=\lvert H\rvert$이다. 이 **fixed-field theorem**은 다음 linear-algebra 논증으로 이해할 수 있다.

$h=\lvert H\rvert$라고 하자. $L^H$ 위에서 independent한 원소 $x_1,\ldots,x_m$이 $m>h$개 있다고 가정하면, 다음 $h$개의 방정식은 nonzero solution $y\in L^m$을 갖는다.

$$
\sum_{j=1}^{m}\sigma(x_j)y_j=0,
\qquad \sigma\in H
\tag{26}
$$

미지수보다 방정식이 적기 때문이다. 그중 $0$이 아닌 좌표가 가장 적은 solution을 고른다. 그 좌표 하나로 전체를 나누어 해당 좌표를 $1$로 맞추자. 모든 계수와 미지수에 임의의 $\tau\in H$를 적용하면 방정식들의 순서만 바뀐다. $\tau\sigma$도 $H$의 원소들을 한 번씩 거치기 때문이다. 따라서 $\tau(y)-y$도 solution이다. $1$로 맞춘 좌표는 $1-1=0$이 되고 원래 $0$인 좌표는 여전히 $0$이므로, 차이가 $0$이 아니라면 nonzero 좌표 수가 더 줄어든다. 이는 처음에 최소로 골랐다는 조건에 어긋난다. 따라서 모든 $y_j$는 $H$가 고정해야 한다. 그러면 (26)에서 $\sigma=e$인 행이 $x_j$ 사이의 nonzero $L^H$-linear relation을 주어 independence에 모순이다. 따라서 dimension은 $h$ 이하다. 한편 degree가 $d$인 extension의 base-fixing embedding은 많아야 $d$개인데, $H$의 원소들이 이미 서로 다른 embedding $h$개이므로 dimension은 $h$ 이상이기도 하다.

임의의 intermediate field $E$에 대해 $L/E$도 separable splitting field이므로 $\lvert\operatorname{Gal}(L/E)\rvert=[L:E]$다. 이 subgroup의 fixed field는 $E$를 포함하고, 그 위로 $L$까지의 extension degree도 $[L:E]$와 같다. Tower law에 따라 이 fixed field는 $E$일 수밖에 없다. 반대로 $H\subseteq\operatorname{Gal}(L/L^H)$이고 양쪽 order가 모두 $[L:L^H]=\lvert H\rvert$이므로 두 subgroup도 같다.

실수 field $\mathbb Q(\sqrt[3]2)$는 normality가 왜 필요한지 보여 준다. Degree는 $3$인데 $\mathbb Q$-automorphism은 identity 하나뿐이다. 나머지 두 embedding의 image는 이 실수 field 밖에 있다.

</details>

## 8. Cubic: nonabelian이어도 radical로 풀리는 예
{: id="8-cubic-서로-commute하지-않아도-radical로-풀릴-수-있다" }

$L=\mathbb Q(\sqrt[3]2,\omega)$를 다시 보자. 이 extension의 degree는 $6$이다. $\alpha=\sqrt[3]2$로 두고 세 근 $\alpha,\omega\alpha,\omega^2\alpha$에 차례로 $1,2,3$이라는 label을 붙이자. 세 근을 어디로 보내는지 정하면 automorphism이 결정된다. Characteristic zero의 splitting field이므로 (25)에 따라 automorphism은 여섯 개다. 세 label의 permutation도 여섯 개뿐이므로 모든 permutation이 실제로 나타나고, Galois group은 $S_3$다.

$\rho=(123)$는 세 근을 한 칸씩 회전시킨다.

$$
\rho(\alpha)=\omega\alpha,\qquad
\rho(\omega)=\omega.
\tag{27}
$$

Complex conjugation $c=(23)$는 실근을 고정하고 나머지 두 근을 바꾼다.

$$
c(\alpha)=\alpha,\qquad c(\omega)=\omega^2.
\tag{28}
$$

$\alpha$에만 적용해 봐도 합성 순서에 따라 결과가 달라진다. $\rho c(\alpha)=\omega\alpha$이지만 $c\rho(\alpha)=\omega^2\alpha$다. 첫 번째는 $c$가 실근 $\alpha$를 그대로 둔 다음 $\rho$가 회전시키는 순서다. 두 번째는 먼저 회전시킨 다음 $c$가 실수가 아닌 두 근을 맞바꾼다. $\rho^3=c^2=e$는 세 번 회전하거나 두 번 반사하면 모든 근이 제자리로 돌아온다는 뜻이다. $c\rho c^{-1}=\rho^{-1}$은 반사를 전후에 적용하면 회전 방향이 뒤집힌다는 관계로, 삼각형에서도 똑같이 성립한다.

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-05-cubic.svg" alt="Cubic의 세 근이 놓인 complex plane에서 120도 회전과 conjugation에 의한 reflection을 비교한 그림." width="960" height="550">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-05-cubic-mobile.svg" alt="Cubic의 세 근이 놓인 complex plane에서 120도 회전과 conjugation에 의한 reflection을 비교한 그림. 모바일 배치." width="480" height="960">
</div>
*Figure 5. $x^3-2$의 세 근은 반지름이 $\alpha=\sqrt[3]2$인 원 위에 놓인다. 이 근들에 $\rho$를 적용하면 $120^\circ$ 회전한다. Complex conjugation $c$는 real axis를 기준으로 반사하므로 $\alpha$는 그대로 두고 실수가 아닌 두 근만 바꾼다.*

그렇다고 noncommutative하다는 이유만으로 radical 공식이 불가능해지는 것은 아니다. 세 근 $\alpha,\omega\alpha,\omega^2\alpha$는 이미 radical로 표현되어 있다. 따라서 이 cubic처럼 radical로 풀리는 경우와, radical로 풀리지 않는 quintic을 구분하려면 다른 조건이 필요하다.

### Cardano 공식에 square root와 cube root가 등장하는 이유

일반적인 cubic에서는 근을 $x=u+v$로 놓는다. 전개하면 $uv$가 들어 있는 항이 생기는데, 이 곱을 적절히 정하면 $u^3,v^3$에 대한 두 조건만 남고 quadratic을 얻는다. 그 quadratic을 풀 때 square root가, $u,v$를 구할 때 cube root가 등장한다. Group이 nonabelian이어도 radical만으로 근을 구하는 계산은 가능하다.

<details markdown="1">
<summary>Cardano 공식의 유도와 계수를 넣어 계산하는 예</summary>

$Ax^3+Bx^2+Cx+D=0$에서 $A\ne0$이라고 하자. $A$로 나누고 $x=y-B/(3A)$를 대입하면 $y^2$ 항이 상쇄된다. 변수 이름을 다시 $x$로 쓰면 quadratic 항이 없는 **depressed cubic**을 얻는다.

$$
x^3+px+q=0.
\tag{29}
$$

$x=u+v$를 대입하면 $u^3+v^3+q+(3uv+p)(u+v)=0$이다. $uv=-p/3$, $u^3+v^3=-q$가 되도록 고르면 $u^3v^3=-p^3/27$이므로 $u^3,v^3$는 다음 quadratic의 두 근이다.

$$
t^2+qt-\frac{p^3}{27}=0.
\tag{30}
$$

Quadratic formula를 적용하면

$$
\begin{aligned}
u^3&=-\frac q2+\sqrt{\Delta},\\
v^3&=-\frac q2-\sqrt{\Delta},\\
\Delta&=\left(\frac q2\right)^2
       +\left(\frac p3\right)^3.
\end{aligned}
\tag{31}
$$

을 얻는다. 예를 들어 $x^3+3x+1=0$에서는 $p=3$, $q=1$, $\Delta=5/4$다. 따라서 $u^3=(-1+\sqrt5)/2$, $v^3=(-1-\sqrt5)/2$다. 양수인 첫 번째 값의 실수 cube root를 $u$로 고르고 $v=-1/u$로 정하면 $uv=-1$이 자동으로 성립한다. $v$를 세제곱하면 필요한 $v^3$ 값도 맞는지 확인할 수 있다. 그러므로 $u+v$는 원래 cubic의 근이다.

$0$이 아닌 복소수에는 cube root가 세 개 있다. 여기서 **branch**를 고른다는 것은 이 세 값 중 하나를 선택한다는 뜻이다. $u,v$는 $uv=-p/3$도 만족해야 하므로 각각 따로 고르면 안 된다. $u\ne0$이면 $u$를 먼저 고른 뒤 $v=-p/(3u)$로 정하면 된다. $\omega^ku$, $\omega^{-k}v$로 함께 바꾸면 곱은 그대로 유지되며, multiplicity를 포함해 세 근을 모두 얻는다. $p=0$이면 $x^3=-q$를 바로 풀면 되고, $p=q=0$도 여기에 포함된다.

최종 근이 모두 실수여도 계산 중에는 complex radical이 필요할 수 있다. Radical로 풀린다는 정의는 이런 중간값도 허용한다.

</details>

## 9. Normal subgroup, quotient, commutator

Group 자체는 nonabelian이어도 일부 변환을 하나로 묶으면 abelian group을 만들 수 있다. 이렇게 변환들을 묶어 만든 group을 **quotient group**이라고 한다.

앞서 본 quartic의 group $G=\lbrace e,\sigma,\tau,\sigma\tau\rbrace $를 쓰자. $\sigma$는 $\sqrt2$의 부호를 바꾸고 $\tau$는 $\sqrt3$의 부호를 바꾼다. $H=\lbrace e,\sigma\rbrace $로 놓고, $H$의 각 원소에 왼쪽에서 $\tau$를 합성하면 $\tau H=\lbrace\tau e,\tau\sigma\rbrace=\lbrace\tau,\sigma\tau\rbrace$다. 두 부호 변환이 commute하기 때문에 마지막 등호가 성립한다. 네 변환은 각각 $H$와 $\tau H$ 중 어느 한 집합에만 속한다.

일반적으로 $gH=\lbrace gh:h\in H\rbrace $를 **left coset**이라고 한다. Subgroup의 모든 원소에 같은 map $g$를 왼쪽에서 합성해 만든 집합이다. 이 연산은 $g^{-1}$로 되돌릴 수 있으므로 원소가 합쳐지거나 빠지지 않는다. 따라서 모든 coset의 원소 수는 $\lvert H\rvert$로 같고, 서로 다른 coset은 겹치지 않는다. **Index** $[G:H]$는 이 coset의 개수다. 여기서는 $2$이며, $4=2\cdot2$는 다음 식의 한 예다.

$$
\lvert G\rvert=[G:H]\,\lvert H\rvert.
\tag{32}
$$

### 작은 field에서 구별되지 않는 변환들
{: id="quotient는-제한한-뒤의-작용이-같은-변환들을-묶는다" }

이제 $E=\mathbb Q(\sqrt3)$에 속한 수만 보자. $H$의 두 원소는 모두 $E$에서 identity로 작용한다. $\sigma$가 더 큰 field에서는 identity가 아니지만 $\sqrt3$의 부호는 바꾸지 않기 때문이다. 반면 $\tau H$의 두 원소는 모두 $\sqrt3$를 $-\sqrt3$로 보낸다. $L$ 위의 네 변환을 $E$에 제한하면 서로 다른 변환은 두 개만 남는다.

이제 각 coset을 $E$에서 구별할 수 없는 map들을 모은 하나의 **class**로 취급한다. $H$는 $E$를 그대로 두는 class, $\tau H$는 $\sqrt3$의 부호를 바꾸는 class다. 곱셈은 $H\cdot H=H$, $H\cdot\tau H=\tau H\cdot H=\tau H$, $\tau H\cdot\tau H=H$가 된다. 마지막 곱에서 각 class의 $\tau$를 고르면 $\tau^2=e\in H$이고, 대신 $\sigma\tau$를 골라도 결과는 여전히 $H$ 안에 있다.

이 두 class가 이루는 group이 $G/H\cong C_2$다. $\cong$는 **isomorphic**하다는 뜻으로, 두 class에 $C_2$의 두 원소 이름을 붙이면 곱셈 규칙이 완전히 같다는 말이다. Quotient는 변환의 절반을 지우거나 각 변환을 $H$로 나누는 연산이 아니다. $E$에서 같은 작용을 하는 변환들을 하나의 원소로 취급하는 것이다.

**Group homomorphism**은 한 group에서 다른 group으로 가면서 합성을 보존하는 map $\phi$, 즉 $\phi(gh)=\phi(g)\phi(h)$를 만족하는 map이다. 여기서 $\phi(g)$는 $g$를 $E$에 **restriction**한 map이다. 변환 규칙은 그대로 쓰되 입력을 $E$의 원소로만 한정한다. 먼저 합성한 뒤 제한해도, 각각 제한한 뒤 합성해도 결과가 같으므로 homomorphism이다. 직접 쓰면 $\phi(e)=\phi(\sigma)=\operatorname{id}_E$이고, $\phi(\tau)=\phi(\sigma\tau)$는 $E$에서 부호를 바꾸는 map이다.

**Kernel**은 도착 group의 identity로 보내지는 입력들의 집합이며, 이 예에서는 정확히 $H$다. Linear map의 kernel과 비슷하지만 zero vector 대신 group의 identity를 쓴다. 이 restriction은 $E/F$의 automorphism 두 개를 모두 얻으므로 **surjective**하다. 반면 서로 다른 $e,\sigma$의 image가 같으므로 **injective**하지는 않다. Quotient를 만들면 큰 field에서 서로 다른 map이 하나로 합쳐질 수 있지만, 작은 field에서 가능한 작용은 모두 남는다.

### Normality가 필요한 이유

Coset끼리 곱한 결과는 각 coset에서 어느 **representative**, 즉 대표 원소를 골랐는지에 따라 달라지면 안 된다. $gN$과 $hN$의 곱을 $ghN$으로 정하려면, $g,h$를 각각 같은 class의 다른 원소로 바꾸어도 결과가 같아야 한다. 이를 보장하는 조건이 **normal subgroup**이다. $N$이 normal하다는 것은 모든 $g\in G$에 대해 $gNg^{-1}=N$이라는 뜻이다. 여기서 $gNg^{-1}$는 $n\in N$인 모든 $gng^{-1}$를 모은 집합이다. 이 연산으로 subgroup 안의 원소가 서로 바뀔 수는 있어도, subgroup이라는 집합 자체는 달라지지 않아야 한다.

$h$를 $ghg^{-1}$로 바꾸는 연산을 **group conjugation**이라고 한다. Permutation에서는 $h$가 움직이는 label을 $g$로 바꿔 붙이는 것과 같다. 예를 들어 $(123)(12)(132)=(23)$이고, 일반적으로 $g(ab)g^{-1}=(g(a)\,g(b))$다. 앞서 나온 complex conjugation $z\mapsto\bar z$와는 구분해야 한다. Group conjugation은 group 원소를 다른 group 원소로 바꾸는 연산이고, complex conjugation은 수에 작용하는 특정한 map이다.

$V_4$ 예에서는 모든 원소가 commute하므로 $ghg^{-1}=h$다. 예를 들어 $\tau\sigma\tau^{-1}=\sigma$이므로 $H$는 normal하다. Nonabelian group에서는 항상 그렇지 않다. $H\triangleleft G$는 “$H$가 $G$의 normal subgroup이다”라는 표기다. Finite Galois extension에서는 field와 normal subgroup 사이에 다음 관계가 성립한다.

$$
\begin{gathered}
E/F\text{ is Galois}\iff H\triangleleft G,\\
\operatorname{Gal}(E/F)\cong G/H.
\end{gathered}
\tag{33}
$$

여기서 $G=\operatorname{Gal}(L/F)$, $E=L^H$다. **Normal**이라는 같은 단어를 쓰지만 대상에 따라 정의가 다르다. Field extension $E/F$에서는 $F$ 위의 irreducible polynomial이 $E$에 근 하나를 가지면 모든 근이 $E$에 있다는 조건이다. Subgroup $H$에서는 conjugation이 $H$라는 집합을 바꾸지 않는다는 조건이다. 식 (33)은 이 두 조건을 연결한다. $H$가 normal이면 $L$의 모든 automorphism은 $E$를 다시 $E$로 보낸다. 이 map들을 $E$에 제한하면 $E/F$의 모든 automorphism을 얻으며, restriction의 kernel은 $H$다. 이 restriction으로 얻는 group이 $G/H$다.

<details markdown="1">
<summary>S3에서 quotient를 만들 수 없는 subgroup의 예</summary>

Cubic 예에서 $\rho=(123)$, $c=(23)$, $H=\lbrace e,c\rbrace $로 두자. 직접 합성하면 $\rho H=\lbrace (123),(12)\rbrace $이지만 $H\rho=\lbrace (123),(13)\rbrace $이므로 left coset과 right coset이 다르다. Label을 바꾸면 $\rho H\rho^{-1}=\lbrace e,(13)\rbrace \ne H$가 된다. 따라서 $H$는 normal하지 않다.

실제로 class의 곱이 하나로 정해지지 않는다. $e,c$는 같은 class $H$의 원소인데, 각각 오른쪽에 $\rho$를 곱하면 $e\rho=(123)$, $c\rho=(13)$이다. $(13)$은 $\rho H$ 안에 없으므로 두 결과는 서로 다른 left coset에 속한다. 따라서 “$H$와 $\rho H$를 곱한다”는 계산이 대표 원소에 따라 달라진다.

이 subgroup의 fixed field는 실수 field $\mathbb Q(\sqrt[3]2)$이며, $\mathbb Q$ 위의 Galois extension이 아니다. Splitting field의 automorphism 중에는 이 실수 field 밖으로 수를 보내는 것도 있다. 따라서 모든 automorphism을 이 field에서 자기 자신으로 가는 map으로 제한할 수는 없다.

반면 rotation으로 이루어진 $N=\lbrace e,\rho,\rho^2\rbrace $는 normal하다. Reflection은 $\rho$와 $\rho^{-1}$을 바꿀 뿐 집합 자체는 바꾸지 않는다. 두 coset은 세 rotation과 세 reflection이며, $S_3/N\cong C_2$를 이룬다. 이는 fixed field $\mathbb Q(\omega)$의 $\mathbb Q$ 위의 automorphism 두 개에 대응한다.

</details>

### Commutator와 abelian quotient
{: id="commutativity를-강제했을-때-사라져야-하는-부분" }

$g,h$의 **commutator**는

$$
[g,h]=ghg^{-1}h^{-1}.
\tag{34}
$$

다. $gh=[g,h]hg$이므로 두 합성 순서의 차이를 나타내며, $gh=hg$일 때, 그리고 그때만 identity가 된다. 여기서는 곱셈과 inverse로 정의한 group commutator를 사용하며, matrix에서 쓰는 $gh-hg$와는 다르다. 예를 들어 오른쪽부터 합성하면 $[(12),(23)]=(132)$다. 네 변환을 차례로 적용하면 $1\mapsto3$, $2\mapsto1$, $3\mapsto2$임을 확인할 수 있다.

모든 commutator를 포함하는 가장 작은 subgroup을 **derived subgroup** $G'$이라고 한다. Commutator만 모으는 데서 그치지 않고, 곱과 inverse를 계속 추가해 subgroup을 만들어야 한다. Prime 기호는 이 연산의 이름을 붙인 것이며 미분을 뜻하지 않는다. $V_4$처럼 abelian인 group에서는 모든 commutator가 $e$이므로 $V_4'=\lbrace e\rbrace$다. $S_3$에서는 방금 얻은 3-cycle이 $\lbrace e,(123),(132)\rbrace $를 생성한다. 여섯 변환으로 만드는 commutator를 모두 계산하면 이 집합에 속하므로, 이것이 $S_3'$이다. 이 group은 order가 $3$인 cyclic group이다.

Commutator를 conjugation하면 다시 commutator가 되므로 $G'$은 normal하다.

$$
k[g,h]k^{-1}=[kgk^{-1},khk^{-1}].
\tag{35}
$$

$G/G'$에서는 모든 commutator가 identity class에 속하므로 quotient가 abelian이다. 반대로 $G/N$이 abelian이면 모든 commutator가 $N$에 속해야 하므로 $G'\subseteq N$이다. 즉 $G'$의 원소들은 abelian quotient를 만들 때 반드시 identity와 같은 class로 묶인다.

이 연산을 반복하면

$$
\begin{gathered}
G^{(0)}=G,\\
G^{(j+1)}=(G^{(j)})'.
\end{gathered}
\tag{36}
$$

이라는 **derived series**를 얻는다. 위첨자 $(j)$는 원소의 거듭제곱이 아니라 연산을 반복한 횟수다. 매 단계에서 처음 group을 다시 쓰는 것이 아니라, 바로 앞 단계에서 얻은 group의 derived subgroup을 구한다. 이 과정을 반복해 $\lbrace e\rbrace $에 도달하는 finite group을 **solvable**이라고 한다. Cubic에서는 $S_3'=\lbrace e,(123),(132)\rbrace $가 cyclic이므로 abelian이다. 이 세 원소 사이의 commutator는 모두 $e$이므로 그다음 derived subgroup은 $\lbrace e\rbrace $다. Order는 $6\to3\to1$로 줄어든다. 처음 group이 nonabelian이어도 마지막에는 identity에 도달할 수 있다.

동치인 정의도 있다. 각 subgroup이 바로 앞 subgroup에서 normal하고, 연속한 두 subgroup의 quotient가 abelian인 chain이 $\lbrace e\rbrace $까지 이어지면 solvable하다. 각 quotient는 변환들을 나누어 살펴볼 때 얻는 abelian 단계 하나에 해당한다. Derived series가 identity에 도달하면 그 자체가 이런 chain이다. 반대로 identity group에 도달하지 않는다면, 다른 subgroup chain을 골라도 이 조건을 만족시킬 수 없다.

<details markdown="1">
<summary>다른 chain을 골라도 derived series의 결론을 피할 수 없는 이유</summary>

$G=N_0\supseteq N_1\supseteq\cdots\supseteq N_r=\lbrace e\rbrace$에서 각 $N_{j+1}$이 $N_j$의 normal subgroup이고 $N_j/N_{j+1}$이 abelian이라고 하자. $N_j$의 commutator는 이 quotient에서 identity가 되므로 $N_j'\subseteq N_{j+1}$이다. $G^{(0)}=N_0$에서 출발하면 차례로 $G^{(1)}\subseteq N_1$, $G^{(2)}\subseteq N_2$를 얻고, 마지막에는 $G^{(r)}\subseteq N_r=\lbrace e\rbrace$다. 이런 chain이 하나라도 있다면 derived series도 반드시 identity에 도달한다.

</details>

## 10. Radical 공식이 있으면 Galois group이 solvable한 이유
{: id="10-radical로-풀리면-galois-group이-solvable한-이유" }

유한한 radical expression은 root를 하나씩 차례로 구하며 계산할 수 있다. 각 단계까지 얻은 수들을 담는 field를 이어 놓은 것이 **radical tower**다.

$$
\begin{gathered}
F=K_0\subseteq\cdots\subseteq K_r,\\
K_j=K_{j-1}(\alpha_j),\\
\alpha_j^{m_j}\in K_{j-1}.
\end{gathered}
\tag{37}
$$

여기서 $j$는 단계 번호, 정수 $m_j\ge2$는 몇 제곱근을 구하는지, $\alpha_j$는 그중 선택한 root 하나를 뜻한다. $\alpha_j^{m_j}\in K_{j-1}$이라는 조건은 **radicand**, 즉 근호 안에 들어가는 값이 앞 단계에서 이미 준비되어 있다는 뜻이다. $K_{j-1}(\alpha_j)$에는 기존 field와 $\alpha_j$뿐 아니라 이들을 사칙연산으로 조합한 수들도 모두 포함된다.

예를 들어 $K_0=\mathbb Q$에 $\sqrt3$를 추가해 $K_1$을 만든 뒤, $\xi=\sqrt{2+\sqrt3}$를 추가한다고 하자. 첫 단계는 지수 $m_1=2$, radicand $3$이고, 둘째 단계는 지수 $m_2=2$, radicand $2+\sqrt3\in K_1$이다. 사칙연산의 결과는 현재 field에 속하지만, 새로운 radical을 구하려면 더 큰 field가 필요할 수 있다. 매번 field가 커져야 하는 것은 아니다. 이미 들어 있는 root를 추가하면 field는 그대로다.

Polynomial이 radical로 풀리려면 그 splitting field가 마지막 radical field에 **포함**되어야 한다. 두 field가 같을 필요는 없다. 공식의 중간 계산에 근을 표현하는 데 꼭 필요한 것보다 더 많은 수가 쓰일 수 있기 때문이다.

먼저 현재 field $K$에 모든 $m$-th root of unity, 즉 $m$제곱하면 $1$이 되는 수들이 들어 있다고 하자. 거듭제곱으로 이 $m$개를 모두 얻을 수 있는 primitive root $\zeta_m$를 하나 고른다. $\beta^m=a\in K$, $a\ne0$이면 $K$를 고정하는 automorphism은 $\beta$를 $x^m-a$의 다른 근으로 보내야 한다. 가능한 값은

$$
\beta,\ \zeta_m\beta,\ldots,\zeta_m^{m-1}\beta.
\tag{38}
$$

뿐이다. 이 근들이 모두 $K(\beta)$에 있으므로 $K(\beta)$는 $K$ 위에서 $x^m-a$의 splitting field다. Characteristic zero에서는 Galois extension이 된다. $K$를 고정하는 각 map은 $\beta\mapsto\zeta_m^k\beta$라는 선택으로 결정된다. 지수가 각각 $k,\ell$인 두 map을 합성하면 $\beta\mapsto\zeta_m^{k+\ell}\beta$가 된다. $\zeta_m$를 고정하므로 다음 map을 적용해도 이 multiplier는 바뀌지 않기 때문이다. $\zeta_m^m=1$이어서 지수는 $m$마다 다시 돌아온다. 지수를 더하는 순서는 결과에 영향을 주지 않으므로 이 map들은 commute한다. 따라서 group은 $C_m$ 전체일 수도, 그보다 작은 subgroup일 수도 있지만 어느 경우든 abelian이다.

Cubic에서 이를 직접 확인할 수 있다. $K=\mathbb Q(\omega)$ 위에서 $\beta=\sqrt[3]2$를 $\omega\beta$로 보내는 map은 세 번 적용하면 identity가 된다. $\omega^2\beta$를 한 번 더 보내면 $\omega^3\beta=\beta$로 돌아오기 때문이다. 지수 $2$인 map을 두 번 적용하면 $\omega^4\beta=\omega\beta$다. $2+2=4$를 modulo $3$에서 $1$로 계산하는 것과 같다. Modulo $3$은 $3$으로 나눈 나머지만 남긴다는 뜻이다. $\mathbb Q$ 위의 전체 group은 $S_3$였지만, $\omega$를 먼저 추가하면 남은 radical 단계의 group은 $C_3$다.

Root of unity가 중요한 이유도 여기에 있다. 원래의 실수 extension $\mathbb Q(\sqrt[3]2)/\mathbb Q$는 Galois가 아니다. 따라서 임의의 radical tower를 그대로 cyclic Galois extension들의 tower라고 생각하면 안 된다.

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-06-radical-tower.svg" alt="Q(zeta5) 위에서 x to the fifth minus 2의 다섯 근에 72도 회전을 적용했을 때의 cyclic orbit." width="960" height="560">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-06-radical-tower-mobile.svg" alt="Q(zeta5) 위에서 x to the fifth minus 2의 다섯 근에 72도 회전을 적용했을 때의 cyclic orbit. 모바일 배치." width="480" height="650">
</div>
*Figure 6. $K=\mathbb Q(\zeta_5)$, $\beta=\sqrt[5]2$일 때 $K(\beta)/K$의 degree는 $5$다. Automorphism $\sigma$는 $K$를 고정하고 $\beta$를 $\zeta_5\beta$로 보낸다. 다섯 근 위에서 이 작용은 $72^\circ$ 회전으로 나타나며 $C_5$를 생성한다.*

Tower 전체에서는 앞 단계의 radicand도 map에 따라 바뀔 수 있다는 점을 처리해야 한다. Algebraic number의 **$F$-conjugate**는 그 수의 $F$ 위 minimal polynomial의 근들이다. 같은 뜻으로, $F$를 고정하고 사칙연산을 보존하되 더 큰 field로 값을 보낼 수도 있는 map이 그 수를 보낼 수 있는 값들이다. 예를 들어 $2+\sqrt3$의 $\mathbb Q$ 위 minimal polynomial은 $t^2-4t+1$이므로 conjugate는 $2+\sqrt3$, $2-\sqrt3$다. 둘 다 실수다. 여기서 conjugate는 complex conjugation으로 얻는 값만을 뜻하지 않으며, group 원소의 conjugation과도 다른 개념이다.

Root of unity와 각 conjugate radicand에 해당하는 근들을 모두 추가하면 tower의 각 단계를 Galois extension으로 확장할 수 있다. 그러면 앞 단계 field의 map을 더 큰 field의 map으로 확장할 수 있다. 각 restriction의 kernel에는 앞 단계 field는 고정하고 새 radical 선택만 바꾸는 map들이 들어 있다. 새 radical마다 multiplier의 지수를 좌표 하나에 기록하면, 합성은 각 좌표를 extraction exponent $m_j$에 대한 modulo로 더하는 연산이 된다. 이 덧셈들은 모두 commute하므로 radical을 여러 개 함께 추가해도 kernel은 abelian이다. 따라서 앞 단계 group에 abelian 단계를 하나씩 더해 가는 구조가 되고, 유한 번 반복한 전체 group은 solvable하다. 이를 원래 polynomial의 splitting field에 제한해 얻는 quotient도 solvable하다. **따라서 유한한 radical 공식이 존재하려면 Galois group이 solvable해야 한다.** 계산 중에 complex radical을 사용해도 이 조건은 달라지지 않는다.

<details markdown="1">
<summary>Radical tower를 포함하는 solvable Galois extension 구성하기</summary>

Finite extension의 **normal closure**는 그 extension을 포함하는 가장 작은 normal extension이다. Generator들의 base-field conjugate가 모두 들어 있어야 한다. 예를 들어 실수 field $\mathbb Q(\sqrt[3]2)$에는 $x^3-2$의 근 두 개가 빠져 있어 normal하지 않다. 그 근들까지 포함한 $\mathbb Q(\sqrt[3]2,\omega)$가 normal closure이며, characteristic zero에서는 Galois extension이다. 아래 구성에서는 최소한의 normal closure보다 더 많은 수를 추가할 수도 있지만, 원래 근들을 포함한다는 목적에는 문제가 없다.

Characteristic zero인 $F$의 algebraic closure 하나 안에서 모든 field를 생각하자. 계산에 필요한 algebraic root들을 함께 담는 field다. 모든 extraction exponent $m_j$의 배수인 $N$을 고르고 $T_0=F(\zeta_N)$로 시작한다. 이 field의 automorphism은 $\zeta_N$를 여전히 primitive인 $\zeta_N^u$로 보낸다. 이 map들을 합성하면 지수를 modulo $N$에서 곱하게 되며, 곱하는 순서는 상관없다. 따라서 첫 Galois group은 abelian이다.

그다음 단계마다 radicand의 $F$ 위의 모든 conjugate $a$에 대해 $x^{m_j}-a$의 모든 근을 추가한다. 이렇게 얻은 $T_j$는 원래 radical tower를 포함하며, $T_j/F$는 모두 finite Galois extension이다.

$T_{j-1}$을 고정하는 map은 새로 추가한 각 nonzero radical에 root of unity를 곱한다. 선택한 radical마다 multiplier를 좌표 하나에 기록하자. Map을 합성하면 각 좌표끼리 곱하므로 이 group은 abelian이다. $\operatorname{Gal}(T_{j-1}/F)$로 가는 restriction은 surjective하다. 앞 단계 field의 automorphism마다 이를 확장한 $T_j$의 automorphism이 하나 이상 있다는 뜻이다. 그 kernel은 방금 설명한 $\operatorname{Gal}(T_j/T_{j-1})$다. 앞 단계 group이 solvable이면 derived 연산을 충분히 반복한 결과는 그 group에서 identity로 보이므로 abelian kernel에 들어 있다. 여기서 한 번 더 반복하면 identity에 도달한다. 이를 차례로 적용하면 $\operatorname{Gal}(T_r/F)$가 solvable함을 알 수 있다.

원래 polynomial의 splitting field가 $L$이면 $L\subseteq T_r$다. Restriction으로

$$
\operatorname{Gal}(T_r/F)
\twoheadrightarrow\operatorname{Gal}(L/F).
\tag{39}
$$

를 얻는다. 두 갈래 화살표는 restriction이 surjective하다는 뜻이다. $L/F$의 모든 automorphism이 restriction으로 얻어진다. Kernel은 $L$에서 보면 identity인 map들의 group $\operatorname{Gal}(T_r/L)$이므로, 도착 group은 이 map들을 하나로 묶은 quotient다. Homomorphism은 곱과 inverse를 보존하므로 commutator도 commutator로 보낸다. 따라서 identity에 도달하는 derived series의 image도 identity에 도달하고, quotient 역시 solvable하다. 그러므로

$$
\begin{gathered}
f\text{ solvable by radicals}\\
\Longrightarrow\\
\operatorname{Gal}(L/F)\text{ solvable}.
\end{gathered}
\tag{40}
$$

가 성립한다.

각 단계의 확장을 구체적으로 구성하자. $T_{j-1}/F$가 Galois이고 $K_{j-1}$을 포함하며, $\alpha_j^{m_j}=a_j$라고 하자. 다음 polynomial들의 모든 근을 추가해 $T_j$를 만든다.

$$
\begin{gathered}
x^{m_j}-\sigma(a_j),\\
\sigma\in\operatorname{Gal}(T_{j-1}/F)
\end{gathered}
\tag{41}
$$

모든 $F$-embedding이 이 polynomial들과 그 근들을 서로 바꾸므로 normality가 성립한다. Characteristic zero이므로 separability도 성립한다. $\alpha_j$가 포함되어 있으므로 $K_j$도 포함된다.

각 nonzero conjugate radicand마다 nonzero root $\beta$를 하나씩 고르자. $\zeta_{m_j}\in T_{j-1}$이므로 이 근들만 추가해도 extension 전체를 생성한다. $T_{j-1}$을 고정하는 $g$에 대해 비율 $g(\beta)/\beta$를 각각 기록하면 root-of-unity group들의 곱으로 가는 injective homomorphism을 얻는다. 합성할 때 비율은 좌표별로 곱해지고, 모든 비율이 $1$인 map은 모든 generator를 고정하므로 field 전체의 identity이기 때문이다. Radicand가 $0$이면 extension이 생기지 않는다. 이 논증은 원래 radical tower가 Galois라고 가정하지 않는다. [Sharifi, Theorem 10.3.10](https://www.math.ucla.edu/~sharifi/notes/algebra-ch10.html)

</details>

## 11. Solvable group에서 radical을 얻는 방법
{: id="11-solvable-group에서-radical을-얻는-방법-eigenvector로-증명하기" }

앞 절의 명제는 역방향도 성립한다. 두 방향을 합친 것이 **Galois's solvability criterion**이다. Characteristic zero에서는

$$
\boxed{
\begin{gathered}
f\text{ solvable by radicals}\\
\iff\\
\operatorname{Gal}(L/F)\text{ solvable}
\end{gathered}}
\tag{42}
$$

이며, $L$은 $F$ 위에서 $f$의 splitting field다. [Milne, Theorem 3.28 and §5](https://www.jmilne.org/math/CourseNotes/FT.pdf)

Quadratic에서 eigenvector와 radical의 관계를 확인해 보자. $E=\mathbb Q(\sqrt2)$의 automorphism $\sigma(a+b\sqrt2)=a-b\sqrt2$는 basis $(1,\sqrt2)$에 대해 matrix $\operatorname{diag}(1,-1)$로 표현된다. Eigenvector $\beta=\sqrt2$의 eigenvalue는 $-1$이다. 제곱에는 이 부호 변화가 남지 않는다. $\sigma(\beta^2)=(-\beta)^2=\beta^2$이므로 $\beta^2$는 fixed field $\mathbb Q$의 원소이고, 여기서는 $2$다. 이 고정된 값에서 $\beta$를 구하는 것이 square root를 한 번 구하는 계산이다. 핵심은 eigenvector가 있다는 사실만이 아니라, 그 vector를 적절히 거듭제곱하면 group의 모든 map이 고정하는 값이 된다는 점이다.

Galois group이 cyclic인 Galois extension을 **cyclic extension**이라고 한다. 그 degree가 소수 $p$일 때도 적절한 eigenvector를 고르면 그 $p$제곱이 fixed field의 원소가 된다. 필요한 root of unity가 있으면 이런 단계를 radical 하나를 추가해 만들 수 있다는 뜻이다. Solvable group의 abelian quotient 단계들은 order가 소수인 cyclic 단계로 더 나눌 수 있다. 예를 들어 $S_3\supset C_3\supset\lbrace e\rbrace$의 quotient order는 $2,3$이므로 square-root 단계와 cube-root 단계에 대응한다. Galois correspondence로 group chain의 방향을 뒤집으면 이에 대응하는 field extension들을 얻는다. Radical 공식이 불가능함을 보이는 데는 앞 절의 정방향만으로 충분하다. 역방향은 group이 solvable하면 실제로 공식이 존재하는 이유를 설명한다.

<details markdown="1">
<summary>Cyclic extension에서 eigenvector로 radical을 얻는 증명</summary>

$E/K$가 Galois이고 그 group이 prime order $p$인 cyclic group이라고 하자. Generator를 $\sigma$라 하고 $\zeta_p\in K$라고 가정한다. Group의 map들은 $I,\sigma,\ldots,\sigma^{p-1}$이며 $I$는 identity operator다. Dimension 공식 (25)에 따라 $E$는 $K$ 위의 $p$차원 vector space다. $\sigma$는 $K$를 고정하므로 $K$-linear operator이기도 하다.

Polynomial에는 수 대신 operator도 대입할 수 있다. 예를 들어 $q(t)=t^2-1$에 $\sigma$를 대입하면 $q(\sigma)=\sigma^2-I$다. 이렇게 대입한 결과가 zero operator가 되는 nonzero monic polynomial 중 degree가 가장 작은 것을 **operator minimal polynomial**이라고 한다. 앞에서 수의 minimal polynomial은 그 수의 거듭제곱 사이의 관계였다면, 여기서는 map의 거듭제곱 사이의 관계다. Quadratic의 map $\operatorname{diag}(1,-1)$에서는 이것이 $t^2-1$이다. 제곱하면 $I$가 되지만, $\sigma-I$와 $\sigma+I$ 어느 것도 zero operator는 아니기 때문이다.

$\sigma^p=I$이므로 operator minimal polynomial은 $t^p-1$을 나눈다. 이를 보려면 $\sigma$를 대입했을 때 $0$이 되는 polynomial을 $q$로 나누면 된다. 나머지에도 $\sigma$를 대입하면 $0$인데 degree는 $q$보다 작으므로, minimality에 따라 나머지 자체가 $0$이어야 한다. 여기서는

$$
t^p-1=\prod_{j=0}^{p-1}(t-\zeta_p^j).
\tag{43}
$$

이다. Characteristic zero에서는 일차인수가 모두 서로 다르다. 서로 다른 일차인수의 곱을 대입했을 때 $0$이 되는 operator는 각 eigenvalue에 해당하는 성분으로 모든 vector를 분해할 수 있다. 따라서 $\sigma$는 **$K$ 위에서 diagonalizable**하다. $K$의 계수로 나타낸 eigenvector들로 basis를 잡으면 matrix가 diagonal이 된다는 뜻이다. 아래에 따로 접어 둔 성분 분해 계산으로 이 eigenbasis가 존재함을 직접 확인할 수 있다. 모든 eigenvalue가 $1$이라면 그 basis에서 $\sigma=I$여야 한다. 하지만 $\sigma$는 identity가 아니므로 eigenvalue가 $\zeta_p^j\ne1$인 nonzero eigenvector $\beta$가 존재한다.

$$
\begin{gathered}
\sigma(\beta)=\zeta_p^j\beta\\
\Longrightarrow
\sigma(\beta^p)=\beta^p.
\end{gathered}
\tag{44}
$$

$\sigma$가 $\beta^p$를 고정하면 그 거듭제곱들도 모두 $\beta^p$를 고정한다. 이것들이 cyclic group의 전체 원소이므로 $\beta^p$는 fixed field $K$에 속한다. $0\le k<p$에서 각 image는 $\sigma^k(\beta)=\zeta_p^{jk}\beta$다. $p$가 소수이고 $j\ne0$이므로 $\zeta_p^{jk}=\zeta_p^{j\ell}$이려면 이 범위에서는 $k=\ell$이어야 한다. 따라서 이 $p$개의 image는 모두 다르다. 이들은 **수** $\beta$의 $K$ 위 minimal polynomial의 근이므로 그 degree는 적어도 $p$다. 한편 그 degree가 $[E:K]=p$보다 클 수는 없으므로

$$
E=K(\beta),\qquad \beta^p\in K.
\tag{45}
$$

를 얻는다. Cyclic prime-degree Galois extension을 radical 하나를 추가하는 단계로 바꾼 것이다.

Finite solvable group의 abelian quotient 단계는 cyclic prime-order 단계들로 더 나눌 수 있다. 예를 들어 $V_4\supset\lbrace e,\sigma\rbrace\supset\lbrace e\rbrace$의 두 quotient는 order가 모두 $2$다. $V_4$ 자체는 cyclic이 아니어도 이렇게 나눌 수 있다. 일반적인 finite abelian group에서는 전체 group보다 작은 **proper subgroup**을 고르되, 더 큰 proper subgroup에 포함되지 않는 것으로 고른다. 그 quotient에는 identity group과 자기 자신 사이의 subgroup이 없다. 따라서 identity가 아닌 원소를 하나 고르면 그것만으로 quotient 전체를 생성해야 하므로 cyclic이다. 그 order는 소수여야 한다. Cyclic group의 order가 $a,b>1$인 합성수 $ab$이고 generator가 $g$라면, $g^a$의 거듭제곱들이 order가 $b$인 subgroup을 만들어 두 끝점 사이에 들어가기 때문이다. 선택한 subgroup 안에서도 이 과정을 반복한다. 원래 chain의 abelian quotient마다 이 분해를 적용하면 cyclic prime-order quotient들로 된 chain을 얻는다. Galois correspondence로 이에 대응하는 field extension들을 구하면 포함관계는 반대가 되고, 각 degree는 해당 prime order와 같다.

$G=\operatorname{Gal}(L/F)$에 대해 $F'=F(\zeta_{\lvert G\rvert})$로 두고, $L$과 $F'$를 함께 포함하는 가장 작은 field를 $L'=LF'$라고 쓰자. 필요한 root of unity가 원래 $L$에 들어 있다는 보장은 없다. $\zeta_{\lvert G\rvert}^{\lvert G\rvert}=1$이므로 이를 추가하는 과정 자체도 radical adjunction이다.

$L'/F'$는 확장한 base field 위에서 같은 polynomial의 splitting field다. 이 extension의 automorphism을 $L$에 제한하면 $G$로 가는 homomorphism을 얻는다. 이 map은 injective하다. 두 map이 $L$에서 같다면, 둘 다 $F'$를 고정하므로 $F'$에서도 같다. 따라서 이들로 만드는 모든 사칙연산 결과, 즉 $L'$ 전체에서 두 map이 같아야 한다. 그러므로 새 group의 곱셈 규칙은 $G$의 어떤 subgroup과 정확히 같고, 여전히 solvable하다. 새 group의 derived subgroup들은 $G$의 derived subgroup 안에 포함되므로 함께 identity에 도달하기 때문이다. 그 group의 chain을 세분할 때 나오는 prime order는 모두 $\lvert G\rvert$의 divisor이므로 $F'$에는 각 단계에 필요한 root of unity가 모두 있다. 이제 $F'$ 위에서 cyclic 단계마다 위 논증을 적용하면 $L'$을 포함하는 radical tower를 얻는다. 원래 polynomial의 모든 근도 이 tower에 포함된다.

</details>

<details markdown="1">
<summary>여기서 eigenbasis가 존재하는 이유</summary>

**Projector**는 $P^2=P$인 linear operator다. 한 성분을 남기는 연산이어서, 한 번 더 적용해도 결과가 같다는 뜻이다. Quadratic 예에서 $P_0=(I+\sigma)/2$는 $a+b\sqrt2$를 $a$로 보내고, $P_1=(I-\sigma)/2$는 $b\sqrt2$로 보낸다. 각각 한 eigenspace는 남기고 다른 eigenspace는 지운다. 둘의 합은 $I$이므로 두 결과를 더하면 원래 입력을 얻는다.

$j=0,\ldots,p-1$에 대해 linear operator $P_j=\frac1p\sum_{k=0}^{p-1}\zeta_p^{-jk}\sigma^k$를 정의하자. Eigenvalue가 $\zeta_p^{\ell}$인 eigenvector에 적용하면 $\frac1p\sum_{k=0}^{p-1}\zeta_p^{(\ell-j)k}$를 곱하는 셈이다. 이 값은 $\ell=j$이면 $1$, 아니면 $0$이다. 앞의 성분 선택 연산을 일반화한 것이다. Eigenbasis가 있다고 미리 가정하지 않고도 필요한 eigenvector들을 모두 얻을 수 있음을 확인하자. 합의 index를 한 칸 옮기고 $\sigma^p=I$를 이용하면 $\sigma P_j=\zeta_p^j P_j$다. 따라서 $P_j$의 image에 들어 있는 nonzero vector는 모두 eigenvalue $\zeta_p^j$에 대응한다.

또한 $\sum_jP_j=I$다. $\sigma^k$의 계수들을 더하면 $k=0$에서는 $1$, $1\le k<p$에서는 $0$이 된다. 후자는 $z^p=1$, $z\ne1$일 때 geometric sum $1+z+\cdots+z^{p-1}$이 $0$이기 때문이다. 이 합에 $z-1$을 곱하면 $z^p-1=0$이 된다. 따라서 모든 vector를 $v=\sum_jP_jv$처럼 eigenvector들의 합으로 표현할 수 있다. 각 $P_j$는 이 성분들에 identity 또는 zero로 작용하므로 $P_j^2=P_j$도 성립한다. 서로 다른 eigenvalue의 eigenspace들은 independent하므로 각 eigenspace의 basis를 모으면 eigenbasis가 된다. 이 유한한 합으로 diagonalization이 성립함을 직접 확인할 수 있다.

</details>

## 12. 2차·3차·4차방정식에는 왜 공식이 있을까?

두 label만 서로 바꾸고 나머지는 그대로 두는 permutation을 **transposition**이라고 한다. $(12)$라면 $1$과 $2$만 바꾼다. 모든 permutation은 이런 swap들을 합성해서 만들 수 있다. 예를 들어 $(123)=(12)(23)$에서 오른쪽 swap부터 적용하면 $1\to2$, $2\to3$, $3\to1$이 된다. Swap을 짝수 번 쓰면 **even**, 홀수 번 쓰면 **odd**라고 한다. 같은 swap을 두 번 끼워 넣을 수 있으므로 사용 횟수 자체는 하나로 정해지지 않는다. 하지만 짝수인지 홀수인지는 바뀌지 않는다.

이유는 determinant로 확인할 수 있다. 각 label에 basis vector $e_1,\ldots,e_n$을 하나씩 대응시키자. **Permutation matrix** $P_\pi$는 $e_i$를 $e_{\pi(i)}$로 보내는 matrix다. 각 column에는 같은 basis vector들이 순서만 바뀌어 들어간다. $(12)$에서는 $e_1,e_2$에 해당하는 두 column을 바꾸므로 determinant가 $-1$이다. $(123)$의 처음 세 column은 $e_2,e_3,e_1$이며, 두 번 바꿔 얻을 수 있으므로 determinant는 $+1$이다. 이 vector들은 label을 표시하기 위한 것이다. 근의 실제 값이나 complex plane에서의 좌표가 아니다.

Matrix의 곱은 permutation의 합성에 대응하고, 곱의 determinant는 각각의 determinant를 곱한 값이다. 따라서 같은 permutation을 swap $k$개의 곱으로 어떻게 쓰더라도 $\det P_\pi=(-1)^k$는 같다. 이 **sign**으로 even과 odd를 구분한다. Even permutation들을 모으면 **alternating group** $A_n$이 된다. Identity와 even permutation들의 곱·inverse는 모두 determinant가 $+1$이기 때문이다. 9절의 용어로는 sign map의 kernel, 즉 이 map이 $+1$로 보내는 permutation들의 집합이다.

$n\ge2$일 때 transposition 하나를 정해서 곱하면, even permutation 하나마다 odd permutation 하나가 짝지어진다. 따라서 $n!$개의 permutation 중 정확히 절반이 even이다. $A_3$의 원소는 $3$개, $A_4$는 $12$개, $A_5$는 $60$개다. Derived series에 등장하는 원소 수가 여기서 나온다.

$n\le4$에서는 derived series가 identity에 도달한다.

$$
\begin{aligned}
S_2&\supset\{e\},\\
S_3&\supset A_3\supset\{e\},\\
S_4&\supset A_4\supset V_4\supset\{e\}.
\end{aligned}
\tag{46}
$$

따라서 이 $S_n$들의 모든 subgroup도 solvable하다. Subgroup에서 commutator를 구하면 그 결과는 원래 $S_n$의 해당 derived subgroup에 포함되기 때문이다.

여기서 $S_4$ 안의 $V_4$는 double transposition으로 이루어진 $\lbrace e,(12)(34),(13)(24),(14)(23)\rbrace $다. Identity를 제외하면 각각 서로 겹치지 않는 swap 두 개를 수행한다. 예를 들어 $(12)(34)$는 $1,2$를 바꾸면서 $3,4$도 바꾼다. 앞에서 본 독립적인 부호 변화와 곱셈 규칙은 같은 Klein four-group이지만, 여기서는 네 label에 작용한다. 네 permutation은 모두 even이다.

$S_4$ chain의 quotient order는 차례로 $24/12=2$, $12/4=3$, $4/1=4$다. 마지막 $V_4$는 abelian이지만 cyclic은 아니다. Identity가 아닌 원소 하나를 반복해도 그 원소와 $e$만 얻기 때문이다. 그 사이에 subgroup $\lbrace e,(12)(34)\rbrace$를 넣으면 마지막 부분을 order가 $2$인 quotient 두 개로 나눌 수 있다. 결국 order가 $2$와 $3$인 cyclic 단계들만 있으면 된다. Radical–solvability 대응에 따르면 각각 square root와 cube root에 해당하므로 quartic 공식에 새로운 종류의 radical은 필요하지 않다.

### Quartic의 pairing을 추적하는 cubic resolvent
{: id="세-가지-pairing이-quartic의-cubic-단계를-설명한다" }

네 근의 label을 순서 없는 두 쌍으로 묶는 방법은 세 가지다.

$$
\begin{gathered}
12\mid34,\\
13\mid24,\\
14\mid23.
\end{gathered}
\tag{47}
$$

예를 들어 $(12)$는 $12\mid34$를 그대로 두고 $13\mid24$와 $14\mid23$을 바꾼다. 네 label을 permutation하면 세 pairing도 함께 바뀌는 것이다. 이 대응으로 homomorphism $S_4\to S_3$를 얻는다. 세 pairing을 모두 고정하는 kernel이 바로 double-transposition subgroup $V_4$다.

**Resolvent**는 원래 근들을 조합한 값을 근으로 갖는 보조 polynomial이다. Quartic의 근을 $a,b,c,d$라 하면, 세 pairing에 각각 $ab+cd$, $ac+bd$, $ad+bc$를 대응시킨다. 세 값을 더하면 $ab+ac+ad+bc+bd+cd$다. 이는 $(x-a)(x-b)(x-c)(x-d)$를 전개했을 때의 $x^2$ 계수와 같다. 두 개씩 곱한 값의 합과 세 값을 모두 곱한 값도 원래 quartic의 계수로 다시 쓸 수 있으며, 아래 계산에 그 식을 적었다. 합과 전체 곱에는 마이너스 부호를 붙이고, 두 개씩 곱한 값의 합은 그대로 쓰면 monic cubic의 나머지 세 계수를 얻는다. 그 cubic의 근이 바로 세 pairing 값이다. 네 근을 미리 알아야 resolvent를 만들 수 있는 것은 아니다.

이 cubic을 풀면 세 pairing 값을 얻고, 남은 quadratic 단계들에서 각각의 근을 구한다. 네 근 자체를 바로 구하는 대신, 근들을 두 쌍으로 묶었을 때의 값을 먼저 구하는 것이다. Quartic 공식에 cubic을 푸는 과정이 포함되는 이유다. 특별한 계수에서는 pairing 값이 겹칠 수 있으므로, 언제나 서로 다르다고 가정해서는 안 된다.

<details markdown="1">
<summary>앞서 본 quartic의 resolvent 직접 계산하기</summary>

$a=\sqrt2$, $b=-\sqrt2$, $c=\sqrt3$, $d=-\sqrt3$로 두자. 세 pairing에 다음 값을 대응시킬 수 있다.

- $12\mid34$: $y_1=ab+cd=-5$.
- $13\mid24$: $y_2=ac+bd=2\sqrt6$.
- $14\mid23$: $y_3=ad+bc=-2\sqrt6$.

따라서 이 값들을 근으로 갖는 cubic은 $(t+5)(t^2-24)=0$이다. 전개하면 $t^3+5t^2-24t-120=0$이 된다.

예를 들어 $t^2$의 계수는 $-(y_1+y_2+y_3)=5$, $t$의 계수는 $y_1y_2+y_1y_3+y_2y_3=-24$, constant coefficient는 $-y_1y_2y_3=-120$이다. 세 linear factor를 곱해서 얻는 익숙한 계수 공식이다.

더 구체적으로 monic quartic을 $x^4+A x^3+B x^2+C x+D$라고 쓰자. Pairing 식들을 직접 전개하면 다음 관계를 얻는다.

- $y_1+y_2+y_3=B$.
- $y_1y_2+y_1y_3+y_2y_3=A C-4D$.
- $y_1y_2y_3=A^2D+C^2-4BD$.

따라서 근을 몰라도 $A,B,C,D$에서 resolvent의 세 계수를 모두 구할 수 있다. 위의 예에서 $A=C=0$, $B=-5$, $D=6$을 넣으면 앞서 계산한 계수들이 그대로 나온다. 이 예에서는 원래 quartic의 Galois group이 $S_4$보다 작아서 resolvent도 linear factor와 quadratic factor로 나뉜다. 세 pairing을 쓰는 방법이 적용된다는 것이지, 모든 quartic의 Galois group이 $S_4$라는 뜻은 아니다.

</details>

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-07-derived-series.svg" alt="Derived step마다 group order를 표시한 log scale 그래프. S3와 S4는 1에 도달하고 S5는 60에서 더 줄어들지 않는다." width="960" height="520">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-07-derived-series-mobile.svg" alt="Derived step마다 group order를 표시한 log scale 그래프. S3와 S4는 1에 도달하고 S5는 60에서 더 줄어들지 않는다. 모바일 배치." width="480" height="600">
</div>
*Figure 7. 정수 단계 $j$마다 현재 group의 commutator들이 생성하는 subgroup으로 바꾸고, 남은 원소 수를 점으로 표시했다. 세로축은 log scale이므로 같은 높이 차이는 같은 차이가 아니라 같은 비율을 나타낸다. $S_3$와 $S_4$는 identity만 있는 order $1$에 도달하지만, $S_5$는 원소가 $60$개인 $A_5$에서 더 줄어들지 않는다.*

## 13. $S_5$의 derived series가 identity에 도달하지 못하는 이유

결정적인 차이는 다음 계산에 있다.

$$
S_5'=A_5,\qquad A_5'=A_5.
\tag{48}
$$

따라서 derived series는 identity에 도달하지 않는다.

$$
\begin{gathered}
S_5\supset A_5=A_5'=\cdots,\\
120\longrightarrow60\longrightarrow60\longrightarrow\cdots.
\end{gathered}
\tag{49}
$$

$S_3$에서는 commutator들이 세 rotation으로 이루어진 abelian group을 생성했고, 그다음 단계에서 identity만 남았다. 반면 $S_5$에서 남은 $60$개의 symmetry로 다시 commutator를 구하면 그 group 전체가 생성된다. 단순히 원소 수가 $120$으로 많아서 생기는 문제가 아니다.

구체적으로 $g=(12)(34)$, $h=(12)(35)$를 보자. 둘 다 even이지만 $[g,h]=(345)$라는 3-cycle을 만든다. $(345)$는 $3\to4\to5\to3$으로 보내고, 네 연산을 모두 마친 뒤에는 $1,2$를 그대로 둔다는 뜻이다. 이 계산의 label을 even permutation으로 바꾸면 모든 3-cycle을 얻는다. 그 3-cycle들을 합성하면 모든 even permutation을 만들 수 있다. 이를 $A_5$ 전체를 **생성한다(generate)**고 한다. 따라서 $A_5'=A_5$다. Identity가 아닌 commutator 하나를 찾는 것보다 강한 결론이다. Commute하지 않을 뿐 아니라, derived 연산을 반복해도 $60$개 원소가 모두 다시 생성된다.

Derived series가 identity에 도달하지 않으므로 다른 subgroup chain을 골라도 abelian quotient들만으로 identity에 이를 수 없다. 따라서 이 group은 solvable하지 않다.

<details markdown="1">
<summary>3-cycle들이 A5 전체를 다시 생성하는 이유</summary>

$S_5$의 commutator는 항상 even이다. $g,h,g^{-1},h^{-1}$의 sign을 모두 곱하면 $1$이기 때문이다. 따라서 $S_5'\subseteq A_5$다. 반대로 label 하나가 겹치는 두 transposition의 commutator는 3-cycle이고, 이를 conjugation하면 모든 3-cycle을 얻는다. 3-cycle들이 모든 even permutation을 생성하므로 $S_5'=A_5$다.

Even permutation을 짝수 개의 transposition의 곱으로 쓰고 두 개씩 묶어 보자. 같은 transposition 두 개는 상쇄되고, label 하나가 겹치는 두 개의 곱은 3-cycle이다. 서로 겹치지 않는 두 개도 다음처럼 쓸 수 있다.

$$
(ab)(cd)=(acb)(acd),
\tag{50}
$$

합성은 오른쪽부터 적용한다. 따라서 모든 쌍을 3-cycle들의 곱으로 바꿀 수 있다.

이제 $A_5$ 안에서만 계산하자. 다음 두 permutation은 모두 even이다.

$$
\begin{gathered}
g=(12)(34),\\
h=(12)(35).
\end{gathered}
\tag{51}
$$

직접 합성하면

$$
[g,h]=(345).
\tag{52}
$$

이다. Derived subgroup은 normal하므로, $A_5'$의 permutation에 $A_5$의 원소로 label을 다시 붙여도 $A_5'$ 안에 남는다. 즉, 모든 $k\in A_5$에 대해 $k(345)k^{-1}=(k(3)\ k(4)\ k(5))$를 포함한다. 이것이 $(345)$의 $A_5$-conjugate다. 여기서 conjugation은 cycle의 세 label을 일관되게 바꾸는 연산이다.

원하는 3-cycle을 만들도록 $3,4,5$의 image를 정해 permutation을 하나 고르자. 이것이 odd이면 오른쪽에 $(12)$를 합성한다. 원래 사용하지 않았던 두 label만 바꾸므로 $3,4,5$의 image는 그대로이고 parity만 even으로 바뀐다. 달리 말하면 $(12)$가 $(345)$와 commute하므로 conjugation 결과는 바뀌지 않는다. 따라서 모든 3-cycle을 even permutation으로 label을 바꿔 얻을 수 있다.

따라서 $A_5'$은 모든 3-cycle을 포함하고, 이들이 생성하는 $A_5$ 전체도 포함한다. 이것으로 (48)이 증명된다. 더 강한 group-theory 정리를 따로 가정할 필요는 없다.

</details>

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-08-a5.svg" alt="다섯 label이 commutator의 네 연산을 거치는 경로. 1과 2도 중간 단계에서는 서로 바뀐다." width="960" height="560">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-08-a5-mobile.svg" alt="다섯 label이 commutator의 네 연산을 거치는 경로. 1과 2도 중간 단계에서는 서로 바뀐다. 모바일 배치." width="480" height="910">
</div>
*Figure 8. 다섯 label에 $h^{-1},g^{-1},h,g$를 차례로 적용한 경로다. PC에서는 왼쪽에서 오른쪽으로, 모바일에서는 위에서 아래로 읽는다. 회색 $1,2$도 중간에는 움직이지만 네 연산을 모두 마치면 원래 위치로 돌아온다. 색칠한 세 label은 $(345)$를 이룬다. 이 cycle과 그 conjugate들이 $A_5$ 전체를 생성하므로 $A_5'=A_5$다.*

### Group 계산을 코드로 구현하기

**Generator**는 합성에 사용할 기본 변환이고, **generated subgroup**은 identity와 generator·inverse를 반복해서 합성해 얻는 모든 변환의 집합이다. 예를 들어 3-cycle $r=(123)$ 하나만 generator로 주어도 생성되는 subgroup은 $\lbrace e,r,r^2\rbrace$다. 세 번 적용하면 $e$로 돌아온다. 여기에 swap $(12)$까지 generator로 추가하면 $S_3$의 여섯 permutation을 모두 만들 수 있다.

Derived subgroup을 계산할 때는 commutator들이 다음 subgroup의 generator가 된다. 개별 commutator의 목록에서 멈추지 말고, 이들을 합성해 얻는 변환까지 모두 포함해야 한다. 더는 새로운 permutation이 나오지 않을 때까지 합성 결과를 추가하는 계산이 **closure**다.

<details markdown="1">
<summary>Python으로 subgroup을 생성하고 derived series 계산하기</summary>

Permutation은 각 label의 image를 나열한 tuple로 표현한다. 예를 들어 `(1, 0, 2)`는 label $0$과 $1$을 바꾼다. Index `i`에 있는 값이 `i`가 이동할 label이므로, `(1, 2, 0)`은 $0\to1\to2\to0$으로 보낸다. 본문의 cycle notation은 label이 $1$부터 시작하지만 코드의 index는 $0$부터 시작한다. `compose(p,q)`는 `q`를 먼저 적용한 뒤 `p`를 적용한다.

`group_closure`에서 `found`는 지금까지 찾은 permutation, `pending`은 아직 합성 결과를 조사하지 않은 permutation이다. Generator와 그 inverse를 적용하며 `pending`이 빌 때까지 새 permutation을 찾는다. 출발점에서 도달할 수 있는 vertex를 모두 찾는 graph search와 같은 방식이다. `derived_subgroup`은 먼저 commutator들을 모은 뒤 이 closure 계산을 실행한다.

```python
from itertools import permutations

def compose(p, q):
    """Apply q first, then p."""
    return tuple(p[q[i]] for i in range(len(p)))

def inverse(p):
    result = [0] * len(p)
    for i, image in enumerate(p):
        result[image] = i
    return tuple(result)

def group_closure(generators, n):
    """Generate the subgroup, rather than just listing its generators."""
    identity = tuple(range(n))
    generators = set(map(tuple, generators))
    generators |= {inverse(g) for g in generators}
    found = {identity}
    pending = [identity]
    while pending:
        current = pending.pop()
        for generator in generators:
            candidate = compose(current, generator)
            if candidate not in found:
                found.add(candidate)
                pending.append(candidate)
    return found

def commutator(g, h):
    return compose(compose(compose(g, h), inverse(g)), inverse(h))

def derived_subgroup(group):
    # Products of commutators must be included, not just the commutators.
    group = set(group)
    n = len(next(iter(group)))
    generators = {commutator(g, h) for g in group for h in group}
    return group_closure(generators, n)

def derived_orders(group):
    group = set(group)
    orders = [len(group)]
    while len(group) > 1:
        next_group = derived_subgroup(group)
        orders.append(len(next_group))
        if next_group == group:  # A nontrivial fixed point is an obstruction.
            break
        group = next_group
    return orders

for n, expected in [(3, [6, 3, 1]), (4, [24, 12, 4, 1]),
                    (5, [120, 60, 60])]:
    actual = derived_orders(permutations(range(n)))
    assert actual == expected
    print(f"S_{n}: {actual}")
```

출력되는 order는 $S_3$에서 $6,3,1$, $S_4$에서 $24,12,4,1$, $S_5$에서 $120,60,60$이다. 이 코드는 해당 finite group의 원소를 모두 열거해 앞의 증명을 계산으로 확인한다. 모든 polynomial을 조사하거나 Galois's criterion 자체를 증명하는 것은 아니다.

</details>

{% include interactive/article-lab.html id="galois-derived" %}

$C_5$, $S_3$, $S_4$, $A_5$, $S_5$를 비교해 보자. 실행 전에 identity에 도달할지 예상해 보면 좋다. $C_5$는 abelian이고, $S_3$와 $S_4$는 nonabelian이지만 solvable하며, $A_5$와 $S_5$는 nonsolvable하다. 다음 subgroup은 언제나 이전 group에 포함되므로, 같은 order가 반복되면 실제 subgroup도 같다는 뜻이다. Order가 $1$에 도달하면 solvable하고, $60$에서 멈추면 그렇지 않다.

## 14. Galois group이 S₅인 5차방정식
{: id="14-galois-group이-s_5인-5차방정식-하나" }

$S_5$가 nonsolvable이라는 사실만으로는 5차방정식의 근의 공식이 없다고 결론 내릴 수 없다. Galois group이 실제로 $S_5$인 5차방정식도 있어야 한다. 다음 polynomial이 그 예다.

$$
f(x)=x^5-5x+1.
\tag{53}
$$

계수는 모두 유리수이고 Galois group은 $S_5$ 전체다. Irreducibility와 nonreal root의 개수만으로 이를 증명할 수 있다. 근삿값은 필요하지 않다.

### 14.1 변수 치환으로 Eisenstein's criterion 적용하기
{: id="141-irreducibility-변수를-이동하면-eisensteins-criterion이-보인다" }

**$\mathbb Q$ 위에서 irreducible하다**는 것은 유리계수 polynomial 중 차수가 $1$ 이상인 두 개의 곱으로 나눌 수 없다는 뜻이다. 예를 들어 $x^2-2$는 $\mathbb Q$ 위에서 irreducible하지만, $\mathbb R$ 위에서는 $(x-\sqrt2)(x+\sqrt2)$로 나뉜다. 어떤 수를 계수로 허용하는지가 중요하다. 앞서 본 quartic $(x^2-2)(x^2-3)$는 이미 유리계수 factor 두 개의 곱으로 주어졌으므로 $\mathbb Q$ 위에서 reducible하다.

Quintic에 rational root가 없다는 사실만으로는 충분하지 않다. Linear factor는 배제할 수 있어도 quadratic과 cubic의 곱일 가능성은 남기 때문이다. **Eisenstein's criterion**은 가능한 차수의 factorization을 한꺼번에 배제하는 방법이다. 정수 계수를 갖는 polynomial과 소수 $p$에 대해, leading coefficient는 $p$의 배수가 아니고 나머지 계수는 모두 $p$의 배수이되 constant coefficient는 $p^2$의 배수가 아니면 된다. 이 조건을 만족하면 $\mathbb Q$ 위에서 irreducible하다.

원래 $f$의 계수에는 이 조건이 바로 보이지 않는다. $x-1$을 대입하면

$$
\begin{aligned}
f(x-1)&=x^5-5x^4+10x^3\\
      &\quad-10x^2+5.
\end{aligned}
\tag{54}
$$

이다. 높은 차수부터 계수를 나열하면 $(1,-5,10,-10,0,5)$다. Leading coefficient는 $1$이고, 나머지는 $0$을 포함해 모두 $5$의 배수다. Constant coefficient인 $5$는 $25$의 배수가 아니다. 따라서 소수 $5$에 대해 **Eisenstein's criterion**을 만족하며, $f(x-1)$은 $\mathbb Q$ 위에서 irreducible하다.

이 변수 치환은 되돌릴 수 있다. $f(x)$가 두 factor의 곱으로 나뉜다면, 각 factor에 $x-1$을 대입해 $f(x-1)$도 같은 방식으로 나눌 수 있다. 반대 방향은 $x+1$을 대입하면 된다. 따라서 $f$ 역시 $\mathbb Q$ 위에서 irreducible하다.

<details markdown="1">
<summary>Modulo 5 계산으로 Eisenstein's criterion 증명하기</summary>

**Modulo $5$** 계산은 수를 $5$로 나눈 나머지만 남기고, 이를 $0,1,2,3,4$ 중 하나로 나타내는 계산이다. 예를 들어 $7$은 $2$, $-1$은 $4$, $-5$는 $0$이 된다. 따라서 $3+4$의 결과는 $2$, $2\cdot3$의 결과는 $1$이다. $7\equiv2\pmod5$는 두 수의 나머지가 같다는 표기다. 이 다섯 나머지가 이루는 finite field가 $\mathbb F_5$다. $1,2,3,4$의 inverse가 각각 $1,3,2,4$이므로 $0$이 아닌 값으로 나누는 것도 가능하다.

Integer polynomial을 modulo $5$로 줄인다는 것은 계수를 각각 나머지로 바꾼다는 뜻이다. 덧셈과 곱셈을 해도 등식이 유지되므로, 원래 $A B=F$였다면 계수를 바꾼 뒤에도 같은 등식이 성립한다. (54)에서는 $x^5$만 남는다. 아래에서 polynomial 위의 bar는 계수를 나머지로 바꾸었다는 뜻이며, complex conjugation이 아니다.

Leading coefficient가 $1$인 polynomial을 **monic**이라고 한다. (54)가 유리계수 polynomial 두 개의 곱으로 나뉜다고 가정하자. **Gauss's lemma**에 따르면 양의 차수를 갖는 monic integer polynomial $A(x),B(x)$의 곱으로도 쓸 수 있다. 두 factor의 leading coefficient는 modulo $5$에서도 $1$이므로 차수가 줄어들지 않는다. 양쪽의 계수를 modulo $5$로 계산하면

$$
\overline A(x)\,\overline B(x)=x^5.
\tag{55}
$$

이다. 두 factor는 각각 $x$의 양의 정수 거듭제곱이어야 한다. 곱의 constant coefficient가 $0$이므로 적어도 한 factor의 constant coefficient가 $0$이다. 그 factor에서 $x$를 하나 묶어 낸 뒤 약분하자. 같은 과정을 곱이 $1$이 될 때까지 반복하면, 다섯 개의 $x$가 두 monic factor에 나뉘고 다른 factor는 남지 않는다. 두 factor의 차수가 양수이므로 각각 적어도 하나의 $x$를 포함한다.

따라서 원래 두 integer polynomial의 constant coefficient는 모두 $5$의 배수다. 그 곱은 $25$의 배수여야 하는데, (54)의 constant coefficient는 $5$이므로 모순이다.

이 계산은 원래 문제의 base field를 $\mathbb F_5$로 바꾼 것이 아니다. $\mathbb Q$ 위에서 factorization이 가능하다면, 그로부터 얻은 integer polynomial의 등식은 modulo $5$에서도 성립해야 한다. 하지만 (55)와 constant coefficient를 비교하면 모순이 생긴다.

</details>

<details markdown="1">
<summary>Gauss's lemma: 유리계수 factor를 정수 계수로 바꾸는 원리</summary>

곱의 계수가 정수라고 해서 처음부터 factor의 계수도 정수일 필요는 없다. 예를 들어 $(x/2+1/2)(2x-2)=x^2-1$이다. 숫자 부분을 분리하면 $(1/2)(x+1)\cdot2(x-1)$이고, $1/2$와 $2$가 상쇄되어 정수 계수인 $(x+1)(x-1)$이 남는다. Gauss's lemma는 primitive integer polynomial에서 항상 이런 식으로 숫자 부분을 정리해 정수 계수 factor를 얻을 수 있다는 정리다.

Integer polynomial의 계수들의 양의 최대공약수를 **content**라고 한다. 예를 들어 $2x+2$의 content는 $2$이고, $x+1$과 $2x+1$의 content는 각각 $1$이다. Content가 $1$이면 **primitive**라고 한다. Primitive root of unity의 “primitive”와는 다른 뜻이다. Content로 나누면 정수 계수 polynomial을 숫자 부분과 primitive polynomial로 분리할 수 있다.

Primitive integer polynomial 두 개를 곱한 결과도 primitive다. 그렇지 않다면 어떤 소수 $p$가 곱의 모든 계수를 나눈다. 계수를 modulo $p$로 바꾸면 $\mathbb F_p$ 위에서 $0$이 아닌 두 polynomial의 곱이 $0$이 된다. 하지만 곱의 leading coefficient는 $0$이 아닌 두 leading coefficient의 곱이므로 $0$일 수 없다. 여기서는 $p$가 소수라는 조건이 중요하다. 소수가 아닌 수로 나눈 나머지들은 field를 이루지 않을 수도 있다.

이제 primitive integer polynomial $F$의 유리계수 factorization에서 분모를 없애고, 얻은 정수 계수 factor들의 content를 분리하자. 그러면 $F=r A B$로 쓸 수 있다. $A,B$는 primitive integer polynomial이고 $r$은 유리수다. $A B$도 primitive다. $r=u/v$를 기약분수로 쓰고 $v>0$으로 두자. $F$의 계수가 모두 정수이므로 $v$는 $A B$의 모든 계수를 나눠야 하고, 따라서 $v=1$이다. $F$ 역시 primitive이므로 $\lvert u\rvert=1$이다. 결국 $F=\pm A B$라는 정수 계수 factorization을 얻는다. $F$가 monic이면 두 factor의 정수 leading coefficient의 곱은 $1$이므로, 부호를 조정해 둘 다 monic으로 만들 수 있다. 이것이 (55)에서 사용한 Gauss's lemma다.

</details>

### 14.2 실근 세 개와 nonreal conjugate pair 하나
{: id="142-실근-세-개와-실수가-아닌-conjugate-pair-하나" }

실근이 몇 개인지 알면 complex conjugation이 다섯 label에 어떻게 작용하는지도 알 수 있다. 근의 정확한 식은 필요하지 않고, 증가와 감소가 바뀌는 turning point만 찾으면 된다. Derivative는

$$
f'(x)=5(x^4-1).
\tag{56}
$$

이다. $x<-1$에서는 양수, $-1<x<1$에서는 음수, $x>1$에서는 양수다. 따라서 함수는 첫 구간에서 증가하고, 가운데 구간에서 감소한 뒤, 마지막 구간에서 다시 증가한다. 두 turning point에서의 값은

$$
f(-1)=5,\qquad f(1)=-3.
\tag{57}
$$

이다. Leading term이 $x^5$이므로 왼쪽으로 갈수록 함수값은 $-\infty$로, 오른쪽으로 갈수록 $+\infty$로 간다. 함수는 연속이고 세 구간 각각에서 한 방향으로만 변한다. 따라서 $(-\infty,-1)$, $(-1,1)$, $(1,\infty)$에서 각각 정확히 한 번씩 $0$을 지난다.

Complex conjugation은 $a+bi$를 $a-bi$로 보낸다. 예를 들어 $2+3i$와 $2-3i$를 서로 바꾸고, 실수는 모두 그대로 둔다. $f$의 계수가 실수이므로 $f(\overline z)=\overline{f(z)}$다. 따라서 실수가 아닌 근이 있으면 그 conjugate도 근이다.

Separability에 의해 repeated root가 없으므로, 나머지 두 근은 하나의 nonreal conjugate pair를 이룬다. Complex conjugation은 세 실근을 그대로 두고 이 두 근만 서로 바꾼다. 덧셈과 곱셈을 보존하고, $\mathbb Q$를 고정하며, splitting field를 같은 field로 보내므로 $\mathbb Q$-automorphism이다. 근들에 대한 작용은 transposition 하나다. 처음부터 특별한 label을 골라야 하는 것은 아니다. 실수가 아닌 두 근에 $4,5$라는 label을 붙이면 이 automorphism이 $(45)$로 나타난다.

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-09-quintic.svg" alt="Quintic의 실제 곡선과 complex roots. Conjugation은 실수가 아닌 한 쌍만 바꾼다." width="960" height="570">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-09-quintic-mobile.svg" alt="Quintic의 실제 곡선과 complex roots. Conjugation은 실수가 아닌 한 쌍만 바꾼다. 모바일 배치." width="480" height="940">
</div>
*Figure 9. 함수 그래프에는 세 교점과 critical value $f(-1)=5$, $f(1)=-3$을, complex plane에는 다섯 근을 모두 표시했다. Conjugation은 세 실근을 그대로 두고 $z,\bar z$만 서로 바꾸므로 transposition 하나로 나타난다. 실근이 정확히 세 개라는 사실은 derivative의 부호로 확인한다.*

### 14.3 Transitivity와 transposition이 S₅를 결정한다
{: id="143-irreducible-polynomial의-근에는-group이-transitively-작용한다" }

한 근에 가능한 automorphism을 모두 적용했을 때 얻는 근들의 집합을 **orbit**이라고 한다. 6절의 quartic $(x^2-2)(x^2-3)$에서 $\sqrt2$의 orbit은 네 근 전체가 아니라 $\lbrace \sqrt2,-\sqrt2\rbrace $다. 제곱이 $2$인 수를 제곱이 $3$인 수로 보낼 수는 없기 때문이다. 나머지 두 근은 다른 orbit인 $\lbrace \sqrt3,-\sqrt3\rbrace $를 이룬다.

근 전체가 하나의 orbit이면 group의 작용이 **transitive**하다고 한다. 어느 근을 골라도 적절한 automorphism으로 다른 모든 근에 대응시킬 수 있다는 뜻이다. $\mathbb Q$ 위의 irreducible polynomial은 이 조건을 만족한다. Orbit이 둘 이상이면 각각에서 $\mathbb Q$ 위의 factor를 얻을 수 있어 irreducibility와 모순이기 때문이다. 앞의 quartic에서는 두 orbit이 각각 $x^2-2$, $x^2-3$에 해당한다.

<details markdown="1">
<summary>한 orbit의 근들로 base field 위의 factor를 만드는 방법</summary>

Characteristic이 $0$인 field $F$ 위의 irreducible polynomial을 생각하자. 그 splitting field를 $L$, Galois group을 $G=\operatorname{Gal}(L/F)$라고 하자. Orbit $O$에 속한 근들을 모아 $q_O(x)=\prod_{\beta\in O}(x-\beta)$를 만든다.

$G$의 모든 원소는 $O$ 안의 근들을 서로 바꾼다. 곱하는 순서만 달라지므로 $q_O$는 변하지 않는다. 따라서 $q_O$의 계수는 $G$의 모든 원소가 고정한다. 7절의 fixed-field theorem에서 $L^G=F$이므로, 이 계수들은 $L$에 속할 뿐 아니라 $F$에도 속한다.

$q_O$의 차수는 양수이고, $L[x]$에서 원래 polynomial의 factor다. 두 polynomial의 계수가 모두 $F$에 있으므로 $F[x]$에서 monic polynomial $q_O$로 나눠도 나머지는 똑같이 $0$이다. 원래 polynomial이 irreducible하므로 차수가 더 낮은 nonconstant factor는 있을 수 없다. 따라서 $O$는 모든 근을 포함하고, group의 작용은 transitive하다.

</details>

따라서 이 방정식의 group $G\le S_5$는 transitive하고 transposition을 포함한다. 각각의 조건만으로는 $G=S_5$라고 할 수 없다. Five-cycle 하나만 반복해도 어느 label이든 다른 모든 label로 보낼 수 있지만, 생성되는 permutation은 다섯 개뿐이다. Swap 하나만으로는 두 개뿐이다. 하지만 label의 개수가 소수일 때 이 두 조건을 함께 만족하면 full symmetric group을 얻는다.

이미 알고 있는 swap이 $(45)$라고 하자. $g$가 $4$를 $1$로, $5$를 $3$으로 보낸다면 $g(45)g^{-1}=(13)$도 가능한 swap이다. 근의 label마다 점을 하나씩 그리고 이를 **vertex**라고 부르자. 이렇게 얻는 swap의 두 label을 선으로 연결하고, 그 선을 **edge**라고 한다. 이 graph의 vertex는 group의 원소가 아니라 **근의 label**이고, edge는 가능한 **두 label 사이의 swap**이다.

Edge를 연달아 따라가는 경로를 **path**라고 한다. 이런 경로로 서로 도달할 수 있는 vertex를 빠짐없이 모은 묶음이 **connected component**다. 모든 vertex가 한 묶음에 속하면 graph가 **connected**하다고 한다.

$G$의 원소는 모든 vertex와 edge의 label을 일관되게 바꾸므로, connected component 하나를 다른 component 하나에 대응시킨다. Transitive하다는 것은 어느 component의 vertex든 다른 component의 vertex로 보낼 수 있다는 뜻이므로 모든 component의 크기는 같다. $5$는 소수이므로 크기 $5$인 component 하나이거나 크기 $1$인 component 다섯 개여야 한다. 실제 edge가 있으므로 두 번째 경우는 불가능하다. Vertex가 $6$개라면 크기 $2$나 $3$인 component들로 나뉠 수도 있다. 바로 이 점에서 소수 차수라는 조건이 쓰인다.

Graph가 connected하면 edge에 해당하는 swap들을 합성해서 임의의 두 vertex를 바꿀 수 있다. 예를 들어 path $1$–$2$–$3$에서는 $(12)(23)(12)=(13)$이다. 오른쪽부터 적용하면 양 끝점만 바뀌고 가운데 vertex는 그대로 남는다. 긴 path에서도 같은 방법을 반복하면 된다. 이렇게 모든 transposition을 만들 수 있으므로 모든 permutation을 만들 수 있다.

<details markdown="1">
<summary>Conjugate로 만든 graph가 $S_5$를 생성하는 이유</summary>

이미 포함된 transposition을 $(a\,b)$라고 하고, 다섯 근의 label을 vertex로 하는 graph를 만들자. 각 $g\in G$에 대해 다음 edge를 그린다.

$$
\{g(a),g(b)\}.
\tag{58}
$$

이 edge의 두 vertex를 바꾸는 변환은 $g(a\,b)g^{-1}$이므로 $G$에 속한다. $h\in G$를 적용하면 $g$로 만든 edge가 $hg$로 만든 edge로 바뀐다. 따라서 graph는 보존되고, 하나의 connected component는 다른 component에 일대일로 대응한다. Transitivity에 의해 어떤 component의 vertex든 다른 component의 vertex로 보낼 수 있으므로 모든 component의 크기는 같다. 그 크기는 $5$를 나누어야 한다. Edge가 있으므로 component 크기가 $1$일 수는 없고, $5$가 소수이므로 전체 graph가 connected하다.

Connected graph의 edge transposition들로 모든 permutation을 만들 수 있다. Path의 한 부분에서 swap $(u\,v)$를 이미 만들었고 다음 edge가 $(v\,w)$라고 하자. $u,v,w$가 서로 다르면 $(u\,v)(v\,w)(u\,v)=(u\,w)$이므로, 한 vertex 더 떨어진 swap도 만들 수 있다. 첫 edge부터 이 과정을 반복하면 simple path의 두 끝점을 바꾸는 swap을 얻는다. 모든 transposition으로 $S_5$를 생성할 수 있으므로 $G$는 $S_5$ 전체를 포함하고, 따라서 둘은 같다. 이 논증에서는 $5$가 소수라는 조건이 중요하다.

</details>

세 단계를 종합하면

$$
\operatorname{Gal}(f/\mathbb Q)=S_5.
\tag{59}
$$

이다. $\operatorname{Gal}(f/\mathbb Q)$는 $f$의 $\mathbb Q$ 위의 splitting field의 Galois group을 뜻한다. [Milne, §4, prime-degree criterion](https://www.jmilne.org/math/CourseNotes/FT.pdf)

## 15. 모든 5차방정식을 radical로 푸는 공식은 없다
{: id="15-결론-모든-5차방정식에-적용되는-radical-공식은-없다" }

이 방정식에 대해 증명한 내용을 정리하면 다음과 같다.

$$
\begin{gathered}
f(x)=x^5-5x+1,\\
\text{irreducible,}\\
\text{exactly two nonreal roots}\\
\Longrightarrow
\operatorname{Gal}(f/\mathbb Q)=S_5,\\
S_5'=A_5,\qquad A_5'=A_5,\\
\Longrightarrow
f\text{ not solvable}\\
\text{by radicals}.
\end{gathered}
\tag{60}
$$

유리계수를 갖는 **모든** 5차방정식에 적용되는 radical 공식이 있다면 이 방정식에도 적용되어야 한다. 하지만 이 방정식은 radical로 풀리지 않는다. 따라서 그런 일반적인 근의 공식은 존재할 수 없다.

중간에 complex radical을 사용해도 결론은 달라지지 않는다. 처음부터 허용한 연산이기 때문이다. Nested radical을 더 많이 사용해도 마찬가지다. 유한 번 사용한다면 radical tower로 나타낼 수 있으므로 같은 solvability 조건을 만족해야 한다.

<details markdown="1">
<summary>이 방정식은 근 하나도 radical로 나타낼 수 없다</summary>

이 irreducible polynomial은 근 하나도 $\mathbb Q$의 radical extension에 들어가지 않는다. 근 하나가 들어간다고 가정하면, 10절의 normal-closure 구성으로 그 근의 모든 conjugate를 solvable Galois group을 갖는 extension에 포함시킬 수 있다. 그런데 그 근의 minimal polynomial이 $f$이므로 다섯 근이 모두 포함되어야 한다. 이는 (59), (40)과 모순이다.

여기서도 irreducibility가 중요하다. 근 하나의 minimal polynomial이 $f$의 더 작은 factor가 아니라 $f$ 전체이므로, 한 근에 대한 가정으로 나머지 네 근까지 같은 extension에 포함된다는 결론을 얻는다.

</details>

그렇다고 모든 5차방정식을 radical로 풀 수 없다는 뜻은 아니다. 예를 들어

$$
\begin{gathered}
x^5-2=0\\
\Longrightarrow
x_k=\sqrt[5]2\,\zeta_5^k,\\
k=0,\ldots,4.
\end{gathered}
\tag{61}
$$

이다. $\zeta_5$는 primitive fifth root of unity이며, $1$의 fifth root를 구할 때 허용되는 선택 중 하나다. 여기서 primitive는 $1,\zeta_5,\zeta_5^2,\zeta_5^3,\zeta_5^4$가 모두 다르고, 다섯 번째 거듭제곱에서야 $1$로 돌아온다는 뜻이다. 따라서 위의 근들은 모두 radical expression이다. $\mathbb Q$ 위의 splitting field는 order가 $20$인 solvable Galois group을 갖는다. 이 group은 $C_5$가 아니다. $\zeta_5$를 먼저 추가한 뒤에 남는 degree-five 단계가 cyclic이다.

<details markdown="1">
<summary>Radical로 풀리는 이 quintic의 group order가 20인 이유</summary>

Root of unity를 추가해 얻은 field를 **cyclotomic field**라고 한다. 여기서는 $\mathbb Q(\zeta_5)$다. $(x-1)(x^4+x^3+x^2+x+1)=x^5-1$이므로 $1$이 아닌 fifth root들은 $x^4+x^3+x^2+x+1=0$을 만족한다. 이 quartic에 $x+1$을 대입하면 $x^4+5x^3+10x^2+10x+5$가 되어 소수 $5$에 대한 Eisenstein 조건을 만족한다. 따라서 이 quartic이 minimal polynomial이고 field degree는 $4$다.

Automorphism은 $u=1,2,3,4$에 대해 $\zeta_5$를 $\zeta_5^u$로 보낸다. 이 map들을 합성하면 exponent를 곱한 뒤 modulo $5$로 계산한다. $2$를 거듭제곱하면 modulo $5$에서 $1,2,4,3,1$ 순서가 나오므로, $\zeta_5\mapsto\zeta_5^2$를 반복해 네 automorphism을 모두 얻는다. 따라서 group은 $C_4$다.

$L=\mathbb Q(\sqrt[5]2,\zeta_5)$라고 하자. Degree가 $4$인 extension에 fifth root 하나를 추가하므로 전체 degree는 최대 $20$이다. 한편 $L$에는 degree가 $4$인 cyclotomic field와 degree가 $5$인 field $\mathbb Q(\sqrt[5]2)$가 모두 들어 있다. 후자의 defining polynomial은 소수 $2$에 대한 Eisenstein 조건을 만족한다. 따라서 전체 degree는 $4$와 $5$의 배수여야 하므로 $[L:\mathbb Q]=20$이다.

$\zeta_5$를 고정하는 subgroup의 order는 $20/4=5$다. 이 subgroup은 $\sqrt[5]2$를 $\sqrt[5]2\,\zeta_5^k$로 보내며, 반복 적용은 exponent를 modulo $5$로 더하는 계산이므로 $C_5$를 이룬다. Quotient는 $\zeta_5$를 바꾸는 네 가지 선택에 대응하는 $C_4$다. 둘 다 abelian이므로 전체 group은 solvable하다. Fifth root의 다섯 가지 선택뿐 아니라 root of unity를 바꾸는 automorphism도 함께 세어야 한다.

</details>

## 16. Radical 공식이 없어도 근삿값은 구할 수 있다
{: id="16-radical-공식이-없어도-수치적으로-근을-구할-수-있다" }

(53)의 세 실근은 $[-3,-1]$, $[-1,1]$, $[1,3]$에 각각 들어 있다. 세 구간 모두 양 끝점이 유리수이고, 두 끝점에서 함수값의 부호가 다르다. **Bisection**은 이 부호 차이를 유지하면서 구간의 폭을 매번 절반으로 줄인다. 함수가 continuous하므로 좁힌 구간에도 근이 있다.

가운데 구간에서 $f(-1)=5$, $f(1)=-3$이다. Midpoint $0$에서는 $f(0)=1$이므로 두 끝점에서 함수값의 부호가 다른 구간 $[0,1]$을 남긴다. 다음 midpoint $1/2$에서는 $f(1/2)=-47/32$이므로 이번에는 $[0,1/2]$을 남긴다. 근이 있는 구간을 좁혀 가는 계산이며, 근을 radical expression으로 바꾸는 과정은 아니다.

여기서 “정확하다”는 말을 구분해서 쓸 필요가 있다. 근을 “$[0,1/2]$ 안에 있는 $x^5-5x+1$의 유일한 zero”라고 지정할 수 있고, 유리수 계산으로 그 근을 포함하는 구간을 원하는 만큼 좁힐 수도 있다. 둘 다 정확한 수학적 설명이지만 finite radical expression은 아니다. Bisection을 유한 번 실행해 얻는 값은 일반적으로 근 자체가 아니라 오차가 보장된 유리수 근삿값이다.

<details markdown="1">
<summary>유리수로 정확히 계산하는 코드와 오차 보장</summary>

처음 구간의 폭이 $W$이면 $k$번 반복한 뒤

$$
\begin{gathered}
W_k=\frac{W}{2^k},\\
\lvert x_{\mathrm{mid}}-\alpha\rvert
\le\frac{W}{2^{k+1}}.
\end{gathered}
\tag{62}
$$

이다. 이 세 구간에서는 함수가 한 방향으로만 변하므로 근이 하나씩만 존재한다. 아래 코드는 끝점을 자릿수가 제한된 floating-point 근삿값이 아니라 정확한 유리수로 계산한다. 따라서 부호 판정과 구간 폭의 bound에는 roundoff, 즉 정해진 자릿수까지만 수를 저장하면서 생기는 오차가 들어가지 않는다.

출력하는 오차 bound는 exact midpoint가 아니라 소수점 아래 12자리로 표시한 값에 대한 것이다. 표시값과 midpoint 사이의 차이를 구간 폭의 절반에 더한 뒤, $10^{-15}$ 단위로 올림한다.

```python
from fractions import Fraction
from math import ceil

def quintic(x):
    return x**5 - 5*x + 1

def bisect_exact(f, left, right, steps=50):
    left, right = Fraction(left), Fraction(right)
    if left >= right or steps < 0:
        raise ValueError("Use an increasing interval and nonnegative steps")
    fl, fr = f(left), f(right)
    if fl == 0:
        return left, left
    if fr == 0:
        return right, right
    if fl * fr > 0:
        raise ValueError("The endpoints must have opposite signs")
    for _ in range(steps):
        midpoint = (left + right) / 2
        fm = f(midpoint)
        if fm == 0:
            return midpoint, midpoint
        if fl * fm < 0:
            right = midpoint
        else:
            left, fl = midpoint, fm
    return left, right

for bracket in [(-3, -1), (-1, 1), (1, 3)]:
    left, right = bisect_exact(quintic, *bracket)
    midpoint = (left + right) / 2
    assert right - left <= Fraction(2, 2**50)
    displayed = f"{float(midpoint):.12f}"
    error_bound = abs(Fraction(displayed) - midpoint) + (right - left) / 2
    # Both terms are exact; rounding upward keeps the printed bound conservative.
    bound_units = ceil(error_bound * 10**15)
    print(f"root ≈ {displayed}; error ≤ {bound_units}e-15")
```

실근의 근삿값은 각각 $-1.541651684105$, $0.200064102630$, $1.440500397342$다. 나머지 두 근은 약 $-0.049456407934\pm1.499441367239\,i$다. 이 값들은 근의 위치를 보여 준다. Galois group은 근삿값으로 판정한 것이 아니라 14절의 논증으로 증명했다.

</details>

{% include interactive/article-lab.html id="galois-quintic" %}

각 단계를 실행하기 전에 midpoint에서의 부호를 보고 어느 쪽 절반이 남을지 예상한 뒤, 구간 폭을 확인해 보자. $x^5-2$를 선택해도 같은 알고리즘이 수렴한다. Radical로 풀리는지와 관계없이 근삿값을 구할 수 있으며, 수치 계산의 수렴 여부로 radical solvability를 판정할 수는 없다. Residual $\lvert f(x)\rvert$가 작다는 것은 함수값이 $0$에 가깝다는 뜻이지, 그 자체로 $x$의 오차를 보장하지는 않는다. 여기서는 부호가 다른 두 끝점으로 근을 포함하는 구간을 유지하기 때문에 오차를 보장할 수 있다.

<details markdown="1">
<summary>선택 사항: SymPy로 Galois group과 근삿값 계산하기</summary>

앞의 핵심 Python 예제는 standard library만 사용하지만, 아래 코드에는 SymPy가 필요하다. `galois_group`은 symbolic computation으로 group을 구하고, `nroots`는 근삿값을 구한다. 두 함수는 서로 다른 일을 한다. API는 SymPy 1.14.0에서 확인했다. [SymPy number-field documentation](https://docs.sympy.org/latest/modules/polys/numberfields.html)

```python
import sympy as sp
from sympy.polys.numberfields import galois_group

x = sp.Symbol("x")
f = x**5 - 5*x + 1
assert sp.expand(f.subs(x, x - 1)) == x**5 - 5*x**4 + 10*x**3 - 10*x**2 + 5
assert sp.Poly(f, x).is_irreducible
group, contained_in_alternating = galois_group(f, x)
assert group.order() == 120 and not group.is_solvable
assert not contained_in_alternating
print("Galois-group order:", group.order())
print("Numerical roots:", sp.nroots(f))
```

Library에서 permutation의 곱을 정의하는 순서는 앞의 `compose`와 다를 수 있다. Group order와 solvability는 어느 convention을 써도 같지만, 개별 permutation을 비교할 때는 합성 순서를 맞춰야 한다.

</details>

## 17. 연습문제: radical로 풀 수 있는지 판단하기
{: id="17-연습문제-어떤-조건이-radical-표현을-가로막는가" }

문제에 필요한 것은 앞에서 구분한 내용들이다. 유한한 공식과 근삿값을 계속 개선하는 계산은 다르고, linear combination을 보존하는 것과 곱셈을 보존하는 것도 다르다. Nonabelian이라는 사실만으로 derived series가 identity에 도달하지 못한다고 할 수도 없다. Field와 subgroup 문제에서는 더 많은 수를 고정할수록 가능한 automorphism이 줄어든다는 점을 떠올리면 된다. 새로운 정리나 symbolic-algebra package는 필요하지 않다.

<details markdown="1">
<summary>문제 그림 한눈에 보기</summary>

<div class="galois-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-10-exercises.svg" alt="Radical expression tree, 좌표 stretch, root-pair exchange, field chain, derived-series 원소 수로 구성한 문제 입력." width="960" height="1180">
</div>
<div class="galois-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-10-25-galois-from-scratch/fig-10-exercises-mobile.svg" alt="Radical expression tree, 좌표 stretch, root-pair exchange, field chain, derived-series 원소 수로 구성한 문제 입력. 모바일 배치." width="480" height="1740">
</div>
*Figure 10. 다섯 interactive 문제에 주어진 조건을 모았다. 2번의 좌표는 basis $(1,\sqrt2)$에 대한 유리수 계수이고, 3번에서는 $2,3$을 고정한다. 4번에는 $\sigma$와 field $E$를, 마지막 panel에는 derived subgroup의 원소 수를 표시했다.*

</details>

{% include interactive/article-lab.html id="galois-quiz" %}

다섯 문제로 radical expression과 근사 계산의 차이, linearity와 multiplicativity의 차이, 근의 permutation이 automorphism이 되는 조건, field와 subgroup 사이의 반대 포함관계, nonabelian과 nonsolvable의 차이를 확인할 수 있다. 보기를 고르면 맞거나 틀린 이유가 나온다. 이전·다음 문제로 이동해도 선택한 답은 유지되며, Reset을 누르면 모두 초기화된다.

### Coding exercise: generator로 만든 group의 solvability 판정하기
{: id="coding-exercise-generator-목록과-generated-group은-다르다" }

`generated_solvable(generators, n)`을 구현하자. 입력은 $n$개의 label에 대한 올바른 zero-based permutation들이다. 먼저 이 permutation들로 subgroup을 생성하고, 그 group에서 derived subgroup을 반복해서 구한다. Series가 identity에 도달하면 `True`, identity가 아닌 subgroup에서 멈추면 `False`를 반환한다. 처음 group의 원소 수만으로는 판정할 수 없다.

예를 들어 `generated_solvable([(1, 0, 2), (1, 2, 0)], 3)`의 반환값은 `True`여야 한다. 여기서 `n=3`은 label `0, 1, 2`의 개수이지 **group order가 아니다**. 첫 tuple은 label `0`과 `1`을 바꾸고, 두 번째는 `0 → 1 → 2 → 0`으로 보낸다. 두 generator로 만든 closure는 $S_3$의 여섯 원소 전체다. Derived subgroup의 order가 $6\to3\to1$로 줄어들어 identity에 도달하므로, nonabelian group이지만 답은 `True`다.

아래 $S_5$ test도 generator는 두 개지만 closure의 order는 $120$이다. Derived subgroup의 order가 $120\to60\to60$으로 바뀌다가 identity가 아닌 subgroup에서 멈추므로 `False`를 반환해야 한다. 반환할 것은 order 목록이 아니라 이 Boolean이다.

13절의 `group_closure`와 `derived_subgroup`을 사용하면 되며, 두 함수를 다시 구현할 필요는 없다. `group`은 permutation tuple들을 담은 Python set이다. 따라서 `len(group)`은 원소 수이고, `next_group == group`은 원소 수만이 아니라 실제 집합이 같은지 비교한다. Generator 목록이 비어 있어도 closure에는 identity가 들어간다. 이 경우에는 loop를 건너뛰고 `True`를 반환한다.

Starter code의 TODO 두 곳에는 다음 subgroup을 구하는 함수와, identity가 아닌 group에서 더는 줄어들지 않을 때 반환할 값을 넣으면 된다. 아직 완성하지 않은 연습용 코드이므로 TODO를 채우기 전에 호출하면 `NotImplementedError`가 발생한다.

```python
def generated_solvable(generators, n):
    """Complete the two TODOs; input permutations are assumed valid."""
    group = group_closure(generators, n)
    while len(group) > 1:
        # TODO 1: compute the subgroup generated by all commutators.
        next_group = None
        if next_group is None:
            raise NotImplementedError("Replace TODO 1 with a derived subgroup")
        # TODO 2: what does equality imply when len(group) > 1?
        if next_group == group:
            raise NotImplementedError("Return the correct Boolean at a fixed point")
        group = next_group
    return True
```

아래 test case에는 nonabelian이면서 답이 `True`인 group과 `False`인 group이 모두 있다. Generator가 주어져 있으므로 별도의 group-theory library 없이 풀 수 있다.

<details markdown="1">
<summary>실행 가능한 테스트와 reference solution</summary>

```python
def reference_generated_solvable(generators, n):
    group = group_closure(generators, n)
    while len(group) > 1:
        next_group = derived_subgroup(group)
        if next_group == group:
            return False
        group = next_group
    return True

SOLVABILITY_CASES = [
    ("identity", [], 5, True),
    ("C5", [(1, 2, 3, 4, 0)], 5, True),
    ("V4", [(1, 0, 3, 2), (2, 3, 0, 1)], 4, True),
    ("S3", [(1, 0, 2), (1, 2, 0)], 3, True),
    ("S4", [(1, 0, 2, 3), (1, 2, 3, 0)], 4, True),
    ("A5", [(1, 2, 0, 3, 4), (1, 2, 3, 4, 0)], 5, False),
    ("S5", [(1, 0, 2, 3, 4), (1, 2, 3, 4, 0)], 5, False),
]

def test_solvability(candidate):
    for name, generators, n, expected in SOLVABILITY_CASES:
        actual = candidate(generators, n)
        assert actual is expected, (name, actual, expected)
    return len(SOLVABILITY_CASES)

assert test_solvability(reference_generated_solvable) == 7
print("Reference solution: 7/7 cases")
# After completing the starter, run: test_solvability(generated_solvable)
```

Derived subgroup은 현재 group에 포함된다. 그보다 작은 **proper subgroup**이면 원소 수가 줄어들고, 그렇지 않으면 현재 group과 같으므로 이후에도 바뀌지 않는다. Order는 양의 정수이므로 이 loop는 결국 order $1$에 도달하거나 identity가 아닌 fixed point에서 멈춘다.

테스트는 주어진 group 일곱 개에 대한 결과를 확인한다. 이 알고리즘이 solvability를 올바르게 판정하는 이유는 테스트 통과 여부가 아니라, solvability의 정의와 위의 descending-chain 논증에 있다.

</details>

## 18. 5차방정식의 근의 공식이 없다는 말의 정확한 의미

Field extension은 사칙연산 규칙을 그대로 유지하면서 새로운 수를 추가하는 방법이다. Extension degree는 base field를 scalar로 쓸 때 필요한 basis vector의 개수다. Galois group은 base field 위의 대수적 관계를 모두 보존하는 근의 permutation들로 이루어진다. Label을 마음대로 바꾼 permutation이 항상 이 관계를 보존하는 것은 아니다.

Radical로 풀 수 있는 polynomial의 Galois group은 abelian quotient를 갖는 단계들로 나누어 identity까지 줄일 수 있다. 4차까지의 symmetric group과 그 subgroup들은 이 조건을 만족한다. 반면 $S_5$의 commutator들은 $A_5$를 생성하고, $A_5$ 안에서 다시 commutator를 구해도 $A_5$ 전체가 생성된다.

$x^5-5x+1$은 바로 이 이유로 radical로 풀리지 않는다. 그 근들은 algebraic number로서 잘 정의되어 있고, 원하는 정확도까지 근삿값도 구할 수 있다. 불가능한 것은 유리수에서 출발해 사칙연산과 radical만을 유한 번 사용하여 그 근들을 정확히 나타내는 것이다.

## 참고문헌

- [J. S. Milne, *Fields and Galois Theory*](https://www.jmilne.org/math/CourseNotes/FT.pdf): extension degree, finite Galois correspondence, solvability criterion, prime-degree polynomial의 실근 수를 이용한 판별 조건.
- [Keith Conrad, *Simple Radical Extensions*](https://kconrad.math.uconn.edu/blurbs/galoistheory/simpleradical.pdf): simple radical extension의 degree와 radical root의 선택.
- [Romyar Sharifi, *Abstract Algebra*, Chapter 10](https://www.math.ucla.edu/~sharifi/notes/algebra-ch10.html): radical extension과 solvability theorem.
- [John Beachy, *Galois Theory*](https://faculty.niu.edu/math_beachy/aaol/galois.shtml): polynomial 방정식을 푼다는 것의 field-theoretic 의미.
- [SymPy, Number Fields](https://docs.sympy.org/latest/modules/polys/numberfields.html): 선택 사항인 exact Galois-group 계산의 API.
