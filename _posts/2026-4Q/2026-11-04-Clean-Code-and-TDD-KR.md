---
title: "Clean Code와 TDD: 읽기 쉽고 고치기 쉬운 코드"
description: "워크숍 신청을 처리하는 Python 코드에서 정원 제한, 대기 순서, 입력 보존을 테스트로 확인하며 이름과 함수의 역할을 정리한다."
date: 2026-11-03 21:00:00 +0900
categories: [Software Engineering]
tags: [software-engineering, clean-code, python, refactoring, software-testing, tdd]
math: true
interactive_labs: [cleancode]
pin: false
topic: software-engineering
lang: ko
translation_key: clean-code-and-tdd
image:
  path: /assets/img/posts/2026-11-04-clean-code-tdd/cover-flat.png
  alt: "정원 1명에 A가 확정, B가 대기 중일 때 A를 취소한다. 실제 확정 명단은 [A]에서 [B]로 바뀌고 refactoring 뒤에도 [B]를 유지한다. 기대값은 세 단계 모두 [B]다."
  width: 1672
  height: 941
  fit: contain
  hide_caption: true
---

[Python + pytest 실습 프로젝트 ZIP]({{ site.baseurl }}/assets/code/clean-code-tdd-workshop.zip) · [핵심 구현 코드]({{ site.baseurl }}/assets/code/clean-code-tdd/workshop.py) · [실제 테스트 실행 기록]({{ site.baseurl }}/assets/data/cleancode/tdd-runs.json)

지금은 맞는 결과를 내는 프로그램도 다음 기능을 추가하려고 보면 고치기 어려울 수 있다. 정원이 두 명인 워크숍에 A와 B가 참가 확정이고 C와 D가 대기 중이라고 하자. A가 취소하면 B와 C가 확정 명단에 남고 D는 계속 대기해야 한다. 코드에서 이 규칙은 어디에 있을까? 그 부분을 고치면 다른 동작도 달라질까? 무엇을 확인해야 안심할 수 있을까?

여기서 **clean code**는 동작과 역할, 수정해야 할 지점을 이해하기 쉬운 코드를 뜻한다. **Test-driven development**, 줄여서 **TDD**는 테스트를 먼저 작성하고 그 테스트에 맞춰 구현하는 개발 방식이다. 작은 동작 하나를 정해 테스트로 표현하고, 이를 통과하도록 구현한 뒤, 동작을 유지하면서 구조를 개선할 필요가 있는지 살펴본다. 둘은 연결되지만 같은 뜻은 아니다. [Kent Beck, *Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd), [Martin Fowler, *Test Driven Development*](https://martinfowler.com/bliki/TestDrivenDevelopment.html).

Tim Peters는 *The Zen of Python*에서 다음과 같이 썼다.

> Readability counts.

— [PEP 20](https://peps.python.org/pep-0020/). 읽기 쉬운 코드가 중요하다는 뜻이다. 그렇다고 짧게 쓰기만 하면 되는 것은 아니다. 짧은 식이 규칙을 감출 수도 있고, 조금 긴 이름이 의도를 더 분명히 드러낼 수도 있다. 테스트는 수정한 동작을 확인하는 데 도움이 되지만, 설계에 관한 모든 판단을 대신하지는 않는다.

## 1. 함수의 줄 수보다 바꾸려는 동작부터 보자

“확정 참가자가 취소하면 가장 먼저 대기한 사람에게 자리를 준다”는 요청을 생각해 보자. 참가자를 찾고, 명단을 바꾸고, 파일을 저장하고, 메시지를 출력하는 일이 한 함수에 섞여 있다면 한 번에 신경 써야 할 것이 많아진다. 함수 이름까지 `process(data, x, flag)`라면 각 인자가 무엇인지 본문을 읽어 가며 알아내야 한다.

확인할 질문은 구체적이다.

- 파일 저장 코드를 읽지 않고도 취소 규칙을 찾을 수 있는가?
- 대기 순서도 결과의 일부인가?
- 함수를 호출하면 호출자가 가지고 있던 명단까지 바뀌는가?
- 테스트가 대기 명단의 첫 사람과 마지막 사람을 구별하는가?

이 질문들에 답하기 쉽도록 이름과 함수의 역할을 정리하고 테스트를 작성한다. 함수의 줄 수를 줄이는 것 자체가 목표는 아니다. Google의 code review 지침도 불필요한 복잡성과 아직 필요하지 않은 일반화를 살펴보라고 권한다. 모든 함수에 적용할 수 있는 길이 하나를 제시하는 방식은 아니다. [Google Engineering Practices, *What to look for in a code review*](https://google.github.io/eng-practices/review/reviewer/looking-for.html).

예제는 함수, 조건문, list, dict를 알면 시작할 수 있다. 함수의 **interface**는 다른 코드가 그 함수를 사용하는 방법이다. 이름, 인자, 결과, 문서에 정한 오류 처리가 여기에 포함된다. **Implementation**은 그 동작을 실제로 수행하는 내부 코드다. `cancel(state, "A")`처럼 호출하기로 정했다고 해서 내부의 반복문이나 보조 함수(**helper**)까지 정해진 것은 아니다. 다른 코드에서 호출하도록 공개한 함수들과 그 사용 규약을 프로그램의 **API**라고 한다. 웹서비스에만 쓰는 말은 아니다.

## 2. 워크숍의 규칙을 먼저 분명히 정하자

**Contract**는 허용하는 입력과 호출자가 기대할 수 있는 결과를 정한 약속이다. 상태는 정확히 세 필드를 가진 dict로 나타낸다.

```python
state = {
    "capacity": 2,
    "confirmed": ["A", "B"],
    "waiting": ["C", "D"],
}
```

`capacity`는 양의 정수다. `confirmed`와 `waiting`은 참가자 ID를 순서대로 담은 list다. ID는 비어 있지 않은 문자열을 그대로 비교한다. `"A"`와 `"a"`는 서로 다르며, 공백을 지우거나 실제 사용자 계정의 이름 규칙을 검사하지 않는다.

동작은 다음과 같이 정한다.

1. 자리가 남아 있으면 신청한 ID를 `confirmed` 끝에 넣는다.
2. 정원이 찼으면 `waiting` 끝에 넣는다.
3. 어느 명단에든 이미 있는 ID는 다시 넣지 않는다.
4. 확정 참가자가 취소하면 그 ID를 빼고, 대기자가 있다면 맨 앞의 한 명을 승급한다.
5. 대기자가 취소하면 그 ID만 뺀다. 없는 ID의 취소 요청은 아무것도 바꾸지 않는다.
6. 인자로 받은 dict와 그 안의 list는 바꾸지 않는다. 값이 그대로인 경우에도 새 dict와 독립적인 두 list를 반환한다.

이전 상태를 그대로 남겨 두면 처리 전후를 비교할 수 있다. 반환된 list도 따로 만들기 때문에, 나중에 결과를 수정하더라도 이전 명단은 바뀌지 않는다.

**FIFO**는 *first in, first out*의 약자다. 먼저 들어온 항목을 먼저 꺼낸다는 뜻이다. C가 D보다 먼저 기다렸으므로 C를 먼저 승급하고, 나머지 참가자들의 상대적인 순서는 유지한다. A를 취소했을 때 기대하는 상태는 다음과 같다.

```python
expected = {
    "capacity": 2,
    "confirmed": ["B", "C"],
    "waiting": ["D"],
}
```

**Invariant**는 유효한 연산 전후에 항상 성립해야 하는 조건이다. 여기서는 확정 인원이 정원을 넘지 않고, 두 명단 전체에 같은 ID가 두 번 나오지 않아야 한다. 대기자가 있다면 확정 명단은 가득 차 있어야 한다. 하지만 이 조건들만 확인해서는 FIFO까지 검증할 수 없다. C 대신 D를 승급해도 세 조건을 모두 만족할 수 있기 때문이다.

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-01-state.svg" alt="A를 취소하면 B는 남고, 대기 명단의 맨 앞 C가 승급하며 D는 계속 대기한다." width="960" height="670">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-01-state-mobile.svg" alt="취소와 FIFO 승급을 위아래로 배치한 모바일 그림." width="480" height="1000">
</div>
*Figure 1. 정원은 두 명이다. A가 빠지면 B가 명단 앞쪽으로 이동하고, 가장 먼저 대기한 C가 그 뒤에 추가된다. 남은 대기자는 D다. 숫자는 실제 좌석 번호가 아니라 list의 index다.*

{% include interactive/article-lab.html id="cleancode-state" %}

A, B, C, D를 차례로 신청한 뒤 A를 취소해 보자. 결과를 보기 전에 두 명단이 어떻게 바뀔지 예상해 보면 좋다. 이번에는 대기자를 취소하거나 이미 있는 ID로 다시 신청해 보자. 변화가 없는 경우도 contract에 포함된 동작이다.

<details markdown="1">
<summary>입력 검사와 이 예제에서 다루는 범위</summary>

다운로드한 구현은 상태의 필드, 정수인 정원, list의 내용, ID 중복, 인원 조건을 검사한다. `True`와 `False`는 참·거짓을 나타내는 Python의 `bool` 값이다. Python은 이를 여러 정수 연산에서 1과 0처럼 다루지만, 여기서는 정원으로 허용하지 않는다. 잘못된 상태나 ID를 받으면 `ValueError`를 발생시킨다. 이는 값이 허용 조건에 맞지 않는다는 것을 알리는 exception이다. 공백만 있는 문자열도 비어 있지는 않으므로, 이 단순한 ID 규칙에서는 허용한다.

정원 변경은 마지막 과제에서 추가한다. 그전까지 하나의 실행 중인 프로그램이 요청을 하나씩 순서대로 처리한다. 파일 저장을 붙인다고 동시 신청 처리, 인증, 프로그램이 중단돼도 안전한 저장까지 구현되는 것은 아니다. 그런 기능에는 별도의 요구사항과 검사가 필요하다.

</details>

## 3. Assertion은 실제 결과와 따로 정한 기대값을 비교한다

**Test**는 상황을 준비하고, 동작을 실행한 뒤, 기대한 결과가 나오는지 확인하는 코드다. 그중 결과를 확인하는 식이 **assertion**이다.

`from workshop import cancel`은 `workshop.py`에서 `cancel` 함수를 가져와 쓸 수 있게 한다.

여기서 사용하는 일반 실행 모드에서 `assert 조건`은 조건이 참이면 다음 줄로 진행하고, 거짓이면 `AssertionError`를 발생시킨다. pytest는 이를 테스트 실패로 보고한다. [Python, *The assert statement*](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).

`==`는 dict의 키와 값을 비교하며, 키를 적은 순서는 상관없다. 다만 `confirmed`처럼 값이 list라면 항목의 순서까지 비교한다. 따라서 `["B", "C"]`와 `["C", "B"]`는 서로 다른 결과다. [Python, *Built-in Types*](https://docs.python.org/3/library/stdtypes.html).

```python
from workshop import cancel


def test_cancelling_confirmed_promotes_first_waiting():
    before = {
        "capacity": 2,
        "confirmed": ["A", "B"],
        "waiting": ["C", "D"],
    }

    actual = cancel(before, "A")

    assert actual == {
        "capacity": 2,
        "confirmed": ["B", "C"],
        "waiting": ["D"],
    }
    assert before == {
        "capacity": 2,
        "confirmed": ["A", "B"],
        "waiting": ["C", "D"],
    }
```

첫 assertion은 반환한 상태를 확인하고, 둘째는 호출자가 넘긴 원본 상태가 유지됐는지 확인한다. 반환값이 맞더라도 원본까지 바꿔 버리는 구현이 있을 수 있기 때문이다.

이 구성을 **Arrange–Act–Assert**라고도 부른다. `before`를 준비하고, `cancel`을 호출하고, 결과를 검사하는 순서다.

기대값은 워크숍의 규칙에서 정한다. `expected = cancel(before, "A")`로 기대값을 만든 뒤 다른 `cancel` 호출 결과와 비교하면 같은 버그를 양쪽에 반복할 수 있다. **Test oracle**은 테스트가 정답을 판단하는 근거를 뜻한다. 여기서는 FIFO 규칙을 보고 직접 적은 상태가 그 역할을 한다. Beck도 테스트를 통과시키려고 실제 계산 결과를 기대값에 그대로 붙여 넣는 일을 경계한다. [*Canon TDD*, “Make it Pass”](https://newsletter.kentbeck.com/p/canon-tdd).

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-02-test.svg" alt="같은 취소 입력에서 FIFO 규칙의 기대값과 일부러 잘못 승급한 결과를 비교해 C와 D의 불일치를 찾는다." width="960" height="810">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-02-test-mobile.svg" alt="같은 취소 입력에서 FIFO 기대값과 일부러 잘못 만든 결과의 ID를 비교하는 모바일 그림." width="480" height="1060">
</div>
*Figure 2. Expected는 FIFO 규칙에서 따로 정한 값이다. 오류가 있는 구현은 C 대신 D를 승급한다. 확정 인원은 둘 다 두 명이지만, 명단의 값과 순서는 다르므로 assertion이 실패한다.*

## 4. pytest로 실제 테스트 실행하기

**pytest**는 test runner다. 테스트 함수를 찾아 실행하고, 실패한 부분을 보여 준다. 프로젝트를 내려받아 압축을 푼 뒤 `workshop.py`와 `pytest.ini`가 있는 폴더에서 terminal을 열고 실행한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

둘째 명령은 macOS와 Linux에서 가상환경을 활성화한다. Windows PowerShell에서는 `.\.venv\Scripts\Activate.ps1`를 쓴다. **가상환경**은 프로젝트에서 설치한 Python 패키지를 다른 프로젝트와 분리한다. [Python, `venv`](https://docs.python.org/3/library/venv.html#how-venvs-work).

`python -m pytest`는 선택한 Python 인터프리터로 pytest를 실행한다. 실습 프로젝트는 pytest 8.4.2를 사용한다.

이 명령은 완성된 예제 코드를 검사하므로 설치가 잘됐다면 테스트가 통과해야 한다. 5절의 단계별 TDD 기록에서는 아직 완성되지 않은 구현과, 다음 수정을 이끌어 내는 실패를 비교한다.

앞 절의 테스트를 `tests/test_walkthrough.py`로 저장했다면 `python -m pytest -q tests/test_walkthrough.py`로 그 파일만 실행할 수 있다. `test_*.py`와 `test_*` 같은 이름은 pytest가 테스트를 찾는 규칙이다. 결과의 `F`는 실패, `.`는 통과를 나타낸다. `[100%]`는 실행이 끝났다는 표시이지, 모든 테스트를 통과했다는 뜻이 아니다. [pytest, *Get Started*](https://docs.pytest.org/en/stable/getting-started.html).

실패 보고서에는 실제 결과와 기대값의 차이, assertion이 있는 위치가 나온다. 그 실패가 확인하려던 동작에서 발생했는지 살펴보자. `["B", "C"]`여야 할 결과가 `["B", "D"]`라면 승급 순서를 확인해야 한다. pytest를 찾을 수 없다면 설치나 실행 환경을 확인해야 한다. 화면이 빨갛다는 사실만으로는 어느 쪽인지 알 수 없다.

<details markdown="1">
<summary>프로젝트 파일 구성과 의도한 실패 구별하기</summary>

```text
clean-code-tdd/
    workshop.py
    storage.py
    exercise.py
    exercise_solution.py
    tests/
    exercise_tests/
    requirements.txt
    pytest.ini
    README.md
```

`exercise_tests`는 미완성인 학생용 코드를 검사하므로 마지막 과제를 시작할 때 따로 실행한다.

`python -m pytest -q tests/test_walkthrough.py -x`는 그 파일의 첫 실패에서 실행을 멈춘다. `python -m pytest --version`으로 설치한 버전을 확인할 수 있다. 정답 코드는 별도 파일에 두었다. 정답을 읽거나 실행해도 작성 중인 함수가 덮어써지지 않는다.

</details>

## 5. TDD는 다음에 확인할 동작 하나를 고른다

**Red–Green–Refactor**는 작은 개발 주기의 세 단계를 뜻한다.

1. **Red.** 다음에 필요한 동작이 없거나 잘못되어 테스트가 실패한다.
2. **Green.** 그 테스트와 기존 테스트를 통과하도록 구현을 바꾼다.
3. **Refactor.** 개선할 부분이 있으면 동작을 유지하면서 내부 구조를 다듬는다.

먼저 확인할 상황의 목록을 만든다. 첫 신청, 두 번째 신청, 중복 신청, 정원 초과, 대기 순서, 취소, 승급, 없는 ID 등이 있다. 이 모든 것을 미리 테스트 코드로 완성하는 대신, 항목 하나를 골라 실행 가능한 테스트로 만든다. 그 주기를 마치면 다음 항목을 고르고, 새로 발견한 상황은 목록에 추가한다. Beck은 테스트 목록과 다음 테스트 하나의 작성을 분명히 구분한다. Fowler는 테스트를 먼저 생각하는 과정이 내부 구현보다 interface를 먼저 고민하게 한다고 설명한다. [Beck, *Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd), [Fowler, *Test Driven Development*](https://martinfowler.com/bliki/TestDrivenDevelopment.html).

**Green은 현재 실행한 테스트가 통과했다는 뜻이다.** 목록에 적은 모든 기능을 구현했다는 뜻은 아니다. 버그가 내는 값에 맞춰 기대값을 바꾸는 것도 올바른 Green 단계가 아니다.

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-03-cycle.svg" alt="실패한 구체적인 assertion, 통과시키기 위한 동작 추가, 같은 테스트로 확인하는 구조 변경." width="960" height="730">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-03-cycle-mobile.svg" alt="구체적인 Red Green Refactor 단계를 세로로 배치한 그림." width="480" height="960">
</div>
*Figure 3. 정원은 한 명이고 A가 확정, B가 대기 중이다. A를 취소한 뒤의 확정 명단이 `[A] → [B] → [B]`로 바뀐다. 숫자는 저장된 전체 테스트의 통과·실패 개수이며, 회색 경로는 다음에 확인할 동작을 고르는 과정이다.*

{% include interactive/article-lab.html id="cleancode-tdd" %}

이 단계별 실습은 각 Python 코드 버전을 실제로 실행해 저장한 기록을 보여 준다. Red 단계마다 테스트를 하나 추가하고, 다음 Green 단계에서 그 동작을 구현한다. 대기자 취소를 먼저 구현한 뒤, 확정 참가자를 취소했을 때 FIFO 순서로 승급하는 경우를 추가한다. 마지막 Refactor 단계에서는 직전 Green 단계와 같은 테스트 다섯 개로 구조 변경을 확인한다.

추가한 테스트와 구현, 실패 이유를 함께 비교해 보자. 기록에 담긴 코드 버전은 유효한 입력이 들어온다고 가정한다. 완성본 `workshop.py`는 contract에 명시한 입력 조건도 검사한다. 브라우저는 저장된 기록을 보여 주며 Python이나 pytest를 직접 실행하지 않는다. [실행 기록 파일]({{ site.baseurl }}/assets/data/cleancode/tdd-runs.json)에는 사용한 명령, 버전, 결과, 코드가 들어 있다.

## 6. 첫 신청부터 정원 초과까지 하나씩 구현하기

처음에는 빈자리가 있을 때 신청자를 확정 명단에 넣는 경우부터 구현한다. `new_workshop(2)`는 정원이 두 명이고 `confirmed`와 `waiting`이 모두 빈 새 상태를 만든다. 테스트는 여기에 A를 신청하면 확정 명단에 A가 들어가고 대기 명단은 비어 있는지 확인한다.

```python
from workshop import new_workshop, register


def test_first_registration():
    actual = register(new_workshop(2), "A")
    assert actual == {
        "capacity": 2,
        "confirmed": ["A"],
        "waiting": [],
    }
```

처음에는 모든 요청을 `confirmed`에 추가하는 구현으로도 이 테스트를 통과할 수 있다. 하지만 중복 신청과 정원 초과에서는 틀린 결과가 나온다. 다음 테스트에는 참가자 이름만 바꾸기보다, 아직 구현하지 않은 중복 신청이나 정원 초과 상황을 넣는 편이 도움이 된다.

```python
from workshop import new_workshop, register


def test_duplicate_registration_changes_nothing():
    once = register(new_workshop(2), "A")
    twice = register(once, "A")
    assert twice == {
        "capacity": 2,
        "confirmed": ["A"],
        "waiting": [],
    }


def test_waiting_duplicate_changes_nothing():
    before = {
        "capacity": 1,
        "confirmed": ["A"],
        "waiting": ["B"],
    }
    assert register(before, "B") == {
        "capacity": 1,
        "confirmed": ["A"],
        "waiting": ["B"],
    }


def test_registration_at_capacity_joins_waiting():
    full = register(new_workshop(1), "A")
    actual = register(full, "B")
    assert actual == {
        "capacity": 1,
        "confirmed": ["A"],
        "waiting": ["B"],
    }
```

두 중복 테스트는 확정된 ID와 대기 중인 ID의 재신청을 각각 확인한다. 따라서 새 ID를 넣기 전에 두 명단을 모두 확인해야 한다. 정원이 찬 테스트를 추가하면 어느 명단에 넣을지 결정해야 한다. 여기서 **boundary**, 즉 경계는 확정 인원이 정원과 같아지는 지점이다. 정원이 한 명이고 이미 한 명이 확정이라면 다음 신청자는 대기해야 한다. 자리가 넉넉한 경우만 검사해서는 이 조건의 `<`와 `<=`를 구별할 수 없다.

이 테스트들은 내부 코드를 특정한 방식으로 나누라고 요구하지 않는다. 인자 두 개를 받는 함수로 시작하고, 예제를 추가하며 필요한 역할이 드러나면 구현을 정리할 수 있다. 지금 필요한 동작과 관계없는 구조까지 미리 만들 필요는 없다.

<details markdown="1">
<summary>입력 검사와 명단 배치 규칙을 구분해서 읽기</summary>

상태 검사는 `validate_state`에 맡긴다. `register_worked`는 참가자 ID의 형식을 확인한 뒤 중복 여부와 자리 배치를 판단한다.

```python
from workshop import validate_state


def register_worked(state, participant_id):
    validate_state(state)
    if not isinstance(participant_id, str) or not participant_id:
        raise ValueError("participant_id must be a nonempty string")

    # Copy both lists: the input remains the caller's unchanged state.
    result = {
        "capacity": state["capacity"],
        "confirmed": list(state["confirmed"]),
        "waiting": list(state["waiting"]),
    }
    already_registered = (
        participant_id in result["confirmed"]
        or participant_id in result["waiting"]
    )
    if already_registered:
        return result

    if len(result["confirmed"]) < result["capacity"]:
        result["confirmed"].append(participant_id)
    else:
        result["waiting"].append(participant_id)
    return result
```

이미 신청한 ID라면 바로 반환한다. 그러면 이후 코드에서는 중복 신청을 제외한 경우만 생각하면 된다. 이렇게 특정 경우에 먼저 반환해 중첩을 줄이는 패턴을 **guard clause**라고 한다. [Fowler, *Replace Nested Conditional with Guard Clauses*](https://refactoring.com/catalog/replaceNestedConditionalWithGuardClauses.html).

</details>

## 7. 취소 테스트에서는 인원수뿐 아니라 순서도 확인해야 한다

취소 뒤에 확정자가 두 명, 대기자가 한 명이라고 해도 누가 승급했는지는 알 수 없다. 3절에서 대기자를 두 명 둔 이유가 여기에 있다. FIFO와 LIFO가 서로 다른 결과를 내도록 한 것이다.

**LIFO**는 *last in, first out*, 나중에 들어온 항목을 먼저 꺼내는 방식이다. 이 정책이라면 D가 먼저 승급한다. 다른 프로그램에서는 올바른 정책일 수 있지만, 여기서 정한 contract에는 맞지 않는다. 테스트는 요구사항을 표현해야 한다. 익숙한 자료구조의 처리 순서라고 해서 서로 바꾸어 써도 되는 것은 아니다.

취소도 경우를 나누어 생각해야 한다. 대기자가 있을 때의 확정 취소, 대기자가 없을 때의 확정 취소, 대기 취소, 없는 ID의 취소는 기대 결과가 다르다. 대기 취소에서는 확정 명단이 바뀌면 안 된다. 없는 ID라면 두 명단의 값과 순서가 모두 같아야 한다. 테스트는 이런 공개된 결과를 검사한다. 내부 helper를 몇 번 호출해야 하는지까지 정할 필요는 없다.

<details markdown="1">
<summary>겉의 dict만 복사하면 안 되는 이유</summary>

```python
before = {
    "capacity": 2,
    "confirmed": ["A"],
    "waiting": [],
}
shallow = dict(before)
shallow["confirmed"].append("B")
assert before["confirmed"] == ["A", "B"]
```

`dict(before)`는 바깥 dict를 새로 만들지만 안쪽 list는 공유한다. **Aliasing**은 서로 다른 참조가 같은 객체를 가리키는 경우를 뜻한다. 어느 참조로 `append`를 호출해도 같은 list가 바뀐다. Python의 단순 대입 자체는 객체를 복사하지 않는다. [Python, *copy — Shallow and deep copy operations*](https://docs.python.org/3/library/copy.html).

`register_worked`처럼 각 list를 따로 복사하면 이 상태 구조에서는 충분하다. List의 원소가 내용을 바꿀 수 없는 문자열이기 때문이다. 더 복잡하게 중첩된 모든 구조에 그대로 적용할 수 있는 deep-copy 방법이라는 뜻은 아니다. List처럼 그 자리에서 내용을 바꿀 수 있는 객체를 **mutable**하다고 한다. 문자열은 내용을 그 자리에서 바꿀 수 없으므로 **immutable**하다.

입력을 바꾸지 않는 것은 이 예제에서 선택한 API의 약속이다. 입력을 바꾸는 API를 TDD로 개발할 수도 있다. 그때는 바뀌어야 하는 상태를 테스트로 확인해야 한다.

</details>

## 8. 이름과 함수의 역할을 정리하는 기준

이름이 분명하면 역할을 알아내기 위해 읽어야 하는 코드가 줄어든다. `participant_id`는 화면에 보여 줄 이름이 아니라 식별자라는 점을 드러낸다. `confirmed`와 `waiting`은 `list1`과 `list2`가 감추는 두 명단의 역할을 설명한다. PEP 8도 외부에 공개하는 이름은 내부 구현 방식보다 사용자가 쓰는 방식에 맞추라고 권한다. [PEP 8, *Naming conventions*](https://peps.python.org/pep-0008/#naming-conventions).

**Cohesion**은 한곳에 모아 둔 역할들이 얼마나 밀접하게 관련되어 있는지를 말한다. 신청 여부를 검사하고 어느 명단에 넣을지 판단하는 일은 같은 워크숍 규칙에 속한다. Terminal에 출력할 문장을 만드는 일은 목적이 다르다. **Coupling**은 코드 사이의 의존관계다. 규칙을 계산하는 함수가 특정 파일을 직접 열면, 그 규칙을 테스트할 때 파일도 준비해야 한다.

그렇다고 식 하나마다 함수를 만들 필요는 없다. `is_registered(state, participant_id)`는 반복해서 확인하는 조건에 의미 있는 이름을 붙여 준다. 반면 `do_part_two`라는 helper는 코드를 한 번 더 찾아 들어가게 할 뿐, 그 함수의 역할은 알려 주지 못한다. Fowler의 *Extract Function*은 함수를 추출하는 방법을 보여 준다. 어디에서 도움이 되는지는 실제 코드에 맞춰 판단해야 한다. [Refactoring catalog](https://refactoring.com/catalog/extractFunction.html).

주석에는 코드만으로 쉽게 알 수 없는 정보를 남긴다. 순서를 지켜야 하는 이유, 호환성 조건, 겉보기에는 좋아 보이는 수정이 틀리는 이유가 그 예다. `x += 1` 위의 “x를 증가시킨다”는 주석은 정보를 거의 더하지 않는다. “대기 순서는 contract이므로 맨 앞 ID부터 승급한다”는 주석은 이유를 설명한다. 코드를 바꿀 때 주석도 함께 확인해야 한다. [PEP 8, *Comments*](https://peps.python.org/pep-0008/#comments).

비슷한 코드를 너무 일찍 하나로 묶는 일에 대해 Beck은 이렇게 썼다.

> Duplication is a hint, not a command.

— [*Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd). 중복은 살펴볼 단서이지, 무조건 없애라는 명령은 아니라는 뜻이다. 비슷해 보이는 식도 서로 다른 정책을 구현하고 있어서 나중에 따로 바뀔 수 있다. 반대로 같은 워크숍 규칙이 여러 곳에 반복되면 수정할 때 일부를 놓치기 쉽다. 모양이 비슷한지보다 같은 이유로 바뀌는 코드인지가 중요하다.

## 9. Refactoring은 약속한 동작을 유지한다

Fowler의 정의에는 다음 표현이 들어 있다.

> without changing its observable behavior.

— [*Refactoring*](https://refactoring.com/). 외부에서 관찰할 수 있는 동작은 바꾸지 않는다는 뜻이다.

이 API에서는 반환한 상태, 명단 순서, 문서에 정한 오류, 호출자의 입력이 바뀌는지가 관찰할 동작에 포함된다. 지역변수 이름을 바꾸면서 이 동작들을 유지할 수 있다. 하지만 대기 명단의 앞이 아니라 뒤에서 승급하도록 바꾸면 동작이 달라진다. 코드가 짧아졌더라도 refactoring이라고 할 수 없다.

실제로는 테스트를 통과시킨 뒤, 구조 한 부분을 바꾸고, 다시 테스트하고, diff를 읽는 식으로 진행한다. **Diff**는 두 버전 사이에서 달라진 줄을 보여 준다. **Private helper**는 외부 코드에 사용을 약속한 함수가 아니라 내부 구현에 쓰는 보조 함수다. Python에서는 흔히 이름 앞에 underscore를 붙여 표시하지만, 접근을 막는 문법은 아니다. 테스트가 helper의 이름이나 호출 방식에 의존하면 올바른 refactoring에도 실패할 수 있다. 공개된 규칙 자체를 바꾸려는 경우에는 명세와 테스트를 함께 검토하고, 기능 변경이나 버그 수정으로 분명히 구분한다.

두 함수에서 반복하는 준비 작업은 분명하다. 입력 상태를 검사하고, 두 명단을 각각 복사하는 일이다. 프로젝트에서는 이 작업을 `copy_state`로 묶는다.

```python
from workshop import validate_state


def copy_state(state):
    """Validate state and return a copy with independent nested lists."""
    validate_state(state)
    return {
        "capacity": state["capacity"],
        "confirmed": state["confirmed"].copy(),
        "waiting": state["waiting"].copy(),
    }
```

`register`와 `cancel`은 이 helper를 호출한 뒤 각자의 규칙을 처리한다. 참가자 ID 검사와 확정·대기 명단에 누구를 넣을지 결정하는 코드는 각 함수에 남는다. 공통 준비 코드를 한곳으로 옮겨도 자리를 받는 사람은 달라지지 않는다. [Fowler, *Extract Function*](https://refactoring.com/catalog/extractFunction.html).

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-04-refactor.svg" alt="같은 입력과 결과를 유지하면서 내부의 중복된 준비 코드를 역할에 맞게 정리한다." width="960" height="810">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-04-refactor-mobile.svg" alt="공개된 contract와 내부 구조 변경을 구별하는 모바일 그림." width="480" height="1300">
</div>
*Figure 4. 두 함수에서 반복하던 상태 검사와 두 list의 복사를 공통 helper인 `copy_state`로 옮긴다. 참가자 ID 검사와 신청·취소 규칙은 각 함수에 남는다. 연결선은 추출한 코드와 호출의 관계를 나타낸다.*

조건문의 중첩을 줄이는 것도 refactoring의 한 예다. 이미 신청한 ID를 다시 받은 경우처럼 추가할 일이 없으면 바로 반환한다. 그러면 이후 코드에서는 나머지 경우만 다루면 된다. 이 패턴이 **guard clause**다. 아래 실습은 Figure 4의 함수 추출과는 별도로, 이 조건문 정리를 비교한다.

{% include interactive/article-lab.html id="cleancode-refactor" %}

입력을 “정원이 차면 B는 대기”로 두고 비교해 보자. 중첩 조건문과 guard clause 구현은 모두 A를 확정 명단에 두고 B를 대기 명단에 넣는다. 경계 조건이 잘못된 버전을 고르면 B까지 확정한다. 바뀐 비교식이 정확히 정원이 찬 경우를 놓치기 때문이다. 결과가 다르면 반례가 된다. 준비한 입력에서 모두 같더라도 모든 허용 입력에서 같다는 증명은 아니다.

## 10. 테스트도 준비한 상황이 잘 보여야 한다

준비 코드의 반복이 많아도 테스트의 의도를 읽기 어렵고, 준비 과정을 helper 안에 너무 많이 숨겨도 읽기 어렵다. **Fixture**는 pytest가 테스트에 필요한 데이터나 자원을 제공하는 장치다. 테스트 함수의 인자 이름으로 fixture를 요청하면, pytest가 그 fixture 함수를 실행하고 반환한 값을 넣어 준다.

```python
import pytest
from workshop import cancel


@pytest.fixture
def full_workshop():
    return {
        "capacity": 2,
        "confirmed": ["A", "B"],
        "waiting": ["C", "D"],
    }


def test_waiting_cancellation_keeps_confirmed(full_workshop):
    actual = cancel(full_workshop, "C")
    assert actual["confirmed"] == ["A", "B"]
    assert actual["waiting"] == ["D"]
```

`@pytest.fixture`는 **decorator**다. 바로 아래 함수에 표시나 동작을 추가하는 Python 문법이다. 여기서는 그 함수를 fixture로 등록한다. **Test suite**는 실행할 테스트들의 모음이다.

Fixture의 **scope**는 pytest가 반환된 값을 언제까지 재사용할지 정한다. 기본값인 `function` scope에서는 이 fixture를 요청하는 테스트마다 함수를 다시 호출한다. 위 함수는 호출할 때 dict와 list를 새로 만들므로 테스트끼리 데이터를 공유하지 않는다. 다만 scope가 객체를 자동으로 복사해 주는 것은 아니다. 전역 list를 반환하는 fixture라면 그 list는 여전히 공유된다. [pytest, *How to use fixtures*](https://docs.pytest.org/en/stable/how-to/fixtures.html).

**Parametrization**은 테스트 함수 하나를 여러 입력 사례로 실행하는 기능이다. 잘못된 정원을 검사하는 짧은 예를 보자.

```python
import pytest
from workshop import new_workshop


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        new_workshop(capacity)
```

`with pytest.raises(ValueError):`는 들여쓴 코드에서 `ValueError`가 발생해야 한다는 검사다. Exception이 발생하지 않거나 다른 종류가 발생하면 테스트가 실패한다. 입력 사례의 값은 자동으로 복사되지 않으므로, list나 dict를 바꾸는 테스트에서는 공유 여부에도 주의해야 한다. [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html), [exception 검사](https://docs.pytest.org/en/stable/getting-started.html#assert-that-a-certain-exception-is-raised).

**Isolation**은 어떤 테스트가 다른 테스트의 실행 순서나 남겨 둔 데이터에 의존하지 않는다는 뜻이다. 매번 새로 준비한 상태를 쓰고, 숨겨진 전역 list를 없애고, 임시 파일을 나누면 이를 지키기 쉽다. 테스트는 검사하려는 동작 때문에 실패해야 한다. 다른 테스트가 준비 데이터를 바꾸었기 때문에 실패한다면 원인을 파악하기가 어려워진다.

## 11. 규칙을 계산하는 코드와 파일 작업을 나누자

파일을 읽고 쓰는 일은 **I/O**, 즉 input/output이다. 함수의 인자와 반환값 외에 외부 자원을 사용한다. 파일이 없거나 권한이 없어서 실패할 수도 있다. C를 대기자로 넣을지 결정하는 데 파일 이름을 알 필요는 없다.

핵심 연산은 입력 상태를 바꾸거나 I/O를 하지 않고, 주어진 값으로 결과를 계산한다. 허용된 입력 범위에서는 이런 함수를 **pure function**이라고 한다. 같은 입력 값이면 같은 결과를 내고, 외부에서 관찰할 수 있는 side effect가 없다는 뜻이다. **Side effect**는 반환값 외의 관찰 가능한 변화다. 호출자가 가진 list를 바꾸거나, 출력하거나, 파일을 쓰는 일이 여기에 해당한다. 여기서는 pure function이 유용하지만, 모든 TDD가 pure function만 다루어야 하는 것은 아니다.

저장 코드를 담은 `storage.py`는 관련 정의들을 모은 Python 파일, 즉 **module**이다. 이 module은 상태를 **JSON**이라는 텍스트 형식으로 바꾸거나 다시 읽어 온다. Python의 `json` module은 인코딩과 디코딩을 처리한다. 그 결과가 워크숍의 contract까지 만족하는지는 `validate_state` 같은 입력 검사 함수가 확인해야 한다. `{"capacity": -3}`은 올바른 JSON이지만 유효한 워크숍 상태는 아니다. [Python JSON 문서](https://docs.python.org/3/library/json.html).

테스트에서는 실제 파일 접근을 작은 함수로 대신할 수도 있다. 이런 대역을 **test double**이라고 한다. **Stub**은 미리 준비한 값을 반환하고, **spy**는 전달받은 값을 기록한다. Figure 5는 이런 대역을 쓰는 검사와 실제 파일을 저장하고 읽는 검사를 비교한다. [Fowler, *Test Double*](https://martinfowler.com/bliki/TestDouble.html).

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-05-boundaries.svg" alt="상태 값으로 규칙을 계산하는 핵심 함수와 외부 파일 읽기·쓰기를 구분하고, integration test에서는 실제 임시 파일을 사용한다." width="960" height="850">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-05-boundaries-mobile.svg" alt="워크숍 규칙과 실제 파일 I/O의 경계를 나타낸 모바일 그림." width="480" height="1250">
</div>
*Figure 5. 핵심 규칙의 계산, 실제 JSON 파일을 저장하고 다시 읽는 검사, 파일 접근을 대역으로 바꾼 검사를 구별한다. 세 번째 경로에서 stub은 준비한 상태를 반환하고, spy는 저장 함수에 전달한 값을 기록한다. 이런 대역을 test double이라고 하며, 이 경로에서는 실제 파일에 접근하지 않는다.*

**Unit test**는 작게 정한 동작 단위를 집중해서 확인한다. 여기서는 파일 없이 워크숍 API를 호출할 수 있다. **Integration test**는 여러 부분이 함께 작동하는지 확인한다. 검사할 범위를 구분하는 이름이지, unit test가 반드시 함수 하나만 호출해야 한다거나 integration test가 원격 서비스에 접속해야 한다는 뜻은 아니다. [Fowler, *Unit Test*](https://martinfowler.com/bliki/UnitTest.html), [*Integration Test*](https://martinfowler.com/bliki/IntegrationTest.html).

다음 integration test에서는 실제 JSON 변환, 파일 쓰기·읽기, 다시 변환하기, 상태 검사를 함께 실행한다.

```python
from storage import load_workshop, save_workshop
from workshop import new_workshop, register


def test_workshop_survives_a_json_round_trip(tmp_path):
    path = tmp_path / "workshop.json"
    expected = register(new_workshop(2), "A")

    save_workshop(path, expected)
    actual = load_workshop(path)

    assert actual == expected
```

`tmp_path`는 이 테스트만을 위한 임시 폴더를 제공하는 기본 fixture다. `tmp_path / "workshop.json"`의 `/`는 임시 폴더 경로와 파일 이름을 연결하는 연산이다. 숫자를 나누는 연산이 아니다. 실제 참가자 파일을 덮어쓰지 않는다. 저장한 것을 다시 읽는 round trip은 유용하지만, 읽기와 쓰기에 서로 맞물리는 버그가 있으면 놓칠 수 있다. 프로젝트는 파일 내용과 잘못된 입력도 따로 검사한다. [pytest, 임시 폴더와 파일](https://docs.pytest.org/en/stable/how-to/tmp_path.html).

## 12. 외부 의존성을 바꿔 끼우는 이유

함수가 동작하는 데 필요한 다른 함수나 외부 자원을 **dependency**라고 한다. 예를 들어 시작 상태를 파일에서 읽어 오려면 파일을 읽는 함수가 필요하다. Python에서는 함수도 값처럼 다른 함수에 인자로 넘길 수 있다. 이렇게 필요한 기능을 밖에서 전달하는 방식을 **dependency injection**이라고 한다.

```python
from workshop import register


def register_with_storage(participant_id, load, save):
    updated = register(load(), participant_id)
    save(updated)
    return updated
```

호출하는 쪽은 `load`와 `save`에 파일 이름이 아니라 함수를 넘긴다. 위 코드에서는 `load()`로 시작 상태를 얻고, `save(updated)`에 계산된 상태를 전달한다. 신청 규칙은 `register`에 그대로 둔다. 전달하는 함수에 따라 실제 파일을 읽고 쓸 수도 있고, 테스트에서 준비한 값만 주고받을 수도 있다.

<details markdown="1">
<summary>일반 함수로 만드는 test double</summary>

```python
from workshop import new_workshop


def test_registration_can_be_checked_without_a_file():
    saved = []

    def load_known_state():
        return new_workshop(1)

    def record_save(state):
        saved.append(state)

    actual = register_with_storage("A", load_known_state, record_save)
    assert actual["confirmed"] == ["A"]
    assert saved == [actual]
```

Python에서는 함수 안에 다른 함수를 정의할 수 있다. `record_save`는 바깥 테스트의 `saved` list에 전달받은 상태를 기록한다. 이 테스트는 계산된 상태가 `save` 함수에 제대로 전달됐는지 확인한다.

위 코드에서 `load_known_state`가 stub이고, `record_save`가 spy다. **Fake**는 메모리 안에서만 동작하는 저장소처럼 단순하지만 실제로 작동하는 구현을 제공한다. **Mock**은 보통 어떤 함수가 어떤 인자로 호출돼야 하는지도 미리 정한다. Library마다 용어 사용에 차이가 있으므로 이름보다 실제로 대신하는 역할을 살펴보는 편이 중요하다. [Fowler, *Test Double*](https://martinfowler.com/bliki/TestDouble.html).

이 대역을 사용한 테스트가 통과해도 JSON 변환이나 파일 권한, 실제 저장까지 확인되는 것은 아니다. 따라서 규칙을 계산하는 테스트와 실제 파일 작업을 확인하는 테스트를 따로 둔다.

</details>

## 13. 테스트가 통과해도 놓칠 수 있는 버그

Assertion이 없는 테스트는 잘못된 함수도 실행만 하고 통과할 수 있다.

```python
def broken_total():
    return 99


def test_executes_but_checks_nothing():
    broken_total()
```

필요한 합계가 얼마인지 확인하는 assertion이 없다. **Code coverage**는 실행 중 코드의 어느 부분을 지났는지 나타낸다. Line coverage나 branch coverage처럼 무엇을 세는지에 따라 종류가 나뉜다. Line coverage는 실행한 줄을 세고, branch coverage는 조건의 각 결과처럼 갈라지는 경로를 확인한다. 어느 쪽도 assertion이 요구사항에 맞는지, 중요한 오답을 구별하는지는 알려 주지 않는다. Fowler도 coverage를 테스트하지 않은 부분을 찾는 도구로 설명하며, 테스트 품질을 보장하는 숫자로 보지 않는다. [*Test Coverage*](https://martinfowler.com/bliki/TestCoverage.html).

이 예제에서는 더 구체적으로 물어볼 수 있다. “이 테스트들이 특정 실수를 알아차릴까?” 정원 비교식을 바꾸거나, 중복 검사를 빼거나, 대기 명단의 반대쪽에서 승급하거나, 입력을 직접 바꾸는 구현을 일부러 만들어 보자. 이렇게 작은 결함을 넣은 버전을 **mutant**라고 한다. 테스트가 이런 변경을 알아차리는지 확인하는 방법이 **mutation testing**이다. 아래 실습은 목록에 적은 몇 가지 결함만 검사한다. 모든 종류의 mutation을 자동 생성하는 도구는 아니다. 실제 Python 도구와 사용 조건은 [mutmut GitHub](https://github.com/boxed/mutmut)에서 확인할 수 있다.

<div class="cleancode-test-matrix-wrapper" role="region" aria-labelledby="cleancode-matrix-title" tabindex="0">
<table id="cleancode-test-matrix">
<caption id="cleancode-matrix-title">각 테스트가 어떤 오류를 검출하는가?</caption>
<colgroup><col class="cc-matrix-test"><col><col><col><col></colgroup>
<thead><tr>
<th scope="col" aria-label="오류별 검사"><span class="cc-matrix-full">오류별 검사</span><span class="cc-matrix-brief" aria-hidden="true">검사</span></th>
<th scope="col" aria-label="잘못된 정원 비교"><span class="cc-matrix-full">잘못된 정원 비교</span><span class="cc-matrix-brief" aria-hidden="true">정원</span></th>
<th scope="col" aria-label="중복 신청 허용"><span class="cc-matrix-full">중복 신청 허용</span><span class="cc-matrix-brief" aria-hidden="true">중복</span></th>
<th scope="col" aria-label="반대쪽에서 승급"><span class="cc-matrix-full">반대쪽에서 승급</span><span class="cc-matrix-brief" aria-hidden="true">LIFO</span></th>
<th scope="col" aria-label="원본 상태 변경"><span class="cc-matrix-full">원본 상태 변경</span><span class="cc-matrix-brief" aria-hidden="true">입력</span></th>
</tr></thead>
<tbody>
<tr data-test="boundary"><th scope="row" aria-label="T1 · 정원이 찬 경우"><span class="cc-matrix-full">T1 · 정원이 찬 경우</span><span class="cc-matrix-brief" aria-hidden="true">T1</span></th>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">오류 검출</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
</tr>
<tr data-test="duplicate"><th scope="row" aria-label="T2 · 중복 ID"><span class="cc-matrix-full">T2 · 중복 ID</span><span class="cc-matrix-brief" aria-hidden="true">T2</span></th>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">오류 검출</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
</tr>
<tr data-test="fifo"><th scope="row" aria-label="T3 · FIFO 승급"><span class="cc-matrix-full">T3 · FIFO 승급</span><span class="cc-matrix-brief" aria-hidden="true">T3</span></th>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">오류 검출</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
</tr>
<tr data-test="nonmutation"><th scope="row" aria-label="T4 · 입력 보존"><span class="cc-matrix-full">T4 · 입력 보존</span><span class="cc-matrix-brief" aria-hidden="true">T4</span></th>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">이 테스트로는 검출하지 못함</span></td>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">오류 검출</span></td>
</tr>
</tbody>
</table>
</div>
<div class="cleancode-matrix-mobile-key">
<p><strong>T1</strong> 정원이 찬 경우</p>
<p><strong>T2</strong> 중복 ID</p>
<p><strong>T3</strong> FIFO 승급</p>
<p><strong>T4</strong> 입력 보존</p>
</div>
<p class="cleancode-matrix-legend"><strong>● 오류 검출:</strong> 해당 오류 구현에서 테스트가 실패한다. <strong>—</strong> 이 테스트만으로는 해당 오류를 검출하지 못한다.</p>

*Table 1. 각 셀은 아래 실습과 같은 테스트·오류 구현 조합을 실행한 결과다.*

{% include interactive/article-lab.html id="cleancode-tests" %}

빈자리에 신청하는 테스트 하나만 골라 보자. 어떤 결함이 남는가? 중복, 정확히 정원이 찬 경우, FIFO, 입력 보존 검사를 하나씩 추가해 보자. 검출한 개수뿐 아니라 어느 입력에서 결과가 달라지는지도 확인해야 한다. 목록의 모든 결함을 잡아도 목록 밖의 결함이나 잘못 정한 명세까지 해결되는 것은 아니다.

<details markdown="1">
<summary>구체적인 예와 함께 검사할 수 있는 property</summary>

**Property**는 여러 입력에서 공통으로 성립해야 하는 성질이다. 같은 ID를 두 번 신청해도 한 번 신청한 결과와 값이 같아야 한다는 것이 한 예다. 이 성질을 **idempotent**하다고 한다. 그렇다고 이 성질이나 인원수 제한만으로 FIFO 예제를 대신할 수는 없다.

ID를 몇 개로 제한해 가능한 유효 상태를 모두 나열하고, 각 연산 뒤 invariant가 유지되는지 검사할 수도 있다. 이 결과는 정해 둔 유한 입력 범위에 대한 것이다.

</details>

## 14. 테스트가 없는 기존 코드부터 고쳐야 한다면

기존 코드에는 테스트가 없는 경우도 많다. 구조를 바꾸기 전에 다른 코드에서 호출하는 함수 하나를 골라, 같은 결과를 재현할 수 있는 입력을 넣어 본다. 현재 결과를 관찰해 테스트에 기록하는 것이 **characterization test**다. 지금 어떻게 동작하는지 기록하는 것이지, 그 동작이 바람직하다고 자동으로 판단하는 것은 아니다. Alberto Savoia는 Michael Feathers가 제시한 이 개념을 [*Working Effectively With Characterization Tests*](https://www.artima.com/weblogs/viewpost.jsp?thread=198296)에서 직접 작성한 예제로 설명한다.

현재 취소 코드가 C보다 D를 먼저 승급한다고 하자. 처음 characterization test에는 그 사실을 기록해 구조 변경과 동작 변경을 따로 확인할 수 있다. 합의한 요구사항이 FIFO라면 기존의 LIFO assertion을 검토해 고치거나 제거하고, C를 기대하는 테스트에 맞춰 동작을 의도적으로 수정한다. 서로 모순되는 두 assertion을 그대로 두면 둘 다 통과시킬 수 없다. 버그까지 포함해 기존 동작을 유지하는 일과 버그를 수정하는 일은 다르다. 테스트 이름과 변경 설명에서도 구별해 두는 편이 좋다.

확인할 동작과 준비할 입력을 좁혀서 시작한다. 파일 접근 때문에 규칙만 검사하기 어렵다면 필요한 만큼 분리해 테스트할 수 있게 만든다. 모든 module을 다시 짜면서 업무 규칙까지 동시에 바꾸지는 않는다. 기존 코드를 test-first로 만들었다고 할 수는 없지만, 동작을 유지하는 작은 변경은 지금부터 적용할 수 있다. [Fowler, *Refactoring*](https://refactoring.com/).

<details markdown="1">
<summary>도구를 도입하고 함수를 나누는 것만으로는 해결되지 않는 문제</summary>

Formatter는 들여쓰기를 일정하게 만들 수 있지만 어떤 취소 규칙이 필요한지는 정하지 못한다. **Type hint**는 `participant_id: str`처럼 기대하는 자료형을 적는 표시다. 이것만으로 실행 중의 값이 자동 검사되지는 않는다. [Python typing 문서](https://docs.python.org/3/library/typing.html)도 이 점을 명시한다. 함수 하나를 여러 개로 나누어도 새 이름들이 의미를 설명하지 못하면 오히려 여기저기 찾아다녀야 한다.

List에서 ID를 찾으려면 최악의 경우 모든 참가자를 확인해야 하고, list 전체를 복사할 때도 모든 항목을 거친다. 이 작은 순차 실행 프로그램에서는 이런 작업량이 참가자 수에 비례해 커질 수 있다. Helper를 더 만든다고 이 증가율이 좋아지는 것은 아니다. 명단이 매우 커지거나, 요청을 동시에 처리하거나, 프로그램이 중단돼도 저장한 내용을 잃지 않아야 한다면 그 요구에 맞춰 자료구조와 설계를 바꿀 수 있다. 필요하기 전에 모두 넣으면 이해해야 할 contract부터 늘어난다.

</details>

## 15. 실패를 읽고 작은 기능을 추가해 보자

{% include interactive/article-lab.html id="cleancode-quiz" %}

문제에는 실제 상태, 코드, 실패 메시지가 주어진다. 각 보기마다 해설이 있다. 보기를 선택하기 전에 어떤 결과를 확인하면 정답과 오답을 구별할 수 있을지 생각해 보자.

### Coding task: 정원을 늘리면 대기자를 순서대로 승급시키기

`exercise.py`의 `increase_capacity(state, new_capacity)`를 구현하자. 정원을 바꾸는 기능은 기존 API에 없으므로, 이 과제는 refactoring이 아니라 기능 추가다.

- `new_capacity`는 `bool`을 제외한 양의 정수여야 한다.
- 현재보다 작으면 `ValueError`를 발생시키고, 같으면 상태의 값은 바꾸지 않는다.
- 커졌다면 새로 비는 자리만큼 대기자를 FIFO 순서로 승급한다.
- 기존 확정 명단의 순서는 유지한다. 승급하지 않은 대기자의 순서도 유지한다.
- 인자로 받은 상태와 그 안의 list는 바꾸지 않는다.
- 정원이 그대로여도 새 dict와 독립적인 `confirmed`, `waiting` list를 반환한다.
- 입력 상태는 2절의 contract를 만족해야 한다. 잘못된 상태를 받으면 `ValueError`를 발생시킨다.

정원이 두 명이고 확정 명단이 `["A", "B"]`, 대기 명단이 `["C", "D"]`일 때 정원을 세 명으로 늘리면 확정 명단은 `["A", "B", "C"]`, 대기 명단은 `["D"]`가 된다. 다섯 명으로 늘리면 대기자 둘을 모두 승급하고 한 자리는 비어 있게 된다.

구현 전에 확인할 경우를 적어 보자. 같은 정원, 한 자리 추가, 여러 자리 추가, 대기자보다 많은 자리, 대기자가 없는 경우, 잘못된 입력, 입력 보존이 있다. 그중 하나를 골라 테스트를 쓰고 실패를 확인한 뒤, 기존 테스트까지 통과하도록 구현한다.

```bash
python -m pytest -q exercise_tests
```

스타터는 일부러 `NotImplementedError`를 발생시킨다. 정답은 별도 파일인 `exercise_solution.py`에 있으며 이를 import해도 작성 중인 함수를 바꾸지 않는다. README에 정답 검사 방법과 공개 테스트의 실행 방법이 있다. 특정 helper 이름이나 알고리즘을 써야 한다는 숨겨진 조건은 없다.

<details markdown="1">
<summary>정답을 열기 전에 볼 수 있는 힌트</summary>

새 정원에서 빈자리가 몇 개인지 먼저 계산한다. 빈자리 수와 대기자 수 중 작은 값만큼 대기 명단 앞에서 ID를 가져와 확정 명단의 복사본에 붙이고, 나머지를 새 대기 명단으로 남긴다. Python slice를 쓰면 앞부분과 나머지를 분명히 나눌 수 있다.

승급 인원수만 확인하면 가져온 순서가 뒤집혀도 놓칠 수 있다. 반환한 상태만 확인하면 입력을 바꾼 버그를 놓칠 수 있다. 테스트에서 둘 다 구별할 수 있어야 한다. 테스트를 통과한 뒤에는 승급할 명단 부분에 이름을 붙이는 것이 읽는 데 도움이 되는지 살펴보자.

[참고 구현]({{ site.baseurl }}/assets/code/clean-code-tdd/exercise_solution.py) · [실습 시작 코드]({{ site.baseurl }}/assets/code/clean-code-tdd/exercise.py)

</details>

## 16. Clean Code와 TDD의 효과와 한계

Interface가 분명하면 함수를 사용하는 쪽에서 알아야 할 내부 정보가 줄어든다. 테스트는 요구사항의 구체적인 예를 실행 가능한 형태로 남긴다. 작은 refactoring은 구조 변경을 검토하기 쉽게 만든다. TDD는 다음에 필요한 동작을 정하고 짧은 주기로 결과를 확인하며 이 작업들을 연결한다.

그렇다고 명세의 정확성, 모든 중요한 동작에 대한 검사, 언제나 좋은 설계, 모든 프로젝트에서의 개발 속도 향상이 보장되는 것은 아니다. 테스트도 유지해야 하는 코드다. 사소한 내부 구현에 묶인 테스트는 동작에 영향이 없는 수정도 어렵게 만들 수 있다. 느리거나 결과가 일정하지 않은 테스트도 확인을 어렵게 한다. **Flaky test**는 검사하려는 동작이 바뀌지 않았는데도 결과가 달라지는 테스트다. 공유된 mutable state나 통제하지 않은 실행 시간에 의존하는 경우가 그 예다. [pytest의 flaky test 설명](https://docs.pytest.org/en/stable/explanation/flaky.html)에서 이런 원인을 다룬다.

성능, 접근성, 보안, 동시 실행은 각각의 요구사항과 적절한 검사가 필요하다. 대기 순서 테스트를 통과했다는 사실이 이런 속성까지 검증했다는 근거는 아니다.

다음에 필요한 동작을 정하고, 실수를 구별할 예를 고르고, 실패를 읽고, 코드를 바꾼다. 그다음 수정할 사람도 이해할 수 있도록 코드를 정리한다.

## 참고문헌과 실행 자료

### 개념과 원문 인용

- [Kent Beck, *Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd) — 테스트 목록, 다음 테스트 하나, 구현 수정, 성급한 추상화를 피하는 판단. 8절의 짧은 인용은 원문 그대로다.
- [Martin Fowler, *Test Driven Development*](https://martinfowler.com/bliki/TestDrivenDevelopment.html) — Red–Green–Refactor와 interface를 먼저 생각하는 방식.
- [Martin Fowler, *Refactoring*](https://refactoring.com/) — 동작을 유지하면서 구조를 바꾸는 작업. 9절의 짧은 표현은 원문 그대로 인용했다.
- [Refactoring catalog: Extract Function](https://refactoring.com/catalog/extractFunction.html), [Guard Clauses](https://refactoring.com/catalog/replaceNestedConditionalWithGuardClauses.html) — 구체적인 변경 방법. 모든 코드에 강제할 점수표는 아니다.
- [Tim Peters, PEP 20](https://peps.python.org/pep-0020/), [PEP 8](https://peps.python.org/pep-0008/) — Python의 설계 지침, 이름, 주석. 도입부의 두 단어 인용도 원문 그대로다.
- [Google Engineering Practices, code review 지침](https://google.github.io/eng-practices/review/reviewer/looking-for.html) — 복잡성, 기능 검토, 아직 필요하지 않은 일반화.
- [Martin Fowler, *Test Coverage*](https://martinfowler.com/bliki/TestCoverage.html), [*Test Double*](https://martinfowler.com/bliki/TestDouble.html) — coverage 해석과 테스트 대역의 구분.
- [Martin Fowler, *Unit Test*](https://martinfowler.com/bliki/UnitTest.html), [*Integration Test*](https://martinfowler.com/bliki/IntegrationTest.html) — 작은 동작과 여러 부분의 연동을 검사하는 방법. 검사할 범위를 함수 개수 하나로 정하지는 않는다.

### 문서, GitHub, 예제 파일

- [pytest: Get Started](https://docs.pytest.org/en/stable/getting-started.html), [fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html), [parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html), [임시 파일](https://docs.pytest.org/en/stable/how-to/tmp_path.html).
- [pytest GitHub](https://github.com/pytest-dev/pytest) — runner의 구현, 테스트, 프로젝트 문서.
- [mutmut GitHub](https://github.com/boxed/mutmut) — Python mutation 도구. [Alberto Savoia의 characterization test 예제](https://www.artima.com/weblogs/viewpost.jsp?thread=198296) — 현재 동작을 관찰해 기록하는 방법.
- [Python typing 문서](https://docs.python.org/3/library/typing.html), [pytest flaky test 설명](https://docs.pytest.org/en/stable/explanation/flaky.html) — type annotation의 runtime 한계와 통제하지 않은 테스트 상태.
- [Python: copying](https://docs.python.org/3/library/copy.html), [JSON](https://docs.python.org/3/library/json.html) — 예제에서 사용하는 언어·library의 동작.
- [Python: assert 문](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) — 조건 검사와 `AssertionError`. 일반 `assert`를 생략할 수 있는 `python -O` 옵션은 이 검사에 사용하지 않는다.
- [전체 실습 프로젝트 ZIP]({{ site.baseurl }}/assets/code/clean-code-tdd-workshop.zip), [workshop.py]({{ site.baseurl }}/assets/code/clean-code-tdd/workshop.py), [README]({{ site.baseurl }}/assets/code/clean-code-tdd/README.md).
- [Python 개발 단계의 실제 실행 기록]({{ site.baseurl }}/assets/data/cleancode/tdd-runs.json), [구현 간 비교에 사용한 사례]({{ site.baseurl }}/assets/data/cleancode/workshop-fixtures.json).
