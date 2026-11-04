---
title: "Clean Code and TDD: Writing Code You Can Understand and Change"
description: "A Python workshop registration example uses clear names and small refactorings while testing capacity limits, queue order, and input preservation."
date: 2026-11-03 21:00:00 +0900
categories: [Software Engineering]
tags: [software-engineering, clean-code, python, refactoring, software-testing, tdd]
math: true
interactive_labs: [cleancode]
pin: false
topic: software-engineering
lang: en
translation_key: clean-code-and-tdd
image:
  path: /assets/img/posts/2026-11-04-clean-code-tdd/cover-flat.png
  alt: "Capacity one, A confirmed and B waiting: the actual confirmed list changes from [A] to [B], then stays [B] after refactoring; the expected cancellation result remains [B]."
  width: 1672
  height: 941
  fit: contain
  hide_caption: true
---

[Python + pytest project ZIP]({{ site.baseurl }}/assets/code/clean-code-tdd-workshop.zip) · [Core implementation]({{ site.baseurl }}/assets/code/clean-code-tdd/workshop.py) · [Recorded test runs]({{ site.baseurl }}/assets/data/cleancode/tdd-runs.json)

A program can produce the right answer today and still be difficult to change tomorrow. Suppose a workshop has two seats. A and B are confirmed; C and D are waiting. A cancels. The next state should confirm B and C, with D still waiting. Where is that rule in the code? What else might a change to it affect? How can we check?

**Clean code** is used here as a practical goal: code whose behavior, responsibilities, and reasons for change are understandable. **Test-driven development**, or **TDD**, is a development workflow: choose a small behavior, write an executable test, make it pass, and consider improving the structure while retaining the tested behavior. These are related ideas, not synonyms. [Kent Beck, *Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd), [Martin Fowler, *Test Driven Development*](https://martinfowler.com/bliki/TestDrivenDevelopment.html).

Tim Peters's *Zen of Python* puts one useful design preference in two words:

> Readability counts.

— [PEP 20](https://peps.python.org/pep-0020/). A preference still needs judgment. A shorter expression can hide a policy; a longer name can reveal one. Tests can help check changes, but they cannot decide every design question for us.

## 1. Start with a change, not a rule about line counts

Consider the request: “When a confirmed participant cancels, offer the seat to the first person waiting.” A function that mixes participant lookup, list changes, file writes, and printing gives us several things to reason about at once. A function named `process(data, x, flag)` also forces us to discover what its arguments mean by reading its body.

The useful questions are concrete:

- Can we locate the cancellation policy without reading file-handling code?
- Is the order of the waiting list part of the result?
- Does calling the function unexpectedly change data the caller still uses?
- Can a test distinguish the first waiting person from the last?

Names, decomposition, and tests are valuable when they answer these questions. They are not ends in themselves. Google's code-review guidance similarly treats unnecessary complexity and speculative generality as review concerns, rather than prescribing one universal function size. [Google Engineering Practices, *What to look for in a code review*](https://google.github.io/eng-practices/review/reviewer/looking-for.html).

The examples need functions, conditions, lists, and dictionaries. A function's **interface** means how another piece of code uses it: its name, arguments, result, and documented errors. Its **implementation** is the code that produces that behavior. Choosing `cancel(state, "A")` as an interface does not yet decide whether its internals use a loop or call another supporting function, a **helper**. The functions intended for other code to call form the program's **API**, its exposed programming interface; an API need not be a web service.

## 2. Give the workshop an exact contract

A **contract** states what inputs an operation accepts and what the caller may expect. Our state is a dictionary with exactly three fields:

```python
state = {
    "capacity": 2,
    "confirmed": ["A", "B"],
    "waiting": ["C", "D"],
}
```

`capacity` is a positive integer. `confirmed` and `waiting` are ordered lists of participant IDs. An ID is an exact, nonempty string; `"A"` and `"a"` are different IDs. We do not trim spaces or implement a user-account naming policy.

The rules are:

1. Registration with a free seat appends the ID to `confirmed`.
2. Registration when full appends the ID to `waiting`.
3. An ID already in either list is not added again.
4. Cancelling a confirmed ID removes it and promotes the first waiting ID, if one exists.
5. Cancelling a waiting ID removes only that ID. Cancelling an unknown ID changes nothing.
6. Operations leave the supplied dictionary and its lists unchanged, and return a new dictionary with independent lists—even when the values do not change.

Keeping the old state intact lets us compare before and after. Independent returned lists also prevent a later edit through the result from changing the old snapshot.

**FIFO** means *first in, first out*: C was waiting before D, so C is promoted first. We preserve the relative order of everyone else. The state returned by cancelling A is:

```python
expected = {
    "capacity": 2,
    "confirmed": ["B", "C"],
    "waiting": ["D"],
}
```

An **invariant** is a condition that must hold before and after every valid operation. Here, confirmed attendance never exceeds capacity, IDs never occur twice across the two lists, and a nonempty waiting list implies that the confirmed list is full. These conditions constrain states; they do not by themselves prove FIFO order. A program can preserve all three while promoting D incorrectly.

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-01-state.svg" alt="Cancellation removes A, keeps B, promotes C from the head of the waiting queue, and leaves D waiting." width="960" height="670">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-01-state-mobile.svg" alt="Cancellation and FIFO promotion, arranged vertically for mobile." width="480" height="1000">
</div>
*Figure 1. Capacity is two. Removing A compacts B toward the front; C leaves the head of the waiting list and is appended after B. D remains waiting. Numbers denote list indices, not physical seat numbers.*

{% include interactive/article-lab.html id="cleancode-state" %}

Register A, B, C, and D, then cancel A. Predict both lists before inspecting the result. Next cancel someone waiting, and try registering an existing ID. The unchanged cases are part of the contract, not missing features.

<details markdown="1">
<summary>Input validation and the limits of this model</summary>

The downloadable implementation checks the exact state shape, integer capacity, list contents, uniqueness, and occupancy invariants. `True` and `False` are Python's `bool` values for truth and falsehood. Python treats them like 1 and 0 in many integer operations, but this contract rejects them as capacities. Invalid state or ID input raises `ValueError`, an exception reporting an unacceptable value. A whitespace-only string is nonempty and therefore allowed by this deliberately small ID policy.

Capacity remains fixed until the final exercise. One running program handles requests one at a time. File persistence does not add concurrent booking safety, authentication, or crash-safe writes. Those would require additional requirements and tests; nothing in the small state model establishes them.

</details>

## 3. An assertion compares an answer with an independent expectation

A **test** is executable code that sets up a situation, performs an operation, and checks an expectation. An **assertion** is the check itself.

`from workshop import cancel` makes the `cancel` function from `workshop.py` available to this test.

In the normal Python mode used here, `assert condition` continues when the condition is true and raises `AssertionError` when it is false; pytest reports that exception as a test failure. [Python, *The assert statement*](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).

The `==` operator compares dictionary keys and values; dictionary key order does not matter. For list values such as `confirmed`, it compares both the items and their order. Thus `["B", "C"]` and `["C", "B"]` are different results. [Python, *Built-in Types*](https://docs.python.org/3/library/stdtypes.html).

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

The first assertion checks the result. The second checks that the caller's state was not changed: returning the right answer does not by itself prevent an implementation from changing its input too.

This three-part arrangement is often called **Arrange–Act–Assert**: prepare `before`, call `cancel`, then check.

The expected state comes from the workshop rule, not a second call to `cancel`. Writing `expected = cancel(before, "A")` and comparing another call with it could reproduce the same mistake twice. A **test oracle** is whatever tells a test the expected answer. Here it is the explicitly written state justified by the FIFO rule. Beck specifically warns against copying computed results into expected values merely to make tests pass. [*Canon TDD*, “Make it Pass”](https://newsletter.kentbeck.com/p/canon-tdd).

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-02-test.svg" alt="An independently specified FIFO expectation is compared with a deliberately faulty cancellation result; C and D expose the mismatch." width="960" height="810">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-02-test-mobile.svg" alt="The same cancellation input leads to independent FIFO expected values and a deliberately faulty result, with the mismatched IDs highlighted." width="480" height="1060">
</div>
*Figure 2. The expected value is fixed independently by the FIFO rule. A deliberately faulty implementation promotes D instead of C. Both results confirm two people, but their identities and order differ, so the assertion fails.*

## 4. Run a real test with pytest

**pytest** is a Python test runner: it discovers test functions, executes them, and reports failures. In the downloaded project, open a terminal in the directory containing `workshop.py` and `pytest.ini`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The second command activates the environment on macOS and Linux. In Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`. A **virtual environment** keeps a project's installed packages separate from other Python projects. [Python, `venv`](https://docs.python.org/3/library/venv.html#how-venvs-work).

Running `python -m pytest` uses the selected Python interpreter to run pytest. The workshop project pins pytest 8.4.2.

These commands run the completed reference implementation, so a successful installation should produce a passing run. The step-by-step TDD records in section 5 instead show incomplete implementations and the failures that guide their next change.

Save the preceding test as `tests/test_walkthrough.py` to run it separately with `python -m pytest -q tests/test_walkthrough.py`. Names such as `test_*.py` and `test_*` are part of pytest's discovery conventions. A leading `F` means failure; `.` means pass. The progress marker `[100%]` means that execution finished, not that all tests passed. [pytest, *Get Started*](https://docs.pytest.org/en/stable/getting-started.html).

A failure report compares actual and expected values and points to the assertion. Ask whether that is the intended failure. An assertion showing `["B", "D"]` instead of `["B", "C"]` exposes the promotion rule. A missing pytest installation exposes a setup problem. The red color alone does not explain which.

<details markdown="1">
<summary>Project layout, expected failures, and useful commands</summary>

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

The separate `exercise_tests` directory targets unfinished student code. Run it explicitly when starting section 15.

Use `python -m pytest -q tests/test_walkthrough.py -x` to stop after that file's first failure. Use `python -m pytest --version` to inspect the installed runner. Keep your working file and the reference solution separate, so reading or running the solution cannot silently overwrite your attempt.

</details>

## 5. TDD chooses one next behavior

**Red–Green–Refactor** names a small development cycle.

1. **Red.** The next test fails because the requested behavior is absent or wrong.
2. **Green.** Change the implementation until that test and the existing tests pass.
3. **Refactor.** Improve the structure where it helps while preserving behavior.

Start with a list of scenarios, not a large pile of speculative test code: first registration, another registration, duplicate registration, full capacity, waiting order, cancellation, promotion, missing ID. Turn one item into a runnable test, complete its cycle, then select the next. Newly noticed cases can join the list. That distinction between a test list and one executable next test is explicit in Beck's description; Fowler connects test-first work with thinking about the interface before its internals. [Beck, *Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd), [Fowler, *Test Driven Development*](https://martinfowler.com/bliki/TestDrivenDevelopment.html).

**Green means that the current tests pass.** It does not mean that every scenario on the list is implemented. Nor is changing expected values to agree with a bug a legitimate way of completing the cycle.

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-03-cycle.svg" alt="A concrete failing assertion, a behavior change that makes tests pass, and a structure-only change checked against the same tests." width="960" height="730">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-03-cycle-mobile.svg" alt="Concrete Red Green Refactor development stages in mobile layout." width="480" height="960">
</div>
*Figure 3. With capacity one, A confirmed and B waiting, cancelling A should return confirmed `[B]`. The output progresses `[A] → [B] → [B]`; counts describe the full saved test runs. The gray return selects the next behavior.*

{% include interactive/article-lab.html id="cleancode-tdd" %}

The stepper contains saved executions of actual Python snapshots. Each Red checkpoint adds one test; the next Green checkpoint implements that behavior. Waiting-list cancellation is added before confirmed cancellation with FIFO promotion. The final Refactor checkpoint retains the same five tests as the preceding Green checkpoint.

Compare the new test, implementation, and reported failure. The recorded snapshots assume valid inputs; the complete `workshop.py` file adds the contract's input validation. The page reads those records rather than running Python or pytest in your browser. The [downloadable records]({{ site.baseurl }}/assets/data/cleancode/tdd-runs.json) retain the command, version, outcome, and code used.

## 6. Build registration without jumping to the final design

The first behavior is small: an unregistered participant can take an available seat. `new_workshop(2)` creates a new state with capacity two and empty `confirmed` and `waiting` lists. The test asks registration to add A to `confirmed` and leave `waiting` empty.

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

A first implementation might append to `confirmed` for every call. It can satisfy this test and still be wrong for duplicates and full workshops. The next useful test is not another spelling of the same case:

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

The two duplicate cases check confirmed and waiting IDs separately, motivating a check of both lists before adding an ID. The full case motivates deciding which list receives it. The **boundary** is equality with capacity: with one confirmed participant and capacity one, the next participant must wait. Merely testing a comfortably empty workshop does not distinguish `<` from `<=` in that decision.

These tests do not force a particular internal arrangement. The interface can remain two ordinary arguments. As new cases reveal responsibilities, we can reorganize the implementation without adding machinery that no current requirement needs.

<details markdown="1">
<summary>The registration policy, with input validation kept explicit</summary>

The following worked version uses the package's validation functions but spells out the policy. The complete package also documents its error contract.

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

The early return handles the no-change case before the seat decision. It is called a **guard clause**: a condition that exits early so the remaining path need not be nested inside its opposite. The pattern is useful when it makes the cases clearer; replacing every conditional with an early return is not a correctness rule. [Fowler, *Replace Nested Conditional with Guard Clauses*](https://refactoring.com/catalog/replaceNestedConditionalWithGuardClauses.html).

</details>

## 7. Cancellation tests need more than an attendance count

Suppose cancellation returns two confirmed participants and one waiting participant. That does not tell us *which* participant was promoted. The assertion in section 3 deliberately uses two waiting IDs so FIFO and LIFO produce different answers.

**LIFO**, *last in, first out*, would promote D first. It is a valid policy in other contexts, but violates this contract. Tests should express the requirement, not assume that every familiar data-structure policy is interchangeable.

Different cancellation cases deserve different expectations: confirmed with a queue, confirmed without a queue, waiting, and unknown. For a waiting cancellation, confirmed attendance must not change. For an unknown ID, both lists keep their values and ordering. The tests check these publicly visible outcomes rather than insisting that an internal helper be called a particular number of times.

<details markdown="1">
<summary>Python's shallow-copy trap, with a concrete counterexample</summary>

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

`dict(before)` creates another outer dictionary, but the nested list is still shared. **Aliasing** means that two references lead to the same object. Appending through either reference changes that object. Python assignment itself does not copy objects. [Python, *copy — Shallow and deep copy operations*](https://docs.python.org/3/library/copy.html).

Copying each list, as `register_worked` does, is sufficient for this state shape because its elements are immutable strings. It is not a general deep-copy recipe for arbitrarily nested structures. A **mutable** object can be changed in place; lists are mutable. A string's contents cannot be changed in place, so strings are **immutable**.

Avoiding input mutation is our API choice. TDD can also be used to build an explicitly mutating API; its tests would need to check that different contract.

</details>

## 8. Clean names and useful boundaries

A good name reduces the amount of code a reader must inspect before understanding its role. `participant_id` distinguishes an identifier from a display name. `confirmed` and `waiting` distinguish two policies that `list1` and `list2` conceal. PEP 8's guidance on public names favors usage rather than implementation details. [PEP 8, *Naming conventions*](https://peps.python.org/pep-0008/#naming-conventions).

**Cohesion** describes how closely the responsibilities inside a piece of code belong together. Registration checks and list placement belong to the same workshop decision; formatting terminal messages has a different purpose. **Coupling** concerns dependencies between pieces: when a policy function directly opens a specific file, a policy test must also deal with that file.

Do not respond by extracting every expression into a function. `is_registered(state, participant_id)` may give a recurring domain question a useful name. A helper named `do_part_two` adds navigation without explaining a responsibility. Fowler's *Extract Function* gives the transformation; the decision about where it helps still depends on the example. [Refactoring catalog](https://refactoring.com/catalog/extractFunction.html).

Comments should record information the code cannot easily express: a required ordering policy, a compatibility constraint, or why a seemingly tempting change is wrong. A comment saying “increment x” above `x += 1` adds little. A comment saying “promote the earliest waiting ID; order is contractual” explains a reason. Keep comments synchronized with code. [PEP 8, *Comments*](https://peps.python.org/pep-0008/#comments).

Beck gives a useful warning about extracting common code too soon:

> Duplication is a hint, not a command.

— [*Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd). Two similar expressions can implement different policies that will change independently. Conversely, repeating the same workshop policy in several functions gives a future change several places to become inconsistent. The reason for sharing matters more than visual similarity.

## 9. Refactoring keeps the agreed behavior

Fowler's definition includes the phrase:

> without changing its observable behavior.

— [*Refactoring*](https://refactoring.com/).

For our API, observable behavior includes the result, list order, documented errors, and whether caller-owned input changes. Renaming a local variable can preserve that behavior. Changing promotion from the queue's front to its back does not, even if the new code is shorter.

A practical sequence is: make the tests pass; change one structural detail; rerun them; inspect the diff, meaning the lines changed between versions. A **private helper** is an internal supporting function, not one of the documented entry points callers rely on. Python commonly marks such names with a leading underscore; this convention does not prevent access. If a test depends on that helper's old name, it may resist a legitimate refactoring. If the public rule changes, update the specification and its tests consciously; call that a feature change or bug fix, not merely refactoring.

The repeated preparation is a concrete responsibility: check the state, then make independent copies of its two lists. The project names that operation `copy_state`:

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

Both `register` and `cancel` call this helper before applying their own rules. The participant-ID check and the decision about who is confirmed remain in `register` and `cancel`. Extracting the helper changes where shared preparation lives, not which participant gets the seat. [Fowler, *Extract Function*](https://refactoring.com/catalog/extractFunction.html).

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-04-refactor.svg" alt="Two internal arrangements retain the same workshop input and result while separating named responsibilities." width="960" height="810">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-04-refactor-mobile.svg" alt="Same public contract and reorganized internals, mobile layout." width="480" height="1300">
</div>
*Figure 4. Repeated state validation and the copying of both lists move into the shared `copy_state` helper. Participant-ID validation and registration/cancellation policies stay in their respective functions. Connections identify extracted code and helper calls.*

A different refactoring reduces nested conditions. A **guard clause** returns early once a case is settled—for example, when an ID is already registered—so the following code handles only the remaining cases. The next lab compares that change, rather than the helper extraction shown in Figure 4.

{% include interactive/article-lab.html id="cleancode-refactor" %}

Keep the full-capacity input selected. Both the nested and guard-clause versions keep A confirmed and put B on the waiting list. Select the wrong-boundary version: it confirms B too, because the changed comparison does not recognize an exactly full workshop. A different result is a counterexample; agreement on the supplied inputs is a finite check, not a proof for every permitted input.

## 10. Test code should reveal its setup

Repeated setup can hide a test's purpose, but excessive setup helpers can hide it too. A **fixture** is pytest's mechanism for supplying test data or resources. A function parameter requests a fixture by name; pytest runs the fixture function and supplies its return value.

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

The `@pytest.fixture` line is a **decorator**: syntax that marks or wraps the function below it. Here it registers the function as a fixture. A **test suite** is the collection of tests being run.

A fixture's **scope** controls how long pytest reuses its result. With the default `function` scope, pytest calls this fixture separately for each requesting test. Here the function constructs a new dictionary and new lists each time, so tests do not share them. Scope does not copy objects: a fixture that returns a shared global list would still share that list. [pytest, *How to use fixtures*](https://docs.pytest.org/en/stable/how-to/fixtures.html).

**Parametrization** runs one test function with several supplied cases. A small invalid-capacity example keeps each input visible:

```python
import pytest
from workshop import new_workshop


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        new_workshop(capacity)
```

`with pytest.raises(ValueError):` tells pytest to expect that exception from the indented code. If no exception occurs, or a different kind occurs, the test fails. Parameter values are not automatically copied, so list or dictionary cases still need care when a test changes them. [pytest parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html), [exception assertions](https://docs.pytest.org/en/stable/getting-started.html#assert-that-a-certain-exception-is-raised).

**Isolation** means that one test does not depend on another having run first or on leftovers it leaves behind. Fresh setup, no hidden global list, and separate temporary files make that easier. A test should fail because of the behavior it checks, not because another test happened to mutate its preparation.

## 11. Keep decisions separate from file operations

Reading or writing a file is **I/O**, input/output. It reaches outside the function's ordinary arguments and returned value. It can fail because a path is missing or permissions deny access. The registration rule does not need to know a filename to decide whether C waits.

Our core operations compute a result from the supplied state without changing it or performing I/O. That makes them examples of **pure functions** on the accepted inputs: the same input values determine the same result, without externally visible side effects. A **side effect** is an observable change outside a returned value, such as modifying a caller's list, printing, or writing a file. Purity is useful here; it is not a prerequisite for all TDD.

The storage file, `storage.py`, is a **module**: a Python file holding related definitions. It converts state to and from **JSON**, a text format for exchanging structured values. Python's `json` module handles encoding and decoding; the input-checking function `validate_state` separately checks that decoded data satisfies our contract. Valid JSON such as `{"capacity": -3}` is not a valid workshop. [Python JSON documentation](https://docs.python.org/3/library/json.html).

We can also replace file access with small functions written for a test. Such a replacement is a **test double**. A **stub** supplies a prepared answer, while a **spy** records what it receives. Figure 5 contrasts these replacements with an actual file write and read. [Fowler, *Test Double*](https://martinfowler.com/bliki/TestDouble.html).

<div class="cleancode-figure-desktop">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-05-boundaries.svg" alt="Workshop decisions use state values; file loading and saving remain outside, with a real temporary file used by integration tests." width="960" height="850">
</div>
<div class="cleancode-figure-mobile">
<img src="{{ site.baseurl }}/assets/img/posts/2026-11-04-clean-code-tdd/fig-05-boundaries-mobile.svg" alt="Policy and real file I/O boundaries, mobile layout." width="480" height="1250">
</div>
*Figure 5. Distinguish the pure rule calculation, a real JSON-file round trip, and a test that replaces file access. In the third route, a stub returns a known state and a spy records the value passed for saving. Both are test doubles; neither performs real file I/O here.*

A **unit test** focuses on a small chosen unit of behavior; ours can call the workshop API without files. An **integration test** checks cooperating pieces together. These names describe the scope of the check, not a requirement that a unit be exactly one function or that integration involve a remote service. [Fowler, *Unit Test*](https://martinfowler.com/bliki/UnitTest.html), [*Integration Test*](https://martinfowler.com/bliki/IntegrationTest.html).

The following integration test checks the real encoder, file write, file read, decoder, and validation:

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

`tmp_path` is a built-in fixture supplying a temporary directory unique to this test. In `tmp_path / "workshop.json"`, `/` joins the directory path and filename; with these path objects it is not numeric division. No real participant file is overwritten. A successful round trip is useful, but can miss matching bugs in reader and writer; the project also checks file contents and malformed input separately. [pytest, temporary directories](https://docs.pytest.org/en/stable/how-to/tmp_path.html).

## 12. Replace an external dependency only when that helps the test

A **dependency** is another function or external resource that code needs to do its work. For example, loading the starting state requires a file-reading function. Python allows a function itself to be passed as an argument. Supplying the needed operations from outside is **dependency injection**:

```python
from workshop import register


def register_with_storage(participant_id, load, save):
    updated = register(load(), participant_id)
    save(updated)
    return updated
```

The caller supplies `load` and `save` as functions, not filenames. Inside this function, `load()` obtains the starting state and `save(updated)` receives the result. The registration rule stays in `register`; the supplied functions can access a real file or stand in for file access during a test.

<details markdown="1">
<summary>A test double made from ordinary functions</summary>

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

The inner functions are ordinary Python functions. `record_save` can access the enclosing test's `saved` list and record what it receives. The test checks that the resulting state is handed to the storage boundary.

Here `load_known_state` is the stub and `record_save` is the spy. A **fake** goes further by providing a simplified working implementation, such as in-memory storage. A **mock** usually adds predeclared expectations about which calls it should receive. Terminology varies by library; the role matters more than the label. [Fowler, *Test Double*](https://martinfowler.com/bliki/TestDouble.html).

This test double does not prove that JSON encoding, disk permissions, or persistence works. Keep the real temporary-file test too. Conversely, a file test alone may not pinpoint a mistake in the queue policy. Different boundaries answer different questions.

</details>

## 13. What passing tests can miss

An assertion-free test can run a broken function and still pass:

```python
def broken_total():
    return 99


def test_executes_but_checks_nothing():
    broken_total()
```

There is no assertion about the required total. **Code coverage** reports which code a run executed, with variants such as line and branch coverage. Line coverage checks executed lines; branch coverage checks alternatives such as the outcomes of a condition. Neither tells us whether assertions express the right requirements or distinguish relevant wrong answers. Fowler recommends coverage as a way to find untested areas, not as a numerical guarantee of test quality. [*Test Coverage*](https://martinfowler.com/bliki/TestCoverage.html).

We can ask a sharper question in this workshop: does our test set notice a specific mistake? Deliberately change the capacity comparison, remove duplicate checking, promote from the wrong end, or mutate the input. Each small faulty implementation is a **mutant**. Checking whether tests detect such changes is the idea behind **mutation testing**. The activity below evaluates a small explicitly listed collection of faults; it is not an exhaustive mutation system. For an actual Python mutation tool and its operating limits, see the [mutmut project on GitHub](https://github.com/boxed/mutmut).

<div class="cleancode-test-matrix-wrapper" role="region" aria-labelledby="cleancode-matrix-title" tabindex="0">
<table id="cleancode-test-matrix">
<caption id="cleancode-matrix-title">Which test detects each isolated bug?</caption>
<colgroup><col class="cc-matrix-test"><col><col><col><col></colgroup>
<thead><tr>
<th scope="col" aria-label="Focused test"><span class="cc-matrix-full">Focused test</span><span class="cc-matrix-brief" aria-hidden="true">Test</span></th>
<th scope="col" aria-label="Wrong capacity check"><span class="cc-matrix-full">Wrong capacity check</span><span class="cc-matrix-brief" aria-hidden="true">Cap.</span></th>
<th scope="col" aria-label="Duplicate ID allowed"><span class="cc-matrix-full">Duplicate ID allowed</span><span class="cc-matrix-brief" aria-hidden="true">Dup.</span></th>
<th scope="col" aria-label="Wrong queue end"><span class="cc-matrix-full">Wrong queue end</span><span class="cc-matrix-brief" aria-hidden="true">LIFO</span></th>
<th scope="col" aria-label="Input changed"><span class="cc-matrix-full">Input changed</span><span class="cc-matrix-brief" aria-hidden="true">Input</span></th>
</tr></thead>
<tbody>
<tr data-test="boundary"><th scope="row" aria-label="T1 · Full workshop"><span class="cc-matrix-full">T1 · Full workshop</span><span class="cc-matrix-brief" aria-hidden="true">T1</span></th>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">Detected</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
</tr>
<tr data-test="duplicate"><th scope="row" aria-label="T2 · Duplicate ID"><span class="cc-matrix-full">T2 · Duplicate ID</span><span class="cc-matrix-brief" aria-hidden="true">T2</span></th>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">Detected</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
</tr>
<tr data-test="fifo"><th scope="row" aria-label="T3 · FIFO promotion"><span class="cc-matrix-full">T3 · FIFO promotion</span><span class="cc-matrix-brief" aria-hidden="true">T3</span></th>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">Detected</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
</tr>
<tr data-test="nonmutation"><th scope="row" aria-label="T4 · Input unchanged"><span class="cc-matrix-full">T4 · Input unchanged</span><span class="cc-matrix-brief" aria-hidden="true">T4</span></th>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="false"><span aria-hidden="true">—</span><span class="visually-hidden">Not detected by this test</span></td>
<td data-detected="true" class="cc-matrix-detected"><span aria-hidden="true">●</span><span class="visually-hidden">Detected</span></td>
</tr>
</tbody>
</table>
</div>
<div class="cleancode-matrix-mobile-key">
<p><strong>T1</strong> Full workshop</p>
<p><strong>T2</strong> Duplicate ID</p>
<p><strong>T3</strong> FIFO promotion</p>
<p><strong>T4</strong> Input unchanged</p>
</div>
<p class="cleancode-matrix-legend"><strong>● Detected:</strong> the test fails on that faulty implementation. <strong>—</strong> this test does not expose that bug.</p>

*Table 1. Each cell comes from running the same test–fault combinations as the activity.*

{% include interactive/article-lab.html id="cleancode-tests" %}

Start with only a free-seat case. Which faults survive? Add duplicate, exact-capacity, FIFO, and no-mutation checks individually. Inspect the counterexample, not just the detection count. Detecting all listed faults still says nothing about unlisted faults or a mistaken specification.

<details markdown="1">
<summary>Properties complement examples</summary>

A **property** is a statement expected to hold over a class of inputs. For instance, registering the same ID twice should have the same value result as registering it once. The term **idempotent** names that particular property. Neither this nor an attendance bound replaces the explicit FIFO example.

One useful small exhaustive check enumerates valid states over a few IDs and checks invariants after every operation. It establishes those results for that finite input set.

</details>

## 14. Add safety around existing code

Existing code often has no tests. Before changing its structure, choose one of the functions called by other code, prepare an input that reproduces the result, observe that result, and record it in a test. Such a **characterization test** documents current behavior. It does not automatically declare that behavior desirable. Alberto Savoia demonstrates this observe-then-record approach in [*Working Effectively With Characterization Tests*](https://www.artima.com/weblogs/viewpost.jsp?thread=198296), crediting Michael Feathers for the concept.

Suppose the current cancellation promotes D before C. A characterization test may initially capture that fact so a structural change can be checked independently. If the agreed requirement is FIFO, deliberately update or retire the contradictory LIFO assertion, introduce the expectation of C, and fix the behavior. Keeping both contradictory assertions would make the suite impossible to satisfy. Preserving a bug and correcting a bug are different jobs; the test name and change description should make the distinction clear.

Start with a narrow behavior and inputs you can control. If file access prevents a focused policy test, a small separation can make the test possible. Avoid rewriting every module while simultaneously changing its business rules. The existing code was not developed test-first, but small behavior-preserving changes can begin now. [Fowler, *Refactoring*](https://refactoring.com/).

<details markdown="1">
<summary>Where style rules and extra abstractions stop helping</summary>

A formatter can make indentation consistent but cannot decide which cancellation rule users need. A **type hint**, such as `participant_id: str`, documents an expected type but does not automatically validate values while the program runs. [Python's typing documentation](https://docs.python.org/3/library/typing.html) makes that distinction explicit. Splitting one function into many can make navigation harder when the new names convey no meaning.

Searching a list may require checking every participant ID; copying the whole list also visits every entry. In this small sequential program, those operations can therefore take work proportional to the roster size. More helpers do not improve that growth rate. Larger rosters, concurrent requests, or a requirement to preserve stored data after a program interruption may justify different data structures and architecture. Adding them before they are needed creates more contracts to understand.

</details>

## 15. Read the failures, then implement a small extension

{% include interactive/article-lab.html id="cleancode-quiz" %}

The questions use concrete state, code, and failure messages. Each answer has its own explanation. Before choosing an answer, state what observation would distinguish the alternatives.

### Coding task: increase capacity without losing queue order

Implement `increase_capacity(state, new_capacity)` in `exercise.py`. This is a new feature, not a refactoring of the fixed-capacity rule.

- `new_capacity` must be a positive integer, excluding `bool`.
- A smaller value raises `ValueError`; an equal value leaves the state's value unchanged.
- A larger value promotes as many waiting IDs as there are new free seats, in FIFO order.
- Existing confirmed order is retained. Everyone not promoted keeps their waiting order.
- The supplied state and its lists remain unchanged.
- Return a new dictionary and independent `confirmed` and `waiting` lists, even when capacity is unchanged.
- The input state must satisfy the contract in section 2. An invalid state raises `ValueError`.

With capacity two, confirmed `["A", "B"]`, and waiting `["C", "D"]`, increasing to three gives confirmed `["A", "B", "C"]` and waiting `["D"]`. Increasing to five promotes both waiting IDs and leaves one seat free.

Write down the cases before implementation: same capacity, one new seat, several new seats, fewer waiting people than seats, no waiting people, invalid input, and input preservation. Choose one, write a test, inspect its failure, then implement enough to pass it and the earlier tests.

```bash
python -m pytest -q exercise_tests
```

The starter deliberately raises `NotImplementedError`. The reference implementation is a separate file, `exercise_solution.py`; importing it does not replace your function. The project README explains the reference check and the published cases. There is no hidden requirement to use a particular helper name or algorithm.

<details markdown="1">
<summary>Hints before opening the reference solution</summary>

First compute how many places are free in the new capacity. Take at most that many IDs from the front of the old waiting list, append them to a copy of the confirmed list, and retain the rest as the new waiting list. Python slices make these two queue segments explicit.

Checking only the number promoted can miss a reversed prefix. Checking only the returned state can miss a mutated input. The tests should make both errors visible. After the suite passes, decide whether naming the promoted segment makes the implementation easier to read.

[Reference implementation]({{ site.baseurl }}/assets/code/clean-code-tdd/exercise_solution.py) · [Student starter]({{ site.baseurl }}/assets/code/clean-code-tdd/exercise.py)

</details>

## 16. Benefits and limits of clean code and TDD

Clear interfaces reduce the amount of internal detail a caller needs to understand. Explicit tests preserve executable examples of requirements. Small refactorings make structural changes easier to inspect. TDD connects those activities by giving the next behavior a short feedback cycle.

None of this guarantees a correct specification, complete test coverage of relevant behavior, universally good architecture, or faster delivery in every project. Tests also require maintenance. Tests tied to incidental internals can make harmless changes expensive, and slow or nondeterministic tests can weaken feedback. A **flaky test** changes outcome without the intended behavior changing, for example because it depends on shared mutable state or uncontrolled timing. [pytest's explanation of flaky tests](https://docs.pytest.org/en/stable/explanation/flaky.html) discusses such uncontrolled state.

Performance, accessibility, security, and concurrent behavior need their own requirements and appropriate checks. A passing queue-policy suite does not establish those unrelated properties.

The useful discipline is specific: state the next behavior, choose an example that could expose a real mistake, inspect the failure, make the change, and keep the code understandable for the next person who must change it.

## References and runnable material

### Concepts and original wording

- [Kent Beck, *Canon TDD*](https://newsletter.kentbeck.com/p/canon-tdd) — test lists, one next runnable test, implementation changes, and restraint in abstraction. The short quotation in section 8 is verbatim.
- [Martin Fowler, *Test Driven Development*](https://martinfowler.com/bliki/TestDrivenDevelopment.html) — Red–Green–Refactor and interface-first thinking.
- [Martin Fowler, *Refactoring*](https://refactoring.com/) — behavior-preserving structural changes; the quoted phrase in section 9 is verbatim.
- [Refactoring catalog: Extract Function](https://refactoring.com/catalog/extractFunction.html) and [Guard Clauses](https://refactoring.com/catalog/replaceNestedConditionalWithGuardClauses.html) — concrete transformations, not mandatory style scores.
- [Tim Peters, PEP 20](https://peps.python.org/pep-0020/) and [PEP 8](https://peps.python.org/pep-0008/) — Python design preferences, naming, and comments. The opening two-word quotation is verbatim.
- [Google Engineering Practices, code-review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html) — complexity, functionality, and avoiding speculative generality.
- [Martin Fowler, *Test Coverage*](https://martinfowler.com/bliki/TestCoverage.html) and [*Test Double*](https://martinfowler.com/bliki/TestDouble.html) — interpreting coverage and distinguishing replacement collaborators.
- [Martin Fowler, *Unit Test*](https://martinfowler.com/bliki/UnitTest.html) and [*Integration Test*](https://martinfowler.com/bliki/IntegrationTest.html) — small behavior checks and checks of cooperating pieces; neither term dictates one universal test boundary.

### Documentation, source repositories, and this example

- [pytest: Get Started](https://docs.pytest.org/en/stable/getting-started.html), [fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html), [parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html), and [temporary files](https://docs.pytest.org/en/stable/how-to/tmp_path.html).
- [pytest source on GitHub](https://github.com/pytest-dev/pytest) — runner implementation, tests, and project documentation.
- [mutmut on GitHub](https://github.com/boxed/mutmut) — a Python mutation tool; [Alberto Savoia's characterization-test example](https://www.artima.com/weblogs/viewpost.jsp?thread=198296) — recording existing behavior.
- [Python typing documentation](https://docs.python.org/3/library/typing.html) and [pytest flaky-test guidance](https://docs.pytest.org/en/stable/explanation/flaky.html) — runtime annotation limits and uncontrolled test state.
- [Python: copying](https://docs.python.org/3/library/copy.html) and [JSON](https://docs.python.org/3/library/json.html) — language and library behavior used by the examples.
- [Python: the assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement) — conditions, `AssertionError`, and why ordinary `assert` checks should not be run with `python -O`, which can remove them.
- [Complete workshop project ZIP]({{ site.baseurl }}/assets/code/clean-code-tdd-workshop.zip), [workshop.py]({{ site.baseurl }}/assets/code/clean-code-tdd/workshop.py), and [README]({{ site.baseurl }}/assets/code/clean-code-tdd/README.md).
- [Recorded Python development runs]({{ site.baseurl }}/assets/data/cleancode/tdd-runs.json) and [cross-implementation fixtures]({{ site.baseurl }}/assets/data/cleancode/workshop-fixtures.json).
