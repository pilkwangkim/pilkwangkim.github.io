# Clean Code / TDD workshop

Python 3.13.5 and pytest 8.4.2 were used to check this download. The code uses
only the Python standard library; pytest is its sole test dependency.

## Run the GREEN example

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

`pytest.ini` selects `tests/`, not the deliberately unfinished exercise.
On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
On Windows Command Prompt (`cmd.exe`), use `.\.venv\Scripts\activate.bat`.

```python
from workshop import cancel, new_workshop, register

state = new_workshop(1)
state = register(state, "Ada")
state = register(state, "Bo")
state = register(state, "Cy")
state = cancel(state, "Ada")
assert state == {"capacity": 1, "confirmed": ["Bo"], "waiting": ["Cy"]}
```

## State contract

- Exactly three dictionary keys: `capacity`, `confirmed`, `waiting`.
- `capacity` is a positive integer. Booleans and floats are rejected.
- Both participant lists are Python lists of exact, nonempty strings.
  Case and whitespace are preserved: `"Ada"`, `"ada"`, and `" "` are distinct
  valid IDs. This is a teaching ID contract, not a username-validation policy.
- No ID occurs twice, either within one list or across the lists.
- Confirmed count cannot exceed capacity. If waiting is nonempty, confirmed is full.
- `register` confirms a free seat or appends to waiting. Existing IDs are value no-ops.
- `cancel` removes a confirmed ID and promotes the first waiting ID, removes a
  waiting ID without another change, or leaves values unchanged for an unknown ID.
- Every operation returns a new dictionary and independent nested lists, including
  value no-ops. The input is never mutated. This is this example's design choice,
  not a requirement imposed by TDD.
- Invalid states, IDs, or capacities raise `ValueError`.

`workshop.py` is the complete implementation. `tdd_snapshots.py` contains smaller,
earlier versions with deliberately incomplete behavior and a growing test suite.
Those snapshots use valid inputs only; the final module adds state/input validation.
The repository command `python tools/build_cleancode_assets.py` executes each
snapshot in a fresh temporary directory to record real pytest outcomes.

## JSON boundary

```python
from storage import load_workshop, save_workshop
from workshop import new_workshop, register

save_workshop("workshop.json", new_workshop(1))
state = load_workshop("workshop.json")
save_workshop("workshop.json", register(state, "Ada"))
```

`load_workshop` parses JSON and validates the resulting state. `save_workshop`
validates before opening the destination. Missing files, malformed JSON, and
other file I/O errors propagate. The integration tests use pytest's temporary
directory fixture, `tmp_path`, and do not touch a real registration file.

This is a single-process sequential example. A successful test does not provide
file locking, atomic replacement, crash recovery, concurrent-update protection,
database transactions, authentication, or a production registration service.

## Exercise: increase capacity

Implement `increase_capacity(state, new_capacity)` in `exercise.py`.

1. An equal capacity is a value no-op.
2. A smaller capacity raises `ValueError`.
3. A larger capacity promotes as many waiting IDs as fit, in FIFO order.
4. Preserve the input and return independent lists, even for a value no-op.
5. Validate state and positive integer capacity under the same contract.

The published tests specify exact expected values and errors:

```sh
python -m pytest exercise_tests -q
```

That command is intentionally RED until you implement the starter: its function
raises `NotImplementedError`. This exercise RED is distinct from the recorded
TDD snapshots, whose RED steps fail behavioral assertions.

A separate solution is in `exercise_solution.py`; it does not overwrite the starter.
Run exactly the same published contract tests against it:

```sh
WORKSHOP_EXERCISE_SOLUTION=1 python -m pytest exercise_tests -q
```

PowerShell: set `$env:WORKSHOP_EXERCISE_SOLUTION="1"`, run the pytest command,
then remove it with `Remove-Item Env:WORKSHOP_EXERCISE_SOLUTION`.

## 한국어 실행 안내

기본 `python -m pytest -q`는 완성된 예제의 `tests/`만 실행한다.
`exercise.py`의 미완성 과제는 `python -m pytest exercise_tests -q`로 따로
실행한다. 과제를 구현하기 전에는 `NotImplementedError`로 실패한다.
`exercise_solution.py`의 정답은 스타터를 바꾸지 않으며, 위의 환경변수를
설정하면 같은 과제 테스트로 확인할 수 있다.

ID는 공백 제거·대소문자 통일 없이 원문 그대로 비교한다. 입력 상태와 중첩
리스트를 바꾸지 않는다. JSON 예제는 순차 실행만 다루며 파일 잠금이나 원자적
저장, 동시 신청 처리를 보장하지 않는다.
