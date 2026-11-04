"""Published contract tests shared by the starter and the separate solution.

Default: python -m pytest exercise_tests -q
Solution: WORKSHOP_EXERCISE_SOLUTION=1 python -m pytest exercise_tests -q
"""

import os
from copy import deepcopy

import pytest

if os.environ.get("WORKSHOP_EXERCISE_SOLUTION") == "1":
    from exercise_solution import increase_capacity
else:
    from exercise import increase_capacity


def test_same_capacity_is_a_value_noop():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    assert increase_capacity(state, 1) == state


def test_larger_capacity_promotes_in_fifo_order():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy", "Di"]}
    assert increase_capacity(state, 3) == {
        "capacity": 3, "confirmed": ["Ada", "Bo", "Cy"], "waiting": ["Di"]
    }


def test_larger_capacity_can_promote_everyone_and_leave_free_seats():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    assert increase_capacity(state, 4) == {
        "capacity": 4, "confirmed": ["Ada", "Bo"], "waiting": []
    }


def test_larger_capacity_without_waiting():
    state = {"capacity": 2, "confirmed": ["Ada"], "waiting": []}
    assert increase_capacity(state, 4) == {
        "capacity": 4, "confirmed": ["Ada"], "waiting": []
    }


@pytest.mark.parametrize("new_capacity", [1, 2, 4])
def test_input_is_unchanged_and_nested_lists_are_independent(new_capacity):
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy"]}
    original = deepcopy(state)
    updated = increase_capacity(state, new_capacity)
    assert state == original
    assert updated is not state
    assert updated["confirmed"] is not state["confirmed"]
    assert updated["waiting"] is not state["waiting"]
    updated["confirmed"].append("changed result")
    updated["waiting"].append("changed result")
    assert state == original


def test_smaller_capacity_is_rejected_without_mutating_input():
    state = {"capacity": 2, "confirmed": ["Ada"], "waiting": []}
    original = deepcopy(state)
    with pytest.raises(ValueError):
        increase_capacity(state, 1)
    assert state == original


@pytest.mark.parametrize("new_capacity", [0, -1, True, False, 1.0, "3", None])
def test_invalid_new_capacity_is_rejected(new_capacity):
    with pytest.raises(ValueError):
        increase_capacity({"capacity": 1, "confirmed": [], "waiting": []}, new_capacity)


@pytest.mark.parametrize("state", [
    {},
    {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Ada"]},
    {"capacity": 2, "confirmed": ["Ada"], "waiting": ["Bo"]},
])
def test_invalid_input_state_is_rejected(state):
    with pytest.raises(ValueError):
        increase_capacity(state, 3)
