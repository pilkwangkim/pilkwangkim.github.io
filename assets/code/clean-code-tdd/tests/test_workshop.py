from copy import deepcopy

import pytest

from workshop import cancel, new_workshop, register, validate_state


def test_new_workshop():
    assert new_workshop(2) == {"capacity": 2, "confirmed": [], "waiting": []}


def test_new_workshops_do_not_share_lists():
    first = new_workshop(2)
    second = new_workshop(2)
    first["confirmed"].append("Ada")
    assert second["confirmed"] == []


def test_register_confirms_a_free_seat():
    assert register(new_workshop(1), "Ada") == {
        "capacity": 1, "confirmed": ["Ada"], "waiting": []
    }


def test_register_appends_to_waiting_when_full():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    assert register(state, "Cy") == {
        "capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy"]
    }


@pytest.mark.parametrize("participant_id", ["Ada", "Bo"])
def test_duplicate_registration_is_a_value_noop(participant_id):
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    assert register(state, participant_id) == state


def test_ids_are_exact_nonempty_strings():
    state = register(new_workshop(3), "Ada")
    state = register(state, "ada")
    state = register(state, " ")
    assert state["confirmed"] == ["Ada", "ada", " "]


def test_cancel_confirmed_promotes_first_waiting_participant():
    state = {"capacity": 2, "confirmed": ["Ada", "Bo"], "waiting": ["Cy", "Di"]}
    assert cancel(state, "Ada") == {
        "capacity": 2, "confirmed": ["Bo", "Cy"], "waiting": ["Di"]
    }


def test_cancel_confirmed_without_waiting_leaves_a_free_seat():
    state = {"capacity": 2, "confirmed": ["Ada", "Bo"], "waiting": []}
    assert cancel(state, "Ada") == {
        "capacity": 2, "confirmed": ["Bo"], "waiting": []
    }


def test_cancel_waiting_only_removes_that_participant():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy", "Di"]}
    assert cancel(state, "Cy") == {
        "capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Di"]
    }


def test_cancel_unknown_id_is_a_value_noop():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    assert cancel(state, "unknown") == state


@pytest.mark.parametrize("operation,participant_id", [
    (register, "Cy"), (register, "Ada"), (register, "Bo"),
    (cancel, "Ada"), (cancel, "Bo"), (cancel, "unknown"),
])
def test_operations_preserve_input_and_return_independent_lists(operation, participant_id):
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    original = deepcopy(state)
    updated = operation(state, participant_id)
    assert state == original
    assert updated is not state
    assert updated["confirmed"] is not state["confirmed"]
    assert updated["waiting"] is not state["waiting"]
    updated["confirmed"].append("changed result")
    updated["waiting"].append("changed result")
    assert state == original


@pytest.mark.parametrize("capacity", [0, -1, True, False, 1.0, "2", None])
def test_new_workshop_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        new_workshop(capacity)


@pytest.mark.parametrize("participant_id", ["", None, 1, True, [], {}])
@pytest.mark.parametrize("operation", [register, cancel])
def test_operations_reject_invalid_ids(operation, participant_id):
    state = new_workshop(1)
    with pytest.raises(ValueError):
        operation(state, participant_id)
    assert state == new_workshop(1)


INVALID_STATES = [
    None,
    {},
    {"capacity": 1, "confirmed": []},
    {"capacity": 1, "confirmed": [], "waiting": [], "extra": 1},
    {"capacity": 0, "confirmed": [], "waiting": []},
    {"capacity": True, "confirmed": [], "waiting": []},
    {"capacity": 1, "confirmed": (), "waiting": []},
    {"capacity": 1, "confirmed": [], "waiting": "Ada"},
    {"capacity": 2, "confirmed": ["Ada", "Ada"], "waiting": []},
    {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Ada"]},
    {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Bo"]},
    {"capacity": 1, "confirmed": ["Ada", "Bo"], "waiting": []},
    {"capacity": 2, "confirmed": ["Ada"], "waiting": ["Bo"]},
    {"capacity": 1, "confirmed": [""], "waiting": []},
    {"capacity": 1, "confirmed": [1], "waiting": []},
]


@pytest.mark.parametrize("state", INVALID_STATES)
def test_invalid_state_is_rejected_before_operations(state):
    original = deepcopy(state)
    with pytest.raises(ValueError):
        validate_state(state)
    for operation in [register, cancel]:
        with pytest.raises(ValueError):
            operation(state, "Ada")
    assert state == original


def test_operations_preserve_invariants_through_a_longer_sequence():
    state = new_workshop(2)
    for participant_id in ["Ada", "Bo", "Cy", "Di", "Ada", "Cy"]:
        state = register(state, participant_id)
        validate_state(state)
    for participant_id in ["Bo", "missing", "Di", "Ada", "Cy"]:
        state = cancel(state, participant_id)
        validate_state(state)
    assert state == new_workshop(2)
