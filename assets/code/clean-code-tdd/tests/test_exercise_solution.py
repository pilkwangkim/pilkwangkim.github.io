"""The default GREEN suite tests the solution without replacing the starter."""

from exercise_solution import increase_capacity


def test_solution_promotes_fifo_without_changing_input():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy"]}
    assert increase_capacity(state, 2) == {
        "capacity": 2, "confirmed": ["Ada", "Bo"], "waiting": ["Cy"]
    }
    assert state == {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy"]}
