"""One solution; the starter in exercise.py remains unchanged."""

from workshop import copy_state


def increase_capacity(state, new_capacity):
    """Increase capacity; preserve state and promote as many as fit, FIFO."""
    updated = copy_state(state)
    if type(new_capacity) is not int or new_capacity <= 0:
        raise ValueError("new_capacity must be a positive integer")
    if new_capacity < updated["capacity"]:
        raise ValueError("new_capacity cannot be smaller than current capacity")
    updated["capacity"] = new_capacity
    free_seats = new_capacity - len(updated["confirmed"])
    promoted = updated["waiting"][:free_seats]
    updated["confirmed"].extend(promoted)
    updated["waiting"] = updated["waiting"][free_seats:]
    return updated
