"""A sequential, in-memory workshop waiting list.

Each operation returns a new dictionary and new lists. Participant IDs are
case-sensitive, exact strings: no trimming or other normalization is performed.
"""


def _validate_capacity(capacity):
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")


def _validate_participant_id(participant_id):
    if not isinstance(participant_id, str) or participant_id == "":
        raise ValueError("participant_id must be a nonempty string")


def validate_state(state):
    """Raise ValueError unless state satisfies the workshop contract."""
    if not isinstance(state, dict) or set(state) != {
        "capacity", "confirmed", "waiting"
    }:
        raise ValueError("state must contain capacity, confirmed, and waiting")
    _validate_capacity(state["capacity"])
    confirmed = state["confirmed"]
    waiting = state["waiting"]
    if not isinstance(confirmed, list) or not isinstance(waiting, list):
        raise ValueError("confirmed and waiting must be lists")
    participants = confirmed + waiting
    for participant_id in participants:
        _validate_participant_id(participant_id)
    if len(set(participants)) != len(participants):
        raise ValueError("participant IDs must be unique across both lists")
    if len(confirmed) > state["capacity"]:
        raise ValueError("confirmed cannot exceed capacity")
    if waiting and len(confirmed) != state["capacity"]:
        raise ValueError("a waiting list requires all confirmed seats to be full")


def copy_state(state):
    """Validate state and return a copy with independent nested lists."""
    validate_state(state)
    return {
        "capacity": state["capacity"],
        "confirmed": state["confirmed"].copy(),
        "waiting": state["waiting"].copy(),
    }


def new_workshop(capacity):
    """Create an empty workshop with a positive integer capacity."""
    _validate_capacity(capacity)
    return {"capacity": capacity, "confirmed": [], "waiting": []}


def register(state, participant_id):
    """Confirm a free seat, or append to the FIFO waiting list.

    Registering an already present ID is a value-level no-op, not an error.
    """
    updated = copy_state(state)
    _validate_participant_id(participant_id)
    if participant_id in updated["confirmed"] or participant_id in updated["waiting"]:
        return updated
    if len(updated["confirmed"]) < updated["capacity"]:
        updated["confirmed"].append(participant_id)
    else:
        updated["waiting"].append(participant_id)
    return updated


def cancel(state, participant_id):
    """Cancel one registration; a vacated confirmed seat promotes FIFO."""
    updated = copy_state(state)
    _validate_participant_id(participant_id)
    if participant_id in updated["confirmed"]:
        updated["confirmed"].remove(participant_id)
        if updated["waiting"]:
            updated["confirmed"].append(updated["waiting"].pop(0))
    elif participant_id in updated["waiting"]:
        updated["waiting"].remove(participant_id)
    return updated
