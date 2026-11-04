import json
from copy import deepcopy

import pytest

from storage import load_workshop, save_workshop
from workshop import cancel, new_workshop, register


def test_json_round_trip(tmp_path):
    path = tmp_path / "workshop.json"
    state = {"capacity": 1, "confirmed": ["아다"], "waiting": ["Bo"]}
    original = deepcopy(state)
    save_workshop(path, state)
    assert load_workshop(path) == state
    assert state == original
    assert "아다" in path.read_text(encoding="utf-8")


def test_sequential_load_change_save(tmp_path):
    path = tmp_path / "workshop.json"
    save_workshop(path, new_workshop(1))
    save_workshop(path, register(load_workshop(path), "Ada"))
    save_workshop(path, register(load_workshop(path), "Bo"))
    save_workshop(path, cancel(load_workshop(path), "Ada"))
    assert load_workshop(path) == {
        "capacity": 1, "confirmed": ["Bo"], "waiting": []
    }


@pytest.mark.parametrize("invalid_state", [
    {},
    {"capacity": True, "confirmed": [], "waiting": []},
    {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Ada"]},
    {"capacity": 2, "confirmed": ["Ada"], "waiting": ["Bo"]},
])
def test_load_validates_parsed_json(tmp_path, invalid_state):
    path = tmp_path / "workshop.json"
    path.write_text(json.dumps(invalid_state), encoding="utf-8")
    with pytest.raises(ValueError):
        load_workshop(path)


def test_load_reports_malformed_json(tmp_path):
    path = tmp_path / "workshop.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_workshop(path)


def test_load_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workshop(tmp_path / "missing.json")


def test_invalid_save_does_not_overwrite_existing_file(tmp_path):
    path = tmp_path / "workshop.json"
    save_workshop(path, new_workshop(1))
    original_text = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        save_workshop(path, {"capacity": 0, "confirmed": [], "waiting": []})
    assert path.read_text(encoding="utf-8") == original_text
