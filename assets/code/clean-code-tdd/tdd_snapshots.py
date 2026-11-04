"""Source for the recorded TDD steps, executed by the repository asset builder.

These deliberately small snapshots use valid inputs only. The final workshop.py
also validates the complete published contract. A RED step is a failing assertion,
not an import error or an unfinished-environment error.
"""

from textwrap import dedent


START = '''\
def new_workshop(capacity):
    return {"capacity": capacity, "confirmed": [], "waiting": []}


'''

COPY = '''\
    updated = {
        "capacity": state["capacity"],
        "confirmed": state["confirmed"].copy(),
        "waiting": state["waiting"].copy(),
    }
'''

REGISTER_STUB = "def register(state, participant_id):\n" + COPY + "    return updated\n"
REGISTER_CONFIRM = "def register(state, participant_id):\n" + COPY + '''\
    updated["confirmed"].append(participant_id)
    return updated
'''
REGISTER_WAIT = "def register(state, participant_id):\n" + COPY + '''\
    if len(updated["confirmed"]) < updated["capacity"]:
        updated["confirmed"].append(participant_id)
    else:
        updated["waiting"].append(participant_id)
    return updated
'''
REGISTER_DUPLICATE = "def register(state, participant_id):\n" + COPY + '''\
    if participant_id in updated["confirmed"] or participant_id in updated["waiting"]:
        return updated
    if len(updated["confirmed"]) < updated["capacity"]:
        updated["confirmed"].append(participant_id)
    else:
        updated["waiting"].append(participant_id)
    return updated
'''
CANCEL_STUB = "\n\ndef cancel(state, participant_id):\n" + COPY + "    return updated\n"
CANCEL_WAITING = "\n\ndef cancel(state, participant_id):\n" + COPY + '''\
    if participant_id in updated["waiting"]:
        updated["waiting"].remove(participant_id)
    return updated
'''
CANCEL_COMPLETE = "\n\ndef cancel(state, participant_id):\n" + COPY + '''\
    if participant_id in updated["confirmed"]:
        updated["confirmed"].remove(participant_id)
        if updated["waiting"]:
            updated["confirmed"].append(updated["waiting"].pop(0))
    elif participant_id in updated["waiting"]:
        updated["waiting"].remove(participant_id)
    return updated
'''

TEST_CONFIRM = '''\
from workshop import new_workshop, register


def test_free_seat_is_confirmed():
    assert register(new_workshop(1), "Ada") == {
        "capacity": 1, "confirmed": ["Ada"], "waiting": []
    }
'''
TEST_WAIT = TEST_CONFIRM + '''\


def test_full_workshop_adds_to_waiting():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": []}
    assert register(state, "Bo") == {
        "capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]
    }
'''
TEST_DUPLICATE = TEST_WAIT + '''\


def test_duplicate_id_is_a_value_noop():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo"]}
    assert register(state, "Ada") == state
    assert register(state, "Bo") == state
'''
TEST_CANCEL_WAITING = TEST_DUPLICATE.replace(
    "from workshop import new_workshop, register",
    "from workshop import cancel, new_workshop, register",
) + '''\


def test_cancel_waiting_only_removes_that_id():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy"]}
    assert cancel(state, "Bo") == {
        "capacity": 1, "confirmed": ["Ada"], "waiting": ["Cy"]
    }
'''
TEST_CANCEL = TEST_CANCEL_WAITING + '''\


def test_cancel_confirmed_promotes_first_waiting():
    state = {"capacity": 1, "confirmed": ["Ada"], "waiting": ["Bo", "Cy"]}
    assert cancel(state, "Ada") == {
        "capacity": 1, "confirmed": ["Bo"], "waiting": ["Cy"]
    }
'''

REFACTORED = START + '''\
def copy_state(state):
    return {
        "capacity": state["capacity"],
        "confirmed": state["confirmed"].copy(),
        "waiting": state["waiting"].copy(),
    }


def register(state, participant_id):
    updated = copy_state(state)
    if participant_id in updated["confirmed"] or participant_id in updated["waiting"]:
        return updated
    if len(updated["confirmed"]) < updated["capacity"]:
        updated["confirmed"].append(participant_id)
    else:
        updated["waiting"].append(participant_id)
    return updated


def cancel(state, participant_id):
    updated = copy_state(state)
    if participant_id in updated["confirmed"]:
        updated["confirmed"].remove(participant_id)
        if updated["waiting"]:
            updated["confirmed"].append(updated["waiting"].pop(0))
    elif participant_id in updated["waiting"]:
        updated["waiting"].remove(participant_id)
    return updated
'''


def snapshot(identifier, stage, title_en, title_ko, code, test, passed, failed,
             reason_en, reason_ko):
    return {
        "id": identifier,
        "stage": stage,
        "title": {"en": title_en, "ko": title_ko},
        "code": dedent(code),
        "test": dedent(test),
        "expected_passed": passed,
        "expected_failed": failed,
        "failure_reason": {"en": reason_en, "ko": reason_ko},
    }


SNAPSHOTS = [
    snapshot("confirm-red", "red", "A free seat should confirm Ada", "빈자리에 Ada를 확정한다",
             START + REGISTER_STUB, TEST_CONFIRM, 0, 1,
             "The assertion fails: confirmed is still empty.",
             "확정 명단이 비어 있어 assertion이 실패한다."),
    snapshot("confirm-green", "green", "Confirm the first participant", "첫 참가자를 확정한다",
             START + REGISTER_CONFIRM, TEST_CONFIRM, 1, 0,
             "The single free-seat test passes; full workshops are not tested yet.",
             "빈자리 테스트 하나가 통과한다. 만석 동작은 아직 검사하지 않는다."),
    snapshot("waiting-red", "red", "A full workshop needs a waiting list", "만석이면 대기 명단이 필요하다",
             START + REGISTER_CONFIRM, TEST_WAIT, 1, 1,
             "Bo was appended to confirmed instead of waiting.",
             "Bo가 대기 명단이 아니라 확정 명단에 추가되어 실패한다."),
    snapshot("waiting-green", "green", "Choose confirmed or waiting", "확정과 대기 명단을 나눈다",
             START + REGISTER_WAIT, TEST_WAIT, 2, 0,
             "Both free-seat and full-workshop tests pass.",
             "빈자리와 만석 테스트가 모두 통과한다."),
    snapshot("duplicate-red", "red", "Registering twice must not duplicate an ID", "같은 ID가 두 번 추가되면 안 된다",
             START + REGISTER_WAIT, TEST_DUPLICATE, 2, 1,
             "An existing ID was appended again.",
             "이미 있는 ID가 다시 추가되어 실패한다."),
    snapshot("duplicate-green", "green", "Keep an existing registration unchanged", "이미 신청한 ID는 그대로 둔다",
             START + REGISTER_DUPLICATE, TEST_DUPLICATE, 3, 0,
             "The duplicate test covers IDs in confirmed and waiting.",
             "중복 테스트는 확정과 대기 명단의 ID를 모두 검사한다."),
    snapshot("waiting-cancel-red", "red", "Cancelling a waiting ID must remove it", "대기 중인 ID를 취소하면 명단에서 빼야 한다",
             START + REGISTER_DUPLICATE + CANCEL_STUB, TEST_CANCEL_WAITING, 3, 1,
             "The new waiting-cancellation assertion fails because the stub leaves Bo in the queue.",
             "임시 구현이 Bo를 대기 명단에 그대로 두어 새로 추가한 대기 취소 테스트가 실패한다."),
    snapshot("waiting-cancel-green", "green", "Remove only the cancelled waiting ID", "취소한 대기 ID만 명단에서 뺀다",
             START + REGISTER_DUPLICATE + CANCEL_WAITING, TEST_CANCEL_WAITING, 4, 0,
             "The four tests pass. Confirmed cancellation and FIFO promotion have not been implemented yet.",
             "네 테스트가 통과한다. 확정 참가자의 취소와 FIFO 승급은 아직 구현하지 않았다."),
    snapshot("cancel-red", "red", "Cancelling a confirmed ID must promote FIFO", "확정 참가자가 취소하면 맨 앞 대기자를 승급시킨다",
             START + REGISTER_DUPLICATE + CANCEL_WAITING, TEST_CANCEL, 4, 1,
             "Only the new confirmed-cancellation assertion fails: Ada remains confirmed instead of Bo.",
             "확정 취소 테스트 하나가 새로 실패한다. Bo를 승급시키지 않고 Ada를 확정 명단에 그대로 두기 때문이다."),
    snapshot("cancel-green", "green", "Remove and promote in FIFO order", "취소하고 FIFO 순서로 승급한다",
             START + REGISTER_DUPLICATE + CANCEL_COMPLETE, TEST_CANCEL, 5, 0,
             "All five tests pass after implementing cancellation.",
             "취소를 구현한 뒤 다섯 테스트가 모두 통과한다."),
    snapshot("copy-refactor", "refactor", "Extract the repeated state copy", "반복되는 복사 코드를 helper로 묶는다",
             REFACTORED, TEST_CANCEL, 5, 0,
             "The same five tests still pass; no requirement or test changed.",
             "요구사항과 테스트를 바꾸지 않고 같은 다섯 테스트가 통과한다."),
]
