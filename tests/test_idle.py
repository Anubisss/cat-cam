import numpy as np

from catcam import idle as idle_module
from catcam.idle import Idle

DT = 1 / 30
RELAXED = {name: 0.05 for name in idle_module.FIDGETS}


def _run(idle, start, seconds, head=lambda t: np.zeros(3), scores=lambda t: RELAXED, talking=lambda t: False):
    """Feeds idle seconds of you at 30 fps from start; returns the times and its (head, jaw, tongue)."""
    times = start + DT * np.arange(1, int(seconds / DT) + 1)
    out = []
    for t in times:
        idle.update(head(t), scores(t), talking(t), DT, t)
        out.append((idle.head.copy(), idle.jaw, idle.tongue))
    return times, out


def _moves(out) -> float:
    """How far the animal's head moved on its own."""
    return max(np.linalg.norm(head) for head, _, _ in out)


def test_nothing_happens_while_you_have_just_been_still():
    idle = Idle(np.random.default_rng(0))
    _, out = _run(idle, 0.0, idle_module.STILL_AFTER - 0.2)
    assert _moves(out) == 0 and idle.amount == 0


def test_when_you_are_still_it_breathes_and_looks_around():
    idle = Idle(np.random.default_rng(0))
    _run(idle, 0.0, idle_module.STILL_AFTER + idle_module.FADE_IN)
    assert idle.amount == 1
    _, out = _run(idle, 10.0, 30.0)
    nods = np.array([head[0] for head, _, _ in out])
    assert np.ptp(nods) > idle_module.BREATHING[1]  # it breathes
    assert max(abs(head[1]) for head, _, _ in out) > 0.05  # and turns to look around


def test_a_dog_pants_with_its_tongue_out_now_and_then():
    idle = Idle(np.random.default_rng(0))
    _, out = _run(idle, 0.0, 60.0)
    jaws = np.array([jaw for _, jaw, _ in out])
    tongues = np.array([tongue for _, _, tongue in out])
    assert jaws.max() > 0.04 and tongues.max() > 0.5  # pants...
    assert (jaws == 0).mean() > 0.3  # ...in bouts
    assert (jaws >= 0).all()


def _idling():
    idle = Idle(np.random.default_rng(1))
    _run(idle, 0.0, 20.0)
    assert idle.amount == 1
    return idle


def test_turning_your_head_stops_it():
    idle = _idling()
    _, out = _run(idle, 20.0, 0.5, head=lambda t: np.array([0, 0.8 * (t - 20), 0]))
    assert idle.amount == 0 and _moves(out[-1:]) == 0


def test_talking_stops_it():
    idle = _idling()
    _run(idle, 20.0, 0.4, talking=lambda t: True)
    assert idle.amount == 0 and idle.jaw == 0 and idle.tongue == 0


def test_moving_your_mouth_stops_it():
    idle = _idling()
    chatter = lambda t: {**RELAXED, "jawOpen": 0.3 * (1 + np.sin(2 * np.pi * 4 * t)) / 2}  # noqa: E731
    _run(idle, 20.0, 0.5, scores=chatter)
    assert idle.amount == 0


def test_blinking_does_not_stop_it():
    idle = _idling()
    blinks = lambda t: {**RELAXED, "eyeBlinkLeft": float(t % 1 < 0.15), "eyeBlinkRight": float(t % 1 < 0.15)}  # noqa: E731
    _run(idle, 20.0, 3.0, scores=blinks)
    assert idle.amount == 1


def test_it_acts_on_its_own_while_you_are_away():
    idle = Idle(np.random.default_rng(0))
    _run(idle, 0.0, 10.0, scores=lambda t: None)
    assert idle.amount == 1
