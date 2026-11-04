import numpy as np
import pytest

from catcam.ears import FLATTEN, PERK, TWITCH_EVERY, EarMotion, bend, brow_angles


@pytest.fixture(scope="module")
def meshes(real_cats):
    from catcam.cat_mesh import CatMesh, Renderer

    renderer = Renderer((640, 360))
    return [(cat, CatMesh(renderer, cat.sprite, cat.points, (0, 0, 0))) for cat in real_cats]


@pytest.mark.gl
def test_every_cat_has_two_ears_above_its_face(meshes):
    for cat, mesh in meshes:
        assert [ear.side for ear in mesh.ears] == [-1, 1], cat.name
        for ear in mesh.ears:
            assert not ear.weights[:478].any(), cat.name  # the face never moves with the ears
            tip = mesh.rest[ear.weights.argmax()]
            assert tip[1] < mesh.rest[10, 1], cat.name  # above the forehead
            assert (tip[0] - mesh.pivot[0]) * ear.side > 0, cat.name  # on its own side


@pytest.mark.gl
def test_flattened_ears_tip_outward_and_perked_ones_inward(meshes):
    for cat, mesh in meshes:
        for ear in mesh.ears:
            tip = ear.weights.argmax()
            flat = bend(mesh.rest, mesh.ears, (0.3, 0.3))
            perked = bend(mesh.rest, mesh.ears, (-0.15, -0.15))
            assert (flat[tip, 0] - mesh.rest[tip, 0]) * ear.side > 0, cat.name
            assert (perked[tip, 0] - mesh.rest[tip, 0]) * ear.side < 0, cat.name
        # Only one ear moves when only one is asked to.
        one = bend(mesh.rest, mesh.ears, (0.3, 0.0))
        right = mesh.ears[1].weights > 0
        np.testing.assert_array_equal(one[right & (mesh.ears[0].weights == 0)],
                                      mesh.rest[right & (mesh.ears[0].weights == 0)])


def test_brows_steer_the_ears():
    np.testing.assert_allclose(brow_angles({}), 0)
    frown = brow_angles({"browDownLeft": 1.0, "browDownRight": 1.0})
    np.testing.assert_allclose(frown, FLATTEN)
    raised = brow_angles({"browInnerUp": 1.0})
    np.testing.assert_allclose(raised, PERK)
    # The person's right brow is the image's left ear.
    one = brow_angles({"browDownRight": 1.0})
    assert one[0] == FLATTEN and one[1] == 0


def test_ears_follow_the_brows_smoothly():
    motion = EarMotion(np.random.default_rng(0))
    motion.twitch_at = np.inf  # no twitches
    frown = {"browDownLeft": 1.0, "browDownRight": 1.0}
    first = motion.update(frown, 1 / 30, 0.0)
    assert 0 < first[0] < FLATTEN
    for i in range(30):
        last = motion.update(frown, 1 / 30, i / 30)
    np.testing.assert_allclose(last, FLATTEN, atol=1e-3)


def test_an_ear_twitches_now_and_then():
    motion = EarMotion(np.random.default_rng(0))
    times = np.arange(0, 2 * TWITCH_EVERY[1], 1 / 30)
    angles = np.array([motion.update({}, 1 / 30, t) for t in times])
    assert np.abs(angles).max() > 0.1  # a twitch happened
    assert (np.abs(angles) < 1e-9).mean() > 0.5  # but mostly the ears are still
    # Without a face the ears still twitch.
    still = EarMotion(np.random.default_rng(1))
    assert max(np.abs(still.update(None, 1 / 30, t)).max() for t in times) > 0.1
