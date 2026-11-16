import numpy as np

from wave_part_dual.classical_cv import rolling_shutter, sample_rggb, simulate_camera


def test_sample_rggb_uses_the_expected_color_at_each_location() -> None:
    rgb = np.array(
        [
            [[10, 11, 12], [20, 21, 22]],
            [[30, 31, 32], [40, 41, 42]],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(sample_rggb(rgb), [[10, 21], [31, 42]])


def test_simulate_camera_is_deterministic_and_preserves_shape() -> None:
    rgb = np.full((6, 7, 3), 100, dtype=np.uint8)

    first = simulate_camera(rgb, seed=9)
    second = simulate_camera(rgb, seed=9)

    assert first.shape == rgb.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_rolling_shutter_keeps_first_row_and_shifts_last_row() -> None:
    image = np.arange(15).reshape(3, 5)
    distorted = rolling_shutter(image, total_shift=2)

    assert np.array_equal(distorted[0], image[0])
    assert np.array_equal(distorted[-1], np.roll(image[-1], 2))
