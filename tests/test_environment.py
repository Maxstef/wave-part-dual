import cv2
import numpy as np

import wave_part_dual


def test_core_environment_is_usable() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255

    dilated = cv2.dilate(image, np.ones((3, 3), dtype=np.uint8))

    assert wave_part_dual.__doc__
    assert np.count_nonzero(dilated) == 9
