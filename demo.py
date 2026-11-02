import json

import cv2
import numpy as np

from measure import measure


def synthetic_example():
    y, x = np.indices((512, 512))
    green = (120 + (x + y) % 48).astype(np.uint8)
    bgr = np.stack((green - 20, green, green + 40), axis=-1)
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(mask, (256, 256), (205, 137), 0, 0, 360, 1, -1)
    for offset in range(-90, 100, 15):
        cv2.line(bgr, (65, 256 + offset), (430, 280 - offset), (40, 45, 160), 2)
    return measure(bgr, mask)


if __name__ == "__main__":
    print(json.dumps(synthetic_example(), allow_nan=False))
