import argparse
import json

import cv2
import numpy as np
from skimage.morphology import binary_erosion, disk

from features import analysis_params, calculate_fractal_dimension, extract_intermediate_steps


def measure(image, labels):
    if image is None or labels is None:
        raise ValueError("Could not decode the image or label mask")
    if image.shape[:2] != (512, 512) or labels.shape != (512, 512):
        raise ValueError("Expected a 512 x 512 image and matching label mask")
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected merged labels: 0 background, 1 conjunctiva, 2 cornea")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi = binary_erosion(labels == 1, disk(10)).astype(np.uint8)
    if not np.any(rgb[:, :, 1][roi > 0]):
        raise ValueError("No measurable green-channel pixels inside the eroded ROI")
    _, redness, vessels, skeleton = extract_intermediate_steps(rgb, roi, analysis_params)
    rs = float(redness[vessels > 0].sum() / roi.sum())
    fd = calculate_fractal_dimension(skeleton)
    return {"redness_score": rs, "fractal_dimension": float(fd) if np.isfinite(fd) else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure a study-format image and merged mask")
    parser.add_argument("image")
    parser.add_argument("mask")
    args = parser.parse_args()
    print(json.dumps(measure(cv2.imread(args.image), cv2.imread(args.mask, 0)), allow_nan=False))
