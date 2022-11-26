from typing import List

import cv2
import numpy as np
import torch

from openpose.src.body import Body


def crop_from_keypoints(image: np.ndarray, keypoints: np.ndarray, include_buffer: int) -> np.ndarray:
    x = keypoints[:, 0].astype(int)
    y = keypoints[:, 1].astype(int)
    top, bottom, left, right = min(y), max(y), min(x), max(x)
    top = max(0, top-include_buffer)
    bottom = min(image.shape[0], bottom+include_buffer)
    left = max(0, left-include_buffer)
    right = min(image.shape[1], right+include_buffer)
    crop = image[top:bottom, left:right, :]
    return crop


def get_bodies(image: np.ndarray, body_model: Body, num_required_points: int, include_buffer: int) -> List[np.ndarray]:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    keypoints, subset = body_model(image)
    valid_bodies = []
    for body in subset:

        # The last dimension of each subset contains the number of POCO body keypoints found. See openpose for more info
        if body[-1] == num_required_points:
            body_keypoint_ids = body[:18].astype(int)
            body_keypoints = keypoints[body_keypoint_ids]
            valid_bodies.append(crop_from_keypoints(image, body_keypoints, include_buffer))

    return valid_bodies



