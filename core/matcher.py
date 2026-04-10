import numpy as np
import time
import cv2

from core.SIFT import (
    build_gaussian_pyramid,
    build_dog_pyramid,
    detect_keypoints,
    extract_descriptors,
)


# SSD
def ssd_distance(desc1, desc2):
    return np.sum((desc1 - desc2) ** 2)


# NCC
def ncc_distance(desc1, desc2):
    d1 = desc1 - np.mean(desc1)
    d2 = desc2 - np.mean(desc2)

    denom = (np.linalg.norm(d1) * np.linalg.norm(d2)) + 1e-8
    ncc = np.sum(d1 * d2) / denom

    return 1 - ncc


# SSD matching
def match_descriptors_ssd(descs1, descs2):
    matches = []

    for i, d1 in enumerate(descs1):
        best_j = None
        best_score = float('inf')

        for j, d2 in enumerate(descs2):
            score = ssd_distance(d1, d2)
            if score < best_score:
                best_score = score
                best_j = j

        matches.append((i, best_j, best_score))

    return matches, 0.0


# NCC matching
def match_descriptors_ncc(descs1, descs2):
    matches = []

    for i, d1 in enumerate(descs1):
        best_j = None
        best_score = float('inf')

        for j, d2 in enumerate(descs2):
            score = ncc_distance(d1, d2)
            if score < best_score:
                best_score = score
                best_j = j

        matches.append((i, best_j, best_score))

    return matches, 0.0

# -------------------------------------------
# Helper: SIFT Pipeline
# -------------------------------------------
def compute_sift_descriptors(image):
    """
    Full SIFT pipeline (CORE responsibility)
    """
    if image is None:
        return np.empty((0, 128), dtype=np.float32), []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gaussian_pyramid = build_gaussian_pyramid(gray)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)

    keypoints = detect_keypoints(dog_pyramid)
    descriptors, valid_keypoints, _ = extract_descriptors(keypoints, gaussian_pyramid)

    return descriptors, valid_keypoints


# -------------------------------------------
# Full Matching Pipeline
# -------------------------------------------
def run_matching_pipeline(img1, img2, method="ssd"):
    """
    Returns:
        matches, keypoints1, keypoints2, elapsed_time
    """

    start = time.time()

    descs1, kp1 = compute_sift_descriptors(img1)
    descs2, kp2 = compute_sift_descriptors(img2)

    if descs1.size == 0 or descs2.size == 0:
        return [], kp1, kp2, 0.0

    if method == "ssd":
        matches, _ = match_descriptors_ssd(descs1, descs2)
    else:
        matches, _ = match_descriptors_ncc(descs1, descs2)

    # sorting (CORE responsibility)
    matches = sorted(matches, key=lambda x: x[2])

    elapsed = time.time() - start

    return matches, kp1, kp2, elapsed