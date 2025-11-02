import cv2
import numpy as np
from utils import load_images_from_folder
import config

imgs, files = load_images_from_folder(config.DATASET_PATH)
orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

keyframes = []

for i, img in enumerate(imgs):
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        continue

    for idx, old_kp, old_des in keyframes:
        
        if abs(i - idx) < 30:
            continue

        matches = bf.knnMatch(des, old_des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 50:
            pts1 = np.float32([kp[m.queryIdx].pt for m in good])
            pts2 = np.float32([old_kp[m.trainIdx].pt for m in good])
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            if F is not None and mask is not None and mask.sum() > 100:
                print(f"True loop closure between frame {i} and {idx}, inliers: {mask.sum()}")

    if i % 10 == 0:
        keyframes.append((i, kp, des))
