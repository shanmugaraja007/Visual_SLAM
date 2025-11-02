import cv2
from utils import load_images_from_folder
import config

imgs, files = load_images_from_folder(config.DATASET_PATH)
orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


video_out = cv2.VideoWriter(
    'matches_output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),20,(1280, 480)           
)

for i in range(len(imgs) - 1):
    img1, img2 = imgs[i], imgs[i + 1]
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        continue

    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    vis = cv2.resize(vis, (1280, 480))
    video_out.write(vis)
    cv2.imshow('ORB Matching Video', vis)
    if cv2.waitKey(30) & 0xFF == 27:
        break

video_out.release()
cv2.destroyAllWindows()
