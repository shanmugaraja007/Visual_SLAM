import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_images_from_folder, Rt_to_transform
import config



imgs, files = load_images_from_folder(config.DATASET_PATH)
orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

poses = [np.eye(4)]
K = config.K



plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Visual Odometry Trajectory (x vs z)")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.grid(True)
ax.invert_yaxis()

trajectory_line, = ax.plot([], [], "-o", color="blue", markersize=3, label="Trajectory")
start_marker = ax.scatter([], [], color="green", s=80, label="Start")
end_marker = ax.scatter([], [], color="red", s=80, label="End")
ax.legend()



xs, zs = [], []



for i in range(len(imgs) - 1):
    img1, img2 = imgs[i], imgs[i + 1]

    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        poses.append(poses[-1])
        continue

    
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        poses.append(poses[-1])
        continue

    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        poses.append(poses[-1])
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    T = Rt_to_transform(R, t)
    newpose = poses[-1] @ np.linalg.inv(T)
    poses.append(newpose)

    
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imshow("ORB Keypoint Matches", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    
    xs = [p[0, 3] for p in poses]
    zs = [p[2, 3] for p in poses]
    trajectory_line.set_xdata(xs)
    trajectory_line.set_ydata(zs)

    
    start_marker.set_offsets([[xs[0], zs[0]]])
    end_marker.set_offsets([[xs[-1], zs[-1]]])

    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)


plt.ioff()
plt.show()
cv2.destroyAllWindows()



trajectory_array = np.array([[p[0, 3], p[1, 3], p[2, 3]] for p in poses])
np.savetxt("trajectory.txt", trajectory_array, fmt="%.6f")

print("Trajectory saved as trajectory.txt")
print("Visualization complete")
