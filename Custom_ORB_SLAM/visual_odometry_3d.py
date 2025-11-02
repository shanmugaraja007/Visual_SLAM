import cv2
import numpy as np
import plotly.graph_objects as go
from utils import load_images_from_folder, Rt_to_transform
import config



imgs, files = load_images_from_folder(config.DATASET_PATH)
orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
K = config.K

poses = [np.eye(4)]



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



trajectory = np.array([[p[0, 3], p[1, 3], p[2, 3]] for p in poses])


np.savetxt("trajectory.txt", trajectory, fmt="%.6f")
print(" Trajectory saved as trajectory.txt")



fig = go.Figure()


fig.add_trace(go.Scatter3d(
    x=trajectory[:, 0],
    y=trajectory[:, 1],
    z=trajectory[:, 2],
    mode='lines+markers',
    line=dict(color='royalblue', width=4),
    marker=dict(size=4, color=np.linspace(0, 1, len(trajectory)), colorscale='Viridis'),
    name='Trajectory'
))


fig.add_trace(go.Scatter3d(
    x=[trajectory[0, 0]],
    y=[trajectory[0, 1]],
    z=[trajectory[0, 2]],
    mode='markers',
    marker=dict(color='green', size=8, symbol='circle'),
    name='Start'
))

fig.add_trace(go.Scatter3d(
    x=[trajectory[-1, 0]],
    y=[trajectory[-1, 1]],
    z=[trajectory[-1, 2]],
    mode='markers',
    marker=dict(color='red', size=8, symbol='diamond'),
    name='End'
))


fig.update_layout(
    title='3D Visual Odometry Trajectory (Plotly)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=True
)

fig.show()