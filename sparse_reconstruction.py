"""
sparse_reconstruction_with_trajectory.py

Produces a sparse 3D reconstruction from a keyframe-based visual odometry pipeline,
and visualizes the trajectory and sparse points using Plotly.

Files required in same folder:
 - config.py  (DATASET_PATH, K, ORB_NFEATURES)
 - utils.py   (load_images_from_folder, Rt_to_transform)

Run:
 python sparse_reconstruction_with_trajectory.py
"""
import os
import cv2
import numpy as np
import plotly.graph_objects as go
from utils import load_images_from_folder, Rt_to_transform
import config

# ------- Parameters (tune these) -------
KEYFRAME_STEP = 3       # use every 3rd frame as a keyframe
MIN_MATCHES = 30        # min good matches to accept triangulation
MAX_POINT_DEPTH = 100.0 # max z-distance for points (in meters)
VO_MATCH_THRESHOLD = 0.75
DUPLICATE_VOXEL = 0.01  # voxel size (m) used for deduplication

# ------- Prepare ORB & matcher -------
orb = cv2.ORB_create(nfeatures=config.ORB_NFEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
K = config.K.astype(np.float64)

# ------- Load images -------
imgs, files = load_images_from_folder(config.DATASET_PATH)
if len(imgs) == 0:
    raise ValueError(f"No images found in {config.DATASET_PATH}")

print(f"Loaded {len(imgs)} frames.")

# ------- Build keyframes and compute VO (poses) -------
keyframes = []      # list of (idx, kp, des, img)
poses = [np.eye(4)] # world poses; pose_0 = identity

# Initialize with frame 0 as keyframe
kp_prev, des_prev = orb.detectAndCompute(imgs[0], None)
keyframes.append((0, kp_prev, des_prev, imgs[0]))

# We'll compute relative transforms between consecutive keyframes and integrate to get poses.
prev_pose = np.eye(4)

for i in range(KEYFRAME_STEP, len(imgs), KEYFRAME_STEP):
    img_cur = imgs[i]
    kp_cur, des_cur = orb.detectAndCompute(img_cur, None)

    if des_prev is None or des_cur is None:
        print(f"Frame {i}: skipping (no descriptors).")
        # still append last known pose as placeholder to keep indices aligned
        poses.append(prev_pose.copy())
        continue

    # match descriptors between last keyframe and current frame
    matches = bf.knnMatch(des_prev, des_cur, k=2)
    good = [m for m, n in matches if m.distance < VO_MATCH_THRESHOLD * n.distance]

    if len(good) < MIN_MATCHES:
        print(f"Frame {i}: insufficient matches ({len(good)}); skipping.")
        poses.append(prev_pose.copy())
        # still keep current as keyframe to attempt triang later? we'll update anyway
        keyframes.append((i, kp_cur, des_cur, img_cur))
        kp_prev, des_prev = kp_cur, des_cur
        continue

    pts1 = np.float32([kp_prev[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp_cur[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print(f"Frame {i}: essential matrix not found; skipping.")
        poses.append(prev_pose.copy())
        keyframes.append((i, kp_cur, des_cur, img_cur))
        kp_prev, des_prev = kp_cur, des_cur
        continue

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    T_rel = Rt_to_transform(R, t)                 # transform from cur_cam -> prev_cam convention from recoverPose
    # integrate to world: new_world = prev_world * inv(T_rel)
    new_pose = prev_pose @ np.linalg.inv(T_rel)
    poses.append(new_pose)
    prev_pose = new_pose.copy()

    # store keyframe
    keyframes.append((i, kp_cur, des_cur, img_cur))
    kp_prev, des_prev = kp_cur, des_cur

    print(f"Keyframe {i} added. Matches: {len(good)}. Pose #{len(poses)-1} computed.")

# If there are fewer poses than keyframes due to skipping, ensure we have same count by padding
# (Not strictly necessary — just a safeguard)
while len(poses) < len(keyframes):
    poses.append(poses[-1].copy())

# ------- Triangulate between consecutive keyframes and accumulate 3D points -------
all_points = []
all_colors = []  # optional: color sampling from image (if using color images)
used_pairs = 0

for k in range(len(keyframes)-1):
    idx1, kp1, des1, img1 = keyframes[k]
    idx2, kp2, des2, img2 = keyframes[k+1]

    # Match between the two keyframes
    if des1 is None or des2 is None:
        continue
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < VO_MATCH_THRESHOLD * n.distance]

    if len(good) < MIN_MATCHES:
        # skip pairs with few matches
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Estimate essential and recover finer pose just for triangulation stability
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        continue
    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    # Use the previously computed absolute poses instead of chaining R_rel/t_rel
    pose1 = poses[k]   # world <- cam1
    pose2 = poses[k+1] # world <- cam2

    # camera projection matrices for triangulation (camera coordinates)
    # P_cam = K [R|t] where R,t are camera-to-world? We need projection from camera to image.
    # We want P1 and P2 such that X_image = P * X_cam.
    # For triangulatePoints we feed projection matrices of the two cameras relative to same world.
    # Let world_T_cam = inverse(world_pose) -> cam_T_world
    # But simpler: We want P = K * [R_cam, t_cam] where R_cam and t_cam are rotation and translation from world to camera.
    def world_to_cam_pose(world_pose):
        # world_pose is 4x4 that maps local camera frame -> world (world = world_pose * cam)
        # invert it to get world->cam
        cam_T_world = np.linalg.inv(world_pose)
        R = cam_T_world[:3,:3]
        t = cam_T_world[:3,3].reshape(3,1)
        return R, t

    R1, t1 = world_to_cam_pose(pose1)
    R2, t2 = world_to_cam_pose(pose2)

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    # triangulate (expects 2xN float points)
    points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # shape (4, N)
    pts3d = (points4d[:3, :] / points4d[3, :]).T  # shape (N, 3)

    # filter by finite and reasonable depth (in camera coordinates)
    # compute depth in each camera: Z = (R * X + t)[2]
    cam2_coords = (R2 @ pts3d.T + t2).T
    cam1_coords = (R1 @ pts3d.T + t1).T

    valid_mask = np.isfinite(pts3d).all(axis=1) & (cam1_coords[:,2] > 0.1) & (cam1_coords[:,2] < MAX_POINT_DEPTH) & \
                 (cam2_coords[:,2] > 0.1) & (cam2_coords[:,2] < MAX_POINT_DEPTH)

    pts3d = pts3d[valid_mask]
    if pts3d.shape[0] == 0:
        continue

    # Optionally sample color from first keyframe if color images are available (we used grayscale in loader)
    # Here we try to access color image if exists on disk; if not, use gray intensity as color.
    colors = None
    try:
        # try to read color versions if available (same filename)
        img1_color_path = os.path.join(config.DATASET_PATH, files[idx1])
        img1_color = cv2.imread(img1_color_path, cv2.IMREAD_COLOR)
        if img1_color is not None:
            colors = []
            for m_idx, m in enumerate(good):
                if not valid_mask[m_idx]:
                    continue
                pt = kp1[m.queryIdx].pt
                x, y = int(round(pt[0])), int(round(pt[1]))
                h, w = img1_color.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    bgr = img1_color[y, x]
                    colors.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))
                else:
                    colors.append((200,200,200))
            colors = np.array(colors)
    except Exception:
        colors = None

    # Append points to global list
    all_points.append(pts3d)
    if colors is not None and colors.shape[0] == pts3d.shape[0]:
        all_colors.append(colors)
    else:
        all_colors.append(np.full((pts3d.shape[0], 3), 200, dtype=np.uint8))

    used_pairs += 1
    print(f"Triangulated {pts3d.shape[0]} pts from keyframe pair ({idx1},{idx2}).")

print(f"Triangulated from {used_pairs} pairs.")

# ------- Merge and deduplicate points -------
if len(all_points) == 0:
    print("No 3D points were triangulated. Exiting.")
    exit(0)

all_points = np.vstack(all_points)
all_colors = np.vstack(all_colors)

# voxel-grid deduplication: round coords to nearest voxel and unique
vox = DUPLICATE_VOXEL
keys = np.round(all_points / vox).astype(np.int32)
# create unique index by converting rows to bytes
_, unique_idx = np.unique(keys.view([('', keys.dtype)]*keys.shape[1]), return_index=True)
unique_points = all_points[unique_idx]
unique_colors = all_colors[unique_idx]

print(f"Total 3D points before dedup: {all_points.shape[0]}, after dedup: {unique_points.shape[0]}")

# ------- Save PLY -------
def write_ply(filename, points, colors=None):
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        if colors is not None:
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            p = points[i]
            if colors is not None:
                c = colors[i].astype(int)
                f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

ply_name = "sparse_map.ply"
write_ply(ply_name, unique_points, unique_colors)
print(f"Saved PLY: {ply_name}")

# ------- Save trajectory -------
trajectory = np.array([[p[0,3], p[1,3], p[2,3]] for p in poses])
np.savetxt("trajectory.txt", trajectory, fmt="%.6f")
print("Saved trajectory.txt")

# ------- Plot with Plotly -------
fig = go.Figure()

# plot sparse points
fig.add_trace(go.Scatter3d(
    x=unique_points[:,0],
    y=unique_points[:,1],
    z=unique_points[:,2],
    mode='markers',
    marker=dict(size=2, color=['rgb(%d,%d,%d)'%(c[0],c[1],c[2]) for c in unique_colors], opacity=0.8),
    name='Sparse Points'
))

# plot trajectory
fig.add_trace(go.Scatter3d(
    x=trajectory[:,0],
    y=trajectory[:,1],
    z=trajectory[:,2],
    mode='lines+markers',
    line=dict(width=4, color='blue'),
    marker=dict(size=3, color=np.linspace(0,1,trajectory.shape[0]), colorscale='Viridis'),
    name='Trajectory'
))

# start/end markers
fig.add_trace(go.Scatter3d(x=[trajectory[0,0]], y=[trajectory[0,1]], z=[trajectory[0,2]],
                          mode='markers', marker=dict(color='green', size=6), name='Start'))
fig.add_trace(go.Scatter3d(x=[trajectory[-1,0]], y=[trajectory[-1,1]], z=[trajectory[-1,2]],
                          mode='markers', marker=dict(color='red', size=6), name='End'))

fig.update_layout(title='Sparse 3D Reconstruction + Trajectory',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  margin=dict(l=0, r=0, b=0, t=40))
fig.show()
