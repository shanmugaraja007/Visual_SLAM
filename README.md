# Python ORB Visual SLAM Framework

This repository implements a **basic Visual SLAM framework in Python** using ORB features.
Unlike **ORB-SLAM2**, which is written in C++ and optimized for speed, this project demonstrates the **conceptual workflow of feature-based SLAM** in Python, including:

* Feature detection
* Feature matching
* Visual odometry (pose estimation)
* Loop closure detection
* Pose Graph Optimization(in-progress)
* Sparse 3D reconstruction

The code is modular, easy to understand, and provides visualization of both **camera trajectory** and **sparse 3D map points**.


##  Requirements

Python 3.10.4 and the following libraries:

```bash
pip install opencv-python numpy plotly
```

Also, include:

* `config.py` → dataset path, ORB parameters, camera intrinsic matrix `K`
* `utils.py` → helper functions: `load_images_from_folder()`, `Rt_to_transform()`

---

## 📷 Camera Intrinsics

This project uses **pinhole camera model** intrinsics, defined as:

```python
FX = 718.856
FY = 718.856
CX = 607.1928
CY = 185.2157

K = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
])
```

* `FX` / `FY` → focal lengths in pixels along X and Y axes
* `CX` / `CY` → principal point coordinates (optical center)
* `K` → camera intrinsic matrix

>  **Important:** These values correspond to the **KITTI outdoor dataset**.
> For your own dataset or camera, you must **replace these intrinsics** with the correct values.

**How to update:**

1. **For a calibrated camera:** Use the intrinsic matrix from camera calibration.
2. **For new datasets:** Check dataset documentation; most datasets provide `K`.
3. **Update in** `config.py`:

```python
K = np.array([
    [YOUR_FX, 0, YOUR_CX],
    [0, YOUR_FY, YOUR_CY],
    [0, 0, 1]
])
```

**Why it matters:**
The intrinsic matrix is used for:

* Computing the **Essential matrix** (`cv2.findEssentialMat`)
* Recovering **relative camera pose** (`cv2.recoverPose`)
* Triangulating 3D points

Using incorrect intrinsics will lead to **incorrect scale, trajectory, and 3D reconstruction**.


### 1️. `feature_detection.py`

     python feature_detection.py
     
     
Detect and visualize ORB keypoints in each frame.



**Output:**

* Video: `kp_detection_output.mp4` showing keypoints per frame.
<video width="640" controls>
  <source src="kp_detection_output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
**Improvements:**

* Use other detectors (SIFT, AKAZE) for more robustness.
* Store descriptors for downstream tasks (matching, pose estimation).

---

### 2️⃣ `feature_matching.py`

**Purpose:**
Match ORB features between consecutive frames for motion analysis.

**Algorithm:**

1. Detect and compute ORB descriptors for each frame.
2. Use **Brute-Force Matcher** with Hamming distance.
3. Perform KNN matching (`k=2`).
4. Apply **Lowe’s ratio test (0.75)** to remove ambiguous matches.
5. Visualize matches with `cv2.drawMatches()`.

**Output:**

* Video: `matches_output.mp4` showing matched features.

**Improvements:**

* Apply **RANSAC with Essential/Fundamental matrix** for robust outlier removal.
* Integrate matched points into **pose estimation** pipeline.

---

### 3️⃣ `visual_odometry.py`

**Purpose:**
Estimate camera trajectory using consecutive ORB feature matches.

**Algorithm:**

1. Detect ORB keypoints and compute descriptors.
2. Match descriptors between consecutive frames.
3. Filter matches using Lowe’s ratio test.
4. Compute **Essential Matrix** with `cv2.findEssentialMat()`.
5. Recover relative pose (`R, t`) using `cv2.recoverPose()`.
6. Integrate poses to get camera trajectory in world coordinates.
7. Visualize trajectory using Plotly.

**Output:**

* 3D camera trajectory visualization.
* `trajectory.txt` storing XYZ positions per frame.

**Improvements:**

* Incorporate **bundle adjustment** for more accurate trajectory.
* Handle missing descriptors with interpolation or keyframe selection.

---

### 4️⃣ `loop_closure.py`

**Purpose:**
Detect potential loop closures to improve pose accuracy.

**Algorithm:**

1. Maintain a list of **keyframes** (every N-th frame).
2. Compare current frame with past keyframes beyond a temporal threshold.
3. Match ORB features and filter with Lowe’s ratio test.
4. Verify geometric consistency using `cv2.findFundamentalMat()` + RANSAC.
5. Report strong loop closure candidates.

**Output:**

* Prints loop closure frames and inliers count.

**Improvements:**

* Use BoW (Bag-of-Words) for faster candidate retrieval.
* Integrate confirmed loop closures into pose graph optimization.

---

### 5️⃣ `sparse_reconstruction_with_trajectory.py`

**Purpose:**
Build a **sparse 3D map** using keyframe triangulation and visualize trajectory + points.

**Algorithm:**

1. Select keyframes at a fixed step.
2. Match ORB features between consecutive keyframes.
3. Filter matches using Lowe’s ratio test and RANSAC.
4. Recover relative pose and integrate to get absolute pose.
5. Triangulate matched points using camera projection matrices.
6. Deduplicate points using voxel grid filtering.
7. Save 3D points as **PLY** and trajectory as **TXT**.
8. Visualize with Plotly (trajectory + sparse map points).

**Output:**

* `sparse_map.ply` → 3D point cloud.
* `trajectory.txt` → camera trajectory.
* Interactive 3D Plotly visualization.

**Improvements:**

* Integrate color information for RGB point cloud.
* Implement **dense reconstruction** using multi-view stereo.
* Add **bundle adjustment** to improve triangulation accuracy.

---

## ▶ Running the Project

```bash
# Step 1: Feature detection
python feature_detection.py

# Step 2: Feature matching
python feature_matching.py

# Step 3: Visual Odometry
python visual_odometry.py

# Step 4: Loop closure detection (optional)
python loop_closure.py

# Step 5: Sparse 3D reconstruction + trajectory
python sparse_reconstruction_with_trajectory.py
```

---

## 📷 Outputs

| Script                                   | Output                                                           |
| ---------------------------------------- | ---------------------------------------------------------------- |
| feature_detection.py                     | `kp_detection_output.mp4`                                        |
| feature_matching.py                      | `matches_output.mp4`                                             |
| visual_odometry.py                       | 3D trajectory Plotly visualization, `trajectory.txt`             |
| loop_closure.py                          | Console logs of detected loop closures                           |
| sparse_reconstruction_with_trajectory.py | Sparse 3D point cloud `sparse_map.ply`, trajectory visualization |

---

## ⚡ Improvements & Future Work

1. Implement **RANSAC + bundle adjustment** for more accurate pose estimation.
2. Replace ORB with **SIFT / AKAZE / SuperPoint** for robustness in textureless environments.
3. Use **Pose Graph Optimization** after loop closure detection.
4. Upgrade to **dense reconstruction** for complete 3D scene mapping.
5. Integrate with **ROS** for real-time Visual SLAM pipeline.

---

## 📚 References

1. Mur-Artal, R., Montiel, J.M.M., Tardós, J.D. *ORB-SLAM2: An
