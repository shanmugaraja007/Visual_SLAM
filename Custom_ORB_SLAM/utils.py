import cv2
import numpy as np
import os
from math import sin, cos
import matplotlib.pyplot as plt

def load_images_from_folder(folder, ext=('png','jpg','jpeg')):
    imgs = []
    files = [f for f in sorted(os.listdir(folder)) if f.split('.')[-1].lower() in ext]
    for f in files:
            imgs.append(cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR))
    return imgs, files


def Rt_to_transform(R, t):
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.ravel()
        return T


def transform_to_Rt(T):
        R = T[:3,:3]
        t = T[:3,3].reshape(3,1)
        return R, t


def draw_trajectory(poses, show=True, save_path=None):
    import matplotlib.pyplot as plt
    xs = [p[0,3] for p in poses]
    zs = [p[2,3] for p in poses]
    plt.figure(figsize=(6,6))
    plt.plot(xs, zs, '-o', color='blue', markersize=3)
    plt.scatter(xs[0], zs[0], color='green', s=80, label='Start')
    plt.scatter(xs[-1], zs[-1], color='red', s=80, label='End')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title('Trajectory (x vs z)')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    if show:
        plt.show()


def kps_to_np(kps):
   return np.array([kp.pt for kp in kps], dtype=np.float32)