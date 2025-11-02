import numpy as np
from scipy.optimize import least_squares


def v2t(v):
        x,y,th = v
        c = np.cos(th); s = np.sin(th)
        T = np.array([[c,-s,x],[s,c,y],[0,0,1]])
        return T


def t2v(T):
    x = T[0,2]; y = T[1,2]
    th = np.arctan2(T[1,0], T[0,0])
    return np.array([x,y,th])


def residuals(x, edges):
    res = []
    for (i,j,rel) in edges:
        xi = x[3*i:3*i+3]
        xj = x[3*j:3*j+3]
        Ti = v2t(xi); Tj = v2t(xj)
        Tij = np.linalg.inv(Ti).dot(Tj)
        rel_est = t2v(Tij)
        res.extend(rel_est - rel)
    return np.array(res)


if __name__=='__main__':
    N=6
    poses = [np.array([i*1.0, 0.0, 0.0]) for i in range(N)]
    x0 = np.hstack(poses)
    edges = [(i, i+1, np.array([1.0, 0.0, 0.0])) for i in range(N-1)]
    edges.append((N-1, 0, np.array([-5.0, 0.1, 0.02])))
    res = least_squares(residuals, x0, args=(edges,))
    x_opt = res.x
    for i in range(N):
      print('Pose', i, x_opt[3*i:3*i+3])