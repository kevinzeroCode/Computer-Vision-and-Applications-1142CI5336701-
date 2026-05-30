"""
error_map.py   (EVALUATION ONLY -- GroundTruth.ply is read here and ONLY here)

Quantify the reconstruction against GroundTruth.ply:
  * densely sample the GT mesh faces -> dense reference cloud (handles the
    coarsely-tessellated turntable correctly);
  * align the reconstruction to GT by a rotation about z (the only free DOF:
    both share origin, scale and the z axis -- the offset is the unknown
    frame-0 turntable azimuth), refined by a few rigid-ICP steps.  This
    alignment is a *diagnostic*; it is never fed back into reconstruct.py;
  * report recon->GT (accuracy) and GT->recon (coverage) distances;
  * write a jet-coloured error PLY and a histogram PNG.

Usage:  python error_map.py
"""

import numpy as np
import os, re
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GT_PLY = os.path.join(HERE, 'GroundTruth.ply')
RECON  = os.path.join(HERE, 'M11415015.xyz')


def load_gt_mesh(path):
    data = open(path, 'rb').read()
    he = data.find(b'end_header\n') + len(b'end_header\n')
    nv = int(re.search(rb'element vertex (\d+)', data).group(1))
    nf = int(re.search(rb'element face (\d+)', data).group(1))
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])
    V = np.frombuffer(data, dtype=dt, count=nv, offset=he)
    verts = np.stack([V['x'], V['y'], V['z']], 1).astype(np.float64)
    # faces: variable-length list (uchar count + count*uint32); fan-triangulate
    off = he + nv * 16
    buf = np.frombuffer(data, dtype=np.uint8, offset=off)
    tris = []
    p = 0
    for _ in range(nf):
        n = buf[p]
        ids = buf[p + 1:p + 1 + 4 * n].view('<u4')
        for k in range(1, n - 1):
            tris.append((ids[0], ids[k], ids[k + 1]))
        p += 1 + 4 * n
    return verts, np.array(tris, dtype=np.int64)


def sample_mesh(verts, faces, n=400000):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    prob = area / area.sum()
    pick = np.random.choice(len(faces), size=n, p=prob)
    u = np.random.rand(n, 1); w = np.random.rand(n, 1)
    over = (u + w > 1).ravel(); u[over] = 1 - u[over]; w[over] = 1 - w[over]
    a, b, c = v0[pick], v1[pick], v2[pick]
    return a + u * (b - a) + w * (c - a)


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def icp(src, tree, iters=12):
    """Rigid point-to-point ICP refinement (src -> GT tree). Returns aligned src."""
    P = src.copy()
    for _ in range(iters):
        d, idx = tree.query(P)
        Q = tree.data[idx]
        mp, mq = P.mean(0), Q.mean(0)
        H = (P - mp).T @ (Q - mq)
        U, _, Vt = np.linalg.svd(H)
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1; Rm = Vt.T @ U.T
        t = mq - Rm @ mp
        P = (Rm @ P.T).T + t
    return P


def main():
    np.random.seed(0)
    recon = np.loadtxt(RECON)
    rec = recon[:, :3]
    print(f'Reconstruction: {len(rec)} points')

    verts, faces = load_gt_mesh(GT_PLY)
    gt = sample_mesh(verts, faces)
    print(f'GT mesh: {len(verts)} verts, {len(faces)} faces -> {len(gt)} sampled pts')
    tree_gt = cKDTree(gt)

    # ---- align: yaw search (+ optional mirror) then rigid ICP (diagnostic) ----
    best = None
    sub = rec[np.random.choice(len(rec), min(8000, len(rec)), replace=False)]
    for mirror in (1, -1):
        S = sub.copy(); S[:, 0] *= mirror
        for yaw in range(0, 360, 2):
            P = (rot_z(yaw) @ S.T).T
            d, _ = tree_gt.query(P)
            m = d.mean()
            if best is None or m < best[0]:
                best = (m, yaw, mirror)
    _, yaw, mirror = best
    print(f'Coarse align: mirror={mirror} yaw={yaw} deg (mean d={best[0]:.3f} mm)')
    rec_a = rec.copy(); rec_a[:, 0] *= mirror
    rec_a = (rot_z(yaw) @ rec_a.T).T
    rec_a = icp(rec_a, tree_gt)

    # ---- bidirectional distances ----
    d_r2g, _ = tree_gt.query(rec_a)                       # accuracy
    tree_rec = cKDTree(rec_a)
    d_g2r, _ = tree_rec.query(gt)                         # coverage

    def stats(d, tag):
        print(f'  {tag:9s} mean={d.mean():.3f}  median={np.median(d):.3f}  '
              f'RMS={np.sqrt((d**2).mean()):.3f}  95%={np.percentile(d,95):.3f}  '
              f'max={d.max():.3f} mm')
    print('Distances (mm):')
    stats(d_r2g, 'recon->GT')
    stats(d_g2r, 'GT->recon')
    cov = (d_g2r < 1.0).mean() * 100
    print(f'  Coverage (GT within 1.0 mm of a recon point): {cov:.1f}%')

    # ---- coloured error PLY (recon points, jet by recon->GT) ----
    cmax = np.percentile(d_r2g, 98)
    t = np.clip(d_r2g / max(cmax, 1e-6), 0, 1)
    cmap = (plt.get_cmap('jet')(t)[:, :3] * 255).astype(np.uint8)
    write_ply(os.path.join(HERE, 'error_map.ply'), rec_a, cmap)

    # ---- histogram ----
    plt.figure(figsize=(7, 4))
    plt.hist(d_r2g, bins=120, range=(0, np.percentile(d_r2g, 99)),
             color='#3070b0', edgecolor='none')
    plt.axvline(np.median(d_r2g), color='r', ls='--',
                label=f'median={np.median(d_r2g):.3f} mm')
    plt.xlabel('recon -> GT distance (mm)'); plt.ylabel('count')
    plt.title('Reconstruction accuracy vs GroundTruth'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'error_hist.png'), dpi=110)
    print('Saved error_map.ply, error_hist.png')


def write_ply(path, pts, cols):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n')


if __name__ == '__main__':
    main()
