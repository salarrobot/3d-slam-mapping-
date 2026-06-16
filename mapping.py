"""
3D occupancy-grid mapping (log-odds).

The robot poses are known (this is the *mapping* half of SLAM), and every
LiDAR scan is fused into a voxel grid using the classic log-odds update:

  * every voxel the ray passes through before its end point gets evidence
    that it is FREE   (l += l_free),
  * the voxel at the ray's end point gets evidence that it is OCCUPIED
    (l += l_occ),

clamped to keep the map responsive. A voxel is "occupied" once its
log-odds exceed a threshold.

Reference: occupancy grid mapping, Thrun/Burgard/Fox ch. 9; OctoMap.
"""

import numpy as np


class OccupancyGrid3D:
    def __init__(self, world, res=0.4,
                 l_occ=0.85, l_free=0.4, clamp=6.0, occ_thresh=0.0):
        self.res = res
        self.min = world.min.copy()
        self.max = world.max.copy()
        self.dims = np.ceil((self.max - self.min) / res).astype(int)
        self.log = np.zeros(self.dims, dtype=np.float32)

        self.l_occ = l_occ
        self.l_free = l_free
        self.clamp = clamp
        self.occ_thresh = occ_thresh

    # ----------------------- coordinate helpers ------------------------
    def to_idx(self, pts):
        return np.floor((pts - self.min) / self.res).astype(int)

    def in_bounds(self, idx):
        return np.all((idx >= 0) & (idx < self.dims), axis=1)

    # --------------------------- integration ---------------------------
    def integrate(self, origin, dirs, dists, hit, max_range):
        """Fuse one scan (vectorized over all rays)."""
        # --- free space: sample points along every ray up to its hit ---
        step = self.res * 0.5
        S = int(np.ceil(max_range / step))
        t = (np.arange(S) + 0.5) * step                      # [S]
        pts = origin[None, None, :] + t[None, :, None] * dirs[:, None, :]
        free = t[None, :] < (dists[:, None] - self.res)      # [R, S]
        fidx = self.to_idx(pts[free])
        m = self.in_bounds(fidx)
        fidx = fidx[m]
        np.subtract.at(self.log, (fidx[:, 0], fidx[:, 1], fidx[:, 2]),
                       self.l_free)

        # --- occupied: the end point of every real return ---
        ends = origin[None, :] + dists[hit, None] * dirs[hit]
        oidx = self.to_idx(ends)
        m2 = self.in_bounds(oidx)
        oidx = oidx[m2]
        np.add.at(self.log, (oidx[:, 0], oidx[:, 1], oidx[:, 2]), self.l_occ)

        np.clip(self.log, -self.clamp, self.clamp, out=self.log)

    # ----------------------------- queries -----------------------------
    def occupied_mask(self):
        return self.log > self.occ_thresh

    def occupied_centers(self):
        """World coordinates of the centres of all occupied voxels."""
        ijk = np.argwhere(self.occupied_mask())
        return self.min + (ijk + 0.5) * self.res

    def n_occupied(self):
        return int(np.count_nonzero(self.occupied_mask()))

    def top_down(self):
        """2D floor-plan: highest log-odds over z at each (x, y) column."""
        return self.log.max(axis=2).T            # [Ny, Nx] for imshow
