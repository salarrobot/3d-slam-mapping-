"""
3D simulation world for the mapping robot.

Contains:
  * Box           - an axis-aligned 3D obstacle / wall (with height),
                    plus a slab-method ray-cast and a wire-frame drawer.
  * World         - the room: outer walls, interior obstacles and a floor,
                    with a ray-cast that returns the nearest surface hit.
  * Drone         - a flying robot that follows a smooth 3D loop.
  * Lidar3D       - a spherical 3D LiDAR that returns a point cloud.

Conventions:  pose = [x, y, z, yaw].  All lengths in metres, angles in rad.
"""

from dataclasses import dataclass

import numpy as np


# =====================================================================
# ----------------------------- PARAMETERS ----------------------------
# =====================================================================
WORLD_MIN = np.array([0.0, 0.0, 0.0])
WORLD_MAX = np.array([16.0, 12.0, 4.0])
WALL_H = 4.0                # height of the outer walls
WALL_T = 0.3               # wall thickness


# =====================================================================
# ------------------------------- BOX ---------------------------------
# =====================================================================
@dataclass
class Box:
    """Axis-aligned box defined by its min and max corner."""
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    color: str = "#6c7a89"

    @property
    def lo(self):
        return np.array([self.xmin, self.ymin, self.zmin])

    @property
    def hi(self):
        return np.array([self.xmax, self.ymax, self.zmax])

    def ray(self, o, d, t_max):
        """Slab method. Returns distance to first forward hit, or None."""
        tmin, tmax = 1e-6, t_max
        lo, hi = self.lo, self.hi
        for i in range(3):
            if abs(d[i]) < 1e-12:
                if o[i] < lo[i] or o[i] > hi[i]:
                    return None
            else:
                inv = 1.0 / d[i]
                t1 = (lo[i] - o[i]) * inv
                t2 = (hi[i] - o[i]) * inv
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        return tmin

    def edges(self):
        """The 12 edges of the box as pairs of 3D points (for wire-frame)."""
        x0, y0, z0 = self.xmin, self.ymin, self.zmin
        x1, y1, z1 = self.xmax, self.ymax, self.zmax
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        idx = [(0, 1), (1, 2), (2, 3), (3, 0),
               (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)]
        return [(c[a], c[b]) for a, b in idx]

    def draw(self, ax, color=None, lw=1.0, alpha=0.6):
        col = color or self.color
        for (p, q) in self.edges():
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                    color=col, lw=lw, alpha=alpha)


# =====================================================================
# ------------------------------ WORLD --------------------------------
# =====================================================================
def _build_room():
    """Outer walls (with height), interior obstacles of varied heights."""
    X0, Y0, _ = WORLD_MIN
    X1, Y1, _ = WORLD_MAX
    t, h = WALL_T, WALL_H
    walls = [
        Box(X0, Y0, 0, X1, Y0 + t, h, "#8a96a3"),        # south wall
        Box(X0, Y1 - t, 0, X1, Y1, h, "#8a96a3"),        # north wall
        Box(X0, Y0, 0, X0 + t, Y1, h, "#8a96a3"),        # west wall
        Box(X1 - t, Y0, 0, X1, Y1, h, "#8a96a3"),        # east wall
    ]
    interior = [
        # central tall pillar the drone orbits around
        Box(7.3, 5.3, 0, 8.7, 6.7, 3.5, "#c0653a"),
        # corner furniture / crates of different heights (outside flight ring)
        Box(1.5, 1.5, 0, 3.5, 3.5, 1.2, "#3a78c0"),
        Box(12.5, 8.5, 0, 14.5, 10.5, 1.5, "#3a78c0"),
        Box(1.5, 8.5, 0, 3.5, 10.5, 2.0, "#3a78c0"),
        Box(12.5, 1.5, 0, 14.5, 3.5, 2.5, "#3a78c0"),
        # two thin partition walls near the long walls
        Box(8.0, 9.6, 0, 11.0, 10.2, 2.5, "#4aa06a"),
        Box(5.0, 1.8, 0, 8.0, 2.4, 2.0, "#4aa06a"),
    ]
    return walls + interior


class World:
    def __init__(self):
        self.boxes = _build_room()
        self.min = WORLD_MIN.copy()
        self.max = WORLD_MAX.copy()

    def raycast(self, o, d, t_max):
        """Nearest hit distance along ray (origin o, unit dir d), clamped
        to t_max. Returns (dist, hit_bool)."""
        best = t_max
        for b in self.boxes:
            t = b.ray(o, d, best)
            if t is not None and t < best:
                best = t
        # floor plane z = 0 (only counts inside the room footprint)
        if abs(d[2]) > 1e-9:
            tf = (0.0 - o[2]) / d[2]
            if 1e-6 < tf < best:
                hx, hy = o[0] + tf * d[0], o[1] + tf * d[1]
                if self.min[0] <= hx <= self.max[0] and \
                   self.min[1] <= hy <= self.max[1]:
                    best = tf
        return best, best < t_max - 1e-6

    def draw(self, ax, lw=1.0, alpha=0.45):
        for b in self.boxes:
            b.draw(ax, lw=lw, alpha=alpha)


# =====================================================================
# ------------------------------ DRONE --------------------------------
# =====================================================================
class Drone:
    """Flies a smooth elliptical loop around the central pillar, gently
    changing altitude. Deterministic given the step count."""

    def __init__(self, n_steps, loops=2.0):
        self.n = n_steps
        self.loops = loops
        self.cx, self.cy = 8.0, 6.0
        self.rx, self.ry = 4.0, 2.5
        self.z0, self.zamp = 2.0, 0.9
        self.k = 0

    def pose_at(self, k):
        ang = 2.0 * np.pi * self.loops * (k / self.n)
        x = self.cx + self.rx * np.cos(ang)
        y = self.cy + self.ry * np.sin(ang)
        z = self.z0 + self.zamp * np.sin(2.0 * ang)
        # yaw = travel direction (tangent of the ellipse)
        yaw = np.arctan2(self.ry * np.cos(ang), -self.rx * np.sin(ang))
        return np.array([x, y, z, yaw])

    def step(self):
        p = self.pose_at(self.k)
        self.k += 1
        return p


# =====================================================================
# ----------------------------- LIDAR 3D ------------------------------
# =====================================================================
class Lidar3D:
    """Spherical LiDAR: a grid of azimuth x elevation rays."""

    def __init__(self, world, n_az=36, n_el=12, max_range=14.0,
                 el_min_deg=-55.0, el_max_deg=55.0, range_std=0.025, seed=1):
        self.world = world
        self.max_range = max_range
        self.range_std = range_std
        self.rng = np.random.default_rng(seed)

        az = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
        el = np.deg2rad(np.linspace(el_min_deg, el_max_deg, n_el))
        AZ, EL = np.meshgrid(az, el)
        AZ, EL = AZ.ravel(), EL.ravel()
        # ray directions in the sensor frame (yaw applied per scan)
        self.base_dirs = np.stack(
            [np.cos(EL) * np.cos(AZ),
             np.cos(EL) * np.sin(AZ),
             np.sin(EL)], axis=1)

    def scan(self, pose):
        x, y, z, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        dirs = self.base_dirs @ Rz.T
        origin = np.array([x, y, z])

        dists = np.empty(len(dirs))
        hit = np.empty(len(dirs), dtype=bool)
        for i in range(len(dirs)):
            t, h = self.world.raycast(origin, dirs[i], self.max_range)
            dists[i], hit[i] = t, h

        # range noise on real returns
        n = int(hit.sum())
        if n:
            dists[hit] += self.rng.normal(0.0, self.range_std, n)
        dists = np.clip(dists, 0.05, self.max_range)
        points = origin + dists[:, None] * dirs
        return {"origin": origin, "dirs": dirs, "dists": dists,
                "hit": hit, "points": points}
