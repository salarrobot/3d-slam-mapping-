"""
Run the 3D mapping robot.

A drone flies an autonomous loop through a room of 3D walls/obstacles,
scanning with a spherical 3D LiDAR. Every scan is fused into a log-odds
occupancy-voxel grid, so the map is *built up* as the robot explores.

Outputs (written to ./outputs):
    map3d.png       reconstructed 3D voxel map vs. ground-truth wireframe
    pointcloud.png  accumulated raw LiDAR point cloud
    topdown.png     recovered 2D floor-plan (top-down occupancy)
    progress.png    mapped-voxel count over time
    mapping.gif     rotating animation of the map being built

Usage:
    python run_mapping.py                 # full run + GIF
    python run_mapping.py --no-anim       # plots only (fast)
    python run_mapping.py --steps 360 --loops 3 --fps 20
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment3d import World, Drone, Lidar3D, WORLD_MIN, WORLD_MAX
from mapping import OccupancyGrid3D

DT = 0.1
CMAP = "viridis"


# =====================================================================
# ------------------------------ helpers ------------------------------
# =====================================================================
def setup_3d(ax, title=None):
    ax.set_xlim(WORLD_MIN[0], WORLD_MAX[0])
    ax.set_ylim(WORLD_MIN[1], WORLD_MAX[1])
    ax.set_zlim(0, WORLD_MAX[2])
    ax.set_box_aspect((WORLD_MAX[0], WORLD_MAX[1], WORLD_MAX[2] * 2.0))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)


def scatter_voxels(ax, centers, size=10):
    if len(centers) == 0:
        return
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c=centers[:, 2], cmap=CMAP, vmin=0, vmax=WORLD_MAX[2],
               s=size, marker="s", depthshade=True, edgecolors="none")


# =====================================================================
# ----------------------------- simulate ------------------------------
# =====================================================================
def simulate(steps, loops, stride):
    world = World()
    drone = Drone(steps, loops=loops)
    lidar = Lidar3D(world)
    grid = OccupancyGrid3D(world)

    traj, cloud, n_occ_hist, frames = [], [], [], []
    for k in range(steps):
        pose = drone.step()
        scan = lidar.scan(pose)
        grid.integrate(scan["origin"], scan["dirs"], scan["dists"],
                       scan["hit"], lidar.max_range)

        traj.append(pose[:3].copy())
        cloud.append(scan["points"][scan["hit"]])
        n_occ_hist.append(grid.n_occupied())

        if k % stride == 0:
            occ = grid.occupied_centers()
            if len(occ) > 5000:                       # cap for speed/size
                sel = np.random.default_rng(k).choice(len(occ), 5000, False)
                occ = occ[sel]
            frames.append({"k": k, "pose": pose[:3].copy(), "occ": occ})

    return world, grid, np.array(traj), np.vstack(cloud), \
        np.array(n_occ_hist), frames


# =====================================================================
# ------------------------------- plots -------------------------------
# =====================================================================
def plot_map3d(world, grid, traj, outdir):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, "Reconstructed 3D occupancy map  (colour = height)")
    world.draw(ax, lw=1.0, alpha=0.25)                 # ground-truth wireframe
    scatter_voxels(ax, grid.occupied_centers(), size=14)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="crimson",
            lw=1.6, label="drone path")
    ax.legend(loc="upper left")
    ax.view_init(elev=30, azim=-60)
    path = os.path.join(outdir, "map3d.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_pointcloud(cloud, traj, outdir):
    pts = cloud
    if len(pts) > 40000:
        sel = np.random.default_rng(0).choice(len(pts), 40000, False)
        pts = pts[sel]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_3d(ax, f"Accumulated LiDAR point cloud  ({len(cloud):,} points)")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap=CMAP,
               vmin=0, vmax=WORLD_MAX[2], s=2, alpha=0.5, edgecolors="none")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="crimson", lw=1.6)
    ax.view_init(elev=30, azim=-60)
    path = os.path.join(outdir, "pointcloud.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_topdown(grid, traj, outdir):
    fig, ax = plt.subplots(figsize=(9, 7))
    img = grid.top_down()
    ax.imshow(img, origin="lower", cmap="magma",
              extent=[WORLD_MIN[0], WORLD_MAX[0], WORLD_MIN[1], WORLD_MAX[1]])
    ax.plot(traj[:, 0], traj[:, 1], color="cyan", lw=1.8, label="drone path")
    ax.set_title("Recovered floor-plan (top-down occupancy)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    path = os.path.join(outdir, "topdown.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_progress(n_occ_hist, outdir):
    t = np.arange(len(n_occ_hist)) * DT
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, n_occ_hist, color="#2a7", lw=2)
    ax.fill_between(t, n_occ_hist, color="#2a7", alpha=0.2)
    ax.set_title("Mapping progress")
    ax.set_xlabel("time [s]"); ax.set_ylabel("occupied voxels in map")
    ax.grid(alpha=0.3)
    path = os.path.join(outdir, "progress.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


# =====================================================================
# ----------------------------- animation -----------------------------
# =====================================================================
def render_animation(world, frames, traj, outdir, fps):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw(i):
        ax.cla()
        fr = frames[i]
        k = fr["k"]
        setup_3d(ax, f"Building the 3D map   —   t = {k * DT:5.1f} s   "
                     f"({len(fr['occ']):,} voxels)")
        world.draw(ax, lw=0.8, alpha=0.18)
        scatter_voxels(ax, fr["occ"], size=10)
        ax.plot(traj[:k + 1, 0], traj[:k + 1, 1], traj[:k + 1, 2],
                color="crimson", lw=1.5)
        p = fr["pose"]
        ax.scatter([p[0]], [p[1]], [p[2]], color="red", s=60, marker="^")
        ax.view_init(elev=28, azim=-60 + 0.6 * i)      # slow orbit
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    gif = os.path.join(outdir, "mapping.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return gif


# =====================================================================
# -------------------------------- main -------------------------------
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--loops", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every Nth step as an animation frame")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Flying {args.steps} steps ({args.loops} loops) and mapping ...")
    world, grid, traj, cloud, n_occ_hist, frames = simulate(
        args.steps, args.loops, args.stride)

    total = int(np.prod(grid.dims))
    dims = tuple(int(d) for d in grid.dims)
    print(f"  voxel grid          : {dims}  ({total:,} voxels)")
    print(f"  LiDAR points fused  : {len(cloud):,}")
    print(f"  occupied voxels     : {grid.n_occupied():,}")

    print("Saving plots ...")
    for p in (plot_map3d(world, grid, traj, args.outdir),
              plot_pointcloud(cloud, traj, args.outdir),
              plot_topdown(grid, traj, args.outdir),
              plot_progress(n_occ_hist, args.outdir)):
        print(f"  -> {p}")

    if not args.no_anim:
        print(f"Rendering animation ({len(frames)} frames) ...")
        print(f"  -> {render_animation(world, frames, traj, args.outdir, args.fps)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
