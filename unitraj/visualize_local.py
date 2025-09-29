import time
import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

# Project imports
from unitraj.models import build_model               # factory that creates model objects based on Hydra config
from unitraj.datasets import build_dataset           # factory that creates dataset/dataloader utils based on config
from unitraj.datasets.common_utils import rotate_points_along_z  # helper for SE(2) transforms in XY plane
from unitraj.utils.utils import set_seed             # reproducible seed setter for numpy/torch/etc.

# ---------------------------------------------------------------------
# Map-feature conversion utilities
# ---------------------------------------------------------------------
# This dictionary maps our custom COSMOS map categories into the “NuScenes-like”
# categories that UniTraj expects. It’s a minimal mapping used by visualization.
COSMOS_TO_NUSCENE_TYPE = {
    "CROSSWALK": "CROSSWALK",
    "DRIVABLE_AREA": "DRIVABLE_AREA",
    "LANE_SURFACE_STREET": "LANE_SURFACE_STREET",
    "LANE_BIKE_LANE": "LANE_BIKE_LANE",
}

def polygon_to_centerline_directional(polygon_coords: np.ndarray, lane_name: str) -> np.ndarray:
    """
    Convert a lane polygon to a simple 2-point “centerline” segment, with a heuristic
    about orientation. This produces a minimal polyline to overlay on the map.

    Args:
        polygon_coords: (N, 2) polygon vertices in meters, ordered.
        lane_name:      string like "XY_nv" / "XY_wv" indicating lane orientation.

    Returns:
        (2, 2) array with the two end points of a centerline segment.
        If input polygon is invalid/empty, we just return the raw polygon (fallback).
    """
    poly = Polygon(polygon_coords)
    if (not poly.is_valid) or poly.is_empty:
        return polygon_coords  # do nothing if invalid

    # Heuristic parsing of lane_name to infer a primary direction for drawing
    lane_name = lane_name.lower()
    is_vertical = any(d in lane_name for d in ["nv", "sv"])
    is_horizontal = any(d in lane_name for d in ["wv", "ev"])

    xs = polygon_coords[:, 0]
    ys = polygon_coords[:, 1]

    # If vertical, center x and span y-range
    if is_vertical:
        x_center = (xs.min() + xs.max()) / 2
        return np.array([[x_center, ys.min()], [x_center, ys.max()]], dtype=np.float32)
    # If horizontal, center y and span x-range
    if is_horizontal:
        y_center = (ys.min() + ys.max()) / 2
        return np.array([[xs.min(), y_center], [xs.max(), y_center]], dtype=np.float32)

    # Fallback: use the minimum rotated rectangle, pick the long axis midpoints
    min_rect = poly.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords[:-1])  # 4 corners, drop the closing corner
    v1, v2 = coords[1] - coords[0], coords[2] - coords[1]
    if np.linalg.norm(v1) < np.linalg.norm(v2):
        m0, m1 = (coords[0] + coords[1]) / 2, (coords[2] + coords[3]) / 2
    else:
        m0, m1 = (coords[1] + coords[2]) / 2, (coords[3] + coords[0]) / 2
    return np.array([m0, m1], dtype=np.float32)

def build_map_features(csv_path: str, div: float = 20.0, img_h_px: int = 836) -> dict:
    """
    Vectorize map features from a CSV labeled in image pixels into “meter” space.

    Key steps:
    - Flip Y axis: image origin is top-left; we want a standard XY world with Y up.
      We do: y' = H - y, where H is the image height in pixels.
    - Scale: pixel units to meters by dividing with `div` (pixels-per-meter).
    - For DRIVABLE_AREA we keep only the polygon. For other lanes, we also create a
      simple 2-point centerline (polyline) for quick rendering.

    Args:
        csv_path: CSV file path with columns `region_shape_attributes` and `region_attributes`
                  (VIA/VGG style annotations).
        div:      pixels-per-meter divisor; 20 px ≈ 1 m by default.
        img_h_px: raw image height in pixels (used for Y-flip).

    Returns:
        dict: lane_id -> {
                 "type": str,
                 "polygon": (N,2) float32 in meters,
                 "polyline": (2,2) float32 (optional)
              }
    """

    df = pd.read_csv(csv_path)
    out = {}
    for rsa, ra in zip(df["region_shape_attributes"], df["region_attributes"]):
        rsa, ra = json.loads(rsa), json.loads(ra)
        lane_name = ra["location"] + ra["type"][0]  # e.g., "AB_nv"

        # Build polygon in pixel coords -> flip Y -> scale to meters
        polygon = np.array(list(zip(rsa["all_points_x"], rsa["all_points_y"])), dtype=np.float32)
        polygon[:, 1] = img_h_px - polygon[:, 1]  # flip Y to make "up" positive
        polygon /= div                           # convert pixels to meters

        mtype = COSMOS_TO_NUSCENE_TYPE.get(ra.get("map_feature", ""), "unknown")
        rec = {"type": mtype, "polygon": polygon}

        # Non-drivable lanes also get a simple centerline segment
        if mtype != "DRIVABLE_AREA":
            rec["polyline"] = polygon_to_centerline_directional(polygon, lane_name)

        out[f"lane_{lane_name}"] = rec
    return out

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def transform_to_global(local_points, center_xyz, center_heading):
    """
    Transform local XY points into global/world coordinates using a 2D rotation
    (about Z) and a translation given by the center object’s world pose.

    Args:
        local_points: (..., 2) array-like in local frame.
        center_xyz:   (>=2,) array-like for world translation (x, y, [z, ...]).
        center_heading: scalar yaw in radians.

    Returns:
        Same shape as input local_points, but in world frame (XY only).
    """
    pts = np.asarray(local_points, dtype=np.float32).reshape(-1, 2)[None, :, :]  # shape (1, N, 2)
    ang = np.array([center_heading], dtype=np.float32)                           # shape (1,)
    world = rotate_points_along_z(pts, ang)[0, :, :2]                            # rotate in XY
    world += center_xyz[:2]                                                      # translate
    return world.reshape(local_points.shape)

# ---------------------------------------------------------------------
# Plot style palette and primitives
# ---------------------------------------------------------------------
PALETTE = {
    "raster": dict(),  # reserved
    "lanes": dict(line="dimgray", fill="lightgray"),
    "traj": dict(past="tab:green", gt="tab:blue", best="crimson", others="gold"),
}

def add_raster_background(ax, image_path, div=20.0):
    """
    Draw a raster map image (meters-scaled) as the background.
    """
    img = plt.imread(image_path)
    h, w = img.shape[:2]
    extent = [0, w / div, 0, h / div]  # map pixels to meters for consistent scale
    ax.imshow(img, extent=extent, origin="lower", interpolation="none", alpha=1.0, zorder=0)

def add_vector_map(ax, map_features: dict):
    """
    Plot vectorized lane polygons + optional centerlines.
    """
    for lane in map_features.values():
        poly = lane["polygon"]
        if poly.shape[0] > 0:
            xs = np.r_[poly[:, 0], poly[0, 0]]
            ys = np.r_[poly[:, 1], poly[0, 1]]
            ax.plot(xs, ys, color="lightgray", linewidth=0.8, zorder=1)
        if "polyline" in lane:
            pl = lane["polyline"]
            ax.plot(pl[:, 0], pl[:, 1], color="dimgray", linewidth=1.0, alpha=0.9, zorder=2)

def add_dataset_map(ax, map_polylines_local, center_world, center_heading):
    """
    Plot dataset-provided map polylines, transformed from local to world coords.
    """
    for lane in map_polylines_local[..., :2]:
        if np.allclose(lane, 0):
            continue
        world = transform_to_global(lane, center_world, center_heading)
        ax.plot(world[:, 0], world[:, 1], color="lightgray", linewidth=0.8, alpha=0.9, zorder=1)

def draw_center_trajs(ax, sample, center_idx, center_world, center_heading):
    """
    Plot the center agent’s past trajectory (history) and ground-truth future.
    Both are transformed into world coordinates for a consistent map overlay.
    """
    past = transform_to_global(sample["obj_trajs"][center_idx, :, :2], center_world, center_heading)
    gt   = transform_to_global(sample["obj_trajs_future_state"][center_idx, :, :2], center_world, center_heading)

    # History in green (dashed), with circular markers
    ax.plot(past[:, 0], past[:, 1], color=PALETTE["traj"]["past"], linestyle="--",
            linewidth=1.4, alpha=0.9, zorder=3)
    ax.scatter(past[:, 0], past[:, 1], color=PALETTE["traj"]["past"], s=10, alpha=0.9, marker="o", zorder=3)

    # GT future in blue (dashed), with circular markers
    ax.plot(gt[:, 0], gt[:, 1], color=PALETTE["traj"]["gt"], linestyle="--",
            linewidth=1.4, alpha=0.9, zorder=3)
    ax.scatter(gt[:, 0], gt[:, 1], color=PALETTE["traj"]["gt"], s=10, alpha=0.9, marker="o", zorder=3)

def draw_predictions(ax, preds_local, gt_future_local, center_world, center_heading):
    """
    Plot the model’s multi-modal predictions (M modes). We rank modes by ADE
    w.r.t. ground-truth future and highlight the best (lowest ADE) mode.

    Args:
        preds_local:     (M, T, 2) predicted future points in the local frame
        gt_future_local: (T, 2) ground-truth future points in local frame
        center_world:    world pose translation
        center_heading:  world pose yaw
    """
    # Compute ADE per mode vs GT (in local frame)
    ade = np.linalg.norm(preds_local - gt_future_local[None, :, :2], axis=-1).mean(-1)  # (M,)
    order = np.argsort(ade)
    best = order[0]

    # Light alpha for “non-best” modes to reduce visual clutter
    alpha_line_others = 0.12
    alpha_mark_others = 0.06

    for m in order:
        world = transform_to_global(preds_local[m], center_world, center_heading)
        if m == best:
            # Best mode: thick red line + triangles
            ax.plot(world[:, 0], world[:, 1], color=PALETTE["traj"]["best"], linewidth=2.2, alpha=1.0, zorder=5)
            ax.scatter(world[:, 0], world[:, 1], color=PALETTE["traj"]["best"], s=18, alpha=1.0, marker="^", zorder=5)
        else:
            # Other modes: thin, dashed, pale gold
            ax.plot(world[:, 0], world[:, 1], color=PALETTE["traj"]["others"],
                    linewidth=0.9, linestyle="--", alpha=alpha_line_others, zorder=4)
            ax.scatter(world[:, 0], world[:, 1], color=PALETTE["traj"]["others"],
                       s=12, alpha=alpha_mark_others, marker="^", zorder=4)

# ---------------------------------------------------------------------
# Main entry: load config, build model/dataset, run inference, visualize
# ---------------------------------------------------------------------
def main():
    """
    End-to-end flow:
    1) Load Hydra config (config.yaml + defaults -> method/autobot.yaml).
    2) Merge top-level cfg with method cfg so model.build sees flat keys (e.g., hidden_size).
    3) Build model in eval mode; optionally load checkpoint.
    4) Build validation dataset and pick a scene to visualize.
    5) Optional: load external map (CSV or raster image) or use dataset map.
    6) For each agent in the selected scene:
         - collate a single-sample batch
         - move tensors to device
         - forward the model (no grad)
         - draw history, GT future, and predictions
    7) Save a PNG with all overlays.
    """
    # 1) Hydra: initialize from an explicit config directory and compose the full config.
    #    You can override values inline by adding strings to `overrides` (see commented examples).
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=os.path.join("/home/cz2678/Documents/trajectory/UniTraj/unitraj/configs")
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                # Example overrides (uncomment and set your paths/ids if needed):
                # "ckpt_path=/content/drive/MyDrive/COSMOS_Unitraj/Unitraj_main/unitraj/unitraj_ckpt/autobot_real/epoch=64-val/brier_fde=0.65.ckpt",
                # "vis_scene_id=YOUR_SCENE_ID",
            ],
        )

    # 2) Match the project’s training/eval script behavior:
    #    - Disable struct to allow adding/overwriting keys.
    #    - Merge `cfg.method` into the root so model builders can index keys directly (e.g., cfg['hidden_size']).
    #    - Flag evaluation mode.
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)   # required so keys like hidden_size exist at the top level
    cfg["eval"] = True

    # 3) Device & model build
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()

    # Optional checkpoint loading: expects a Lightning-style dict with ["state_dict"]
    if cfg.get("ckpt_path", None):
        ckpt = torch.load(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[ok] Loaded model: {cfg.ckpt_path}")

    # 4) Build validation dataset from config and pick a target scene.
    #    `val_set.data_loaded_keys` looks like: ["sceneX-agentY", ...]
    val_set = build_dataset(cfg, val=True)
    target_scene_id = cfg.get("vis_scene_id") or val_set.data_loaded_keys[0].split("-")[0]
    idxs = [i for i, k in enumerate(val_set.data_loaded_keys) if target_scene_id in k]
    print(f"scene_id={target_scene_id}  agents={len(idxs)}")

    # Extract a base world pose from the first sample so we can place everything consistently.
    first = val_set[idxs[0]]
    center_world = first["center_objects_world"][:3]  # (x, y, z, ...)
    center_heading = first["center_objects_world"][6] # yaw in radians

    # 5) Optional external map choice: "vector" (CSV), "raster" (image), or "dataset" (built-in polylines).
    vis_mode = "vector"
    map_features = None
    if vis_mode == "vector":
        csv_path = "/home/cz2678/Documents/trajectory/UniTraj/unitraj/map_features.csv"
        map_features = build_map_features(csv_path)
        print(f"[ok] loaded vector map: {csv_path}  lanes={len(map_features)}")
    elif vis_mode == "raster":
        img_path = "/home/cz2678/Documents/trajectory/UniTraj/unitraj/cosmos-view.png"
        assert os.path.exists(img_path), f"missing image: {img_path}"
        print(f"[ok] using raster background: {img_path}")

    # 6) Figure/axes setup — turn off spines/ticks for a clean map view
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw the chosen base map
    if vis_mode == "raster":
        add_raster_background(ax, img_path)
    elif vis_mode == "vector":
        add_vector_map(ax, map_features)
    else:  # "dataset"
        add_dataset_map(ax, first["map_polylines"], center_world, center_heading)

    # 7) Iterate agents in the scene, run forward once per agent, and plot results.
    times_ms = []  # per-agent forward timing for a tiny perf summary
    for idx in tqdm(idxs, desc="agents"):
        sample = val_set[idx]

        # Collate a single-sample “batch” using dataset’s collate_fn for proper tensor shapes.
        batch = val_set.collate_fn([sample])

        # UniTraj model forward expects {"input_dict": ...}; we keep only that branch.
        batch = {"input_dict": batch["input_dict"]}

        # Move every tensor in the input_dict to the active device.
        for k in batch["input_dict"]:
            if isinstance(batch["input_dict"][k], torch.Tensor):
                batch["input_dict"][k] = batch["input_dict"][k].to(device)

        # Optional CUDA sync before/after measuring elapsed time for consistent profiling.
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output, _ = model(batch)  # forward pass returns (output_dict, aux_dict)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        print(f"[time] agent {len(times_ms)}/{len(idxs)} forward: {dt_ms:.2f} ms")

        # Some models compute probabilistic heads on a second call (kept from original script)
        with torch.no_grad():
            output, _ = model(batch)

        # Center object index and its pose per-sample (used to transform local -> world)
        center_idx = int(sample["track_index_to_predict"])
        c_world = sample["center_objects_world"][:3]
        c_head = sample["center_objects_world"][6]

        # Draw past and GT future for the center agent
        draw_center_trajs(ax, sample, center_idx, c_world, c_head)

        # Extract predictions (B=1): output["predicted_trajectory"] has shape (B, M, T, 4)
        # We take first batch item, all modes M, all T, XY only.
        preds_local = output["predicted_trajectory"][0, :, :, :2].detach().cpu().numpy()  # (M, T, 2)

        # Ground-truth future in local frame for the same agent (T, 2)
        gt_future_local = sample["obj_trajs_future_state"][center_idx, :, :2]

        # Overlay predictions ranked by ADE, highlight best mode
        draw_predictions(ax, preds_local, gt_future_local, c_world, c_head)

    # 8) Print a quick timing summary and save the figure
    if len(times_ms):
        trimmed = times_ms[3:] if len(times_ms) > 3 else times_ms  # skip warm-up
        mean_ms = float(np.mean(trimmed))
        med_ms = float(np.median(trimmed))
        p90_ms = float(np.percentile(trimmed, 90))
        fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        print("[time] summary (single-agent forward) "
              f"N={len(trimmed)} | mean={mean_ms:.2f} ms | median={med_ms:.2f} ms "
              f"| p90={p90_ms:.2f} ms | ~{fps:.1f} FPS")

    out_name = f"1autobot_cosmos_{vis_mode}.png"
    plt.tight_layout(pad=0.1)
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"[ok] saved to {out_name}")

# Standard Python entry-guard. Required when this code is executed as a script/module.
if __name__ == "__main__":
    main()
