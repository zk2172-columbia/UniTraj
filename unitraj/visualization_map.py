# viz_scene.py
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
from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.datasets.common_utils import rotate_points_along_z
from unitraj.utils.utils import set_seed

# =========================
# Map features (CSV -> vector lanes)
# =========================
COSMOS_TO_NUSCENE_TYPE = {
    "CROSSWALK": "CROSSWALK",
    "DRIVABLE_AREA": "DRIVABLE_AREA",
    "LANE_SURFACE_STREET": "LANE_SURFACE_STREET",
    "LANE_BIKE_LANE": "LANE_BIKE_LANE",
}

def polygon_to_centerline_directional(polygon_coords: np.ndarray, lane_name: str) -> np.ndarray:
    poly = Polygon(polygon_coords)
    if (not poly.is_valid) or poly.is_empty:
        return polygon_coords

    lane_name = lane_name.lower()
    is_vertical = any(d in lane_name for d in ["nv", "sv"])
    is_horizontal = any(d in lane_name for d in ["wv", "ev"])

    xs = polygon_coords[:, 0]
    ys = polygon_coords[:, 1]

    if is_vertical:
        x_center = (xs.min() + xs.max()) / 2
        return np.array([[x_center, ys.min()], [x_center, ys.max()]], dtype=np.float32)
    if is_horizontal:
        y_center = (ys.min() + ys.max()) / 2
        return np.array([[xs.min(), y_center], [xs.max(), y_center]], dtype=np.float32)

    # fallback: minimum rotated rectangle’s long axis midpoints
    min_rect = poly.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords[:-1])
    v1, v2 = coords[1] - coords[0], coords[2] - coords[1]
    if np.linalg.norm(v1) < np.linalg.norm(v2):
        m0, m1 = (coords[0] + coords[1]) / 2, (coords[2] + coords[3]) / 2
    else:
        m0, m1 = (coords[1] + coords[2]) / 2, (coords[3] + coords[0]) / 2
    return np.array([m0, m1], dtype=np.float32)

def build_map_features(csv_path: str, div: float = 20.0, img_h_px: int = 836) -> dict:
    """
    Build vectorized map features from CSV.
    - Flip Y: y' = H - y  (to make up go up)
    - Scale to meters by /div
    - Skip DRIVABLE_AREA polyline (keep polygon only)
    """
    df = pd.read_csv(csv_path)
    out = {}
    for rsa, ra in zip(df["region_shape_attributes"], df["region_attributes"]):
        rsa, ra = json.loads(rsa), json.loads(ra)
        lane_name = ra["location"] + ra["type"][0]

        polygon = np.array(list(zip(rsa["all_points_x"], rsa["all_points_y"])), dtype=np.float32)
        polygon[:, 1] = img_h_px - polygon[:, 1]
        polygon /= div

        mtype = COSMOS_TO_NUSCENE_TYPE.get(ra.get("map_feature", ""), "unknown")
        rec = {"type": mtype, "polygon": polygon}

        if mtype != "DRIVABLE_AREA":
            rec["polyline"] = polygon_to_centerline_directional(polygon, lane_name)

        out[f"lane_{lane_name}"] = rec
    return out

# =========================
# Geometry helpers
# =========================
def transform_to_global(local_points, center_xyz, center_heading):
    pts = np.asarray(local_points, dtype=np.float32).reshape(-1, 2)[None, :, :]  # (1,N,2)
    ang = np.array([center_heading], dtype=np.float32)
    world = rotate_points_along_z(pts, ang)[0, :, :2]
    world += center_xyz[:2]
    return world.reshape(local_points.shape)

# =========================
# Plot primitives
# =========================
PALETTE = {
    "raster": dict(),  # not used
    "lanes": dict(line="dimgray", fill="lightgray"),
    "traj": dict(past="tab:green", gt="tab:blue", best="crimson", others="gold"),
}

def add_raster_background(ax, image_path, div=20.0):
    img = plt.imread(image_path)
    h, w = img.shape[:2]
    extent = [0, w / div, 0, h / div]  # meters
    ax.imshow(img, extent=extent, origin="lower", interpolation="none", alpha=1.0, zorder=0)

def add_vector_map(ax, map_features: dict):
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
    for lane in map_polylines_local[..., :2]:
        if np.allclose(lane, 0):
            continue
        world = transform_to_global(lane, center_world, center_heading)
        ax.plot(world[:, 0], world[:, 1], color="lightgray", linewidth=0.8, alpha=0.9, zorder=1)

def draw_center_trajs(ax, sample, center_idx, center_world, center_heading):
    past = transform_to_global(sample["obj_trajs"][center_idx, :, :2], center_world, center_heading)
    gt   = transform_to_global(sample["obj_trajs_future_state"][center_idx, :, :2], center_world, center_heading)

    ax.plot(past[:, 0], past[:, 1], color=PALETTE["traj"]["past"], linestyle="--",
            linewidth=1.4, alpha=0.9, zorder=3)
    ax.scatter(past[:, 0], past[:, 1], color=PALETTE["traj"]["past"], s=10, alpha=0.9, marker="o", zorder=3)

    ax.plot(gt[:, 0], gt[:, 1], color=PALETTE["traj"]["gt"], linestyle="--",
            linewidth=1.4, alpha=0.9, zorder=3)
    ax.scatter(gt[:, 0], gt[:, 1], color=PALETTE["traj"]["gt"], s=10, alpha=0.9, marker="o", zorder=3)

def draw_predictions(ax, preds_local, gt_future_local, center_world, center_heading):
    # rank modes by ADE vs GT
    ade = np.linalg.norm(preds_local - gt_future_local[None, :, :2], axis=-1).mean(-1)  # (M,)
    order = np.argsort(ade)
    best = order[0]

    # fade others more aggressively
    alpha_line_others = 0.12
    alpha_mark_others = 0.06

    for m in order:
        world = transform_to_global(preds_local[m], center_world, center_heading)
        if m == best:
            ax.plot(world[:, 0], world[:, 1], color=PALETTE["traj"]["best"], linewidth=2.2, alpha=1.0, zorder=5)
            ax.scatter(world[:, 0], world[:, 1], color=PALETTE["traj"]["best"], s=18, alpha=1.0, marker="^", zorder=5)
        else:
            ax.plot(world[:, 0], world[:, 1], color=PALETTE["traj"]["others"],
                    linewidth=0.9, linestyle="--", alpha=alpha_line_others, zorder=4)
            ax.scatter(world[:, 0], world[:, 1], color=PALETTE["traj"]["others"],
                       s=12, alpha=alpha_mark_others, marker="^", zorder=4)

# =========================
# Main
# =========================
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    """
    Add to your hydra config (or override from CLI):
      vis_mode: raster | vector | dataset
      map_image_path:  "./cosmos_camera_topdown.jpg"
      map_features_csv:"./map_features.csv"
      ckpt_path:        "..."  (optional)
      vis_scene_id:     null   (auto picks first)
    Usage:
      python viz_scene.py vis_mode=raster map_image_path=./first_map.jpg
      python viz_scene.py vis_mode=vector map_features_csv=./map_features.csv
      python viz_scene.py vis_mode=dataset
    """
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg["eval"] = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).eval()
    if cfg.get("ckpt_path", None):
        ckpt = torch.load(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[ok] Loaded model: {cfg.ckpt_path}")

    val_set = build_dataset(cfg, val=True)

    # pick a scene
    target_scene_id = cfg.get("vis_scene_id") or val_set.data_loaded_keys[0].split("-")[0]
    idxs = [i for i, k in enumerate(val_set.data_loaded_keys) if target_scene_id in k]
    print(f"scene_id={target_scene_id}  agents={len(idxs)}")

    # base world pose from first sample
    first = val_set[idxs[0]]
    center_world = first["center_objects_world"][:3]
    center_heading = first["center_objects_world"][6]

    # optional external maps
    vis_mode = "vector"
    map_features = None
    if vis_mode == "vector":
        csv_path = "/home/wz2708/unitraj/unitraj/map_features.csv"
        map_features = build_map_features(csv_path)
        print(f"[ok] loaded vector map: {csv_path}  lanes={len(map_features)}")
    elif vis_mode == "raster":
        img_path = "/home/wz2708/unitraj/unitraj/cosmos-view.png"
        assert os.path.exists(img_path), f"missing image: {img_path}"
        print(f"[ok] using raster background: {img_path}")

    # canvas
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    # draw base map
    if vis_mode == "raster":
        add_raster_background(ax, img_path)
    elif vis_mode == "vector":
        add_vector_map(ax, map_features)
    else:  # dataset
        add_dataset_map(ax, first["map_polylines"], center_world, center_heading)

    # iterate agents in scene
    times_ms = []
    for idx in tqdm(idxs, desc="agents"):
        sample = val_set[idx]
        batch = val_set.collate_fn([sample])
        batch = {"input_dict": batch["input_dict"]}
        for k in batch["input_dict"]:
            if isinstance(batch["input_dict"][k], torch.Tensor):
                batch["input_dict"][k] = batch["input_dict"][k].to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output, _ = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)
        print(f"[time] agent {len(times_ms)}/{len(idxs)} forward: {dt_ms:.2f} ms")

        with torch.no_grad():
            output, _ = model(batch)

        center_idx = int(sample["track_index_to_predict"])
        c_world = sample["center_objects_world"][:3]
        c_head = sample["center_objects_world"][6]

        draw_center_trajs(ax, sample, center_idx, c_world, c_head)

        preds_local = output["predicted_trajectory"][0, :, :, :2].detach().cpu().numpy()  # (M,T,2)
        gt_future_local = sample["obj_trajs_future_state"][center_idx, :, :2]
        draw_predictions(ax, preds_local, gt_future_local, c_world, c_head)

    if len(times_ms):
        trimmed = times_ms[3:] if len(times_ms) > 3 else times_ms
        mean_ms = float(np.mean(trimmed))
        med_ms = float(np.median(trimmed))
        p90_ms = float(np.percentile(trimmed, 90))
        fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        print("[time] summary (single-agent forward) "
              f"N={len(trimmed)} | mean={mean_ms:.2f} ms | median={med_ms:.2f} ms "
              f"| p90={p90_ms:.2f} ms | ~{fps:.1f} FPS")
    out_name = f"wayformer_real_cosmos3_{vis_mode}.png"
    plt.tight_layout(pad=0.1)
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"[ok] saved to {out_name}")

if __name__ == "__main__":
    main()
