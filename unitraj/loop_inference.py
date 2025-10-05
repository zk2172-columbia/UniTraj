import os, json, time
import numpy as np
import torch
import shutil
from typing import Dict, Tuple, Any, List, Optional

import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.datasets.common_utils import rotate_points_along_z
from unitraj.utils.utils import set_seed
from unitraj.visualization_map import build_map_features, add_vector_map
from unitraj.build_pkl import build_custom_pkl

DIV = 20.0
IMG_H = 836

# Coordinate System Tools
def local_to_world_m(local_xy: np.ndarray, center_world_xyz: np.ndarray, center_heading: float) -> np.ndarray:
    """(T,2) local meters -> (T,2) world meters"""
    pts = local_xy.reshape(1, -1, 2).astype(np.float32)    # (1,T,2)
    ang = np.array([center_heading], dtype=np.float32)     # (1,)
    world = rotate_points_along_z(pts, ang)[0, :, :2]      # (T,2)
    world += center_world_xyz[:2]
    return world

def world_m_to_pixel(world_xy: np.ndarray, div: float = DIV, img_h_px: int = IMG_H) -> np.ndarray:
    px = world_xy * div
    px[:, 1] = img_h_px - px[:, 1]
    return px

# Helper functions for extracting fields
def get_agent_id(sample: dict, idx_fallback: int) -> str:
    if "track_id_to_predict" in sample:
        return str(sample["track_id_to_predict"])
    return f"agent_{idx_fallback:03d}"

def get_agent_type(sample: Dict[str, Any], center_idx: int) -> str:
    if "object_type_to_predict" in sample:
        return str(sample["object_type_to_predict"])
    if "obj_trajs_type" in sample:
        try:
            return str(sample["obj_trajs_type"][center_idx])
        except Exception:
            pass
    return "VEHICLE"

def get_center_pose(sample: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    c_world = sample["center_objects_world"][:3]     # (x,y,z) meters
    c_head = float(sample["center_objects_world"][6])
    return c_world, c_head

# 1) Single round: reasoning for all agents in "one pkl scene"
def infer_scene_once(cfg,
                     model: torch.nn.Module,
                     round_dir: str,
                     device: Optional[str] = None) -> Dict[str, Any]:
    """
    Input:
    - cfg: Your hydra/project configuration (DictConfig, keep the original structure)
    - model: The built and .eval() model
    - round_dir: The directory for this round (only one pkl file is placed in it: scene.pkl)
    Return pack:
    - scene_id: str
    - agents: List[str]
    - center_pose: {aid: (center_world_xyz(3,), center_heading)}
    - types: {aid: "VEHICLE"/"PEDESTRIAN"}
    - pred12_local: {aid: (12,2)} # Prediction for this round (local coordinates)
    - hist_world: {aid: (8,2)} # Only used when round==0 (world coordinates)
    - gt_world: {aid: (12,2)} or without this key # If the sample has GT, return the world coordinates
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Only read this round of directories
    cfg["val_data_path"] = [round_dir]
    val_set = build_dataset(cfg, val=True)

    target_scene_id = val_set.data_loaded_keys[0].split("-")[0]
    idxs = [i for i, k in enumerate(val_set.data_loaded_keys) if target_scene_id in k]

    agents_order: List[str] = []
    center_pose: Dict[str, Tuple[np.ndarray, float]] = {}
    types: Dict[str, str] = {}
    pred12_local: Dict[str, np.ndarray] = {}
    hist_world: Dict[str, np.ndarray] = {}
    gt_world: Dict[str, np.ndarray] = {}

    for i, idx in enumerate(tqdm(idxs, desc="[infer] agents")):
        sample = val_set[idx]

        batch = val_set.collate_fn([sample])
        batch = {"input_dict": batch["input_dict"]}
        for k in batch["input_dict"]:
            if isinstance(batch["input_dict"][k], torch.Tensor):
                batch["input_dict"][k] = batch["input_dict"][k].to(device)

        with torch.no_grad():
            output, _ = model(batch)

        center_idx = int(sample["track_index_to_predict"])
        aid = get_agent_id(sample, idx_fallback=i)
        atype = get_agent_type(sample, center_idx)
        c_world, c_head = get_center_pose(sample)

        # predict (local, meter)
        preds_local = output["predicted_trajectory"][0, :, :, :2].detach().cpu().numpy()  # (M,12,2)
        gt_future_local = sample["obj_trajs_future_state"][center_idx, :, :2]
        gt_world[aid] = local_to_world_m(gt_future_local, c_world, c_head)

        if gt_future_local is not None and len(gt_future_local) > 0:
            # If have GT, select with ADE
            ade = np.linalg.norm(preds_local - gt_future_local[None, :, :2], axis=-1).mean(-1)
            best = np.argmin(ade)
        else:
            # No GT → Choose by probability (if any)
            if "predicted_probability" in output:
                probs = output["predicted_probability"][0].detach().cpu().numpy()
                best = int(np.argmax(probs))
            else:
                best = 0  # fallback: first mode
        pred12 = preds_local[best]
        pred12_local[aid] = pred12

        # First pass: History & GT (world coordinates, meters)
        # History: obj_trajs is (N, T_hist, 4) or similar, only xy is used here
        if "obj_trajs" in sample:
            past_local = sample["obj_trajs"][center_idx, :, :2]  # (Th,2) usually Th=8
            past_world = local_to_world_m(past_local, c_world, c_head)
            # Insurance: only take the first 8 frames
            hist_world[aid] = past_world[:8, :]
        # Record ID
        agents_order.append(aid)
        center_pose[aid] = (c_world, c_head)
        types[aid] = atype

    pack = {
        "scene_id": target_scene_id,
        "agents": agents_order,
        "center_pose": center_pose,
        "types": types,
        "pred12_local": pred12_local,
        "hist_world": hist_world,   # 首轮会用到
    }
    if len(gt_world):
        pack["gt_world"] = gt_world
    return pack

# 2) Multiple rounds: All agents in the scene advance synchronously + overlay vector maps to draw a "single picture"
def rollout_scene(initial_pkl,
                  cfg,
                  out_dir: str = "unitraj/rollouts_scene",
                  max_rounds: int = 10,
                  small_motion_thresh_m: float = 0.05,
                  small_motion_last_k: int = 3,
                  map_features: Optional[Dict[str, Any]] = None,
                  draw_vector_map_fn: Optional[Any] = None) -> None:
    """
    - initial_pkl: Initial PKL path
    - cfg: where to save pkl data
    - out_dir: Scrolling output root directory
    - map_features: Pass in the constructed vector map (in meters); if None, no map is drawn.
    - draw_vector_map_fn: Function handle: fn(ax, map_features) → Draw on ax; if None, no map is drawn.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build model
    model = build_model(cfg).to(device).eval()
    if cfg.get("ckpt_path"):
        ckpt = torch.load(cfg["ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[ok] Loaded model: {cfg['ckpt_path']}")

    # round_000 directory + put a scene.pkl
    fn = os.path.basename(initial_pkl)
    round0_dir = os.path.join(out_dir, "round_000")
    os.makedirs(round0_dir, exist_ok=True)
    current_pkl = os.path.join(round0_dir, fn)
    if os.path.abspath(initial_pkl) != os.path.abspath(current_pkl):
        shutil.copy2(initial_pkl, current_pkl)

    # Record across rounds: history/GT/multi-round predictions for each agent (all world coordinates, meters)
    traj_record: Dict[str, Dict[str, Any]] = {}
    agents_in_scene: List[str] = []
    rounds_done = 0

    for r in range(max_rounds):
        round_dir = os.path.join(out_dir, f"round_{r:03d}")
        os.makedirs(round_dir, exist_ok=True)
        current_pkl = os.path.join(round_dir, fn)
        if not os.path.exists(current_pkl):
            # Not the first round: build_custom_pkl has already existed because we copied it in the first round.
            # If this doesn't exist, it means the previous round didn't produce any output; just stop.
            print(f"[stop] missing {current_pkl}")
            break

        # Single-round reasoning
        pack = infer_scene_once(cfg, model, round_dir, device=device)
        agents = pack["agents"]
        types = pack["types"]
        center_pose = pack["center_pose"]
        pred12_local = pack["pred12_local"]
        if r == 0:
            agents_in_scene = agents[:]  # record order
            # initial traj_record：history/GT（world coordinates）
            for aid in agents_in_scene:
                traj_record[aid] = {
                    "type": types[aid],
                    "hist_world": pack["hist_world"].get(aid, None),
                    "gt_world":   pack.get("gt_world", {}).get(aid, None),
                    "preds_world": []
                }

        # Record the current round of predictions (world coordinates)
        # and prepare the next round of history (pixel coordinates)
        all_last_means: List[float] = []
        next_histories_px: Dict[str, np.ndarray] = {}

        # When round==0, the order and id of the first round are fixed
        if r == 0:
            agents_in_scene = pack["agents"][:]  # Record the stable id of the first round

        # Align by position each subsequent round
        assert len(pack["agents"]) == len(agents_in_scene), "The number of samples in this round is inconsistent with the first round"
        for i, aid0 in enumerate(agents_in_scene):
            # get the id of the corresponding position in this round
            # this id may change across rounds, so we only use it to get the value)
            aid_now = pack["agents"][i]
            c_world, c_head = center_pose[aid_now]
            pred12 = pred12_local[aid_now]

            pred12_world = local_to_world_m(pred12, c_world, c_head)
            # Write back to the first round id
            traj_record[aid0]["preds_world"].append(pred12_world)

            next_hist_world = pred12_world[-8:, :]
            next_hist_px = world_m_to_pixel(next_hist_world, div=DIV, img_h_px=IMG_H).astype(np.float32)
            # use the first round ID all the time to keep consistent
            next_histories_px[aid0] = next_hist_px

        rounds_done = r + 1

        # Termination Condition
        # The last K steps of displacement for all agents are small
        # TODO: Robuster conditions, like control each agent's termination
        if len(all_last_means) and all(m < small_motion_thresh_m for m in all_last_means):
            print(f"[stop] round={r} small-motion for all agents (≤{small_motion_thresh_m} m over last {small_motion_last_k} steps)")
            break

        # Produce the next round of pkl and put it under new sub-folder round_{r+1}/scene.pkl
        next_dir = os.path.join(out_dir, f"round_{r+1:03d}")
        os.makedirs(next_dir, exist_ok=True)
        next_pkl = os.path.join(next_dir, fn)

        # Note: build_custom_pkl will do: pixel → (flip y + /div), fill the future trajectory with 0 and valid
        build_custom_pkl(
            agents_histories=next_histories_px,
            agents_types={aid: traj_record[aid]["type"] for aid in agents_in_scene},
            output_path=next_pkl,
            scenario_id=100000 + (r + 1)
        )

    # Draw one picture contains all trajectories
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    # draw vector map
    if map_features is not None and draw_vector_map_fn is not None:
        draw_vector_map_fn(ax, map_features)

    # History = green, GT = blue; Prediction = gradient by round number (red → yellow)
    # Predicted maximum round number = completed rounds_done
    if rounds_done <= 0:
        rounds_done = 1
    cmap_colors = plt.cm.autumn(np.linspace(0, 1, rounds_done))  # round0, round1, ...

    for aid in agents_in_scene:
        rec = traj_record[aid]

        # history
        if rec["hist_world"] is not None:
            hw = rec["hist_world"]
            ax.plot(hw[:, 0], hw[:, 1], color="tab:green", linestyle="--", linewidth=1.5, alpha=0.9, zorder=3)
            ax.scatter(hw[:, 0], hw[:, 1], color="tab:green", s=12, marker="o", alpha=0.9, zorder=3)

        # ground truth
        if rec["gt_world"] is not None:
            gw = rec["gt_world"]
            ax.plot(gw[:, 0], gw[:, 1], color="tab:blue", linestyle="--", linewidth=1.5, alpha=0.9, zorder=3)
            ax.scatter(gw[:, 0], gw[:, 1], color="tab:blue", s=12, marker="x", alpha=0.9, zorder=3)

        # Multi-round prediction
        for r, pw in enumerate(rec["preds_world"]):
            clr = cmap_colors[r]
            ax.plot(pw[:, 0], pw[:, 1], color=clr, linewidth=1.2, alpha=0.7, zorder=4)
            ax.scatter(pw[:, 0], pw[:, 1], color=clr, s=10, alpha=0.7, marker="^", zorder=4)

    out_png = os.path.join(out_dir, "rollout_all_agents.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved overlay figure: {out_png}")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # import config
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg["eval"] = True

    initial_pkl = "/home/wz2708/unitraj/unitraj/rollouts_demo/sd_cosmos_51069330948.pkl"
    map_features = build_map_features(csv_path="unitraj/map_features.csv", div=20.0, img_h_px=836)
    rollout_scene(
        initial_pkl=initial_pkl,    # initial scene pkl we use for loop inference, whose parent folder must contain only this file
        cfg=cfg,                    # config in unitraj/configs
        out_dir="unitraj/rollouts_scene_wayformer", # output folder, must clean without any files
        max_rounds=3,               # number of iterations we want to loop
        small_motion_thresh_m=0.05, # terminal conditions
        small_motion_last_k=3,      # terminal conditions
        map_features=map_features,
        draw_vector_map_fn=add_vector_map
    )
if __name__ == "__main__":
    main()