import os, json, time
import numpy as np
import torch
import shutil
from typing import Dict, Tuple, Any, List, Optional
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.datasets.common_utils import rotate_points_along_z
from unitraj.utils.utils import set_seed
from unitraj.visualization_map import build_map_features, add_vector_map
from unitraj.gmm_scene import build_gmm_scene
from unitraj.build_pkl import build_custom_pkl_all

from scipy.optimize import linear_sum_assignment

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
            fde = np.linalg.norm(preds_local[:, -1, :2] - gt_future_local[-1, :2], axis=-1)
            # best = int(np.argmin(fde))
            print(f"[debug] agent={aid}  ADE_min={ade.min():.3f}  FDE_min={fde.min():.3f}  "
                  f"ADE@argminFDE={ade[np.argmin(fde)]:.3f}  FDE@argminADE={fde[np.argmin(ade)]:.3f}")

            best = np.argmin(fde)

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
        "hist_world": hist_world,
    }
    if len(gt_world):
        pack["gt_world"] = gt_world
    return pack

# 2) Multiple rounds: All agents in the scene advance synchronously + overlay vector maps to draw a "single picture"
def rollout_scene(scene_json_path: str,
                  cfg,
                  out_dir: str = "unitraj/rollouts_scene",
                  max_rounds: int = 10,
                  small_motion_thresh_m: float = 0.05,
                  small_motion_last_k: int = 3,
                  map_features: Optional[Dict[str, Any]] = None,
                  draw_vector_map_fn: Optional[Any] = None) -> None:
    """
    Drives the loop from the gmm_scene JSON (meters)
    - All agents enter the scene simultaneously by default with t0=0
    - Relaxed strategy: Enter the last round even if k < 12; when writing pkl, padded the future segments to 12 with the last frame; writeback only replaces the actual k frames.
    - Alignment: Nearest neighbor matching based on the "last historical frame world coordinate" to avoid ID/order inconsistencies
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read GMM scene JSON (meters)
    with open(scene_json_path, "r") as f:
        scene = json.load(f)
    meta = scene["meta"]
    dt = float(meta.get("dt", 0.4))
    obs_len, pred_len = 8, 12

    # Create index: id -> agent dict; and convert prior into np.ndarray for easy in-place writing
    agents_by_id: Dict[str, Dict[str, Any]] = {}
    for ag in scene["agents"]:
        ag["prior_xy"] = np.asarray(ag["prior_xy"], dtype=np.float32)  # (T,2) meters
        agents_by_id[ag["id"]] = ag
    # backup original prior for easier visualization
    orig_prior_xy = {aid: ag["prior_xy"].copy() for aid, ag in agents_by_id.items()}

    # model building
    model = build_model(cfg).to(device).eval()
    if cfg.get("ckpt_path"):
        ckpt = torch.load(cfg["ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"[ok] Loaded model: {cfg['ckpt_path']}")

    # trajectory recording for visualization
    traj_record: Dict[str, Dict[str, Any]] = {}
    rounds_done = 0

    # Tool: Constructs the pkl of round_r (8+12, padded to 12 under the relaxed policy)
    # according to the current round r, and saves the order of active_ids
    def build_round_pkl_from_scene(r: int, round_dir: str) -> Tuple[str, List[str]]:
        cursor = obs_len + pred_len * r  # 8, 20, 32, ...
        agents_hist_px: Dict[str, np.ndarray] = {}
        agents_fut_px: Dict[str, np.ndarray] = {}
        agents_types: Dict[str, str] = {}
        active_ids: List[str] = []

        for aid, ag in agents_by_id.items():
            prior = ag["prior_xy"]               # (T,2) meters
            T = len(prior)
            if T < obs_len:                      # Not even 8 frames of history → Never appear
                continue
            if cursor > T:                       # Window has exceeded its bounds → Inactive
                continue
            if cursor - obs_len < 0:             # history window is not enough
                continue

            # history 8 frames
            hist_m = prior[cursor - obs_len: cursor]  # (8,2)

            # The next k frames (k = 0..12);
            # if k==0, do not enter;
            # if 0<k<12, repeat the last frame to make it 12
            k = min(pred_len, max(0, T - cursor))
            if k <= 0:
                continue
            fut_m = prior[cursor: cursor + k]         # (k,2)
            if k < pred_len:
                pad = np.repeat(fut_m[-1][None, :], pred_len - k, axis=0)
                fut_m = np.concatenate([fut_m, pad], axis=0)  # (12,2)

            # meters -> pixels (with y-flip)
            hist_px = world_m_to_pixel(hist_m.copy(), div=DIV, img_h_px=IMG_H).astype(np.float32)
            fut_px  = world_m_to_pixel(fut_m.copy(),  div=DIV, img_h_px=IMG_H).astype(np.float32)

            agents_hist_px[aid] = hist_px
            agents_fut_px[aid]  = fut_px
            agents_types[aid]   = "PEDESTRIAN" if ag.get("typ") in (0, "ped") else "VEHICLE"
            active_ids.append(aid)

            # Initialize drawing record
            if aid not in traj_record:
                traj_record[aid] = {
                    "type": agents_types[aid],
                    "hist_world": hist_m.copy(),   # round0
                    "gt_world":   None,
                    "preds_world": []
                }

        if len(active_ids) == 0:
            return "", active_ids  # no activate agents this round

        os.makedirs(round_dir, exist_ok=True)
        pkl_path = os.path.join(round_dir, "sd_cosmos_100000.pkl")
        build_custom_pkl_all(
            agents_histories=agents_hist_px,
            agents_futures=agents_fut_px,
            agents_types=agents_types,
            output_path=pkl_path,
            scenario_id=100000 + r
        )
        # Record the order of active_ids written in this round (sidecar)
        with open(os.path.join(round_dir, "_active_ids.json"), "w") as f:
            json.dump(active_ids, f)
        return pkl_path, active_ids

    # round_000: Generate PKL and infer using r=0 window ([0..7]+[8..19], with tail padding)
    round_dir = os.path.join(out_dir, "round_000")
    pkl_path, active_ids = build_round_pkl_from_scene(r=0, round_dir=round_dir)
    if not pkl_path:
        print("[stop] round_000 has no active agents (not enough 8+1 frames).")
        return

    # main loop
    for r in range(max_rounds):
        round_dir = os.path.join(out_dir, f"round_{r:03d}")
        pkl_path  = os.path.join(round_dir, "sd_cosmos_100000.pkl")
        if not os.path.exists(pkl_path):
            print(f("[stop] missing {pkl_path}"))
            break

        # single round inference
        pack = infer_scene_once(cfg, model, round_dir, device=device)
        center_pose = pack["center_pose"]
        pred12_local = pack["pred12_local"]
        returned_ids = pack["agents"]  # Dataset side id order

        # Position-based nearest neighbor matching: ds_id → aid, aligning the center pose with the written-back object
        cursor = obs_len + pred_len * r

        # 1) Dataset side: the world coordinates of the last historical frame (as the observation anchor point)
        ds_ids = returned_ids
        ds_obs_last = []
        for ds_id in ds_ids:
            hw = pack["hist_world"].get(ds_id, None)  # (8,2) world meters
            ds_obs_last.append(None if hw is None or len(hw) == 0 else hw[-1])

        # 2) On our scene side: the last frame of the current window history (prior[cursor-1]) is used as the prior anchor point
        with open(os.path.join(round_dir, "_active_ids.json"), "r") as f:
            active_ids_this_round = json.load(f)
        anchor_aids = []
        anchor_pts = []
        for aid in active_ids_this_round:
            ag = agents_by_id.get(aid)
            if ag is None:
                continue
            prior = ag["prior_xy"]
            if cursor-1 < 0 or cursor-1 >= len(prior):
                continue
            anchor_aids.append(aid)
            anchor_pts.append(prior[cursor-1])

        anchor_pts = np.asarray(anchor_pts, dtype=np.float32)

        # Use Hungarian to match ds samples with our aid (based on the last frame world coordinates)
        valid_ds = [(i, ds_id, pt) for i, (ds_id, pt) in enumerate(zip(ds_ids, ds_obs_last)) if pt is not None]
        valid_aids = [(j, aid, anchor_pts[j]) for j, aid in enumerate(anchor_aids)]
        if len(valid_ds) == 0 or len(valid_aids) == 0:
            continue

        D = np.zeros((len(valid_ds), len(valid_aids)), dtype=np.float32)
        for ii, (_, _, pt) in enumerate(valid_ds):
            D[ii] = np.linalg.norm(valid_aids[0][2] - pt)
        # computing distance matrix
        for ii, (_, _, pt) in enumerate(valid_ds):
            for jj, (_, _, ap) in enumerate(valid_aids):
                D[ii, jj] = np.linalg.norm(pt - ap)

        # threshold filer
        thr = 3.0  # m
        M = D.copy()
        M[M > thr] = 1e6

        row_ind, col_ind = linear_sum_assignment(M)

        # Establish a one-to-one mapping from ds_id to aid (filter out over-threshold pairs)
        pairings = []
        for r_idx, c_idx in zip(row_ind, col_ind):
            if M[r_idx, c_idx] >= 1e6:
                continue
            ds_i, ds_id, _ = valid_ds[r_idx]
            aid_j, aid, _ = valid_aids[c_idx]
            pairings.append((ds_id, aid))

        for ds_id, aid in pairings:
            ag = agents_by_id.get(aid)
            prior = ag["prior_xy"]
            T = len(prior)
            if cursor >= T:
                continue
            c_world, c_head = center_pose[ds_id]
            pred12 = pred12_local[ds_id]
            pred12_world = local_to_world_m(pred12, c_world, c_head)
            k = min(pred_len, max(0, T - cursor))
            if k <= 0:
                continue
            prior[cursor: cursor + k, :] = pred12_world[:k]

            # log for visualization
            if aid not in traj_record:
                traj_record[aid] = {
                    "type": ("PEDESTRIAN" if ag.get("typ") in (0, "ped") else "VEHICLE"),
                    "hist_world": None, "gt_world": None, "preds_world": []
                }
            traj_record[aid]["preds_world"].append(pred12_world.copy())
            # Next round of history
            if cursor + pred_len - obs_len >= 0:
                hist_world_next = prior[cursor + pred_len - obs_len: cursor + pred_len]
                traj_record[aid]["hist_world"] = hist_world_next.copy()

        rounds_done = r + 1

        # Generate the next round of pkl (r+1), and stop if there is no active agent
        next_dir = os.path.join(out_dir, f"round_{r+1:03d}")
        _, next_active = build_round_pkl_from_scene(r=r+1, round_dir=next_dir)
        if len(next_active) == 0:
            print(f"[stop] round_{r+1:03d}: no active agents left.")
            break

    # Visualization: original prior + final prior + prediction segments for each round
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    if map_features is not None and draw_vector_map_fn is not None:
        draw_vector_map_fn(ax, map_features)

    if rounds_done <= 0:
        rounds_done = 1
    cmap_colors = plt.cm.autumn(np.linspace(0, 1, rounds_done))
    clr_prior_orig  = (0.75, 0.75, 0.75, 0.9)
    clr_prior_final = (0.10, 0.10, 0.10, 0.95)

    for aid, ag in agents_by_id.items():
        # # Final prior (after writeback)
        # prf = ag["prior_xy"]
        # if prf is not None and len(prf) > 1:
        #     ax.plot(prf[:, 0], prf[:, 1], color="tab:green", linewidth=1.6, zorder=2)
        #     ax.scatter(prf[:, 0], prf[:, 1], color="tab:green", s=10, alpha=0.9, marker="x", zorder=4)

        # Original prior (entire GMM + spline)
        pr0 = orig_prior_xy.get(aid, None)
        if pr0 is not None and len(pr0) > 1:
            ax.plot(pr0[:, 0], pr0[:, 1], color="tab:red", linewidth=1.0, linestyle="--", zorder=1)
            ax.scatter(pr0[:, 0], pr0[:, 1], color="tab:red", s=10, alpha=0.9, marker="o", zorder=4)

        # Each round of prediction
        rec = traj_record.get(aid, None)
        if rec is not None and "preds_world" in rec:
            for rr, seg in enumerate(rec["preds_world"]):
                if seg is None or len(seg) == 0:
                    continue
                clr = cmap_colors[rr]
                ax.plot(seg[:, 0], seg[:, 1], color="tab:blue", linewidth=1.2, alpha=0.9, zorder=3)
                ax.scatter(seg[:, 0], seg[:, 1], color="tab:blue", s=10, alpha=0.9, marker="^", zorder=4)

    out_png = os.path.join(out_dir, "rollout_all_agents.png")
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved overlay figure: {out_png}")



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg["eval"] = True

    # 1. Generate prior scenarios using GMM
    print("[stage1] Building GMM scene ...")
    scene = build_gmm_scene(
        gmms_dir="unitraj/gmm_files",
        n_ped=10, n_veh=10,
        ped_dir="all", veh_dir="all",     # options: "ns"/"ew"/"all"
        dt=0.4, obs_len=8, n_wpts=20,
        n_skip=0, min_ll=-96.0, div=20.0,
        seed=0,
    )
    scene_json_path = "unitraj/scene_gmm.json"
    with open(scene_json_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"[ok] scene JSON saved → {scene_json_path}")

    # 2. refine trajectories by calling rollout_scene
    map_features = build_map_features(csv_path="unitraj/map_features.csv", div=20.0, img_h_px=836)
    rollout_scene(
        scene_json_path=scene_json_path,  # Global Planning files
        cfg=cfg,
        out_dir="unitraj/rollouts_scene_gmm6",
        max_rounds=5,
        small_motion_thresh_m=0.05,
        small_motion_last_k=3,
        map_features=map_features,
        draw_vector_map_fn=add_vector_map
    )

if __name__ == "__main__":
    main()