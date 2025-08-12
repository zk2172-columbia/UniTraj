import os
import sys
import pickle
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from omegaconf import OmegaConf

# ---- Manual Configuration ----
# Training Data train root directory (contains cosmos_0, cosmos_1, ...)
TRAIN_ROOT = "/home/wz2708/unitraj/unitraj/data_samples/cosmos/metadrive3/cosmos_l20_f2p5/train"
# Output clustering file path
OUTPUT_PKL = "models/mtr/metadrive3_64_center_4p8s.pkl"

# Data and loading settings
PAST_LEN = 8
FUTURE_LEN = 12           # 4.8s @ 2.5Hz
LOAD_NUM_WORKERS = 8
BATCH_SIZE = 1024
RANDOM_SEED = 42

# Optional: Maximum number of samples (to avoid excessive memory usage)
MAX_SAMPLES_PER_CLASS = None   # if there is no limit, set it to None

# Required object type (MTR supports VEHICLE/PEDESTRIAN/CYCLIST)
OBJECT_TYPES = ['VEHICLE', 'PEDESTRIAN']


def find_train_paths(root: str):
    """Recursively collect all cosmos_* subdirectories under TRAIN_ROOT"""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"TRAIN_ROOT not found: {root}")
    paths = sorted([p for p in glob(os.path.join(root, "cosmos_*")) if os.path.isdir(p)])
    if not paths:
        raise FileNotFoundError(f"No cosmos_* subfolders under {root}")
    return paths


def build_cfg(train_paths):
    """Construct a minimal viable cfg (bypassing Hydra)"""
    # BaseDataset/MTRDataset will read these keys
    cfg = OmegaConf.create({
        'seed': RANDOM_SEED,
        'load_num_workers': LOAD_NUM_WORKERS,
        'train_data_path': train_paths,
        'val_data_path': [],
        'cache_path': './cache',
        'max_data_num': [None] * len(train_paths),
        'starting_frame': [0] * len(train_paths),

        'past_len': PAST_LEN,
        'future_len': FUTURE_LEN,
        'trajectory_sample_interval': 1,

        'object_type': OBJECT_TYPES,
        'line_type': ['lane','stop_sign','road_edge','road_line','crosswalk','speed_bump'],
        'masked_attributes': ['z_axis','size'],

        'only_train_on_ego': False,
        'center_offset_of_map': [0.0, 0.0],

        'use_cache': False,
        'overwrite_cache': True,
        'store_data_in_memory': False,

        'max_num_agents': 64,
        'map_range': 100,
        'max_num_roads': 384,
        'max_points_per_lane': 30,
        'manually_split_lane': False,
        'point_sampled_interval': 1,
        'num_points_each_polyline': 20,
        'vector_break_dist_thresh': 1.0,
    })
    return cfg


def main():
    try:
        from unitraj.datasets import MTR_dataset
    except ImportError:
        proj_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(proj_root)
        from unitraj.datasets import MTR_dataset

    # 1) Collect training data path
    train_paths = find_train_paths(TRAIN_ROOT)
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    print(f"Found {len(train_paths)} train splits:")
    for p in train_paths:
        print("  -", p)

    # 2) Construct cfg and create dataset & loader
    cfg = build_cfg(train_paths)
    dataset = MTR_dataset.MTRDataset(cfg)   # is_validation=False by default
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=cfg.load_num_workers,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    # 3) Cluster Collection
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    vehicle_pts = []
    ped_pts = []
    cyc_pts = []

    total = 0
    for batch in loader:
        inputs = batch['input_dict']
        # (B, T, 4) -> only use (x,y)
        ego_future_xy = inputs['center_gt_trajs'][..., :2]  # [B, T, 2]
        # The last valid time index of each sample
        last_idx = inputs['center_gt_final_valid_idx'].to(torch.int64)  # [B]
        # Get the corresponding final (x, y)
        # Gather requires shape alignment: [B, 1, 1] -> repeat to [B, 1, 2]
        final_xy = torch.gather(
            ego_future_xy, 1,
            last_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
        ).squeeze(1)  # [B, 2]

        types = inputs['center_objects_type']  # numpy array
        v_mask = (types == 1)
        p_mask = (types == 2)
        c_mask = (types == 3)

        if v_mask.any():
            vehicle_pts.append(final_xy[v_mask].cpu().numpy())
        if p_mask.any():
            ped_pts.append(final_xy[p_mask].cpu().numpy())
        if c_mask.any():
            cyc_pts.append(final_xy[c_mask].cpu().numpy())

        total += len(types)
        if total % 5000 == 0:
            print(f"Scanned {total} samples...")

    def _concat_limit(chunks):
        if not chunks:
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.concatenate(chunks, axis=0)
        if MAX_SAMPLES_PER_CLASS is not None and len(arr) > MAX_SAMPLES_PER_CLASS:
            idx = np.random.choice(len(arr), size=MAX_SAMPLES_PER_CLASS, replace=False)
            arr = arr[idx]
        return arr

    vehicle_arr = _concat_limit(vehicle_pts)
    ped_arr = _concat_limit(ped_pts)
    cyc_arr = _concat_limit(cyc_pts)

    print(f"Collected points -> VEHICLE: {len(vehicle_arr)}, PEDESTRIAN: {len(ped_arr)}, CYCLIST: {len(cyc_arr)}")

    # 4) KMeans(n_clusters=64)
    kmeans = KMeans(n_clusters=64, n_init='auto', random_state=RANDOM_SEED)
    cluster_dict = {}

    if len(vehicle_arr) > 0:
        print("Clustering VEHICLE...")
        cluster_dict['VEHICLE'] = kmeans.fit(vehicle_arr).cluster_centers_
    if len(ped_arr) > 0:
        print("Clustering PEDESTRIAN...")
        cluster_dict['PEDESTRIAN'] = kmeans.fit(ped_arr).cluster_centers_

    # 5) save
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(cluster_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved clusters to: {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
