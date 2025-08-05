import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.datasets.common_utils import rotate_points_along_z
from unitraj.utils.utils import set_seed


def transform_to_global(local_points, center_xyz, center_heading):
    """
    将局部坐标(local_points: (...,2))转换为全局坐标
    """
    pts = np.asarray(local_points, dtype=np.float32).reshape(-1, 2)
    pts = pts[None, :, :]  # (1,N,2)
    angle = np.array([center_heading], dtype=np.float32)
    global_pts = rotate_points_along_z(pts, angle)[0, :, :2]
    global_pts += center_xyz[:2]
    return global_pts.reshape(local_points.shape)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 构建模型并加载权重
    model = build_model(cfg).to(device)
    model.eval()
    if cfg.get('ckpt_path', None):
        ckpt = torch.load(cfg.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"Loaded model checkpoint: {cfg.ckpt_path}")

    # 2. 构建验证集
    val_set = build_dataset(cfg, val=True)

    # 3. 选择一个场景ID
    target_scene_id = cfg.get('vis_scene_id', None)
    if target_scene_id is None:
        first_key = val_set.data_loaded_keys[0]
        target_scene_id = first_key.split('-')[0]
    print(f"Visualizing scene_id={target_scene_id}")

    # 4. 找到该场景的所有 agent 样本索引
    idxs = [i for i, k in enumerate(val_set.data_loaded_keys) if target_scene_id in k]
    print(f"Found {len(idxs)} agents in scene {target_scene_id}")

    # 5. 可视化
    fig, ax = plt.subplots(figsize=(10, 10))

    # 地图绘制（转换到全局坐标）
    first_sample = val_set[idxs[0]]
    center_world = first_sample['center_objects_world'][:3]
    center_heading = first_sample['center_objects_world'][6]

    map_polylines_local = first_sample['map_polylines'][..., :2]  # (num_roads,num_points,2)
    for lane in map_polylines_local:
        if np.allclose(lane, 0):  # padding lane
            continue
        lane_global = transform_to_global(lane, center_world, center_heading)
        ax.plot(lane_global[:, 0], lane_global[:, 1], color='grey', linewidth=0.8, alpha=0.5)
    # 地图绘制（转换到全局坐标）
    # map_polylines_local = first_sample['map_polylines'][..., :2]
    # for lane in map_polylines_local:
    #     if np.allclose(lane, 0):  # padding lane
    #         continue
    #     lane_global = transform_to_global(lane, center_world, center_heading)
    #     ax.plot(lane_global[:, 0], lane_global[:, 1],
    #             color='dimgray', linewidth=1.0, alpha=0.8)  # <-- 提高清晰度

    # 遍历场景中的每个 agent
    for idx in tqdm(idxs):
        sample = val_set[idx]
        batch = val_set.collate_fn([sample])
        batch = {'input_dict': batch['input_dict']}
        for k in batch['input_dict']:
            if isinstance(batch['input_dict'][k], torch.Tensor):
                batch['input_dict'][k] = batch['input_dict'][k].to(device)

        # 模型预测
        with torch.no_grad():
            output, _ = model(batch)

        center_idx = int(sample['track_index_to_predict'])
        center_world = sample['center_objects_world'][:3]
        center_heading = sample['center_objects_world'][6]

        # 历史轨迹（绿色）
        past_traj_global = transform_to_global(sample['obj_trajs'][center_idx, :, :2],
                                               center_world, center_heading)
        ax.plot(past_traj_global[:, 0], past_traj_global[:, 1],
                color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.scatter(past_traj_global[:, 0], past_traj_global[:, 1],
                   color='green', s=15, alpha=0.8, marker='o')

        # 未来GT轨迹（蓝色）
        gt_future_global = transform_to_global(sample['obj_trajs_future_state'][center_idx, :, :2],
                                               center_world, center_heading)
        ax.plot(gt_future_global[:, 0], gt_future_global[:, 1],
                color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.scatter(gt_future_global[:, 0], gt_future_global[:, 1],
                   color='blue', s=15, alpha=0.8, marker='o')

        # 所有预测轨迹
        preds_local = output['predicted_trajectory'][0, :, :, :2].cpu().numpy()  # (num_modes, T, 2)
        ade_all = np.linalg.norm(preds_local - sample['obj_trajs_future_state'][center_idx, :, :2][None,:,:],
                                 axis=-1).mean(-1)
        best_mode = np.argmin(ade_all)

        for mode_idx, pred_future_local in enumerate(preds_local):
            pred_future_global = transform_to_global(pred_future_local, center_world, center_heading)
            if mode_idx == best_mode:
                ax.plot(pred_future_global[:, 0], pred_future_global[:, 1],
                        color='red', linewidth=2.0, alpha=1.0)
                ax.scatter(pred_future_global[:, 0], pred_future_global[:, 1],
                           color='red', s=20, alpha=1.0, marker='^')
            else:
                ax.plot(pred_future_global[:, 0], pred_future_global[:, 1],
                        color='gold', linestyle='--', linewidth=1.0, alpha=0.3)
                ax.scatter(pred_future_global[:, 0], pred_future_global[:, 1],
                           color='gold', s=15, alpha=0.3, marker='^')

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("scene_prediction_global.png", dpi=200)
    print("Visualization saved to scene_prediction_global.png")


if __name__ == '__main__':
    main()
