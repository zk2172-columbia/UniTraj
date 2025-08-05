import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.datasets.common_utils import rotate_points_along_z
from unitraj.utils.utils import set_seed


def transform_to_global(local_traj, center_xyz, center_heading):
    """
    将局部轨迹转换到全局坐标系
    local_traj: (T,2)
    center_xyz: (3,)
    center_heading: float
    """
    local_traj = np.asarray(local_traj, dtype=np.float32).reshape(-1, 2)
    local_traj = local_traj[None, :, :]  # (1,T,2)
    angle = np.array([center_heading], dtype=np.float32)
    global_traj = rotate_points_along_z(local_traj, angle)[0, :, :2]
    global_traj += center_xyz[:2]
    return global_traj


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

    # 5. 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))
    first_sample = val_set[idxs[0]]
    map_polylines = first_sample['map_polylines'][..., :2]  # (num_roads,num_points,2)
    for lane in map_polylines:
        if np.allclose(lane, 0):  # padding lane
            continue
        ax.plot(lane[:, 0], lane[:, 1], color='grey', linewidth=0.5, alpha=0.5)

    # 6. 遍历场景中的每个 agent
    colors = cm.get_cmap('tab20', len(idxs))
    for c_idx, idx in enumerate(tqdm(idxs)):
        sample = val_set[idx]

        # 构建 batch 输入给模型
        batch = val_set.collate_fn([sample])
        batch = {'input_dict': batch['input_dict']}
        for k in batch['input_dict']:
            if isinstance(batch['input_dict'][k], torch.Tensor):
                batch['input_dict'][k] = batch['input_dict'][k].to(device)

        # 模型预测
        with torch.no_grad():
            output, _ = model(batch)

        # 当前中心 agent index
        center_idx = int(sample['track_index_to_predict'])

        # 历史轨迹（绿）
        past_traj_local = sample['obj_trajs'][center_idx, :, :2]
        past_traj_global = transform_to_global(
            past_traj_local,
            sample['center_objects_world'][:3],
            sample['center_objects_world'][6]
        )

        # GT未来轨迹（蓝）
        gt_future_local = sample['obj_trajs_future_state'][center_idx, :, :2]
        gt_future_global = transform_to_global(
            gt_future_local,
            sample['center_objects_world'][:3],
            sample['center_objects_world'][6]
        )

        # 预测未来轨迹（红，取第一条mode）
        pred_future_local = output['predicted_trajectory'][0, 0, :, :2].cpu().numpy()
        pred_future_global = transform_to_global(
            pred_future_local,
            sample['center_objects_world'][:3],
            sample['center_objects_world'][6]
        )

        # 调试打印
        print("Past traj global:", past_traj_global[:3])
        print("GT future global:", gt_future_global[:3])
        print("Pred future global:", pred_future_global[:3])

        # 绘制三条轨迹
        ax.plot(past_traj_global[:, 0], past_traj_global[:, 1],
                color='green', linewidth=2, alpha=0.8)
        ax.plot(gt_future_global[:, 0], gt_future_global[:, 1],
                color='blue', linewidth=2, alpha=0.8)
        ax.plot(pred_future_global[:, 0], pred_future_global[:, 1],
                color='red', linewidth=2, alpha=0.8)

        # 在轨迹终点画个小圆点
        ax.scatter(pred_future_global[-1, 0], pred_future_global[-1, 1],
                   color='red', s=20)
        ax.scatter(gt_future_global[-1, 0], gt_future_global[-1, 1],
                   color='blue', s=20)

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("scene_prediction.png", dpi=150)
    print("Visualization saved to scene_prediction.png")


if __name__ == '__main__':
    main()
