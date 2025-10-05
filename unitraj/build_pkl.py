import numpy as np
import pickle
from collections import defaultdict
import os, json, pickle, time, gc, math
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from collections import defaultdict

# Map feature type mapping (COSMOS -> NuScenes-ish)
COSMOS_TO_NUSCENE_TYPE = {
    "CROSSWALK": "CROSSWALK",
    "DRIVABLE_AREA": "DRIVABLE_AREA",
    "LANE_SURFACE_STREET": "LANE_SURFACE_STREET",
    "LANE_BIKE_LANE": "LANE_BIKE_LANE",
}

# MAP: polygon -> directional centerline
def polygon_to_centerline_directional(polygon_coords: np.ndarray, lane_name: str) -> np.ndarray:
    if polygon_coords.shape[0] < 3:
        return polygon_coords.astype(np.float32)

    poly = Polygon(polygon_coords)
    if (not poly.is_valid) or poly.is_empty:
        return polygon_coords.astype(np.float32)

    lane_name = str(lane_name).lower()
    is_vertical   = any(d in lane_name for d in ["nv", "sv"])
    is_horizontal = any(d in lane_name for d in ["wv", "ev"])
    xs = polygon_coords[:, 0]
    ys = polygon_coords[:, 1]

    if is_vertical:
        x_center = (xs.min() + xs.max()) / 2.0
        return np.array([[x_center, ys.min()], [x_center, ys.max()]], dtype=np.float32)

    if is_horizontal:
        y_center = (ys.min() + ys.max()) / 2.0
        return np.array([[xs.min(), y_center], [xs.max(), y_center]], dtype=np.float32)

    # fallback: long axis of minimum rotated rectangle
    min_rect = poly.minimum_rotated_rectangle
    coords = np.array(min_rect.exterior.coords[:-1])
    v1, v2 = coords[1] - coords[0], coords[2] - coords[1]
    if np.linalg.norm(v1) < np.linalg.norm(v2):
        m0, m1 = (coords[0] + coords[1]) / 2, (coords[2] + coords[3]) / 2
    else:
        m0, m1 = (coords[1] + coords[2]) / 2, (coords[3] + coords[0]) / 2
    return np.array([m0, m1], dtype=np.float32)

def build_map_features(csv_path = "unitraj/map_features.csv", div = 20.0, img_h_px = 836):
    """
    Vectorize map once from the VIA CSV.
    - Flip y to Cartesian (upwards),
    - scale pixels->meters by 1/div,
    - compute a centerline polyline for non-DRIVABLE_AREA polygons.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for rsa, ra in zip(df["region_shape_attributes"], df["region_attributes"]):
        rsa, ra = json.loads(rsa), json.loads(ra)
        lane_name = ra.get("location", "L") + str(ra.get("type", ["x"])[0])
        polygon = np.array(list(zip(rsa["all_points_x"], rsa["all_points_y"])), dtype=np.float32)
        polygon[:, 1] = img_h_px - polygon[:, 1]  # flip y
        polygon /= div

        mtype = COSMOS_TO_NUSCENE_TYPE.get(ra.get("map_feature", ""), "unknown")
        rec = {"type": mtype, "polygon": polygon}

        if mtype != "DRIVABLE_AREA":
            rec["polyline"] = polygon_to_centerline_directional(polygon, lane_name)

        out[f"lane_{lane_name}"] = rec
    return out

# TODO: need to be modified according to different scene
def build_traffic_lights_synth(seq_len=20, div=20.0):
    """
    Create 8 traffic-light entries (ev, wv, nv, sv, ep, wp, np, sp) from a simple 4-phase vehicle matrix.
    Pattern: 10 steps EW green (NS red), then 10 steps NS green (EW red); repeat if needed.
    """
    # Base vehicle lights for [ev, wv, nv, sv], 1=GO, 0=STOP
    phase_len = 10
    tfl_raw = np.zeros((seq_len, 4), dtype=np.int64)
    for t in range(seq_len):
        ew_green = (t // phase_len) % 2 == 0
        if ew_green:
            tfl_raw[t, :] = [1, 1, 0, 0]  # ev, wv, nv, sv
        else:
            tfl_raw[t, :] = [0, 0, 1, 1]

    # Duplicate to 8 columns (vehicle+ped), match previous indexing
    tfl = tfl_raw[:, [0, 0, 2, 2, 1, 1, 3, 3]]  # [ev,ep,nv,np,wv,wp,sv,sp]

    # Stop points (pixel coords from your code) -> meters
    lanes  = ['ev','wv','nv','sv','ep','wp','np','sp']
    stops_px = [(38,498),(800,449),(381,122),(447,800),(167,439),(676,443),(417,228),(420,679)]

    traffic_lights = {}
    for i, (name, pt) in enumerate(zip(lanes, stops_px)):
        lane_id = f"lane_{name}"
        sp = np.array([pt[0]/div, (836-pt[1])/div, 0.0], dtype=np.float32)  # flip y + scale
        traffic_lights[lane_id] = {
            "type": "TRAFFIC_LIGHT",
            "state": {
                "object_state": [
                    "LANE_STATE_GO" if int(val)==1 else "LANE_STATE_STOP"
                    for val in tfl[:, i]
                ]
            },
            "lane": lane_id,
            "controlled_lane_ids": [lane_id] if name.endswith('v') else [],
            "stop_point": sp,
            "metadata": {
                "type": "TRAFFIC_LIGHT",
                "track_length": seq_len,
                "object_id": lane_id,
            }
        }
    return traffic_lights

def compute_continuous_valid_length(valid_mask: np.ndarray) -> int:
    max_len = curr_len = 0
    for v in valid_mask:
        if v:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0
    return max_len

def compute_moving_distance(positions: np.ndarray, valid_mask: np.ndarray) -> float:
    if positions.shape[1] > 2:
        positions = positions[:, :2]
    valid_positions = positions[valid_mask]
    if len(valid_positions) < 2:
        return 0.0
    diffs = valid_positions[1:] - valid_positions[:-1]
    dist = np.linalg.norm(diffs, axis=1)
    return dist.sum()

def build_object_summary(tracks: dict) -> dict:
    summary = {}
    for object_id, track in tracks.items():
        valid = track["state"]["valid"]
        position = track["state"]["position"]
        obj_type = track["type"]
        track_length = len(position)
        summary[object_id] = {
            "type": obj_type,
            "object_id": object_id,
            "track_length": track_length,
            "valid_length": int(np.sum(valid)),
            "continuous_valid_length": compute_continuous_valid_length(valid),
            "moving_distance": float(compute_moving_distance(position, valid)),
        }
    return summary

def build_number_summary(object_summary, traffic_lights, map_features, seq_len=20):
    types_set = set()
    type_counts = defaultdict(int)
    moving_set = set()
    moving_counts = defaultdict(int)

    for obj in object_summary.values():
        obj_type = obj["type"]
        types_set.add(obj_type)
        type_counts[obj_type] += 1
        if obj["moving_distance"] > 0.1:
            moving_set.add(obj_type)
            moving_counts[obj_type] += 1

    tfl_count = len(traffic_lights)
    tfl_types = {v["type"] for v in traffic_lights.values()}
    tfl_each_step = {str(i): tfl_count for i in range(seq_len)}

    num_map_features = len(map_features)

    return {
        "num_objects": sum(type_counts.values()),
        "object_types": types_set,
        "num_objects_each_type": dict(type_counts),
        "num_moving_objects": sum(moving_counts.values()),
        "num_moving_objects_each_type": dict(moving_counts),
        "num_traffic_lights": tfl_count,
        "num_traffic_light_types": tfl_types,
        "num_traffic_light_each_step": tfl_each_step,
        "num_map_features": num_map_features,
        "map_height_diff": float("-inf"),
    }

# Final package function
def build_custom_pkl(agents_histories, agents_types, output_path=None, scenario_id=0, fps=2.5, div=20):
    """
    Construct a MetaDrive-formatted pkl file from scratch.
    - agents_histories: dict {agent_id: np.ndarray(8,2)} Input history trajectory
    - agents_types: dict {agent_id: "VEHICLE"/"PEDESTRIAN"}
    - map_features: dict (generated by build_map_features)
    - output_path: str (optional). If given, writes the pkl file.
    """

    obs_len, tgt_len = 8, 12
    seq_len = obs_len + tgt_len

    # ==== central anchor ====
    center = np.array([416,416,0], np.float32) / div
    tracks = {}
    tracks["center"] = {
        "type":"ANCHOR",
        "state":{
            "position":np.tile(center[None], (seq_len,1)),
            "heading":np.zeros(seq_len,np.float32),
            "length":np.ones(seq_len,np.float32)/100,
            "width": np.ones(seq_len,np.float32)/100,
            "height":np.ones(seq_len,np.float32)/100,
            "velocity":np.zeros((seq_len,2),np.float32),
            "valid":np.ones(seq_len, bool),
        },
        "metadata":{"type":"ANCHOR","track_length":seq_len,"object_id":"center"}
    }

    tracks_to_predict = {}
    index = 1

    # ==== agents ====
    for agent_id, hist in agents_histories.items():
        obj_type = agents_types.get(agent_id)
        pos = np.zeros((seq_len,3), np.float32)
        val = np.zeros(seq_len, bool)

        # history (0..7)
        hist = hist.astype(np.float32).copy()
        # Flip the y-axis (same as the original build_tracks: obs[:,1] = 836 - obs[:,1])
        hist[:, 1] = 836 - hist[:, 1]
        # Pixel → Meter
        hist /= 20.0

        pos[:obs_len, :2] = hist
        # val[:obs_len] = True
        val[:] = True
        # Set the future segment (8..19) to zero
        # Velocity is only calculated when two consecutive frames are valid
        vel = np.zeros((seq_len,2), np.float32)
        dv = pos[1:obs_len,:2] - pos[:obs_len-1,:2]
        vel[1:obs_len,:] = dv * fps
        vel[0] = vel[1]

        tracks[agent_id] = {
            "type": obj_type,
            "state": {
                "position": pos,
                "heading": np.zeros(seq_len, np.float32),
                "length": np.ones(seq_len, np.float32),
                "width":  np.ones(seq_len, np.float32),
                "height": np.ones(seq_len, np.float32),
                "velocity": vel,
                "valid": val,
            },
            "metadata": {"type":obj_type, "track_length":seq_len, "object_id":agent_id}
        }

        tracks_to_predict[agent_id] = {
            "track_index": index,
            "track_id": agent_id,
            "difficulty": 0,
            "object_type": obj_type
        }
        index += 1

    # ==== metadata ====
    map_features = build_map_features()
    traffic_lights = build_traffic_lights_synth()
    object_summary = build_object_summary(tracks)
    number_summary = build_number_summary(object_summary, traffic_lights, map_features, seq_len)

    scenario = {
        "id": f"Custom_{scenario_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": seq_len,
        "metadata": {
            "ts": np.arange(seq_len, dtype=np.float32) / fps,
            "metadrive_processed": False,
            "coordinate": "metadrive",
            "source_file": "custom_input",
            "dataset": "cosmos",
            "scenario_id": str(scenario_id),
            "sdc_id": "center",
            "tracks_to_predict": tracks_to_predict,
            "object_summary": object_summary,
            "number_summary": number_summary,
        },
        "tracks": tracks,
        "dynamic_map_states": traffic_lights,
        "map_features": map_features,
    }

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(scenario, f)

    return scenario

if __name__ == "__main__":
    # make two agents' historical trajectories
    def make_straight_line(px0, px1, T):
        """Linear interpolation inclusive of endpoints in pixel space (T points)."""
        x = np.linspace(px0[0], px1[0], T, dtype=np.float32)
        y = np.linspace(px0[1], px1[1], T, dtype=np.float32)
        return np.stack([x, y], axis=1)

    histories = {
        "veh1": make_straight_line(px0=(300, 416), px1=(415, 416), T=8),
        "veh2": make_straight_line(px0=(416, 550), px1=(416, 430), T=8),
        "veh3": make_straight_line(px0=(150, 300), px1=(415, 416), T=8),
        "ped1": make_straight_line(px0=(300, 416), px1=(150, 300), T=8),
        "ped2": make_straight_line(px0=(300, 416), px1=(450, 500), T=8),
    }
    types = {"veh1": "VEHICLE","veh2": "VEHICLE","veh3": "VEHICLE", "ped1": "PEDESTRIAN", "ped2": "PEDESTRIAN"}

    # build pkl
    scenario = build_custom_pkl(
        agents_histories=histories,
        agents_types=types,
        output_path="unitraj/assemble/sd_cosmos_12345.pkl",
        scenario_id=12345
    )

