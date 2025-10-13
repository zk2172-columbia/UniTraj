"""
Read from GMM file -> Sample -> Unify splines to dt=0.4s -> Return to "Scene Container" (meter-based).
No PKL/visualization/model inference is performed to maintain decoupling.
"""

from __future__ import annotations
from typing import Dict, Any, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import copy
from scipy.interpolate import make_interp_spline

# GMM Processing
# -----------------------------
def load_stats(gmms_dir: str):
    """
    Read GMM and ToD statistics
    Returns a tuple, which the outer main function unpacks and uses.
    """
    with open(f"{gmms_dir}/peds_wpt_gmms.pkl", "rb") as f:
        ped_wpt_gmm = pickle.load(f)
    with open(f"{gmms_dir}/vehs_wpt_gmms.pkl", "rb") as f:
        veh_wpt_gmm = pickle.load(f)

    with open(f"{gmms_dir}/peds_typ_gmms.pkl", "rb") as f:
        ped_typ_gmm = pickle.load(f)
    with open(f"{gmms_dir}/vehs_typ_gmms.pkl", "rb") as f:
        veh_typ_gmm = pickle.load(f)

    with open(f"{gmms_dir}/peds_tod_gmms.pkl", "rb") as f:
        ped_tod_gmm = pickle.load(f)
    with open(f"{gmms_dir}/vehs_tod_gmms.pkl", "rb") as f:
        veh_tod_gmm = pickle.load(f)

    ped_day_cnts = pd.read_csv(f"{gmms_dir}/peds_tods.csv")
    veh_day_cnts = pd.read_csv(f"{gmms_dir}/vehs_tods.csv")

    return (
        ped_wpt_gmm, veh_wpt_gmm,
        ped_typ_gmm, veh_typ_gmm,
        ped_tod_gmm, veh_tod_gmm,
        ped_day_cnts, veh_day_cnts,
    )

def _renorm_with_eps(weights, eps=1e-12):
    w = weights.astype(float).copy()
    # Fix the all 0 situation (extremely small probability)
    if np.all(w == 0):
        w += eps
    # preventing log(0)
    w[w == 0] = eps
    w /= w.sum()
    return w

def build_wpt_dir_gmms(wpt_gmm, ns_comp, ew_comp):
    wpt_gmm_ns = copy.copy(wpt_gmm)
    w_ns = np.zeros_like(wpt_gmm.weights_)
    w_ns[ns_comp] = wpt_gmm.weights_[ns_comp]
    wpt_gmm_ns.weights_ = _renorm_with_eps(w_ns)

    wpt_gmm_ew = copy.copy(wpt_gmm)
    w_ew = np.zeros_like(wpt_gmm.weights_)
    w_ew[ew_comp] = wpt_gmm.weights_[ew_comp]
    wpt_gmm_ew.weights_ = _renorm_with_eps(w_ew)
    return wpt_gmm_ns, wpt_gmm_ew

def generate_wpts(
    wpt_gmm,
    n_wpts: int,
    min_ll: float,
    div: float,
    n: int,
    drop_component: Optional[int] = 8,
) -> pd.DataFrame:
    """
    Draw n samples from the specified waypoint using a GMM and filter by log-likelihood.
    The output DataFrame has coordinates in meters (internal implementation).
    """
    data, _ = wpt_gmm.sample(n)
    cols = (
        ["pxs", "pys", "pxe", "pye", "vxs", "vys", "vxe", "vye"]
        + [f"wpx{i}" for i in range(n_wpts)]
        + [f"wpy{i}" for i in range(n_wpts)]
        + ["t"]
    )
    samples = pd.DataFrame(data, columns=cols)
    tps = wpt_gmm.predict(samples)
    lls = wpt_gmm.score_samples(samples)
    samples["tp"] = tps
    samples["ll"] = lls

    # Empirical filtering: remove certain components (default 8) with low likelihood
    if drop_component is not None:
        samples = samples[tps != drop_component]
    samples = samples.groupby("tp", group_keys=False).apply(
        lambda g: g[g["ll"] > min_ll]
    )

    # Coordinates are unified to meters
    meter_cols = [c for c in samples.columns if c not in ("tp", "t", "ll")]
    samples.loc[:, meter_cols] = samples.loc[:, meter_cols] / div
    return samples

def _sample_enough(gmm, want, *, n_wpts, min_ll, div, chunk=200, drop_component=8):
    keep = []
    while len(keep) < want:
        df = generate_wpts(gmm, n_wpts=n_wpts, min_ll=min_ll, div=div,
                           n=chunk, drop_component=drop_component)
        # Optional: Make sure t is long enough to give at least 8 frames of history
        df = df[df["t"] >= 8 * 0.4]
        keep.append(df)
        if sum(len(x) for x in keep) >= 10_000:  # Preventing infinite loops
            break
    out = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    return out.head(want)

# Splines and Packing
# -----------------------------
def row_to_prior(
    row: pd.Series,
    dt: float,
    n_wpts: int,
    n_skip: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single sample: 20 landmarks + start and end speeds -> Construct a spline
    Sample dt on [0, t] to obtain the prior (in meters)
    Return:
    prior_xy: (T, 2)
    prior_t: (T,)
    """
    # 20 waypoints (meters)
    wpx = np.array([row[f"wpx{i}"] for i in range(n_wpts)], dtype=float)
    wpy = np.array([row[f"wpy{i}"] for i in range(n_wpts)], dtype=float)
    wps = np.stack([wpx, wpy], axis=1)  # (20, 2)

    # Start and end speed (m/s)
    v_start = np.array([row["vxs"], row["vys"]], dtype=float)
    v_end = np.array([row["vxe"], row["vye"]], dtype=float)

    t_total = float(row["t"])

    # Use parameterization with 20 waypoints and evenly mapped the time axis to [0, t_total]
    bspl = make_interp_spline(
        np.linspace(0.0, t_total, wps.shape[0]),  # 20 equal parts
        wps,
        bc_type=([(1, v_start)], [(1, v_end)]),   # First derivative boundary condition = endpoint velocity
    )

    # Use uniform dt sampling; n_skip at both ends can be trimmed as needed
    tt = np.arange(0.0, t_total, dt)
    if n_skip > 0:
        tt = tt[n_skip:-n_skip]
    prior_xy = bspl(tt).astype(float)             # (T, 2)
    prior_t = tt.astype(float)                    # (T,)
    return prior_xy, prior_t


def bundle_agent(
    typ: Literal["ped", "veh"],
    agent_id: str,
    row: pd.Series,
    prior_xy: np.ndarray,
    prior_t: np.ndarray,
    n_wpts: int,
) -> Dict[str, Any]:
    """Pack all fields of the agent into a unified dictionary (meter unit)"""
    agent = {
        "id": agent_id,
        "typ": typ,
        "t_total": float(row["t"]),
        "component": int(row["tp"]),
        "loglik": float(row["ll"]),
        "start_xy": [float(row["pxs"]), float(row["pys"])],
        "end_xy":   [float(row["pxe"]), float(row["pye"])],
        "start_v":  [float(row["vxs"]), float(row["vys"])],
        "end_v":    [float(row["vxe"]), float(row["vye"])],
        "waypoints": [[float(row[f"wpx{i}"]), float(row[f"wpy{i}"])] for i in range(n_wpts)],
        "prior_xy": prior_xy.tolist(),            # meter
        "prior_t": prior_t.tolist(),              # second
        "hist8_xy": prior_xy[:8].tolist(),       # Convenient for subsequent pkl package
    }
    return agent


def _pick_gmm_by_dir(
    wpt_gmm, dir_sel: Literal["ns", "ew", "all"],
    ns_ew_pair: Tuple[Any, Any]  # (gmm_ns, gmm_ew)
):
    if dir_sel == "ns":
        return ns_ew_pair[0]
    if dir_sel == "ew":
        return ns_ew_pair[1]
    return wpt_gmm  # "all": Use original blend (no direction restriction)


# scene container
# -----------------------------
def build_gmm_scene(
    gmms_dir: str,
    *,
    n_ped: int,
    n_veh: int,
    ped_dir: Literal["ns", "ew", "all"] = "ns",
    veh_dir: Literal["ns", "ew", "all"] = "ns",
    # The direction component
    ped_ns_comp: List[int] = (0, 1, 2, 3, 4, 5),
    ped_ew_comp: List[int] = (6, 7, 8, 9, 10, 11),
    veh_ns_comp: List[int] = (1, 2, 3, 4, 6, 8),
    veh_ew_comp: List[int] = (0, 5, 7, 9, 10, 11),
    dt: float = 0.4,
    obs_len: int = 8,
    n_wpts: int = 20,
    n_skip: int = 0,
    min_ll: float = -96.0,
    div: float = 20.0,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Outputs "the complete prior trajectory and information of all agents in this scene" (unit: meters)

    Return dict structure：
    {
      "meta": {...},
      "agents": [ {...}, {...}, ... ],
      "gmms_meta": {...}
    }
    """
    if seed is not None:
        np.random.seed(seed)

    (
        ped_wpt_gmm, veh_wpt_gmm,
        ped_typ_gmm, veh_typ_gmm,
        ped_tod_gmm, veh_tod_gmm,
        ped_day_cnts, veh_day_cnts,
    ) = load_stats(gmms_dir)

    # Constructing NS/EW submixes
    ped_wpt_ns, ped_wpt_ew = build_wpt_dir_gmms(ped_wpt_gmm, list(ped_ns_comp), list(ped_ew_comp))
    veh_wpt_ns, veh_wpt_ew = build_wpt_dir_gmms(veh_wpt_gmm, list(veh_ns_comp), list(veh_ew_comp))

    # filtering directions
    ped_gmm_sel = _pick_gmm_by_dir(ped_wpt_gmm, ped_dir, (ped_wpt_ns, ped_wpt_ew))
    veh_gmm_sel = _pick_gmm_by_dir(veh_wpt_gmm, veh_dir, (veh_wpt_ns, veh_wpt_ew))

    # Sampling (coordinates are already in meters)
    ped_df = generate_wpts(ped_gmm_sel, n_wpts=n_wpts, min_ll=min_ll, div=div, n=n_ped)
    veh_df = generate_wpts(veh_gmm_sel, n_wpts=n_wpts, min_ll=min_ll, div=div, n=n_veh)
    # ped_df = _sample_enough(ped_gmm_sel, n_ped, n_wpts=n_wpts, min_ll=min_ll, div=div)
    # veh_df = _sample_enough(veh_gmm_sel, n_veh, n_wpts=n_wpts, min_ll=min_ll, div=div)

    agents: List[Dict[str, Any]] = []

    # pedestrian
    for i, row in ped_df.reset_index(drop=True).iterrows():
        prior_xy, prior_t = row_to_prior(row, dt=dt, n_wpts=n_wpts, n_skip=n_skip)
        agent = bundle_agent("ped", f"p{i}", row, prior_xy, prior_t, n_wpts=n_wpts)
        agents.append(agent)

    # vehicles
    for i, row in veh_df.reset_index(drop=True).iterrows():
        prior_xy, prior_t = row_to_prior(row, dt=dt, n_wpts=n_wpts, n_skip=n_skip)
        agent = bundle_agent("veh", f"v{i}", row, prior_xy, prior_t, n_wpts=n_wpts)
        agents.append(agent)

    scene: Dict[str, Any] = {
        "meta": {
            "dt": float(dt),
            "obs_len": int(obs_len),
            "n_wpts": int(n_wpts),
            "div": float(div),
            "coord": "meters",
        },
        "agents": agents,
        "gmms_meta": {
            "ped": {"ns_components": list(ped_ns_comp), "ew_components": list(ped_ew_comp)},
            "veh": {"ns_components": list(veh_ns_comp), "ew_components": list(veh_ew_comp)},
        },
    }
    return scene

if __name__ == "__main__":
    scene = build_gmm_scene(
        gmms_dir="gmm_files",
        n_ped=5, n_veh=5,
        ped_dir="all", veh_dir="all",  # Can be changed to "ns"/"ew"/"all"
        dt=0.4, obs_len=8, n_wpts=20,
        n_skip=0, min_ll=-96.0, div=20.0,
        seed=0,
    )

    # scene["agents"] contains the full prior of all agents (in meters)
    print("agents:", len(scene["agents"]))
    print("meta:", scene["meta"])
    print("example agent keys:", scene["agents"][0].keys())
    print("first agent prior length:", len(scene["agents"][0]["prior_xy"]))
    print("first agent hist8:", scene["agents"][0]["hist8_xy"])
