"""
PRISM — LiDAR Processor
========================
Python port of the C++ lidarPreProcess ROS2 package.

Pipeline (mirrors the C++ nodes):
    Raw PointCloud2
        → Statistical Outlier Removal   (sorFilter.cpp)
        → Voxel Downsampling            (lidarVoxel.cpp)
        → Grid-Adaptive Ground Removal  (lidarGroundRemove.cpp)
        → Range-Adaptive DBSCAN         (EuclideanClusters.cpp / boundingBox.cpp)
        → AABB Bounding Boxes           → LiDARDetection objects

Output LiDARDetection has:
    range_m     — radial distance to cluster centroid (metres)
    bearing_deg — bearing from forward axis (+ = right, − = left)
    lateral_m   — lateral offset from ego centreline (metres, + = right)
    distance_m  — forward distance (metres, along ego X axis)
    height_m    — cluster vertical extent
    n_points    — number of points in cluster
    centroid    — (x, y, z) in sensor/ego frame
    bbox_min/max — AABB corners

Usage:
    proc = LiDARProcessor()
    dets = proc.process(cloud_np)  # cloud_np is (N,4) float32 [x,y,z,intensity]
    for det in dets:
        print(f"  {det.distance_m:.1f}m fwd, {det.lateral_m:.1f}m lat")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from prism.utils.common import get_logger

logger = get_logger("LiDARProcessor")


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class LiDARDetection:
    """
    A detected obstacle from LiDAR processing.
    Coordinates are in LiDAR/ego sensor frame:
        x = forward, y = left, z = up
    """
    centroid:    np.ndarray       # (x, y, z) metres — cluster centre
    bbox_min:    np.ndarray       # (x, y, z) AABB min corner
    bbox_max:    np.ndarray       # (x, y, z) AABB max corner

    range_m:     float = 0.0     # radial distance sqrt(x²+y²)
    bearing_deg: float = 0.0     # bearing: arctan2(y, x) — +ve = left
    distance_m:  float = 0.0     # forward distance (x component)
    lateral_m:   float = 0.0     # lateral offset  (y component, +ve = left)
    height_m:    float = 0.0     # cluster vertical size
    n_points:    int   = 0

    @property
    def threat_zone(self) -> str:
        if self.range_m < 5:   return "CRITICAL"
        if self.range_m < 15:  return "CLOSE"
        if self.range_m < 30:  return "MEDIUM"
        return "FAR"

    def __repr__(self) -> str:
        return (f"LiDARDet(fwd={self.distance_m:.1f}m lat={self.lateral_m:.1f}m "
                f"range={self.range_m:.1f}m  h={self.height_m:.2f}m  n={self.n_points})")


# ── Processing parameters ─────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # SOR (Statistical Outlier Removal)
    "sor_k":              20,       # k nearest neighbours
    "sor_std_ratio":       1.0,     # std dev multiplier

    # Voxel downsampling
    "voxel_size":          0.10,    # metres — leaf size

    # Ground removal (mirrors lidarGroundRemove.cpp defaults)
    "grid_res":            0.5,     # metres per cell
    "height_margin":       0.25,    # metres above cell minimum to classify as ground
    "max_range":          40.0,     # metres — ignore points beyond this
    "dilation_steps":      3,       # cells to dilate ground map
    "min_pts_in_cell":     1,

    # Z passthrough (filter above/below these heights in sensor frame)
    "z_min":              -3.0,
    "z_max":              10.0,

    # Range-adaptive DBSCAN (mirrors boundingBox.cpp defaults)
    "eps_near":            0.20,    # metres at range 0
    "eps_range_factor":    0.05,    # +5% per metre
    "min_pts_near":        5,
    "min_pts_far":         2,
    "far_range":          20.0,
    "min_cluster_near":   30,
    "min_cluster_far":     5,
    "far_cluster_range":  20.0,
    "max_cluster_size": 50000,

    # Cluster filters (mirrors BoundingBoxNode)
    "max_box_dimension":  15.0,     # metres any axis
    "min_box_volume":      0.01,    # m³
    "max_ground_clearance": 0.5,    # m — cluster z_min must be below this
    "min_point_density":   5.0,     # pts/m³
    "density_range_factor": 0.05,
    "merge_distance":       1.5,    # metres — merge nearby AABBs

    # Forward-sector filter (only keep objects ahead of vehicle)
    "min_forward_dist":    0.5,     # metres — ignore objects behind sensor
    "max_forward_dist":   50.0,
    "max_lateral_dist":   20.0,     # metres left/right
}


# ── Statistical Outlier Removal ───────────────────────────────────────────────

def _sor_filter(cloud: np.ndarray, k: int = 20, std_ratio: float = 1.0) -> np.ndarray:
    """
    Remove statistical outliers.
    For each point, compute mean distance to k nearest neighbours.
    Points with mean distance > global_mean + std_ratio * global_std are removed.
    """
    if len(cloud) < k + 1:
        return cloud

    try:
        from sklearn.neighbors import KDTree
        xyz = cloud[:, :3]
        tree = KDTree(xyz)
        dists, _ = tree.query(xyz, k=k + 1)  # includes self
        mean_dists = dists[:, 1:].mean(axis=1)  # exclude self
        global_mean = mean_dists.mean()
        global_std  = mean_dists.std()
        threshold   = global_mean + std_ratio * global_std
        mask = mean_dists <= threshold
        return cloud[mask]
    except ImportError:
        # sklearn not available — skip SOR
        return cloud


# ── Voxel Downsampling ────────────────────────────────────────────────────────

def _voxel_downsample(cloud: np.ndarray, voxel_size: float = 0.10) -> np.ndarray:
    """
    Voxel grid downsampling — keep one centroid per voxel cell.
    Fast pure-numpy implementation.
    """
    if len(cloud) == 0 or voxel_size <= 0:
        return cloud

    xyz    = cloud[:, :3]
    coords = np.floor(xyz / voxel_size).astype(np.int32)

    # Unique voxel keys
    keys   = coords[:, 0] * 1_000_003 + coords[:, 1] * 1_009 + coords[:, 2]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)

    # Average within each voxel
    n_voxels = len(counts)
    result   = np.zeros((n_voxels, cloud.shape[1]), dtype=np.float32)
    np.add.at(result, inv, cloud)
    result /= counts[:, None]

    return result


# ── Grid-Adaptive Ground Removal ──────────────────────────────────────────────

def _ground_removal(
    cloud: np.ndarray,
    grid_res: float       = 0.5,
    height_margin: float  = 0.25,
    max_range: float      = 40.0,
    dilation_steps: int   = 3,
    min_pts_in_cell: int  = 1,
) -> np.ndarray:
    """
    Port of lidarGroundRemove.cpp.

    Up axis = Z (LiDAR frame).
    Grid plane is XY.
    For each grid cell find min Z, then remove points within height_margin above it.
    """
    if len(cloud) == 0:
        return cloud

    xyz = cloud[:, :3]

    # Range filter
    ranges_sq = xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2
    in_range  = ranges_sq <= (max_range ** 2)

    # Project to grid
    inv_res = 1.0 / grid_res
    ix = np.floor(xyz[:, 0] * inv_res).astype(np.int32)
    iy = np.floor(xyz[:, 1] * inv_res).astype(np.int32)
    h  = xyz[:, 2]

    # Build grid: cell → min height
    grid: dict = {}
    for i in range(len(cloud)):
        if not in_range[i]:
            continue
        key = (int(ix[i]), int(iy[i]))
        if key not in grid:
            grid[key] = {"min_h": float(h[i]), "cnt": 1}
        else:
            if h[i] < grid[key]["min_h"]:
                grid[key]["min_h"] = float(h[i])
            grid[key]["cnt"] += 1

    # Remove cells with too few points
    if min_pts_in_cell > 1:
        grid = {k: v for k, v in grid.items() if v["cnt"] >= min_pts_in_cell}

    # Dilation — propagate min height to neighbouring empty cells
    dx = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(dilation_steps):
        to_add = []
        for (gx, gy), ci in grid.items():
            for ddx, ddy in dx:
                nk = (gx + ddx, gy + ddy)
                if nk not in grid:
                    to_add.append((nk, ci["min_h"]))
        for nk, mh in to_add:
            if nk not in grid:  # first-come wins
                grid[nk] = {"min_h": mh, "cnt": 0}

    # Classify: non-ground if height > cell_min + margin
    keep = np.ones(len(cloud), dtype=bool)
    for i in range(len(cloud)):
        if not in_range[i]:
            continue  # out-of-range points are kept
        key = (int(ix[i]), int(iy[i]))
        if key not in grid:
            continue  # no ground reference → keep
        local_ground = grid[key]["min_h"]
        if float(h[i]) <= local_ground + height_margin:
            keep[i] = False

    return cloud[keep]


# ── Range-Adaptive DBSCAN ─────────────────────────────────────────────────────

def _range_adaptive_dbscan(
    cloud: np.ndarray,
    eps_near: float       = 0.20,
    eps_range_factor: float = 0.05,
    min_pts_near: int     = 5,
    min_pts_far:  int     = 2,
    far_range:    float   = 20.0,
    min_cluster_near: int = 30,
    min_cluster_far:  int = 5,
    far_cluster_range: float = 20.0,
    max_cluster_size: int = 50000,
) -> List[np.ndarray]:
    """
    Port of the range-adaptive DBSCAN from boundingBox.cpp.
    Returns list of point arrays, one per cluster.
    """
    if len(cloud) == 0:
        return []

    try:
        from sklearn.neighbors import BallTree
    except ImportError:
        logger.warning("sklearn not available — falling back to fixed-radius DBSCAN")
        return _fixed_dbscan(cloud, eps_near, min_pts_near, min_cluster_near, max_cluster_size)

    xyz    = cloud[:, :3].astype(np.float64)
    ranges = np.sqrt((xyz ** 2).sum(axis=1))

    # Per-point adaptive eps
    eps_arr = eps_near * (1.0 + eps_range_factor * ranges)

    # Per-point adaptive minPts
    t = np.clip(ranges / max(far_range, 0.1), 0.0, 1.0)
    min_pts_arr = np.round(min_pts_near * (1 - t) + min_pts_far * t).astype(int)
    min_pts_arr = np.maximum(min_pts_arr, 1)

    n = len(cloud)
    labels = np.full(n, -1, dtype=np.int32)   # -1 = unvisited
    queued = np.zeros(n, dtype=bool)
    cluster_id = 0

    # Build BallTree once (query with varying radius per point)
    tree = BallTree(xyz)

    from collections import deque
    for i in range(n):
        if labels[i] != -1:
            continue

        neighbours = tree.query_radius([xyz[i]], r=eps_arr[i])[0]
        if len(neighbours) < min_pts_arr[i]:
            labels[i] = -2   # noise
            continue

        # Start new cluster
        labels[i] = cluster_id
        cluster_pts = 1
        seeds = deque()
        for nb in neighbours:
            if nb != i and not queued[nb]:
                queued[nb] = True
                seeds.append(nb)

        while seeds:
            q = seeds.popleft()
            if labels[q] in (-2, -1):
                labels[q] = cluster_id
                cluster_pts += 1

            if labels[q] != -2:
                q_neighbours = tree.query_radius([xyz[q]], r=eps_arr[q])[0]
                if len(q_neighbours) >= min_pts_arr[q]:
                    for nb in q_neighbours:
                        if not queued[nb] and labels[nb] == -1:
                            queued[nb] = True
                            seeds.append(nb)
                        elif labels[nb] == -2:
                            labels[nb] = cluster_id
                            cluster_pts += 1

            if cluster_pts > max_cluster_size:
                break

        # Adaptive min cluster size based on seed range
        tc = min(1.0, ranges[i] / max(far_cluster_range, 0.1))
        min_size_i = max(1, int(round(min_cluster_near * (1 - tc) + min_cluster_far * tc)))

        mask = labels == cluster_id
        actual_size = mask.sum()
        if min_size_i <= actual_size <= max_cluster_size:
            cluster_id += 1
        else:
            labels[mask] = -2  # discard

    # Collect clusters
    clusters = []
    for cid in range(cluster_id):
        mask = labels == cid
        if mask.sum() > 0:
            clusters.append(cloud[mask])

    return clusters


def _fixed_dbscan(
    cloud: np.ndarray,
    eps: float,
    min_pts: int,
    min_cluster: int,
    max_cluster: int,
) -> List[np.ndarray]:
    """Fallback simple DBSCAN using sklearn if BallTree unavailable."""
    try:
        from sklearn.cluster import DBSCAN
        xyz = cloud[:, :3]
        labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(xyz)
        clusters = []
        for cid in set(labels):
            if cid < 0:
                continue
            mask = labels == cid
            size = mask.sum()
            if min_cluster <= size <= max_cluster:
                clusters.append(cloud[mask])
        return clusters
    except ImportError:
        logger.warning("sklearn unavailable — no clustering performed")
        return []


# ── AABB + LiDARDetection builder ─────────────────────────────────────────────

def _build_detection(cluster: np.ndarray) -> Optional[LiDARDetection]:
    """Build a LiDARDetection from a cluster point array."""
    if len(cluster) == 0:
        return None

    xyz = cluster[:, :3]
    mn  = xyz.min(axis=0)
    mx  = xyz.max(axis=0)
    ctr = (mn + mx) / 2.0

    x_fwd  = float(ctr[0])
    y_left = float(ctr[1])
    z_ctr  = float(ctr[2])

    rng = float(np.sqrt(x_fwd**2 + y_left**2))
    bearing = float(np.degrees(np.arctan2(y_left, x_fwd)))

    return LiDARDetection(
        centroid   = np.array([x_fwd, y_left, z_ctr], dtype=np.float32),
        bbox_min   = mn.astype(np.float32),
        bbox_max   = mx.astype(np.float32),
        range_m    = rng,
        bearing_deg= bearing,
        distance_m = x_fwd,
        lateral_m  = -y_left,  # convention: positive = right; LiDAR y=left
        height_m   = float(mx[2] - mn[2]),
        n_points   = len(cluster),
    )


# ── Box-gap merge (Union-Find) ────────────────────────────────────────────────

def _box_gap(mn_a, mx_a, mn_b, mx_b) -> float:
    dx = max(0.0, max(mn_a[0], mn_b[0]) - min(mx_a[0], mx_b[0]))
    dy = max(0.0, max(mn_a[1], mn_b[1]) - min(mx_a[1], mx_b[1]))
    dz = max(0.0, max(mn_a[2], mn_b[2]) - min(mx_a[2], mx_b[2]))
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def _merge_nearby(
    dets: List[LiDARDetection],
    merge_dist: float = 1.5,
) -> List[LiDARDetection]:
    """
    Merge detections whose AABBs are within merge_dist of each other.
    Port of the Union-Find merge in boundingBox.cpp.
    """
    if len(dets) <= 1:
        return dets

    n = len(dets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            gap = _box_gap(
                dets[i].bbox_min, dets[i].bbox_max,
                dets[j].bbox_min, dets[j].bbox_max,
            )
            if gap <= merge_dist:
                unite(i, j)

    # Group by root
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged = []
    for members in groups.values():
        all_mn = np.stack([dets[m].bbox_min for m in members]).min(axis=0)
        all_mx = np.stack([dets[m].bbox_max for m in members]).max(axis=0)
        ctr    = (all_mn + all_mx) / 2.0
        total_pts = sum(dets[m].n_points for m in members)

        x_fwd  = float(ctr[0])
        y_left = float(ctr[1])
        z_ctr  = float(ctr[2])
        rng    = float(np.sqrt(x_fwd**2 + y_left**2))

        merged.append(LiDARDetection(
            centroid    = ctr.astype(np.float32),
            bbox_min    = all_mn.astype(np.float32),
            bbox_max    = all_mx.astype(np.float32),
            range_m     = rng,
            bearing_deg = float(np.degrees(np.arctan2(y_left, x_fwd))),
            distance_m  = x_fwd,
            lateral_m   = -y_left,
            height_m    = float(all_mx[2] - all_mn[2]),
            n_points    = total_pts,
        ))

    return merged


# ── Main processor ────────────────────────────────────────────────────────────

class LiDARProcessor:
    """
    Full LiDAR processing pipeline — Python port of lidarPreProcess C++ package.

    Input:  (N, 3) or (N, 4) float32 numpy array [x, y, z, (intensity)]
            in Velodyne sensor frame: x=forward, y=left, z=up

    Output: list of LiDARDetection objects
    """

    def __init__(self, params: Optional[dict] = None):
        p = DEFAULT_PARAMS.copy()
        if params:
            p.update(params)
        self.p = p
        logger.info("LiDARProcessor ready (grid-adaptive ground removal + range-adaptive DBSCAN)")

    def process(self, cloud: np.ndarray) -> List[LiDARDetection]:
        """
        Run full pipeline.  Returns list of LiDARDetection, sorted by range.
        """
        if cloud is None or len(cloud) == 0:
            return []

        # Ensure (N,4) shape
        if cloud.shape[1] == 3:
            cloud = np.column_stack([cloud, np.zeros(len(cloud), dtype=np.float32)])

        # 1. Z passthrough
        z_mask = (cloud[:, 2] >= self.p["z_min"]) & (cloud[:, 2] <= self.p["z_max"])
        cloud  = cloud[z_mask]
        if len(cloud) == 0:
            return []

        # 2. SOR filter (removes sensor noise)
        cloud = _sor_filter(cloud, k=self.p["sor_k"], std_ratio=self.p["sor_std_ratio"])

        # 3. Voxel downsample
        cloud = _voxel_downsample(cloud, voxel_size=self.p["voxel_size"])

        # 4. Ground removal
        cloud = _ground_removal(
            cloud,
            grid_res        = self.p["grid_res"],
            height_margin   = self.p["height_margin"],
            max_range       = self.p["max_range"],
            dilation_steps  = self.p["dilation_steps"],
            min_pts_in_cell = self.p["min_pts_in_cell"],
        )
        if len(cloud) == 0:
            return []

        # 5. Clustering
        clusters = _range_adaptive_dbscan(
            cloud,
            eps_near          = self.p["eps_near"],
            eps_range_factor  = self.p["eps_range_factor"],
            min_pts_near      = self.p["min_pts_near"],
            min_pts_far       = self.p["min_pts_far"],
            far_range         = self.p["far_range"],
            min_cluster_near  = self.p["min_cluster_near"],
            min_cluster_far   = self.p["min_cluster_far"],
            far_cluster_range = self.p["far_cluster_range"],
            max_cluster_size  = self.p["max_cluster_size"],
        )

        # 6. Build detections + per-cluster filters
        raw_dets = []
        for cluster in clusters:
            det = self._filter_cluster(cluster)
            if det is not None:
                raw_dets.append(det)

        # 7. Merge nearby AABBs
        dets = _merge_nearby(raw_dets, merge_dist=self.p["merge_distance"])

        # 8. Forward-sector filter (only keep objects ahead of vehicle)
        dets = [
            d for d in dets
            if (self.p["min_forward_dist"] <= d.distance_m <= self.p["max_forward_dist"])
            and abs(d.lateral_m) <= self.p["max_lateral_dist"]
        ]

        # Sort by range
        dets.sort(key=lambda d: d.range_m)
        return dets

    def _filter_cluster(self, cluster: np.ndarray) -> Optional[LiDARDetection]:
        """Apply per-cluster filters matching BoundingBoxNode in C++."""
        xyz = cluster[:, :3]
        mn  = xyz.min(axis=0)
        mx  = xyz.max(axis=0)

        sx, sy, sz = (mx - mn)
        vol = max(float(sx * sy * sz), 1e-6)

        # a) Ground clearance — cluster must touch near the ground
        if float(mn[2]) > self.p["max_ground_clearance"]:
            return None

        # b) Size sanity
        if sx > self.p["max_box_dimension"] or sy > self.p["max_box_dimension"] or \
           sz > self.p["max_box_dimension"]:
            return None
        if vol < self.p["min_box_volume"]:
            return None

        # c) Point density — scale threshold by range
        ctr = (mn + mx) / 2.0
        cluster_range = float(np.sqrt(ctr[0]**2 + ctr[1]**2 + ctr[2]**2))
        eff_thresh = self.p["min_point_density"] / (
            1.0 + self.p["density_range_factor"] * cluster_range
        )
        density = len(cluster) / vol
        if density < eff_thresh:
            return None

        return _build_detection(cluster)

    def process_to_metric_dets(self, cloud: np.ndarray) -> list:
        """
        Process LiDAR cloud and return objects in the same format as
        MetricDetection from metric_depth.py, for easy pipeline fusion.
        Each returned dict: {distance_m, lateral_m, range_m, source='lidar'}
        """
        dets = self.process(cloud)
        return [
            {
                "distance_m": d.distance_m,
                "lateral_m":  d.lateral_m,
                "range_m":    d.range_m,
                "height_m":   d.height_m,
                "n_points":   d.n_points,
                "bearing_deg": d.bearing_deg,
                "centroid":   d.centroid,
                "source":     "lidar",
            }
            for d in dets
        ]
