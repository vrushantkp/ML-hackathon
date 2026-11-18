"""
PathSnap — Data Loader Module
Handles robust loading, validation, and geographic projection for raw trip files.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

@dataclass
class TripData:
    trip_id: str
    partition: str
    track_df: pd.DataFrame
    nodes_df: pd.DataFrame
    arcs_df: pd.DataFrame
    route_df: Optional[pd.DataFrame]
    track_xy: np.ndarray      # Shape (N, 2) in meters
    nodes_xy: np.ndarray      # Shape (M, 2) in meters
    mean_lat: float
    scale_x: float
    scale_y: float

def project_coordinates(lon: np.ndarray, lat: np.ndarray, mean_lat: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Projects WGS84 angular coordinates (lon, lat in degrees) to local metric coordinates (x, y in meters)
    using Equirectangular projection with standard parallel at mean_lat.
    Appropriate for city/regional scale map matching without spatial distortion.
    """
    if mean_lat is None:
        mean_lat = float(np.mean(lat))
    scale_x = 111320.0 * np.cos(np.radians(mean_lat))
    scale_y = 110540.0
    x = lon * scale_x
    y = lat * scale_y
    return x, y, mean_lat, scale_x, scale_y

def load_trip(trip_id: Union[str, int], base_raw_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"), partition: Optional[str] = None) -> TripData:
    """
    Loads and validates all trip files (.track, .nodes, .arcs, .route).
    Searches in 'train', 'student_test', or custom directory if partition is None.
    """
    trip_id_str = str(trip_id).zfill(8)
    trip_dir = None
    resolved_partition = partition
    
    if partition:
        candidate_dir = os.path.join(base_raw_dir, partition, trip_id_str)
        if os.path.exists(candidate_dir):
            trip_dir = candidate_dir
    else:
        for p in ['train', 'student_test', 'evaluator_test']:
            candidate_dir = os.path.join(base_raw_dir, p, trip_id_str)
            if os.path.exists(candidate_dir):
                trip_dir = candidate_dir
                resolved_partition = p
                break
                
    if trip_dir is None:
        raise FileNotFoundError(f"Trip {trip_id} not found in {base_raw_dir} (searched train, student_test).")
        
    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")
    
    if not os.path.exists(track_file):
        raise FileNotFoundError(f"Missing track file: {track_file}")
    if not os.path.exists(nodes_file):
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")
    if not os.path.exists(arcs_file):
        raise FileNotFoundError(f"Missing arcs file: {arcs_file}")
        
    # Read files with strict whitespace parsing
    track_df = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'], dtype={'lon': np.float64, 'lat': np.float64, 'time': np.float64})
    nodes_df = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat'], dtype={'lon': np.float64, 'lat': np.float64})
    arcs_df = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['u', 'v'], dtype={'u': np.int64, 'v': np.int64})
    
    route_df = None
    if os.path.exists(route_file):
        route_df = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'], dtype={'arc_id': np.int64})
        
    # Data validation
    assert not track_df.isna().any().any(), f"Trip {trip_id}: track contains NaNs"
    assert (track_df['lon'].between(-180, 180)).all(), f"Trip {trip_id}: invalid longitude"
    assert (track_df['lat'].between(-90, 90)).all(), f"Trip {trip_id}: invalid latitude"
    assert (np.diff(track_df['time']) >= 0).all(), f"Trip {trip_id}: timestamps non-monotonic"
    
    n_nodes = len(nodes_df)
    assert (arcs_df['u'] >= 0).all() and (arcs_df['u'] < n_nodes).all(), f"Trip {trip_id}: arc 'u' node out of range"
    assert (arcs_df['v'] >= 0).all() and (arcs_df['v'] < n_nodes).all(), f"Trip {trip_id}: arc 'v' node out of range"
    
    if route_df is not None:
        n_arcs = len(arcs_df)
        assert (route_df['arc_id'] >= 0).all() and (route_df['arc_id'] < n_arcs).all(), f"Trip {trip_id}: route arc_id out of range"
        
    # Coordinate Projection
    mean_lat = float(track_df['lat'].mean())
    tx, ty, _, scale_x, scale_y = project_coordinates(track_df['lon'].values, track_df['lat'].values, mean_lat)
    nx, ny, _, _, _ = project_coordinates(nodes_df['lon'].values, nodes_df['lat'].values, mean_lat)
    
    track_xy = np.column_stack([tx, ty])
    nodes_xy = np.column_stack([nx, ny])
    
    # Aliases for consistent naming across modules
    track_df['timestamp'] = track_df['time']
    arcs_df['from_node_id'] = arcs_df['u']
    arcs_df['to_node_id'] = arcs_df['v']
    
    u_pts = nodes_xy[arcs_df['u'].values]
    v_pts = nodes_xy[arcs_df['v'].values]
    arcs_df['length'] = np.sqrt(np.sum((v_pts - u_pts)**2, axis=1))
    
    return TripData(
        trip_id=trip_id,
        partition=resolved_partition or "unknown",
        track_df=track_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        route_df=route_df,
        track_xy=track_xy,
        nodes_xy=nodes_xy,
        mean_lat=mean_lat,
        scale_x=scale_x,
        scale_y=scale_y
    )
