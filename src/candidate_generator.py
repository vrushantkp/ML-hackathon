"""
PathSnap — Candidate Generator Module
Builds spatial index (STRtree) over directed road network segments.
Performs adaptive spatial candidate search (50m default, expandable to 75m/100m/150m).
Calculates exact point-to-segment perpendicular distance, projection ratio, bearing, and length.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from shapely import distance as shapely_distance

class CandidateGenerator:
    def __init__(self, nodes_xy: np.ndarray, arcs_df: pd.DataFrame, initial_radius: float = 50.0, max_radius: float = 150.0, max_candidates: int = 15):
        """
        Initializes STRtree spatial index over all directed road network segments.
        nodes_xy: Shape (M, 2) in local metric meters.
        arcs_df: Columns ['u', 'v'].
        """
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.max_candidates = max_candidates
        self.nodes_xy = nodes_xy
        self.arcs_df = arcs_df
        
        # Build LineStrings in metric coordinates
        u_idx = arcs_df['u'].values
        v_idx = arcs_df['v'].values
        
        self.u_coords = nodes_xy[u_idx]  # Shape (N_arcs, 2)
        self.v_coords = nodes_xy[v_idx]  # Shape (N_arcs, 2)
        
        # Precompute segment vectors, lengths, bearings
        delta = self.v_coords - self.u_coords
        self.arc_lengths = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        # Avoid division by zero for degenerate segments
        self.arc_lengths_safe = np.maximum(self.arc_lengths, 1e-4)
        
        # Bearing in degrees: 0=North, 90=East, 180=South, 270=West
        # delta[:, 0] is dx (East), delta[:, 1] is dy (North)
        self.arc_bearings = (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360.0) % 360.0
        
        # Vectorized LineString creation
        # Create Shapely LineStrings from coordinate pairs
        coords = np.stack([self.u_coords, self.v_coords], axis=1)  # (N_arcs, 2, 2)
        self.lines = [LineString(c) for c in coords]
        self.tree = STRtree(self.lines)
        
    def generate_candidates_for_points(self, track_xy: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Finds candidate road arcs for an array of GPS points (N, 2).
        Returns a DataFrame of candidate records and candidate generation statistics.
        """
        n_points = len(track_xy)
        pts = [Point(p[0], p[1]) for p in track_xy]
        
        # Query 1: Initial search radius (50m)
        point_indices, arc_indices = self.tree.query(pts, predicate='dwithin', distance=self.initial_radius)
        
        # Check which points have 0 candidates
        points_with_cands = set(point_indices)
        all_points = set(range(n_points))
        missing_points = list(all_points - points_with_cands)
        
        expanded_records = []
        radius_expansion_count = 0
        
        # Adaptive radius expansion for missing points
        if missing_points:
            for pt_idx in missing_points:
                pt_geom = pts[pt_idx]
                found = False
                for r in [75.0, 100.0, self.max_radius]:
                    radius_expansion_count += 1
                    sub_pts, sub_arcs = self.tree.query([pt_geom], predicate='dwithin', distance=r)
                    if len(sub_arcs) > 0:
                        for a_idx in sub_arcs:
                            expanded_records.append((pt_idx, a_idx))
                        found = True
                        break
                if not found:
                    # Fallback to nearest arc if still empty
                    nearest_arc = self.tree.nearest(pt_geom)
                    expanded_records.append((pt_idx, nearest_arc))
                    
        # Combine initial and expanded candidates
        if expanded_records:
            exp_p, exp_a = zip(*expanded_records)
            all_pt_indices = np.concatenate([point_indices, np.array(exp_p, dtype=np.int64)])
            all_arc_indices = np.concatenate([arc_indices, np.array(exp_a, dtype=np.int64)])
        else:
            all_pt_indices = point_indices
            all_arc_indices = arc_indices
            
        # Compute exact point-to-segment geometry and features
        p_coords = track_xy[all_pt_indices]
        a_coords = self.u_coords[all_arc_indices]
        b_coords = self.v_coords[all_arc_indices]
        
        ab = b_coords - a_coords
        ap = p_coords - a_coords
        
        ab_len_sq = self.arc_lengths_safe[all_arc_indices] ** 2
        t = np.clip(np.sum(ap * ab, axis=-1) / ab_len_sq, 0.0, 1.0)
        proj = a_coords + t[:, None] * ab
        
        perp_dists = np.sqrt(np.sum((p_coords - proj) ** 2, axis=-1))
        
        cand_df = pd.DataFrame({
            'obs_idx': all_pt_indices,
            'arc_id': all_arc_indices,
            'perp_distance': perp_dists,
            'projection_ratio': t,
            'arc_length': self.arc_lengths[all_arc_indices],
            'arc_bearing': self.arc_bearings[all_arc_indices],
            'u': self.arcs_df.iloc[all_arc_indices]['u'].values,
            'v': self.arcs_df.iloc[all_arc_indices]['v'].values
        })
        
        # Sort by (obs_idx, perp_distance) and prune to top max_candidates per observation
        cand_df = cand_df.sort_values(['obs_idx', 'perp_distance'])
        cand_df['candidate_rank'] = cand_df.groupby('obs_idx').cumcount() + 1
        cand_df = cand_df[cand_df['candidate_rank'] <= self.max_candidates].copy()
        
        stats = {
            'n_points': n_points,
            'total_candidates_generated': len(cand_df),
            'avg_candidates_per_obs': float(len(cand_df) / n_points),
            'radius_expansions_performed': radius_expansion_count,
            'obs_with_expanded_radius': len(missing_points),
            'min_candidates': int(cand_df.groupby('obs_idx').size().min()),
            'max_candidates': int(cand_df.groupby('obs_idx').size().max())
        }
        
        return cand_df, stats

def generate_candidates_for_trip(
    trip_data: Any,
    initial_radius: float = 50.0,
    max_radius: float = 150.0,
    max_candidates: int = 15
) -> pd.DataFrame:
    """
    Convenience function to generate candidates directly from a TripData object.
    """
    generator = CandidateGenerator(
        nodes_xy=trip_data.nodes_xy,
        arcs_df=trip_data.arcs_df,
        initial_radius=initial_radius,
        max_radius=max_radius,
        max_candidates=max_candidates
    )
    cand_df, _ = generator.generate_candidates_for_points(trip_data.track_xy)
    return cand_df

