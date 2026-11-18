"""
PathSnap — Map Visualization Module
Generates interactive multi-layer Folium maps comparing:
1. Raw GPS trajectory (Blue circles / markers)
2. Ground-truth reference route (Green polyline)
3. Classifier-only Top-1 prediction (Orange dashed polyline)
4. HMM Viterbi reconstructed route (Red solid polyline)
Supports evaluation mode to suppress ground truth display.
"""

import os
import folium
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

from src.data_loader import TripData

def extract_arc_coordinates(arc_ids: List[int], nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> List[List[float]]:
    """
    Extracts ordered [lat, lon] coordinate path corresponding to a sequence of arc IDs.
    """
    coords = []
    # Fast lookup arrays
    node_lats = nodes_df['lat'].values
    node_lons = nodes_df['lon'].values
    arc_u = arcs_df['from_node_id'].values
    arc_v = arcs_df['to_node_id'].values
    
    last_node = None
    for arc_id in arc_ids:
        if arc_id < 0 or arc_id >= len(arc_u):
            continue
        u = int(arc_u[arc_id])
        v = int(arc_v[arc_id])
        
        u_lat, u_lon = float(node_lats[u]), float(node_lons[u])
        v_lat, v_lon = float(node_lats[v]), float(node_lons[v])
        
        if last_node != u:
            coords.append([u_lat, u_lon])
        coords.append([v_lat, v_lon])
        last_node = v
        
    return coords

def build_folium_route_map(
    trip_data: TripData,
    gt_route_arcs: Optional[List[int]] = None,
    classifier_arcs: Optional[List[int]] = None,
    hmm_reconstructed_arcs: Optional[List[int]] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    show_ground_truth: bool = True
) -> folium.Map:
    """
    Builds a multi-layer Folium map for a trip.
    """
    track_df = trip_data.track_df
    nodes_df = trip_data.nodes_df
    arcs_df = trip_data.arcs_df
    
    center_lat = float(track_df['lat'].mean())
    center_lon = float(track_df['lon'].mean())
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='cartodbpositron',
        control_scale=True
    )
    
    # Layer 1: GPS Observations (Blue Dots)
    gps_group = folium.FeatureGroup(name="GPS Observations (Raw Track)", show=True)
    gps_coords = track_df[['lat', 'lon']].values
    
    # Add polyline connecting raw GPS
    folium.PolyLine(
        locations=gps_coords.tolist(),
        color="#2980B9",
        weight=2,
        opacity=0.4,
        dash_array="4, 6"
    ).add_to(gps_group)
    
    # Add markers for start and end
    if len(gps_coords) > 0:
        folium.Marker(
            location=gps_coords[0].tolist(),
            popup="GPS Start",
            icon=folium.Icon(color="blue", icon="play")
        ).add_to(gps_group)
        folium.Marker(
            location=gps_coords[-1].tolist(),
            popup="GPS End",
            icon=folium.Icon(color="darkblue", icon="stop")
        ).add_to(gps_group)
        
    # Sample GPS circle markers if trajectory is large
    step = max(1, len(gps_coords) // 200)
    for idx in range(0, len(gps_coords), step):
        lat, lon = gps_coords[idx]
        t = track_df['timestamp'].iloc[idx] if 'timestamp' in track_df else idx
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2980B9",
            fill=True,
            fill_color="#3498DB",
            fill_opacity=0.7,
            popup=f"Obs #{idx}<br>Time: {t}s<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        ).add_to(gps_group)
        
    gps_group.add_to(m)
    
    # Layer 2: Ground Truth Route (Green Solid) - Only shown if permitted
    if show_ground_truth and (gt_route_arcs is not None or trip_data.route_df is not None):
        ref_arcs = gt_route_arcs if gt_route_arcs is not None else trip_data.route_df['arc_id'].tolist()
        gt_coords = extract_arc_coordinates(ref_arcs, nodes_df, arcs_df)
        
        if gt_coords:
            gt_group = folium.FeatureGroup(name="Ground Truth Route (Reference)", show=True)
            folium.PolyLine(
                locations=gt_coords,
                color="#27AE60",
                weight=6,
                opacity=0.75,
                tooltip=f"Ground Truth Route ({len(ref_arcs)} arcs)"
            ).add_to(gt_group)
            gt_group.add_to(m)
            
    # Layer 3: Classifier-Only Prediction (Orange Dashed)
    if classifier_arcs is not None and len(classifier_arcs) > 0:
        clf_coords = extract_arc_coordinates(classifier_arcs, nodes_df, arcs_df)
        if clf_coords:
            clf_group = folium.FeatureGroup(name="Classifier Only (Top-1 Prediction)", show=False)
            folium.PolyLine(
                locations=clf_coords,
                color="#E67E22",
                weight=4,
                opacity=0.8,
                dash_array="6, 6",
                tooltip=f"Classifier Only ({len(classifier_arcs)} arcs)"
            ).add_to(clf_group)
            clf_group.add_to(m)
            
    # Layer 4: HMM Reconstructed Route (Red Solid)
    if hmm_reconstructed_arcs is not None and len(hmm_reconstructed_arcs) > 0:
        hmm_coords = extract_arc_coordinates(hmm_reconstructed_arcs, nodes_df, arcs_df)
        if hmm_coords:
            hmm_group = folium.FeatureGroup(name="PathSnap HMM Reconstructed Route", show=True)
            folium.PolyLine(
                locations=hmm_coords,
                color="#E74C3C",
                weight=5,
                opacity=0.9,
                tooltip=f"HMM Reconstructed ({len(hmm_reconstructed_arcs)} arcs)"
            ).add_to(hmm_group)
            hmm_group.add_to(m)

    # Layer Control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Title / Metrics Box in HTML
    if metrics_summary:
        box_html = f"""
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 280px; height: auto; 
                    background-color: white; border:2px solid #2C3E50; z-index:9999; font-size:12px;
                    padding: 10px; border-radius: 6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <b style="font-size:13px; color:#2C3E50;">Trip {trip_data.trip_id} Summary</b><hr style="margin:5px 0;">
            <b>Observations:</b> {len(track_df)}<br>
            <b>Top-1 Accuracy:</b> {metrics_summary.get('top1_accuracy', 'N/A')}<br>
            <b>Route Continuity:</b> {metrics_summary.get('continuity', 'N/A')}<br>
            <b>HMM Arc Count:</b> {len(hmm_reconstructed_arcs) if hmm_reconstructed_arcs else 'N/A'}<br>
            <b>Reference Arc Count:</b> {len(gt_route_arcs) if gt_route_arcs else (len(trip_data.route_df) if trip_data.route_df is not None else 'N/A')}<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(box_html))
        
    return m
