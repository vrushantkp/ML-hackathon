"""
PathSnap — Exploratory Data Analysis (EDA) Script
Audits and analyzes all 80 train trips and 10 student_test trips.
Computes dataset inventory, GPS statistics, graph topology, gap analysis, and Folium maps.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

def analyze_trip(trip_id, trip_dir, partition):
    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")
    
    files_exist = {
        'track': os.path.exists(track_file),
        'nodes': os.path.exists(nodes_file),
        'arcs': os.path.exists(arcs_file),
        'route': os.path.exists(route_file)
    }
    file_sizes = {
        'track_bytes': os.path.getsize(track_file) if files_exist['track'] else 0,
        'nodes_bytes': os.path.getsize(nodes_file) if files_exist['nodes'] else 0,
        'arcs_bytes': os.path.getsize(arcs_file) if files_exist['arcs'] else 0,
        'route_bytes': os.path.getsize(route_file) if files_exist['route'] else 0,
    }
    
    track_df = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    n_points = len(track_df)
    
    nan_count = track_df.isna().sum().sum()
    valid_coords = (
        (track_df['lon'] >= -180) & (track_df['lon'] <= 180) &
        (track_df['lat'] >= -90) & (track_df['lat'] <= 90)
    ).all()
    
    times = track_df['time'].values
    time_diffs = np.diff(times)
    is_monotonic = bool((time_diffs > 0).all())
    median_dt = float(np.median(time_diffs)) if len(time_diffs) > 0 else 0.0
    pct_1s = float((np.abs(time_diffs - 1.0) < 0.01).mean() * 100) if len(time_diffs) > 0 else 0.0
    gaps_gt_2s = int((time_diffs > 2.0).sum()) if len(time_diffs) > 0 else 0
    gaps_gt_5s = int((time_diffs > 5.0).sum()) if len(time_diffs) > 0 else 0
    max_gap_s = float(np.max(time_diffs)) if len(time_diffs) > 0 else 0.0
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    
    mean_lat = track_df['lat'].mean()
    scale_x = 111320.0 * np.cos(np.radians(mean_lat))
    scale_y = 110540.0
    
    track_x = track_df['lon'].values * scale_x
    track_y = track_df['lat'].values * scale_y
    step_dists = np.sqrt(np.diff(track_x)**2 + np.diff(track_y)**2)
    total_track_length_m = float(np.sum(step_dists)) if len(step_dists) > 0 else 0.0
    avg_speed_mps = float(total_track_length_m / duration_s) if duration_s > 0 else 0.0
    
    nodes_df = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs_df = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['u', 'v'])
    n_nodes = len(nodes_df)
    n_arcs = len(arcs_df)
    
    isolated_nodes = n_nodes - len(set(arcs_df['u']).union(set(arcs_df['v'])))
    
    route_df = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'])
    n_route_arcs = len(route_df)
    unique_route_arcs = int(route_df['arc_id'].nunique())
    
    route_arc_ids = route_df['arc_id'].values
    valid_arc_ids = bool((route_arc_ids >= 0).all() and (route_arc_ids < n_arcs).all())
    
    route_connected_count = 0
    if valid_arc_ids and n_route_arcs > 1:
        u_seq = arcs_df.iloc[route_arc_ids]['u'].values
        v_seq = arcs_df.iloc[route_arc_ids]['v'].values
        route_connected_count = int((v_seq[:-1] == u_seq[1:]).sum())
        route_continuity_pct = float(route_connected_count / (n_route_arcs - 1) * 100)
    else:
        route_continuity_pct = 100.0 if n_route_arcs <= 1 else 0.0
        
    return {
        'trip_id': trip_id,
        'partition': partition,
        'n_gps_points': n_points,
        'duration_s': duration_s,
        'nan_count': nan_count,
        'valid_coords': valid_coords,
        'is_monotonic': is_monotonic,
        'median_dt': median_dt,
        'pct_1s_sampling': pct_1s,
        'gaps_gt_2s': gaps_gt_2s,
        'gaps_gt_5s': gaps_gt_5s,
        'max_gap_s': max_gap_s,
        'track_length_km': round(total_track_length_m / 1000.0, 3),
        'avg_speed_kmh': round(avg_speed_mps * 3.6, 2),
        'min_lon': round(float(track_df['lon'].min()), 6),
        'max_lon': round(float(track_df['lon'].max()), 6),
        'min_lat': round(float(track_df['lat'].min()), 6),
        'max_lat': round(float(track_df['lat'].max()), 6),
        'n_nodes': n_nodes,
        'n_arcs': n_arcs,
        'isolated_nodes': isolated_nodes,
        'n_route_arcs': n_route_arcs,
        'unique_route_arcs': unique_route_arcs,
        'route_continuity_pct': round(route_continuity_pct, 2),
        'track_size_kb': round(file_sizes['track_bytes'] / 1024.0, 1),
        'nodes_size_kb': round(file_sizes['nodes_bytes'] / 1024.0, 1),
        'arcs_size_kb': round(file_sizes['arcs_bytes'] / 1024.0, 1),
        'route_size_kb': round(file_sizes['route_bytes'] / 1024.0, 1)
    }

def generate_eda_map(trip_id, trip_dir, out_html):
    """Generates a fast, interactive Folium map for a trip."""
    track_df = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes_df = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs_df = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
    route_df = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.route"), sep=r'\s+', header=None, names=['arc_id'])
    
    node_coords = nodes_df[['lat', 'lon']].values
    
    center_lat = float(track_df['lat'].mean())
    center_lon = float(track_df['lon'].mean())
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
    
    # 1. Nearby road network: filter arcs near the track
    lat_pad = 0.005
    lon_pad = 0.005
    min_lat, max_lat = track_df['lat'].min() - lat_pad, track_df['lat'].max() + lat_pad
    min_lon, max_lon = track_df['lon'].min() - lon_pad, track_df['lon'].max() + lon_pad
    
    u_vals = arcs_df['u'].values
    v_vals = arcs_df['v'].values
    u_lats = node_coords[u_vals, 0]
    u_lons = node_coords[u_vals, 1]
    v_lats = node_coords[v_vals, 0]
    v_lons = node_coords[v_vals, 1]
    
    nearby_mask = (
        (u_lats >= min_lat) & (u_lats <= max_lat) & (u_lons >= min_lon) & (u_lons <= max_lon) &
        (v_lats >= min_lat) & (v_lats <= max_lat) & (v_lons >= min_lon) & (v_lons <= max_lon)
    )
    
    nearby_indices = np.where(nearby_mask)[0]
    if len(nearby_indices) > 600:
        nearby_indices = nearby_indices[::2]
        
    road_lines = [
        [[float(u_lats[i]), float(u_lons[i])], [float(v_lats[i]), float(v_lons[i])]]
        for i in nearby_indices
    ]
    
    fg_roads = folium.FeatureGroup(name="Nearby Road Network (Grey)", show=True)
    folium.PolyLine(road_lines, color="#888888", weight=1.5, opacity=0.5).add_to(fg_roads)
    fg_roads.add_to(m)
    
    # 2. Ground truth route (Green)
    route_arcs = route_df['arc_id'].values
    ru_vals = u_vals[route_arcs]
    rv_vals = v_vals[route_arcs]
    
    route_segments = [
        [[float(node_coords[ru, 0]), float(node_coords[ru, 1])], [float(node_coords[rv, 0]), float(node_coords[rv, 1])]]
        for ru, rv in zip(ru_vals, rv_vals)
    ]
    
    fg_route = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
    folium.PolyLine(route_segments, color="#2ECC71", weight=4.5, opacity=0.9, tooltip=f"Trip {trip_id} Reference Route").add_to(fg_route)
    fg_route.add_to(m)
    
    # 3. Raw GPS Points (Blue)
    fg_gps = folium.FeatureGroup(name="Raw Noisy GPS Points (Blue)", show=True)
    step = max(1, len(track_df) // 400)
    for _, row in track_df.iloc[::step].iterrows():
        folium.CircleMarker(
            location=[float(row['lat']), float(row['lon'])],
            radius=2.5,
            color="#2980B9",
            fill=True,
            fill_color="#3498DB",
            fill_opacity=0.7,
            popup=f"t={row['time']}s"
        ).add_to(fg_gps)
    fg_gps.add_to(m)
    
    # Markers
    folium.Marker([float(track_df.iloc[0]['lat']), float(track_df.iloc[0]['lon'])], popup="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(track_df.iloc[-1]['lat']), float(track_df.iloc[-1]['lon'])], popup="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"Saved EDA map to {out_html}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    plots_dir = os.path.join(base_dir, "artifacts", "plots")
    results_dir = os.path.join(base_dir, "artifacts", "results")
    maps_dir = os.path.join(base_dir, "artifacts", "maps")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    
    train_dir = os.path.join(raw_dir, "train")
    test_dir = os.path.join(raw_dir, "student_test")
    
    csv_path = os.path.join(results_dir, "dataset_summary.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing summary from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        records = []
        print("Auditing 80 training trips...")
        train_trips = sorted(os.listdir(train_dir))
        for tid in train_trips:
            tdir = os.path.join(train_dir, tid)
            records.append(analyze_trip(tid, tdir, "train"))
            
        print("Auditing 10 student testing trips...")
        test_trips = sorted(os.listdir(test_dir))
        for tid in test_trips:
            tdir = os.path.join(test_dir, tid)
            records.append(analyze_trip(tid, tdir, "student_test"))
            
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"Wrote dataset summary to {csv_path}")
    
    # Generate Plots
    sns.set_theme(style="whitegrid")
    
    # 1. GPS points distribution per trip
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="n_gps_points", hue="partition", bins=20, kde=True, palette="Set1")
    plt.title("Distribution of GPS Observations per Trip (PathSnap)")
    plt.xlabel("Number of GPS Observations")
    plt.ylabel("Trip Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_gps_observations_dist.png"), dpi=200)
    plt.close()
    
    # 2. Sampling interval & gap analysis
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="partition", y="pct_1s_sampling", hue="partition", palette="Pastel1", legend=False)
    plt.title("Percentage of 1-second Sampling per Trip")
    plt.ylabel("% of Exactly 1.0s dt Steps")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_sampling_pct1s.png"), dpi=200)
    plt.close()
    
    # 3. Trajectory Length vs Route Arcs Count
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="track_length_km", y="n_route_arcs", hue="partition", s=70, palette="Dark2")
    plt.title("Physical Trajectory Length (km) vs Reference Route Arcs")
    plt.xlabel("Trajectory Length (km)")
    plt.ylabel("Number of Road Arcs in Route")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_track_length_vs_arcs.png"), dpi=200)
    plt.close()
    
    # 4. Gaps distribution
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="trip_id", y="gaps_gt_2s", hue="partition")
    plt.xticks([])
    plt.title("Irregular GPS Gaps (dt > 2.0s) per Trip Across Partitions")
    plt.xlabel("Trips (Sorted)")
    plt.ylabel("Count of Gaps > 2s")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_gps_gaps.png"), dpi=200)
    plt.close()
    
    # Generate sample Folium map for trip 00000001 and trip 00000000
    generate_eda_map("00000001", os.path.join(train_dir, "00000001"), os.path.join(maps_dir, "eda_trip_00000001.html"))
    generate_eda_map("00000000", os.path.join(test_dir, "00000000"), os.path.join(maps_dir, "eda_trip_00000000.html"))
    
    summary_stats = {
        'total_trips': len(df),
        'train_trips': int((df['partition'] == 'train').sum()),
        'student_test_trips': int((df['partition'] == 'student_test').sum()),
        'total_gps_points': int(df['n_gps_points'].sum()),
        'avg_gps_points_per_trip': float(df['n_gps_points'].mean()),
        'min_gps_points': int(df['n_gps_points'].min()),
        'max_gps_points': int(df['n_gps_points'].max()),
        'total_trajectory_km': float(df['track_length_km'].sum()),
        'avg_trajectory_km': float(df['track_length_km'].mean()),
        'overall_pct_1s_sampling': float(df['pct_1s_sampling'].mean()),
        'trips_with_gaps_gt_5s': int((df['gaps_gt_5s'] > 0).sum()),
        'avg_route_arcs': float(df['n_route_arcs'].mean()),
        'overall_route_continuity_pct': float(df['route_continuity_pct'].mean()),
        'all_coords_valid': bool(df['valid_coords'].all()),
        'all_timestamps_monotonic': bool(df['is_monotonic'].all()),
        'nan_observations': int(df['nan_count'].sum())
    }
    
    with open(os.path.join(results_dir, "eda_metrics.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)
    print("Summary statistics written to eda_metrics.json:")
    print(json.dumps(summary_stats, indent=2))

if __name__ == "__main__":
    main()
