"""
Unit tests for HMM Viterbi decoding and continuous route reconstruction.
"""

import numpy as np
import pandas as pd
import networkx as nx
import pytest

from src.hmm_decoder import build_road_graph, RoadMatchHMM
from src.metrics import compute_trip_route_continuity

def test_hmm_viterbi_and_continuity():
    # Construct a simple synthetic road graph
    # Node 0 -> Node 1 (arc 0, len 10)
    # Node 1 -> Node 2 (arc 1, len 10)
    # Node 2 -> Node 3 (arc 2, len 10)
    # Node 3 -> Node 4 (arc 3, len 10)
    arcs_df = pd.DataFrame([
        {'from_node_id': 0, 'to_node_id': 1, 'length': 10.0},
        {'from_node_id': 1, 'to_node_id': 2, 'length': 10.0},
        {'from_node_id': 2, 'to_node_id': 3, 'length': 10.0},
        {'from_node_id': 3, 'to_node_id': 4, 'length': 10.0},
    ])
    
    G, edge_to_arc = build_road_graph(arcs_df)
    hmm = RoadMatchHMM(G, arcs_df, sigma_z=10.0, beta=5.0)
    
    # Check shortest path
    path_len = hmm.get_shortest_path_dist(0, 3)
    assert abs(path_len - 30.0) < 1e-4
    
    path_arcs = hmm.get_shortest_path_arcs(0, 3)
    assert path_arcs == [0, 1, 2]
    
    # Test route reconstruction with gap:
    # Say Viterbi chose arc 0, then skipped to arc 3
    chosen_arcs = [0, 3]
    full_route = hmm.reconstruct_continuous_route(chosen_arcs)
    # Intermediate arcs 1 and 2 should be spliced
    assert full_route == [0, 1, 2, 3]
    cont = compute_trip_route_continuity(full_route, arcs_df)
    assert cont == 100.0
