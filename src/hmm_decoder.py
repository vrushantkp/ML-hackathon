"""
PathSnap — Hidden Markov Model (HMM) Viterbi Map-Matching & Route Reconstruction
Implements spatial-temporal emission modeling and graph-topological transition modeling
with dynamic programming Viterbi decoding and continuous shortest-path route assembly.
"""

import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

def build_road_graph(arcs_df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[Tuple[int, int], int]]:
    """
    Builds a directed NetworkX graph from arcs_df.
    Nodes are node_ids, edges have attributes 'arc_id' and 'weight' (length).
    Also returns a mapping from (from_node, to_node) -> arc_id.
    """
    G = nx.DiGraph()
    edge_to_arc = {}
    
    for arc_id, row in arcs_df.iterrows():
        u = int(row['from_node_id'])
        v = int(row['to_node_id'])
        # Arc length if present, else fallback to 10.0
        length = float(row['length']) if 'length' in row else 10.0
        
        G.add_edge(u, v, arc_id=int(arc_id), weight=max(length, 0.1))
        edge_to_arc[(u, v)] = int(arc_id)
        
    return G, edge_to_arc

class RoadMatchHMM:
    """
    Hidden Markov Model map-matching decoder.
    Hidden States: Candidate directed road segments for each GPS observation.
    Observations: 2D projected metric coordinates and timestamps.
    """
    def __init__(
        self,
        G: nx.DiGraph,
        arcs_df: pd.DataFrame,
        sigma_z: float = 10.0,
        beta: float = 5.0,
        disconnect_penalty: float = -35.0
    ):
        self.G = G
        self.arcs_df = arcs_df
        self.sigma_z = sigma_z
        self.beta = beta
        self.disconnect_penalty = disconnect_penalty
        self.log_beta = math.log(max(beta, 1e-4))
        
        # Precompute arc lookup arrays
        self.arc_u = arcs_df['from_node_id'].values.astype(int)
        self.arc_v = arcs_df['to_node_id'].values.astype(int)
        if 'length' in arcs_df.columns:
            self.arc_len = arcs_df['length'].values.astype(float)
        else:
            self.arc_len = np.full(len(arcs_df), 10.0)
            
        self.path_cache: Dict[Tuple[int, int], Optional[float]] = {}
        self.path_arc_cache: Dict[Tuple[int, int], Optional[List[int]]] = {}

    def get_shortest_path_dist(self, u: int, v: int, cutoff: float = 200.0) -> Optional[float]:
        if u == v:
            return 0.0
        pair = (u, v)
        if pair in self.path_cache:
            return self.path_cache[pair]
            
        try:
            # single_source_dijkstra strictly limits search to cutoff meters
            dist, _ = nx.single_source_dijkstra(self.G, u, target=v, cutoff=cutoff, weight='weight')
            self.path_cache[pair] = dist
            return dist
        except (KeyError, nx.NetworkXNoPath, nx.NodeNotFound):
            self.path_cache[pair] = None
            return None

    def get_shortest_path_arcs(self, u: int, v: int, cutoff: float = 300.0) -> Optional[List[int]]:
        """
        Returns list of arc_ids along the shortest path from u to v.
        """
        if u == v:
            return []
        pair = (u, v)
        if pair in self.path_arc_cache:
            return self.path_arc_cache[pair]
            
        try:
            _, node_path = nx.single_source_dijkstra(self.G, u, target=v, cutoff=cutoff, weight='weight')
            arc_path = []
            for i in range(len(node_path) - 1):
                nu, nv = node_path[i], node_path[i+1]
                edge_data = self.G.get_edge_data(nu, nv)
                if edge_data and 'arc_id' in edge_data:
                    arc_path.append(int(edge_data['arc_id']))
                else:
                    return None
            self.path_arc_cache[pair] = arc_path
            return arc_path
        except (KeyError, nx.NetworkXNoPath, nx.NodeNotFound):
            self.path_arc_cache[pair] = None
            return None

    def compute_emission_log_prob(self, cand: Dict[str, Any]) -> float:
        """
        Computes emission log-probability combining ML classifier confidence and GPS measurement error.
        ln P(z_t | c_i) = ln(max(P_ML, 1e-4)) - (dist^2) / (2 * sigma_z^2)
        """
        prob_ml = float(cand.get('prob_pos', 0.5))
        dist = float(cand.get('perp_distance', 25.0))
        
        log_ml = math.log(max(prob_ml, 1e-4))
        log_spatial = - (dist ** 2) / (2.0 * (self.sigma_z ** 2))
        return log_ml + log_spatial

    def compute_transition_log_prob(
        self,
        c_prev: Dict[str, Any],
        c_curr: Dict[str, Any],
        d_euclid: float,
        dt: float
    ) -> float:
        """
        Computes transition log-probability based on road network routing plausibility.
        """
        arc_prev = int(c_prev['arc_id'])
        arc_curr = int(c_curr['arc_id'])
        
        # Case 1: Vehicle stays on the same road segment
        if arc_prev == arc_curr:
            proj_prev = float(c_prev.get('projection_ratio', 0.0))
            proj_curr = float(c_curr.get('projection_ratio', 0.0))
            arc_l = self.arc_len[arc_prev]
            d_net = abs(proj_curr - proj_prev) * arc_l
            diff = abs(d_net - d_euclid)
            # Small forward progress bonus
            bonus = 0.5 if proj_curr >= proj_prev else -1.0
            return - self.log_beta - (diff / self.beta) + bonus

        # Retrieve source and target nodes
        v_prev = self.arc_v[arc_prev]
        u_curr = self.arc_u[arc_curr]
        
        proj_prev = float(c_prev.get('projection_ratio', 0.5))
        proj_curr = float(c_curr.get('projection_ratio', 0.5))
        l_prev = self.arc_len[arc_prev]
        l_curr = self.arc_len[arc_curr]
        
        rem_prev = max(0.0, (1.0 - proj_prev) * l_prev)
        ent_curr = max(0.0, proj_curr * l_curr)

        # Case 2: Direct topological successor (end of arc_prev connects directly to start of arc_curr)
        if v_prev == u_curr:
            d_net = rem_prev + ent_curr
            diff = abs(d_net - d_euclid)
            # Continuity bonus for direct topological connection
            return - self.log_beta - (diff / self.beta) + 1.5

        # Case 3: Transition across intermediate graph nodes
        # Fast local urban search capped at 200m
        cutoff = min(200.0, max(50.0, d_euclid * 2.0 + 30.0))
        d_path = self.get_shortest_path_dist(v_prev, u_curr, cutoff=cutoff)
        
        if d_path is not None:
            d_net = rem_prev + d_path + ent_curr
            diff = abs(d_net - d_euclid)
            return - self.log_beta - (diff / self.beta)
        else:
            # Topological disconnection / unreachable
            return self.disconnect_penalty

    def decode(
        self,
        candidates_by_obs: List[List[Dict[str, Any]]],
        track_xy: np.ndarray,
        timestamps: np.ndarray
    ) -> List[int]:
        """
        Performs Viterbi decoding over candidate sequences.
        Returns:
            List of chosen arc_ids, one per observation.
        """
        T = len(candidates_by_obs)
        if T == 0:
            return []
            
        # Step 1: Initialize Viterbi table and backpointers
        V = []
        backpointers = []
        
        first_cands = candidates_by_obs[0]
        K0 = len(first_cands)
        v0 = np.zeros(K0)
        for i, c in enumerate(first_cands):
            v0[i] = self.compute_emission_log_prob(c)
        V.append(v0)
        backpointers.append(np.zeros(K0, dtype=int))
        
        # Step 2: Forward recurrence
        for t in range(1, T):
            prev_cands = candidates_by_obs[t - 1]
            curr_cands = candidates_by_obs[t]
            
            K_prev = len(prev_cands)
            K_curr = len(curr_cands)
            
            vt = np.full(K_curr, -np.inf)
            bp_t = np.zeros(K_curr, dtype=int)
            
            d_euclid = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))
            dt = float(timestamps[t] - timestamps[t - 1])
            
            for j, c_j in enumerate(curr_cands):
                emit_log_p = self.compute_emission_log_prob(c_j)
                
                best_score = -np.inf
                best_i = 0
                
                for i, c_i in enumerate(prev_cands):
                    trans_log_p = self.compute_transition_log_prob(c_i, c_j, d_euclid, dt)
                    score = V[t - 1][i] + trans_log_p + emit_log_p
                    if score > best_score:
                        best_score = score
                        best_i = i
                        
                vt[j] = best_score
                bp_t[j] = best_i
                
            V.append(vt)
            backpointers.append(bp_t)
            
        # Step 3: Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        chosen_indices = [best_last_idx]
        
        for t in range(T - 1, 0, -1):
            prev_idx = backpointers[t][chosen_indices[-1]]
            chosen_indices.append(int(prev_idx))
            
        chosen_indices.reverse()
        
        # Extract arc IDs
        chosen_arcs = []
        for t, idx in enumerate(chosen_indices):
            chosen_arcs.append(int(candidates_by_obs[t][idx]['arc_id']))
            
        return chosen_arcs

    def reconstruct_continuous_route(self, chosen_arc_ids: List[int]) -> List[int]:
        """
        Converts observation-level decoded arc IDs into a topologically continuous
        reference route sequence by removing duplicates and splicing missing intermediate arcs.
        """
        if not chosen_arc_ids:
            return []
            
        # Deduplicate consecutive identical arcs
        compact_arcs = [chosen_arc_ids[0]]
        for a in chosen_arc_ids[1:]:
            if a != compact_arcs[-1]:
                compact_arcs.append(a)
                
        # Splice gaps
        reconstructed = []
        for i in range(len(compact_arcs)):
            arc_curr = compact_arcs[i]
            reconstructed.append(arc_curr)
            
            if i < len(compact_arcs) - 1:
                arc_next = compact_arcs[i + 1]
                v_curr = self.arc_v[arc_curr]
                u_next = self.arc_u[arc_next]
                
                # If disconnected, attempt to find shortest path
                if v_curr != u_next:
                    intermediate_arcs = self.get_shortest_path_arcs(v_curr, u_next, cutoff=400.0)
                    if intermediate_arcs:
                        reconstructed.extend(intermediate_arcs)
                        
        return reconstructed
