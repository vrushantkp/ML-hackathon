"""
Unit tests for feature engineering, causality checks, and angular differences.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import compute_circular_heading_difference, FEATURE_COLUMNS
from src.data_loader import load_trip
from src.candidate_generator import generate_candidates_for_trip
from src.feature_engineering import extract_features_for_trip

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

def test_circular_heading_difference():
    # 0 deg and 10 deg -> 10 deg
    diff1 = compute_circular_heading_difference(0.0, 10.0)
    assert abs(diff1 - 10.0) < 1e-4

    # 355 deg and 5 deg -> 10 deg (with circular wraparound)
    diff2 = compute_circular_heading_difference(355.0, 5.0)
    assert abs(diff2 - 10.0) < 1e-4

    # 10 deg and 350 deg -> 20 deg
    diff3 = compute_circular_heading_difference(10.0, 350.0)
    assert abs(diff3 - 20.0) < 1e-4

    # 90 deg and 270 deg -> 180 deg
    diff4 = compute_circular_heading_difference(90.0, 270.0)
    assert abs(diff4 - 180.0) < 1e-4

def test_feature_extraction_completeness_and_causality():
    trip = load_trip("00000000", RAW_DIR, partition="student_test")
    # Subsample track to 20 points for fast unit test
    sub_track = trip.track_df.iloc[:20].copy()
    trip.track_df = sub_track
    trip.track_xy = trip.track_xy[:20]
    
    cands_df = generate_candidates_for_trip(trip, initial_radius=50.0)
    features_df = extract_features_for_trip(trip, cands_df)
    
    # 1. Verify all 11 feature columns are present
    for col in FEATURE_COLUMNS:
        assert col in features_df.columns
        assert not features_df[col].isna().any(), f"NaN found in feature {col}"
        
    # 2. Verify causality: non-negative speeds and positive dt
    first_obs_cands = features_df[features_df['obs_idx'] == 0]
    assert (first_obs_cands['vehicle_speed'] >= 0.0).all()
    assert (first_obs_cands['displacement_m'] >= 0.0).all()
    assert (first_obs_cands['dt_s'] > 0.0).all()
    assert (first_obs_cands['dt_s'] == 1.0).all()
