"""
Unit tests for data loader and geographic projection.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_trip, project_coordinates

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

def test_load_existing_trip():
    trip = load_trip("00000000", RAW_DIR, partition="student_test")
    assert trip.trip_id == "00000000"
    assert trip.partition == "student_test"
    assert len(trip.track_df) > 0
    assert len(trip.nodes_df) > 0
    assert len(trip.arcs_df) > 0
    assert trip.route_df is not None
    assert len(trip.route_df) > 0

def test_coordinate_projection_properties():
    trip = load_trip("00000000", RAW_DIR, partition="student_test")
    # Coordinates in meters must be positive and reasonable
    assert trip.track_xy.shape == (len(trip.track_df), 2)
    assert trip.nodes_xy.shape == (len(trip.nodes_df), 2)
    # Check that scale factors match Germany (lat ~ 49.85)
    assert 60000 < trip.scale_x < 80000
    assert 110000 < trip.scale_y < 111500

def test_load_nonexistent_trip():
    with pytest.raises(FileNotFoundError):
        load_trip("99999999", RAW_DIR, partition="student_test")
