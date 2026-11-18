"""
Integration tests for FastAPI ML Backend endpoints.
Verifies health, dataset listing, route reconstruction, and server-side ground truth isolation.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "PathSnap ML Backend"
    assert "available_models" in data

def test_trips_listing_endpoint():
    response = client.get("/api/trips")
    assert response.status_code == 200
    data = response.json()
    assert data["total_trips"] > 0
    trip_ids = [t["trip_id"] for t in data["trips"]]
    assert "00000000" in trip_ids

def test_predict_endpoint_and_isolation():
    req_body = {
        "model_name": "best_model",
        "use_hmm": True,
        "sigma_z": 10.0,
        "beta": 5.0
    }
    response = client.post("/api/predict/00000000", json=req_body)
    assert response.status_code == 200
    res = response.json()
    
    assert res["trip_id"] == "00000000"
    assert "reconstructed_route" in res
    assert len(res["reconstructed_route"]) > 0
    assert res["reconstructed_route_length"] > 0
    assert "route_polyline" in res
    assert "raw_gps_polyline" in res
    assert res["inference_time_ms"] > 0
    
    # CRITICAL: Verify NO ground truth references are leaked to client
    assert "ground_truth_route" not in res
    assert "gt_route" not in res
    assert "route_df" not in res

def test_evaluate_endpoint():
    response = client.get("/api/evaluate/00000000")
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "00000000"
    assert data["has_ground_truth"] is True
    assert "evaluation_metrics" in data
    em = data["evaluation_metrics"]
    assert "reconstructed_route_continuity_pct" in em
    assert "sequence_alignment_ratio_pct" in em
    assert "arc_jaccard_overlap_pct" in em
