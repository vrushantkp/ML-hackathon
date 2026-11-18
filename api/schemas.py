"""
PathSnap — API Pydantic Schemas
Defines request and response data models for the ML backend service.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "PathSnap ML Backend"
    version: str = "1.0.0"
    device: str = "Laptop 1 (ML Engine)"
    default_model: str
    available_models: List[str]
    system_time: str

class TripMetadata(BaseModel):
    trip_id: str
    partition: str
    observation_count: int
    duration_seconds: float
    start_timestamp: float
    end_timestamp: float
    mean_sampling_interval_s: float
    bounding_box: Dict[str, float]
    has_ground_truth: bool

class TripsListResponse(BaseModel):
    total_trips: int
    trips: List[TripMetadata]

class PredictionRequest(BaseModel):
    model_name: Optional[str] = "histgradientboosting"
    use_hmm: bool = True

class ReconstructedPoint(BaseModel):
    obs_idx: int
    timestamp: float
    lat: float
    lon: float
    chosen_arc_id: int
    confidence: float

class PredictionResponse(BaseModel):
    trip_id: str
    partition: str
    model_used: str
    use_hmm: bool
    observation_count: int
    candidate_count: int
    reconstructed_route: List[int]       # Final sequence of arc IDs
    reconstructed_route_length: int
    classifier_top1_route: List[int]     # Top-1 classifier predicted arc IDs
    trajectory_points: List[ReconstructedPoint]
    route_polyline: List[List[float]]    # [lat, lon] sequence for map display
    raw_gps_polyline: List[List[float]]  # [lat, lon] sequence of GPS
    ground_truth_polyline: Optional[List[List[float]]] = []     # Ground-truth reference polyline
    classifier_top1_polyline: Optional[List[List[float]]] = []  # Raw classifier Top-1 polyline
    misaligned_segments: Optional[List[List[List[float]]]] = []  # Route segments where model deviated from ground truth
    missed_gt_segments: Optional[List[List[List[float]]]] = []   # Ground truth segments missed by model
    misaligned_arc_count: Optional[int] = 0
    misalignment_rate_pct: Optional[float] = 0.0
    top1_accuracy_pct: Optional[float] = None
    hmm_observation_accuracy_pct: Optional[float] = None
    hmm_config_used: Optional[Dict[str, Any]] = None
    inference_time_ms: float
    reconstruction_continuity_pct: float

class EvaluationResponse(BaseModel):
    trip_id: str
    partition: str
    has_ground_truth: bool
    evaluation_metrics: Optional[Dict[str, Any]] = None
    message: str
