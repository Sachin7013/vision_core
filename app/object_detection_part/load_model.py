from ultralytics import YOLO
from typing import Dict

_model_cache: Dict[str, YOLO] = {}


def _get_or_load_model(model_id: str) -> YOLO:
    """Load a YOLO model by ``model_id`` using a simple in-memory cache."""
    if model_id in _model_cache:
        return _model_cache[model_id]
    model = YOLO(model_id)
    _model_cache[model_id] = model
    return model