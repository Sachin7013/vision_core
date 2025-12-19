from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from av import VideoFrame
from ultralytics import YOLO

from app.models import AgentRule
from app.object_detection_part.load_model import _get_or_load_model
from app.object_detection_part.object_detection import (
    annotate_frame_with_detections,
    run_detection,
)
from app.rule_engine import rule as rules_module

@dataclass
class AgentRuntime:
    """In-memory helper representing a running agent.

    For now we keep this very small: just the model, list of rules and
    target FPS. This structure is designed so it can later be connected
    to the live RTSP frames inside ``sender_stream.py``.
    """

    agent_id: str
    camera_id: str
    model: YOLO
    rules: List[AgentRule]
    fps: int

def build_agent_runtime_from_doc(agent_doc: Dict[str, Any]) -> Optional[AgentRuntime]:
    """Build a single AgentRuntime from a MongoDB agent document.

    Returns None if the document is incomplete (e.g., missing model_ids).
    """
    camera_id = agent_doc.get("camera_id")
    agent_id = agent_doc.get("agent_id")
    model_ids = agent_doc.get("model_ids") or []
    if not camera_id or not agent_id or not model_ids:
        return None
    model_id = model_ids[0]
    model = _get_or_load_model(model_id)

    rules_data = agent_doc.get("rules") or []
    rules: List[AgentRule] = []
    for rd in rules_data:
        rules.append(AgentRule.parse_obj(rd))

    fps = int(agent_doc.get("fps", 5))
    return AgentRuntime(agent_id=agent_id, camera_id=camera_id, model=model, rules=rules, fps=fps)


def run_rules_for_agent(
    runtime: AgentRuntime,
    detections: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Apply all rules for a single agent to a detection list.

    Rule implementations live in ``rules.py``. We dynamically dispatch
    based on ``rule.type`` so new rules can be added without changing
    this engine module.
    """
    any_match = False
    kept: List[Dict[str, Any]] = []

    for rule in runtime.rules:
        # Look up a handler function on rules_module whose name matches
        # the rule.type value (e.g. "class_presence", "class_count").
        handler = getattr(rules_module, rule.type, None)
        if handler is None:
            print("error: Unknown rule type: " + rule.type)
            # Unknown rule type; skip silently for now.
            continue

        matched, filtered = handler(rule, detections)
        if matched:
            any_match = True
            kept.extend(filtered)
            print(
                f"[agent:{runtime.agent_id}] ALERT: rule '{rule.label}' matched "
                f"for class '{rule.target_class}'"
            )

    return any_match, kept


def process_frame_for_agent(runtime: AgentRuntime, frame: VideoFrame) -> VideoFrame:
    """Process a single frame for exactly one agent.

    Unlike process_frame_for_camera (which aggregates across multiple agents),
    this function only runs detection and rules for the provided AgentRuntime.
    It annotates the frame only with detections that matched the agent's rules.
    """
    try:
        bgr = frame.to_ndarray(format="bgr24")
    except Exception as exc:
        print(f"[engine] Failed to convert frame for agent {runtime.agent_id}: {exc}")
        return frame
    if bgr is None or bgr.size == 0:
        return frame

    dets = run_detection(runtime.model, bgr)
    matched, filtered = run_rules_for_agent(runtime, dets)
    if not matched or not filtered:
        return frame

    annotated = annotate_frame_with_detections(bgr, filtered)
    new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
    new_frame.pts = frame.pts
    new_frame.time_base = frame.time_base
    return new_frame
