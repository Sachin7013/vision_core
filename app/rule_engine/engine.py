from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import time

from av import VideoFrame
from ultralytics import YOLO

from app.db import get_agents_collection
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


def build_agent_runtimes_for_camera(camera_id: str) -> List[AgentRuntime]:
    """Build AgentRuntime objects for all *running* agents on a camera.

    This is a synchronous helper intended to be called from the
    streaming process (e.g. inside ``sender_stream.py``). It looks up
    active agents in MongoDB and loads their requested models.
    """
    coll = get_agents_collection()
    docs = list(coll.find({"camera_id": camera_id, "status": "running"}))
    runtimes: List[AgentRuntime] = []
    for doc in docs:
        agent_id = doc.get("agent_id")
        model_ids = doc.get("model_ids") or []
        if not model_ids:
            continue
        # For now we take the first model_id only.
        model_id = model_ids[0]
        model = _get_or_load_model(model_id)

        rules_data = doc.get("rules") or []
        rules: List[AgentRule] = []
        for rd in rules_data:
            # ``AgentRule`` knows how to map the incoming ``class`` key
            # into ``target_class``.
            rules.append(AgentRule.parse_obj(rd))

        fps = int(doc.get("fps", 5))
        runtimes.append(AgentRuntime(agent_id=agent_id, camera_id=camera_id, model=model, rules=rules, fps=fps))
    return runtimes


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


# Per-camera cache for the last detection timestamp so we can throttle
# how often YOLO is run based on the agents' configured FPS.
_camera_last_detection_time: Dict[str, float] = {}


# ==================================================================================
# FRAME PROCESSING - ENGINE HANDLES DETECTION AND RULE APPLICATION
# ==================================================================================
# The engine receives active agents from agent_scheduler and processes frames.
# Agent triggering is ONLY done by the scheduler - this module handles the
# detection and rule logic once an agent has been activated.


def process_frame_for_camera(camera_id: str, frame: VideoFrame) -> VideoFrame:
    """Main entry point for frame processing.

    This is the PRIMARY ENTRY POINT used by sender_stream.py to process each
    video frame. The flow is:

    1. Get running agents from scheduler (ONLY active agents returned)
    2. Run detection and rules for each agent
    3. Annotate frame with matched detections
    4. Return processed frame

    CRITICAL: Agent triggering/identification is handled entirely by
    agent_scheduler.py. This module only processes frames for agents
    that are already active (status='running').

    Args:
        camera_id: The camera identifier
        frame: The incoming VideoFrame to process

    Returns:
        VideoFrame with annotated detections (or original if no matches)

    FLOW DIAGRAM:
        sender_stream.py
            ↓
        process_frame_for_camera(camera_id, frame)
            ↓
        get_running_agents_for_camera() [from scheduler]
            ↓
        For each running agent:
            - Run YOLO detection
            - Apply agent's rules
            - Collect matched detections
            ↓
        Annotate frame with matched detections
            ↓
        Return frame to sender_stream.py
    """

    # =========================================================================
    # STEP 1: GET ACTIVE AGENTS
    # =========================================================================
    # IMPORTANT: This calls agent_scheduler.get_running_agents_for_camera()
    # which ONLY returns agents with status='running'. The scheduler ensures
    # the database is always up-to-date, so we just query it here.
    #
    # This is the ONLY place where agents are fetched - no loops elsewhere.

    from app.agent_scheduler import get_running_agents_for_camera

    agents = get_running_agents_for_camera(camera_id)
    if not agents:
        # No active agents on this camera - return frame unchanged
        # Log this occasionally to debug (not every frame to avoid spam)
        if time.time() % 30 < 0.1:  # Log once every ~30 seconds
            print(f"[engine] ℹ️ No running agents for camera '{camera_id}'")
        return frame
    
    print(f"[engine] 🎯 Found {len(agents)} running agent(s) for camera '{camera_id}'")
    for agent in agents:
        print(f"[engine]    - Agent: {agent.agent_id} (FPS: {agent.fps})")

    # =========================================================================
    # STEP 2: THROTTLE DETECTION TO CONFIGURED FPS
    # =========================================================================
    # We don't run YOLO on every frame. Instead, we calculate the maximum FPS
    # across all agents and throttle accordingly. This reduces compute load.

    max_fps = max((rt.fps for rt in agents if rt.fps > 0), default=1)
    min_interval = 1.0 / max_fps
    now = time.time()
    last_det = _camera_last_detection_time.get(camera_id, 0.0)
    if (now - last_det) < min_interval:
        # Not enough time has passed - skip detection for this frame
        return frame
    _camera_last_detection_time[camera_id] = now

    # =========================================================================
    # STEP 3: CONVERT FRAME TO NUMPY ARRAY
    # =========================================================================
    # YOLO requires NumPy BGR format for detection

    try:
        bgr = frame.to_ndarray(format="bgr24")
    except Exception as exc:
        print(f"[engine] Failed to convert frame for detection on {camera_id}: {exc}")
        return frame
    if bgr is None or bgr.size == 0:
        return frame

    # =========================================================================
    # STEP 4: RUN DETECTION AND RULES FOR EACH AGENT
    # =========================================================================
    # For each running agent, we:
    # 1. Run YOLO detection on the frame
    # 2. Apply the agent's rules to filter detections
    # 3. Keep detections that matched at least one rule
    #
    # This is where the actual processing happens. Each agent independently
    # detects objects and applies its own rules.

    rule_detections: List[Dict[str, Any]] = []
    for rt in agents:
        # Run YOLO detection
        dets = run_detection(rt.model, bgr)

        # Apply this agent's rules to filter detections
        matched, filtered = run_rules_for_agent(rt, dets)
        if matched and filtered:
            # At least one rule matched - keep these detections
            rule_detections.extend(filtered)
            print(
                f"[engine] Agent '{rt.agent_id}' on camera '{camera_id}': "
                f"Detection processed, {len(filtered)} matched detections"
            )

    # =========================================================================
    # STEP 5: ANNOTATE FRAME AND RETURN
    # =========================================================================
    # If nothing matched any rules, return the original frame.
    # Otherwise, draw the detections that matched rules.

    if not rule_detections:
        # No detections matched any rules - return original frame
        return frame

    # Draw only detections that matched rules
    annotated = annotate_frame_with_detections(bgr, rule_detections)
    new_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
    new_frame.pts = frame.pts
    new_frame.time_base = frame.time_base
    return new_frame
