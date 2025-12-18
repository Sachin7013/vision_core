"""
DATA MODELS - WHAT IS DATA?
============================
Models are like templates that define what information we store.

Think of it like:
- Model = A form with fields to fill
- When we create a camera, we fill the camera form
- When we create an agent, we fill the agent form
"""

from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


# ===========================
# CAMERA MODELS - Store Camera Information
# ===========================

class CameraBase(BaseModel):
    """
    Basic camera information.
    
    Fields:
    - user_id: Which user owns this camera
    - camera_id: Unique name for this camera (like "cam01")
    - camera_name: Human-readable name (like "Front Door")
    - rtsp_url: Where to get the video stream (like "rtsp://192.168.1.100:554/stream")
    - device_id: Optional hardware ID
    """
    user_id: str
    camera_id: str
    camera_name: str
    rtsp_url: str
    device_id: str | None = None


class CameraCreate(BaseModel):
    """
    Form for creating a new camera.
    
    This is what we receive from the frontend/API.
    Field names might be different from CameraBase (that's OK, we convert them).
    """
    id: str                    # Camera ID
    owner_user_id: str        # User ID
    name: str                 # Camera name
    stream_url: str           # RTSP URL
    device_id: str | None = None


class CameraOut(CameraBase):
    """Camera data returned to user (includes when it was created)."""
    created_at: datetime


# ===========================
# WEBRTC CONFIG - For Video Streaming
# ===========================

class WebRTCConfig(BaseModel):
    """
    Configuration for video streaming to browser.
    
    Fields:
    - signaling_url: Server URL for handshake
    - ice_servers: Servers to help with network connection
    """
    signaling_url: str
    ice_servers: List[Dict[str, Any]]


# ===========================
# AGENT MODELS - Store Automation Rules
# ===========================

class AgentRule(BaseModel):
    """
    A single rule/condition for an agent.
    
    Example rule: "If you detect a PERSON, alert me"
    
    Fields:
    - type: Rule type (e.g., "class_presence" = object exists)
    - target_class: What to look for (e.g., "person")
    - label: Human name for this rule
    - min_count: Minimum number needed (e.g., "at least 2 people")
    - action: What to do when rule matches (e.g., "send alert")
    """
    type: str
    target_class: str = Field(alias="class")  # Maps "class" from API to "target_class"
    label: str
    min_count: int | None = None
    action: str | None = None

    class Config:
        allow_population_by_field_name = True


class AgentCreate(BaseModel):
    """
    Form for creating a new automation agent.
    
    An agent = a set of rules that run on a camera
    
    Fields:
    - agent_id: Unique ID for this agent
    - task_name: What is this agent for? (e.g., "Person Detection")
    - task_type: Type of task (e.g., "object_detection")
    - camera_id: Which camera to monitor
    - model_ids: Which AI models to use (e.g., ["yolov8n.pt"])
    - fps: How many frames per second to check (e.g., 5 = check 5 times/second)
    - rules: List of rules to check (e.g., [detect person, detect car, etc])
    - start_at: When to activate this agent
    - end_at: When to stop this agent
    - status: Current state (pending, running, terminated)
    
    SIMPLE EXAMPLE:
    {
        "agent_id": "agent_001",
        "task_name": "Person Detection",
        "camera_id": "cam01",
        "model_ids": ["yolov8n.pt"],
        "fps": 5,
        "start_at": "2025-12-18T09:00:00",
        "end_at": "2025-12-18T17:00:00",
        "rules": [
            {
                "type": "class_presence",
                "class": "person",
                "label": "Found a person"
            }
        ]
    }
    """
    agent_id: str
    task_name: str
    task_type: str
    camera_id: str
    source_uri: str
    model_ids: List[str]
    fps: int = 5  # Default: check 5 frames per second
    run_mode: str
    rules: List[AgentRule] = []
    status: str = "pending"  # Will change to "running" when time arrives
    start_at: datetime
    end_at: datetime
    created_at: datetime | None = None


class AgentOut(AgentCreate):
    """Agent data returned to user (includes database ID)."""
    id: str  # MongoDB ID
