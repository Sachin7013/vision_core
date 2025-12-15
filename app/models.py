from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel


class CameraBase(BaseModel):
    """Normalized camera fields used inside Vision backend and for responses."""

    user_id: str
    camera_id: str
    camera_name: str
    rtsp_url: str
    device_id: str | None = None


class CameraCreate(BaseModel):
    """Payload shape coming from Samit's backend.

    Example JSON:
    {
      "id": "CAM-43C1E6AFB726",         # camera_id
      "owner_user_id": "6928...",       # user_id
      "name": "home_camera",            # camera_name
      "stream_url": "rtsp://...",       # rtsp_url
      "device_id": "DEV-790EECBE4CF2"   # optional extra metadata
    }
    """

    id: str
    owner_user_id: str
    name: str
    stream_url: str
    device_id: str | None = None


class CameraOut(CameraBase):
    created_at: datetime


class WebRTCConfig(BaseModel):
    """Configuration returned to the frontend for WebRTC setup.

    Contains the full signaling URL (including viewer client ID) and
    the list of STUN/TURN ICE servers the browser should use.
    """

    signaling_url: str
    ice_servers: List[Dict[str, Any]]
