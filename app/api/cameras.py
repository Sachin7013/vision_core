from datetime import datetime
from typing import List

import os

from fastapi import APIRouter, HTTPException

from app.db import get_cameras_collection
from app.models import CameraCreate, CameraOut, WebRTCConfig

router = APIRouter(prefix="/api", tags=["cameras"])


@router.post("/cameras", response_model=CameraOut)
def add_camera(camera: CameraCreate) -> CameraOut:
    """Register or update a camera sent from Samit's backend.

    Samit's JSON payload:
    {
      "id": "CAM-43C1E6AFB726",          # camera_id
      "owner_user_id": "6928...",        # user_id
      "name": "home_camera",             # camera_name
      "stream_url": "rtsp://...",        # rtsp_url
      "device_id": "DEV-790EECBE4CF2"    # optional
    }

    We normalize this into our internal schema:
    {
      "user_id": ..., "camera_id": ..., "camera_name": ..., "rtsp_url": ...,
      "device_id": ..., "created_at": ...
    }
    """
    coll = get_cameras_collection()

    doc = {
        "user_id": camera.owner_user_id,
        "camera_id": camera.id,
        "camera_name": camera.name,
        "rtsp_url": camera.stream_url,
        "device_id": camera.device_id,
        "created_at": datetime.utcnow(),
    }

    coll.update_one(
        {"user_id": doc["user_id"], "camera_id": doc["camera_id"]},
        {"$set": doc},
        upsert=True,
    )

    saved = coll.find_one(
        {"user_id": doc["user_id"], "camera_id": doc["camera_id"]},
        {"_id": 0},
    )
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save camera")

    return CameraOut(**saved)


@router.get("/cameras", response_model=List[CameraOut])
def list_cameras(user_id: str) -> List[CameraOut]:
    """List all cameras belonging to a specific user."""
    coll = get_cameras_collection()
    docs = coll.find({"user_id": user_id}, {"_id": 0}).sort("camera_id")
    return [CameraOut(**d) for d in docs]


@router.get("/webrtc-config", response_model=WebRTCConfig)
def get_webrtc_config(user_id: str) -> WebRTCConfig:
    """Return signaling URL and ICE servers for a given user.

    The frontend calls this *after* the user is logged in on Samit's side.
    The `user_id` must match the `owner_user_id` used when cameras were added.

    This endpoint does **not** expose raw env secrets in code; it only
    reads TURN/STUN configuration from environment variables at runtime
    and returns the minimal data the browser needs (`iceServers`).
    """

    coll = get_cameras_collection()
    has_camera = coll.find_one({"user_id": user_id})
    if not has_camera:
        raise HTTPException(status_code=404, detail="No cameras found for this user_id in Vision database")

    signaling_ws = os.getenv("SIGNALING_WS")
    if not signaling_ws:
        raise HTTPException(status_code=500, detail="SIGNALING_WS environment variable is not configured on Vision backend")

    signaling_url = signaling_ws.rstrip("/") + f"/viewer:{user_id}"

    ice_servers = [
        {"urls": "stun:stun.l.google.com:19302"},
    ]

    aws_turn_ip = os.getenv("AWS_TURN_IP")
    aws_turn_port = os.getenv("AWS_TURN_PORT")
    aws_turn_user = os.getenv("AWS_TURN_USER")
    aws_turn_pass = os.getenv("AWS_TURN_PASS")

    if aws_turn_ip and aws_turn_port and aws_turn_user and aws_turn_pass:
        ice_servers.append(
            {
                "urls": [
                    f"turn:{aws_turn_ip}:{aws_turn_port}?transport=udp",
                    f"turn:{aws_turn_ip}:{aws_turn_port}?transport=tcp",
                ],
                "username": aws_turn_user,
                "credential": aws_turn_pass,
            }
        )

    return WebRTCConfig(signaling_url=signaling_url, ice_servers=ice_servers)
