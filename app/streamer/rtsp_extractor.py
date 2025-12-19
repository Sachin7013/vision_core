"""
RTSP Extraction Module - Handles RTSP camera stream extraction and frame validation.
Uses MediaPlayer from aiortc to connect to RTSP sources.
"""
import asyncio
import time
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame

from app.shared_hub.hub import SharedFrameHub
from app.db import get_agents_collection

# Small cache to avoid pinging DB on every frame just to know if any agents
# are running for a camera. The scheduler updates status every few seconds,
# so a short TTL here is fine.
_agent_presence_cache = {}
_PRESENCE_TTL_SEC = 1.0


async def check_player_frames(player, label, timeout=3.0):
    """
    Try to receive a single frame from player.video to ensure the RTSP source is healthy.
    
    Args:
        player: MediaPlayer instance
        label: Camera label for logging
        timeout: Maximum time to wait for a frame (seconds)
        
    Returns:
        bool: True if frame was successfully received, False otherwise
    """
    if not getattr(player, "video", None):
        print(f"[debug] {label}: No video attribute on player")
        return False
    try:
        frame = await asyncio.wait_for(player.video.recv(), timeout=timeout)
        if frame is None:
            print(f"[debug] {label}: recv returned None")
            return False
        print(f"[debug] {label}: got frame pts={getattr(frame, 'pts', '?')} size={getattr(frame,'width','?')}x{getattr(frame,'height','?')}")
        return True
    except asyncio.TimeoutError:
        print(f"[debug] {label}: recv() timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[debug] {label}: recv() exception: {e}")
        return False


async def create_rtsp_player(rtsp_url, label):
    """
    Create a MediaPlayer for the given RTSP URL and validate the connection.
    
    Args:
        rtsp_url: RTSP URL of the camera stream
        label: Camera label/ID for logging
        
    Returns:
        tuple: (label, player, is_healthy)
               - label: Camera label
               - player: MediaPlayer instance (None if creation failed)
               - is_healthy: Boolean indicating if frames were successfully received
    """
    try:
        print(f"[pusher] Creating MediaPlayer for {label}: {rtsp_url}")
        player = MediaPlayer(
            rtsp_url,
            format="rtsp",
            options={"rtsp_transport": "tcp", "stimeout": "5000000"}
        )
        await asyncio.sleep(0.5)  # allow ffmpeg to spin up
        
        ok = await check_player_frames(player, label, timeout=3.0)
        if not ok:
            # try one more short retry
            print(f"[pusher] {label}: retrying frame check")
            await asyncio.sleep(1.0)
            ok = await check_player_frames(player, label, timeout=3.0)
        
        if not ok:
            print(f"[pusher] ⚠️ {label}: no frames detected (RTSP may be wrong or camera offline)")
        else:
            print(f"[pusher] ✅ {label}: frames detected")
        
        return (label, player, ok)
    except Exception as e:
        print(f"[pusher] ❌ Error creating player for {label}: {e}")
        return (label, None, False)


def has_running_agents_for_camera(camera_id: str) -> bool:
    """Return True if there is at least one running agent for this camera."""
    now = time.time()
    cached = _agent_presence_cache.get(camera_id)
    if cached and (now - cached[0] < _PRESENCE_TTL_SEC):
        return cached[1]
    try:
        coll = get_agents_collection()
        # Quick existence check; limit=1 to keep it lightweight
        has_any = coll.count_documents({"camera_id": camera_id, "status": "running"}, limit=1) > 0
        _agent_presence_cache[camera_id] = (now, has_any)
        return has_any
    except Exception:
        _agent_presence_cache[camera_id] = (now, False)
        return False


def fanout_frame(camera_id: str, frame: VideoFrame) -> VideoFrame:
    """
    Publish every camera frame to the shared hub so agents (when running)
    can consume the latest frames immediately. Also return the original
    frame unchanged for the live path.
    """
    try:
        SharedFrameHub.instance().publish(camera_id, frame)
    finally:
        return frame


def subscribe_to_camera(camera_id: str):
    """Subscribe to shared frames for a camera; returns an asyncio.Queue."""
    return SharedFrameHub.instance().subscribe(camera_id)


def unsubscribe_from_camera(camera_id: str, q) -> None:
    """Unsubscribe a previously created queue from the camera channel."""
    SharedFrameHub.instance().unsubscribe(camera_id, q)
