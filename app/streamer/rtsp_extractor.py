"""
RTSP Extraction Module - Handles RTSP camera stream extraction and frame validation.
Uses MediaPlayer from aiortc to connect to RTSP sources.
"""
import asyncio
from aiortc.contrib.media import MediaPlayer


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
