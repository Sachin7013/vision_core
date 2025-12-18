"""
WebRTC Pusher - Streams multiple RTSP cameras with YOLOv8 pose detection
Sends video to a remote viewer via WebRTC with TURN relay support
"""
import asyncio
import json
import os
import time
from dotenv import load_dotenv
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    VideoStreamTrack,
)
from aiortc.contrib.signaling import candidate_from_sdp
import websockets

from app.db import get_cameras_collection
from app.rule_engine.engine import process_frame_for_camera
from app.streamer.rtsp_extractor import create_rtsp_player, check_player_frames

# Load environment variables from .env file
load_dotenv()

# ============ Configuration from .env file ============
SIGNALING_WS = os.getenv("SIGNALING_WS")  # WebSocket server URL for signaling
if not SIGNALING_WS:
    raise RuntimeError("SIGNALING_WS environment variable is required for live-sender")

# ============ TURN Server Configuration (for NAT traversal) ============
AWS_TURN_IP = os.getenv("AWS_TURN_IP")
AWS_TURN_PORT = os.getenv("AWS_TURN_PORT")
AWS_TURN_USER = os.getenv("AWS_TURN_USER")
AWS_TURN_PASS = os.getenv("AWS_TURN_PASS")

# ============ ICE Servers Setup ============
# Start with Google's public STUN server
ICE_SERVERS = [RTCIceServer(urls="stun:stun.l.google.com:19302")]

# Add TURN server if credentials are provided (for better NAT traversal)
if AWS_TURN_IP and AWS_TURN_PORT and AWS_TURN_USER and AWS_TURN_PASS:
    ICE_SERVERS += [
        RTCIceServer(
            urls=f"turn:{AWS_TURN_IP}:{AWS_TURN_PORT}?transport=udp",
            username=AWS_TURN_USER,
            credential=AWS_TURN_PASS,
        ),
        RTCIceServer(
            urls=f"turn:{AWS_TURN_IP}:{AWS_TURN_PORT}?transport=tcp",
            username=AWS_TURN_USER,
            credential=AWS_TURN_PASS,
        ),
    ]


class ProxyVideoTrack(VideoStreamTrack):
    """Wrapper track that forwards frames from RTSP source to WebRTC.

    The ``camera_id`` is used as both a human-readable label and the
    WebRTC track ID so that the viewer can map incoming tracks back to
    the correct camera.
    """
    def __init__(self, source_track, camera_id):
        super().__init__()
        self.source = source_track
        self.label = camera_id
        # Use camera_id as the ID to ensure uniqueness across tracks
        self._id = camera_id
        print("========trackid========", camera_id)

    @property
    def id(self):
        """Return unique track ID"""
        return self._id

    @property
    def kind(self):
        """Return track kind (always 'video')"""
        return getattr(self.source, "kind", "video")

    async def recv(self):
        """Receive frame from source and hand it off to the rule engine.

        The rule engine decides whether to run object detection, apply
        any active agent rules, and annotate the frame. If there are no
        active agents for this camera, or no rules match, the original
        frame is returned unchanged.
        """
        frame = await self.source.recv()
        return process_frame_for_camera(self.label, frame)


async def run_single_session():
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ICE_SERVERS))
    print(f"[pusher] PeerConnection created. TURN: {AWS_TURN_IP}:{AWS_TURN_PORT if AWS_TURN_IP else ''}")

    @pc.on("iceconnectionstatechange")
    def on_ice_state():
        print("[pusher] ICE state:", pc.iceConnectionState)

    @pc.on("connectionstatechange")
    def on_conn_state():
        print("[pusher] Connection state:", pc.connectionState)
        if pc.connectionState == "failed":
            print("[pusher] ⚠️ Connection failed, will attempt to recover")
        elif pc.connectionState == "disconnected":
            print("[pusher] ⚠️ Connection disconnected, waiting for reconnection")

    @pc.on("icegatheringstatechange")
    def on_gather_state():
        print("[pusher] ICE gathering state:", pc.iceGatheringState)

    players = []

    cameras_coll = get_cameras_collection()
    distinct_user_ids = cameras_coll.distinct("user_id")
    if not distinct_user_ids:
        print("[pusher] ❌ No users/cameras found in database. Please add a camera first.")
        await pc.close()
        return

    if len(distinct_user_ids) > 1:
        print(f"[pusher] ❌ Multiple user_ids found in database: {distinct_user_ids}. This Jetson instance is expected to serve exactly one user.")
        await pc.close()
        return

    user_id = distinct_user_ids[0]
    print(f"[pusher] Resolved user_id from MongoDB: {user_id}")

    camera_docs = list(cameras_coll.find({"user_id": user_id}))
    if not camera_docs:
        print(f"[pusher] ❌ No cameras found for user_id={user_id}")
        await pc.close()
        return

    async def create_player(rtsp_url, label):
        result = await create_rtsp_player(rtsp_url, label)
        label, player, ok = result
        if player is not None:
            players.append((label, player))
        return result

    # create players for each camera of this user
    player_infos = []
    for cam in camera_docs:
        camera_id = cam.get("camera_id")
        rtsp_url = cam.get("rtsp_url")
        if not camera_id or not rtsp_url:
            print(f"[pusher] ⚠️ Skipping camera with missing data: {cam}")
            continue
        info = await create_player(rtsp_url, camera_id)
        player_infos.append(info)

    # If no players, exit
    active_infos = [info for info in player_infos if info[1] is not None]
    if not active_infos:
        print("[pusher] ❌ No players created, exiting")
        await pc.close()
        return

    # Add separate transceivers: one per camera. This forces separate m=video lines in SDP.
    async def add_transceiver_for(player_tuple):
        label, player = player_tuple
        if player is None:
            print(f"[pusher] Skipping {label}: player is None")
            return False
        try:
            proxied = ProxyVideoTrack(player.video, label)
            # Preferred approach: add a separate transceiver for each track.
            # We attempt to attach proxied directly to addTransceiver if supported.
            try:
                transceiver = pc.addTransceiver(proxied, direction="sendonly")
                # Some aiortc versions return a transceiver; the sender will be created.
                print(f"[pusher] Added transceiver for {label}. transceiver={transceiver}")
            except TypeError:
                # Fallback: add transceiver by kind, then replace sender.track
                transceiver = pc.addTransceiver(kind="video", direction="sendonly")
                sender = transceiver.sender
                try:
                    # replace_track may be available; try it.
                    await sender.replace_track(proxied)
                    print(f"[pusher] Replaced transceiver sender.track for {label}")
                except Exception:
                    # fallback to addTrack (less ideal)
                    sender = pc.addTrack(proxied)
                    print(f"[pusher] Fallback: used addTrack for {label}; sender={sender}")
            # Log sender info if possible
            try:
                s = transceiver.sender
                print(f"[pusher] sender for {label}: id={getattr(s,'id',None)} track={getattr(s,'track',None)}")
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[pusher] ❌ Error adding transceiver for {label}: {e}")
            return False

    # Add transceivers for each created player
    for label, player, _ok in active_infos:
        await add_transceiver_for((label, player))

    print(f"[pusher] Completed adding transceivers for {len(active_infos)} cameras")

    camera_client_id = f"camera:{user_id}"
    viewer_client_id = f"viewer:{user_id}"

    # Connect signaling
    ws_url = SIGNALING_WS.rstrip("/") + "/" + camera_client_id
    print("[pusher] Connecting to signaling server:", ws_url)

    try:
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
            print("[pusher] ✅ Signaling connected")

            @pc.on("icecandidate")
            async def on_local_ice(candidate):
                try:
                    if candidate is None:
                        print("[pusher] ✅ Local ICE gathering finished")
                        await ws.send(json.dumps({"type":"ice","from":camera_client_id,"to":viewer_client_id,"candidate":{}}))
                        return
                    if "relay" in candidate.to_sdp():
                        print("[pusher] 🔄 Sending TURN candidate")
                    msg = {
                        "type":"ice",
                        "from": camera_client_id,
                        "to": viewer_client_id,
                        "candidate": {
                            "candidate": candidate.to_sdp(),
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex
                        }
                    }
                    await ws.send(json.dumps(msg))
                    print("[pusher] Sent ICE candidate")
                except Exception as e:
                    print("[pusher] ❌ Error sending ICE candidate:", e)

            # create offer
            print("[pusher] Creating SDP offer...")
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            print("[pusher] Local description (offer) set. Printing short SDP for debug:")
            sdp = pc.localDescription.sdp
            print("----- SDP START (first 1600 chars) -----")
            print(sdp[:1600])
            print("----- SDP END -----")

            # send offer
            offer_msg = {"type":"offer","from": camera_client_id, "to": viewer_client_id, "sdp": pc.localDescription.sdp}
            await ws.send(json.dumps(offer_msg))
            print("[pusher] ✅ Offer sent to viewer")

            # Keep track of connection state
            connection_established = False
            last_activity = time.time()
            keepalive_task = None

            async def keepalive_loop():
                """Send fall alerts and pings while connection is active.

                If the peer connection enters a failed/closed state, proactively
                close the signaling WebSocket so the current session can end and
                a fresh one can be created by run_forever().
                """
                print("[pusher] Keep-alive loop started")
                last_heartbeat = time.time()
                try:
                    while not ws.closed:
                        await asyncio.sleep(1)
                        current_time = time.time()

                        # Monitor peer connection state
                        if pc.connectionState in ["failed", "closed", "disconnected"]:
                            print("[pusher] ⚠️ Peer connection state is", pc.connectionState,
                                  "- closing signaling WebSocket to restart session")
                            try:
                                await ws.close()
                            except Exception as e:
                                print(f"[pusher] ⚠️ Error closing WebSocket after PC failure: {e}")
                            break

                        # Send keep-alive ping every 10 seconds to maintain signaling connection
                        if current_time - last_heartbeat > 10:
                            try:
                                if ws and not ws.closed:
                                    await ws.send(json.dumps({
                                        "type": "ping",
                                        "from": camera_client_id,
                                        "to": viewer_client_id
                                    }))
                                    last_heartbeat = current_time
                            except Exception as e:
                                print(f"[pusher] Keep-alive ping failed: {e}")
                                break
                except asyncio.CancelledError:
                    print("[pusher] Keep-alive loop cancelled")
                    raise
                except Exception as e:
                    print(f"[pusher] Keep-alive loop error: {e}")

            # handle incoming messages with separate keep-alive task
            try:
                keepalive_task = asyncio.create_task(keepalive_loop())
                async for raw in ws:
                    last_activity = time.time()
                    try:
                        message = json.loads(raw)
                    except Exception as e:
                        print("[pusher] ⚠️ Invalid JSON:", e)
                        continue

                    typ = message.get("type")
                    if typ == "answer":
                        print("[pusher] Received answer: setting remote description")
                        try:
                            answer = RTCSessionDescription(sdp=message["sdp"], type="answer")
                            await pc.setRemoteDescription(answer)
                            print("[pusher] ✅ Remote description set")
                            connection_established = True
                        except Exception as e:
                            print("[pusher] ❌ setRemoteDescription failed:", e)
                    elif typ == "ice":
                        try:
                            candidate_data = message.get("candidate") or {}
                            candidate_str = candidate_data.get("candidate")
                            if not candidate_str:
                                await pc.addIceCandidate(None)
                                print("[pusher] ✅ Remote ICE end (added None)")
                                continue
                            candidate = candidate_from_sdp(candidate_str)
                            candidate.sdpMid = candidate_data.get("sdpMid")
                            candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
                            await pc.addIceCandidate(candidate)
                            print("[pusher] Added remote ICE candidate")
                        except Exception as e:
                            print("[pusher] ⚠️ Failed to add remote ICE:", e)
                    else:
                        print("[pusher] ⚠️ Unknown message type:", typ)
            except asyncio.CancelledError:
                print("[pusher] Message handling cancelled")
                raise
            finally:
                if keepalive_task is not None:
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass
                print("[pusher] Message loop finished, ending session")

    except Exception as e:
        print("[pusher] ❌ Signaling/WS exception:", e)
    finally:
        for label, player in players:
            try:
                if player is not None:
                    print(f"[pusher] Stopping player for {label}")
                    player.stop()
            except Exception as e:
                print(f"[pusher] ⚠️ Error stopping player for {label}: {e}")
        print("[pusher] Closing peer connection")
        await pc.close()


async def run_forever(retry_delay: float = 3.0):
    while True:
        print("[pusher] === Starting new WebRTC session ===")
        try:
            await run_single_session()
        except asyncio.CancelledError:
            print("[pusher] Streaming loop cancelled, shutting down")
            raise
        except Exception as e:
            print(f"[pusher] ❌ Unexpected error in WebRTC session: {e}")
        print(f"[pusher] Session ended, restarting in {retry_delay} seconds...")
        await asyncio.sleep(retry_delay)


if __name__ == "__main__":
    print("="*60)
    print("WebRTC Multi-Camera Pusher with YOLOv8 Pose Detection")
    print("User ID: will be resolved dynamically from MongoDB")
    print(f"Signaling URL: {SIGNALING_WS}")
    print("="*60)
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        print("\n[pusher] Stopped by user")
    except Exception as e:
        print("[pusher] Fatal:", e)

