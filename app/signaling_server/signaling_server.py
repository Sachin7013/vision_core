# # server_signaling.py
import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
import uvicorn

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [SIGNALING] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve viewer.html
BASE_DIR = Path(__file__).resolve().parent
VIEWER_FILE = BASE_DIR / "viewer.html"

# Global state - track clients and their subscriptions
class ClientState:
    def __init__(self, client_id: str, websocket: WebSocket):
        self.client_id = client_id
        self.websocket = websocket
        self.is_camera = client_id.startswith("camera")
        self.subscribed_cameras: Set[str] = set()
        self.last_offer: str = None
        self.last_answer: str = None

clients: Dict[str, ClientState] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Enhanced signaling with multi-viewer support and connection stability"""
    await websocket.accept()
    
    is_camera = client_id.startswith("camera")
    role = "🎥 Camera" if is_camera else "👁️ Viewer"
    
    logger.info(f"✅ {role} '{client_id}' connected")
    client = ClientState(client_id, websocket)
    clients[client_id] = client
    
    # If a viewer connects after its camera, immediately replay the last stored offer
    # from that camera so the WebRTC handshake can start without restarting the sender.
    if not is_camera:
        try:
            # Expected IDs: "viewer:<user_id>" and "camera:<user_id>"
            if ":" in client_id:
                _, user_part = client_id.split(":", 1)
                camera_peer_id = f"camera:{user_part}"
                camera_client = clients.get(camera_peer_id)
                if camera_client and camera_client.last_offer:
                    offer_msg = {
                        "type": "offer",
                        "from": camera_peer_id,
                        "to": client_id,
                        "sdp": camera_client.last_offer,
                    }
                    await websocket.send_text(json.dumps(offer_msg))
                    logger.info(
                        f"📤 Replayed stored offer from '{camera_peer_id}' to late viewer '{client_id}'"
                    )
        except Exception as e:
            logger.error(f"❌ Failed to replay stored offer to '{client_id}': {e}")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)
            except Exception as e:
                logger.error(f"❌ Receive error from {client_id}: {e}")
                break
            
            msg_type = msg.get("type")
            target = msg.get("to")
            
            # ===== OFFER: Camera sends offer to viewer(s) =====
            if msg_type == "offer":
                if is_camera:
                    # Store offer for new viewers
                    client.last_offer = msg.get("sdp")
                    logger.info(f"📨 Camera '{client_id}' sent offer")
                    
                    # Forward to specific viewer or all viewers
                    if target and target in clients:
                        try:
                            await clients[target].websocket.send_text(data)
                            logger.info(f"📤 Forwarded offer to viewer '{target}'")
                        except Exception as e:
                            logger.error(f"❌ Failed to forward offer to {target}: {e}")
                    else:
                        # Broadcast to all connected viewers
                        for viewer_id, viewer in clients.items():
                            if not viewer.is_camera:
                                try:
                                    await viewer.websocket.send_text(data)
                                    logger.info(f"📤 Broadcast offer to viewer '{viewer_id}'")
                                except Exception as e:
                                    logger.error(f"❌ Failed to broadcast to {viewer_id}: {e}")
            
            # ===== ANSWER: Viewer sends answer back to camera =====
            elif msg_type == "answer":
                if not is_camera:
                    # Store answer
                    client.last_answer = msg.get("sdp")
                    logger.info(f"📨 Viewer '{client_id}' sent answer")
                    
                    # Forward to camera
                    if target and target in clients:
                        try:
                            await clients[target].websocket.send_text(data)
                            logger.info(f"📤 Forwarded answer to camera '{target}'")
                        except Exception as e:
                            logger.error(f"❌ Failed to forward answer: {e}")
            
            # ===== ICE: Forward ICE candidates =====
            elif msg_type == "ice":
                if target and target in clients:
                    try:
                        await clients[target].websocket.send_text(data)
                        logger.debug(f"🔄 Forwarded ICE candidate to '{target}'")
                    except Exception as e:
                        logger.error(f"❌ Failed to forward ICE: {e}")
            
            # ===== ICE-COMPLETE: Signal ICE gathering done =====
            elif msg_type == "ice-complete":
                if target and target in clients:
                    try:
                        await clients[target].websocket.send_text(data)
                        logger.info(f"✅ Forwarded ICE-complete to '{target}'")
                    except Exception as e:
                        logger.error(f"❌ Failed to forward ICE-complete: {e}")
            
            # ===== PING: Keep-alive heartbeat =====
            elif msg_type == "ping":
                try:
                    pong_msg = {"type": "pong", "from": "signaling"}
                    await websocket.send_text(json.dumps(pong_msg))
                except Exception as e:
                    logger.error(f"❌ Ping/pong error: {e}")
            
            else:
                logger.debug(f"Unknown message type: {msg_type}")
    
    except WebSocketDisconnect:
        logger.info(f"🔌 {role} '{client_id}' disconnected")
    except Exception as e:
        logger.error(f"❌ Error in {client_id}: {e}")
    
    finally:
        if client_id in clients:
            del clients[client_id]
        logger.info(f"📊 Active clients: {len(clients)}")

@app.get("/")
async def root():
    return {"message":"Signaling server running"}

@app.get("/viewer", response_class=HTMLResponse)
async def serve_viewer():
    if not VIEWER_FILE.exists():
        return HTMLResponse("viewer.html not found", status_code=404)
    return VIEWER_FILE.read_text(encoding="utf-8")

if __name__ == "__main__":
    uvicorn.run("server_signaling:app", host="0.0.0.0", port=8000, log_level="info")

# ============================================================================================