# signaling_server.py
import json
import logging
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [SIGNALING] %(levelname)s: %(message)s'
)
logger = logging.getLogger("signaling")

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Client State
# --------------------------------------------------
class Client:
    def __init__(self, client_id: str, ws: WebSocket):
        self.id = client_id
        self.ws = ws
        # camera OR agent are publishers
        self.is_publisher = client_id.startswith("camera") or client_id.startswith("agent")
        self.last_offer = None

clients: Dict[str, Client] = {}

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def get_publisher_id_for_viewer(viewer_id: str) -> str | None:
    """
    viewer:<user>:<camera>            -> camera:<user>
    viewer:<user>:<agent>             -> agent:<user>:<agent>
    """
    parts = viewer_id.split(":")
    if len(parts) == 2:
        # viewer:user
        return f"camera:{parts[1]}"
    if len(parts) == 3:
        # viewer:user:camera_id OR viewer:user:agent_id
        return f"agent:{parts[1]}:{parts[2]}"
    return None

# --------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------
@app.websocket("/ws/{client_id}")
async def ws_endpoint(ws: WebSocket, client_id: str):
    await ws.accept()
    client = Client(client_id, ws)
    clients[client_id] = client

    role = "PUBLISHER" if client.is_publisher else "VIEWER"
    logger.info(f"✅ {role} connected: {client_id}")

    # --------------------------------------------------
    # If viewer joins late → replay last offer
    # --------------------------------------------------
    if not client.is_publisher:
        publisher_id = get_publisher_id_for_viewer(client_id)
        if publisher_id and publisher_id in clients:
            publisher = clients[publisher_id]
            if publisher.last_offer:
                await ws.send_text(json.dumps({
                    "type": "offer",
                    "from": publisher_id,
                    "to": client_id,
                    "sdp": publisher.last_offer
                }))
                logger.info(f"📤 Replayed offer {publisher_id} → {client_id}")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            msg_type = msg.get("type")
            target = msg.get("to")

            # --------------------------------------------------
            # OFFER (camera / agent → viewer)
            # --------------------------------------------------
            if msg_type == "offer" and client.is_publisher:
                client.last_offer = msg.get("sdp")
                logger.info(f"📨 OFFER from {client.id}")

                if target and target in clients:
                    await clients[target].ws.send_text(data)
                    logger.info(f"📤 OFFER forwarded → {target}")

            # --------------------------------------------------
            # ANSWER (viewer → camera / agent)
            # --------------------------------------------------
            elif msg_type == "answer" and not client.is_publisher:
                if target and target in clients:
                    await clients[target].ws.send_text(data)
                    logger.info(f"📤 ANSWER forwarded → {target}")

            # --------------------------------------------------
            # ICE
            # --------------------------------------------------
            elif msg_type == "ice":
                if target and target in clients:
                    await clients[target].ws.send_text(data)

            # --------------------------------------------------
            # PING
            # --------------------------------------------------
            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        logger.info(f"🔌 Disconnected: {client_id}")
    finally:
        clients.pop(client_id, None)
        logger.info(f"📊 Active clients: {len(clients)}")

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
