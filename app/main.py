import asyncio
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.cameras import router as cameras_router
from app.api.agents import router as agents_router
from app.agent_scheduler import start_agent_scheduler


app = FastAPI(title="Vision Core Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Path and module name for the WebRTC sender process
SENDER_MODULE = "app.streamer.sender_stream"
SENDER_PATH = APP_DIR / "streamer" / "sender_stream.py"


@app.on_event("startup")
async def start_live_sender_background() -> None:
    """Spawn the live-sender process in the background when the API starts.

    We use subprocess.Popen instead of asyncio.create_subprocess_exec because
    the default event loop on Windows used by uvicorn does not implement
    subprocess support, which would raise NotImplementedError.
    """

    if not SENDER_PATH.exists():
        # Nothing to start if the script is missing
        return

    # Run as a module so that `app` is treated as a proper package and
    # imports like `from app.object_detection_part.object_detection import ...` work reliably.
    cmd = [sys.executable, "-m", SENDER_MODULE]
    # Run from project root so Python finds the `app` package
    try:
        subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    except Exception:
        # Fail silently here so the API can still start even if the
        # live-sender process cannot be spawned for some reason.
        # Detailed errors will appear if you run live-sender manually.
        pass


@app.on_event("startup")
async def start_background_schedulers() -> None:
    """Start background tasks such as the agent status scheduler."""
    await start_agent_scheduler()


@app.get("/")
async def root():
    return {"message": "Vision Core Backend is running"}


app.include_router(cameras_router)
app.include_router(agents_router)
