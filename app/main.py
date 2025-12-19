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

SENDER_MODULE = "app.streamer.sender_stream"
SENDER_PATH = APP_DIR / "streamer" / "sender_stream.py"


@app.on_event("startup")
async def start_live_sender_background() -> None:
    if not SENDER_PATH.exists():
        return

    cmd = [sys.executable, "-m", SENDER_MODULE]
    try:
        subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    except Exception:
        pass


@app.on_event("startup")
async def start_background_schedulers() -> None:
    await start_agent_scheduler()


@app.get("/")
async def root():
    return {"message": "Vision Core Backend is running"}


app.include_router(cameras_router)
app.include_router(agents_router)
