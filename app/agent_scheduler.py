"""
AGENT SCHEDULER - WHO RUNS AND WHEN?
=====================================
This file automatically turns agents ON and OFF based on time.

Think of it like:
- Your agent has a start time (e.g., 9:00 AM)
- Your agent has an end time (e.g., 5:00 PM)
- The scheduler checks: "Is it time yet?"
- If YES → Turn agent ON
- If NO → Keep it OFF
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Dict

from app.db import get_agents_collection
from app.shared_hub.hub import SharedFrameHub
from app.rule_engine.engine import (
    AgentRuntime,
    build_agent_runtime_from_doc,
    process_frame_for_agent,
)

# Track background runner tasks per agent
_agent_tasks: Dict[str, asyncio.Task] = {}


async def _agent_status_loop(interval_seconds: int = 10) -> None:

    coll = get_agents_collection()
    
    print("⏰ Agent scheduler started (checks every 10 seconds)")

    while True:
        # Get current time
        now = datetime.utcnow()
        
        try:
            # Get ALL agents that are not already terminated
            agents = coll.find({"status": {"$ne": "terminated"}})
            
            for agent_doc in agents:
                start_at = agent_doc.get("start_at")
                end_at = agent_doc.get("end_at")
                current_status = agent_doc.get("status", "pending")
                agent_id = agent_doc.get("agent_id")
                run_mode = str(agent_doc.get("run_mode", "scheduled")).lower()

                # Skip if missing times
                if not start_at or not end_at:
                    continue

                # DECIDE THE NEW STATUS
                if run_mode == "continuous":
                    # Always run regardless of time window
                    new_status = "running"
                else:
                    if now < start_at:
                        new_status = "pending"
                    elif start_at <= now < end_at:
                        new_status = "running"
                    else:
                        new_status = "terminated"

                # If status changed, update database
                if new_status != current_status:
                    coll.update_one(
                        {"_id": agent_doc["_id"]},
                        {"$set": {"status": new_status}}
                    )
                    print(f"📊 Agent '{agent_id}': {current_status} → {new_status}")

                # Ensure runner is started/stopped based on effective status
                effective_status = new_status
                if effective_status == "running":
                    if agent_id not in _agent_tasks:
                        runtime = build_agent_runtime_from_doc(agent_doc)
                        if runtime is not None:
                            _agent_tasks[agent_id] = asyncio.create_task(_run_agent(runtime))
                else:
                    task = _agent_tasks.pop(agent_id, None)
                    if task is not None:
                        task.cancel()

        except Exception as exc:
            print(f"⚠️ Error in scheduler: {exc}")
            # Keep running even if error occurs (auto-recovery)

        # Wait before checking again
        await asyncio.sleep(interval_seconds)


async def _run_agent(runtime: AgentRuntime) -> None:
    """Continuously pull latest frames for the agent's camera and process at the agent's FPS."""
    hub = SharedFrameHub.instance()
    interval = 1.0 / max(1, int(runtime.fps or 1))
    channel_out = f"agent:{runtime.agent_id}"
    try:
        while True:
            await asyncio.sleep(interval)
            frame = hub.get_latest(runtime.camera_id)
            if frame is None:
                continue
            processed = process_frame_for_agent(runtime, frame)
            hub.publish(channel_out, processed)
    except asyncio.CancelledError:
        return


async def start_agent_scheduler() -> None:
    asyncio.create_task(_agent_status_loop())
