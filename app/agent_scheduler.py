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
from typing import List

from app.db import get_agents_collection
from app.rule_engine.engine import AgentRuntime, build_agent_runtimes_for_camera


async def _agent_status_loop(interval_seconds: int = 5) -> None:
    """
    Background task that updates agent status automatically.
    
    This runs continuously in the background checking the time.
    
    Status changes:
    pending    → Agent not started yet (waiting for start_at time)
    running    → Agent is active (between start_at and end_at)
    terminated → Agent is done (end_at time has passed)
    
    Args:
        interval_seconds: How often to check (in seconds)
    """
    coll = get_agents_collection()
    
    print("⏰ Agent scheduler started (checks every 5 seconds)")

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

                # Skip if missing times
                if not start_at or not end_at:
                    continue

                # DECIDE THE NEW STATUS
                if now < start_at:
                    # Not yet time to start
                    new_status = "pending"
                elif start_at <= now < end_at:
                    # Time is between start and end → RUN IT!
                    new_status = "running"
                else:
                    # Time has passed → STOP IT
                    new_status = "terminated"

                # If status changed, update database
                if new_status != current_status:
                    coll.update_one(
                        {"_id": agent_doc["_id"]},
                        {"$set": {"status": new_status}}
                    )
                    print(f"📊 Agent '{agent_id}': {current_status} → {new_status}")

        except Exception as exc:
            print(f"⚠️ Error in scheduler: {exc}")
            # Keep running even if error occurs (auto-recovery)

        # Wait before checking again
        await asyncio.sleep(interval_seconds)


def get_running_agents_for_camera(camera_id: str) -> List[AgentRuntime]:
    """
    Get all ACTIVE agents for a camera RIGHT NOW.
    
    This is the key function that says:
    "Which agents should process frames for this camera?"
    
    IMPORTANT: Only agents with status='running' are returned!
    
    Args:
        camera_id: Which camera
    
    Returns:
        List of agents ready to use (empty list if none running)
    
    EXAMPLE:
    If we have 3 agents on camera "cam01":
    - agent_A: status='pending' (not time yet)
    - agent_B: status='running' (active!)
    - agent_C: status='terminated' (already done)
    
    This function returns ONLY [agent_B]
    """
    return build_agent_runtimes_for_camera(camera_id)


async def start_agent_scheduler() -> None:
    """Start the background agent status scheduler.

    This should be called from FastAPI's startup event; it spawns a
    detached asyncio task that runs for the lifetime of the process.

    RESPONSIBILITIES:
        1. Continuously updates agent status based on timing
        2. Maintains "source of truth" in database for agent state
        3. Does NOT execute agents (that's engine.py's job)

    FLOW DIAGRAM:
        start_agent_scheduler()
            ↓
        _agent_status_loop() [runs continuously]
            ↓
        Updates agent.status in database every 50 seconds
            ↓
        When status='running', engine.py picks up the agent
            ↓
        engine.py calls get_running_agents_for_camera()
            ↓
        Only running agents are processed for frame detection
    """
    asyncio.create_task(_agent_status_loop())
