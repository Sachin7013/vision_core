from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pymongo import ReturnDocument

from app.db import get_agents_collection, get_cameras_collection
from app.models import AgentCreate, AgentOut


router = APIRouter(prefix="/api", tags=["agents"])


def _mongo_agent_to_out(doc: dict) -> AgentOut:
    """Convert a raw MongoDB document into an AgentOut model.

    We expose the MongoDB ``_id`` as a simple ``id`` string field.
    """
    if not doc:
        raise ValueError("Agent document is empty")
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return AgentOut(**doc)


@router.post("/agents", response_model=AgentOut)
async def upsert_agent(agent: AgentCreate) -> AgentOut:
    """Create or update an agent configuration.

    This endpoint is intended to be called by the external Samits backend.
    If an agent with the same ``agent_id`` already exists, it will be
    replaced; otherwise, a new document is created.
    """
    coll = get_agents_collection()

    # Ensure the referenced camera exists; this also allows us to
    # validate that the camera_id is known in our system.
    cameras_coll = get_cameras_collection()
    camera_doc = cameras_coll.find_one({"camera_id": agent.camera_id})
    if not camera_doc:
        raise HTTPException(status_code=404, detail="Camera not found for given camera_id")

    now = datetime.now(timezone.utc)
    payload = agent.dict(by_alias=True)
    if not payload.get("created_at"):
        payload["created_at"] = now

    # Upsert by agent_id so repeated calls for the same agent simply
    # update the existing document.
    result = coll.find_one_and_update(
        {"agent_id": payload["agent_id"]},
        {"$set": payload},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return _mongo_agent_to_out(result)


@router.get("/agents", response_model=List[AgentOut])
async def list_agents(camera_id: Optional[str] = None, status: Optional[str] = None) -> List[AgentOut]:
    """List agents, optionally filtered by camera_id and/or status."""
    coll = get_agents_collection()
    query: dict = {}
    if camera_id:
        query["camera_id"] = camera_id
    if status:
        query["status"] = status

    docs = list(coll.find(query))
    return [_mongo_agent_to_out(d) for d in docs]


@router.get("/agents/{agent_id}", response_model=AgentOut)
async def get_agent(agent_id: str) -> AgentOut:
    """Fetch a single agent by its agent_id."""
    coll = get_agents_collection()
    doc = coll.find_one({"agent_id": agent_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _mongo_agent_to_out(doc)
