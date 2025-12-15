import os
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env if present
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "vision_core")

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    """Return a shared MongoClient instance."""
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client


def get_database():
    """Return the main application database."""
    client = get_client()
    return client[MONGODB_DB_NAME]


def get_cameras_collection():
    """Convenience helper to get the cameras collection."""
    db = get_database()
    coll = db["cameras"]
    # Ensure a unique index on (user_id, camera_id) so we do not store duplicates
    coll.create_index([("user_id", 1), ("camera_id", 1)], unique=True)
    return coll
