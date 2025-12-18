"""
DATABASE SETUP
==============
This file connects to MongoDB and gives us easy access to our data.

Think of it like:
- MongoDB = A filing cabinet
- Database = One drawer in the cabinet
- Collection = A folder in the drawer
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient

# Load settings from .env file
load_dotenv()

# MongoDB connection details
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "vision_core")

# Store connection (we only create it once to save memory)
_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    """
    Get MongoDB connection.
    
    We create it once and reuse it (called "singleton pattern").
    This saves memory instead of creating new connections every time.
    """
    global _client
    if _client is None:
        print("📡 Connecting to MongoDB...")
        _client = MongoClient(MONGODB_URI)
    return _client


def get_database():
    """Get the main database where all our data is stored."""
    client = get_client()
    return client[MONGODB_DB_NAME]


def get_cameras_collection():
    """
    Get the cameras collection (where we store camera information).
    
    A collection is like a table in Excel with rows and columns.
    Each row = one camera
    
    Returns: cameras collection from MongoDB
    """
    db = get_database()
    coll = db["cameras"]
    
    # Make sure we don't have duplicate cameras for same user+camera combo
    coll.create_index([("user_id", 1), ("camera_id", 1)], unique=True)
    
    return coll


def get_agents_collection():
    """
    Get the agents collection (where we store automation rules/agents).
    
    A collection is like a table where:
    Each row = one automation agent
    
    Returns: agents collection from MongoDB
    """
    db = get_database()
    coll = db["agents"]
    
    # Make sure each agent_id is unique (no duplicates)
    coll.create_index([("agent_id", 1)], unique=True)
    
    return coll
