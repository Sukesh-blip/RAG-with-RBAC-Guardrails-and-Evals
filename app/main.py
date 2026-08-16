"""
FastAPI backend exposing /chat and /ingest.
"""

import sys
from pathlib import Path

# Allow imports from project root when running via uvicorn app.main:app
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from monitoring.cost_tracker import get_usage_summary
from rbac.access_control import VALID_ROLES
from ingestion.loader import run_ingestion
from agents.graph import run_agent

app = FastAPI(title="Agentic RAG RBAC Chatbot")


class ChatRequest(BaseModel):
    role: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    retrieved_count: int
    out_of_scope: bool
    pii_redacted: list
    sub_queries: list
    token_usage: dict
    cost_alerts: list


@app.get("/")
def root():
    return {"status": "ok", "valid_roles": sorted(VALID_ROLES)}


@app.post("/ingest")
def ingest():
    try:
        summary = run_ingestion()
        return {"status": "success", **summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.role not in VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{request.role}'. Valid roles: {sorted(VALID_ROLES)}",
        )
    try:
        result = run_agent(request.role, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost/usage")
def cost_usage():
    return get_usage_summary()