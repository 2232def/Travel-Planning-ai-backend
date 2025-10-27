from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# from ..app.services.retriever import QdrantRetriever
from app.services.llm import generate_query_or_respond

router = APIRouter(prefix="/api")

class AskRequest(BaseModel):
    question: str
    k: int | None = 5

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ask")
def ask(request: AskRequest):

    state = {"messages" : [{"role": "user", "content": request.question}]}
    try:
        out = generate_query_or_respond(state)
        msg = out["messages"][-1]
        content = getattr(msg, "content", None) or getattr(msg, "text", None)
        return {"answer":  content}

    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e)) 