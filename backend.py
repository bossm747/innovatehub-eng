import os
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import json
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI()

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy model/provider list (extend as needed)
AVAILABLE_MODELS = [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "deepseek", "model": "deepseek-reasoner"},
    # Add more models/providers here
]

@app.get("/models")
def list_models():
    """List available LLM models/providers."""
    return {"models": AVAILABLE_MODELS}

async def call_llm(model: str, messages: List[Dict[str, Any]]):
    """
    Production-ready LLM API integration for OpenAI and DeepSeek.
    Streams responses as AG UI events. Add more providers as needed.
    """
    provider = None
    for m in AVAILABLE_MODELS:
        if m["model"] == model:
            provider = m["provider"]
            break
    if not provider:
        yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'error', 'error': 'Unknown model'})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'started'})}\n\n"
    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'error', 'error': 'Missing OpenAI API key'})}\n\n"
                return
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                delta = json.loads(data)
                                content = delta.get("choices", [{}])[0].get("delta", {}).get("content")
                                if content:
                                    yield f"data: {json.dumps({'type': 'text-delta', 'value': content})}\n\n"
                            except Exception:
                                continue
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'error', 'error': 'Missing DeepSeek API key'})}\n\n"
                return
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data = line[len("data: "):]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                delta = json.loads(data)
                                content = delta.get("choices", [{}])[0].get("delta", {}).get("content")
                                if content:
                                    yield f"data: {json.dumps({'type': 'text-delta', 'value': content})}\n\n"
                            except Exception:
                                continue
        else:
            yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'error', 'error': 'Provider not implemented'})}\n\n"
            return
    except Exception as e:
        yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'error', 'error': str(e)})}\n\n"
        return
    yield f"data: {json.dumps({'type': 'lifecycle', 'status': 'completed'})}\n\n"

@app.post("/chat")
async def chat(request: Request):
    """Accept chat request and stream AG UI events (SSE)."""
    body = await request.json()
    model = body.get("model", "gpt-4o")
    messages = body.get("messages", [])
    return StreamingResponse(
        call_llm(model, messages),
        media_type="text/event-stream"
    )

# ---
# Extend with more endpoints (tool calls, knowledge base, etc.) as needed
# --- 