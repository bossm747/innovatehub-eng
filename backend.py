import os
from fastapi import FastAPI, Request, Response, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import json
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel
from crawl4ai import crawl_url

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

class CrawlRequest(BaseModel):
    url: str
    max_depth: int = 1
    markdown: bool = True

class CrawlResponse(BaseModel):
    url: str
    content: str
    status: str
    error: str = None

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

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_endpoint(req: CrawlRequest):
    try:
        result = await crawl_url(req.url, max_depth=req.max_depth, markdown=req.markdown)
        return CrawlResponse(url=req.url, content=result["content"], status="success")
    except Exception as e:
        return CrawlResponse(url=req.url, content="", status="error", error=str(e))

# --- MCP (Model Context Protocol) compatibility ---

MCP_TOOLS = [
    {
        "name": "crawl",
        "description": "Crawl a web page and return its content as markdown.",
        "parameters": {
            "url": {"type": "string", "description": "The URL to crawl."},
            "max_depth": {"type": "integer", "default": 1, "description": "How deep to crawl."},
            "markdown": {"type": "boolean", "default": True, "description": "Return markdown output."}
        },
        "returns": {"type": "object", "properties": {"content": {"type": "string"}}}
    },
    # Add more tools here (file management, terminal, etc.)
]

@app.get("/mcp/tools")
def mcp_list_tools():
    """List available tools in MCP format."""
    return {"tools": MCP_TOOLS}

@app.post("/mcp/tool-call")
async def mcp_tool_call(
    tool_name: str = Body(...),
    parameters: dict = Body(...)
):
    """Invoke a tool by name (MCP format)."""
    if tool_name == "crawl":
        req = CrawlRequest(**parameters)
        result = await crawl_endpoint(req)
        return {"result": result.dict()}
    # Add more tool dispatches here
    return {"error": f"Tool '{tool_name}' not found."}

# --- End MCP section ---

# ---
# Extend with more endpoints (tool calls, knowledge base, etc.) as needed
# --- 