import os
from fastapi import FastAPI, Request, Response, Body, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import json
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field
from crawl4ai import crawl_url
import subprocess
from stagehand import Stagehand, BrowserAction

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

# --- File Management Endpoints ---

class ListFilesRequest(BaseModel):
    path: str = Field(default=".", description="Directory to list")

class ListFilesResponse(BaseModel):
    files: list

@app.post("/files/list", response_model=ListFilesResponse)
def list_files(req: ListFilesRequest):
    try:
        files = os.listdir(req.path)
        return ListFilesResponse(files=files)
    except Exception as e:
        return ListFilesResponse(files=[f"Error: {str(e)}"])

class ReadFileRequest(BaseModel):
    path: str

class ReadFileResponse(BaseModel):
    content: str

@app.post("/files/read", response_model=ReadFileResponse)
def read_file(req: ReadFileRequest):
    try:
        with open(req.path, "r", encoding="utf-8") as f:
            content = f.read()
        return ReadFileResponse(content=content)
    except Exception as e:
        return ReadFileResponse(content=f"Error: {str(e)}")

class WriteFileRequest(BaseModel):
    path: str
    content: str

class WriteFileResponse(BaseModel):
    status: str

@app.post("/files/write", response_model=WriteFileResponse)
def write_file(req: WriteFileRequest):
    try:
        with open(req.path, "w", encoding="utf-8") as f:
            f.write(req.content)
        return WriteFileResponse(status="success")
    except Exception as e:
        return WriteFileResponse(status=f"Error: {str(e)}")

class DeleteFileRequest(BaseModel):
    path: str

class DeleteFileResponse(BaseModel):
    status: str

@app.post("/files/delete", response_model=DeleteFileResponse)
def delete_file(req: DeleteFileRequest):
    try:
        os.remove(req.path)
        return DeleteFileResponse(status="success")
    except Exception as e:
        return DeleteFileResponse(status=f"Error: {str(e)}")

# --- Terminal/Automation Endpoint ---
class TerminalExecRequest(BaseModel):
    command: str

class TerminalExecResponse(BaseModel):
    output: str
    status: str

@app.post("/terminal/exec", response_model=TerminalExecResponse)
def terminal_exec(req: TerminalExecRequest):
    try:
        # Security: Only allow safe commands in production!
        result = subprocess.run(req.command, shell=True, capture_output=True, text=True, timeout=10)
        return TerminalExecResponse(output=result.stdout + result.stderr, status="success")
    except Exception as e:
        return TerminalExecResponse(output="", status=f"Error: {str(e)}")

# --- Register as MCP tools ---
MCP_TOOLS.extend([
    {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {"path": {"type": "string", "default": "."}},
        "returns": {"type": "object", "properties": {"files": {"type": "array"}}}
    },
    {
        "name": "read_file",
        "description": "Read a file's contents.",
        "parameters": {"path": {"type": "string"}},
        "returns": {"type": "object", "properties": {"content": {"type": "string"}}}
    },
    {
        "name": "write_file",
        "description": "Write or update a file.",
        "parameters": {"path": {"type": "string"}, "content": {"type": "string"}},
        "returns": {"type": "object", "properties": {"status": {"type": "string"}}}
    },
    {
        "name": "delete_file",
        "description": "Delete a file.",
        "parameters": {"path": {"type": "string"}},
        "returns": {"type": "object", "properties": {"status": {"type": "string"}}}
    },
    {
        "name": "terminal_exec",
        "description": "Execute a shell command (use with caution).",
        "parameters": {"command": {"type": "string"}},
        "returns": {"type": "object", "properties": {"output": {"type": "string"}, "status": {"type": "string"}}}
    },
])

# --- Stagehand Browser Automation Endpoint ---
class StagehandRequest(BaseModel):
    action: str
    url: str = None
    selector: str = None
    text: str = None

class StagehandResponse(BaseModel):
    status: str
    result: str = None
    error: str = None

stagehand = Stagehand()

@app.post("/stagehand", response_model=StagehandResponse)
async def stagehand_endpoint(req: StagehandRequest):
    try:
        if req.action == "open_url":
            await stagehand.open_page(req.url)
            return StagehandResponse(status="success", result=f"Opened {req.url}")
        elif req.action == "click":
            await stagehand.click(req.selector)
            return StagehandResponse(status="success", result=f"Clicked {req.selector}")
        elif req.action == "extract_text":
            text = await stagehand.extract_text(req.selector)
            return StagehandResponse(status="success", result=text)
        else:
            return StagehandResponse(status="error", error="Unknown action")
    except Exception as e:
        return StagehandResponse(status="error", error=str(e))

# --- Register Stagehand as MCP tool ---
MCP_TOOLS.append({
    "name": "stagehand",
    "description": "Browser automation: open URL, click, extract text, etc.",
    "parameters": {
        "action": {"type": "string", "description": "Action to perform (open_url, click, extract_text)"},
        "url": {"type": "string", "description": "URL to open (for open_url)"},
        "selector": {"type": "string", "description": "CSS selector (for click/extract)"},
        "text": {"type": "string", "description": "Text to type/click (future)"}
    },
    "returns": {"type": "object", "properties": {"status": {"type": "string"}, "result": {"type": "string"}, "error": {"type": "string"}}}
})

@app.post("/mcp/tool-call")
async def mcp_tool_call(
    tool_name: str = Body(...),
    parameters: dict = Body(...)
):
    if tool_name == "crawl":
        req = CrawlRequest(**parameters)
        result = await crawl_endpoint(req)
        return {"result": result.dict()}
    elif tool_name == "list_files":
        req = ListFilesRequest(**parameters)
        result = list_files(req)
        return {"result": result.dict()}
    elif tool_name == "read_file":
        req = ReadFileRequest(**parameters)
        result = read_file(req)
        return {"result": result.dict()}
    elif tool_name == "write_file":
        req = WriteFileRequest(**parameters)
        result = write_file(req)
        return {"result": result.dict()}
    elif tool_name == "delete_file":
        req = DeleteFileRequest(**parameters)
        result = delete_file(req)
        return {"result": result.dict()}
    elif tool_name == "terminal_exec":
        req = TerminalExecRequest(**parameters)
        result = terminal_exec(req)
        return {"result": result.dict()}
    elif tool_name == "stagehand":
        req = StagehandRequest(**parameters)
        result = await stagehand_endpoint(req)
        return {"result": result.dict()}
    return {"error": f"Tool '{tool_name}' not found."}

# --- End MCP section ---

# ---
# Extend with more endpoints (tool calls, knowledge base, etc.) as needed
# --- 