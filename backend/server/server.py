import json
import os
from typing import Dict, List
from fastapi import FastAPI, Request, File, UploadFile, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import shutil

from backend.server.server_utils import (
    get_config_dict,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_state_communication
)

# Load environment variables
load_dotenv()

# Get logger instance
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str


class ConfigRequest(BaseModel):
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    DOC_PATH: str
    RETRIEVER: str
    GOOGLE_API_KEY: str = ''
    GOOGLE_CX_KEY: str = ''
    BING_API_KEY: str = ''
    SEARCHAPI_API_KEY: str = ''
    SERPAPI_API_KEY: str = ''
    SERPER_API_KEY: str = ''
    SEARX_URL: str = ''
    XAI_API_KEY: str
    DEEPSEEK_API_KEY: str


# App initialization
app = FastAPI()

# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Initialize state manager
app_state = {}

# Startup event


@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    os.makedirs(DOC_PATH, exist_ok=True)


# Routes


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/files/")
async def list_files():
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/start")
async def start_task(request: Request):
    """Start a research task"""
    try:
        data = await request.json()
        
        # Initialize task state
        task_state = {
            "task": data.get("task"),
            "status": "starting",
            "logs": []
        }
        app_state["current_task"] = task_state

        # Start task processing
        report = await handle_state_communication(task_state)
        return JSONResponse(content={"report": report})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/chat")
async def chat(request: Request):
    """Handle chat messages"""
    try:
        data = await request.json()
        
        # Update chat state
        chat_state = app_state.get("current_task", {})
        chat_state.update({
            "command": "chat",
            "chat_data": data
        })

        # Process chat message
        response = await handle_state_communication(chat_state)
        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads"""
    try:
        result = await handle_file_upload(file, DOC_PATH)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.delete("/api/delete/{filename}")
async def delete_file(filename: str):
    """Handle file deletion"""
    try:
        return await handle_file_deletion(filename, DOC_PATH)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/config")
async def update_config(request: Request):
    """Update configuration"""
    try:
        data = await request.json()
        config = get_config_dict(**data)
        update_environment_variables(config)
        return JSONResponse(content={"message": "Configuration updated successfully"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/execute")
async def execute(request: Request):
    """Execute multi-agent task"""
    try:
        return await execute_multi_agents(app_state)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/state")
async def get_state():
    """Get current application state"""
    return JSONResponse(content=app_state)
