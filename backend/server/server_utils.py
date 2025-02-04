import json
import os
import re
import time
import shutil
from typing import Dict, List, Any
from fastapi.responses import JSONResponse, FileResponse
from icis_researcher.document.document import DocumentLoader
from backend.utils import write_md_to_pdf, write_md_to_word, write_text_to_md
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomLogsHandler:
    """Custom handler for logging messages"""

    def __init__(self, state: Dict, task: str):
        """Initialize the handler"""
        self.state = state
        self.task = task
        self.logs = []

    async def emit(self, data):
        """Store log data and update state"""
        # Add to logs list
        self.logs.append(data)
        
        # Update state
        if self.state:
            self.state["logs"].append(data)
            await copilotkit_emit_state(self.state.get("config"), self.state)

    async def handle_task(self, task_data):
        """Handle a research task"""
        try:
            # Initialize state
            self.state = {
                "task": task_data,
                "logs": [],
                "config": None,
                "status": "running"
            }

            # Create researcher instance
            researcher = Researcher(
                task=task_data,
                state=self.state,
                logs_handler=self
            )

            # Run research
            report = await researcher.run()

            # Update final state
            self.state["status"] = "completed"
            self.state["report"] = report
            await copilotkit_emit_state(self.state.get("config"), self.state)

            return report

        except Exception as e:
            # Update error state
            self.state["status"] = "error"
            self.state["error"] = str(e)
            await copilotkit_emit_state(self.state.get("config"), self.state)
            raise e

async def handle_start_command(state: Dict, data: str):
    """Handle the start command"""
    try:
        # Parse task data
        json_data = json.loads(data)
        task = json_data.get("task")

        # Create logs handler with state and task
        logs_handler = CustomLogsHandler(state, task)

        # Initialize task state
        state.update({
            "task": task,
            "logs": [],
            "status": "starting"
        })
        await copilotkit_emit_state(state.get("config"), state)

        # Run research task
        report = await run_research_task(
            task,
            state,
            logs_handler
        )

        # Update final state
        state["status"] = "completed"
        state["report"] = report
        await copilotkit_emit_state(state.get("config"), state)

        return report

    except Exception as e:
        # Update error state
        state["status"] = "error" 
        state["error"] = str(e)
        await copilotkit_emit_state(state.get("config"), state)
        raise e

async def handle_chat(state: Dict, data: str):
    """Handle chat messages"""
    try:
        json_data = json.loads(data)
        message = json_data.get("message")

        # Update chat state
        state["chat_message"] = message
        await copilotkit_emit_state(state.get("config"), state)

        # Process chat message
        response = await process_chat_message(message, state)

        # Update response state
        state["chat_response"] = response
        await copilotkit_emit_state(state.get("config"), state)

        return response

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        await copilotkit_emit_state(state.get("config"), state)
        raise e

async def update_file_paths(state: Dict, file_paths: Dict[str, str]):
    """Update file paths in state"""
    state["file_paths"] = file_paths
    await copilotkit_emit_state(state.get("config"), state)

async def run_research_task(task: str, state: Dict, stream_output: bool = True) -> str:
    """Run a research task"""
    try:
        # Initialize research state
        state.update({
            "task": task,
            "status": "running",
            "stream_output": stream_output
        })
        await copilotkit_emit_state(state.get("config"), state)

        # Create researcher instance
        researcher = Researcher(
            task=task,
            state=state
        )

        # Run research
        report = await researcher.run()

        # Update final state
        state["status"] = "completed"
        state["report"] = report
        await copilotkit_emit_state(state.get("config"), state)

        return report

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        await copilotkit_emit_state(state.get("config"), state)
        raise e

async def handle_state_communication(state: Dict):
    """Handle state-based communication"""
    try:
        while True:
            # Check state for new commands
            if state.get("command"):
                command = state["command"]
                
                if command == "start":
                    await handle_start_command(state, state.get("task_data"))
                elif command == "chat":
                    await handle_chat(state, state.get("chat_data"))
                
                # Clear processed command
                state["command"] = None
                await copilotkit_emit_state(state.get("config"), state)

            # Sleep briefly to prevent busy waiting
            await asyncio.sleep(0.1)

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        await copilotkit_emit_state(state.get("config"), state)

class Researcher:
    def __init__(self, query: str, state: Dict, logs_handler: CustomLogsHandler):
        self.query = query
        self.state = state
        self.logs_handler = logs_handler

    async def run(self) -> str:
        """Conduct research and return report"""
        # Generate unique ID for this research task
        self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(self.query)}"

        # Initialize logs handler with research ID
        self.logs_handler.state = self.state

        # Run research
        report = await self.researcher.conduct_research()

        # Generate the files
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{self.query}")
        file_paths = await generate_report_files(report, sanitized_filename)

        # Get the JSON log path that was created by CustomLogsHandler
        json_relative_path = os.path.relpath(self.logs_handler.state.get("log_file"))

        # Update file paths in state
        await update_file_paths(self.state, {
            "pdf": file_paths["pdf"],
            "docx": file_paths["docx"],
            "md": file_paths["md"],
            "json": json_relative_path
        })

        return report

def sanitize_filename(filename: str) -> str:
    # Split into components
    prefix, timestamp, *task_parts = filename.split('_')
    task = '_'.join(task_parts)
    
    # Calculate max length for task portion
    # 255 - len("outputs/") - len("task_") - len(timestamp) - len("_.json") - safety_margin
    max_task_length = 255 - 8 - 5 - 10 - 6 - 10  # ~216 chars for task
    
    # Truncate task if needed
    truncated_task = task[:max_task_length] if len(task) > max_task_length else task
    
    # Reassemble and clean the filename
    sanitized = f"{prefix}_{timestamp}_{truncated_task}"
    return re.sub(r"[^\w\s-]", "", sanitized).strip()

async def generate_report_files(report: str, filename: str) -> Dict[str, str]:
    pdf_path = await write_md_to_pdf(report, filename)
    docx_path = await write_md_to_word(report, filename)
    md_path = await write_text_to_md(report, filename)
    return {"pdf": pdf_path, "docx": docx_path, "md": md_path}

async def handle_file_upload(file, DOC_PATH: str) -> Dict[str, str]:
    file_path = os.path.join(DOC_PATH, os.path.basename(file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File uploaded to {file_path}")

    document_loader = DocumentLoader(DOC_PATH)
    await document_loader.load()

    return {"filename": file.filename, "path": file_path}

async def handle_file_deletion(filename: str, DOC_PATH: str) -> JSONResponse:
    file_path = os.path.join(DOC_PATH, os.path.basename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File deleted: {file_path}")
        return JSONResponse(content={"message": "File deleted successfully"})
    else:
        print(f"File not found: {file_path}")
        return JSONResponse(status_code=404, content={"message": "File not found"})

async def execute_multi_agents(manager) -> Any:
    state = manager.active_state
    if state:
        report = await run_research_task(
            state.get("task"),
            state
        )
        return {"report": report}
    else:
        return JSONResponse(status_code=400, content={"message": "No active state"})

async def handle_state_communication(state: Dict):
    """Handle state-based communication"""
    try:
        while True:
            # Check state for new commands
            if state.get("command"):
                command = state["command"]
                
                if command == "start":
                    await handle_start_command(state, state.get("task_data"))
                elif command == "chat":
                    await handle_chat(state, state.get("chat_data"))
                
                # Clear processed command
                state["command"] = None
                await copilotkit_emit_state(state.get("config"), state)

            # Sleep briefly to prevent busy waiting
            await asyncio.sleep(0.1)

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        await copilotkit_emit_state(state.get("config"), state)

def get_config_dict(
    langchain_api_key: str, openai_api_key: str, tavily_api_key: str,
    google_api_key: str, google_cx_key: str, bing_api_key: str,
    searchapi_api_key: str, serpapi_api_key: str, serper_api_key: str, searx_url: str
) -> Dict[str, str]:
    return {
        "LANGCHAIN_API_KEY": langchain_api_key or os.getenv("LANGCHAIN_API_KEY", ""),
        "OPENAI_API_KEY": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "TAVILY_API_KEY": tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
        "GOOGLE_API_KEY": google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        "GOOGLE_CX_KEY": google_cx_key or os.getenv("GOOGLE_CX_KEY", ""),
        "BING_API_KEY": bing_api_key or os.getenv("BING_API_KEY", ""),
        "SEARCHAPI_API_KEY": searchapi_api_key or os.getenv("SEARCHAPI_API_KEY", ""),
        "SERPAPI_API_KEY": serpapi_api_key or os.getenv("SERPAPI_API_KEY", ""),
        "SERPER_API_KEY": serper_api_key or os.getenv("SERPER_API_KEY", ""),
        "SEARX_URL": searx_url or os.getenv("SEARX_URL", ""),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "DOC_PATH": os.getenv("DOC_PATH", "./my-docs"),
        "RETRIEVER": os.getenv("RETRIEVER", ""),
        "EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "")
    }

def update_environment_variables(config: Dict[str, str]):
    for key, value in config.items():
        os.environ[key] = value

def extract_command_data(json_data: Dict) -> tuple:
    return (
        json_data.get("task"),
        json_data.get("report_type"),
        json_data.get("source_urls"),
        json_data.get("document_urls"),
        json_data.get("tone"),
        json_data.get("headers", {}),
        json_data.get("report_source")
    )
