from dotenv import load_dotenv
import sys
import os
import uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.researcher.multi_agents.agents import ChiefEditorAgent
import asyncio
import json
from app.researcher.icis_researcher.utils.enum import Tone
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

# Run with LangSmith if API key is set
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

def open_task():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to task.json
    task_json_path = os.path.join(current_dir, 'task.json')
    
    with open(task_json_path, 'r') as f:
        task = json.load(f)

    if not task:
        raise Exception("No task found. Please ensure a valid task.json file is present in the multi_agents directory and contains the necessary task information.")

    return task

async def search_node(state: AgentState, config: RunnableConfig):
    query = state.get("query")
    tone = state.get("tone", Tone.Objective)
    
    task = {
        "query": query,
        "tone": tone
    }
    
    chief_editor = ChiefEditorAgent(task, state=state, config=config)
    research_report = await chief_editor.run()
    
    state["research_logs"].append({
        "message": research_report,
        "done": True
    })
    await copilotkit_emit_state(config, state)
    
    return research_report

async def main():
    task = open_task()

    config = copilotkit_customize_config(RunnableConfig())
    state = AgentState()
    state["query"] = task["query"]
    state["tone"] = task.get("tone", Tone.Objective)
    state["research_logs"] = []

    research_report = await search_node(state, config)

    return research_report

if __name__ == "__main__":
    asyncio.run(main())