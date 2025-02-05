import os
import time
import datetime
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from app.researcher.multi_agents.agents.utils.views import print_agent_output
from app.researcher.multi_agents.memory.research import ResearchState
from app.researcher.multi_agents.agents.utils.utils import sanitize_filename
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

# Import agent classes
from app.researcher.multi_agents.agents import \
    WriterAgent, \
    EditorAgent, \
    PublisherAgent, \
    ResearchAgent, \
    HumanAgent


class ChiefEditorAgent:
    """Agent responsible for managing and coordinating editing tasks."""

    def __init__(self, task: dict, state: AgentState, config: RunnableConfig):
        self.task = task
        self.state = state
        self.config = config
        self.output_dir = os.getenv("ICIS_OUTPUT_DIR", "outputs")
        self.tone = task.get("tone")
        self.headers = None  

        # Initialize agents with state and config
        self.agents = {
            "writer": WriterAgent(state=state, config=config),
            "editor": EditorAgent(state=state, config=config),
            "research": ResearchAgent(state=state, config=config),
            "publisher": PublisherAgent(output_dir=self.output_dir, state=state, config=config),
            "human": HumanAgent(state=state, config=config)
        }

    async def run(self):
        try:
            self.state["research_logs"].append({
                "message": f"Starting research task for query: {self.task['query']}",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Run the research task
            research_result = await self.agents["research"].run(self.task["query"])
            
            self.state["research_logs"].append({
                "message": "Research completed, generating report...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Generate the report
            report = await self.agents["writer"].write(research_result)
            
            # Publish the report
            final_report = await self.agents["publisher"].publish(report)
            
            self.state["research_logs"].append({
                "message": "Research report completed and published",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return final_report

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during research: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e
