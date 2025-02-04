import asyncio
from typing import List, Optional, Dict, Any
from icis_researcher.skills.researcher import ResearcherSkill
from icis_researcher.utils.state import AgentState, RunnableConfig
from icis_researcher.utils.messaging import copilotkit_emit_state

class DetailedReport:
    """Generate a detailed research report"""

    def __init__(
        self,
        query: str,
        cfg: Any,
        state: AgentState = None,
        config: RunnableConfig = None
    ):
        """Initialize the detailed report generator.

        Args:
            query (str): The research query.
            cfg (Any): Configuration object.
            state (AgentState, optional): The state object.
            config (RunnableConfig, optional): The config object.
        """
        self.query = query
        self.cfg = cfg
        self.state = state
        self.config = config

    async def generate(self) -> str:
        """Generate the detailed report.

        Returns:
            str: The generated report.
        """
        try:
            if self.state:
                self.state["report_logs"].append({
                    "message": "Generating detailed report...",
                    "done": False
                })
                await copilotkit_emit_state(self.config, self.state)

            # Create researcher
            researcher = ResearcherSkill(
                query=self.query,
                cfg=self.cfg,
                state=self.state,
                config=self.config
            )

            # Conduct research
            report = await researcher.conduct_research()

            if self.state:
                self.state["report_logs"].append({
                    "message": "Detailed report generated",
                    "done": True
                })
                await copilotkit_emit_state(self.config, self.state)

            return report

        except Exception as e:
            if self.state:
                self.state["report_logs"].append({
                    "message": f"Error generating detailed report: {str(e)}",
                    "done": True,
                    "error": True
                })
                await copilotkit_emit_state(self.config, self.state)
            raise e
