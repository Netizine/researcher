from typing import List, Optional, Dict, Any
from icis_researcher.skills.researcher import ResearcherSkill
from icis_researcher.utils.state import AgentState, RunnableConfig
from icis_researcher.utils.messaging import copilotkit_emit_state

class BasicReport:
    """Generate a basic research report"""

    def __init__(
        self,
        query: str,
        cfg: Any,
        state: AgentState = None,
        config: RunnableConfig = None,
        report_type: str = None,
        report_source: str = None,
        source_urls=None,
        document_urls=None,
        tone: Any = None,
        headers=None
    ):
        """Initialize the basic report generator.

        Args:
            query (str): The research query.
            cfg (Any): Configuration object.
            state (AgentState, optional): The state object.
            config (RunnableConfig, optional): The config object.
            report_type (str, optional): The report type.
            report_source (str, optional): The report source.
            source_urls (optional): The source URLs.
            document_urls (optional): The document URLs.
            tone (Any, optional): The tone.
            headers (optional): The headers.
        """
        self.query = query
        self.cfg = cfg
        self.state = state
        self.config = config
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.tone = tone
        self.headers = headers or {}

    async def generate(self) -> str:
        """Generate the basic report.

        Returns:
            str: The generated report.
        """
        try:
            if self.state:
                self.state["report_logs"].append({
                    "message": "Generating basic report...",
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
                    "message": "Basic report generated",
                    "done": True
                })
                await copilotkit_emit_state(self.config, self.state)

            return report

        except Exception as e:
            if self.state:
                self.state["report_logs"].append({
                    "message": f"Error generating basic report: {str(e)}",
                    "done": True,
                    "error": True
                })
                await copilotkit_emit_state(self.config, self.state)
            raise e
