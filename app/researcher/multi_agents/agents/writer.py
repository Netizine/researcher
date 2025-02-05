from datetime import datetime
import json5 as json
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from app.researcher.multi_agents.agents.utils.views import print_agent_output
from app.researcher.multi_agents.agents.utils.llms import call_model

sample_json = """
{
  "table_of_contents": A table of contents in markdown syntax (using '-') based on the research headers and subheaders,
  "introduction": An indepth introduction to the topic in markdown syntax and hyperlink references to relevant sources,
  "conclusion": A conclusion to the entire research based on all research data in markdown syntax and hyperlink references to relevant sources,
  "sources": A list with strings of all used source links in the entire research data in markdown syntax and apa citation format. For example: ['-  Title, year, Author [source url](source)', ...]
}
"""


class WriterAgent:
    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def write(self, research_results: dict) -> str:
        self.state["research_logs"].append({
            "message": "Starting to write research report...",
            "done": False
        })
        await copilotkit_emit_state(self.config, self.state)

        try:
            # Extract initial research and depth results
            initial_research = research_results.get("initial_research", {})
            depth_results = research_results.get("depth_results", [])

            # Combine the research into a coherent report
            report = await self._combine_research(initial_research, depth_results)

            self.state["research_logs"].append({
                "message": "Research report written successfully",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return report

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error writing research report: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _combine_research(self, initial_research: dict, depth_results: list) -> str:
        query = initial_research.get("query")
        data = initial_research.get("data")
        task = initial_research.get("task")
        follow_guidelines = task.get("follow_guidelines")
        guidelines = task.get("guidelines")

        prompt = [
            {
                "role": "system",
                "content": "You are a research writer. Your sole purpose is to write a well-written "
                "research reports about a "
                "topic based on research findings and information.\n ",
            },
            {
                "role": "user",
                "content": f"Today's date is {datetime.now().strftime('%d/%m/%Y')}\n."
                f"Query or Topic: {query}\n"
                f"Research data: {str(data)}\n"
                f"Your task is to write an in depth, well written and detailed "
                f"introduction and conclusion to the research report based on the provided research data. "
                f"Do not include headers in the results.\n"
                f"You MUST include any relevant sources to the introduction and conclusion as markdown hyperlinks -"
                f"For example: 'This is a sample text. ([url website](url))'\n\n"
                f"{f'You must follow the guidelines provided: {guidelines}' if follow_guidelines else ''}\n"
                f"You MUST return nothing but a JSON in the following format (without json markdown):\n"
                f"{sample_json}\n\n",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )

        combined_report = response.get("introduction", "") + "\n\n" + response.get("conclusion", "")

        for result in depth_results:
            combined_report += "\n\n" + result.get("content", "")

        return combined_report
