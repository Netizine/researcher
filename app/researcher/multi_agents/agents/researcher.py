from app.researcher.icis_researcher import GPTResearcher
from colorama import Fore, Style
from app.researcher.multi_agents.agents.utils.views import print_agent_output
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config


class ResearchAgent:
    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config
        self.tone = state.get("tone")

    async def run(self, query: str, verbose: bool = False, report_source: str = None):
        self.state["research_logs"].append({
            "message": f"Running initial research on the following query: {query}",
            "done": False
        })
        await copilotkit_emit_state(self.config, self.state)

        # Run initial research
        research_results = await self.run_initial_research(query)

        # Run in-depth research for each topic
        for topic in research_results.get("topics", []):
            self.state["research_logs"].append({
                "message": f"Running in depth research on the following report topic: {topic}",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Run the in-depth research for this topic
            topic_results = await self.run_depth_research(topic)
            research_results["depth_results"].append(topic_results)

        self.state["research_logs"].append({
            "message": "Research completed",
            "done": True
        })
        await copilotkit_emit_state(self.config, self.state)

        return research_results

    async def run_initial_research(self, query: str):
        # Initialize the researcher
        researcher = GPTResearcher(query=query, report_type="research_report", verbose=False, report_source="web", tone=self.tone)
        # Conduct research on the given query
        await researcher.conduct_research()
        # Write the report
        report = await researcher.write_report()
        return report

    async def run_depth_research(self, topic: str):
        # Initialize the researcher
        researcher = GPTResearcher(query=topic, report_type="subtopic_report", verbose=False, report_source="web", tone=self.tone)
        # Conduct research on the given query
        await researcher.conduct_research()
        # Write the report
        report = await researcher.write_report()
        return report