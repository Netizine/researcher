from .utils.file_formats import \
    write_md_to_pdf, \
    write_md_to_word, \
    write_text_to_md

from .utils.views import print_agent_output
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
import os
from datetime import datetime

class PublisherAgent:
    """Agent responsible for publishing research content"""

    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def publish_research_report(self, research_state: dict, publish_formats: dict):
        layout = self.generate_layout(research_state)
        await self.write_report_by_formats(layout, publish_formats)

        return layout

    def generate_layout(self, research_state: dict):
        sections = '\n\n'.join(f"{value}"
                                 for subheader in research_state.get("research_data")
                                 for key, value in subheader.items())
        references = '\n'.join(f"{reference}" for reference in research_state.get("sources"))
        headers = research_state.get("headers")
        layout = f"""# {headers.get('title')}
#### {headers.get("date")}: {research_state.get('date')}

## {headers.get("introduction")}
{research_state.get('introduction')}

## {headers.get("table_of_contents")}
{research_state.get('table_of_contents')}

{sections}

## {headers.get("conclusion")}
{research_state.get('conclusion')}

## {headers.get("references")}
{references}
"""
        return layout

    async def write_report_by_formats(self, layout:str, publish_formats: dict):
        if publish_formats.get("pdf"):
            await write_md_to_pdf(layout, self.state.get("output_dir", "research_output"))
        if publish_formats.get("docx"):
            await write_md_to_word(layout, self.state.get("output_dir", "research_output"))
        if publish_formats.get("markdown"):
            await write_text_to_md(layout, self.state.get("output_dir", "research_output"))

    async def publish(self, report: str) -> str:
        self.state["research_logs"].append({
            "message": "Publishing final research report based on retrieved data...",
            "done": False
        })
        await copilotkit_emit_state(self.config, self.state)

        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.state.get("output_dir", "research_output"), exist_ok=True)
            
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.md"
            filepath = os.path.join(self.state.get("output_dir", "research_output"), filename)
            
            # Write report to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            
            self.state["research_logs"].append({
                "message": f"Research report published to {filepath}",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)
            
            return report
            
        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error publishing research report: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def run(self, research_state: dict):
        task = research_state.get("task")
        publish_formats = task.get("publish_formats")
        print_agent_output(output="Publishing final research report based on retrieved data...", agent="PUBLISHER")
        final_research_report = await self.publish_research_report(research_state, publish_formats)
        await self.publish(final_research_report)
        return {"report": final_research_report}
