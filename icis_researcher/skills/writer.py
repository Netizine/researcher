from typing import Dict, Optional, Any
import json

from ..utils.llm import construct_subtopics
from ..actions import (
    stream_output,
    generate_report,
    generate_report_with_sections,
    generate_conclusion,
    generate_introduction,
    generate_draft_section_titles,
    write_report_introduction,
    write_conclusion
)
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config


class WriterSkill:
    """Generates reports based on research data."""

    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config
        
        # Initialize state with required fields
        self.state.setdefault("writer_logs", [])
        self.state.setdefault("costs", 0.0)
        self.state.setdefault("draft_state", {})

        self.research_params = {
            "query": state["query"],
            "agent_role_prompt": state["agent_role"] or state["role"],
            "report_type": state["report_type"],
            "report_format": state["report_format"],
            "tone": state["tone"],
            "headers": state["headers"],
        }

    async def write_report(self, context: str, cfg: Any = None) -> str:
        """Write a research report"""
        try:
            self.state["writer_logs"].append({
                "message": "Starting to write research report...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Generate report using LLM
            report = await generate_report(
                query=self.state["query"],
                context=context,
                cfg=cfg or self.state["cfg"],
                cost_callback=self.state.get("add_costs"),
                state=self.state,
                config=self.config
            )

            self.state["writer_logs"].append({
                "message": "Research report completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return report

        except Exception as e:
            self.state["writer_logs"].append({
                "message": f"Error writing report: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def write_report_with_sections(self, context: str, subtopics: list, cfg: Any = None) -> str:
        """Write a research report with sections"""
        try:
            self.state["writer_logs"].append({
                "message": "Starting to write sectioned report...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Generate report with sections
            report = await generate_report_with_sections(
                query=self.state["query"],
                context=context,
                subtopics=subtopics,
                cfg=cfg or self.state["cfg"],
                cost_callback=self.state.get("add_costs"),
                state=self.state,
                config=self.config
            )

            self.state["writer_logs"].append({
                "message": "Sectioned report completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return report

        except Exception as e:
            self.state["writer_logs"].append({
                "message": f"Error writing sectioned report: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def write_conclusion(self, report_body: str, cfg: Any = None) -> str:
        """Write a conclusion for the report"""
        try:
            self.state["writer_logs"].append({
                "message": "Writing conclusion...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Generate conclusion
            conclusion = await generate_conclusion(
                report_body=report_body,
                cfg=cfg or self.state["cfg"],
                cost_callback=self.state.get("add_costs"),
                state=self.state,
                config=self.config
            )

            self.state["writer_logs"].append({
                "message": "Conclusion completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return conclusion

        except Exception as e:
            self.state["writer_logs"].append({
                "message": f"Error writing conclusion: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def write_introduction(self, cfg: Any = None) -> str:
        """Write an introduction for the report"""
        try:
            self.state["writer_logs"].append({
                "message": "Writing introduction...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Generate introduction
            intro = await generate_introduction(
                query=self.state["query"],
                cfg=cfg or self.state["cfg"],
                cost_callback=self.state.get("add_costs"),
                state=self.state,
                config=self.config
            )

            self.state["writer_logs"].append({
                "message": "Introduction completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return intro

        except Exception as e:
            self.state["writer_logs"].append({
                "message": f"Error writing introduction: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def get_subtopics(self):
        """Retrieve subtopics for the research."""
        try:
            self.state["research_logs"].append({
                "message": "Starting to generate subtopics...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            if self.state["verbose"]:
                await stream_output(
                    "logs",
                    "generating_subtopics",
                    f"🌳 Generating subtopics for '{self.state['query']}'...",
                )

            subtopics = await construct_subtopics(
                task=self.state["query"],
                data=self.state["context"],
                config=self.state["cfg"],
                subtopics=self.state["subtopics"],
            )

            self.state["research_logs"].append({
                "message": "Subtopics generation completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            if self.state["verbose"]:
                await stream_output(
                    "logs",
                    "subtopics_generated",
                    f"📊 Subtopics generated for '{self.state['query']}'",
                )

            return subtopics

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during subtopics generation: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def get_draft_section_titles(self, current_subtopic: str):
        """Generate draft section titles for the report."""
        try:
            self.state["research_logs"].append({
                "message": "Starting to generate draft section titles...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            if self.state["verbose"]:
                await stream_output(
                    "logs",
                    "generating_draft_sections",
                    f"📑 Generating draft section titles for '{self.state['query']}'...",
                )

            draft_section_titles = await generate_draft_section_titles(
                query=self.state["query"],
                current_subtopic=current_subtopic,
                context=self.state["context"],
                role=self.state["cfg"].agent_role or self.state["role"],
                config=self.state["cfg"],
                cost_callback=self.state["add_costs"],
            )

            self.state["research_logs"].append({
                "message": "Draft section titles generation completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            if self.state["verbose"]:
                await stream_output(
                    "logs",
                    "draft_sections_generated",
                    f"🗂️ Draft section titles generated for '{self.state['query']}'",
                )

            return draft_section_titles

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during draft section titles generation: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def write_draft(self, research_data: dict) -> str:
        """Write initial draft based on research data"""
        try:
            self.state["research_logs"].append({
                "message": "Starting to write draft...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Generate outline
            outline = await self._generate_outline(research_data)
            
            # Write sections
            sections = []
            for section in outline:
                section_content = await self._write_section(section, research_data)
                sections.append(section_content)

            # Combine sections
            draft = self._combine_sections(sections)

            self.state["research_logs"].append({
                "message": "Draft completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return draft

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error writing draft: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _generate_outline(self, research_data: dict) -> list:
        """Generate outline based on research data"""
        try:
            self.state["research_logs"].append({
                "message": "Generating outline...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Use LLM to generate outline
            outline = await construct_subtopics(
                task=self.state["query"],
                data=self.state["context"],
                config=self.state["cfg"],
                subtopics=self.state["subtopics"],
            )

            self.state["research_logs"].append({
                "message": "Outline generated",
                "done": True,
                "outline": outline
            })
            await copilotkit_emit_state(self.config, self.state)

            return outline

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error generating outline: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _write_section(self, section: dict, research_data: dict) -> str:
        """Write a section of the draft"""
        try:
            self.state["research_logs"].append({
                "message": f"Writing section: {section['title']}",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Use LLM to write section
            content = await generate_draft_section_titles(
                query=self.state["query"],
                current_subtopic=section["title"],
                context=self.state["context"],
                role=self.state["cfg"].agent_role or self.state["role"],
                config=self.state["cfg"],
                cost_callback=self.state["add_costs"],
            )

            self.state["research_logs"].append({
                "message": f"Completed section: {section['title']}",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return content

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error writing section {section['title']}: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    def _combine_sections(self, sections: list) -> str:
        """Combine sections into a complete draft"""
        try:
            # Format and combine sections
            draft = "\n\n".join(sections)

            self.state["research_logs"].append({
                "message": "Combined all sections",
                "done": True
            })
            return draft

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error combining sections: {str(e)}",
                "done": True,
                "error": True
            })
            raise e
