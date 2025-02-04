from typing import Optional, List, Dict, Any, Set
import json

from .config import Config
from .memory import Memory
from .utils.enum import ReportSource, ReportType, Tone
from .llm_provider import GenericLLMProvider
from .vector_store import VectorStoreWrapper

# Research skills
from .skills.researcher import ResearchConductor
from .skills.writer import ReportGenerator
from .skills.context_manager import ContextManager
from .skills.browser import BrowserManager
from .skills.curator import SourceCurator

from .actions import (
    add_references,
    extract_headers,
    extract_sections,
    table_of_contents,
    get_retrievers,
    choose_agent
)

from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

class GPTResearcher:
    """Main researcher agent that coordinates the research process."""

    def __init__(
        self,
        state: AgentState,
        config: RunnableConfig,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_format: str = "markdown",
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        source_urls=None,
        document_urls=None,
        complement_source_urls=False,
        documents=None,
        vector_store=None,
        vector_store_filter=None,
        config_path=None,
    ):
        self.state = state
        self.config = config
        
        # Initialize state with research parameters
        self.state.setdefault("query", query)
        self.state.setdefault("report_type", report_type)
        self.state.setdefault("report_format", report_format)
        self.state.setdefault("report_source", report_source)
        self.state.setdefault("tone", tone)
        self.state.setdefault("source_urls", source_urls or [])
        self.state.setdefault("document_urls", document_urls or [])
        self.state.setdefault("complement_source_urls", complement_source_urls)
        self.state.setdefault("documents", documents or [])
        self.state.setdefault("vector_store", vector_store)
        self.state.setdefault("vector_store_filter", vector_store_filter)
        self.state.setdefault("research_logs", [])
        self.state.setdefault("verbose", True)
        self.state.setdefault("costs", 0.0)
        self.state.setdefault("research_sources", [])
        self.state.setdefault("research_images", [])

        # Load configuration
        self.cfg = Config(config_path) if config_path else Config()
        self.state["cfg"] = self.cfg

        # Initialize skills with state and config
        self.research_conductor: ResearchConductor = ResearchConductor(self)
        self.report_generator: ReportGenerator = ReportGenerator(self)
        self.context_manager: ContextManager = ContextManager(self)
        self.scraper_manager: BrowserManager = BrowserManager(self)
        self.source_curator: SourceCurator = SourceCurator(self)

        # Initialize LLM and memory
        self.llm = GenericLLMProvider(self.cfg)
        self.memory = Memory(
            self.cfg.embedding_provider, self.cfg.embedding_model, **self.cfg.embedding_kwargs
        )

        # Initialize retrievers
        self.retrievers = get_retrievers(self.state["cfg"], self.cfg)

    async def _log_event(self, event_type: str, **kwargs):
        """Helper method to handle logging events"""
        if self.state["verbose"]:
            try:
                if event_type == "tool":
                    await self.state["log_handler"].on_tool_start(kwargs.get('tool_name', ''), **kwargs)
                elif event_type == "action":
                    await self.state["log_handler"].on_agent_action(kwargs.get('action', ''), **kwargs)
                elif event_type == "research":
                    await self.state["log_handler"].on_research_step(kwargs.get('step', ''), kwargs.get('details', {}))
                
                # Add direct logging as backup
                import logging
                research_logger = logging.getLogger('research')
                research_logger.info(f"{event_type}: {json.dumps(kwargs, default=str)}")
                
            except Exception as e:
                import logging
                logging.getLogger('research').error(f"Error in _log_event: {e}", exc_info=True)

    async def conduct_research(self):
        await self._log_event("research", step="start", details={
            "query": self.state["query"],
            "report_type": self.state["report_type"],
            "agent": self.state["agent"],
            "role": self.state["role"]
        })

        if not (self.state["agent"] and self.state["role"]):
            await self._log_event("action", action="choose_agent")
            self.state["agent"], self.state["role"] = await choose_agent(
                query=self.state["query"],
                cfg=self.cfg,
                parent_query=self.state["parent_query"],
                cost_callback=self.add_costs,
                headers=self.state["headers"],
            )
            await self._log_event("action", action="agent_selected", details={
                "agent": self.state["agent"],
                "role": self.state["role"]
            })

        await self._log_event("research", step="conducting_research", details={
            "agent": self.state["agent"],
            "role": self.state["role"]
        })
        self.state["context"] = await self.research_conductor.conduct_research()
        
        await self._log_event("research", step="research_completed", details={
            "context_length": len(self.state["context"])
        })
        await copilotkit_emit_state(self.config, self.state)
        return self.state["context"]

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None) -> str:
        await self._log_event("research", step="writing_report", details={
            "existing_headers": existing_headers,
            "context_source": "external" if ext_context else "internal"
        })
        
        report = await self.report_generator.write_report(
            existing_headers,
            relevant_written_contents,
            ext_context or self.state["context"]
        )
        
        await self._log_event("research", step="report_completed", details={
            "report_length": len(report)
        })
        await copilotkit_emit_state(self.config, self.state)
        return report

    async def write_report_conclusion(self, report_body: str) -> str:
        await self._log_event("research", step="writing_conclusion")
        conclusion = await self.report_generator.write_report_conclusion(report_body)
        await self._log_event("research", step="conclusion_completed")
        await copilotkit_emit_state(self.config, self.state)
        return conclusion

    async def write_introduction(self):
        await self._log_event("research", step="writing_introduction")
        intro = await self.report_generator.write_introduction()
        await self._log_event("research", step="introduction_completed")
        await copilotkit_emit_state(self.config, self.state)
        return intro

    async def get_subtopics(self):
        return await self.report_generator.get_subtopics()

    async def get_draft_section_titles(self, current_subtopic: str):
        return await self.report_generator.get_draft_section_titles(current_subtopic)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        return await self.context_manager.get_similar_written_contents_by_draft_section_titles(
            current_subtopic,
            draft_section_titles,
            written_contents,
            max_results
        )

    # Utility methods
    def get_research_images(self, top_k=10) -> List[Dict[str, Any]]:
        return self.state["research_images"][:top_k]

    def add_research_images(self, images: List[Dict[str, Any]]) -> None:
        self.state["research_images"].extend(images)

    def get_research_sources(self) -> List[Dict[str, Any]]:
        return self.state["research_sources"]

    def add_research_sources(self, sources: List[Dict[str, Any]]) -> None:
        self.state["research_sources"].extend(sources)

    def add_references(self, report_markdown: str, visited_urls: set) -> str:
        return add_references(report_markdown, visited_urls)

    def extract_headers(self, markdown_text: str) -> List[Dict]:
        return extract_headers(markdown_text)

    def extract_sections(self, markdown_text: str) -> List[Dict]:
        return extract_sections(markdown_text)

    def table_of_contents(self, markdown_text: str) -> str:
        return table_of_contents(markdown_text)

    def get_source_urls(self) -> list:
        return list(self.state["visited_urls"])

    def get_research_context(self) -> list:
        return self.state["context"]

    def get_costs(self) -> float:
        return self.state["costs"]

    def set_verbose(self, verbose: bool):
        self.state["verbose"] = verbose

    def add_costs(self, cost: float) -> None:
        if not isinstance(cost, (float, int)):
            raise ValueError("Cost must be an integer or float")
        self.state["costs"] += cost
        if self.state["log_handler"]:
            self._log_event("research", step="cost_update", details={
                "cost": cost,
                "total_cost": self.state["costs"]
            })
        await copilotkit_emit_state(self.config, self.state)
