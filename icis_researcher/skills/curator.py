from typing import Dict, Optional, List
import json
from ..config.config import Config
from ..utils.llm import create_chat_completion
from ..prompts import curate_sources as rank_sources_prompt
from ..actions import stream_output
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from datetime import datetime


class SourceCurator:
    """Ranks sources and curates data based on their relevance, credibility and reliability."""

    def __init__(self, researcher):
        self.researcher = researcher
        self.state = AgentState()
        self.config = RunnableConfig()

        # Initialize state with required fields
        self.state.setdefault("research_logs", [])
        self.state.setdefault("costs", 0.0)
        self.state.setdefault("curator_state", {})

    async def _log_event(self, message: str, done: bool = False, error: bool = False):
        """Log a curator event to state"""
        self.state["research_logs"].append({
            "message": message,
            "done": done,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        await copilotkit_emit_state(self.config, self.state)

    async def curate_sources(
        self,
        source_data: List,
        max_results: int = 10,
    ) -> List:
        """
        Rank sources based on research data and guidelines.
        
        Args:
            query: The research query/task
            source_data: List of source documents to rank
            max_results: Maximum number of top sources to return
            
        Returns:
            str: Ranked list of source URLs with reasoning
        """
        print(f"\n\nCurating {len(source_data)} sources: {source_data}")
        await self._log_event("Evaluating and curating sources by credibility and relevance...")

        response = ""
        try:
            response = await create_chat_completion(
                model=self.researcher.cfg.smart_llm_model,
                messages=[
                    {"role": "system", "content": f"{self.researcher.role}"},
                    {"role": "user", "content": rank_sources_prompt(
                        self.researcher.query, source_data, max_results)},
                ],
                temperature=0.2,
                max_tokens=8000,
                llm_provider=self.researcher.cfg.smart_llm_provider,
                llm_kwargs=self.researcher.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs,
            )

            curated_sources = json.loads(response)
            print(f"\n\nFinal Curated sources {len(source_data)} sources: {curated_sources}")

            await self._log_event(f"Curated {len(curated_sources)} sources", done=True)

            self.state["curated_sources"] = curated_sources

            await self._log_event(f"Verified and ranked top {len(curated_sources)} most reliable sources")

            return curated_sources

        except Exception as e:
            print(f"Error in curate_sources from LLM response: {response}")
            await self._log_event(f"Error curating sources: {str(e)}", done=True, error=True)
            return source_data
