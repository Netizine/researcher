import asyncio
from typing import List, Dict, Optional, Set
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

from ..context.compression import ContextCompressor, WrittenContentCompressor, VectorstoreCompressor
from ..actions.utils import stream_output
from datetime import datetime

class ContextManagerSkill:
    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config
        self.researcher = state["researcher"]
        
        # Initialize state with required fields
        self.state.setdefault("research_logs", [])
        self.state.setdefault("costs", 0.0)
        self.state.setdefault("context_state", {})

    async def _log_event(self, message: str, done: bool = False, error: bool = False):
        """Log a context event to state"""
        self.state["research_logs"].append({
            "message": message,
            "done": done,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        await copilotkit_emit_state(self.config, self.state)

    async def manage_context(self, content: str) -> dict:
        """Manage and update research context"""
        try:
            await self._log_event("Managing research context...")
            
            # Process and update context
            context = await self._process_content(content)
            self._update_context(context)

            await self._log_event("Research context updated", done=True)
            return context

        except Exception as e:
            await self._log_event(f"Error managing context: {str(e)}", done=True, error=True)
            raise e

    async def _process_content(self, content: str) -> dict:
        """Process content and prepare context"""
        try:
            # Process content using your existing implementation
            processed_content = await self._extract_relevant_content(content)
            
            # Format content for better readability
            prettier_content = self._format_content(processed_content)

            await self._log_event(f"Processed content: {prettier_content[:200]}...", done=True)
            return processed_content

        except Exception as e:
            await self._log_event(f"Error processing content: {str(e)}", done=True, error=True)
            raise e

    def _update_context(self, context: dict) -> None:
        """Update the research context in state"""
        if "context" not in self.state:
            self.state["context"] = []
        self.state["context"].append(context)

    async def _extract_relevant_content(self, content: str) -> dict:
        """Extract relevant content from raw content"""
        # Implement content extraction logic here
        query = content
        pages = self.researcher.memory.get_pages()
        context_compressor = ContextCompressor(
            documents=pages, embeddings=self.researcher.memory.get_embeddings()
        )
        return await context_compressor.async_get_context(
            query=query, max_results=10, cost_callback=self.researcher.add_costs
        )

    def _format_content(self, content: dict) -> str:
        """Format content for better readability"""
        # Implement content formatting logic here
        return "\n".join(content)

    async def get_similar_content_by_query_with_vectorstore(self, query, filter): 
        await self._log_event(f"Getting relevant content based on query: {query}")
        vectorstore_compressor = VectorstoreCompressor(self.researcher.vector_store, filter)
        return await vectorstore_compressor.async_get_context(query=query, max_results=8)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        all_queries = [current_subtopic] + draft_section_titles

        async def process_query(query: str) -> Set[str]:
            return set(await self.__get_similar_written_contents_by_query(query, written_contents))

        results = await asyncio.gather(*[process_query(query) for query in all_queries])
        relevant_contents = set().union(*results)
        relevant_contents = list(relevant_contents)[:max_results]

        if relevant_contents:
            prettier_contents = "\n".join(relevant_contents)
            await self._log_event(f"📃 {prettier_contents}", done=True)

        return relevant_contents

    async def __get_similar_written_contents_by_query(self,
                                                      query: str,
                                                      written_contents: List[Dict],
                                                      similarity_threshold: float = 0.5,
                                                      max_results: int = 10
                                                      ) -> List[str]:
        await self._log_event(f"🔎 Getting relevant written content based on query: {query}")
        written_content_compressor = WrittenContentCompressor(
            documents=written_contents,
            embeddings=self.researcher.memory.get_embeddings(),
            similarity_threshold=similarity_threshold
        )
        return await written_content_compressor.async_get_context(
            query=query, max_results=max_results, cost_callback=self.researcher.add_costs
        )
