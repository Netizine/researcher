import asyncio
import random
import json
from typing import Dict, Optional, Any
import logging
from datetime import datetime

from app.researcher.icis_researcher.actions.utils import stream_output
from app.researcher.icis_researcher.actions.query_processing import plan_research_outline, get_search_results
from app.researcher.icis_researcher.document import DocumentLoader, OnlineDocumentLoader, LangChainDocumentLoader
from app.researcher.icis_researcher.utils.enum import ReportSource, ReportType, Tone
from app.researcher.icis_researcher.utils.logging_config import get_json_handler, get_research_logger
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

class ResearchSkill:
    """Manages and coordinates the research process."""

    def __init__(self, state: AgentState, config: RunnableConfig, researcher):
        self.state = state
        self.config = config
        self.researcher = researcher
        self.logger = logging.getLogger("research")
        self.json_handler = get_json_handler()

        # Initialize state with required fields
        self.state.setdefault("researcher_logs", [])
        self.state.setdefault("costs", 0.0)
        self.state.setdefault("research_state", {})

    async def _log_event(self, message: str, done: bool = False, error: bool = False):
        """Log a research event to state"""
        self.state["researcher_logs"].append({
            "message": message,
            "done": done,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        await copilotkit_emit_state(self.config, self.state)

    async def conduct_research(self, query: str) -> dict:
        """Conduct research on a given query"""
        try:
            await self._log_event(f"Starting research for query: {query}")

            # Initialize research state
            if "research_state" not in self.state:
                self.state["research_state"] = {
                    "query": query,
                    "visited_urls": set(),
                    "research_summary": [],
                    "research_depth": 0,
                    "max_depth": self.state.get("max_research_depth", 3)
                }

            # Perform initial search
            search_results = await self._perform_search(query)
            
            # Process and analyze search results
            research_data = await self._process_search_results(search_results)
            
            # Update research state
            self.state["research_state"]["research_summary"].extend(research_data)

            await self._log_event(f"Research completed for query: {query}", done=True)
            return research_data

        except Exception as e:
            await self._log_event(f"Error during research: {str(e)}", done=True, error=True)
            raise e

    async def _perform_search(self, query: str) -> list:
        """Perform web search and retrieve results"""
        try:
            await self._log_event("Searching for relevant sources...")
            
            # Get search results using retrievers
            search_results = []
            for retriever in self.state.get("retrievers", []):
                results = await retriever.search(query)
                search_results.extend(results)

            await self._log_event("Source search completed", done=True)
            return search_results

        except Exception as e:
            await self._log_event(f"Error searching sources: {str(e)}", done=True, error=True)
            raise e

    async def _process_search_results(self, search_results: list) -> list:
        """Process and analyze search results"""
        try:
            await self._log_event("Processing sources...")
            
            processed_results = []
            for result in search_results:
                # Skip if URL already visited
                if result["url"] in self.state["research_state"]["visited_urls"]:
                    continue

                # Extract and analyze content
                content = await self._extract_content(result["url"])
                analysis = await self._analyze_content(content)
                
                processed_results.append({
                    "url": result["url"],
                    "content": content,
                    "analysis": analysis
                })

                # Mark URL as visited
                self.state["research_state"]["visited_urls"].add(result["url"])

            await self._log_event("Source processing completed", done=True)
            return processed_results

        except Exception as e:
            await self._log_event(f"Error processing sources: {str(e)}", done=True, error=True)
            raise e

    async def _extract_content(self, url: str) -> str:
        """Extract content from a URL"""
        try:
            # Use browser skill to extract content
            browser = self.state.get("skills", {}).get("browser")
            if browser:
                return await browser.browse(url)
            return ""

        except Exception as e:
            await self._log_event(f"Error extracting content from {url}: {str(e)}", done=True, error=True)
            raise e

    async def _analyze_content(self, content: str) -> dict:
        """Analyze extracted content"""
        try:
            # Use LLM to analyze content
            llm = self.state.get("llm")
            if llm:
                analysis = await llm.analyze(content)
                return analysis
            return {}

        except Exception as e:
            await self._log_event(f"Error analyzing content: {str(e)}", done=True, error=True)
            raise e

    async def plan_research(self, query):
        await self._log_event(f"Planning research for query: {query}")

        try:
            await stream_output(
                "logs",
                "planning_research",
                f"🌐 Browsing the web to learn more about the task: {query}...",
                None,
            )

            search_results = await get_search_results(query, self.researcher.retrievers[0])
            self.logger.info(
                f"Initial search results obtained: {len(search_results)} results"
            )

            await stream_output(
                "logs",
                "planning_research",
                f"🤔 Planning the research strategy and subtasks...",
                None,
            )

            outline = await plan_research_outline(
                query=query,
                search_results=search_results,
                agent_role_prompt=self.researcher.role,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                report_type=self.researcher.report_type,
                cost_callback=self.researcher.add_costs,
            )
            self.logger.info(f"Research outline planned: {outline}")
            await self._log_event(f"Research outline planned: {outline}", done=True)
            return outline
        except Exception as e:
            await self._log_event(f"Error planning research: {str(e)}", done=True, error=True)
            raise e

    async def _get_context_by_urls(self, urls):
        """Scrapes and compresses the context from the given urls"""
        await self._log_event(f"Getting context from URLs: {urls}")

        try:
            self.logger.info(f"Getting context from URLs: {urls}")

            new_search_urls = await self._get_new_urls(urls)
            self.logger.info(f"New URLs to process: {new_search_urls}")

            scraped_content = await self.researcher.scraper_manager.browse_urls(
                new_search_urls
            )
            self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

            if self.researcher.vector_store:
                self.logger.info("Loading content into vector store")
                self.researcher.vector_store.load(scraped_content)

            context = await self.researcher.context_manager.get_similar_content_by_query(
                self.researcher.query, scraped_content
            )
            self.logger.info(f"Generated context length: {len(context)}")
            await self._log_event(f"Generated context length: {len(context)}", done=True)
            return context
        except Exception as e:
            await self._log_event(f"Error getting context from URLs: {str(e)}", done=True, error=True)
            raise e

    async def _get_context_by_vectorstore(self, query, filter: Optional[dict] = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        await self._log_event(f"Searching vectorstore for query: {query}")

        try:
            context = []
            # Generate Sub-Queries including original query
            sub_queries = await self.plan_research(query)
            # If this is not part of a sub researcher, add original query to research for better results
            if self.researcher.report_type != "subtopic_report":
                sub_queries.append(query)

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subqueries",
                    f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                    None,
                    True,
                    sub_queries,
                )

            # Using asyncio.gather to process the sub_queries asynchronously
            context = await asyncio.gather(
                *[
                    self._process_sub_query_with_vectorstore(sub_query, filter)
                    for sub_query in sub_queries
                ]
            )
            await self._log_event("Vectorstore search completed", done=True)
            return context
        except Exception as e:
            await self._log_event(f"Error searching vectorstore: {str(e)}", done=True, error=True)
            raise e

    async def _get_context_by_web_search(self, query, scraped_data: list = []):
        """Generates the context for the research task by searching the query and scraping the results"""
        await self._log_event(f"Starting web search for query: {query}")

        try:
            self.logger.info(f"Starting web search for query: {query}")

            # Generate Sub-Queries including original query
            sub_queries = await self.plan_research(query)
            self.logger.info(f"Generated sub-queries: {sub_queries}")

            # If this is not part of a sub researcher, add original query to research for better results
            if self.researcher.report_type != "subtopic_report":
                sub_queries.append(query)

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subqueries",
                    f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                    None,
                    True,
                    sub_queries,
                )

            # Using asyncio.gather to process the sub_queries asynchronously
            try:
                context = await asyncio.gather(
                    *[
                        self._process_sub_query(sub_query, scraped_data)
                        for sub_query in sub_queries
                    ]
                )
                self.logger.info(f"Gathered context from {len(context)} sub-queries")
                # Filter out empty results and join the context
                context = [c for c in context if c]
                if context:
                    combined_context = " ".join(context)
                    self.logger.info(f"Combined context size: {len(combined_context)}")
                    await self._log_event(f"Combined context size: {len(combined_context)}", done=True)
                    return combined_context
                return []
            except Exception as e:
                await self._log_event(f"Error during web search: {str(e)}", done=True, error=True)
                raise e
        except Exception as e:
            await self._log_event(f"Error starting web search: {str(e)}", done=True, error=True)
            raise e

    async def _process_sub_query(self, sub_query: str, scraped_data: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        await self._log_event(f"Processing sub query: {sub_query}")

        try:
            if self.json_handler:
                self.json_handler.log_event(
                    "sub_query",
                    {"query": sub_query, "scraped_data_size": len(scraped_data)},
                )

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "running_subquery_research",
                    f"\n🔍 Running research for '{sub_query}'...",
                    None,
                )

            try:
                if not scraped_data:
                    scraped_data = await self._scrape_data_by_urls(sub_query)
                    self.logger.info(f"Scraped data size: {len(scraped_data)}")

                content = (
                    await self.researcher.context_manager.get_similar_content_by_query(
                        sub_query, scraped_data
                    )
                )
                self.logger.info(
                    f"Content found for sub-query: {len(str(content)) if content else 0} chars"
                )

                if content and self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_window",
                        f"📃 {content}",
                        None,
                    )
                elif self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        None,
                    )
                if content:
                    if self.json_handler:
                        self.json_handler.log_event(
                            "content_found",
                            {"sub_query": sub_query, "content_size": len(content)},
                        )
                await self._log_event(f"Content found for sub query: {sub_query}", done=True)
                return content
            except Exception as e:
                await self._log_event(f"Error processing sub query: {str(e)}", done=True, error=True)
                raise e
        except Exception as e:
            await self._log_event(f"Error processing sub query: {str(e)}", done=True, error=True)
            raise e

    async def _process_sub_query_with_vectorstore(
        self, sub_query: str, filter: Optional[dict] = None
    ):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        await self._log_event(f"Processing sub query with vectorstore: {sub_query}")

        try:
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "running_subquery_with_vectorstore_research",
                    f"\n🔍 Running research for '{sub_query}'...",
                    None,
                )

            content = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(
                sub_query, filter
            )

            if content and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_context_window",
                    f"📃 {content}",
                    None,
                )
            elif self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_context_not_found",
                    f"🤷 No content found for '{sub_query}'...",
                    None,
                )
            await self._log_event(f"Content found for sub query with vectorstore: {sub_query}", done=True)
            return content
        except Exception as e:
            await self._log_event(f"Error processing sub query with vectorstore: {str(e)}", done=True, error=True)
            raise e

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """
        await self._log_event(f"Getting new URLs from: {url_set_input}")

        try:
            new_urls = []
            for url in url_set_input:
                if url not in self.researcher.visited_urls:
                    self.researcher.visited_urls.add(url)
                    new_urls.append(url)
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "added_source_url",
                            f"✅ Added source url to research: {url}\n",
                            None,
                            True,
                            url,
                        )

            await self._log_event(f"New URLs: {new_urls}", done=True)
            return new_urls
        except Exception as e:
            await self._log_event(f"Error getting new URLs: {str(e)}", done=True, error=True)
            raise e

    async def _search_relevant_source_urls(self, query):
        await self._log_event(f"Searching for relevant source URLs: {query}")

        try:
            new_search_urls = []

            # Iterate through all retrievers
            for retriever_class in self.researcher.retrievers:
                # Instantiate the retriever with the sub-query
                retriever = retriever_class(query)

                # Perform the search using the current retriever
                search_results = await asyncio.to_thread(
                    retriever.search,
                    max_results=self.researcher.cfg.max_search_results_per_query,
                )

                # Collect new URLs from search results
                search_urls = [url.get("href") for url in search_results]
                new_search_urls.extend(search_urls)

            # Get unique URLs
            new_search_urls = await self._get_new_urls(new_search_urls)
            random.shuffle(new_search_urls)

            await self._log_event(f"Relevant source URLs: {new_search_urls}", done=True)
            return new_search_urls
        except Exception as e:
            await self._log_event(f"Error searching for relevant source URLs: {str(e)}", done=True, error=True)
            raise e

    async def _scrape_data_by_urls(self, sub_query):
        """Runs a sub-query across multiple retrievers and scrapes the resulting URLs.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        await self._log_event(f"Scraping data by URLs: {sub_query}")

        try:
            new_search_urls = await self._search_relevant_source_urls(sub_query)

            # Log the research process if verbose mode is on
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "researching",
                    f"🤔 Researching for relevant information across multiple sources...\n",
                    None,
                )

            # Scrape the new URLs
            scraped_content = await self.researcher.scraper_manager.browse_urls(
                new_search_urls
            )

            if self.researcher.vector_store:
                self.researcher.vector_store.load(scraped_content)

            await self._log_event(f"Scraped data: {scraped_content}", done=True)
            return scraped_content
        except Exception as e:
            await self._log_event(f"Error scraping data by URLs: {str(e)}", done=True, error=True)
            raise e
