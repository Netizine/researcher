import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
import uuid

from icis_researcher.utils.llm import get_llm
from icis_researcher.memory import Memory
from icis_researcher.config.config import Config

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool, tool

from icis_researcher.skills.researcher import ResearcherSkill
from icis_researcher.utils.state import AgentState, RunnableConfig
from icis_researcher.utils.messaging import copilotkit_emit_state
from icis_researcher.config import Config

class ChatAgentWithMemory:
    def __init__(self, report: str, config_path, headers, vector_store=None, state: AgentState = None, config: RunnableConfig = None):
        self.report = report
        self.headers = headers
        self.config = Config(config_path)
        self.vector_store = vector_store
        self.state = state or {}
        self.config = config
        self.graph = self.create_agent()
        self.chat = Chat(state=state, config=config)

    def create_agent(self):
        """Create React Agent Graph"""
        cfg = Config()

        # Retrieve LLM using get_llm with settings from config
        provider = get_llm(
            llm_provider=cfg.smart_llm_provider,
            model=cfg.smart_llm_model,
            temperature=0.35,
            max_tokens=cfg.smart_token_limit,
            **self.config.llm_kwargs,
        ).llm

        # If vector_store is not initialized, process documents and add to vector_store
        if not self.vector_store:
            documents = self._process_document(self.report)
            self.chat_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            self.embedding = Memory(
                cfg.embedding_provider, cfg.embedding_model, **cfg.embedding_kwargs
            ).get_embeddings()
            self.vector_store = InMemoryVectorStore(self.embedding)
            self.vector_store.add_texts(documents)

        # Create the React Agent Graph with the configured provider
        graph = create_react_agent(
            provider,
            tools=[self.vector_store_tool(self.vector_store)],
            checkpointer=MemorySaver(),
        )

        return graph

    def vector_store_tool(self, vector_store) -> Tool:
        """Create Vector Store Tool"""

        @tool
        def retrieve_info(query):
            """
            Consult the report for relevant contexts whenever you don't know something
            """
            retriever = vector_store.as_retriever(k=4)
            return retriever.invoke(query)

        return retrieve_info

    def _process_document(self, report):
        """Split Report into Chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.split_text(report)
        return documents

    async def process_message(self, message: str) -> str:
        """Process a chat message.

        Args:
            message (str): The user's message.

        Returns:
            str: The AI's response.
        """
        try:
            if self.state:
                self.state["chat_logs"].append({
                    "message": "Processing chat message...",
                    "done": False
                })
                await copilotkit_emit_state(self.config, self.state)

            # Process message
            response = await self.chat.process_message(message)

            if self.state:
                self.state["chat_logs"].append({
                    "message": "Chat message processed",
                    "done": True
                })
                self.state["chat_response"] = response
                await copilotkit_emit_state(self.config, self.state)

            return response

        except Exception as e:
            if self.state:
                self.state["chat_logs"].append({
                    "message": f"Error processing chat message: {str(e)}",
                    "done": True,
                    "error": True
                })
                await copilotkit_emit_state(self.config, self.state)
            raise e

    def get_context(self):
        """return the current context of the chat"""
        return self.chat.get_conversation_history()

class Chat:
    """Chat handler for research assistant"""

    def __init__(self, state: AgentState = None, config: RunnableConfig = None):
        """Initialize chat handler.

        Args:
            state (AgentState, optional): The state object.
            config (RunnableConfig, optional): The config object.
        """
        self.state = state or {}
        self.config = config
        self.conversation_history = []

    async def process_message(self, message: str) -> str:
        """Process a chat message.

        Args:
            message (str): The user's message.

        Returns:
            str: The AI's response.
        """
        try:
            if self.state:
                self.state["chat_logs"].append({
                    "message": "Processing chat message...",
                    "done": False
                })
                await copilotkit_emit_state(self.config, self.state)

            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })

            # Process message with researcher
            researcher = ResearcherSkill(
                query=message,
                state=self.state,
                config=self.config
            )
            response = await researcher.process_chat_message(message)

            # Add AI response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            if self.state:
                self.state["chat_logs"].append({
                    "message": "Chat message processed",
                    "done": True
                })
                self.state["chat_response"] = response
                await copilotkit_emit_state(self.config, self.state)

            return response

        except Exception as e:
            if self.state:
                self.state["chat_logs"].append({
                    "message": f"Error processing chat message: {str(e)}",
                    "done": True,
                    "error": True
                })
                await copilotkit_emit_state(self.config, self.state)
            raise e

    def get_conversation_history(self) -> list:
        """Get the conversation history.

        Returns:
            list: List of conversation messages.
        """
        return self.conversation_history
