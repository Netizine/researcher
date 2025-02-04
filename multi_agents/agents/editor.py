from datetime import datetime
import asyncio
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END

from .utils.views import print_agent_output
from .utils.llms import call_model
from ..memory.draft import DraftState
from . import ResearchAgent, ReviewerAgent, ReviserAgent
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config


class EditorAgent:
    """Agent responsible for editing and managing code."""

    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def plan_research(self, research_state: Dict[str, any]) -> Dict[str, any]:
        """
        Plan the research outline based on initial research and task parameters.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with title, date, and planned sections
        """
        try:
            self.state["research_logs"].append({
                "message": "Planning research outline...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            initial_research = research_state.get("initial_research")
            task = research_state.get("task")
            include_human_feedback = task.get("include_human_feedback")
            human_feedback = research_state.get("human_feedback")
            max_sections = task.get("max_sections")

            prompt = self._create_planning_prompt(
                initial_research, include_human_feedback, human_feedback, max_sections)

            plan = await call_model(
                prompt=prompt,
                model=task.get("model"),
                response_format="json",
            )

            self.state["research_logs"].append({
                "message": f"Research outline generated: {plan}",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return {
                "title": plan.get("title"),
                "date": plan.get("date"),
                "sections": plan.get("sections"),
            }

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error planning research: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def run_parallel_research(self, research_state: Dict[str, any]) -> Dict[str, List[str]]:
        """
        Execute parallel research tasks for each section.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with research results
        """
        try:
            task = research_state.get("task")
            sections = research_state.get("sections", [])

            self.state["research_logs"].append({
                "message": f"Running parallel research for {len(sections)} sections",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            agents = self._initialize_agents()
            workflow = self._create_workflow()
            chain = workflow.compile()

            queries = research_state.get("sections")
            title = research_state.get("title")

            self._log_parallel_research(queries)

            final_drafts = [
                chain.ainvoke(self._create_task_input(
                    research_state, query, title))
                for query in queries
            ]
            research_results = [
                result["draft"] for result in await asyncio.gather(*final_drafts)
            ]

            self.state["research_logs"].append({
                "message": "All parallel research completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return {"research_data": research_results}

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during parallel research: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def edit_draft(self, draft: str) -> str:
        """Edit and refine a draft"""
        try:
            self.state["editor_logs"].append({
                "message": "Starting to edit draft...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Analyze and improve draft
            analysis = await self._analyze_draft(draft)
            improved_draft = await self._improve_draft(draft, analysis)
            final_draft = await self._final_review(improved_draft)

            self.state["editor_logs"].append({
                "message": "Draft editing completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return final_draft

        except Exception as e:
            self.state["editor_logs"].append({
                "message": f"Error editing draft: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _analyze_draft(self, draft: str) -> dict:
        """Analyze draft for improvements"""
        try:
            self.state["editor_logs"].append({
                "message": "Analyzing draft...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Use LLM to analyze draft
            analysis = await self.state["llm"].analyze_draft(draft)

            self.state["editor_logs"].append({
                "message": "Draft analysis completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return analysis

        except Exception as e:
            self.state["editor_logs"].append({
                "message": f"Error analyzing draft: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _improve_draft(self, draft: str, analysis: dict) -> str:
        """Make improvements based on analysis"""
        try:
            self.state["editor_logs"].append({
                "message": "Improving draft...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Use LLM to improve draft
            improved_draft = await self.state["llm"].improve_draft(draft, analysis)

            self.state["editor_logs"].append({
                "message": "Draft improvements completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return improved_draft

        except Exception as e:
            self.state["editor_logs"].append({
                "message": f"Error improving draft: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _final_review(self, draft: str) -> str:
        """Perform final review and polish"""
        try:
            self.state["editor_logs"].append({
                "message": "Performing final review...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Use LLM for final review
            final_draft = await self.state["llm"].final_review(draft)

            self.state["editor_logs"].append({
                "message": "Final review completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return final_draft

        except Exception as e:
            self.state["editor_logs"].append({
                "message": f"Error in final review: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    def _create_planning_prompt(self, initial_research: str, include_human_feedback: bool,
                                human_feedback: Optional[str], max_sections: int) -> List[Dict[str, str]]:
        """Create the prompt for research planning."""
        return [
            {
                "role": "system",
                "content": "You are a research editor. Your goal is to oversee the research project "
                           "from inception to completion. Your main task is to plan the article section "
                           "layout based on an initial research summary.\n ",
            },
            {
                "role": "user",
                "content": self._format_planning_instructions(initial_research, include_human_feedback,
                                                              human_feedback, max_sections),
            },
        ]

    def _format_planning_instructions(self, initial_research: str, include_human_feedback: bool,
                                      human_feedback: Optional[str], max_sections: int) -> str:
        """Format the instructions for research planning."""
        today = datetime.now().strftime('%d/%m/%Y')
        feedback_instruction = (
            f"Human feedback: {human_feedback}. You must plan the sections based on the human feedback."
            if include_human_feedback and human_feedback and human_feedback != 'no'
            else ''
        )

        return f"""Today's date is {today}
                   Research summary report: '{initial_research}'
                   {feedback_instruction}
                   \nYour task is to generate an outline of sections headers for the research project
                   based on the research summary report above.
                   You must generate a maximum of {max_sections} section headers.
                   You must focus ONLY on related research topics for subheaders and do NOT include introduction, conclusion and references.
                   You must return nothing but a JSON with the fields 'title' (str) and 
                   'sections' (maximum {max_sections} section headers) with the following structure:
                   '{{title: string research title, date: today's date, 
                   sections: ['section header 1', 'section header 2', 'section header 3' ...]}}'."""

    def _initialize_agents(self) -> Dict[str, any]:
        """Initialize the research, reviewer, and reviser skills."""
        return {
            "research": ResearchAgent(state=self.state, config=self.config),
            "reviewer": ReviewerAgent(state=self.state, config=self.config),
            "reviser": ReviserAgent(state=self.state, config=self.config),
        }

    def _create_workflow(self) -> StateGraph:
        """Create the workflow for the research process."""
        agents = self._initialize_agents()
        workflow = StateGraph(DraftState)

        workflow.add_node("researcher", agents["research"].run_depth_research)
        workflow.add_node("reviewer", agents["reviewer"].run)
        workflow.add_node("reviser", agents["reviser"].run)

        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "reviewer")
        workflow.add_edge("reviser", "reviewer")
        workflow.add_conditional_edges(
            "reviewer",
            lambda draft: "accept" if draft["review"] is None else "revise",
            {"accept": END, "revise": "reviser"},
        )

        return workflow

    def _log_parallel_research(self, queries: List[str]) -> None:
        """Log the start of parallel research tasks."""
        print_agent_output(
            f"Running the following research tasks in parallel: {queries}...",
            agent="EDITOR",
        )

    def _create_task_input(self, research_state: Dict[str, any], query: str, title: str) -> Dict[str, any]:
        """Create the input for a single research task."""
        return {
            "task": research_state.get("task"),
            "topic": query,
            "title": title,
            "headers": self.headers,
        }
