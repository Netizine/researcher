import json
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config


class HumanAgent:
    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def review_plan(self, research_state: dict) -> dict:
        """
        Reviews the research plan and allows for human feedback.
        """
        try:
            self.state["research_logs"].append({
                "message": "Requesting human review of research plan...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # In automated mode, just proceed without human feedback
            if not self.state.get("interactive", False):
                self.state["research_logs"].append({
                    "message": "Automated mode: Proceeding without human feedback",
                    "done": True
                })
                await copilotkit_emit_state(self.config, self.state)
                return {"human_feedback": None}

            # Request human feedback
            self.state["research_logs"].append({
                "message": "Please review the research plan and provide feedback.\nType 'ok' to proceed or provide feedback to revise.",
                "done": False,
                "requires_input": True
            })
            await copilotkit_emit_state(self.config, self.state)

            # Wait for human input through state updates
            # Note: The actual implementation of waiting for input would depend on your frontend
            # For now, we'll simulate accepting the plan
            
            self.state["research_logs"].append({
                "message": "Research plan accepted",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)
            
            return {"human_feedback": None}

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during human review: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e
