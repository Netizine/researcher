from .utils.views import print_agent_output
from .utils.llms import call_model
import json
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

sample_revision_notes = """
{
  "draft": { 
    draft title: The revised draft that you are submitting for review 
  },
  "revision_notes": Your message to the reviewer about the changes you made to the draft based on their feedback
}
"""


class ReviserAgent:
    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def revise(self, draft_state: dict) -> dict:
        try:
            self.state["research_logs"].append({
                "message": "Starting draft revision...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            task = draft_state.get("task")
            feedback = draft_state.get("feedback")
            draft = draft_state.get("draft")

            # Create revision prompt
            prompt = [
                {
                    "role": "system",
                    "content": "You are a research editor. Your task is to revise the given draft based on the feedback provided."
                },
                {
                    "role": "user",
                    "content": f"Please revise this draft based on the following feedback:\nDraft: {draft}\nFeedback: {feedback}"
                }
            ]

            # Get revised draft
            revised_draft = await call_model(
                prompt,
                task.get("model"),
                response_format="json"
            )

            self.state["research_logs"].append({
                "message": "Draft revision completed",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            return {"draft": revised_draft}

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during draft revision: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e
