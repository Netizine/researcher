from app.researcher.multi_agents.agents.utils.views import print_agent_output
from app.researcher.multi_agents.agents.utils.llms import call_model
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

TEMPLATE = """You are an expert research article reviewer. \
Your goal is to review research drafts and provide feedback to the reviser only based on specific guidelines. \
"""


class ReviewerAgent:
    """Agent responsible for reviewing research content"""

    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def review_draft(self, draft_state: dict) -> dict:
        """Review a draft article"""
        try:
            self.state["research_logs"].append({
                "message": "Starting draft review...",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            task = draft_state.get("task")
            guidelines = "- ".join(guideline for guideline in task.get("guidelines"))
            revision_notes = draft_state.get("revision_notes")

            revise_prompt = f"""The reviser has already revised the draft based on your previous review notes with the following feedback:
{revision_notes}\n
Please provide additional feedback ONLY if critical since the reviser has already made changes based on your previous feedback.
If you think the article is sufficient or that non critical revisions are required, please aim to return None.
"""

            review_prompt = f"""You have been tasked with reviewing the draft which was written by a non-expert based on specific guidelines.
Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision.
If not all of the guideline criteria are met, you should send appropriate revision notes.
If the draft meets all the guidelines, please return None.
{revise_prompt if revision_notes else ""}

Guidelines: {guidelines}\nDraft: {draft_state.get("draft")}\n
"""
            prompt = [
                {"role": "system", "content": TEMPLATE},
                {"role": "user", "content": review_prompt},
            ]

            response = await call_model(prompt, model=task.get("model"))

            self.state["research_logs"].append({
                "message": "Draft review completed.",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            if task.get("verbose"):
                print_agent_output(
                    f"Review feedback is: {response}...", agent="REVIEWER"
                )

            if "None" in response:
                return None
            return response

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error during draft review: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def run(self, draft_state: dict):
        task = draft_state.get("task")
        guidelines = task.get("guidelines")
        to_follow_guidelines = task.get("follow_guidelines")
        review = None
        if to_follow_guidelines:
            print_agent_output(f"Reviewing draft...", agent="REVIEWER")

            if task.get("verbose"):
                print_agent_output(
                    f"Following guidelines {guidelines}...", agent="REVIEWER"
                )

            review = await self.review_draft(draft_state)
        else:
            print_agent_output(f"Ignoring guidelines...", agent="REVIEWER")
        return {"review": review}
