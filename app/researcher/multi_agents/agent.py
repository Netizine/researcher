from app.researcher.multi_agents.agents import ChiefEditorAgent
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig

# Initialize state and config
state = AgentState(research_logs=[])
config = RunnableConfig()

# Add task configuration to state
state["task"] = {
    "query": "Is AI in a hype cycle?",
    "max_sections": 3,
    "follow_guidelines": False,
    "model": "gpt-4o",
    "guidelines": [
        "The report MUST be written in APA format",
        "Each sub section MUST include supporting sources using hyperlinks. If none exist, erase the sub section or rewrite it to be a part of the previous section",
        "The report MUST be written in spanish"
    ],
    "verbose": False
}

chief_editor = ChiefEditorAgent(state=state, config=config)
graph = chief_editor.init_research_team()
graph = graph.compile()