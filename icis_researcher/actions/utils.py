from typing import Dict, Any, Callable
from ..utils.logger import get_formatted_logger
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
import datetime

logger = get_formatted_logger()


async def log_event(state: AgentState, config: RunnableConfig, event_type: str, message: str, metadata: dict = None) -> None:
    """Log an event to the state"""
    try:
        # Create event log
        event = {
            "type": event_type,
            "message": message,
            "done": False,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if metadata:
            event.update(metadata)

        # Add to state logs
        if "event_logs" not in state:
            state["event_logs"] = []
        state["event_logs"].append(event)

        # Emit state update
        await copilotkit_emit_state(config, state)

    except Exception as e:
        print(f"Error logging event: {str(e)}")


async def log_error(state: AgentState, config: RunnableConfig, error: Exception, context: str = None) -> None:
    """Log an error to the state"""
    try:
        # Create error log
        error_log = {
            "type": "error",
            "message": str(error),
            "done": True,
            "error": True,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if context:
            error_log["context"] = context

        # Add to state logs
        if "error_logs" not in state:
            state["error_logs"] = []
        state["error_logs"].append(error_log)

        # Emit state update
        await copilotkit_emit_state(config, state)

    except Exception as e:
        print(f"Error logging error: {str(e)}")


async def log_progress(state: AgentState, config: RunnableConfig, step: str, progress: float, message: str = None) -> None:
    """Log progress to the state"""
    try:
        # Create progress log
        progress_log = {
            "type": "progress",
            "step": step,
            "progress": progress,
            "done": progress >= 1.0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if message:
            progress_log["message"] = message

        # Add to state logs
        if "progress_logs" not in state:
            state["progress_logs"] = []
        state["progress_logs"].append(progress_log)

        # Emit state update
        await copilotkit_emit_state(config, state)

    except Exception as e:
        print(f"Error logging progress: {str(e)}")


async def log_result(state: AgentState, config: RunnableConfig, result_type: str, result: Any) -> None:
    """Log a result to the state"""
    try:
        # Create result log
        result_log = {
            "type": "result",
            "result_type": result_type,
            "result": result,
            "done": True,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Add to state logs
        if "result_logs" not in state:
            state["result_logs"] = []
        state["result_logs"].append(result_log)

        # Emit state update
        await copilotkit_emit_state(config, state)

    except Exception as e:
        print(f"Error logging result: {str(e)}")


async def stream_output(
    state: AgentState, config: RunnableConfig, type, content, output, metadata=None
):
    """
    Streams output to the state
    Args:
        state:
        config:
        type:
        content:
        output:

    Returns:
        None
    """
    if type != "images":
        try:
            await log_event(state, config, "output", output, metadata)
        except UnicodeEncodeError:
            # Option 1: Replace problematic characters with a placeholder
            await log_event(state, config, "output", output.encode(
                'cp1252', errors='replace').decode('cp1252'), metadata)


async def safe_send_json(state: AgentState, config: RunnableConfig, data: Dict[str, Any]) -> None:
    """
    Safely send JSON data through a state update.

    Args:
        state (AgentState): The state to send data through.
        config (RunnableConfig): The config to send data through.
        data (Dict[str, Any]): The data to send as JSON.

    Returns:
        None
    """
    try:
        await log_event(state, config, "json", str(data))
    except Exception as e:
        await log_error(state, config, e)


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> float:
    """
    Calculate the cost of API usage based on the number of tokens and the model used.

    Args:
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        model (str): The model used for the API call.

    Returns:
        float: The calculated cost in USD.
    """
    # Define cost per 1k tokens for different models
    costs = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.03,
        "gpt-4-32k": 0.06,
        "gpt-4o": {
            "input": 0.00015,  # $0.15 per 1M input tokens = $0.00015 per 1k tokens
            "output": 0.0006   # $0.60 per 1M output tokens = $0.0006 per 1k tokens
        },
        "gpt-4o-mini": {
            "input": 0.00015,  # $0.15 per 1M input tokens = $0.00015 per 1k tokens
            "output": 0.0006   # $0.60 per 1M output tokens = $0.0006 per 1k tokens
        }
        # Add more models and their costs as needed
    }

    model = model.lower()
    if model not in costs:
        log_error(None, None, Exception(f"Unknown model: {model}. Cost calculation may be inaccurate."))
        return 0.0

    if isinstance(costs[model], dict):
        cost_per_1k_input = costs[model]["input"]
        cost_per_1k_output = costs[model]["output"]
        total_cost = (prompt_tokens / 1000) * cost_per_1k_input + (completion_tokens / 1000) * cost_per_1k_output
    else:
        cost_per_1k = costs[model]
        total_tokens = prompt_tokens + completion_tokens
        total_cost = (total_tokens / 1000) * cost_per_1k

    return total_cost


def format_token_count(count: int) -> str:
    """
    Format the token count with commas for better readability.

    Args:
        count (int): The token count to format.

    Returns:
        str: The formatted token count.
    """
    return f"{count:,}"


async def update_cost(
    state: AgentState,
    config: RunnableConfig,
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> None:
    """
    Update and send the cost information through the state.

    Args:
        state (AgentState): The state to send data through.
        config (RunnableConfig): The config to send data through.
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        model (str): The model used for the API call.

    Returns:
        None
    """
    cost = calculate_cost(prompt_tokens, completion_tokens, model)
    total_tokens = prompt_tokens + completion_tokens

    await log_event(state, config, "cost", f"Total tokens: {format_token_count(total_tokens)}, Prompt tokens: {format_token_count(prompt_tokens)}, Completion tokens: {format_token_count(completion_tokens)}, Total cost: ${cost:.4f}")


def create_cost_callback(state: AgentState, config: RunnableConfig) -> Callable:
    """
    Create a callback function for updating costs.

    Args:
        state (AgentState): The state to send data through.
        config (RunnableConfig): The config to send data through.

    Returns:
        Callable: A callback function that can be used to update costs.
    """
    async def cost_callback(
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> None:
        await update_cost(state, config, prompt_tokens, completion_tokens, model)

    return cost_callback
