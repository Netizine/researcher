# libraries
from __future__ import annotations

import json
import logging
from typing import Optional, Any, Dict, Union, List, Callable

from colorama import Fore, Style
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from app.researcher.icis_researcher.prompts import generate_subtopics_prompt
from app.researcher.icis_researcher.utils.costs import estimate_llm_cost
from app.researcher.icis_researcher.utils.validators import Subtopics
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

def get_llm(llm_provider, **kwargs):
    from app.researcher.icis_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def call_model(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    response_format: str = None,
    state: AgentState = None,
    config: RunnableConfig = None,
    cost_callback: Callable = None,
) -> Any:
    """
    Call the LLM model with the given prompt and parameters.
    """
    try:
        if state:
            state["research_logs"].append({
                "message": "Calling language model...",
                "done": False
            })
            await copilotkit_emit_state(config, state)

        # Call model implementation here
        response = await _call_model_impl(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            cost_callback=cost_callback,
            state=state,
            config=config
        )

        if state:
            state["research_logs"].append({
                "message": "Language model call completed",
                "done": True
            })
            await copilotkit_emit_state(config, state)

        return response

    except Exception as e:
        if state:
            state["research_logs"].append({
                "message": f"Error calling language model: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(config, state)
        raise e


async def _call_model_impl(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    response_format: str = None,
    cost_callback: Callable = None,
    state: AgentState = None,
    config: RunnableConfig = None,
) -> Any:
    """Internal implementation of model calling logic"""
    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 16001:
        raise ValueError(
            f"Max tokens cannot be more than 16,000, but got {max_tokens}")

    # Get the provider from supported providers
    provider = get_llm(model=model, temperature=temperature,
                       max_tokens=max_tokens)

    response = ""
    # create response
    for _ in range(10):  # maximum of 10 attempts
        response = await provider.get_chat_response(prompt)

        if cost_callback:
            llm_costs = estimate_llm_cost(str(prompt), response)
            cost_callback(llm_costs)

        if state:
            state["llm_logs"].append({
                "message": f"Received response from {model} API",
                "done": False
            })
            await copilotkit_emit_state(config, state)

        return response

    logging.error(f"Failed to get response from {model} API")
    raise RuntimeError(f"Failed to get response from {model} API")


async def construct_subtopics(task: str, data: str, config, subtopics: list = []) -> list:
    """
    Construct subtopics based on the given task and data.

    Args:
        task (str): The main task or topic.
        data (str): Additional data for context.
        config: Configuration settings.
        subtopics (list, optional): Existing subtopics. Defaults to [].

    Returns:
        list: A list of constructed subtopics.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        print(f"\n🤖 Calling {config.smart_llm_model}...\n")

        temperature = config.temperature
        # temperature = 0 # Note: temperature throughout the code base is currently set to Zero
        provider = get_llm(
            config.smart_llm_provider,
            model=config.smart_llm_model,
            temperature=temperature,
            max_tokens=config.smart_token_limit,
            **config.llm_kwargs,
        )
        model = provider.llm

        chain = prompt | model | parser

        output = await call_model(
            prompt={
                "task": task,
                "data": data,
                "subtopics": subtopics,
                "max_subtopics": config.max_subtopics
            },
            model=config.smart_llm_model,
            max_tokens=config.smart_token_limit,
            temperature=temperature,
            state=config.state,
            config=config.runnable_config,
            cost_callback=config.cost_callback
        )

        return output

    except Exception as e:
        print("Exception in parsing subtopics : ", e)
        return subtopics
