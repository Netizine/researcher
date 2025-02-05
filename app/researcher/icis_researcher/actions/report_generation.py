import asyncio
from typing import List, Dict, Any
from app.researcher.icis_researcher.config.config import Config
from app.researcher.icis_researcher.utils.llm import create_chat_completion
from app.researcher.icis_researcher.utils.logger import get_formatted_logger
from app.researcher.icis_researcher.prompts import (
    generate_report_introduction,
    generate_draft_titles_prompt,
    generate_report_conclusion,
    get_prompt_by_report_type,
)
from app.researcher.icis_researcher.utils.enum import Tone
from app.researcher.icis_researcher.utils.state import AgentState, RunnableConfig
from app.researcher.icis_researcher.utils.messaging import copilotkit_emit_state

logger = get_formatted_logger()


async def write_report_introduction(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    cost_callback: callable = None,
    state: AgentState = None,
    cfg: RunnableConfig = None
) -> str:
    """
    Generate an introduction for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        state (AgentState, optional): The state object.
        cfg (RunnableConfig, optional): The config object.

    Returns:
        str: The generated introduction.
    """
    try:
        if state:
            state["writer_logs"].append({
                "message": "Generating report introduction...",
                "done": False
            })
            await copilotkit_emit_state(cfg, state)

        introduction = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": generate_report_introduction(
                    question=query,
                    research_summary=context,
                    language=config.language
                )},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        if state:
            state["writer_logs"].append({
                "message": "Report introduction generated",
                "done": True
            })
            await copilotkit_emit_state(cfg, state)
        return introduction
    except Exception as e:
        if state:
            state["writer_logs"].append({
                "message": f"Error generating report introduction: {e}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(cfg, state)
        logger.error(f"Error in generating report introduction: {e}")
    return ""


async def write_conclusion(
    query: str,
    context: str,
    agent_role_prompt: str,
    config: Config,
    cost_callback: callable = None,
    state: AgentState = None,
    cfg: RunnableConfig = None
) -> str:
    """
    Write a conclusion for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        state (AgentState, optional): The state object.
        cfg (RunnableConfig, optional): The config object.

    Returns:
        str: The generated conclusion.
    """
    try:
        if state:
            state["writer_logs"].append({
                "message": "Generating report conclusion...",
                "done": False
            })
            await copilotkit_emit_state(cfg, state)

        conclusion = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": generate_report_conclusion(query=query,
                                                                       report_content=context,
                                                                       language=config.language)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        if state:
            state["writer_logs"].append({
                "message": "Report conclusion generated",
                "done": True
            })
            await copilotkit_emit_state(cfg, state)
        return conclusion
    except Exception as e:
        if state:
            state["writer_logs"].append({
                "message": f"Error generating report conclusion: {e}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(cfg, state)
        logger.error(f"Error in writing conclusion: {e}")
    return ""


async def summarize_url(
    url: str,
    content: str,
    role: str,
    config: Config,
    cost_callback: callable = None,
    state: AgentState = None,
    cfg: RunnableConfig = None
) -> str:
    """
    Summarize the content of a URL.

    Args:
        url (str): The URL to summarize.
        content (str): The content of the URL.
        role (str): The role of the agent.
        config (Config): Configuration object.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        state (AgentState, optional): The state object.
        cfg (RunnableConfig, optional): The config object.

    Returns:
        str: The summarized content.
    """
    try:
        if state:
            state["writer_logs"].append({
                "message": "Summarizing URL content...",
                "done": False
            })
            await copilotkit_emit_state(cfg, state)

        summary = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": f"Summarize the following content from {url}:\n\n{content}"},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        if state:
            state["writer_logs"].append({
                "message": "URL content summarized",
                "done": True
            })
            await copilotkit_emit_state(cfg, state)
        return summary
    except Exception as e:
        if state:
            state["writer_logs"].append({
                "message": f"Error summarizing URL: {e}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(cfg, state)
        logger.error(f"Error in summarizing URL: {e}")
    return ""


async def generate_draft_section_titles(
    query: str,
    current_subtopic: str,
    context: str,
    role: str,
    config: Config,
    cost_callback: callable = None,
    state: AgentState = None,
    cfg: RunnableConfig = None
) -> List[str]:
    """
    Generate draft section titles for the report.

    Args:
        query (str): The research query.
        context (str): Context for the report.
        role (str): The role of the agent.
        config (Config): Configuration object.
        cost_callback (callable, optional): Callback for calculating LLM costs.
        state (AgentState, optional): The state object.
        cfg (RunnableConfig, optional): The config object.

    Returns:
        List[str]: A list of generated section titles.
    """
    try:
        if state:
            state["writer_logs"].append({
                "message": "Generating draft section titles...",
                "done": False
            })
            await copilotkit_emit_state(cfg, state)

        section_titles = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": generate_draft_titles_prompt(
                    current_subtopic, query, context)},
            ],
            temperature=0.25,
            llm_provider=config.smart_llm_provider,
            max_tokens=config.smart_token_limit,
            llm_kwargs=config.llm_kwargs,
            cost_callback=cost_callback,
        )
        if state:
            state["writer_logs"].append({
                "message": "Draft section titles generated",
                "done": True
            })
            await copilotkit_emit_state(cfg, state)
        return section_titles.split("\n")
    except Exception as e:
        if state:
            state["writer_logs"].append({
                "message": f"Error generating draft section titles: {e}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(cfg, state)
        logger.error(f"Error in generating draft section titles: {e}")
    return []


async def generate_report(
    query: str,
    context: str,
    agent_role_prompt: str,
    report_type: str,
    tone: Tone,
    report_source: str,
    cfg: Any,
    cost_callback: callable = None,
    state: AgentState = None,
    config: RunnableConfig = None
) -> str:
    """
    generates the final report
    Args:
        query:
        context:
        agent_role_prompt:
        report_type:
        tone:
        report_source:
        cfg:
        cost_callback:
        state:
        config:

    Returns:
        report:

    """
    generate_prompt = get_prompt_by_report_type(report_type)
    report = ""

    if report_type == "subtopic_report":
        content = f"{generate_prompt(query, [], [], '', context, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words, language=cfg.language)}"
    else:
        content = f"{generate_prompt(query, context, report_source, report_format=cfg.report_format, tone=tone, total_words=cfg.total_words, language=cfg.language)}"
    try:
        if state:
            state["writer_logs"].append({
                "message": "Generating report...",
                "done": False
            })
            await copilotkit_emit_state(config, state)

        report = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": content},
            ],
            temperature=0.35,
            llm_provider=cfg.smart_llm_provider,
            max_tokens=cfg.smart_token_limit,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
        )
        if state:
            state["writer_logs"].append({
                "message": "Report generated",
                "done": True
            })
            await copilotkit_emit_state(config, state)
    except:
        try:
            report = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[
                    {"role": "user", "content": f"{agent_role_prompt}\n\n{content}"},
                ],
                temperature=0.35,
                llm_provider=cfg.smart_llm_provider,
                max_tokens=cfg.smart_token_limit,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
            )
        except Exception as e:
            if state:
                state["writer_logs"].append({
                    "message": f"Error generating report: {e}",
                    "done": True,
                    "error": True
                })
                await copilotkit_emit_state(config, state)
            print(f"Error in generate_report: {e}")

    return report
