from typing import List, Type
from app.researcher.icis_researcher.config.config import Config

def get_retriever(retriever):
    """
    Gets the retriever
    Args:
        retriever: retriever name

    Returns:
        retriever: Retriever class

    """
    match retriever:
        case "google":
            from app.researcher.icis_researcher.retrievers import GoogleSearch

            retriever = GoogleSearch
        case "searx":
            from app.researcher.icis_researcher.retrievers import SearxSearch

            retriever = SearxSearch
        case "searchapi":
            from app.researcher.icis_researcher.retrievers import SearchApiSearch

            retriever = SearchApiSearch
        case "serpapi":
            from app.researcher.icis_researcher.retrievers import SerpApiSearch

            retriever = SerpApiSearch
        case "serper":
            from app.researcher.icis_researcher.retrievers import SerperSearch

            retriever = SerperSearch
        case "duckduckgo":
            from app.researcher.icis_researcher.retrievers import Duckduckgo

            retriever = Duckduckgo
        case "bing":
            from app.researcher.icis_researcher.retrievers import BingSearch

            retriever = BingSearch
        case "arxiv":
            from app.researcher.icis_researcher.retrievers import ArxivSearch

            retriever = ArxivSearch
        case "tavily":
            from app.researcher.icis_researcher.retrievers import TavilySearch

            retriever = TavilySearch
        case "exa":
            from app.researcher.icis_researcher.retrievers import ExaSearch

            retriever = ExaSearch
        case "semantic_scholar":
            from app.researcher.icis_researcher.retrievers import SemanticScholarSearch

            retriever = SemanticScholarSearch
        case "pubmed_central":
            from app.researcher.icis_researcher.retrievers import PubMedCentralSearch

            retriever = PubMedCentralSearch
        case "custom":
            from app.researcher.icis_researcher.retrievers import CustomRetriever

            retriever = CustomRetriever

        case _:
            retriever = None

    return retriever


def get_retrievers(headers, cfg):
    """
    Determine which retriever(s) to use based on headers, config, or default.

    Args:
        headers (dict): The headers dictionary
        cfg (Config): The configuration object

    Returns:
        list: A list of retriever classes to be used for searching.
    """
    # Check headers first for multiple retrievers
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
    # If not found, check headers for a single retriever
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
    # If not in headers, check config for multiple retrievers
    elif cfg.retrievers:
        retrievers = cfg.retrievers
    # If not found, check config for a single retriever
    elif cfg.retriever:
        retrievers = [cfg.retriever]
    # If still not set, use default retriever
    else:
        retrievers = [get_default_retriever().__name__]

    # Convert retriever names to actual retriever classes
    # Use get_default_retriever() as a fallback for any invalid retriever names
    return [get_retriever(r) or get_default_retriever() for r in retrievers]


def get_default_retriever():
    from app.researcher.icis_researcher.retrievers import TavilySearch

    return TavilySearch