"""
tools.py
========
Defines individual tool functions for the ReAct agent.

Each tool is wrapped with Pydantic input validation and includes
a clear docstring for the LLM to understand its purpose.

Available tools:
    - Arxiv Search: Query academic papers on arXiv
    - Wikipedia Search: Look up information on Wikipedia
    - Tavily Web Search: Search the internet for recent information
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Input Schemas
# ---------------------------------------------------------------------------

class ArxivInput(BaseModel):
    """Input schema for the Arxiv search tool."""

    query: str = Field(
        ...,
        description=(
            "A search query for arXiv papers. This can be a paper ID "
            "(e.g., '1706.03762'), a topic (e.g., 'quantum computing'), "
            "or author names."
        ),
    )


class WikipediaInput(BaseModel):
    """Input schema for the Wikipedia search tool."""

    query: str = Field(
        ...,
        description=(
            "A search query for Wikipedia. Use clear, specific terms "
            "for the best results (e.g., 'Artificial Intelligence', "
            "'India', 'Transformer architecture')."
        ),
    )


class TavilyInput(BaseModel):
    """Input schema for the Tavily web search tool."""

    query: str = Field(
        ...,
        description=(
            "A web search query to find recent and relevant information "
            "from the internet. Best for current events, news, and "
            "real-time data."
        ),
    )


# ---------------------------------------------------------------------------
# Tool Factory Functions
# ---------------------------------------------------------------------------

def create_arxiv_tool():
    """
    Create an Arxiv search tool.

    Searches academic papers on arXiv.org. Returns published date, title,
    authors, and a truncated summary for each matching paper.

    Returns:
        ArxivQueryRun: A configured LangChain Arxiv tool instance.

    Raises:
        ImportError: If required packages are not installed.
    """
    try:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper

        api_wrapper = ArxivAPIWrapper(
            top_k_results=5,
            doc_content_chars_max=500,
        )
        tool = ArxivQueryRun(
            api_wrapper=api_wrapper,
            description=(
                "Search for academic papers on arXiv. Useful for finding "
                "research papers, technical reports, and scientific articles. "
                "Input should be a search query string such as a paper ID, "
                "topic, or author name."
            ),
            args_schema=ArxivInput,
        )
        logger.info("Arxiv tool initialized successfully.")
        return tool

    except ImportError as e:
        logger.error("Failed to import Arxiv dependencies: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to initialize Arxiv tool: %s", e)
        raise


def create_wikipedia_tool():
    """
    Create a Wikipedia search tool.

    Searches Wikipedia for general knowledge, biographies, historical events,
    and encyclopedic information.

    Returns:
        WikipediaQueryRun: A configured LangChain Wikipedia tool instance.

    Raises:
        ImportError: If required packages are not installed.
    """
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        api_wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500,
        )
        tool = WikipediaQueryRun(
            api_wrapper=api_wrapper,
            description=(
                "Search Wikipedia for general knowledge, biographies, "
                "historical events, and encyclopedic information. "
                "Input should be a clear, specific search term."
            ),
            args_schema=WikipediaInput,
        )
        logger.info("Wikipedia tool initialized successfully.")
        return tool

    except ImportError as e:
        logger.error("Failed to import Wikipedia dependencies: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to initialize Wikipedia tool: %s", e)
        raise


def create_tavily_tool():
    """
    Create a Tavily web search tool.

    Searches the internet for recent, real-time information including
    news, current events, and up-to-date data.

    Returns:
        TavilySearchResults: A configured LangChain Tavily search tool instance.

    Raises:
        ImportError: If required packages are not installed.
        ValueError: If TAVILY_API_KEY is not set in environment.
    """
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(
            description=(
                "Search the internet for recent and relevant information. "
                "Best for current events, breaking news, real-time data, "
                "and any query that requires up-to-date information. "
                "Input should be a descriptive search query."
            ),
            args_schema=TavilyInput,
        )
        logger.info("Tavily search tool initialized successfully.")
        return tool

    except ImportError as e:
        logger.error("Failed to import Tavily dependencies: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to initialize Tavily tool: %s", e)
        raise


def get_all_tools() -> list:
    """
    Initialize and return all available tools for the ReAct agent.

    Returns:
        list: A list containing [ArxivQueryRun, WikipediaQueryRun, TavilySearchResults].

    Raises:
        RuntimeError: If any tool fails to initialize.
    """
    tools = []
    errors = []

    tool_creators = [
        ("Arxiv", create_arxiv_tool),
        ("Wikipedia", create_wikipedia_tool),
        ("Tavily", create_tavily_tool),
    ]

    for name, creator in tool_creators:
        try:
            tool = creator()
            tools.append(tool)
        except Exception as e:
            error_msg = f"Failed to initialize {name} tool: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    if errors:
        raise RuntimeError(
            "Some tools failed to initialize:\n" + "\n".join(errors)
        )

    logger.info(
        "All tools initialized: %s",
        [t.name for t in tools],
    )
    return tools
