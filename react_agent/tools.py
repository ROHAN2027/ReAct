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
    - News (newsdata.io): Fetch top headlines for a query
    - Python REPL: Execute Python code and return stdout/stderr
    - File Editor: Read, write, and list files in ./agent_workspace
    - Temporal: Get the current UTC date and time
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_WORKSPACE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent_workspace",
)

NEWSDATA_API_BASE = "https://newsdata.io/api/1/latest"
NEWSDATA_TIMEOUT_SECONDS = 15


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


class NewsInput(BaseModel):
    """Input schema for the newsdata.io news tool."""

    query: str = Field(
        ...,
        description=(
            "A keyword or phrase to search for in recent news headlines. "
            "Use clear, specific terms (e.g., 'artificial intelligence', "
            "'climate change', 'stock market')."
        ),
    )


class PythonREPLInput(BaseModel):
    """Input schema for the Python REPL tool."""

    code: str = Field(
        ...,
        description=(
            "A valid Python code snippet to execute. The code should "
            "print its output to stdout. Use this for calculations, "
            "data transformations, logic, or any ephemeral computation."
        ),
    )


# ---------------------------------------------------------------------------
# Tool Factory Functions — Original Tools
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


# ---------------------------------------------------------------------------
# Tool Factory Functions — New Tools
# ---------------------------------------------------------------------------

def create_news_tool():
    """
    Create a News tool that fetches headlines from newsdata.io.

    Returns the top 3 headlines with URLs for a given search query.
    Requires NEWSDATA_API_KEY environment variable.

    Returns:
        A LangChain @tool decorated function.
    """
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        raise ValueError(
            "NEWSDATA_API_KEY environment variable is not set. "
            "Get a free key at https://newsdata.io/"
        )

    @tool("news_search", args_schema=NewsInput)
    def news_search(query: str) -> str:
        """
        Search for recent news headlines using newsdata.io.

        Fetches the top 3 headlines matching the query, including
        their titles and URLs. Use this for breaking news, current
        events, and trending topics.
        """
        try:
            response = requests.get(
                NEWSDATA_API_BASE,
                params={
                    "apikey": api_key,
                    "q": query,
                    "language": "en",
                },
                timeout=NEWSDATA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

            # Handle API-level errors
            if data.get("status") != "success":
                error_detail = data.get("results", {}).get(
                    "message", "Unknown API error"
                )
                return (
                    f"Error: newsdata.io returned an error: {error_detail}. "
                    "Please adjust your input or try a different tool."
                )

            articles = data.get("results", [])
            if not articles:
                return f"No news articles found for query: '{query}'."

            # Format top 3 headlines
            headlines = []
            for i, article in enumerate(articles[:3], 1):
                title = article.get("title", "No title")
                link = article.get("link", "No URL")
                headlines.append(f"{i}. {title}\n   URL: {link}")

            return f"Top news for '{query}':\n\n" + "\n\n".join(headlines)

        except requests.exceptions.Timeout:
            return (
                "Error: newsdata.io request timed out. "
                "Please adjust your input or try a different tool."
            )
        except requests.exceptions.ConnectionError:
            return (
                "Error: Could not connect to newsdata.io. "
                "Please adjust your input or try a different tool."
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                return (
                    "Error: newsdata.io API rate limit exhausted. "
                    "Please adjust your input or try a different tool."
                )
            return (
                f"Error: newsdata.io HTTP error ({e}). "
                "Please adjust your input or try a different tool."
            )
        except Exception as e:
            return (
                f"Error: Unexpected error fetching news: {e}. "
                "Please adjust your input or try a different tool."
            )

    logger.info("News tool initialized successfully.")
    return news_search


def create_python_repl_tool():
    """
    Create a Python REPL tool for executing code.

    Uses langchain_experimental's PythonREPL. The tool executes
    Python code in-process and returns stdout or stderr output.

    Returns:
        A LangChain @tool decorated function.

    Raises:
        ImportError: If langchain_experimental is not installed.
    """
    try:
        from langchain_experimental.utilities import PythonREPL
    except ImportError as e:
        logger.error("Failed to import PythonREPL: %s", e)
        raise

    # Instantiate a single REPL for this session
    repl = PythonREPL()

    @tool("python_repl", args_schema=PythonREPLInput)
    def python_repl(code: str) -> str:
        """
        Execute Python code and return the output.

        Use this tool for calculations, data analysis, math problems,
        string manipulation, or any ephemeral computation. The code
        should use print() to produce output. Do NOT use this for
        persistent file storage — use the file editor tools instead.
        """
        try:
            result = repl.run(code)
            if not result or not result.strip():
                return "Code executed successfully (no output produced)."
            return result.strip()
        except SyntaxError as e:
            return (
                f"Error: Python syntax error: {e}. "
                "Please adjust your input or try a different tool."
            )
        except Exception as e:
            return (
                f"Error: Python execution error: {type(e).__name__}: {e}. "
                "Please adjust your input or try a different tool."
            )

    logger.info("Python REPL tool initialized successfully.")
    return python_repl


def create_file_editor_tools() -> list:
    """
    Create file management tools scoped to ./agent_workspace.

    Provides read_file, write_file, and list_directory tools
    via LangChain's FileManagementToolkit.

    Returns:
        list: A list of file management tool instances.

    Raises:
        ImportError: If required packages are not installed.
    """
    try:
        from langchain_community.agent_toolkits.file_management.toolkit import (
            FileManagementToolkit,
        )
    except ImportError as e:
        logger.error("Failed to import FileManagementToolkit: %s", e)
        raise

    # Ensure workspace directory exists
    os.makedirs(AGENT_WORKSPACE_DIR, exist_ok=True)

    toolkit = FileManagementToolkit(
        root_dir=AGENT_WORKSPACE_DIR,
        selected_tools=["read_file", "write_file", "list_directory"],
    )

    tools = toolkit.get_tools()
    logger.info(
        "File editor tools initialized: %s (root: %s)",
        [t.name for t in tools],
        AGENT_WORKSPACE_DIR,
    )
    return tools


def create_temporal_tool():
    """
    Create a temporal tool that returns the current UTC date and time.

    Returns:
        A LangChain @tool decorated function.
    """

    @tool("get_current_datetime")
    def get_current_datetime() -> str:
        """
        Get the current date and time in UTC.

        Returns a human-readable string with the full date, time, and
        day of the week. Use this tool FIRST when the user's question
        involves dates, scheduling, or time-sensitive planning.
        """
        try:
            now = datetime.now(timezone.utc)
            return now.strftime(
                "Current UTC Date & Time: %A, %B %d, %Y at %I:%M:%S %p UTC"
            )
        except Exception as e:
            return (
                f"Error: Could not retrieve current time: {e}. "
                "Please adjust your input or try a different tool."
            )

    logger.info("Temporal tool initialized successfully.")
    return get_current_datetime


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

def get_all_tools() -> list:
    """
    Initialize and return all available tools for the ReAct agent.

    Returns a flat list of all tool instances:
        - Arxiv, Wikipedia, Tavily (search/research)
        - News (headlines from newsdata.io)
        - Python REPL (code execution)
        - File Editor: read_file, write_file, list_directory
        - Temporal (current UTC datetime)

    Returns:
        list: All tool instances ready for LLM binding.

    Raises:
        RuntimeError: If any critical tool fails to initialize.
    """
    tools = []
    errors = []

    # --- Single-instance tool creators ---
    single_tool_creators = [
        ("Arxiv", create_arxiv_tool),
        ("Wikipedia", create_wikipedia_tool),
        ("Tavily", create_tavily_tool),
        ("News", create_news_tool),
        ("Python REPL", create_python_repl_tool),
        ("Temporal", create_temporal_tool),
    ]

    for name, creator in single_tool_creators:
        try:
            tool = creator()
            tools.append(tool)
        except Exception as e:
            error_msg = f"Failed to initialize {name} tool: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    # --- Multi-instance tool creators (toolkits) ---
    try:
        file_tools = create_file_editor_tools()
        tools.extend(file_tools)
    except Exception as e:
        error_msg = f"Failed to initialize File Editor tools: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    if errors:
        raise RuntimeError(
            "Some tools failed to initialize:\n" + "\n".join(errors)
        )

    logger.info(
        "All tools initialized (%d total): %s",
        len(tools),
        [t.name for t in tools],
    )
    return tools
