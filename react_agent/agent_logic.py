"""
agent_logic.py
==============
Core engine for the ReAct agent.

Responsibilities:
    - Initialize the LLM (ChatGroq) and bind tools
    - Define the LangGraph state schema
    - Build the execution graph with Thought/Action/Observation loop
    - Provide a high-level `run_agent()` function for query execution
    - SafeToolNode for resilient tool execution with error recovery
"""

import logging
from typing import Annotated

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from react_agent.prompts import build_system_message
from react_agent.tools import get_all_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen/qwen3-32b"
MAX_ITERATIONS = 10  # Safety guard against infinite tool-calling loops


# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    State schema for the ReAct agent graph.

    Attributes:
        messages: The conversation history, managed by LangGraph's
                  `add_messages` reducer which handles appending
                  and deduplication.
    """

    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# Safe Tool Node with Error Handling
# ---------------------------------------------------------------------------

class SafeToolNode:
    """
    A wrapper around LangGraph's ToolNode that catches tool execution
    errors and returns them as ToolMessages so the LLM can recover.

    Each tool call is processed individually so that one failing tool
    does not block others in a parallel tool-calling scenario.
    """

    def __init__(self, tools: list):
        self._tool_node = ToolNode(tools)
        self._tool_map = {tool.name: tool for tool in tools}

    def __call__(self, state: AgentState) -> dict:
        """
        Execute tool calls from the last AI message, catching any errors.

        If a tool raises an exception, the error is returned as a
        ToolMessage with a descriptive message so the LLM can observe
        the failure and try a different approach.
        """
        messages = state["messages"]
        last_message = messages[-1]

        results = []

        # Process each tool call individually for granular error handling
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                try:
                    tool = self._tool_map.get(tool_name)
                    if tool is None:
                        raise ValueError(
                            f"Unknown tool: '{tool_name}'. "
                            f"Available tools: {list(self._tool_map.keys())}"
                        )

                    logger.info(
                        "Executing tool '%s' with args: %s",
                        tool_name,
                        tool_args,
                    )
                    result = tool.invoke(tool_args)

                    # Ensure result is always a string
                    result_str = str(result) if result is not None else (
                        "Tool executed successfully (no output)."
                    )

                    results.append(
                        ToolMessage(
                            content=result_str,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )
                    logger.info("Tool '%s' executed successfully.", tool_name)

                except Exception as e:
                    error_msg = (
                        f"Error: {type(e).__name__}: {e}. "
                        "Please adjust your input or try a different tool."
                    )
                    logger.error(
                        "Tool '%s' failed: %s", tool_name, error_msg
                    )
                    results.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )

        return {"messages": results}


# ---------------------------------------------------------------------------
# Agent Graph Builder
# ---------------------------------------------------------------------------

def _create_llm(model: str = DEFAULT_MODEL) -> ChatGroq:
    """
    Initialize the ChatGroq LLM.

    Args:
        model: The model identifier to use on Groq.

    Returns:
        ChatGroq: An initialized LLM instance.
    """
    logger.info("Initializing LLM: %s", model)
    return ChatGroq(model=model)


def create_agent(model: str = DEFAULT_MODEL):
    """
    Build and compile the ReAct agent graph.

    The graph follows this flow:
        START → agent (LLM with tools) ←→ tools (with error handling)
                                        → END (when no tool calls)

    The agent node prepends a ReAct system message that includes:
        - Dynamic tool descriptions
        - Contextual awareness instructions (temporal tool)
        - Tool selection guidance (REPL vs File Editor)

    Args:
        model: The Groq model identifier to use.

    Returns:
        tuple: (compiled_graph, tools_list) — the compiled LangGraph
               StateGraph and the list of tool instances.
    """
    # 1. Initialize all tools (7 individual + 3 from file toolkit = 10 total)
    tools = get_all_tools()
    logger.info(
        "Tools loaded (%d): %s",
        len(tools),
        [t.name for t in tools],
    )

    # 2. Build the system message with tool descriptions
    system_message = build_system_message(tools)

    # 3. Initialize LLM and bind tools
    llm = _create_llm(model)
    llm_with_tools = llm.bind_tools(tools=tools)

    # 4. Define the agent node
    def agent_node(state: AgentState) -> dict:
        """
        The agent reasoning node.

        Prepends the ReAct system message to the conversation,
        invokes the LLM with bound tools, and returns the response.
        """
        messages = state["messages"]

        # Prepend system message if not already present
        if not messages or messages[0].type != "system":
            full_messages = [system_message] + list(messages)
        else:
            full_messages = list(messages)

        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    # 5. Build the state graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", SafeToolNode(tools))

    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # Loop back for multi-step reasoning

    # 6. Compile and return
    graph = builder.compile()
    logger.info("Agent graph compiled successfully.")

    return graph, tools


# ---------------------------------------------------------------------------
# High-Level Execution
# ---------------------------------------------------------------------------

def run_agent(query: str, graph=None, model: str = DEFAULT_MODEL) -> list:
    """
    Run the ReAct agent on a single query.

    Args:
        query: The user's question or request.
        graph: An optional pre-compiled graph. If None, a new one is created.
        model: The Groq model identifier (used only if graph is None).

    Returns:
        list[AnyMessage]: The full list of messages from the conversation,
                          including the agent's reasoning and final answer.
    """
    if graph is None:
        graph, _ = create_agent(model)

    logger.info("Running agent with query: %s", query[:100])

    result = graph.invoke({"messages": query})

    return result["messages"]
