"""
server.py
=========
FastAPI backend for the ReAct Personal Assistant web UI.

Serves the static frontend and provides an SSE-streaming chat API
that wraps the existing LangGraph agent.
"""

import json
import logging
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (initialized at startup)
# ---------------------------------------------------------------------------

agent_graph = None
agent_tools = None
memory_manager = None
session_id = None


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

def load_environment():
    """Load and validate environment variables."""
    load_dotenv()
    required_vars = ["GROQ_API_KEY", "TAVILY_API_KEY", "NEWSDATA_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error("Missing env vars: %s", ", ".join(missing))
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    for var in required_vars:
        os.environ[var] = os.getenv(var)
    logger.info("Environment variables loaded.")


# ---------------------------------------------------------------------------
# Lifespan — initialize agent once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent and memory on server startup."""
    global agent_graph, agent_tools, memory_manager, session_id

    load_environment()
    logger.info("Initializing agent...")

    from react_agent.agent_logic import create_agent
    agent_graph, agent_tools, memory_manager, session_id = create_agent()

    logger.info("Agent ready. Session: %s", session_id)
    logger.info("Tools: %s", [t.name for t in agent_tools])

    yield  # Server runs

    # Shutdown — persist memory
    if memory_manager:
        memory_manager.save()
        logger.info("Memory saved on shutdown.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="ReAct Assistant", lifespan=lifespan)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main chat UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# Chat API — SSE Streaming
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: Request):
    """
    Accept a user message and stream back agent events via SSE.

    Events emitted:
        - thinking: Agent is reasoning
        - tool_call: A tool was invoked {name, args}
        - tool_result: The observation from the tool {name, result}
        - answer: The final agent response {content}
        - memory_saved: Interaction saved to vector DB
        - error: Something went wrong {detail}
    """
    body = await request.json()
    user_message = body.get("message", "").strip()

    if not user_message:
        return {"error": "Empty message"}

    async def event_generator():
        try:
            # Signal that agent is thinking
            yield {
                "event": "thinking",
                "data": json.dumps({"status": "Agent is reasoning..."})
            }

            # Run the agent synchronously in a thread to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: agent_graph.invoke({"messages": user_message})
            )

            messages = result["messages"]

            # Walk through messages and emit events
            for msg in messages:
                if msg.type == "human":
                    continue  # Skip the user's own message

                elif msg.type == "ai":
                    # Check for tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {
                                "event": "tool_call",
                                "data": json.dumps({
                                    "name": tc["name"],
                                    "args": tc["args"],
                                })
                            }
                            await asyncio.sleep(0.05)

                    # If there's content (final or intermediate answer)
                    if msg.content:
                        # Check if this is the LAST ai message
                        # (not followed by tool calls)
                        has_tool_calls = (
                            hasattr(msg, "tool_calls") and msg.tool_calls
                        )
                        if not has_tool_calls:
                            yield {
                                "event": "answer",
                                "data": json.dumps({
                                    "content": msg.content
                                })
                            }

                elif msg.type == "tool":
                    tool_name = getattr(msg, "name", "unknown")
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({
                            "name": tool_name,
                            "result": msg.content[:2000],  # Truncate
                        })
                    }
                    await asyncio.sleep(0.05)

            # Save to memory
            try:
                from react_agent.agent_logic import save_interaction_to_memory
                await loop.run_in_executor(
                    None,
                    lambda: save_interaction_to_memory(
                        messages, memory_manager, session_id, user_message
                    )
                )
                yield {
                    "event": "memory_saved",
                    "data": json.dumps({"status": "Interaction saved to memory"})
                }
            except Exception as e:
                logger.error("Failed to save to memory: %s", e)

        except Exception as e:
            logger.error("Chat error: %s", e)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)})
            }

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Memory Stats API
# ---------------------------------------------------------------------------

@app.get("/api/memory/stats")
async def memory_stats():
    """Return basic memory statistics."""
    if memory_manager is None or memory_manager.vector_store is None:
        return {"total_memories": 0, "session_id": session_id}

    try:
        # FAISS index size
        total = memory_manager.vector_store.index.ntotal
    except Exception:
        total = 0

    return {
        "total_memories": total,
        "session_id": session_id,
    }


@app.get("/api/tools")
async def list_tools():
    """Return the list of available tools."""
    if agent_tools is None:
        return {"tools": []}
    return {
        "tools": [
            {"name": t.name, "description": t.description}
            for t in agent_tools
        ]
    }
