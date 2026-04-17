"""
main.py
=======
Entry point for the ReAct Personal Assistant.

Handles:
    - Environment variable loading and validation
    - Agent initialization
    - Interactive REPL loop for user queries
    - Auto-indexing conversations to long-term memory
"""

import logging
import os
import sys

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

def load_environment():
    """
    Load environment variables from .env file and validate
    that all required API keys are present.

    Raises:
        SystemExit: If required environment variables are missing.
    """
    load_dotenv()

    required_vars = ["GROQ_API_KEY", "TAVILY_API_KEY", "NEWSDATA_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(
            "Missing required environment variables: %s",
            ", ".join(missing),
        )
        print(
            "\n❌ Error: The following environment variables are required "
            "but not set:"
        )
        for var in missing:
            print(f"   - {var}")
        print(
            "\n💡 Create a .env file in the project root with your API keys."
        )
        print("   See .env.example for the expected format.\n")
        sys.exit(1)

    # Ensure keys are available in os.environ for LangChain
    for var in required_vars:
        os.environ[var] = os.getenv(var)

    logger.info("Environment variables loaded successfully.")


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                   🤖 ReAct Personal Assistant                   ║
║                                                                  ║
║  Research : Arxiv | Wikipedia | Tavily | News                    ║
║  Compute  : Python REPL | Temporal                               ║
║  Files    : Read | Write | List (agent_workspace/)               ║
║  Memory   : Long-Term Memory (Vector DB)                         ║
║  Model    : Groq (qwen/qwen3-32b)                               ║
║                                                                  ║
║  Type your question and press Enter.                             ║
║  Type 'quit' or 'exit' to leave.                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_separator():
    """Print a visual separator between interactions."""
    print("\n" + "─" * 62 + "\n")


def display_response(messages: list):
    """
    Display the agent's response messages in a readable format.

    Shows the final AI message by default, with tool calls and
    observations visible for transparency.

    Args:
        messages: The list of messages from the agent execution.
    """
    print_separator()

    for msg in messages:
        try:
            msg.pretty_print()
        except Exception:
            # Fallback for messages without pretty_print
            print(f"[{msg.type}] {getattr(msg, 'content', str(msg))}")

    print_separator()


# ---------------------------------------------------------------------------
# Main REPL Loop
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point: loads environment, initializes the agent,
    and runs an interactive REPL loop.

    Every user–agent interaction is automatically saved to long-term
    memory (FAISS vector store) for retrieval in future sessions.
    """
    # 1. Load and validate environment
    load_environment()

    # 2. Initialize the agent
    print("\n⏳ Initializing agent...")

    try:
        from react_agent.agent_logic import (
            create_agent,
            save_interaction_to_memory,
        )

        graph, tools, memory_manager, session_id = create_agent()
        print(f"✅ Agent ready with tools: {[t.name for t in tools]}")
        print(f"📝 Session ID: {session_id}")

    except Exception as e:
        logger.error("Failed to initialize agent: %s", e)
        print(f"\n❌ Failed to initialize agent: {e}")
        sys.exit(1)

    # 3. Print welcome banner
    print(BANNER)

    # 4. Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()

            # Handle exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                # Persist memory before exiting
                memory_manager.save()
                print("\n💾 Memory saved.")
                print("👋 Goodbye! Thanks for using ReAct Assistant.\n")
                break

            # Skip empty input
            if not user_input:
                print("💡 Please type a question or 'quit' to exit.")
                continue

            # Run the agent
            print("\n🔄 Thinking...")
            result = graph.invoke({"messages": user_input})
            display_response(result["messages"])

            # Auto-index the interaction to long-term memory
            save_interaction_to_memory(
                messages=result["messages"],
                memory_manager=memory_manager,
                session_id=session_id,
                user_input=user_input,
            )

        except KeyboardInterrupt:
            # Persist memory before exiting
            memory_manager.save()
            print("\n\n💾 Memory saved.")
            print("👋 Interrupted. Goodbye!\n")
            break

        except Exception as e:
            logger.error("Error during agent execution: %s", e)
            print(f"\n⚠️  An error occurred: {e}")
            print("Please try again with a different query.\n")


if __name__ == "__main__":
    main()
