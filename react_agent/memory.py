"""
memory.py
=========
Persistent long-term memory for the ReAct agent using FAISS.

Provides a VectorMemoryManager that:
    - Initializes a FAISS vector store with HuggingFace embeddings
    - Persists the index to disk (./agent_memory/) across sessions
    - Supports similarity search for retrieving past conversation context
    - Upserts new interactions with timestamp and session metadata
"""

import logging
import os
from datetime import datetime, timezone

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent_memory",
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# FAISS index file name used by LangChain's FAISS.save_local / load_local
_FAISS_INDEX_FILE = "index.faiss"


# ---------------------------------------------------------------------------
# Vector Memory Manager
# ---------------------------------------------------------------------------

class VectorMemoryManager:
    """
    Manages a persistent FAISS vector store for long-term agent memory.

    On initialization, loads an existing index from `persist_dir` if one
    exists, otherwise creates an empty store on the first `add_interaction`
    call.

    Attributes:
        persist_dir: Directory where the FAISS index is persisted.
        embeddings: The HuggingFace embedding model instance.
        vector_store: The FAISS vector store (None until first document is added
                      or loaded from disk).
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize the memory manager.

        Args:
            persist_dir: Path to the directory for persisting the FAISS index.
            embedding_model_name: HuggingFace model ID for generating embeddings.
        """
        self.persist_dir = persist_dir
        self.vector_store = None

        # Initialize embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                "Embedding model loaded: %s", embedding_model_name
            )
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            raise

        # Attempt to load existing index from disk
        self._load_existing_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_existing_index(self):
        """Load a previously saved FAISS index from disk, if it exists."""
        index_path = os.path.join(self.persist_dir, _FAISS_INDEX_FILE)

        if os.path.exists(index_path):
            try:
                from langchain_community.vectorstores import FAISS

                self.vector_store = FAISS.load_local(
                    self.persist_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    "Loaded existing FAISS index from %s", self.persist_dir
                )
            except Exception as e:
                logger.warning(
                    "Failed to load existing index, starting fresh: %s", e
                )
                self.vector_store = None
        else:
            logger.info(
                "No existing index found at %s — will create on first write.",
                self.persist_dir,
            )

    def save(self):
        """Persist the current FAISS index to disk."""
        if self.vector_store is not None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vector_store.save_local(self.persist_dir)
            logger.info("FAISS index saved to %s", self.persist_dir)
        else:
            logger.debug("No vector store to save (empty memory).")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[str]:
        """
        Search for the most relevant past interactions.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            A list of formatted strings, each containing the interaction
            text and its metadata (timestamp, session_id).
        """
        if self.vector_store is None:
            return ["No long-term memory available yet. This is a fresh start."]

        try:
            docs = self.vector_store.similarity_search(query, k=k)

            if not docs:
                return [f"No relevant memories found for query: '{query}'."]

            results = []
            for i, doc in enumerate(docs, 1):
                timestamp = doc.metadata.get("timestamp", "unknown")
                session_id = doc.metadata.get("session_id", "unknown")
                results.append(
                    f"--- Memory {i} ---\n"
                    f"Session: {session_id}\n"
                    f"Timestamp: {timestamp}\n"
                    f"Content:\n{doc.page_content}\n"
                )

            return results

        except Exception as e:
            logger.error("Memory search failed: %s", e)
            return [f"Error searching memory: {e}"]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_interaction(
        self,
        user_input: str,
        agent_response: str,
        session_id: str,
    ):
        """
        Embed and store a user–agent interaction in the vector store.

        Creates a Document with the combined interaction text and metadata
        (timestamp, session_id), adds it to FAISS, and persists to disk.

        Args:
            user_input: The user's message.
            agent_response: The agent's final response.
            session_id: A unique identifier for the current session.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        document_text = (
            f"User: {user_input}\n\n"
            f"Assistant: {agent_response}"
        )

        doc = Document(
            page_content=document_text,
            metadata={
                "timestamp": timestamp,
                "session_id": session_id,
                "user_input": user_input[:200],  # Truncated for metadata
            },
        )

        try:
            from langchain_community.vectorstores import FAISS

            if self.vector_store is None:
                # First document — create the vector store
                self.vector_store = FAISS.from_documents(
                    [doc], self.embeddings
                )
                logger.info("Created new FAISS index with first interaction.")
            else:
                self.vector_store.add_documents([doc])
                logger.info("Added interaction to existing FAISS index.")

            # Persist immediately
            self.save()

        except Exception as e:
            logger.error("Failed to index interaction: %s", e)
