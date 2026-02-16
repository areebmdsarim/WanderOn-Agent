"""
Thread manager for OpenAI conversation state management.
Handles creation, storage, and retrieval of conversation threads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from loguru import logger


class ThreadManager:
    """
    Manages OpenAI thread state for maintaining conversation context.
    Stores thread info locally to track ongoing conversations.
    """

    def __init__(self, storage_dir: str = ".threads"):
        """
        Initialize the thread manager.

        Args:
            storage_dir: Directory to store thread metadata (defaults to .threads)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.threads: Dict[str, dict] = {}
        self._load_threads()

    def _get_thread_file(self, thread_id: str) -> Path:
        """Get the file path for a thread."""
        return self.storage_dir / f"{thread_id}.json"

    def _load_threads(self) -> None:
        """Load all thread metadata from disk."""
        for thread_file in self.storage_dir.glob("*.json"):
            try:
                with open(thread_file, "r") as f:
                    data = json.load(f)
                    thread_id = data.get("thread_id")
                    if thread_id:
                        self.threads[thread_id] = data
                        logger.debug(f"Loaded thread: {thread_id}")
            except Exception as e:
                logger.error(f"Failed to load thread file {thread_file}: {e}")

    def create_thread(
        self, user_id: str = "default", metadata: Optional[dict] = None
    ) -> str:
        """
        Create a new conversation thread.

        Args:
            user_id: Identifier for the user (defaults to "default")
            metadata: Optional metadata to attach to the thread

        Returns:
            The thread ID
        """
        import uuid

        thread_id = str(uuid.uuid4())
        thread_data = {
            "thread_id": thread_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "metadata": metadata or {},
        }

        self.threads[thread_id] = thread_data
        self._save_thread(thread_id)
        logger.info(f"Created new thread: {thread_id} for user {user_id}")

        return thread_id

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to a thread.

        Args:
            thread_id: The thread ID
            role: Message role ("user" or "assistant")
            content: The message content
        """
        if thread_id not in self.threads:
            logger.warning(f"Thread not found: {thread_id}")
            return

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.threads[thread_id]["messages"].append(message)
        self._save_thread(thread_id)
        logger.debug(f"Added {role} message to thread {thread_id}")

    def get_thread_messages(self, thread_id: str) -> list:
        """
        Get all messages in a thread.

        Args:
            thread_id: The thread ID

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        if thread_id not in self.threads:
            logger.warning(f"Thread not found: {thread_id}")
            return []

        messages = self.threads[thread_id].get("messages", [])
        # Return in LangChain compatible format
        return [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

    def get_thread(self, thread_id: str) -> Optional[dict]:
        """
        Get thread metadata.

        Args:
            thread_id: The thread ID

        Returns:
            Thread data dict or None if not found
        """
        return self.threads.get(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread.

        Args:
            thread_id: The thread ID

        Returns:
            True if deleted, False if not found
        """
        if thread_id not in self.threads:
            logger.warning(f"Thread not found: {thread_id}")
            return False

        del self.threads[thread_id]
        thread_file = self._get_thread_file(thread_id)
        if thread_file.exists():
            thread_file.unlink()
            logger.info(f"Deleted thread: {thread_id}")
        return True

    def list_threads(self, user_id: Optional[str] = None) -> list:
        """
        List all threads, optionally filtered by user_id.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of thread IDs
        """
        threads = []
        for thread_id, thread_data in self.threads.items():
            if user_id is None or thread_data.get("user_id") == user_id:
                threads.append(thread_id)
        return threads

    def _save_thread(self, thread_id: str) -> None:
        """Save thread metadata to disk."""
        if thread_id not in self.threads:
            return

        try:
            thread_file = self._get_thread_file(thread_id)
            with open(thread_file, "w") as f:
                json.dump(self.threads[thread_id], f, indent=2)
                logger.debug(f"Saved thread: {thread_id}")
        except Exception as e:
            logger.error(f"Failed to save thread {thread_id}: {e}")

    def clear_old_threads(self, hours: int = 24) -> int:
        """
        Delete threads older than specified hours.

        Args:
            hours: Number of hours to keep threads for

        Returns:
            Number of threads deleted
        """
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        deleted = 0

        thread_ids_to_delete = []
        for thread_id, thread_data in self.threads.items():
            created_at_str = thread_data.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at < cutoff:
                        thread_ids_to_delete.append(thread_id)
                except Exception as e:
                    logger.warning(f"Could not parse timestamp for {thread_id}: {e}")

        for thread_id in thread_ids_to_delete:
            if self.delete_thread(thread_id):
                deleted += 1

        logger.info(f"Cleaned up {deleted} old threads")
        return deleted


# Global thread manager instance
_thread_manager: Optional[ThreadManager] = None


def get_thread_manager() -> ThreadManager:
    """Get or create the global thread manager."""
    global _thread_manager
    if _thread_manager is None:
        storage_dir = os.getenv("OPENAI_THREADS_DIR", ".threads")
        _thread_manager = ThreadManager(storage_dir)
    return _thread_manager
