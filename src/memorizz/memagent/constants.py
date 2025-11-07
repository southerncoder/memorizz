"""Configuration constants for MemAgent."""

import os

# Configuration constants
DEFAULT_INSTRUCTION = "You are a helpful assistant."
DEFAULT_MAX_STEPS = 20
DEFAULT_TOOL_ACCESS = "private"

# Logging configuration
MEMORIZZ_LOG_LEVEL = os.getenv("MEMORIZZ_LOG_LEVEL", "DEBUG").upper()

# Application modes
APPLICATION_MODES = {
    "assistant": "General purpose assistant",
    "chatbot": "Conversational chatbot",
    "agent": "Task-oriented agent",
}

# Memory types
DEFAULT_MEMORY_TYPES = ["conversation_memory", "semantic_memory"]
