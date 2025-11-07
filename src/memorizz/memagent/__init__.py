"""
MemAgent module.

This module provides a maintainable structure while maintaining
100% backward compatibility with existing code.
"""

# Import core components
from .core import MemAgent

# Optional: Import managers for advanced users
from .managers import (
    CacheManager,
    MemoryManager,
    PersonaManager,
    ToolManager,
    WorkflowManager,
)
from .models import MemAgentConfig, MemAgentModel

# Export all public APIs
__all__ = [
    "MemAgent",
    "MemAgentModel",
    "MemAgentConfig",
    "MemoryManager",
    "ToolManager",
    "CacheManager",
    "PersonaManager",
    "WorkflowManager",
]
