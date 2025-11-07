"""
MemAgent - Backward compatibility wrapper.

This file maintains backward compatibility for existing code that imports
from memorizz.memagent.

Now uses the refactored memagent/ module with unified MemoryProvider interface.
The original implementation is preserved in memagent_original_backup.py for reference.
"""

# Import from the refactored implementation
from .memagent.core import MemAgent
from .memagent.models import MemAgentModel

# Re-export all public APIs to maintain backward compatibility
__all__ = ["MemAgent", "MemAgentModel"]

# This ensures that code like:
#   from memorizz.memagent import MemAgent
# continues to work with both old and new calling conventions
