"""Mock providers for comprehensive testing."""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock


class MockLLMProvider:
    """Comprehensive mock LLM provider for testing."""

    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize with optional predefined responses."""
        self.responses = responses or [
            "This is a mock response.",
            "I understand your question.",
            "Let me help you with that.",
            "Here's the information you requested.",
            "I've completed the task successfully.",
        ]
        self.response_index = 0
        self.call_count = 0
        self.last_messages = None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Mock generate method."""
        self.call_count += 1
        self.last_messages = messages

        # Return next response in cycle
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1

        return response

    def reset(self):
        """Reset mock state."""
        self.response_index = 0
        self.call_count = 0
        self.last_messages = None


class MockMemoryProvider:
    """Comprehensive mock memory provider for testing."""

    def __init__(self):
        """Initialize mock memory provider."""
        self.storage = {}
        self.agents = {}
        self.call_history = []

    def store(self, memory_id: str, memory_unit: Any) -> str:
        """Mock store method."""
        self.call_history.append(("store", memory_id, memory_unit))

        if memory_id not in self.storage:
            self.storage[memory_id] = []

        unit_id = str(uuid.uuid4())
        unit_data = {
            "id": unit_id,
            "memory_id": memory_id,
            "data": memory_unit,
            "timestamp": datetime.now().isoformat(),
        }

        self.storage[memory_id].append(unit_data)
        return unit_id

    def retrieve_by_id(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """Mock retrieve by ID method."""
        self.call_history.append(("retrieve_by_id", unit_id))

        for memory_id, units in self.storage.items():
            for unit in units:
                if unit["id"] == unit_id:
                    return unit
        return None

    def retrieve_by_query(
        self, query: str, memory_id: str, memory_type: Any, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Mock retrieve by query method."""
        self.call_history.append(
            ("retrieve_by_query", query, memory_id, memory_type, limit)
        )

        units = self.storage.get(memory_id, [])
        # Simple relevance simulation - return units that contain query words
        query_words = query.lower().split()
        relevant_units = []

        for unit in units:
            unit_text = str(unit["data"]).lower()
            if any(word in unit_text for word in query_words):
                relevant_units.append(unit)

        return relevant_units[:limit]

    def retrieve_conversation_history_ordered_by_timestamp(
        self, memory_id: str, memory_type: Any, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Mock conversation history retrieval."""
        self.call_history.append(
            ("retrieve_conversation_history", memory_id, memory_type, limit)
        )

        units = self.storage.get(memory_id, [])
        # Sort by timestamp and return recent ones
        sorted_units = sorted(units, key=lambda x: x["timestamp"], reverse=True)
        return sorted_units[:limit]

    def store_memagent(self, agent_data: Dict[str, Any]) -> str:
        """Mock store agent method."""
        agent_id = agent_data.get("agent_id", str(uuid.uuid4()))
        self.agents[agent_id] = agent_data
        self.call_history.append(("store_memagent", agent_id))
        return agent_id

    def retrieve_memagent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Mock retrieve agent method."""
        self.call_history.append(("retrieve_memagent", agent_id))
        return self.agents.get(agent_id)

    def list_memagents(self) -> List[Dict[str, Any]]:
        """Mock list agents method."""
        self.call_history.append(("list_memagents",))
        return list(self.agents.values())

    def delete_by_id(self, memory_id: str) -> bool:
        """Mock delete by ID method."""
        self.call_history.append(("delete_by_id", memory_id))
        if memory_id in self.storage:
            del self.storage[memory_id]
            return True
        return False

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Mock update agent memory IDs method."""
        self.call_history.append(("update_memagent_memory_ids", agent_id, memory_ids))
        if agent_id in self.agents:
            self.agents[agent_id]["memory_ids"] = memory_ids
            return True
        return False

    def get_call_history(self) -> List[tuple]:
        """Get history of all method calls."""
        return self.call_history.copy()

    def reset(self):
        """Reset mock state."""
        self.storage.clear()
        self.agents.clear()
        self.call_history.clear()


class MockToolbox:
    """Mock toolbox for testing tool functionality."""

    def __init__(self, tools: Optional[Dict[str, Any]] = None):
        """Initialize with optional tools."""
        self.tools = tools or {
            "calculator": {
                "metadata": {
                    "name": "calculator",
                    "description": "Simple calculator",
                    "parameters": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                        "operation": {
                            "type": "string",
                            "description": "Operation (+, -, *, /)",
                        },
                    },
                    "required": ["a", "b", "operation"],
                },
                "function": lambda a, b, operation: self._calculator(a, b, operation),
            },
            "text_transformer": {
                "metadata": {
                    "name": "text_transformer",
                    "description": "Transform text",
                    "parameters": {
                        "text": {"type": "string", "description": "Text to transform"},
                        "transform": {
                            "type": "string",
                            "description": "Type of transformation",
                        },
                    },
                    "required": ["text"],
                },
                "function": lambda text, transform="upper": text.upper()
                if transform == "upper"
                else text.lower(),
            },
        }
        self.call_history = []

    def _calculator(self, a: float, b: float, operation: str) -> float:
        """Mock calculator function."""
        self.call_history.append(("calculator", a, b, operation))
        if operation == "+":
            return a + b
        elif operation == "-":
            return a - b
        elif operation == "*":
            return a * b
        elif operation == "/":
            return a / b if b != 0 else float("inf")
        else:
            raise ValueError(f"Unknown operation: {operation}")


class MockPersona:
    """Mock persona for testing."""

    def __init__(
        self,
        name: str = "TestBot",
        role: str = "Assistant",
        traits: List[str] = None,
        expertise: List[str] = None,
    ):
        """Initialize mock persona."""
        self.name = name
        self.role = role
        self.personality_traits = traits or ["helpful", "friendly"]
        self.expertise = expertise or ["testing", "mocking"]
        self.background = "I am a mock persona for testing purposes."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "personality_traits": self.personality_traits,
            "expertise": self.expertise,
            "background": self.background,
        }


class MockSemanticCache:
    """Mock semantic cache for testing."""

    def __init__(self, enabled: bool = True):
        """Initialize mock semantic cache."""
        self.enabled = enabled
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, query: str, session_id: Optional[str] = None) -> Optional[str]:
        """Mock get method."""
        cache_key = f"{query}_{session_id or 'default'}"
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        else:
            self.misses += 1
            return None

    def set(self, query: str, response: str, session_id: Optional[str] = None):
        """Mock set method."""
        cache_key = f"{query}_{session_id or 'default'}"
        self.cache[cache_key] = response

    def clear(self):
        """Mock clear method."""
        self.cache.clear()

    def clear_session(self, session_id: str):
        """Mock clear session method."""
        to_remove = [key for key in self.cache.keys() if key.endswith(f"_{session_id}")]
        for key in to_remove:
            del self.cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "hit_rate": self.hits / (self.hits + self.misses)
            if (self.hits + self.misses) > 0
            else 0,
        }
