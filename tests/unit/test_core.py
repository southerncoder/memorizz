"""Unit tests for MemAgent core functionality."""
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from memorizz.memagent.constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS
from memorizz.memagent.core import MemAgent
from memorizz.memagent.models import MemAgentConfig, MemAgentModel
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid


class TestMemAgentCore:
    """Test the core MemAgent class."""

    @pytest.mark.unit
    def test_memagent_initialization_minimal(self):
        """Test MemAgent initialization with minimal parameters."""
        agent = MemAgent(instruction="Test instruction")

        assert_agent_state_valid(agent)
        assert agent.instruction == "Test instruction"
        assert agent.max_steps == DEFAULT_MAX_STEPS
        assert agent.agent_id is not None
        assert len(agent.agent_id) > 0

    @pytest.mark.unit
    def test_memagent_initialization_full(
        self, mock_llm_provider, mock_memory_provider, sample_tools, sample_persona
    ):
        """Test MemAgent initialization with all parameters."""
        agent = MemAgent(
            model=mock_llm_provider,
            llm_config={"provider": "test", "model": "test-model"},
            tools=sample_tools,
            persona=sample_persona,
            instruction="Full test instruction",
            application_mode="agent",
            max_steps=25,
            memory_provider=mock_memory_provider,
            memory_ids=["test_memory_1", "test_memory_2"],
            agent_id="test_agent_123",
            semantic_cache=True,
            semantic_cache_config={"similarity_threshold": 0.9},
        )

        assert_agent_state_valid(agent)
        assert agent.instruction == "Full test instruction"
        assert agent.max_steps == 25
        assert agent.agent_id == "test_agent_123"
        assert agent.memory_ids == ["test_memory_1", "test_memory_2"]
        assert agent.model == mock_llm_provider
        assert agent.memory_provider == mock_memory_provider

    @pytest.mark.unit
    def test_memagent_managers_initialization(self, mock_memory_provider):
        """Test that all manager components are properly initialized."""
        agent = MemAgent(
            instruction="Test",
            memory_provider=mock_memory_provider,
            semantic_cache=True,
        )

        # Check that managers exist
        assert hasattr(agent, "memory_manager")
        assert hasattr(agent, "tool_manager")
        assert hasattr(agent, "cache_manager")
        assert hasattr(agent, "persona_manager")
        assert hasattr(agent, "workflow_manager")
        assert hasattr(agent, "internet_access_manager")

        # Check memory manager
        assert agent.memory_manager is not None
        assert agent.memory_manager.memory_provider == mock_memory_provider

        # Check cache manager
        assert agent.cache_manager is not None
        assert agent.cache_manager.enabled == True

        # Check other managers
        assert agent.tool_manager is not None
        assert agent.persona_manager is not None
        assert agent.workflow_manager is not None
        assert agent.internet_access_manager is not None
        assert agent.internet_access_manager.is_enabled() is False

    @pytest.mark.unit
    def test_memagent_without_memory_provider(self):
        """Test MemAgent initialization without memory provider."""
        agent = MemAgent(instruction="Test without memory")

        assert_agent_state_valid(agent)
        assert agent.memory_manager is None

    @pytest.mark.unit
    def test_memagent_llm_config_loading(self):
        """Test LLM configuration loading."""
        with patch("memorizz.memagent.core.create_llm_provider") as mock_create:
            mock_llm = Mock()
            mock_create.return_value = mock_llm

            agent = MemAgent(
                instruction="Test LLM config",
                llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            )

            mock_create.assert_called_once_with(
                {"provider": "openai", "model": "gpt-3.5-turbo"}
            )
            assert agent.model == mock_llm

    @pytest.mark.unit
    def test_memagent_tool_initialization(self, sample_tools):
        """Test tool initialization."""
        agent = MemAgent(instruction="Test tools", tools=sample_tools)

        assert_agent_state_valid(agent)
        # Verify tools were added to tool manager
        assert agent.tool_manager is not None
        # Note: Detailed tool testing is in test_tool_manager.py

    @pytest.mark.unit
    def test_memagent_persona_initialization(self, sample_persona):
        """Test persona initialization."""
        agent = MemAgent(instruction="Test persona", persona=sample_persona)

        assert_agent_state_valid(agent)
        assert agent.persona_manager is not None
        # The persona should be set in the persona manager
        # Note: Detailed persona testing is in test_persona_manager.py

    @pytest.mark.unit
    def test_with_entity_memory_toggle(self, mock_memory_provider):
        """Ensure with_entity_memory toggles tools and state."""
        agent = MemAgent(
            instruction="Entity toggle", memory_provider=mock_memory_provider
        )

        assert agent._entity_memory_enabled is False
        agent.with_entity_memory(True)
        assert agent._entity_memory_enabled is True
        assert "entity_memory_lookup" in agent.tool_manager.tools

        agent.with_entity_memory(False)
        assert agent._entity_memory_enabled is False
        assert "entity_memory_lookup" not in agent.tool_manager.tools

    @pytest.mark.unit
    def test_assistant_mode_enables_entity_memory(self, mock_memory_provider):
        """Assistant mode should activate entity memory by default."""
        agent = MemAgent(
            instruction="Assistant with entity memory",
            memory_provider=mock_memory_provider,
            application_mode="assistant",
        )

        assert agent._entity_memory_enabled is True
        assert "entity_memory_lookup" in agent.tool_manager.tools

    @pytest.mark.unit
    def test_builder_entity_memory_toggle(self, mock_memory_provider):
        """Builder helper should pass through entity memory preference."""
        from memorizz.memagent.builders import MemAgentBuilder

        builder = (
            MemAgentBuilder()
            .with_instruction("Test builder entity memory")
            .with_memory_provider(mock_memory_provider)
            .with_entity_memory(True)
        )
        agent = builder.build()

        assert agent._entity_memory_enabled is True


class TestMemAgentRun:
    """Test the MemAgent run method."""

    @pytest.mark.unit
    def test_run_basic_query(self, memagent_with_mocks):
        """Test basic query execution."""
        agent = memagent_with_mocks

        response = agent.run("What is 2+2?")

        assert_agent_response_valid(response)
        # Verify LLM was called
        assert agent.model.generate.called

    @pytest.mark.unit
    def test_run_with_memory_id(self, memagent_with_mocks):
        """Test query execution with specific memory ID."""
        agent = memagent_with_mocks

        response = agent.run("Remember this conversation", memory_id="test_memory_123")

        assert_agent_response_valid(response)
        # Verify memory operations were attempted
        # (Detailed memory testing in test_memory_manager.py)

    @pytest.mark.unit
    def test_run_with_conversation_id(self, memagent_with_mocks):
        """Test query execution with conversation ID."""
        agent = memagent_with_mocks

        response = agent.run("Continue our chat", conversation_id="conv_456")

        assert_agent_response_valid(response)

    @pytest.mark.unit
    def test_run_error_handling(self, mock_memory_provider):
        """Test error handling in run method."""
        # Create agent with failing LLM
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("LLM failed")

        agent = MemAgent(
            model=failing_llm,
            memory_provider=mock_memory_provider,
            instruction="Test error handling",
        )

        response = agent.run("This will fail")

        # Should get error response, not crash
        assert isinstance(response, str)
        assert "error" in response.lower()

    @pytest.mark.unit
    def test_run_without_llm(self, mock_memory_provider):
        """Test run method without LLM model."""
        agent = MemAgent(
            instruction="Test without LLM", memory_provider=mock_memory_provider
        )

        response = agent.run("Test query")

        # Should get error response about missing model
        assert isinstance(response, str)
        assert "no llm model" in response.lower()

    @pytest.mark.unit
    def test_run_context_building(self, memagent_with_mocks):
        """Test that context is properly built for queries."""
        agent = memagent_with_mocks

        # Add some mock conversation history
        memory_id = "test_memory_context"
        agent.memory_ids = [memory_id]

        response = agent.run("Test context", memory_id=memory_id)

        assert_agent_response_valid(response)
        # Verify memory manager was called for context
        assert agent.memory_manager.load_conversation_history.called


class TestMemAgentMethods:
    """Test additional MemAgent methods."""

    @pytest.mark.unit
    def test_load_conversation_history(self, memagent_with_mocks):
        """Test loading conversation history."""
        agent = memagent_with_mocks

        history = agent.load_conversation_history("test_memory")

        # Should return a list (empty or with items)
        assert isinstance(history, list)

    @pytest.mark.unit
    def test_add_tool(self, memagent_with_mocks):
        """Test adding a tool to the agent."""
        agent = memagent_with_mocks

        def new_tool(x: int) -> int:
            return x * 2

        result = agent.add_tool(new_tool)

        # Should return True for successful addition
        assert result is True

    @pytest.mark.unit
    def test_set_persona(self, memagent_with_mocks, sample_persona):
        """Test setting agent persona."""
        agent = memagent_with_mocks

        result = agent.set_persona(sample_persona)

        # Should return True for successful setting
        assert result is True


class TestMemAgentConfig:
    """Test MemAgentConfig class."""

    @pytest.mark.unit
    def test_config_initialization_default(self):
        """Test default config initialization."""
        config = MemAgentConfig()

        assert config.instruction == DEFAULT_INSTRUCTION
        assert config.max_steps == DEFAULT_MAX_STEPS
        assert config.tool_access == "private"
        assert config.semantic_cache == False

    @pytest.mark.unit
    def test_config_initialization_custom(self):
        """Test custom config initialization."""
        config = MemAgentConfig(
            instruction="Custom instruction",
            max_steps=50,
            semantic_cache=True,
            custom_param="custom_value",
        )

        assert config.instruction == "Custom instruction"
        assert config.max_steps == 50
        assert config.semantic_cache == True
        assert config.custom_param == "custom_value"

    @pytest.mark.unit
    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = MemAgentConfig(instruction="Test", max_steps=15, semantic_cache=True)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["instruction"] == "Test"
        assert config_dict["max_steps"] == 15
        assert config_dict["semantic_cache"] == True


class TestMemAgentModel:
    """Test MemAgentModel class."""

    @pytest.mark.unit
    def test_model_initialization_default(self):
        """Test default model initialization."""
        model = MemAgentModel()

        assert model.instruction == DEFAULT_INSTRUCTION
        assert model.max_steps == DEFAULT_MAX_STEPS
        assert model.tool_access == "private"
        assert model.semantic_cache == False
        assert model.application_mode == "assistant"

    @pytest.mark.unit
    def test_model_initialization_custom(self):
        """Test custom model initialization."""
        model = MemAgentModel(
            instruction="Custom model instruction",
            max_steps=30,
            agent_id="custom_agent",
            semantic_cache=True,
            memory_ids=["mem1", "mem2"],
        )

        assert model.instruction == "Custom model instruction"
        assert model.max_steps == 30
        assert model.agent_id == "custom_agent"
        assert model.semantic_cache == True
        assert model.memory_ids == ["mem1", "mem2"]

    @pytest.mark.unit
    def test_model_validation(self):
        """Test model validation."""
        # Should not raise validation errors
        model = MemAgentModel(max_steps=10, instruction="Valid instruction")

        assert model.max_steps == 10
        assert model.instruction == "Valid instruction"

    @pytest.mark.unit
    def test_model_serialization(self):
        """Test model serialization."""
        model = MemAgentModel(
            instruction="Test serialization", max_steps=20, semantic_cache=True
        )

        # Should be able to convert to dict
        model_dict = model.model_dump()

        assert isinstance(model_dict, dict)
        assert model_dict["instruction"] == "Test serialization"
        assert model_dict["max_steps"] == 20
        assert model_dict["semantic_cache"] == True
