import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..enums import MemoryType, Role
from ..llms.llm_factory import create_llm_provider
from .constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS, DEFAULT_TOOL_ACCESS
from .managers import (
    CacheManager,
    MemoryManager,
    PersonaManager,
    ToolManager,
    WorkflowManager,
)

logger = logging.getLogger(__name__)


class MemAgent:
    """
    MemAgent class that orchestrates manager components.

    """

    def __init__(
        self,
        model: Optional[Any] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List, Any]] = None,
        persona: Optional[Any] = None,
        instruction: Optional[str] = None,
        application_mode: Optional[str] = None,
        memory_types: Optional[List[Union[str, Any]]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        memory_provider: Optional[Any] = None,
        memory_ids: Optional[Union[str, List[str]]] = None,
        agent_id: Optional[str] = None,
        tool_access: Optional[str] = DEFAULT_TOOL_ACCESS,
        delegates: Optional[List["MemAgent"]] = None,
        verbose: bool = None,
        embedding_provider: Optional[str] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        semantic_cache: bool = False,
        semantic_cache_config: Optional[Union[Any, Dict[str, Any]]] = None,
    ):
        """Initialize the MemAgent with configuration."""
        # Store configuration
        self.agent_id = agent_id or str(uuid.uuid4())
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.max_steps = max_steps
        self.memory_provider = memory_provider
        self.memory_ids = (
            memory_ids
            if isinstance(memory_ids, list)
            else [memory_ids]
            if memory_ids
            else []
        )

        # Store memory types for tracking which memory systems are active
        self.active_memory_types = self._initialize_memory_types(
            application_mode, memory_types
        )

        # Initialize conversation state tracking
        self._current_conversation_id = None
        self._current_memory_id = None

        # Initialize LLM
        self.model = model
        if not model and llm_config:
            try:
                self.model = create_llm_provider(llm_config)
            except Exception as e:
                logger.warning(f"Could not create LLM from config: {e}")

        # Initialize manager components
        self._initialize_managers(
            memory_provider=memory_provider,
            semantic_cache=semantic_cache,
            semantic_cache_config=semantic_cache_config,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
        )

        # Initialize tools if provided
        if tools:
            self._initialize_tools(tools)

        # Initialize persona if provided
        if persona:
            self.persona_manager.set_persona(persona, self.agent_id, save=False)

        logger.info(
            f"MemAgent {self.agent_id} initialized with memory types: {self.active_memory_types}"
        )

    def _initialize_memory_types(self, application_mode, memory_types):
        """Initialize active memory types based on application mode or explicit memory_types."""
        # If explicit memory_types provided, use them
        if memory_types:
            if isinstance(memory_types, list):
                # Convert string memory types to MemoryType enums if needed
                result = []
                for mt in memory_types:
                    if isinstance(mt, str):
                        try:
                            result.append(MemoryType[mt.upper()])
                        except KeyError:
                            logger.warning(f"Unknown memory type: {mt}")
                    else:
                        result.append(mt)
                return result
            return memory_types

        # Otherwise, derive from application_mode or use defaults
        # Default: conversation + workflow memory for most agents
        return [MemoryType.CONVERSATION_MEMORY, MemoryType.WORKFLOW_MEMORY]

    def _initialize_managers(
        self,
        memory_provider=None,
        semantic_cache=False,
        semantic_cache_config=None,
        embedding_provider=None,
        embedding_config=None,
    ):
        """Initialize all manager components."""
        # Memory Manager
        if memory_provider:
            self.memory_manager = MemoryManager(memory_provider)
        else:
            self.memory_manager = None
            logger.warning("No memory provider - memory functionality disabled")

        # Tool Manager
        self.tool_manager = ToolManager(memory_provider)

        # Cache Manager
        self.cache_manager = CacheManager(
            enabled=semantic_cache,
            config=semantic_cache_config,
            agent_id=self.agent_id,
            memory_id=self.memory_ids[0] if self.memory_ids else None,
            memory_provider=memory_provider,  # ✅ Pass memory provider for persistence
        )

        # Persona Manager
        self.persona_manager = PersonaManager(memory_provider)

        # Workflow Manager
        self.workflow_manager = WorkflowManager()

    def _initialize_tools(self, tools):
        """Initialize tools using the tool manager."""
        if hasattr(tools, "__iter__") and not isinstance(tools, str):
            # List of tools
            for tool in tools:
                self.tool_manager.add_tool(tool)
        else:
            # Single tool or toolbox
            self.tool_manager.add_tool(tools)

    def run(
        self, query: str, memory_id: str = None, conversation_id: str = None
    ) -> str:
        """
        Run the agent with the given query using the new manager architecture.

        Args:
            query: The user's query
            memory_id: Optional memory ID to use (if not provided, uses stored or default)
            conversation_id: Optional conversation ID to use (if not provided, reuses current or creates new)

        Returns:
            The agent's response
        """
        logger.info(f"MemAgent {self.agent_id} executing query: {query[:50]}...")

        try:
            # 1. Prepare IDs with state management
            # memory_id: explicit → stored → default → generate
            if memory_id:
                self._current_memory_id = memory_id
            elif not self._current_memory_id:
                self._current_memory_id = (
                    self.memory_ids[0] if self.memory_ids else str(uuid.uuid4())
                )
            memory_id = self._current_memory_id

            # conversation_id: explicit → stored → generate
            if conversation_id:
                self._current_conversation_id = conversation_id
            elif not self._current_conversation_id:
                self._current_conversation_id = str(uuid.uuid4())
                logger.debug(
                    f"Started new conversation: {self._current_conversation_id}"
                )
            conversation_id = self._current_conversation_id

            # 2. Check semantic cache first
            cached_response = None
            if self.cache_manager.enabled:
                cached_response = self.cache_manager.get_cached_response(
                    query, conversation_id
                )
                if cached_response:
                    logger.info("Returning cached response")
                    self._record_interaction(
                        query, cached_response, memory_id, conversation_id
                    )
                    return cached_response

            # 3. Build context and prompt
            context = self._build_context(query, memory_id)
            system_prompt = self._build_system_prompt()

            # 4. Execute with LLM
            response = self._execute_llm_interaction(system_prompt, query, context)

            # 5. Cache the response
            if self.cache_manager.enabled:
                self.cache_manager.cache_response(query, response, conversation_id)

            # 6. Record interaction in memory
            self._record_interaction(query, response, memory_id, conversation_id)

            logger.info(f"MemAgent {self.agent_id} completed successfully")
            return response

        except Exception as e:
            logger.error(f"MemAgent execution failed: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return error_response

    def _build_context(self, query: str, memory_id: str) -> Dict[str, Any]:
        """Build context for the query using memory manager."""
        context = {"query": query}

        if self.memory_manager:
            # Load conversation history
            try:
                history = self.memory_manager.load_conversation_history(
                    memory_id, limit=10
                )
                context["conversation_history"] = history

                # Get relevant memories
                relevant_memories = self.memory_manager.retrieve_relevant_memories(
                    query=query,
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    memory_id=memory_id,
                    limit=5,
                )
                context["relevant_memories"] = relevant_memories

            except Exception as e:
                logger.warning(f"Failed to build memory context: {e}")

        return context

    def _build_system_prompt(self) -> str:
        """Build system prompt using persona manager."""
        prompt_parts = [self.instruction]

        # Add persona information if available
        persona_prompt = self.persona_manager.get_persona_prompt()
        if persona_prompt:
            prompt_parts.append(persona_prompt)

        # Add tool information
        tools = self.tool_manager.get_tool_metadata()
        if tools:
            tool_descriptions = [
                f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}"
                for tool in tools[:5]
            ]
            if tool_descriptions:
                prompt_parts.append(
                    f"Available tools:\n{chr(10).join(tool_descriptions)}"
                )

        return "\n\n".join(prompt_parts)

    def _execute_llm_interaction(
        self, system_prompt: str, query: str, context: Dict[str, Any]
    ) -> str:
        """Execute the LLM interaction with tool calling support."""
        if not self.model:
            return "Error: No LLM model configured"

        try:
            # Initialize workflow tracking if workflow memory is active
            workflow = None
            if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
                from ..long_term_memory.procedural.workflow.workflow import (
                    Workflow,
                    WorkflowOutcome,
                )

                workflow = Workflow(
                    name="Tool Execution for Query",
                    description=f"Workflow tracking tool usage for: {query[:100]}",
                    memory_id=self._current_memory_id
                    or (self.memory_ids[0] if self.memory_ids else str(uuid.uuid4())),
                    agent_id=self.agent_id,
                    user_query=query,
                )
                logger.debug(f"Created workflow for tracking: {workflow.workflow_id}")

            # Build initial messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation history if available
            history = context.get("conversation_history", [])
            for item in history[-5:]:  # Last 5 interactions
                if isinstance(item, dict):
                    # Handle flat structure (Oracle/direct format)
                    if "role" in item and "content" in item:
                        role = item.get("role")
                        text = item.get("content")
                        if role and text:
                            messages.append({"role": role, "content": text})
                    # Handle nested structure (legacy format)
                    elif "content" in item:
                        content = item["content"]
                        if isinstance(content, dict):
                            role = content.get("role", "user")
                            text = content.get("content", "")
                            if role and text:
                                messages.append({"role": role, "content": text})

            # Add current query
            messages.append({"role": "user", "content": query})

            # Get tool metadata if available and convert to OpenAI format
            tools = None
            if self.tool_manager:
                tool_metadata = self.tool_manager.get_tool_metadata()
                if tool_metadata:
                    # Convert to OpenAI function calling format
                    tools = []
                    for meta in tool_metadata:
                        # Check if already in correct OpenAI format (has nested 'function' key)
                        if (
                            "type" in meta
                            and meta["type"] == "function"
                            and "function" in meta
                        ):
                            # Already in OpenAI format
                            tools.append(meta)
                        else:
                            # Convert to OpenAI format
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": meta.get("name", "unknown"),
                                    "description": meta.get(
                                        "description", "No description"
                                    ),
                                    "parameters": {
                                        "type": "object",
                                        "properties": meta.get("parameters", {}),
                                        "required": meta.get("required", []),
                                    },
                                },
                            }
                            tools.append(openai_tool)

            # Execute main loop with tool calling
            max_iterations = 5
            for iteration in range(max_iterations):
                # Call LLM
                response = self.model.generate(messages, tools=tools)

                # Check if response is a string (no tool calls)
                if isinstance(response, str):
                    # Store workflow before returning (if it exists and has steps)
                    if workflow and workflow.steps:
                        try:
                            workflow.store_workflow(self.memory_provider)
                            logger.info(
                                f"Stored workflow with {len(workflow.steps)} steps"
                            )
                        except Exception as e:
                            logger.error(f"Error storing workflow: {str(e)}")
                    return response

                # Handle tool calls
                if hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message

                    # If no tool calls, return the content
                    if not message.tool_calls:
                        final_content = (
                            message.content
                            if message.content
                            else "I couldn't generate a response."
                        )
                        # Store workflow before returning (if it exists and has steps)
                        if workflow and workflow.steps:
                            try:
                                workflow.store_workflow(self.memory_provider)
                                logger.info(
                                    f"Stored workflow with {len(workflow.steps)} steps"
                                )
                            except Exception as e:
                                logger.error(f"Error storing workflow: {str(e)}")
                        return final_content

                    # Add assistant message with tool calls to history
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in message.tool_calls
                            ],
                        }
                    )

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            import json

                            arguments = json.loads(tool_call.function.arguments)
                        except (json.JSONDecodeError, Exception):
                            arguments = {}

                        logger.info(
                            f"Executing tool: {tool_name} with args: {arguments}"
                        )

                        # Execute the tool
                        error_message = None
                        tool_outcome = WorkflowOutcome.SUCCESS if workflow else None

                        if self.tool_manager:
                            try:
                                result, _ = self.tool_manager.execute_tool(
                                    tool_name, arguments
                                )
                            except Exception as e:
                                result = f"Error executing tool: {str(e)}"
                                error_message = str(e)
                                if workflow:
                                    tool_outcome = WorkflowOutcome.FAILURE
                                logger.error(f"Tool {tool_name} failed: {e}")
                        else:
                            result = "Error: No tool manager available"
                            error_message = "No tool manager available"
                            if workflow:
                                tool_outcome = WorkflowOutcome.FAILURE

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": str(result),
                            }
                        )

                        logger.info(f"Tool {tool_name} returned: {result}")

                        # Track workflow step if workflow memory is active
                        if workflow:
                            # Get tool metadata for the _id
                            tool_entry = {}
                            if self.tool_manager:
                                tool_metadata = self.tool_manager.get_tool_metadata()
                                tool_entry = next(
                                    (
                                        meta
                                        for meta in tool_metadata
                                        if meta.get("name") == tool_name
                                    ),
                                    {},
                                )

                            workflow.add_step(
                                f"Step {len(workflow.steps) + 1}: {tool_name}",
                                {
                                    "_id": (
                                        str(tool_entry.get("_id"))
                                        if tool_entry
                                        else None
                                    ),
                                    "arguments": arguments,
                                    "result": result,
                                    "timestamp": datetime.now().isoformat(),
                                    "error": error_message,
                                },
                            )

                            # Update workflow outcome if any step failed
                            if tool_outcome == WorkflowOutcome.FAILURE:
                                workflow.outcome = WorkflowOutcome.FAILURE

                    # Continue loop to get final response
                    continue

                # Fallback: return any content we got
                fallback_response = "I encountered an unexpected response format."
                # Store workflow before returning (if it exists and has steps)
                if workflow and workflow.steps:
                    try:
                        workflow.store_workflow(self.memory_provider)
                        logger.info(f"Stored workflow with {len(workflow.steps)} steps")
                    except Exception as e:
                        logger.error(f"Error storing workflow: {str(e)}")
                return fallback_response

            # If we exhausted iterations
            final_response = (
                "I reached the maximum number of tool calls. Please try again."
            )

            # Store workflow if it was created and has steps
            if workflow and workflow.steps:
                try:
                    workflow.store_workflow(self.memory_provider)
                    logger.info(f"Stored workflow with {len(workflow.steps)} steps")
                except Exception as e:
                    logger.error(f"Error storing workflow: {str(e)}")
                    # Continue execution even if workflow storage fails

            return final_response

        except Exception as e:
            logger.error(f"LLM interaction failed: {e}")

            # Store workflow even on error if it exists
            if "workflow" in locals() and workflow and workflow.steps:
                try:
                    workflow.store_workflow(self.memory_provider)
                except Exception as workflow_error:
                    logger.error(
                        f"Error storing workflow after exception: {str(workflow_error)}"
                    )

            return f"I encountered an error while processing your request: {str(e)}"

    def _record_interaction(
        self, query: str, response: str, memory_id: str, conversation_id: str
    ):
        """Record the interaction in memory."""
        if not self.memory_manager:
            return

        try:
            # Record user query
            user_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.USER,
                content=query,
                conversation_id=conversation_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
            )
            self.memory_manager.save_memory_unit(user_memory, memory_id)

            # Record assistant response
            assistant_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.ASSISTANT,
                content=response,
                conversation_id=conversation_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
            )
            self.memory_manager.save_memory_unit(assistant_memory, memory_id)

            logger.debug(f"Recorded interaction in memory: {memory_id}")

        except Exception as e:
            logger.warning(f"Failed to record interaction: {e}")

    # Conversation state management methods
    def start_new_conversation(self) -> str:
        """
        Start a new conversation by resetting the conversation_id.

        Returns:
            str: The new conversation_id
        """
        self._current_conversation_id = str(uuid.uuid4())
        logger.info(f"Started new conversation: {self._current_conversation_id}")
        return self._current_conversation_id

    def get_current_conversation_id(self) -> Optional[str]:
        """
        Get the current conversation ID.

        Returns:
            str: Current conversation_id or None
        """
        return self._current_conversation_id

    def get_current_memory_id(self) -> Optional[str]:
        """
        Get the current memory ID.

        Returns:
            str: Current memory_id or None
        """
        return self._current_memory_id

    def reset_conversation_state(self):
        """Reset both conversation_id and memory_id (start completely fresh)."""
        self._current_conversation_id = None
        self._current_memory_id = None
        logger.info("Reset conversation state")

    def enable_semantic_cache(
        self,
        threshold: float = 0.85,
        scope: str = "local",
        embedding_provider: str = None,
        embedding_config: Dict[str, Any] = None,
    ):
        """
        Enable semantic cache on an already-built agent.

        Args:
            threshold: Similarity threshold for cache hits (0.0-1.0, default: 0.85)
            scope: Cache scope - 'local' (agent-specific) or 'global' (all agents)
            embedding_provider: Optional embedding provider (e.g., 'openai', 'ollama')
            embedding_config: Optional configuration for embedding provider

        Returns:
            Self for method chaining

        Example:
            agent.enable_semantic_cache(threshold=0.9, scope='local')
        """
        try:
            pass

            # Create cache configuration
            cache_config = {"similarity_threshold": threshold, "scope": scope}

            # Add embedding provider if specified
            if embedding_provider:
                cache_config["embedding_provider"] = embedding_provider
            if embedding_config:
                cache_config["embedding_config"] = embedding_config

            # Re-initialize the cache manager with new settings
            if self.cache_manager:
                # Enable the existing cache manager
                self.cache_manager.enabled = True
                self.cache_manager.memory_provider = (
                    self.memory_provider
                )  # Ensure provider is set
                self.cache_manager._initialize_cache(
                    config=cache_config,
                    agent_id=self.agent_id,
                    memory_id=self.memory_ids[0] if self.memory_ids else None,
                )
            else:
                # Create a new cache manager if it doesn't exist
                from .managers import CacheManager

                self.cache_manager = CacheManager(
                    enabled=True,
                    config=cache_config,
                    agent_id=self.agent_id,
                    memory_id=self.memory_ids[0] if self.memory_ids else None,
                    memory_provider=self.memory_provider,  # ✅ Pass memory provider
                )

            logger.info(
                f"Semantic cache enabled for agent {self.agent_id} with threshold={threshold}, scope={scope}"
            )
            return self

        except Exception as e:
            logger.error(f"Failed to enable semantic cache: {e}")
            raise

    def disable_semantic_cache(self):
        """
        Disable semantic cache on the agent.

        Returns:
            Self for method chaining
        """
        if self.cache_manager:
            self.cache_manager.enabled = False
            logger.info(f"Semantic cache disabled for agent {self.agent_id}")
        return self

    # Additional methods for compatibility
    def load_conversation_history(self, memory_id: str = None):
        """Load conversation history (delegated to memory manager)."""
        if self.memory_manager:
            memory_id = (
                memory_id
                or self._current_memory_id
                or (self.memory_ids[0] if self.memory_ids else None)
            )
            if memory_id:
                return self.memory_manager.load_conversation_history(memory_id)
        return []

    def add_tool(self, tool, persist: bool = False):
        """Add a tool (delegated to tool manager)."""
        return self.tool_manager.add_tool(tool, persist)

    def set_persona(self, persona, save: bool = True):
        """Set persona (delegated to persona manager)."""
        return self.persona_manager.set_persona(persona, self.agent_id, save)

    def save(self):
        """
        Store the memagent in the memory provider.

        This method stores the memagent configuration and state in the memory provider,
        allowing for persistence and later restoration of the agent.

        Returns:
            MemAgent: Self, for method chaining
        """
        if not self.memory_provider:
            raise ValueError("Cannot save MemAgent: no memory provider configured")

        try:
            # Serialize tools from tool manager
            tools_to_save = self._serialize_tools_for_save()

            # Convert delegates to agent IDs for persistence
            delegate_ids = self._serialize_delegates_for_save()

            # Get semantic cache config for saving
            semantic_cache_config_to_save = self._serialize_semantic_cache_config()

            # Create MemAgentModel with current configuration
            from .models import MemAgentModel

            memagent_to_save = MemAgentModel(
                llm_config=(
                    self.model.get_config()
                    if self.model and hasattr(self.model, "get_config")
                    else None
                ),
                instruction=self.instruction,
                max_steps=self.max_steps,
                memory_ids=self.memory_ids,
                agent_id=self.agent_id,
                persona=(
                    self.persona_manager.current_persona
                    if self.persona_manager
                    else None
                ),
                tools=tools_to_save,
                delegates=delegate_ids if delegate_ids else None,
                semantic_cache=(
                    self.cache_manager.enabled if self.cache_manager else False
                ),
                semantic_cache_config=semantic_cache_config_to_save,
            )

            # Save or update the agent
            if hasattr(self.memory_provider, "store_memagent"):
                if self.agent_id and hasattr(self.memory_provider, "retrieve_memagent"):
                    # Check if agent exists for update vs create
                    try:
                        existing = self.memory_provider.retrieve_memagent(self.agent_id)
                        if existing and hasattr(
                            self.memory_provider, "update_memagent"
                        ):
                            saved_memagent = self.memory_provider.update_memagent(
                                memagent_to_save
                            )
                        else:
                            saved_memagent = self.memory_provider.store_memagent(
                                memagent_to_save
                            )
                    except Exception:
                        # Agent doesn't exist, create new
                        saved_memagent = self.memory_provider.store_memagent(
                            memagent_to_save
                        )
                else:
                    # New agent
                    saved_memagent = self.memory_provider.store_memagent(
                        memagent_to_save
                    )

                # Update agent_id if it was generated
                if not self.agent_id and saved_memagent.get("_id"):
                    self.agent_id = str(saved_memagent["_id"])

                    # Update semantic cache with new agent_id
                    if self.cache_manager and self.cache_manager.enabled:
                        if (
                            hasattr(self.cache_manager, "cache_instance")
                            and self.cache_manager.cache_instance
                        ):
                            self.cache_manager.cache_instance.agent_id = self.agent_id
                            if self.memory_ids:
                                self.cache_manager.cache_instance.memory_id = (
                                    self.memory_ids[0]
                                )

                logger.info(f"MemAgent {self.agent_id} saved successfully")
                return self
            else:
                raise ValueError("Memory provider does not support saving MemAgent")

        except Exception as e:
            logger.error(f"Failed to save MemAgent {self.agent_id}: {e}")
            raise

    def _serialize_tools_for_save(self):
        """Serialize tools for saving."""
        if not self.tool_manager:
            return None

        tools_metadata = self.tool_manager.get_tool_metadata()
        if not tools_metadata:
            return None

        # Determine the memory_id to use for tools
        # Priority: current memory_id → first memory_id → generate new
        agent_memory_id = self._current_memory_id or (
            self.memory_ids[0] if self.memory_ids else None
        )

        # If no memory_id exists, generate one and add it to the agent
        if not agent_memory_id:
            agent_memory_id = str(uuid.uuid4())
            self.memory_ids.append(agent_memory_id)
            self._current_memory_id = agent_memory_id
            logger.info(
                f"Generated new memory_id for agent {self.agent_id}: {agent_memory_id}"
            )

        # Convert tool metadata to serializable format with all necessary fields for TOOLBOX table
        serializable_tools = []
        for tool_meta in tools_metadata:
            if isinstance(tool_meta, dict):
                serializable_tool = {
                    "_id": tool_meta.get("_id")
                    or tool_meta.get("name"),  # Use name as fallback ID
                    "name": tool_meta.get("name"),
                    "description": tool_meta.get("description", ""),
                    "signature": tool_meta.get("signature", ""),
                    "docstring": tool_meta.get(
                        "docstring", tool_meta.get("description", "")
                    ),
                    "parameters": tool_meta.get("parameters", {}),
                    "type": tool_meta.get("type", "function"),
                    "memory_id": tool_meta.get("memory_id")
                    or agent_memory_id,  # Use agent's memory_id
                }
                serializable_tools.append(serializable_tool)

        return serializable_tools if serializable_tools else None

    def _serialize_delegates_for_save(self):
        """Serialize delegate agents for saving."""
        # Note: In the new architecture, delegates would be handled differently
        # This is a placeholder for compatibility
        if hasattr(self, "delegates") and self.delegates:
            delegate_ids = []
            for delegate in self.delegates:
                if hasattr(delegate, "agent_id") and delegate.agent_id:
                    delegate_ids.append(delegate.agent_id)
                    # Ensure delegate is saved
                    try:
                        if hasattr(delegate, "save"):
                            delegate.save()
                    except Exception as e:
                        logger.warning(
                            f"Failed to save delegate {delegate.agent_id}: {e}"
                        )
            return delegate_ids
        return None

    def _serialize_semantic_cache_config(self):
        """Serialize semantic cache configuration for saving."""
        if not self.cache_manager or not self.cache_manager.enabled:
            return None

        try:
            if (
                hasattr(self.cache_manager, "cache_instance")
                and self.cache_manager.cache_instance
            ):
                if hasattr(self.cache_manager.cache_instance, "config"):
                    config = self.cache_manager.cache_instance.config
                    if hasattr(config, "__dict__"):
                        config_dict = config.__dict__.copy()
                        # Convert enums to strings for serialization
                        for key, value in config_dict.items():
                            if hasattr(value, "value"):  # Enum
                                config_dict[key] = value.value
                        return config_dict
        except Exception as e:
            logger.warning(f"Failed to serialize semantic cache config: {e}")

        return None

    @classmethod
    def load(cls, agent_id: str, memory_provider=None, **overrides):
        """
        Load a MemAgent from the memory provider.

        Args:
            agent_id (str): The agent ID to load
            memory_provider: Memory provider to use (optional)
            **overrides: Override parameters for the loaded agent

        Returns:
            MemAgent: The loaded agent instance
        """
        if not memory_provider:
            # Try to import default memory provider
            try:
                from ..memory_provider import MemoryProvider

                memory_provider = MemoryProvider()
            except ImportError:
                raise ValueError(
                    "No memory provider specified and default MemoryProvider not available"
                )

        if not hasattr(memory_provider, "retrieve_memagent"):
            raise ValueError("Memory provider does not support loading MemAgent")

        logger.info(f"Loading MemAgent with agent id {agent_id}...")

        # Retrieve the saved agent
        saved_memagent = memory_provider.retrieve_memagent(agent_id)
        if not saved_memagent:
            raise ValueError(
                f"MemAgent with agent id {agent_id} not found in the memory provider"
            )

        # Reconstruct LLM model
        model_to_load = None
        if hasattr(saved_memagent, "llm_config") and saved_memagent.llm_config:
            try:
                model_to_load = create_llm_provider(saved_memagent.llm_config)
            except Exception as e:
                logger.warning(
                    f"Could not load model from config: {e}. Model will be None."
                )
        elif hasattr(saved_memagent, "model") and saved_memagent.model:
            model_to_load = saved_memagent.model

        # Load delegates if they exist
        loaded_delegates = None
        if hasattr(saved_memagent, "delegates") and saved_memagent.delegates:
            loaded_delegates = []
            for delegate_id in saved_memagent.delegates:
                try:
                    delegate_agent = cls.load(delegate_id, memory_provider)
                    loaded_delegates.append(delegate_agent)
                except Exception as e:
                    logger.warning(f"Could not load delegate agent {delegate_id}: {e}")

        # Reconstruct semantic cache config
        semantic_cache_config_to_load = None
        if (
            hasattr(saved_memagent, "semantic_cache_config")
            and saved_memagent.semantic_cache_config
        ):
            try:
                config_dict = dict(saved_memagent.semantic_cache_config)
                # Convert string scope back to enum if needed
                if "scope" in config_dict and isinstance(config_dict["scope"], str):
                    try:
                        from ..enums.semantic_cache_scope import SemanticCacheScope

                        scope_str = config_dict["scope"].lower()
                        if scope_str == "local":
                            config_dict["scope"] = SemanticCacheScope.LOCAL
                        elif scope_str == "global":
                            config_dict["scope"] = SemanticCacheScope.GLOBAL
                    except ImportError:
                        # If enum not available, keep as string
                        pass
                semantic_cache_config_to_load = config_dict
            except Exception as e:
                logger.warning(f"Failed to reconstruct semantic cache config: {e}")

        # Create new agent instance with loaded configuration
        agent_instance = cls(
            model=overrides.get("model", model_to_load),
            tools=overrides.get("tools", getattr(saved_memagent, "tools", None)),
            persona=overrides.get("persona", getattr(saved_memagent, "persona", None)),
            instruction=overrides.get(
                "instruction", getattr(saved_memagent, "instruction", None)
            ),
            max_steps=overrides.get(
                "max_steps", getattr(saved_memagent, "max_steps", DEFAULT_MAX_STEPS)
            ),
            memory_ids=overrides.get(
                "memory_ids", getattr(saved_memagent, "memory_ids", [])
            ),
            agent_id=agent_id,
            memory_provider=memory_provider,
            delegates=overrides.get("delegates", loaded_delegates),
            semantic_cache=overrides.get(
                "semantic_cache", getattr(saved_memagent, "semantic_cache", False)
            ),
            semantic_cache_config=overrides.get(
                "semantic_cache_config", semantic_cache_config_to_load
            ),
        )

        logger.info(f"MemAgent loaded successfully with agent_id: {agent_id}")
        return agent_instance

    def refresh(self):
        """
        Refresh the MemAgent from the memory provider.

        This method reloads the agent configuration from the memory provider,
        updating the current instance with any changes.

        Returns:
            MemAgent: Self if successful, False if failed
        """
        if not self.memory_provider:
            logger.error("Cannot refresh MemAgent: no memory provider configured")
            return False

        if not self.agent_id:
            logger.error("Cannot refresh MemAgent: no agent_id set")
            return False

        try:
            # Load fresh configuration from memory provider
            if hasattr(self.memory_provider, "retrieve_memagent"):
                saved_memagent = self.memory_provider.retrieve_memagent(self.agent_id)
                if saved_memagent:
                    # Update configuration attributes
                    if hasattr(saved_memagent, "instruction"):
                        self.instruction = saved_memagent.instruction
                    if hasattr(saved_memagent, "max_steps"):
                        self.max_steps = saved_memagent.max_steps
                    if hasattr(saved_memagent, "memory_ids"):
                        self.memory_ids = saved_memagent.memory_ids

                    # Update persona if changed
                    if hasattr(saved_memagent, "persona") and self.persona_manager:
                        self.persona_manager.set_persona(
                            saved_memagent.persona, self.agent_id, save=False
                        )

                    logger.info(f"MemAgent {self.agent_id} refreshed successfully")
                    return self
                else:
                    logger.error(
                        f"MemAgent {self.agent_id} not found in memory provider"
                    )
                    return False
            else:
                logger.error("Memory provider does not support retrieving MemAgent")
                return False

        except Exception as e:
            logger.error(f"Error refreshing MemAgent {self.agent_id}: {e}")
            return False

    def generate_summaries(
        self, days_back: int = 7, max_memories_per_summary: int = 50
    ) -> List[str]:
        """
        Generate summaries by compressing memory units from a specified time period.

        This method collects memory units from the specified time period,
        uses an LLM to compress them into emotionally and situationally relevant summaries,
        and stores them in the summaries collection.

        Parameters:
        -----------
        days_back : int, optional
            Number of days back to include in the summary (default: 7)
        max_memories_per_summary : int, optional
            Maximum number of memory units to include in each summary (default: 50)

        Returns:
        --------
        List[str]
            List of summary IDs that were created
        """
        try:
            import time

            from ..embeddings import get_embedding

            # Calculate time range (days_back days ago to now)
            current_time = time.time()
            start_time = current_time - (days_back * 24 * 60 * 60)

            logger.info(
                f"Generating summaries for agent {self.agent_id} from {days_back} days back"
            )
            logger.info(f"Agent memory_ids: {self.memory_ids}")
            logger.info(f"Current memory_id: {self._current_memory_id}")
            logger.info(f"Time range: {start_time} to {current_time}")

            # Ensure we have memory IDs to search
            memory_ids_to_search = self.memory_ids or []
            if (
                self._current_memory_id
                and self._current_memory_id not in memory_ids_to_search
            ):
                memory_ids_to_search = [self._current_memory_id] + memory_ids_to_search

            if not memory_ids_to_search:
                logger.warning(
                    f"Agent {self.agent_id} has no memory_ids to search for summaries"
                )
                return []

            logger.info(
                f"Searching {len(memory_ids_to_search)} memory_ids: {memory_ids_to_search}"
            )

            # Collect conversation memories from all memory IDs
            all_memories = []
            for memory_id in memory_ids_to_search:
                logger.info(
                    f"Retrieving conversation history for memory_id: {memory_id}"
                )
                try:
                    # Retrieve all conversation history
                    memories = self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(
                        memory_id=memory_id, include_embedding=False
                    )

                    if memories:
                        logger.info(
                            f"Retrieved {len(memories)} raw memories for memory_id: {memory_id}"
                        )

                        # Filter by time range
                        filtered = []
                        for idx, mem in enumerate(memories):
                            mem_timestamp = mem.get("timestamp")
                            original_timestamp = mem_timestamp

                            # Convert timestamp to float if needed
                            if isinstance(mem_timestamp, str):
                                try:
                                    from datetime import datetime

                                    # Try multiple timestamp formats
                                    if "T" in mem_timestamp:
                                        # ISO format
                                        mem_timestamp = datetime.fromisoformat(
                                            mem_timestamp.replace("Z", "+00:00")
                                        ).timestamp()
                                    else:
                                        # Try parsing as float string
                                        mem_timestamp = float(mem_timestamp)
                                except Exception as e:
                                    logger.warning(
                                        f"Could not parse timestamp '{original_timestamp}' at index {idx}: {e}"
                                    )
                                    continue
                            elif hasattr(mem_timestamp, "timestamp"):
                                # datetime object
                                mem_timestamp = mem_timestamp.timestamp()
                            elif not isinstance(mem_timestamp, (int, float)):
                                logger.warning(
                                    f"Unknown timestamp type at index {idx}: {type(mem_timestamp)} = {original_timestamp}"
                                )
                                continue

                            # Convert to float
                            mem_timestamp = float(mem_timestamp)

                            # Debug first few timestamps
                            if idx < 3:
                                logger.info(
                                    f"Memory {idx}: timestamp={mem_timestamp}, start_time={start_time}, current_time={current_time}, in_range={start_time <= mem_timestamp <= current_time}"
                                )

                            if start_time <= mem_timestamp <= current_time:
                                filtered.append(mem)

                        logger.info(
                            f"Found {len(filtered)} memories within time range (out of {len(memories)} total) for memory_id: {memory_id}"
                        )
                        all_memories.extend(filtered)
                    else:
                        logger.info(f"No memories returned for memory_id: {memory_id}")
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve memories for memory_id {memory_id}: {e}"
                    )
                    import traceback

                    logger.debug(traceback.format_exc())

            if not all_memories:
                logger.info(
                    f"No memories found for agent {self.agent_id} in the specified time range"
                )
                return []

            # Sort memories by timestamp
            def get_timestamp(mem):
                ts = mem.get("timestamp", 0)
                if isinstance(ts, str):
                    try:
                        from datetime import datetime

                        return datetime.fromisoformat(
                            ts.replace("Z", "+00:00")
                        ).timestamp()
                    except (ValueError, Exception):
                        return 0
                return float(ts) if isinstance(ts, (int, float)) else 0

            all_memories.sort(key=get_timestamp)

            logger.info(f"Found {len(all_memories)} memory units to summarize")

            # Split memories into chunks and create summaries
            summary_ids = []
            for i in range(0, len(all_memories), max_memories_per_summary):
                memory_chunk = all_memories[i : i + max_memories_per_summary]

                # Generate summary for this chunk
                summary_content = self._compress_memories_with_llm(memory_chunk)

                if summary_content:
                    # Get timestamps for period
                    period_start = get_timestamp(memory_chunk[0])
                    period_end = get_timestamp(memory_chunk[-1])

                    # Get the memory_id from the first memory in the chunk
                    chunk_memory_id = memory_chunk[0].get("memory_id")
                    if not chunk_memory_id:
                        # Fallback to current memory_id or first in list
                        chunk_memory_id = self._current_memory_id or (
                            self.memory_ids[0] if self.memory_ids else "default"
                        )

                    # Create summary document
                    summary_doc = {
                        "memory_id": chunk_memory_id,
                        "agent_id": self.agent_id,
                        "content": summary_content,
                        "period_start": period_start,
                        "period_end": period_end,
                        "memory_units_count": len(memory_chunk),
                        "summary_type": "automatic",
                        "created_at": current_time,
                        "embedding": get_embedding(summary_content),
                    }

                    # Store summary
                    summary_id = self.memory_provider.store(
                        summary_doc, MemoryType.SUMMARIES
                    )
                    summary_ids.append(summary_id)

                    logger.info(
                        f"Created summary {summary_id} for memory_id {chunk_memory_id} covering {len(memory_chunk)} memories"
                    )

            logger.info(
                f"Generated {len(summary_ids)} summaries for agent {self.agent_id}"
            )
            return summary_ids

        except Exception as e:
            logger.error(f"Error generating summaries: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def _compress_memories_with_llm(self, memories: List[Dict]) -> str:
        """
        Use LLM to compress memory units into an emotionally and situationally relevant summary.

        Parameters:
        -----------
        memories : List[Dict]
            List of memory units to compress

        Returns:
        --------
        str
            Compressed summary content
        """
        try:
            # Extract content from memories
            memory_contents = []
            for memory in memories:
                content = memory.get("content", "")
                role = memory.get("role", "")

                if content:
                    if role:
                        memory_contents.append(f"[{role}]: {content}")
                    else:
                        memory_contents.append(content)

            if not memory_contents:
                return ""

            # Create compression prompt
            memories_text = "\n".join(memory_contents)
            compression_prompt = f"""
Analyze the following memory units and create a concise summary that captures:
1. Emotionally significant moments and interactions
2. Situationally relevant context and patterns
3. Key achievements, challenges, or learning experiences
4. Important facts and information learned

Memory Units:
{memories_text}

Provide a comprehensive but concise summary:"""

            # Use the LLM to generate the summary
            if self.model:
                messages = [{"role": "user", "content": compression_prompt}]
                summary = self.model.generate(messages)
                return summary.strip()
            else:
                logger.warning("No LLM model available for memory compression")
                return ""

        except Exception as e:
            logger.error(f"Error compressing memories with LLM: {e}")
            return ""
