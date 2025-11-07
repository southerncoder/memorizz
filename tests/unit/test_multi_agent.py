"""Tests for multi-agent functionality and coordination."""
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import pytest

from memorizz.enums import Role
from memorizz.memagent import MemAgent
from tests.conftest import assert_agent_response_valid, assert_agent_state_valid
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider


class TestMultiAgentBasics:
    """Test basic multi-agent functionality."""

    @pytest.mark.multi_agent
    def test_create_multiple_agents(self):
        """Test creating multiple MemAgent instances."""
        agents = []

        for i in range(3):
            llm_provider = MockLLMProvider([f"I am agent {i+1}, ready to help."])
            memory_provider = MockMemoryProvider()

            agent = MemAgent(
                model=llm_provider,
                memory_provider=memory_provider,
                instruction=f"You are agent {i+1}, specialized in task type {i+1}.",
                agent_id=f"agent_{i+1}",
            )
            agents.append(agent)

        assert len(agents) == 3
        for i, agent in enumerate(agents):
            assert_agent_state_valid(agent)
            assert agent.agent_id == f"agent_{i+1}"
            assert f"agent {i+1}" in agent.instruction

    @pytest.mark.multi_agent
    def test_agents_independent_memory(self):
        """Test that agents maintain independent memory."""
        llm1 = MockLLMProvider(["I'll remember your name is Alice."])
        llm2 = MockLLMProvider(["I'll remember your name is Bob."])

        memory1 = MockMemoryProvider()
        memory2 = MockMemoryProvider()

        agent1 = MemAgent(
            model=llm1,
            memory_provider=memory1,
            instruction="You are agent 1.",
            agent_id="agent_memory_1",
        )

        agent2 = MemAgent(
            model=llm2,
            memory_provider=memory2,
            instruction="You are agent 2.",
            agent_id="agent_memory_2",
        )

        # Each agent learns different information
        response1 = agent1.run("My name is Alice", memory_id="mem1")
        response2 = agent2.run("My name is Bob", memory_id="mem2")

        assert_agent_response_valid(response1)
        assert_agent_response_valid(response2)

        # Verify separate memory providers
        assert agent1.memory_provider != agent2.memory_provider
        assert len(memory1.storage["mem1"]) == 2  # user + assistant
        assert len(memory2.storage["mem2"]) == 2  # user + assistant

        # Memories should be different
        alice_memory = memory1.storage["mem1"]
        bob_memory = memory2.storage["mem2"]

        assert alice_memory != bob_memory

    @pytest.mark.multi_agent
    def test_agents_shared_memory_provider(self):
        """Test agents sharing the same memory provider with different memory IDs."""
        shared_memory = MockMemoryProvider()

        llm1 = MockLLMProvider(["Agent 1 here, I can help with math."])
        llm2 = MockLLMProvider(["Agent 2 here, I can help with writing."])

        agent1 = MemAgent(
            model=llm1,
            memory_provider=shared_memory,
            instruction="You are a math specialist.",
            agent_id="math_agent",
        )

        agent2 = MemAgent(
            model=llm2,
            memory_provider=shared_memory,
            instruction="You are a writing specialist.",
            agent_id="writing_agent",
        )

        # Agents interact with different memory spaces
        response1 = agent1.run("Help me with algebra", memory_id="math_session")
        response2 = agent2.run("Help me write an essay", memory_id="writing_session")

        assert_agent_response_valid(response1)
        assert_agent_response_valid(response2)

        # Both use same provider but different memory spaces
        assert agent1.memory_provider == agent2.memory_provider
        assert "math_session" in shared_memory.storage
        assert "writing_session" in shared_memory.storage
        assert len(shared_memory.storage) == 2


class TestMultiAgentCoordination:
    """Test multi-agent coordination patterns."""

    @pytest.mark.multi_agent
    def test_sequential_agent_handoff(self):
        """Test passing work between agents sequentially."""
        # Agent 1: Data collector
        collector_responses = ["I've collected the data: [1, 2, 3, 4, 5]"]
        collector = MemAgent(
            model=MockLLMProvider(collector_responses),
            memory_provider=MockMemoryProvider(),
            instruction="You collect and prepare data.",
            agent_id="data_collector",
        )

        # Agent 2: Data processor
        processor_responses = ["I've processed the data. Average: 3.0, Sum: 15"]
        processor = MemAgent(
            model=MockLLMProvider(processor_responses),
            memory_provider=MockMemoryProvider(),
            instruction="You process and analyze data.",
            agent_id="data_processor",
        )

        # Agent 3: Report generator
        reporter_responses = [
            "Final Report: The data analysis shows average=3.0, total=15"
        ]
        reporter = MemAgent(
            model=MockLLMProvider(reporter_responses),
            memory_provider=MockMemoryProvider(),
            instruction="You generate final reports.",
            agent_id="report_generator",
        )

        # Sequential workflow
        step1 = collector.run("Collect sample data")
        step2 = processor.run(f"Process this data: {step1}")
        step3 = reporter.run(f"Generate report from: {step2}")

        assert_agent_response_valid(step1)
        assert_agent_response_valid(step2)
        assert_agent_response_valid(step3)

        # Verify information flows through the pipeline
        assert "data" in step1.lower()
        assert "processed" in step2.lower()
        assert "report" in step3.lower()

    @pytest.mark.multi_agent
    def test_parallel_agent_execution(self):
        """Test multiple agents working on tasks in parallel."""

        def create_specialist_agent(specialty, task_response):
            return MemAgent(
                model=MockLLMProvider([task_response]),
                memory_provider=MockMemoryProvider(),
                instruction=f"You are a {specialty} specialist.",
                agent_id=f"{specialty}_specialist",
            )

        # Create specialist agents
        agents = {
            "math": create_specialist_agent("math", "Math result: 42"),
            "science": create_specialist_agent("science", "Science result: H2O"),
            "history": create_specialist_agent("history", "History result: 1776"),
        }

        # Execute tasks in parallel (simulated)
        tasks = {
            "math": "What is 6 * 7?",
            "science": "What is water's chemical formula?",
            "history": "When was the Declaration of Independence signed?",
        }

        results = {}
        for specialty, query in tasks.items():
            results[specialty] = agents[specialty].run(query)

        # Verify all agents completed their tasks
        assert len(results) == 3
        for specialty, result in results.items():
            assert_agent_response_valid(result)
            assert (
                specialty in result.lower()
                or tasks[specialty].split()[2].lower() in result.lower()
            )

    @pytest.mark.multi_agent
    def test_agent_delegation_pattern(self):
        """Test master agent delegating to specialist agents."""
        # Create specialist agents
        math_agent = MemAgent(
            model=MockLLMProvider(["The answer is 15 (3 * 5)"]),
            memory_provider=MockMemoryProvider(),
            instruction="You are a math calculation specialist.",
            agent_id="math_specialist",
        )

        writing_agent = MemAgent(
            model=MockLLMProvider(["Here's a short story about space exploration..."]),
            memory_provider=MockMemoryProvider(),
            instruction="You are a creative writing specialist.",
            agent_id="writing_specialist",
        )

        # Master coordinator agent with delegates
        master_responses = [
            "I'll delegate the math question to my math specialist.",
            "I'll delegate the writing task to my writing specialist.",
        ]

        master_agent = MemAgent(
            model=MockLLMProvider(master_responses),
            memory_provider=MockMemoryProvider(),
            instruction="You coordinate and delegate tasks to specialists.",
            agent_id="master_coordinator",
            delegates=[
                math_agent,
                writing_agent,
            ],  # Note: This assumes delegate support
        )

        # Test delegation (manual simulation for now)
        math_query = "What is 3 * 5?"
        writing_query = "Write a short story about space"

        # Master decides what to do, then delegates
        master_response1 = master_agent.run(f"Handle this math question: {math_query}")
        specialist_result1 = math_agent.run(math_query)

        master_response2 = master_agent.run(
            f"Handle this writing task: {writing_query}"
        )
        specialist_result2 = writing_agent.run(writing_query)

        assert_agent_response_valid(master_response1)
        assert_agent_response_valid(specialist_result1)
        assert_agent_response_valid(master_response2)
        assert_agent_response_valid(specialist_result2)

        assert "15" in specialist_result1
        assert "story" in specialist_result2.lower()


class TestMultiAgentCommunication:
    """Test communication patterns between agents."""

    @pytest.mark.multi_agent
    def test_shared_memory_communication(self):
        """Test agents communicating through shared memory."""
        shared_memory = MockMemoryProvider()
        shared_memory_id = "shared_conversation"

        # Agent A - Asks a question
        agent_a = MemAgent(
            model=MockLLMProvider(
                ["I have a question for Agent B: What's the weather?"]
            ),
            memory_provider=shared_memory,
            instruction="You ask questions and wait for responses.",
            agent_id="questioner",
        )

        # Agent B - Provides answers
        agent_b = MemAgent(
            model=MockLLMProvider(
                ["I see Agent A asked about weather. It's sunny today!"]
            ),
            memory_provider=shared_memory,
            instruction="You answer questions from other agents.",
            agent_id="responder",
        )

        # Simulate communication through shared memory
        response_a = agent_a.run("Ask about the weather", memory_id=shared_memory_id)

        # Agent B reads the shared memory and responds
        # First, let B load the conversation history to see A's question
        history = agent_b.load_conversation_history(shared_memory_id)
        response_b = agent_b.run(
            "Respond to the weather question", memory_id=shared_memory_id
        )

        assert_agent_response_valid(response_a)
        assert_agent_response_valid(response_b)

        # Verify both agents wrote to shared memory
        shared_conversation = shared_memory.storage[shared_memory_id]
        assert len(shared_conversation) == 4  # 2 user queries + 2 agent responses

    @pytest.mark.multi_agent
    def test_message_passing_pattern(self):
        """Test explicit message passing between agents."""
        # Agent 1: Sender
        sender = MemAgent(
            model=MockLLMProvider(["Message sent: Please analyze this data set."]),
            memory_provider=MockMemoryProvider(),
            instruction="You send messages to other agents.",
            agent_id="sender",
        )

        # Agent 2: Receiver
        receiver = MemAgent(
            model=MockLLMProvider(["Message received. Analysis: The data looks good."]),
            memory_provider=MockMemoryProvider(),
            instruction="You receive and process messages from other agents.",
            agent_id="receiver",
        )

        # Simulate message passing
        message = "Please analyze this data: [1, 2, 3, 4, 5]"

        sent_response = sender.run(f"Send this message: {message}")
        received_response = receiver.run(f"Process received message: {message}")

        assert_agent_response_valid(sent_response)
        assert_agent_response_valid(received_response)

        assert "sent" in sent_response.lower() or "message" in sent_response.lower()
        assert (
            "received" in received_response.lower()
            or "analysis" in received_response.lower()
        )

    @pytest.mark.multi_agent
    def test_broadcast_communication(self):
        """Test one agent broadcasting to multiple agents."""
        shared_memory = MockMemoryProvider()
        broadcast_memory_id = "broadcast_channel"

        # Broadcaster agent
        broadcaster = MemAgent(
            model=MockLLMProvider(["Broadcasting: Emergency meeting at 3 PM today!"]),
            memory_provider=shared_memory,
            instruction="You broadcast messages to all agents.",
            agent_id="broadcaster",
        )

        # Multiple listener agents
        listeners = []
        for i in range(3):
            listener = MemAgent(
                model=MockLLMProvider(
                    [f"Listener {i+1}: Message received, will attend meeting."]
                ),
                memory_provider=shared_memory,
                instruction=f"You are listener {i+1}, you receive broadcast messages.",
                agent_id=f"listener_{i+1}",
            )
            listeners.append(listener)

        # Broadcast message
        broadcast_response = broadcaster.run(
            "Send emergency meeting notice", memory_id=broadcast_memory_id
        )

        # All listeners receive and acknowledge
        listener_responses = []
        for listener in listeners:
            response = listener.run(
                "Acknowledge broadcast message", memory_id=broadcast_memory_id
            )
            listener_responses.append(response)

        assert_agent_response_valid(broadcast_response)
        assert (
            "broadcasting" in broadcast_response.lower()
            or "meeting" in broadcast_response.lower()
        )

        for response in listener_responses:
            assert_agent_response_valid(response)
            assert any(
                word in response.lower()
                for word in ["received", "message", "meeting", "attend"]
            )

        # Verify all interactions in shared memory
        broadcast_history = shared_memory.storage[broadcast_memory_id]
        assert (
            len(broadcast_history) == 8
        )  # 1 broadcaster + 3 listeners, each with user query + response


class TestMultiAgentErrorHandling:
    """Test error handling in multi-agent scenarios."""

    @pytest.mark.multi_agent
    def test_agent_failure_isolation(self):
        """Test that failure of one agent doesn't affect others."""
        # Working agent
        working_agent = MemAgent(
            model=MockLLMProvider(["I'm working normally."]),
            memory_provider=MockMemoryProvider(),
            instruction="You are a reliable agent.",
            agent_id="working_agent",
        )

        # Failing agent
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("Agent malfunction")

        failing_agent = MemAgent(
            model=failing_llm,
            memory_provider=MockMemoryProvider(),
            instruction="This agent will fail.",
            agent_id="failing_agent",
        )

        # Test both agents
        working_response = working_agent.run("Are you working?")
        failing_response = failing_agent.run("This will fail")

        # Working agent should succeed
        assert_agent_response_valid(working_response)
        assert "working" in working_response.lower()

        # Failing agent should return error message, not crash
        assert isinstance(failing_response, str)
        assert "error" in failing_response.lower()

    @pytest.mark.multi_agent
    def test_memory_conflict_resolution(self):
        """Test handling of memory conflicts between agents."""
        shared_memory = MockMemoryProvider()

        # Two agents trying to use same memory ID simultaneously
        agent1 = MemAgent(
            model=MockLLMProvider(["Agent 1 storing data..."]),
            memory_provider=shared_memory,
            instruction="You are agent 1.",
            agent_id="concurrent_agent_1",
        )

        agent2 = MemAgent(
            model=MockLLMProvider(["Agent 2 storing data..."]),
            memory_provider=shared_memory,
            instruction="You are agent 2.",
            agent_id="concurrent_agent_2",
        )

        memory_id = "contested_memory"

        # Both agents try to use the same memory space
        response1 = agent1.run("Store my data", memory_id=memory_id)
        response2 = agent2.run("Store my data", memory_id=memory_id)

        assert_agent_response_valid(response1)
        assert_agent_response_valid(response2)

        # Memory should contain both interactions
        contested_memory = shared_memory.storage[memory_id]
        assert len(contested_memory) >= 4  # At least both agent interactions

    @pytest.mark.multi_agent
    def test_cascading_failure_prevention(self):
        """Test preventing cascading failures in agent chains."""
        memory_provider = MockMemoryProvider()

        # Agent 1: Works normally
        agent1 = MemAgent(
            model=MockLLMProvider(["Step 1 completed successfully."]),
            memory_provider=memory_provider,
            instruction="You complete step 1.",
            agent_id="step1_agent",
        )

        # Agent 2: Fails
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("Step 2 failed")

        agent2 = MemAgent(
            model=failing_llm,
            memory_provider=memory_provider,
            instruction="You complete step 2.",
            agent_id="step2_agent",
        )

        # Agent 3: Should handle previous failure gracefully
        agent3 = MemAgent(
            model=MockLLMProvider(["Step 3: Handling previous error gracefully."]),
            memory_provider=memory_provider,
            instruction="You complete step 3 and handle errors.",
            agent_id="step3_agent",
        )

        # Execute chain
        result1 = agent1.run("Execute step 1")
        result2 = agent2.run("Execute step 2")  # This will fail
        result3 = agent3.run(f"Execute step 3, previous result: {result2}")

        assert_agent_response_valid(result1)
        assert "step 1" in result1.lower() and "completed" in result1.lower()

        # Step 2 should fail gracefully
        assert isinstance(result2, str)
        assert "error" in result2.lower()

        # Step 3 should continue despite step 2 failure
        assert_agent_response_valid(result3)
        assert "step 3" in result3.lower()


class TestMultiAgentScenarios:
    """Test realistic multi-agent scenarios."""

    @pytest.mark.multi_agent
    def test_research_team_scenario(self):
        """Test a team of research agents working together."""
        shared_memory = MockMemoryProvider()
        project_memory_id = "research_project"

        # Research coordinator
        coordinator = MemAgent(
            model=MockLLMProvider(
                ["Research project initiated. Assigning tasks to team."]
            ),
            memory_provider=shared_memory,
            instruction="You coordinate research projects.",
            agent_id="research_coordinator",
        )

        # Literature reviewer
        reviewer = MemAgent(
            model=MockLLMProvider(
                ["Literature review completed. Found 15 relevant papers."]
            ),
            memory_provider=shared_memory,
            instruction="You review academic literature.",
            agent_id="literature_reviewer",
        )

        # Data analyst
        analyst = MemAgent(
            model=MockLLMProvider(
                ["Data analysis completed. Statistical significance found."]
            ),
            memory_provider=shared_memory,
            instruction="You analyze research data.",
            agent_id="data_analyst",
        )

        # Report writer
        writer = MemAgent(
            model=MockLLMProvider(["Research report drafted and ready for review."]),
            memory_provider=shared_memory,
            instruction="You write research reports.",
            agent_id="report_writer",
        )

        # Execute research workflow
        coord_response = coordinator.run(
            "Start new research project on AI ethics", memory_id=project_memory_id
        )
        review_response = reviewer.run(
            "Review literature on AI ethics", memory_id=project_memory_id
        )
        analysis_response = analyst.run(
            "Analyze ethics survey data", memory_id=project_memory_id
        )
        report_response = writer.run(
            "Compile research findings into report", memory_id=project_memory_id
        )

        # Verify all steps completed
        responses = [
            coord_response,
            review_response,
            analysis_response,
            report_response,
        ]
        for response in responses:
            assert_agent_response_valid(response)

        assert "research" in coord_response.lower()
        assert "literature" in review_response.lower()
        assert "analysis" in analysis_response.lower()
        assert "report" in report_response.lower()

        # Verify collaborative memory
        project_history = shared_memory.storage[project_memory_id]
        assert len(project_history) == 8  # 4 agents * (query + response)

    @pytest.mark.multi_agent
    def test_customer_support_scenario(self):
        """Test customer support agent coordination."""
        shared_memory = MockMemoryProvider()
        ticket_memory_id = "support_ticket_123"

        # Triage agent
        triage = MemAgent(
            model=MockLLMProvider(
                ["Ticket triaged: Technical issue, routing to Level 2."]
            ),
            memory_provider=shared_memory,
            instruction="You triage support tickets.",
            agent_id="triage_agent",
        )

        # Technical specialist
        tech_specialist = MemAgent(
            model=MockLLMProvider(
                ["Issue diagnosed: Software bug in version 2.1. Fix available."]
            ),
            memory_provider=shared_memory,
            instruction="You handle technical support issues.",
            agent_id="tech_specialist",
        )

        # Follow-up agent
        followup = MemAgent(
            model=MockLLMProvider(
                ["Follow-up completed: Customer satisfied with resolution."]
            ),
            memory_provider=shared_memory,
            instruction="You follow up on resolved tickets.",
            agent_id="followup_agent",
        )

        # Support workflow
        customer_issue = "My app crashes when I try to save files"

        triage_result = triage.run(
            f"Triage this issue: {customer_issue}", memory_id=ticket_memory_id
        )
        tech_result = tech_specialist.run(
            f"Investigate: {customer_issue}", memory_id=ticket_memory_id
        )
        followup_result = followup.run(
            "Check if customer issue is resolved", memory_id=ticket_memory_id
        )

        assert_agent_response_valid(triage_result)
        assert_agent_response_valid(tech_result)
        assert_agent_response_valid(followup_result)

        assert any(
            word in triage_result.lower() for word in ["triaged", "routing", "level"]
        )
        assert any(word in tech_result.lower() for word in ["diagnosed", "bug", "fix"])
        assert any(
            word in followup_result.lower()
            for word in ["follow", "satisfied", "resolved"]
        )

    @pytest.mark.multi_agent
    def test_content_creation_pipeline(self):
        """Test content creation pipeline with multiple agents."""
        # Each agent has independent memory for this pipeline

        # Content strategist
        strategist = MemAgent(
            model=MockLLMProvider(
                ["Content strategy: Focus on beginner tutorials, SEO-optimized."]
            ),
            memory_provider=MockMemoryProvider(),
            instruction="You create content strategies.",
            agent_id="content_strategist",
        )

        # Content writer
        writer = MemAgent(
            model=MockLLMProvider(
                ["Draft article completed: 'Python for Beginners: Getting Started'"]
            ),
            memory_provider=MockMemoryProvider(),
            instruction="You write articles and blog posts.",
            agent_id="content_writer",
        )

        # Editor
        editor = MemAgent(
            model=MockLLMProvider(
                [
                    "Article edited: Grammar corrected, flow improved, ready for publication."
                ]
            ),
            memory_provider=MockMemoryProvider(),
            instruction="You edit and improve written content.",
            agent_id="content_editor",
        )

        # SEO optimizer
        seo_agent = MemAgent(
            model=MockLLMProvider(
                ["SEO optimization complete: Keywords added, meta description written."]
            ),
            memory_provider=MockMemoryProvider(),
            instruction="You optimize content for SEO.",
            agent_id="seo_optimizer",
        )

        # Content pipeline execution
        topic = "Python programming for beginners"

        strategy = strategist.run(f"Create content strategy for: {topic}")
        draft = writer.run(f"Write article based on strategy: {strategy}")
        edited = editor.run(
            f"Edit this draft: {draft[:100]}..."
        )  # Truncate for brevity
        optimized = seo_agent.run(f"SEO optimize this content: {edited[:100]}...")

        pipeline_results = [strategy, draft, edited, optimized]
        for result in pipeline_results:
            assert_agent_response_valid(result)

        assert "strategy" in strategy.lower()
        assert any(word in draft.lower() for word in ["draft", "article", "python"])
        assert any(word in edited.lower() for word in ["edited", "grammar", "ready"])
        assert any(
            word in optimized.lower() for word in ["seo", "keywords", "optimization"]
        )
