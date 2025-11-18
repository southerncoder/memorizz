<div align="center">

# Memorizz 🧠

📊 **[Agent Memory Presentation](https://docs.google.com/presentation/d/1iSu667m5-pOXMrJq_LjkfnfD4V0rW1kbhGaQ2u3TKXQ/edit?usp=sharing)** | 🎥 **[AIEWF Richmond's Talk](https://youtu.be/W2HVdB4Jbjs?si=faaI3cMLc71Efpeu)**

[![PyPI version](https://badge.fury.io/py/memorizz.svg)](https://badge.fury.io/py/memorizz)
[![PyPI Downloads](https://static.pepy.tech/badge/memorizz)](https://pepy.tech/projects/memorizz)

</div>

> **⚠️ IMPORTANT WARNING ⚠️**
>
> **MemoRizz is an EXPERIMENTAL library intended for EDUCATIONAL PURPOSES ONLY.**
>
> **Do NOT use in production environments or with sensitive data.**
>
> This library is under active development, has not undergone security audits, and may contain bugs or breaking changes in future releases.

## Overview

**MemoRizz is a memory management framework for AI agents designed to create memory-augmented agents with explicit memory type allocation based on application mode.**

The framework enables developers to build context and memory aware agents capable of sophisticated information retrieval and storage.

MemoRizz provides flexible single and multi-agent architectures that allow you to instantiate agents with specifically allocated memory types—whether episodic, semantic, procedural, or working memory—tailored to your application's operational requirements.


**Why MemoRizz?**
- 🧠 **Persistent Memory**: Your AI agents remember conversations across sessions
- 🔍 **Semantic Search**: Find relevant information using natural language with AI Vector Search
- 🛠️ **Tool Integration**: Automatically discover and execute functions
- 👤 **Persona System**: Create consistent, specialized agent personalities
- 📁 **Local Filesystem Provider**: Zero-database option powered by FAISS for local development
- 🗄️ **Oracle AI Database**: Built-in integration with Oracle 23ai for advanced vector search and JSON Duality Views
- ⚡ **Semantic Cache**: Speed up responses and reduce costs with intelligent caching

## Key Features

- **Persistent Memory Management**: Long-term memory storage with semantic retrieval
- **MemAgent System**: Complete AI agents with memory, personas, and tools
- **Oracle AI Database Integration**: Leverages Oracle 23ai with native vector search and JSON Relational Duality Views
- **Tool Registration**: Automatically convert Python functions into LLM-callable tools
- **Persona Framework**: Create specialized agent personalities and behaviors
- **Vector Embeddings**: Semantic similarity search across all stored information using Oracle AI Vector Search
- **Semantic Cache**: Intelligent query-response caching with vector similarity matching

## Installation

```bash
pip install memorizz
```

### Prerequisites
- Python 3.7+
- Oracle Database 23ai or higher (for AI Vector Search and JSON Duality Views)
- OpenAI API key (for embeddings and LLM functionality)

Hugging Face embeddings/LLMs are included in the base install, so `pip install memorizz` gives you those dependencies automatically. For other optional stacks, install the matching extra:

```bash
pip install memorizz[ollama]       # Ollama embeddings
pip install memorizz[voyageai]     # VoyageAI embeddings
pip install memorizz[filesystem]   # Filesystem provider + FAISS vector search
```

### Oracle Database Setup

**📖 For complete setup instructions, see [SETUP.md](SETUP.md)**

**Quick Start:**

```bash
# 1. Start Oracle Database (uses lite version by default, 1.78GB)
./install_oracle.sh

# For Apple Silicon: export PLATFORM_FLAG="--platform linux/amd64" && ./install_oracle.sh

# To use full version (9.93GB): export ORACLE_IMAGE_TAG="latest" && ./install_oracle.sh

# 2. Set up database schema
# Option A: Use CLI command (recommended - works for pip-installed users)
memorizz setup-oracle

# Option B: Use examples script (repo-cloned users only)
# Note: CLI command (Option A) is recommended for most users
python examples/setup_oracle_user.py

# The setup automatically detects your database type:
# - Admin mode: Full setup with user creation (local/self-hosted)
# - User-only mode: Uses existing schema (hosted databases like FreeSQL.com)

# 3. Configure credentials (Option A: .env file - recommended)
cp .env.example .env
# Edit .env with your credentials

# Or Option B: Environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ORACLE_USER="memorizz_user"
export ORACLE_PASSWORD="SecurePass123!"
export ORACLE_DSN="localhost:1521/FREEPDB1"
```

**Default Credentials:**
- Admin User: `system`
- Admin Password: `MyPassword123!` (configurable via `ORACLE_ADMIN_PASSWORD`)
- MemoRizz User: `memorizz_user`
- MemoRizz Password: `SecurePass123!` (configurable via `ORACLE_PASSWORD`)

**Customizing Credentials:**

All credentials can be customized using environment variables. See [SETUP.md](SETUP.md) for details.

The setup script will:
- Create the `memorizz_user` with all required privileges
- Set up the relational schema (tables + indexes)
- Create JSON Relational Duality Views
- Verify the complete setup

## Quick Start

### 1. Basic MemAgent Setup

```python
import os
from memorizz.memory_provider.oracle import OracleProvider, OracleConfig
from memorizz.memagent.builders import MemAgentBuilder

# Set up your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Configure Oracle memory provider
oracle_config = OracleConfig(
    user="memorizz_user",
    password="SecurePass123!",
    dsn="localhost:1521/FREEPDB1",
    embedding_provider="openai",
    embedding_config={
        "model": "text-embedding-3-small",
        "api_key": os.environ["OPENAI_API_KEY"]
    }
)
oracle_provider = OracleProvider(oracle_config)

# Create a MemAgent using the builder pattern
agent = (MemAgentBuilder()
    .with_instruction("You are a helpful assistant with persistent memory.")
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

# Save the agent to Oracle
agent.save()

# Start conversing - the agent will remember across sessions
response = agent.run("Hello! My name is John and I'm a software engineer.")
print(response)

# Later in another session...
response = agent.run("What did I tell you about myself?")
print(response)  # Agent remembers John is a software engineer
```

### Filesystem Provider (local development)

Need a zero-dependency backend for local experiments? The filesystem provider persists every memory bucket as JSON files and uses FAISS for vector search (install with `pip install memorizz[filesystem]`). Switching providers is as simple as changing the config:

```python
from pathlib import Path
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

fs_config = FileSystemConfig(
    root_path=Path("~/.memorizz").expanduser(),
    embedding_provider="openai",   # Optional; omit to rely on keyword search
    embedding_config={"model": "text-embedding-3-small"},
)
filesystem_provider = FileSystemProvider(fs_config)

agent = (MemAgentBuilder()
    .with_instruction("You are a helpful assistant with local memory.")
    .with_memory_provider(filesystem_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)
```

All Oracle/Mongo-specific APIs continue to work—`MemAgent.save()`, semantic cache synchronization, and memagent CRUD now write to the configured directory. Vector search automatically falls back to keyword matching if embeddings or FAISS are not configured.

### 2. Enable Internet Access (Tavily – Preferred)

MemoRizz agents can browse the web by attaching an internet access provider. Tavily is the preferred default: the moment `TAVILY_API_KEY` exists MemoRizz wires up the `TavilyProvider` so Deep Research agents gain the `internet_search` and `open_web_page` tools automatically.

```python
import os
from memorizz.internet_access import TavilyProvider
from memorizz.memagent.builders import MemAgentBuilder

tavily = TavilyProvider(
    api_key=os.environ["TAVILY_API_KEY"],
    config={
        "search_depth": "advanced",
        "default_max_results": 8,
        "max_content_chars": 10000,
    },
)

agent = (
    MemAgentBuilder()
    .with_instruction("Use Tavily to research product feedback.")
    .with_memory_provider(oracle_provider)
    .with_llm_config(llm_config)
    .with_internet_access_provider(tavily)
    .build()
)

# Direct helper methods
news_results = agent.search_internet("Latest AI safety news", max_results=3)
article = agent.fetch_url("https://openai.com/blog")

# When running the agent, LLMs can call the browsing tools automatically
response = agent.run("Read the latest AI news and give me two key points.")
```

> ℹ️ Set `TAVILY_API_KEY` and (optionally) `MEMORIZZ_DEFAULT_INTERNET_PROVIDER=tavily`. MemoRizz automatically uses Tavily whenever the key is present and no other provider is supplied. Responses include `content_truncated` metadata when the `max_content_chars` limit trims an extracted page.

### 3. Enable Internet Access (Firecrawl)

Prefer Firecrawl's hybrid search/crawl pipeline? Swap in the bundled `FirecrawlProvider`.

```python
import os
from memorizz.internet_access import FirecrawlProvider

firecrawl = FirecrawlProvider(
    api_key=os.environ.get("FIRECRAWL_API_KEY"),
)

agent = (
    MemAgentBuilder()
    .with_instruction("Research assistant with access to the live web.")
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .with_internet_access_provider(firecrawl)
    .build()
)
```

> ℹ️ Set `FIRECRAWL_API_KEY` in the environment (or pass `api_key=` explicitly). Optional kwargs such as `base_url` and `timeout` can also be provided when instantiating `FirecrawlProvider`.
>
> ⚠️ Browsing APIs return the entire page body, which can overflow an LLM's context window. The `FirecrawlProvider` trims markdown content to 16,000 characters by default and annotates the response with `content_truncated` metadata. Adjust this behavior by passing a config dict (e.g., `max_content_chars`, `include_raw_response`, `max_raw_chars`) to keep scraped pages small enough for downstream prompts.

## Context Window Usage Tracking

MemoRizz agents now surface their token budget consumption so you can keep an eye on the active context window while the agent works.

- **Automatic logging**: Every LLM call records a line such as `Context window usage (iteration_1): 2150/128000 tokens (1.68%) | prompt=2100 completion=50` at `INFO` level.
- **Programmatic access**: Call `memagent.get_context_window_stats()` after any turn to retrieve the most recent snapshot (prompt/completion/total tokens, configured window size, percentage used, and timestamp).
- **Configurable budgets**: Provide `context_window_tokens=...` when constructing the agent (or inside `llm_config`) to override the detected window for custom models.

```python
response = agent.run("Summarize the last project update I gave you yesterday.")

stats = agent.get_context_window_stats()
print(
    f"Used {stats['total_tokens']} tokens (~{stats['percentage_used']:.2f}% of window)"
)
```

When no explicit window is supplied, MemoRizz attempts to infer it from the underlying provider (OpenAI, Azure, or Hugging Face). The stats object is safe to store or expose via telemetry dashboards for proactive monitoring.

📘 See the full walkthrough in `docs/utilities/context_window_stats.md`.

## Documentation

All project documentation lives in the root-level `docs/` directory and is rendered with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

### Local preview

```bash
pip install -e ".[docs]"
make docs-serve            # or: mkdocs serve
```

Open <http://localhost:8000> to view the live site. Use `make docs-build` before committing to ensure the site compiles without warnings.

### Deployment

1. Push changes to `main`. The `Docs` GitHub Action builds the site with `mkdocs build --strict` and uploads it to the `gh-pages` branch.
2. In the repository Settings → Pages panel, select **GitHub Actions** as the source (one-time setup).
3. If you ever need a manual publish, run `mkdocs gh-deploy --force` from your local machine (it uses the same `gh-pages` branch).

Every time you edit a Markdown file or docstring referenced via `mkdocstrings`, the GitHub Action republishes the updated site automatically.

# Table of single agent and multi-agent setups, their descriptions, and links to example notebooks
| Agent Type                | Description                                                                 | Example Notebook                                                                 |
|---------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Single Agent              | A standalone agent with its own memory and persona, suitable for individual tasks | [Single Agent Example](examples/single_agent/memagent_single_agent_demo.ipynb)                      |
| Multi-Agent               | A system of multiple agents collaborating, each with specialized roles and shared memory | [Multi-Agent Example](examples/memagents_multi_agents.ipynb)                        |



# Memory System Components and Examples

| Memory Component | Memory Category | Use Case / Description | Example Notebook |
|------------------|-----------------|------------------------|------------------|
| **Persona** | Semantic Memory | Agent identity, personality, and behavioral consistency | [Persona Example](examples/persona.ipynb) |
| **Knowledge Base** | Semantic Memory | Persistent facts, concepts, and domain knowledge | [Knowledge Base Example](examples/knowledge_base.ipynb) |
| **Toolbox** | Procedural Memory | Registered functions with semantic discovery for LLM execution | [Toolbox Example](examples/toolbox.ipynb) |
| **Workflow** | Procedural Memory | Multi-step process orchestration and execution tracking | [Workflow Example](examples/workflow.ipynb) |
| **Conversation Memory** | Episodic Memory | Interaction history and conversational context | [Single Agent Example](examples/single_agent/memagent_single_agent_demo.ipynb) |
| **Summaries** | Episodic Memory | Compressed episodic experiences and events | [Summarization Example](examples/memagent_summarisation.ipynb) |
| **Working Memory** | Short-term Memory | Active context management and current session state | [Single Agent Example](examples/single_agent/memagent_single_agent_demo.ipynb) |
| **Semantic Cache** | Short-term Memory | Vector-based query-response caching for performance optimization | [Semantic Cache Demo](examples/semantic_cache.ipynb) |
| **Shared Memory** | Multi-Agent Coordination | Blackboard for inter-agent communication and coordination | [Multi-Agent Example](examples/memagents_multi_agents.ipynb) |


### 2. Creating Specialized Agents with Personas

```python
from memorizz.long_term_memory.semantic.persona import Persona, RoleType

# Create a technical expert persona
tech_expert = Persona(
    name="TechExpert",
    role=RoleType.TECHNICAL_EXPERT,
    goals="Help developers solve complex technical problems with detailed explanations.",
    background="10+ years experience in Python, AI/ML, and distributed systems."
)

# Create agent with persona
agent = (MemAgentBuilder()
    .with_instruction("You are a technical expert assistant.")
    .with_persona(tech_expert)
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

# Save agent with persona to Oracle
agent.save()

# Now the agent will respond as a technical expert
response = agent.run("How should I design a scalable microservices architecture?")
```

### 3. Tool Registration and Function Calling

```python
import requests

# Define a tool function
def get_weather(latitude: float, longitude: float) -> float:
    """Get the current temperature for a given latitude and longitude."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m"
    )
    data = response.json()
    return data['current']['temperature_2m']

# Create agent with tools
weather_agent = (MemAgentBuilder()
    .with_instruction("You are a helpful weather assistant.")
    .with_tool(get_weather)
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

# Save agent (tools are persisted to Oracle)
weather_agent.save()

# Agent automatically discovers and uses tools
response = weather_agent.run(
    "What's the weather in New York? (latitude: 40.7128, longitude: -74.0060)"
)
print(response)  # Agent calls get_weather() and provides the temperature
```

### 4. Semantic Cache for Performance Optimization

Speed up your agents and reduce LLM costs with intelligent semantic caching:

```python
# Build agent without cache first
agent = (MemAgentBuilder()
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .with_memory_provider(oracle_provider)
    .build()
)

# Enable semantic cache (stored in Oracle)
agent.enable_semantic_cache(
    threshold=0.85,  # Similarity threshold (0.0-1.0)
    scope='local'    # 'local', 'global', or 'agent'
)

# Similar queries will use cached responses from Oracle
response1 = agent.run("What is the capital of France?")
response2 = agent.run("Tell me France's capital city")  # Cache hit! ⚡
response3 = agent.run("What's the capital of Japan?")   # New query, cache miss
```

**How Semantic Cache Works:**
1. **Store queries + responses** with vector embeddings in Oracle
2. **New query arrives** → generate embedding
3. **Similarity search** in Oracle using AI Vector Search and cosine similarity
4. **Cache hit** (similarity ≥ threshold) → return cached response ⚡
5. **Cache miss** → fallback to LLM + cache new response in Oracle

**Benefits:**
- 🚀 **Faster responses** for similar queries
- 💰 **Reduced LLM costs** by avoiding duplicate API calls
- 🎯 **Configurable precision** with similarity thresholds
- 🔒 **Scoped isolation** by agent, memory, or session ID
- 🗄️ **Persistent caching** in Oracle database with vector search

### 5. Generate Conversation Summaries

Compress long conversation histories into concise summaries:

```python
# After having several conversations with your agent
summary_ids = agent.generate_summaries(
    days_back=7,  # Summarize conversations from the last 7 days
    max_memories_per_summary=50  # Memories per summary chunk
)

print(f"Generated {len(summary_ids)} summaries")

# Summaries are stored in Oracle and can be retrieved
from memorizz.common.memory_type import MemoryType

summaries = oracle_provider.retrieve_by_query(
    query={'agent_id': agent.agent_id},
    memory_store_type=MemoryType.SUMMARIES,
    limit=10
)

for summary in summaries:
    print(f"Summary: {summary['content'][:200]}...")
```

## Core Concepts

### Memory Types

MemoRizz supports different memory categories for organizing information:

- **CONVERSATION_MEMORY**: Chat history and dialogue context
- **WORKFLOW_MEMORY**: Multi-step process information and tool execution tracking
- **LONG_TERM_MEMORY**: Persistent knowledge storage with semantic search
- **SHORT_TERM_MEMORY**: Temporary processing information including semantic cache for query-response optimization
- **PERSONAS**: Agent personality and behavior definitions
- **TOOLBOX**: Function definitions and metadata with semantic discovery
- **SHARED_MEMORY**: Multi-agent coordination and communication
- **MEMAGENT**: Agent configurations and states
- **SUMMARIES**: Compressed summaries of past interactions for efficient memory management

### Oracle JSON Relational Duality Views

MemoRizz leverages Oracle 23ai's JSON Relational Duality Views, which provide:

- **Dual Interface**: Access data as both relational tables and JSON documents
- **Automatic Sync**: Changes in JSON reflect in tables and vice versa
- **Type Safety**: Relational schema ensures data integrity
- **Performance**: Native vector search with Oracle AI Vector Search
- **Scalability**: Enterprise-grade database capabilities

## Advanced Usage

### Custom Memory Providers

Extend the memory provider interface for custom storage backends:

```python
from memorizz.memory_provider.base import MemoryProvider

class CustomMemoryProvider(MemoryProvider):
    def store(self, data, memory_store_type):
        # Your custom storage logic
        pass

    def retrieve_by_query(self, query, memory_store_type, limit=10):
        # Your custom retrieval logic
        pass
```

### Multi-Agent Workflows

Create collaborative agent systems with shared memory in Oracle:

```python
# Create specialized delegate agents
data_analyst = (MemAgentBuilder()
    .with_instruction("You are a data analysis expert.")
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

report_writer = (MemAgentBuilder()
    .with_instruction("You are a report writing specialist.")
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

# Create orchestrator agent with delegates
orchestrator = (MemAgentBuilder()
    .with_instruction("You coordinate between specialists to complete complex tasks.")
    .with_memory_provider(oracle_provider)
    .with_delegates([data_analyst, report_writer])
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": os.environ["OPENAI_API_KEY"]
    })
    .build()
)

# Execute multi-agent workflow
response = orchestrator.run("Analyze our sales data and create a quarterly report.")
```

### Memory Management Operations

Control agent memory persistence in Oracle:

```python
# Save agent state to Oracle
agent.save()

# Load existing agent by ID from Oracle
from memorizz.memagent.core import MemAgent

existing_agent = MemAgent.load(
    agent_id="your-agent-id",
    memory_provider=oracle_provider
)

# Refresh agent from Oracle database
agent.refresh()

# Start a new conversation
agent.start_new_conversation()

# Get current state
conversation_id = agent.get_current_conversation_id()
memory_id = agent.get_current_memory_id()
```

## Architecture

```
┌──────────────────────┐
│   MemAgent           │  ← High-level agent interface
├──────────────────────┤
│   Persona            │  ← Agent personality & behavior
├──────────────────────┤
│   Toolbox            │  ← Function registration & discovery
├──────────────────────┤
│   Memory Provider    │  ← Storage abstraction layer
├──────────────────────┤
│   Vector Search      │  ← Semantic similarity & retrieval
├──────────────────────┤
│   Oracle 23ai        │  ← Persistent storage with:
│   - Relational Tables│     • AI Vector Search
│   - Duality Views   │     • JSON Documents
│   - Vector Indexes  │     • Enterprise Features
└──────────────────────┘
```

## Examples

Check out the `examples/` directory for complete working examples:

- **single_agent/memagent_single_agent_demo.ipynb**: Complete single agent demo with Oracle
- **memagents_multi_agents.ipynb**: Multi-agent collaboration workflows
- **persona.ipynb**: Creating and using agent personas
- **toolbox.ipynb**: Tool registration and function calling
- **workflow.ipynb**: Workflow memory and process tracking
- **knowledge_base.ipynb**: Long-term knowledge management
- **semantic_cache.ipynb**: Semantic cache for performance optimization
- **memagent_summarisation.ipynb**: Conversation summarization

## Configuration

### Oracle Database Setup

**📖 For complete configuration details, see [SETUP.md](SETUP.md)**

**Connection Details (using Docker with defaults):**
- **Host**: `localhost`
- **Port**: `1521`
- **Service Name**: `FREEPDB1`
- **User**: `memorizz_user`
- **Password**: `SecurePass123!` (configurable via `ORACLE_PASSWORD`)

**Hosted Databases (FreeSQL.com, Oracle Cloud, etc.):**
- The setup automatically detects hosted databases and uses user-only mode
- Simply set `ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN` environment variables
- No admin credentials needed - uses your existing schema

**Environment Variables:**

All Oracle credentials and settings can be configured via environment variables:
- `ORACLE_ADMIN_USER` - Admin username (default: `system`)
- `ORACLE_ADMIN_PASSWORD` - Admin password (default: `MyPassword123!`)
- `ORACLE_IMAGE_TAG` - Docker image tag (default: `latest-lite` for 1.78GB, or `latest` for 9.93GB)
- `ORACLE_USER` - MemoRizz user (default: `memorizz_user`)
- `ORACLE_PASSWORD` - MemoRizz password (default: `SecurePass123!`)
- `ORACLE_DSN` - Connection string (default: `localhost:1521/FREEPDB1`)
- `PLATFORM_FLAG` - Docker platform flag (use `--platform linux/amd64` for Apple Silicon)

See [SETUP.md](SETUP.md) for detailed configuration options and manual setup instructions.

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Oracle Connection (optional if using OracleConfig)
export ORACLE_USER="memorizz_user"
export ORACLE_PASSWORD="SecurePass123!"
export ORACLE_DSN="localhost:1521/FREEPDB1"
```

## Troubleshooting

**Common Issues:**

1. **Oracle Connection**: Ensure Oracle is running and accessible
   ```bash
   docker ps  # Check if oracle-memorizz container is running
   docker logs oracle-memorizz  # Check logs
   ```

2. **Vector Search**: Oracle 23ai+ is required for AI Vector Search
3. **API Keys**: Check OpenAI API key is valid and has credits
4. **Duality Views**: Ensure `setup_oracle_user.py` completed successfully
5. **Import Errors**: Ensure you're using the correct import paths shown in examples

## Contributing

This is an educational project. Contributions for learning purposes are welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Educational Resources

This library demonstrates key concepts in:
- **AI Agent Architecture**: Memory, reasoning, and tool use
- **Vector Databases**: Semantic search and retrieval with Oracle AI Vector Search
- **LLM Integration**: Function calling and context management
- **Oracle 23ai Features**: JSON Relational Duality Views and AI capabilities
- **Software Design**: Clean abstractions and extensible architecture

## Why Oracle AI Database?

Oracle Database 23ai provides enterprise-grade features that make it ideal for AI agent memory:

- **Native Vector Search**: Built-in AI Vector Search with multiple distance metrics
- **JSON Duality Views**: Query data as JSON or SQL with automatic synchronization
- **Transactional Consistency**: ACID properties for reliable memory storage
- **Scalability**: Handle millions of memories with enterprise performance
- **Security**: Row-level security, encryption, and comprehensive audit trails
- **Free Tier**: Oracle Database 23ai Free for development and learning

---

**Ready to build memory-augmented AI agents?** Start with the [Quick Start](#quick-start) guide above or explore the [examples](examples/) directory!
