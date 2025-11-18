# Overview

MemoRizz is a composable memory framework for AI agents. It ships opinionated agent builders, configurable memory providers, and a cognitive-inspired architecture so that every memory you store is intentional.

## Architecture at a Glance

```
src/memorizz/
├── long_term_memory/      # semantic, procedural, episodic systems
├── short_term_memory/     # semantic cache + working memory
├── coordination/          # shared memory for multi-agent orchestration
├── memory_provider/       # Oracle, MongoDB, custom backends
└── memagent/              # builders + runtime orchestration
```

Each folder owns the implementation for a specific memory subsystem. Agent presets ("application modes") simply select the right combination of these subsystems.

## Key Capabilities

| Capability | Description | Code Entry Point |
|------------|-------------|------------------|
| Long-term semantic memory | Fact + entity graph storage with embeddings | `long_term_memory/semantic/`
| Procedural memory | Toolboxes and workflows for behavior execution | `long_term_memory/procedural/`
| Episodic memory | Conversation history, summaries, and experiences | `long_term_memory/episodic/`
| Short-term memory | Working context buffer + semantic cache | `short_term_memory/`
| Memory providers | Database-specific persistence logic | `memory_provider/`
| Application modes | Pre-bundled stacks per use case | `enums/application_mode.py`

!!! tip "Map docs to code"
    Every section in this site mirrors these modules. When you update a doc, link back to the concrete module (for example ``::: memorizz.memagent.builders.MemAgentBuilder``) so the rendered API reference always matches the running code.

## Requirements

- Python 3.7+
- An embedding/LLM provider such as OpenAI or Hugging Face
- A memory provider backend (Oracle 23ai/26ai, MongoDB, or your own `MemoryProvider` implementation)

## Next Steps

1. Read through the [Concepts](concepts.md) page to understand each memory type.
2. Pick a provider under [Memory Providers](../memory-providers/oracle.md) and configure credentials.
3. Follow the [Python SDK Quickstart](python-sdk-quickstart.md) to spin up your first `MemAgent`.
