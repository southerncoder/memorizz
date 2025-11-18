# Concepts

MemoRizz models agent cognition around a handful of composable building blocks. Understanding these types makes it easier to reason about what your application mode actually enables.

## Memory Types

| Enum | Purpose | Realization |
|------|---------|-------------|
| `MemoryType.LONG_TERM_MEMORY` | Semantic knowledge base | Namespaces, personas, entity memory |
| `MemoryType.ENTITY_MEMORY` | Structured profile data tied to entities | Attribute/value store with provenance |
| `MemoryType.TOOLBOX` + `MemoryType.WORKFLOW_MEMORY` | Toolbox and workflow behaviors | `long_term_memory/procedural/` |
| `MemoryType.CONVERSATION_MEMORY` | Episodic timeline of interactions | `long_term_memory/episodic/`
| `MemoryType.SUMMARIES` | Cached digests of long conversations | `long_term_memory/episodic/summaries.py`
| `MemoryType.SHORT_TERM_MEMORY` | Working context window | `short_term_memory/working_memory/`
| `MemoryType.SEMANTIC_CACHE` | Fast, short-lived fact lookups | `short_term_memory/semantic_cache/`
| `MemoryType.SHARED_MEMORY` | Coordination between multiple agents | `coordination/shared_memory/`

!!! note
    The `MemoryType` enum lives in `src/memorizz/enums/memory_type.py`. Extending it is the first step when you want to introduce a new storage primitive.

## Memories vs. Providers

- **Memory types** describe *what* your agent can recall.
- **Memory providers** describe *where* the data lives (Oracle, MongoDB, local experiment, etc.).
- **Application modes** (see `src/memorizz/enums/application_mode.py`) simply select the right combination of memories for a task. For example `ASSISTANT` activates conversation history, long-term facts, personas, and summaries; `DEEP_RESEARCH` focuses on toolbox access and shared memory.

## Lifecycle

1. **Capture** – Agents persist facts by calling methods on the active memory types (e.g., saving a persona or upserting entity attributes).
2. **Index** – Providers embed relevant fields using your configured embedding provider.
3. **Retrieve** – During a run, the `MemAgent` orchestrator fetches relevant rows from each memory and mixes them into the prompt stack.
4. **Summarize** – Episodic memory periodically compacts older interactions into summary memories that keep the context window manageable while preserving detail.

## How to Explore Further

- Inspect `src/memorizz/MEMORY_ARCHITECTURE.md` for the full architecture notes that ship with the codebase.
- Use `mkdocstrings` directives inside any doc page to render live API reference blocks, e.g.

```markdown
::: memorizz.memagent.builders.MemAgentBuilder
    handler: python
```

That directive renders directly from the Python source, so your docs always match the SDK version in the repository.
