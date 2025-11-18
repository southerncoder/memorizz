# Semantic Memory

Semantic memory stores canonical facts, personas, and entity attributes that rarely change. In MemoRizz, this maps to `src/memorizz/long_term_memory/semantic/` and is backed by the `MemoryType.LONG_TERM_MEMORY` and `MemoryType.ENTITY_MEMORY` enums.

## Components

- **Knowledge Base** – Vectorized documents segmented by namespace or topic.
- **Personas** – Behavioral instructions, tone, and guardrails that shape agent responses.
- **Entity Memory** – Structured attributes for people, organizations, or devices. The `entity_memory` module exposes helper methods to upsert and query profile fields.

## Typical Operations

```python
kb_id = agent.memory.long_term.save_document(
    namespace="support",
    content="The premium plan includes unlimited vector storage.",
)

agent.memory.entity_memory.upsert(
    entity_id="company_acme",
    attributes={"plan": "premium"},
)
```

The provider automatically embeds the document, stores metadata, and tags the record with the owning agent or namespace.

## When to Use

- Product catalogs and policy manuals
- Persona systems for specialized assistants (support, researcher, interviewer)
- Entity profiles that must persist across sessions and devices

Semantic memory powers long-lived recall. Pair it with episodic memory when you also care about interaction history.
