# Python SDK Quickstart

This walkthrough spins up a fully stateful agent with Oracle as the backing provider. Swap in another provider if you prefer MongoDB or a custom backend.

## 1. Install Dependencies

```bash
pip install -e ".[docs]"         # documentation + tooling
pip install -e ".[oracle]"       # choose oracle/mongodb/ollama/etc. as needed
```

Add or export your provider + LLM credentials (see `.env.example`).

## 2. Bootstrap Oracle (optional)

```bash
./install_oracle.sh          # starts Oracle 23ai locally
memorizz setup-oracle        # prepares schemas and tables
```

The setup script automatically creates the JSON + vector tables for every memory bucket (personas, long-term memory, semantic cache, etc.).

## 3. Configure Embeddings

```python
from memorizz.embeddings import configure_embeddings

configure_embeddings("openai", {
    "model": "text-embedding-3-small",
    "api_key": os.environ["OPENAI_API_KEY"],
})
```

## 4. Build an Agent

```python
from memorizz.memory_provider.oracle import OracleProvider, OracleConfig
from memorizz.memagent.builders import MemAgentBuilder

oracle_provider = OracleProvider(
    OracleConfig(
        user="memorizz_user",
        password="SecurePass123!",
        dsn="localhost:1521/FREEPDB1",
        embedding_provider="openai",
    )
)

agent = (MemAgentBuilder()
    .with_instruction("You are a helpful assistant with persistent memory.")
    .with_memory_provider(oracle_provider)
    .with_llm_config({
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
    })
    .build())
```

## 5. Run and Inspect Memory

```python
response = agent.run("Hello, my name is Leah and I like dark mode UIs.")
print(response)

# Save a structured entity profile
agent.memory.entity_memory.upsert(
    entity_id="leah",
    attributes={"preferences": ["dark mode UIs", "Python"]}
)
```

Check your provider (Oracle, MongoDB) to see the stored JSON, embeddings, and metadata for each memory bucket.

## Where to Go Next

- Review every memory subsystem under [Memory Types](../memory-types/semantic.md).
- Point a different provider at the agent with `MemAgentBuilder().with_memory_provider(...)`.
- Embed API docs inline with ``::: memorizz.memagent.memagent.MemAgent`` to expose parameters inside this site.
