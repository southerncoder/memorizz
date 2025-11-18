# MemoRizz Documentation

MemoRizz helps you build memory-augmented agents that combine long-term knowledge, episodic context, procedural behaviors, and multi-agent coordination. This site keeps the narrative docs, quickstarts, and API references directly inside the repository so the docs never drift from the code that powers them.

## What's Inside

- **Getting Started** walks through the core concepts and SDK setup for your first agent.
- **Memory Types** dives into each cognitive-inspired subsystem and how it maps to the source tree in `src/memorizz/`.
- **Memory Providers** describes the persistence backends (Oracle, MongoDB, or custom) that store the memories.
- **Use Cases** shows how the library stitches memory stacks together for common application modes like assistants or research bots.

!!! info "Docs live with the code"
    Every page in this site is rendered straight from the Markdown under `docs/`. Update a file, run `mkdocs serve`, and the change appears instantly. Merge to `main` and the GitHub Pages workflow publishes the refreshed site automatically.

## Quick Start

```bash
pip install -e ".[docs]"
mkdocs serve
```

Visit <http://localhost:8000> for a hot-reloading docs server. When you're ready to publish, run `mkdocs build --strict` or rely on the provided GitHub Action to deploy to the `gh-pages` branch.

## Need More?

- Check the Python API reference entries embedded throughout the docs via [`mkdocstrings`](https://mkdocstrings.github.io/).
- Browse real workflows in `src/memorizz/examples/` and link them into the docs with snippets or code fences.
- Open an issue or discussion on [GitHub](https://github.com/RichmondAlake/memorizz) if you spot a gap.
