# Context Compactor Injector Filter

Read-only Open WebUI filter that injects the active Context Compactor snapshot into the prompt before the latest user message.

## Purpose

This filter makes the currently active compacted context visible to the model on later turns without rewriting chat history or mutating stored snapshots.

## Design

- Read-only
- Local JSON reads only
- No model calls
- No external HTTP calls
- No database dependency
- Removes prior injected compaction blocks before reinjecting
- Redacts common secret patterns again at injection time
- Fails open by default so chat continues if the store is missing or unreadable

## Default storage path

```text
/app/backend/data/context_compactor/<scope>/
```

## Expected pairing

This filter expects snapshots created by the Context Compactor tool.

The tool manages persistence and activation. The filter only injects the active snapshot.
