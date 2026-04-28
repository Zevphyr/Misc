# Context Compactor Tool

Local Open WebUI tool for storing, retrieving, activating, deactivating, deleting, and pruning compacted context snapshots.

## Purpose

This tool stores model-generated context compaction snapshots as local JSON. It is intended to preserve long-running project, debugging, writing, or research state without relying on an external provider, embedding service, tokenizer, database, or network API.

## Design

- Local JSON storage only
- Scoped snapshot directories
- Atomic writes
- Best-effort local lock files
- Reviewable activation/deactivation
- Secret redaction before save
- No model calls
- No external HTTP calls
- No vendor-specific API dependency

## Default storage path

```text
/app/backend/data/context_compactor/<scope>/
```

## Main functions

- initialize_scope
- save_context_snapshot
- get_active_context
- list_context_snapshots
- get_context_snapshot
- activate_context_snapshot
- deactivate_context
- delete_context_snapshot
- prune_context_snapshots

## Pairing

Use with the Context Compactor Injector filter, which reads the active snapshot and injects it into future chat context.
