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

```/app/backend/data/context_compactor/<scope>/```

## Scope behavior

The injector decides which active snapshot to load using the `scope_source` valve.

# `chat_id_or_default`

Uses the current chat ID as the scope. If no chat ID is available, uses `default_scope`.

Best for strict per-chat isolation.

# `folder_id_or_default`

Uses the current folder ID as the scope, formatted as:

`folder_<folder_id>`

If the chat is not inside a folder, uses `default_scope`.

Best for project/folder-level continuity.

Recommended setting:

`scope_source = folder_id_or_default`
`default_scope = unfiled`

This gives this isolation model:

`Folder A chat -> folder_<folder_A_id>`
`Folder B chat -> folder_<folder_B_id>`
`Unfiled chat  -> unfiled`

Folder contexts stay isolated from each other, and unfiled chats do not read folder snapshots.

# `default_only`

Always uses `default_scope.`

Best for testing or for one deliberately shared global context.

Example test setting:

`scope_source = default_only`
`default_scope = test-context-compactor`
`scope_override`

A per-user override. When set, it ignores `scope_source` and always injects from that explicit scope.

Use this for temporary tests.

## Tool/filter scope alignment

The tool saves snapshots under the `scope_id` passed to `save_context_snapshot`.

The filter only injects snapshots from the scope it resolves at request time.

For injection to work, the saved snapshot scope and the resolved filter scope must match.

Current limitation: the tool does not yet automatically resolve the same chat or folder scope as the filter. Until that is implemented, pass the correct `scope_id` manually when saving snapshots.

## Expected pairing

This filter expects snapshots created by the Context Compactor tool.

The tool manages persistence and activation. The filter only injects the active snapshot.
