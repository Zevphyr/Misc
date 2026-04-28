# Add automatic scope resolution to Context Compactor tool

## Summary

The Context Compactor tool should support automatic scope resolution that matches the Context Compactor Injector filter.

## Current behavior

The tool saves snapshots under the explicit `scope_id` passed to `save_context_snapshot`.

If `scope_id` is empty, the tool falls back to its own `default_scope` valve.

The filter, however, can resolve scope dynamically using `scope_source`, such as:

```text
chat_id_or_default
folder_id_or_default
default_only
```

This can cause mismatches.

Example:

```text
Tool saves snapshot under: test-context-compactor
Filter resolves current chat scope as: folder_abc123
Result: active snapshot is not injected
```

## Shortcoming

The user must manually know and pass the same scope that the filter will later resolve.

This is error-prone, especially for folder-level context isolation, where the correct target scope may be:

```text
folder_<folder_id>
```

The problem is more visible when using:

```text
scope_source = folder_id_or_default
default_scope = unfiled
```

That mode is desirable because it keeps folder contexts isolated from each other and prevents unfiled chats from reading folder snapshots, but the tool does not yet help save snapshots into the matching folder scope.

## Desired behavior

Add a helper function such as:

```text
resolve_context_scope
```

or add an optional mode to `save_context_snapshot`, such as:

```text
scope_id = ""
scope_source = "auto"
```

The tool should then resolve the same effective scope as the filter, if Open WebUI passes the needed metadata to tools.

## Proposed implementation direction

If Open WebUI passes `__metadata__`, `chat_id`, and/or `folder_id` to tools in this environment, reuse the same scope-resolution logic as the filter:

```text
chat_id_or_default
folder_id_or_default
default_only
```

For folder mode:

```text
folder_id -> folder_<folder_id>
no folder -> default_scope
```

Recommended default for project isolation:

```text
scope_source = folder_id_or_default
default_scope = unfiled
```

## Acceptance criteria

- Tool can report the resolved effective scope before saving.
- `save_context_snapshot` can save to the automatically resolved scope.
- Filter and tool use matching scope-resolution behavior.
- Folder contexts remain isolated from each other.
- Unfiled chats use only the unfiled/default scope.
- Unfiled chats do not read folder snapshots.
- No external APIs, model calls, embeddings, databases, or tokenizer dependencies are introduced.
