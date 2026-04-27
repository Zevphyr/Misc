"""
title: Sub Agent
version: 1.0.0
license: GPL-3.0-or-later
required_open_webui_version: 0.9.0
description: Run autonomous, tool-heavy tasks in isolated sub-agent contexts.

SPDX-License-Identifier: GPL-3.0-or-later

This is a sub-agent tool for Open WebUI. It delegate multi-step, tool-heavy work into an
isolated context and return only the sub-agent's final result to the parent
model.

Operational assumptions:
- Native function calling should be enabled for the selected model.
- The tool is designed for Open WebUI 0.9.x async internals.
- Where possible, Open WebUI's own helpers are used so access control, model
  feature gates, built-in tool behavior, and schema conversion stay upstream.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from functools import partial
from typing import Any, Literal, Optional

from fastapi import Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Open WebUI / model-facing schema
# -----------------------------------------------------------------------------


class SubAgentTaskItem(BaseModel):
    """One independent sub-agent task."""

    description: str = Field(
        description=(
            "Brief user-visible status text for this task. Write it in the user's language."
        )
    )
    prompt: str = Field(
        description=(
            "Complete instructions for the sub-agent. Include all context it needs, "
            "because sub-agents do not receive the parent chat history."
        )
    )


# Upstream Open WebUI currently exposes builtins by function name while gating by
# category internally.  These mappings let this tool disable categories without
# trying to reimplement upstream's feature/permission/model checks.
BUILTIN_CATEGORY_TO_FUNCTIONS: dict[str, set[str]] = {
    "time": {"get_current_timestamp", "calculate_timestamp"},
    "web_search": {"search_web", "fetch_url"},
    "image_generation": {"generate_image", "edit_image"},
    "knowledge": {
        "list_knowledge",
        "list_knowledge_bases",
        "search_knowledge_bases",
        "query_knowledge_bases",
        "search_knowledge_files",
        "query_knowledge_files",
        "view_file",
        "view_knowledge_file",
    },
    "chats": {"search_chats", "view_chat"},
    "memory": {
        "search_memories",
        "add_memory",
        "replace_memory_content",
        "delete_memory",
        "list_memories",
    },
    "notes": {"search_notes", "view_note", "write_note", "replace_note_content"},
    "channels": {
        "search_channels",
        "search_channel_messages",
        "view_channel_thread",
        "view_channel_message",
    },
    "code_interpreter": {"execute_code"},
    "skills": {"view_skill"},
    "tasks": {"create_tasks", "update_task"},
    "automations": {
        "create_automation",
        "update_automation",
        "list_automations",
        "toggle_automation",
        "delete_automation",
    },
    "calendar": {
        "search_calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
    },
}

VALVE_TO_BUILTIN_CATEGORY: dict[str, str] = {
    "ENABLE_TIME_TOOLS": "time",
    "ENABLE_WEB_TOOLS": "web_search",
    "ENABLE_IMAGE_TOOLS": "image_generation",
    "ENABLE_KNOWLEDGE_TOOLS": "knowledge",
    "ENABLE_CHAT_TOOLS": "chats",
    "ENABLE_MEMORY_TOOLS": "memory",
    "ENABLE_NOTES_TOOLS": "notes",
    "ENABLE_CHANNELS_TOOLS": "channels",
    "ENABLE_CODE_INTERPRETER_TOOLS": "code_interpreter",
    "ENABLE_SKILLS_TOOLS": "skills",
    "ENABLE_TASK_TOOLS": "tasks",
    "ENABLE_AUTOMATION_TOOLS": "automations",
    "ENABLE_CALENDAR_TOOLS": "calendar",
}

# These Open WebUI builtins can produce citation/source events in the frontend.
CITATION_TOOL_NAMES = {
    "search_web",
    "fetch_url",
    "view_file",
    "view_knowledge_file",
    "query_knowledge_files",
}

TERMINAL_EVENT_TOOL_NAMES = {
    "display_file",
    "write_file",
    "replace_file_content",
    "run_command",
}

SELF_TOOL_FUNCTION_NAMES = {"run_sub_agent", "run_parallel_sub_agents"}

_SKILLS_MANIFEST_START = "<available_skills>"
_SKILLS_MANIFEST_END = "</available_skills>"
_SKILL_TAG_PATTERN = re.compile(r"<skill\s+name=.*?>\n.*?\n</skill>", re.DOTALL)

_CORE_PROCESS_TOOL_RESULT: Optional[Callable[..., Any]] = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


async def maybe_await(value: Any) -> Any:
    """Await only when a compatibility path returns an awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def truncate_text(text: Any, limit: int) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = compact_json(text)
        except Exception:
            text = str(text)
    if limit and limit > 0 and len(text) > limit:
        return f"{text[:limit]}\n\n[SubAgent: truncated {len(text) - limit} characters]"
    return text


def split_csv(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_jsonish_arguments(raw: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Parse native tool-call arguments.

    Native tool calls should provide JSON.  For compatibility with weaker local
    models, a Python-literal fallback can be enabled by the caller separately;
    this function itself keeps the safer default of JSON-only parsing.
    """
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return dict(raw), None
    if not isinstance(raw, str):
        return {}, f"tool arguments must be a JSON object string, got {type(raw).__name__}"
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return {}, f"could not parse tool arguments as JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"tool arguments must decode to an object, got {type(parsed).__name__}"
    return parsed, None


def parse_jsonish_arguments_compat(raw: Any) -> tuple[dict[str, Any], Optional[str]]:
    """JSON-first parser with a literal fallback for imperfect local models."""
    parsed, error = parse_jsonish_arguments(raw)
    if error is None:
        return parsed, None
    if not isinstance(raw, str):
        return parsed, error

    # Compatibility fallback. ast.literal_eval does not execute code, but it is
    # still deliberately last because function-calling arguments should be JSON.
    import ast

    try:
        fallback = ast.literal_eval(raw)
    except Exception:
        return parsed, error
    if not isinstance(fallback, dict):
        return {}, f"tool arguments must decode to an object, got {type(fallback).__name__}"
    return dict(fallback), None


def get_message_text_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return parts
    return []


def extract_from_system_messages(messages: Optional[list], extractor: Callable[[str], Any]) -> list[Any]:
    results: list[Any] = []
    if not isinstance(messages, list):
        return results
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        for text in get_message_text_parts(message.get("content")):
            found = extractor(text)
            if not found:
                continue
            if isinstance(found, list):
                results.extend(found)
            else:
                results.append(found)
    return results


def find_skill_manifest(text: str) -> str:
    start = text.find(_SKILLS_MANIFEST_START)
    if start < 0:
        return ""
    end = text.find(_SKILLS_MANIFEST_END, start)
    if end < 0:
        return ""
    return text[start : end + len(_SKILLS_MANIFEST_END)]


def extract_skill_manifest(messages: Optional[list]) -> str:
    matches = extract_from_system_messages(messages, find_skill_manifest)
    return matches[0] if matches else ""


def extract_user_skill_tags(messages: Optional[list]) -> list[str]:
    return [str(item) for item in extract_from_system_messages(messages, _SKILL_TAG_PATTERN.findall)]


def get_skill_ids(metadata: dict, user_payload: dict, messages: Optional[list]) -> list[str]:
    """Collect skill IDs from known Open WebUI injection locations when present."""
    candidates = [
        metadata.get("skill_ids"),
        metadata.get("__skill_ids__"),
        as_dict(user_payload).get("skill_ids"),
        as_dict(user_payload).get("__skill_ids__"),
    ]
    for value in candidates:
        if isinstance(value, list):
            return [str(item) for item in value if isinstance(item, (str, int))]

    # Some versions only inject skills into the prompt.  IDs cannot be reliably
    # reconstructed from that text, so this intentionally returns [] here.
    return []


def merge_prompt_sections(*sections: Any) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for section in sections:
        if isinstance(section, list):
            nested = merge_prompt_sections(*section)
            if nested and nested not in seen:
                out.append(nested)
                seen.add(nested)
            continue
        if not isinstance(section, str):
            continue
        stripped = section.strip()
        if stripped and stripped not in seen:
            out.append(stripped)
            seen.add(stripped)
    return "\n\n".join(out)


def model_id_from_context(
    *,
    explicit_default: str,
    metadata: Optional[dict],
    model: Optional[dict],
) -> str:
    if explicit_default.strip():
        return explicit_default.strip()
    md = as_dict(metadata)
    md_model = md.get("model")
    if isinstance(md_model, dict) and isinstance(md_model.get("id"), str):
        return md_model["id"]
    if isinstance(md.get("model_id"), str):
        return md["model_id"]
    if isinstance(md.get("model"), str):
        return md["model"]
    if isinstance(model, dict) and isinstance(model.get("id"), str):
        return model["id"]
    return ""


def resolve_model_dict(request: Request, model_id: str, fallback_model: Optional[dict]) -> dict:
    fallback = dict(fallback_model or {})
    try:
        models = getattr(request.app.state, "MODELS", {}) or {}
        resolved = models.get(model_id)
        if isinstance(resolved, dict):
            return resolved
    except Exception:
        pass
    if model_id and not fallback.get("id"):
        fallback["id"] = model_id
    return fallback


def normalize_metadata(
    metadata: Optional[dict],
    *,
    model_id: str,
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> dict:
    """Make a fresh metadata object with the fields builtins commonly expect.

    This avoids stale-reference bugs when callers enriched metadata by rebinding
    instead of mutating the object that had already been put in extra_params.
    """
    normalized = dict(metadata or {})
    normalized["model_id"] = normalized.get("model_id") or model_id

    model_value = normalized.get("model")
    if isinstance(model_value, dict):
        normalized["model"] = {**model_value, "id": model_value.get("id") or model_id}
    elif isinstance(model_value, str):
        normalized["model"] = {"id": model_value}
    else:
        normalized["model"] = {"id": model_id}

    if chat_id:
        normalized["chat_id"] = chat_id
    if message_id:
        normalized["message_id"] = message_id
    normalized.setdefault("files", [])
    normalized.setdefault("features", {})
    normalized.setdefault("tool_ids", [])
    return normalized


def build_extra_params(
    *,
    request: Request,
    user_payload: dict,
    model: dict,
    metadata: dict,
    event_emitter: Optional[Callable[[dict], Any]],
    event_call: Optional[Callable[[dict], Any]],
    chat_id: Optional[str],
    message_id: Optional[str],
    oauth_token: Optional[dict],
    messages: Optional[list],
    skill_ids: Optional[list[str]],
) -> dict:
    return {
        "__request__": request,
        "__user__": user_payload,
        "__model__": model,
        "__metadata__": metadata,
        "__chat_id__": chat_id or metadata.get("chat_id"),
        "__message_id__": message_id or metadata.get("message_id"),
        "__oauth_token__": oauth_token,
        "__event_emitter__": event_emitter,
        "__event_call__": event_call,
        "__files__": metadata.get("files", []),
        "__messages__": messages or [],
        "__skill_ids__": skill_ids or [],
    }


async def emit_event(event_emitter: Optional[Callable[[dict], Any]], event: dict) -> None:
    if not callable(event_emitter):
        return
    try:
        await event_emitter(event)
    except Exception as exc:
        log.debug("[SubAgent] event emission failed: %s", exc)


async def emit_status(
    event_emitter: Optional[Callable[[dict], Any]],
    description: str,
    *,
    done: bool = False,
    limit: int = 1000,
) -> None:
    if not description:
        return
    await emit_event(
        event_emitter,
        {
            "type": "status",
            "data": {
                "description": truncate_text(description.replace("\n", " "), limit),
                "done": done,
            },
        },
    )


async def emit_notification(
    event_emitter: Optional[Callable[[dict], Any]],
    *,
    level: Literal["info", "success", "warning", "error"],
    content: str,
) -> None:
    if not content.strip():
        return
    await emit_event(
        event_emitter,
        {"type": "notification", "data": {"type": level, "content": content.strip()}},
    )


def user_object_from_payload(user_payload: Any) -> Any:
    if user_payload is None or hasattr(user_payload, "id"):
        return user_payload
    if isinstance(user_payload, dict):
        try:
            from open_webui.models.users import UserModel

            return UserModel(**user_payload)
        except Exception:
            from types import SimpleNamespace

            return SimpleNamespace(**user_payload)
    return user_payload


def coerce_user_valves(raw_valves: Any, valves_cls: type[BaseModel]) -> BaseModel:
    if isinstance(raw_valves, valves_cls):
        return raw_valves
    if isinstance(raw_valves, BaseModel):
        with suppress(Exception):
            return valves_cls.model_validate(raw_valves.model_dump())
    if isinstance(raw_valves, dict):
        with suppress(Exception):
            return valves_cls.model_validate(raw_valves)
    return valves_cls.model_validate({})


async def read_request_json(request: Optional[Request]) -> dict:
    if request is None:
        return {}
    body_fn = getattr(request, "body", None)
    if not callable(body_fn):
        return {}
    try:
        raw = await body_fn()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Tool-server / terminal metadata helpers
# -----------------------------------------------------------------------------


def normalize_terminal_id(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


async def resolve_terminal_id(metadata: dict, request: Optional[Request]) -> str:
    body = await read_request_json(request)
    for value in (
        body.get("terminal_id"),
        as_dict(body.get("metadata")).get("terminal_id"),
        metadata.get("terminal_id"),
    ):
        terminal_id = normalize_terminal_id(value)
        if terminal_id:
            return terminal_id
    return ""


def normalize_tool_servers(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


async def resolve_direct_tool_servers(metadata: dict, request: Optional[Request]) -> list[dict]:
    body = await read_request_json(request)
    for value in (
        body.get("tool_servers"),
        as_dict(body.get("metadata")).get("tool_servers"),
        metadata.get("tool_servers"),
    ):
        servers = normalize_tool_servers(value)
        if servers:
            return servers
    return []


def extract_direct_tool_server_prompts(direct_tools: Mapping[str, dict]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for tool in direct_tools.values():
        server = as_dict(tool).get("server")
        if not isinstance(server, dict):
            continue
        prompt = server.get("system_prompt")
        if isinstance(prompt, str):
            stripped = prompt.strip()
            if stripped and stripped not in seen:
                prompts.append(stripped)
                seen.add(stripped)
    return prompts


def build_direct_tools(tool_servers: list[dict]) -> dict[str, dict]:
    """Build direct tool entries from frontend-provided server specs.

    Direct tools are executed through __event_call__ so this tool does not need
    to know each server's protocol details.
    """
    out: dict[str, dict] = {}
    for server in tool_servers:
        specs = server.get("specs")
        if not isinstance(specs, list):
            continue
        server_payload = {k: v for k, v in server.items() if k != "specs"}
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not isinstance(name, str) or not name:
                continue
            out[name] = {
                "tool_id": f"direct:{name}",
                "spec": clean_tool_spec(spec),
                "server": server_payload,
                "direct": True,
                "type": "direct",
            }
    return out


async def execute_direct_tool(
    *,
    name: str,
    params: dict,
    tool: dict,
    extra_params: dict,
) -> Any:
    event_call = extra_params.get("__event_call__")
    if not callable(event_call):
        raise RuntimeError("Direct tool execution requires __event_call__ context")
    metadata = as_dict(extra_params.get("__metadata__"))
    return await event_call(
        {
            "type": "execute:tool",
            "data": {
                "id": str(uuid.uuid4()),
                "name": name,
                "params": params,
                "server": as_dict(tool.get("server")),
                "session_id": metadata.get("session_id"),
            },
        }
    )


def normalize_terminal_tools_result(result: Any, extra_params: dict) -> dict[str, dict]:
    """Accept both old and newer get_terminal_tools return shapes."""
    terminal_tools = result
    terminal_prompt: Optional[str] = None

    if isinstance(result, tuple) and len(result) == 2:
        terminal_tools = result[0]
        if isinstance(result[1], str) and result[1].strip():
            terminal_prompt = result[1].strip()

    if terminal_prompt:
        extra_params["__terminal_system_prompt__"] = terminal_prompt
    else:
        extra_params.pop("__terminal_system_prompt__", None)

    return terminal_tools if isinstance(terminal_tools, dict) else {}


async def emit_terminal_event(
    *,
    name: str,
    params: dict,
    result: Any,
    event_emitter: Optional[Callable[[dict], Any]],
) -> None:
    if name not in TERMINAL_EVENT_TOOL_NAMES:
        return
    if name == "run_command":
        await emit_event(event_emitter, {"type": "terminal:run_command", "data": {}})
        return

    path = params.get("path") if isinstance(params, dict) else None
    if not isinstance(path, str) or not path:
        return

    if name == "display_file":
        parsed = result
        if isinstance(parsed, str):
            with suppress(Exception):
                parsed = json.loads(parsed)
        if isinstance(parsed, dict) and parsed.get("exists") is False:
            return
        await emit_event(event_emitter, {"type": "terminal:display_file", "data": {"path": path}})
    elif name in {"write_file", "replace_file_content"}:
        await emit_event(event_emitter, {"type": f"terminal:{name}", "data": {"path": path}})


# -----------------------------------------------------------------------------
# Schema normalization
# -----------------------------------------------------------------------------


def clean_schema_node(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        non_null = [item for item in schema["anyOf"] if not (isinstance(item, dict) and item.get("type") == "null")]
        if len(non_null) == 1 and isinstance(non_null[0], dict):
            schema.pop("anyOf", None)
            schema.update(non_null[0])
        else:
            schema["anyOf"] = non_null

    if schema.get("default") is None:
        schema.pop("default", None)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key in list(properties.keys()):
            if key.startswith("__"):
                properties.pop(key, None)
            else:
                clean_schema_node(properties[key])

    if isinstance(schema.get("items"), dict):
        clean_schema_node(schema["items"])

    if "type" not in schema and "anyOf" not in schema and "properties" not in schema:
        schema["type"] = "string"

    return schema


def clean_tool_spec(spec: Any) -> dict:
    if not isinstance(spec, dict):
        return {}
    cleaned = copy.deepcopy(spec)
    if isinstance(cleaned.get("parameters"), dict):
        clean_schema_node(cleaned["parameters"])
    return cleaned


def allowed_tool_params_from_spec(spec: dict) -> Optional[set[str]]:
    params = as_dict(spec.get("parameters"))
    properties = params.get("properties")
    if isinstance(properties, dict):
        return set(properties.keys())
    return None


def filter_tool_params(params: dict, spec: dict) -> dict:
    allowed = allowed_tool_params_from_spec(spec)
    if allowed is None:
        return dict(params)
    return {key: value for key, value in params.items() if key in allowed}


def build_tools_param(tools: Mapping[str, dict]) -> Optional[list[dict]]:
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools.values():
        spec = clean_tool_spec(tool.get("spec", {}))
        if spec.get("name"):
            out.append({"type": "function", "function": spec})
    return out or None


# -----------------------------------------------------------------------------
# Open WebUI result/citation processing
# -----------------------------------------------------------------------------


async def process_tool_result(
    *,
    name: str,
    tool_type: str,
    result: Any,
    direct: bool,
    extra_params: dict,
) -> tuple[Any, list, list]:
    global _CORE_PROCESS_TOOL_RESULT

    if _CORE_PROCESS_TOOL_RESULT is None:
        with suppress(Exception):
            from open_webui.utils.middleware import process_tool_result as upstream_process

            _CORE_PROCESS_TOOL_RESULT = upstream_process

    if _CORE_PROCESS_TOOL_RESULT is not None:
        return await maybe_await(
            _CORE_PROCESS_TOOL_RESULT(
                extra_params.get("__request__"),
                name,
                result,
                tool_type,
                direct_tool=direct,
                metadata=as_dict(extra_params.get("__metadata__")),
                user=user_object_from_payload(extra_params.get("__user__")),
            )
        )

    # Conservative compatibility fallback.
    if isinstance(result, tuple):
        result = result[0] if result else ""
    elif direct and isinstance(result, list) and len(result) == 2:
        result = result[0]
    if isinstance(result, (dict, list)):
        result = pretty_json(result)
    elif result is not None and not isinstance(result, str):
        result = str(result)
    return result, [], []


async def emit_citations_if_available(
    *,
    name: str,
    params: dict,
    result: str,
    tool: dict,
    event_emitter: Optional[Callable[[dict], Any]],
) -> None:
    if name not in CITATION_TOOL_NAMES or not result:
        return
    try:
        from open_webui.utils.middleware import get_citation_source_from_tool_result

        sources = get_citation_source_from_tool_result(
            tool_name=name,
            tool_params=params,
            tool_result=result,
            tool_id=tool.get("tool_id", ""),
        )
        for source in sources or []:
            await emit_event(event_emitter, {"type": "source", "data": source})
    except Exception as exc:
        log.debug("[SubAgent] citation extraction failed for %s: %s", name, exc)


# -----------------------------------------------------------------------------
# MCP support
# -----------------------------------------------------------------------------


async def resolve_mcp_tools(
    *,
    request: Request,
    user: Any,
    tool_ids: list[str],
    extra_params: dict,
    debug: bool,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Resolve Open WebUI MCP server IDs into callable tool specs."""
    try:
        from open_webui.utils.mcp.client import MCPClient
    except Exception:
        if debug and tool_ids:
            log.info("[SubAgent] MCP client unavailable; MCP tools skipped")
        return {}, {}

    try:
        from open_webui.utils.access_control import has_connection_access
    except Exception:
        from open_webui.utils.tools import has_tool_server_access as has_connection_access

    from open_webui.env import ENABLE_FORWARD_USER_INFO_HEADERS
    from open_webui.utils.headers import include_user_info_headers
    from open_webui.utils.misc import is_string_allowed

    with suppress(Exception):
        from open_webui.env import (
            FORWARD_SESSION_INFO_HEADER_CHAT_ID,
            FORWARD_SESSION_INFO_HEADER_MESSAGE_ID,
        )
    if "FORWARD_SESSION_INFO_HEADER_CHAT_ID" not in locals():
        FORWARD_SESSION_INFO_HEADER_CHAT_ID = None  # type: ignore[assignment]
        FORWARD_SESSION_INFO_HEADER_MESSAGE_ID = None  # type: ignore[assignment]

    metadata = as_dict(extra_params.get("__metadata__"))
    connections = getattr(getattr(request.app.state, "config", None), "TOOL_SERVER_CONNECTIONS", []) or []

    server_ids: list[str] = []
    seen: set[str] = set()
    for tool_id in tool_ids:
        if not isinstance(tool_id, str) or not tool_id.startswith("server:mcp:"):
            continue
        server_id = tool_id[len("server:mcp:") :].strip()
        if server_id and server_id not in seen:
            seen.add(server_id)
            server_ids.append(server_id)

    tools: dict[str, dict] = {}
    clients: dict[str, Any] = {}

    for server_id in server_ids:
        client = None
        try:
            connection = next(
                (
                    item
                    for item in connections
                    if isinstance(item, dict)
                    and item.get("type") == "mcp"
                    and as_dict(item.get("info")).get("id") == server_id
                ),
                None,
            )
            if not connection:
                await emit_notification(
                    extra_params.get("__event_emitter__"),
                    level="warning",
                    content=f"MCP server '{server_id}' was not found",
                )
                continue
            if not as_dict(connection.get("config")).get("enable", True):
                await emit_notification(
                    extra_params.get("__event_emitter__"),
                    level="warning",
                    content=f"MCP server '{server_id}' is disabled",
                )
                continue

            try:
                has_access = await maybe_await(has_connection_access(user, connection))
            except TypeError:
                has_access = await maybe_await(has_connection_access(user, connection, None))
            if not has_access:
                await emit_notification(
                    extra_params.get("__event_emitter__"),
                    level="warning",
                    content=f"Access denied to MCP server '{server_id}'",
                )
                continue

            headers: dict[str, Any] = {}
            auth_type = connection.get("auth_type", "")
            if auth_type == "bearer":
                headers["Authorization"] = f"Bearer {connection.get('key', '')}"
            elif auth_type == "session":
                token = getattr(getattr(request.state, "token", None), "credentials", "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            elif auth_type == "system_oauth":
                oauth_token = extra_params.get("__oauth_token__")
                if isinstance(oauth_token, dict) and oauth_token.get("access_token"):
                    headers["Authorization"] = f"Bearer {oauth_token['access_token']}"
            elif auth_type in {"oauth_2.1", "oauth_2.1_static"}:
                with suppress(Exception):
                    oauth_token = await request.app.state.oauth_client_manager.get_oauth_token(
                        getattr(user, "id", ""), f"mcp:{server_id}"
                    )
                    if oauth_token and oauth_token.get("access_token"):
                        headers["Authorization"] = f"Bearer {oauth_token['access_token']}"

            configured_headers = connection.get("headers")
            if isinstance(configured_headers, dict):
                headers.update(configured_headers)

            if ENABLE_FORWARD_USER_INFO_HEADERS and user:
                headers = include_user_info_headers(headers, user)
                if FORWARD_SESSION_INFO_HEADER_CHAT_ID and metadata.get("chat_id"):
                    headers[FORWARD_SESSION_INFO_HEADER_CHAT_ID] = metadata["chat_id"]
                if FORWARD_SESSION_INFO_HEADER_MESSAGE_ID and metadata.get("message_id"):
                    headers[FORWARD_SESSION_INFO_HEADER_MESSAGE_ID] = metadata["message_id"]

            function_filter = as_dict(connection.get("config")).get("function_name_filter_list", "")
            if isinstance(function_filter, str):
                function_filter = [item.strip() for item in function_filter.split(",") if item.strip()]
            if not isinstance(function_filter, list):
                function_filter = []

            client = MCPClient()
            lock = asyncio.Lock()
            await client.connect(url=connection.get("url", ""), headers=headers or None)
            specs = await client.list_tool_specs() or []

            def make_callable(mcp_client: Any, upstream_name: str, call_lock: asyncio.Lock) -> Callable[..., Awaitable[Any]]:
                async def tool_function(**kwargs: Any) -> Any:
                    async with call_lock:
                        return await mcp_client.call_tool(upstream_name, function_args=kwargs)

                return tool_function

            prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", server_id)
            loaded = 0
            for spec in specs:
                if not isinstance(spec, dict):
                    continue
                upstream_name = spec.get("name")
                if not isinstance(upstream_name, str) or not upstream_name:
                    continue
                if function_filter and not is_string_allowed(upstream_name, function_filter):
                    continue
                exposed_name = f"{prefix}_{upstream_name}"
                exposed_spec = clean_tool_spec({**spec, "name": exposed_name})
                tools[exposed_name] = {
                    "tool_id": f"server:mcp:{server_id}",
                    "spec": exposed_spec,
                    "callable": make_callable(client, upstream_name, lock),
                    "type": "mcp",
                    "direct": False,
                }
                loaded += 1

            clients[server_id] = client
            if debug:
                log.info("[SubAgent] loaded %s MCP tools from %s", loaded, server_id)
        except Exception as exc:
            log.warning("[SubAgent] failed to load MCP tools from %s: %s", server_id, exc)
            if client is not None:
                with suppress(Exception):
                    await client.disconnect()
            await emit_notification(
                extra_params.get("__event_emitter__"),
                level="warning",
                content=f"Could not load MCP tools from '{server_id}': {exc}",
            )

    return tools, clients


async def cleanup_mcp_clients(clients: Mapping[str, Any]) -> None:
    for client in reversed(list((clients or {}).values())):
        with suppress(Exception):
            await client.disconnect()


# -----------------------------------------------------------------------------
# Tool loading
# -----------------------------------------------------------------------------


def disabled_builtin_names_from_valves(valves: Any) -> set[str]:
    disabled: set[str] = set()
    for valve_name, category in VALVE_TO_BUILTIN_CATEGORY.items():
        if not bool(getattr(valves, valve_name, True)):
            disabled.update(BUILTIN_CATEGORY_TO_FUNCTIONS.get(category, set()))
    return disabled


def apply_tool_name_exclusions(tools: dict[str, dict], names_csv: str) -> None:
    for name in split_csv(names_csv):
        tools.pop(name, None)


def remove_self_tools(tools: dict[str, dict]) -> None:
    for name in SELF_TOOL_FUNCTION_NAMES:
        tools.pop(name, None)


def merge_tools(base: dict[str, dict], incoming: Mapping[str, dict], *, debug: bool, label: str) -> dict[str, dict]:
    out = dict(base)
    for name, tool in incoming.items():
        final_name = name
        if final_name in out:
            tool_id = str(tool.get("tool_id") or tool.get("type") or label)
            candidate = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', tool_id)}_{name}"
            suffix = 2
            while candidate in out:
                candidate = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', tool_id)}_{suffix}_{name}"
                suffix += 1
            if debug:
                log.warning("[SubAgent] tool name collision for %s; exposed as %s", name, candidate)
            final_name = candidate
            tool = {**tool, "spec": {**as_dict(tool.get("spec")), "name": final_name}}
        out[final_name] = dict(tool)
    return out


async def register_view_skill_fallback(tools: dict[str, dict], request: Request, extra_params: dict) -> None:
    """Provide view_skill when a manifest exists but upstream did not inject it."""
    if "view_skill" in tools:
        return
    try:
        from open_webui.tools.builtin import view_skill
        from open_webui.utils.tools import (
            convert_function_to_pydantic_model,
            convert_pydantic_model_to_openai_function_spec,
            get_async_tool_function_and_apply_extra_params,
        )

        callable_fn = await get_async_tool_function_and_apply_extra_params(
            view_skill,
            {
                "__request__": request,
                "__user__": extra_params.get("__user__", {}),
                "__event_emitter__": extra_params.get("__event_emitter__"),
                "__event_call__": extra_params.get("__event_call__"),
                "__metadata__": extra_params.get("__metadata__"),
                "__chat_id__": extra_params.get("__chat_id__"),
                "__message_id__": extra_params.get("__message_id__"),
                "__skill_ids__": extra_params.get("__skill_ids__", []),
            },
        )
        model = convert_function_to_pydantic_model(view_skill)
        spec = clean_tool_spec(convert_pydantic_model_to_openai_function_spec(model))
        tools["view_skill"] = {
            "tool_id": "builtin:view_skill",
            "callable": callable_fn,
            "spec": spec,
            "type": "builtin",
        }
    except Exception as exc:
        log.debug("[SubAgent] could not register view_skill fallback: %s", exc)


async def load_sub_agent_tools(
    *,
    request: Request,
    user: Any,
    valves: Any,
    metadata: dict,
    model: dict,
    extra_params: dict,
    self_tool_id: Optional[str],
) -> tuple[dict[str, dict], dict[str, Any]]:
    from open_webui.utils.tools import get_builtin_tools, get_tools

    try:
        from open_webui.utils.tools import get_terminal_tools
    except Exception:
        get_terminal_tools = None

    debug = bool(getattr(valves, "DEBUG", False))
    event_emitter = extra_params.get("__event_emitter__")

    terminal_id = await resolve_terminal_id(metadata, request)
    if terminal_id:
        metadata["terminal_id"] = terminal_id
        as_dict(extra_params.get("__metadata__"))["terminal_id"] = terminal_id

    direct_tool_servers = await resolve_direct_tool_servers(metadata, request)
    if direct_tool_servers:
        metadata["tool_servers"] = direct_tool_servers
        as_dict(extra_params.get("__metadata__"))["tool_servers"] = direct_tool_servers

    selected_tool_ids = split_csv(getattr(valves, "AVAILABLE_TOOL_IDS", ""))
    if not selected_tool_ids:
        selected_tool_ids = [str(item) for item in metadata.get("tool_ids", []) if isinstance(item, str)]

    excluded_ids = set(split_csv(getattr(valves, "EXCLUDED_TOOL_IDS", "")))
    if self_tool_id:
        excluded_ids.add(self_tool_id)

    selected_tool_ids = [tool_id for tool_id in selected_tool_ids if tool_id not in excluded_ids]
    regular_ids = [tool_id for tool_id in selected_tool_ids if not tool_id.startswith("builtin:")]
    mcp_ids = [tool_id for tool_id in regular_ids if tool_id.startswith("server:mcp:")]
    non_mcp_ids = [tool_id for tool_id in regular_ids if not tool_id.startswith("server:mcp:")]

    if debug:
        log.info("[SubAgent] selected regular tool IDs: %s", regular_ids)
        log.info("[SubAgent] terminal_id: %s", terminal_id or "")
        log.info("[SubAgent] direct tool servers: %s", len(direct_tool_servers))

    tools: dict[str, dict] = {}
    mcp_clients: dict[str, Any] = {}

    if non_mcp_ids:
        try:
            loaded = await get_tools(
                request=request,
                tool_ids=non_mcp_ids,
                user=user,
                extra_params=extra_params,
            )
            tools = merge_tools(tools, loaded or {}, debug=debug, label="regular")
        except Exception as exc:
            log.exception("[SubAgent] regular tool loading failed")
            await emit_notification(event_emitter, level="warning", content=f"Could not load selected tools: {exc}")

    if mcp_ids:
        loaded_mcp, mcp_clients = await resolve_mcp_tools(
            request=request,
            user=user,
            tool_ids=mcp_ids,
            extra_params=extra_params,
            debug=debug,
        )
        tools = merge_tools(tools, loaded_mcp, debug=debug, label="mcp")

    if terminal_id and bool(getattr(valves, "ENABLE_TERMINAL_TOOLS", True)):
        if get_terminal_tools is None:
            if debug:
                log.info("[SubAgent] terminal helper unavailable in this Open WebUI version")
        else:
            try:
                raw_terminal_tools = await maybe_await(
                    get_terminal_tools(
                        request=request,
                        terminal_id=terminal_id,
                        user=user,
                        extra_params=extra_params,
                    )
                )
                terminal_tools = normalize_terminal_tools_result(raw_terminal_tools, extra_params)
                tools = merge_tools(tools, terminal_tools, debug=debug, label="terminal")
            except Exception as exc:
                log.exception("[SubAgent] terminal tool loading failed")
                await emit_notification(event_emitter, level="warning", content=f"Could not load terminal tools: {exc}")

    if direct_tool_servers:
        try:
            direct_tools = build_direct_tools(direct_tool_servers)
            tools = merge_tools(tools, direct_tools, debug=debug, label="direct")
            prompts = extract_direct_tool_server_prompts(direct_tools)
            if prompts:
                extra_params["__direct_tool_server_system_prompts__"] = prompts
            else:
                extra_params.pop("__direct_tool_server_system_prompts__", None)
        except Exception as exc:
            log.exception("[SubAgent] direct tool loading failed")
            extra_params.pop("__direct_tool_server_system_prompts__", None)
            await emit_notification(event_emitter, level="warning", content=f"Could not load direct tools: {exc}")
    else:
        extra_params.pop("__direct_tool_server_system_prompts__", None)

    try:
        builtin_tools = await maybe_await(
            get_builtin_tools(
                request=request,
                extra_params=extra_params,
                features=as_dict(metadata.get("features")),
                model=model,
            )
        )
        disabled_names = disabled_builtin_names_from_valves(valves)
        allow_unknown = bool(getattr(valves, "ALLOW_UNKNOWN_BUILTIN_TOOLS", True))
        known_names = set().union(*BUILTIN_CATEGORY_TO_FUNCTIONS.values())
        filtered: dict[str, dict] = {}
        for name, tool in (builtin_tools or {}).items():
            if name in disabled_names:
                continue
            if name not in known_names and not allow_unknown:
                if debug:
                    log.warning("[SubAgent] unknown builtin skipped by policy: %s", name)
                continue
            filtered[name] = tool
        tools = merge_tools(tools, filtered, debug=debug, label="builtin")
    except Exception as exc:
        log.exception("[SubAgent] builtin tool loading failed")
        await emit_notification(event_emitter, level="warning", content=f"Could not load builtin tools: {exc}")

    remove_self_tools(tools)
    apply_tool_name_exclusions(tools, getattr(valves, "EXCLUDED_TOOL_NAMES", ""))

    if debug:
        log.info("[SubAgent] total exposed tools: %s", len(tools))

    return tools, mcp_clients


# -----------------------------------------------------------------------------
# Inlet filters and prompts
# -----------------------------------------------------------------------------


async def apply_inlet_filters_if_enabled(
    *,
    enabled: bool,
    request: Request,
    model: dict,
    form_data: dict,
    extra_params: dict,
) -> dict:
    if not enabled:
        return form_data
    try:
        from open_webui.models.functions import Functions
        from open_webui.utils.filter import get_sorted_filter_ids, process_filter_functions

        local_extra = dict(extra_params)
        if isinstance(local_extra.get("__user__"), dict):
            local_extra["__user__"] = dict(local_extra["__user__"])

        filter_ids = await maybe_await(
            get_sorted_filter_ids(
                request,
                model,
                as_dict(form_data.get("metadata")).get("filter_ids", []),
            )
        )
        filters = []
        for filter_id in filter_ids or []:
            fn = await maybe_await(Functions.get_function_by_id(filter_id))
            if fn:
                filters.append(fn)
        form_data, _ = await maybe_await(
            process_filter_functions(
                request=request,
                filter_functions=filters,
                filter_type="inlet",
                form_data=form_data,
                extra_params=local_extra,
            )
        )
    except Exception as exc:
        log.debug("[SubAgent] inlet filters skipped after error: %s", exc)
    return form_data


def append_tool_server_prompts(form_data: dict, extra_params: dict) -> dict:
    prompts: list[str] = []
    terminal_prompt = extra_params.get("__terminal_system_prompt__")
    if isinstance(terminal_prompt, str) and terminal_prompt.strip():
        prompts.append(terminal_prompt.strip())
    direct_prompts = extra_params.get("__direct_tool_server_system_prompts__")
    if isinstance(direct_prompts, list):
        prompts.extend(item.strip() for item in direct_prompts if isinstance(item, str) and item.strip())

    merged = merge_prompt_sections(prompts)
    if not merged:
        return form_data

    messages = list(form_data.get("messages", []))
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        first = dict(messages[0])
        content = first.get("content")
        if isinstance(content, list):
            first["content"] = [
                {**part, "text": f"{part.get('text', '')}\n\n{merged}"}
                if isinstance(part, dict) and part.get("type") == "text"
                else part
                for part in content
            ]
        elif isinstance(content, str) and content:
            first["content"] = f"{content}\n\n{merged}"
        else:
            first["content"] = merged
        messages[0] = first
    else:
        messages.insert(0, {"role": "system", "content": merged})
    form_data["messages"] = messages
    return form_data


def add_iteration_note(
    messages: list[dict],
    *,
    iteration: int,
    max_iterations: int,
    role: Literal["user", "system"],
) -> list[dict]:
    note = f"[Sub-agent iteration {iteration}/{max_iterations}]"
    if iteration == max_iterations:
        note += " This is your final tool-call opportunity."

    out = list(messages)
    last = out[-1] if out and isinstance(out[-1], dict) else None
    if role == "user" and isinstance(last, dict) and last.get("role") == "user":
        merged = dict(last)
        content = merged.get("content")
        if isinstance(content, list):
            merged["content"] = content + [{"type": "text", "text": f"\n\n{note}"}]
        elif isinstance(content, str) and content:
            merged["content"] = f"{content}\n\n{note}"
        else:
            merged["content"] = note
        out[-1] = merged
    else:
        out.append({"role": role, "content": note})
    return out


async def create_completion(
    *,
    request: Request,
    user: Any,
    model_id: str,
    messages: list[dict],
    tools: dict[str, dict],
    model: dict,
    extra_params: dict,
    apply_inlet_filters: bool,
) -> Any:
    from open_webui.utils.chat import generate_chat_completion

    metadata = as_dict(extra_params.get("__metadata__"))
    form_data = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "metadata": {
            **metadata,
            "task": "sub_agent",
            "model_id": model_id,
            "filter_ids": metadata.get("filter_ids", []),
        },
    }
    tools_param = build_tools_param(tools)
    if tools_param:
        form_data["tools"] = tools_param

    form_data = await apply_inlet_filters_if_enabled(
        enabled=apply_inlet_filters,
        request=request,
        model=model,
        form_data=form_data,
        extra_params=extra_params,
    )
    form_data = append_tool_server_prompts(form_data, extra_params)

    return await generate_chat_completion(
        request=request,
        form_data=form_data,
        user=user,
        bypass_filter=True,
    )


def parse_completion_response(response: Any) -> tuple[Optional[dict], Optional[str]]:
    if isinstance(response, JSONResponse):
        try:
            data = json.loads(bytes(response.body).decode("utf-8"))
        except Exception:
            return None, f"API error (status {response.status_code}): could not parse error response"
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return None, str(error.get("message") or error)
            if isinstance(error, str):
                return None, error
            return None, str(data.get("message") or data)
        return None, str(data)

    if not isinstance(response, dict):
        return None, f"Unexpected response type: {type(response).__name__}"
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "No response choices returned by model"
    first = choices[0]
    if not isinstance(first, Mapping):
        return None, f"Malformed response: choices[0] is {type(first).__name__}"
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None, f"Malformed response: message is {type(message).__name__}"
    return dict(message), None


def normalize_tool_calls(raw_tool_calls: Any) -> tuple[list[dict], Optional[str]]:
    if not raw_tool_calls:
        return [], None
    if not isinstance(raw_tool_calls, Sequence) or isinstance(raw_tool_calls, (str, bytes)):
        return [], f"tool_calls must be a sequence, got {type(raw_tool_calls).__name__}"

    normalized: list[dict] = []
    for item in raw_tool_calls:
        if not isinstance(item, Mapping):
            continue
        func = item.get("function")
        if not isinstance(func, Mapping):
            continue
        name = func.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = func.get("arguments", "{}")
        if not isinstance(args, str):
            try:
                args = compact_json(args)
            except Exception:
                args = str(args)
        normalized.append({**dict(item), "function": {**dict(func), "arguments": args}})

    if not normalized:
        return [], "model returned tool calls, but none had a valid function name"
    return normalized, None


async def rebind_tool_callable(tool_function: Callable, extra_params: dict) -> Callable:
    try:
        from open_webui.utils.tools import get_updated_tool_function

        return await maybe_await(
            get_updated_tool_function(
                function=tool_function,
                extra_params={
                    "__request__": extra_params.get("__request__"),
                    "__user__": extra_params.get("__user__"),
                    "__model__": extra_params.get("__model__"),
                    "__metadata__": extra_params.get("__metadata__"),
                    "__chat_id__": extra_params.get("__chat_id__"),
                    "__message_id__": extra_params.get("__message_id__"),
                    "__oauth_token__": extra_params.get("__oauth_token__"),
                    "__messages__": extra_params.get("__messages__", []),
                    "__files__": extra_params.get("__files__", []),
                    "__skill_ids__": extra_params.get("__skill_ids__", []),
                    "__event_emitter__": extra_params.get("__event_emitter__"),
                    "__event_call__": extra_params.get("__event_call__"),
                },
            )
        )
    except Exception:
        return tool_function


async def run_with_optional_timeout(awaitable: Awaitable, timeout_seconds: int) -> Any:
    if timeout_seconds and timeout_seconds > 0:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    return await awaitable


async def execute_tool_call(
    *,
    tool_call: dict,
    tools: dict[str, dict],
    extra_params: dict,
    event_emitter: Optional[Callable[[dict], Any]],
    allow_literal_arg_fallback: bool,
    tool_timeout_seconds: int,
) -> dict[str, str]:
    call_id = str(tool_call.get("id") or uuid.uuid4())
    func = as_dict(tool_call.get("function"))
    name = str(func.get("name") or "")
    raw_args = func.get("arguments", "{}")

    parse = parse_jsonish_arguments_compat if allow_literal_arg_fallback else parse_jsonish_arguments
    params, error = parse(raw_args)
    if error:
        return {"tool_call_id": call_id, "content": f"Error parsing arguments for {name or 'tool'}: {error}"}

    if name not in tools:
        return {"tool_call_id": call_id, "content": f"Tool '{name}' not found"}

    tool = tools[name]
    spec = as_dict(tool.get("spec"))
    params = filter_tool_params(params, spec)
    direct = bool(tool.get("direct", False))
    result: Any = None
    files: list = []
    embeds: list = []

    try:
        if direct:
            result = await run_with_optional_timeout(
                execute_direct_tool(name=name, params=params, tool=tool, extra_params=extra_params),
                tool_timeout_seconds,
            )
        else:
            callable_fn = tool.get("callable")
            if not callable(callable_fn):
                raise RuntimeError(f"Tool '{name}' has no callable")
            callable_fn = await rebind_tool_callable(callable_fn, extra_params)
            result = await run_with_optional_timeout(callable_fn(**params), tool_timeout_seconds)

        result, files, embeds = await process_tool_result(
            name=name,
            tool_type=str(tool.get("type") or ""),
            result=result,
            direct=direct,
            extra_params=extra_params,
        )
    except asyncio.TimeoutError:
        result = f"Error: tool '{name}' timed out after {tool_timeout_seconds} seconds"
    except Exception as exc:
        log.exception("[SubAgent] tool execution failed for %s", name)
        result = f"Error executing tool '{name}': {exc}"

    if not isinstance(result, str):
        result = pretty_json(result) if isinstance(result, (dict, list)) else str(result or "")

    await emit_terminal_event(name=name, params=params, result=result, event_emitter=event_emitter)
    if files:
        await emit_event(event_emitter, {"type": "files", "data": {"files": files}})
    if embeds:
        await emit_event(event_emitter, {"type": "embeds", "data": {"embeds": embeds}})
    await emit_citations_if_available(
        name=name,
        params=params,
        result=result,
        tool=tool,
        event_emitter=event_emitter,
    )

    return {"tool_call_id": call_id, "content": result}


async def run_sub_agent_loop(
    *,
    request: Request,
    user: Any,
    model_id: str,
    model: dict,
    messages: list[dict],
    tools: dict[str, dict],
    max_iterations: int,
    event_emitter: Optional[Callable[[dict], Any]],
    extra_params: dict,
    apply_inlet_filters: bool,
    iteration_note_role: Literal["user", "system"],
    status_limit_chars: int,
    max_tool_result_chars: int,
    allow_literal_arg_fallback: bool,
    tool_timeout_seconds: int,
    show_tool_args_in_status: bool,
    show_tool_results_in_status: bool,
) -> str:
    current = list(messages)

    for iteration in range(1, max_iterations + 1):
        await emit_status(
            event_emitter,
            f"Sub-agent iteration {iteration}/{max_iterations}",
            limit=status_limit_chars,
        )

        iteration_messages = add_iteration_note(
            current,
            iteration=iteration,
            max_iterations=max_iterations,
            role=iteration_note_role,
        )

        try:
            response = await create_completion(
                request=request,
                user=user,
                model_id=model_id,
                messages=iteration_messages,
                tools=tools,
                model=model,
                extra_params=extra_params,
                apply_inlet_filters=apply_inlet_filters,
            )
        except Exception as exc:
            log.exception("[SubAgent] completion failed")
            return f"Error during sub-agent completion: {exc}"

        message, error = parse_completion_response(response)
        if error:
            return error
        assert message is not None

        content = message.get("content") or ""
        if content:
            await emit_status(event_emitter, f"[Step {iteration}] Assistant: {content}", limit=status_limit_chars)

        tool_calls, tool_error = normalize_tool_calls(message.get("tool_calls", []))
        if tool_error:
            return f"{tool_error}. Content so far: {content or '(none)'}"
        if not tool_calls:
            return str(content or "")

        await emit_status(
            event_emitter,
            f"[Step {iteration}] Tool calls: {', '.join(as_dict(call.get('function')).get('name', 'unknown') for call in tool_calls)}",
            limit=status_limit_chars,
        )

        current.append({"role": "assistant", "content": str(content or ""), "tool_calls": tool_calls})

        for tool_call in tool_calls:
            func = as_dict(tool_call.get("function"))
            if show_tool_args_in_status:
                await emit_status(
                    event_emitter,
                    f"[Step {iteration}] Args for {func.get('name', 'tool')}: {func.get('arguments', '{}')}",
                    limit=status_limit_chars,
                )

            result = await execute_tool_call(
                tool_call=tool_call,
                tools=tools,
                extra_params={**extra_params, "__messages__": current},
                event_emitter=event_emitter,
                allow_literal_arg_fallback=allow_literal_arg_fallback,
                tool_timeout_seconds=tool_timeout_seconds,
            )
            result_content = truncate_text(result.get("content", ""), max_tool_result_chars)
            if show_tool_results_in_status:
                await emit_status(
                    event_emitter,
                    f"[Step {iteration}] Result: {result_content}",
                    limit=status_limit_chars,
                )
            current.append(
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_call_id", str(uuid.uuid4())),
                    "content": result_content,
                }
            )

    await emit_status(event_emitter, f"Max iterations ({max_iterations}) reached", limit=status_limit_chars)

    final_messages = current + [
        {
            "role": "user",
            "content": (
                "Maximum tool iterations reached. Provide the best final answer using only "
                "the information already gathered. State any remaining uncertainty."
            ),
        }
    ]
    try:
        response = await create_completion(
            request=request,
            user=user,
            model_id=model_id,
            messages=final_messages,
            tools={},
            model=model,
            extra_params=extra_params,
            apply_inlet_filters=apply_inlet_filters,
        )
        message, error = parse_completion_response(response)
        if error:
            return error
        return str((message or {}).get("content") or "")
    except Exception as exc:
        log.exception("[SubAgent] final completion failed")
        return f"Sub-agent reached the iteration limit and finalization failed: {exc}"


# -----------------------------------------------------------------------------
# Public Tools class
# -----------------------------------------------------------------------------


class Tools:
    """Delegate complex work to isolated sub-agent contexts."""

    class Valves(BaseModel):
        DEFAULT_MODEL: str = Field(
            default="",
            description="Model ID used by sub-agents. Empty means use the parent chat's model.",
        )
        MAX_ITERATIONS: int = Field(default=10, ge=1, le=100)
        MAX_PARALLEL_AGENTS: int = Field(default=5, ge=1, le=20)
        SHARE_TOOLS_ACROSS_PARALLEL_AGENTS: bool = Field(
            default=True,
            description=(
                "Load tools once for parallel tasks. Faster and preserves previous behavior. "
                "Disable for maximum isolation with stateful custom tools."
            ),
        )

        AVAILABLE_TOOL_IDS: str = Field(
            default="",
            description=(
                "Comma-separated regular tool IDs available to sub-agents. Empty means use the "
                "tools selected in the parent chat metadata. Builtin categories use the toggles below."
            ),
        )
        EXCLUDED_TOOL_IDS: str = Field(default="", description="Comma-separated regular tool IDs to exclude.")
        EXCLUDED_TOOL_NAMES: str = Field(
            default="",
            description="Comma-separated exposed tool function names to remove after loading.",
        )

        APPLY_INLET_FILTERS: bool = Field(default=True)
        ITERATION_NOTE_ROLE: Literal["user", "system"] = Field(default="user")

        ENABLE_TIME_TOOLS: bool = True
        ENABLE_WEB_TOOLS: bool = True
        ENABLE_IMAGE_TOOLS: bool = True
        ENABLE_KNOWLEDGE_TOOLS: bool = True
        ENABLE_CHAT_TOOLS: bool = True
        ENABLE_MEMORY_TOOLS: bool = True
        ENABLE_NOTES_TOOLS: bool = True
        ENABLE_CHANNELS_TOOLS: bool = True
        ENABLE_TERMINAL_TOOLS: bool = True
        ENABLE_CODE_INTERPRETER_TOOLS: bool = True
        ENABLE_SKILLS_TOOLS: bool = True
        ENABLE_TASK_TOOLS: bool = True
        ENABLE_AUTOMATION_TOOLS: bool = True
        ENABLE_CALENDAR_TOOLS: bool = True
        ALLOW_UNKNOWN_BUILTIN_TOOLS: bool = Field(
            default=True,
            description=(
                "Keep upstream-provided builtin tools that this tool version does not yet recognize. "
                "This improves forward compatibility; disable for a strict allow-list."
            ),
        )

        TOOL_TIMEOUT_SECONDS: int = Field(default=120, ge=0)
        MAX_TOOL_RESULT_CHARS: int = Field(default=20000, ge=0)
        STATUS_LIMIT_CHARS: int = Field(default=1000, ge=100)
        SHOW_TOOL_ARGS_IN_STATUS: bool = Field(
            default=False,
            description="Show tool-call arguments in status events. Disabled by default to avoid leaking secrets.",
        )
        SHOW_TOOL_RESULTS_IN_STATUS: bool = Field(
            default=False,
            description="Show tool results in status events. Disabled by default to avoid leaking sensitive output.",
        )
        ALLOW_LITERAL_ARG_FALLBACK: bool = Field(
            default=True,
            description=(
                "After JSON parsing fails, accept Python literal dicts from imperfect local models. "
                "Disable for strict JSON-only tool arguments."
            ),
        )
        DEBUG: bool = False

    class UserValves(BaseModel):
        SYSTEM_PROMPT: str = Field(
            default=(
                "You are a sub-agent operating autonomously inside an isolated context.\n\n"
                "Rules:\n"
                "1. Complete the delegated task without asking the user for clarification.\n"
                "2. Use available tools when they materially improve correctness.\n"
                "3. Try reasonable alternatives before giving up.\n"
                "4. Keep tool use focused; you have a limited number of iterations.\n"
                "5. Your final response is returned to the parent agent, not directly to the user.\n\n"
                "Final response requirements:\n"
                "- Answer the delegated task directly.\n"
                "- Include relevant evidence, citations, file names, or tool-derived facts when available.\n"
                "- If incomplete, say what was attempted, what failed, and what the parent agent should do next."
            ),
            description="System prompt used for all sub-agent contexts.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def _prepare_context(
        self,
        *,
        __user__: Optional[dict],
        __request__: Optional[Request],
        __model__: Optional[dict],
        __metadata__: Optional[dict],
        __event_emitter__: Optional[Callable[[dict], Any]],
        __event_call__: Optional[Callable[[dict], Any]],
        __chat_id__: Optional[str],
        __message_id__: Optional[str],
        __oauth_token__: Optional[dict],
        __messages__: Optional[list],
    ) -> tuple[Optional[dict], Optional[str]]:
        if __request__ is None:
            return None, compact_json({"error": "Request context not available. Cannot run sub-agent."})
        if __user__ is None:
            return None, compact_json({"error": "User context not available. Cannot run sub-agent."})

        model_id = model_id_from_context(
            explicit_default=self.valves.DEFAULT_MODEL,
            metadata=__metadata__,
            model=__model__,
        )
        if not model_id:
            return None, compact_json({"error": "No model ID available. Set DEFAULT_MODEL in Valves if needed."})

        metadata = normalize_metadata(
            __metadata__,
            model_id=model_id,
            chat_id=__chat_id__,
            message_id=__message_id__,
        )
        model = resolve_model_dict(__request__, model_id, __model__)
        user_obj = user_object_from_payload(__user__)
        skill_ids = get_skill_ids(metadata, __user__, __messages__)
        extra_params = build_extra_params(
            request=__request__,
            user_payload=__user__,
            model=model,
            metadata=metadata,
            event_emitter=__event_emitter__,
            event_call=__event_call__,
            chat_id=__chat_id__,
            message_id=__message_id__,
            oauth_token=__oauth_token__,
            messages=__messages__,
            skill_ids=skill_ids,
        )
        user_valves = coerce_user_valves(as_dict(__user__).get("valves", {}), self.UserValves)
        return {
            "request": __request__,
            "user_payload": __user__,
            "user": user_obj,
            "user_valves": user_valves,
            "model_id": model_id,
            "model": model,
            "metadata": metadata,
            "extra_params": extra_params,
            "skill_manifest": extract_skill_manifest(__messages__),
            "user_skill_tags": extract_user_skill_tags(__messages__),
            "event_emitter": __event_emitter__,
        }, None

    def _build_system_prompt(self, context: dict) -> str:
        sections: list[str] = [context["user_valves"].SYSTEM_PROMPT]
        if self.valves.ENABLE_SKILLS_TOOLS:
            sections.extend(context.get("user_skill_tags") or [])
            if context.get("skill_manifest"):
                sections.append(context["skill_manifest"])
        return merge_prompt_sections(sections)

    async def _load_tools_for_context(
        self,
        *,
        context: dict,
        self_tool_id: Optional[str],
    ) -> tuple[dict[str, dict], dict[str, Any]]:
        tools, mcp_clients = await load_sub_agent_tools(
            request=context["request"],
            user=context["user"],
            valves=self.valves,
            metadata=context["metadata"],
            model=context["model"],
            extra_params=context["extra_params"],
            self_tool_id=self_tool_id,
        )
        if context.get("skill_manifest") and self.valves.ENABLE_SKILLS_TOOLS:
            await register_view_skill_fallback(tools, context["request"], context["extra_params"])
        return tools, mcp_clients

    async def run_sub_agent(
        self,
        description: str,
        prompt: str,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __oauth_token__: Optional[dict] = None,
        __messages__: Optional[list] = None,
    ) -> str:
        """
        Delegate one complex task to an autonomous sub-agent.

        Use this when the task requires multi-step investigation, repeated tool
        use, or enough context that it would clutter the parent conversation.
        The sub-agent does not receive the parent chat history unless you include
        the necessary facts in ``prompt``.

        :param description: Brief status text shown to the user.
        :param prompt: Complete instructions and context for the sub-agent.
        :return: JSON containing the sub-agent result for the parent model.
        """
        context, error = await self._prepare_context(
            __user__=__user__,
            __request__=__request__,
            __model__=__model__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __event_call__=__event_call__,
            __chat_id__=__chat_id__,
            __message_id__=__message_id__,
            __oauth_token__=__oauth_token__,
            __messages__=__messages__,
        )
        if error:
            return error
        assert context is not None

        clean_description = description.strip() or "sub-agent task"
        await emit_status(
            context["event_emitter"],
            f"Starting sub-agent: {clean_description}",
            limit=self.valves.STATUS_LIMIT_CHARS,
        )

        tools, mcp_clients = await self._load_tools_for_context(context=context, self_tool_id=__id__)
        try:
            await emit_status(
                context["event_emitter"],
                f"Sub-agent started with {len(tools)} tools available",
                limit=self.valves.STATUS_LIMIT_CHARS,
            )
            result = await run_sub_agent_loop(
                request=context["request"],
                user=context["user"],
                model_id=context["model_id"],
                model=context["model"],
                messages=[
                    {"role": "system", "content": self._build_system_prompt(context)},
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                max_iterations=self.valves.MAX_ITERATIONS,
                event_emitter=context["event_emitter"],
                extra_params=context["extra_params"],
                apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                status_limit_chars=self.valves.STATUS_LIMIT_CHARS,
                max_tool_result_chars=self.valves.MAX_TOOL_RESULT_CHARS,
                allow_literal_arg_fallback=self.valves.ALLOW_LITERAL_ARG_FALLBACK,
                tool_timeout_seconds=self.valves.TOOL_TIMEOUT_SECONDS,
                show_tool_args_in_status=self.valves.SHOW_TOOL_ARGS_IN_STATUS,
                show_tool_results_in_status=self.valves.SHOW_TOOL_RESULTS_IN_STATUS,
            )
            await emit_status(
                context["event_emitter"],
                f"Sub-agent completed: {clean_description}",
                done=True,
                limit=self.valves.STATUS_LIMIT_CHARS,
            )
            return pretty_json(
                {
                    "note": "The user does not see this result directly; only the parent agent receives it.",
                    "result": result,
                }
            )
        finally:
            await cleanup_mcp_clients(mcp_clients)

    async def run_parallel_sub_agents(
        self,
        tasks: list[SubAgentTaskItem],
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __model__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __id__: Optional[str] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
        __event_call__: Optional[Callable[[dict], Any]] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
        __oauth_token__: Optional[dict] = None,
        __messages__: Optional[list] = None,
    ) -> str:
        """
        Run independent sub-agent tasks concurrently.

        Use this only when each task can be completed without seeing another
        task's result.  Results are returned in the same order as the input list.
        """
        validated, validation_error = self._validate_parallel_tasks(tasks)
        if validation_error:
            return validation_error
        assert validated is not None

        context, error = await self._prepare_context(
            __user__=__user__,
            __request__=__request__,
            __model__=__model__,
            __metadata__=__metadata__,
            __event_emitter__=__event_emitter__,
            __event_call__=__event_call__,
            __chat_id__=__chat_id__,
            __message_id__=__message_id__,
            __oauth_token__=__oauth_token__,
            __messages__=__messages__,
        )
        if error:
            return error
        assert context is not None

        mapping = ", ".join(f"[{i + 1}] {task['description']}" for i, task in enumerate(validated))
        await emit_status(
            context["event_emitter"],
            f"Running {len(validated)} sub-agents: {mapping}",
            limit=self.valves.STATUS_LIMIT_CHARS,
        )

        shared_tools: Optional[dict[str, dict]] = None
        shared_mcp_clients: dict[str, Any] = {}
        if self.valves.SHARE_TOOLS_ACROSS_PARALLEL_AGENTS:
            shared_tools, shared_mcp_clients = await self._load_tools_for_context(context=context, self_tool_id=__id__)

        async def run_one(index: int, task: dict) -> dict:
            async def indexed_emitter(event: dict) -> None:
                if not callable(context["event_emitter"]):
                    return
                if isinstance(event, dict) and event.get("type") == "status" and isinstance(event.get("data"), dict):
                    data = dict(event["data"])
                    if data.get("description"):
                        data["description"] = f"[{index}] {data['description']}"
                    await context["event_emitter"]({"type": "status", "data": data})
                else:
                    await context["event_emitter"](event)

            local_context = dict(context)
            local_extra = dict(context["extra_params"])
            local_extra["__event_emitter__"] = indexed_emitter if callable(context["event_emitter"]) else None
            local_context["extra_params"] = local_extra
            local_context["event_emitter"] = local_extra["__event_emitter__"]

            local_mcp_clients: dict[str, Any] = {}
            tools = shared_tools
            try:
                if tools is None:
                    tools, local_mcp_clients = await self._load_tools_for_context(context=local_context, self_tool_id=__id__)
                result = await run_sub_agent_loop(
                    request=local_context["request"],
                    user=local_context["user"],
                    model_id=local_context["model_id"],
                    model=local_context["model"],
                    messages=[
                        {"role": "system", "content": self._build_system_prompt(local_context)},
                        {"role": "user", "content": task["prompt"]},
                    ],
                    tools=tools or {},
                    max_iterations=self.valves.MAX_ITERATIONS,
                    event_emitter=local_context["event_emitter"],
                    extra_params=local_context["extra_params"],
                    apply_inlet_filters=self.valves.APPLY_INLET_FILTERS,
                    iteration_note_role=self.valves.ITERATION_NOTE_ROLE,
                    status_limit_chars=self.valves.STATUS_LIMIT_CHARS,
                    max_tool_result_chars=self.valves.MAX_TOOL_RESULT_CHARS,
                    allow_literal_arg_fallback=self.valves.ALLOW_LITERAL_ARG_FALLBACK,
                    tool_timeout_seconds=self.valves.TOOL_TIMEOUT_SECONDS,
                    show_tool_args_in_status=self.valves.SHOW_TOOL_ARGS_IN_STATUS,
                    show_tool_results_in_status=self.valves.SHOW_TOOL_RESULTS_IN_STATUS,
                )
                return {"description": task["description"], "result": result}
            except Exception as exc:
                log.exception("[SubAgent] parallel task failed: %s", task.get("description"))
                return {"description": task["description"], "error": str(exc) or type(exc).__name__}
            finally:
                if local_mcp_clients:
                    await cleanup_mcp_clients(local_mcp_clients)

        try:
            results = await asyncio.gather(
                *(run_one(i + 1, task) for i, task in enumerate(validated)),
                return_exceptions=True,
            )
        finally:
            if shared_mcp_clients:
                await cleanup_mcp_clients(shared_mcp_clients)

        processed: list[dict] = []
        for index, item in enumerate(results):
            if isinstance(item, BaseException):
                processed.append(
                    {
                        "description": validated[index]["description"],
                        "error": str(item) or type(item).__name__,
                    }
                )
            else:
                processed.append(item)

        await emit_status(
            context["event_emitter"],
            f"Sub-agents completed: {mapping}",
            done=True,
            limit=self.valves.STATUS_LIMIT_CHARS,
        )
        return pretty_json(
            {
                "note": "The user does not see these results directly; only the parent agent receives them.",
                "results": processed,
            }
        )

    def _validate_parallel_tasks(self, tasks: Any) -> tuple[Optional[list[dict[str, str]]], Optional[str]]:
        if not isinstance(tasks, list):
            return None, compact_json(
                {
                    "error": f"tasks must be a list, got {type(tasks).__name__}",
                    "expected_format": [{"description": "Task summary", "prompt": "Detailed instructions"}],
                }
            )
        if not tasks:
            return None, compact_json({"error": "tasks array is empty"})
        if len(tasks) > self.valves.MAX_PARALLEL_AGENTS:
            return None, compact_json(
                {
                    "error": f"tasks count ({len(tasks)}) exceeds MAX_PARALLEL_AGENTS ({self.valves.MAX_PARALLEL_AGENTS})",
                    "max_parallel_agents": self.valves.MAX_PARALLEL_AGENTS,
                }
            )

        validated: list[dict[str, str]] = []
        for index, raw in enumerate(tasks):
            value = raw
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    return None, compact_json({"error": f"tasks[{index}] must be an object, got unparseable string"})
            try:
                item = value if isinstance(value, SubAgentTaskItem) else SubAgentTaskItem.model_validate(value)
            except Exception as exc:
                if hasattr(exc, "errors"):
                    errors = exc.errors()
                    if errors:
                        first = errors[0]
                        loc = ".".join(str(part) for part in first.get("loc", ()))
                        msg = first.get("msg", "is invalid")
                        return None, compact_json({"error": f"tasks[{index}].{loc} {msg}"})
                return None, compact_json({"error": f"tasks[{index}] is invalid"})

            description = item.description.strip()
            prompt = item.prompt.strip()
            if not description:
                return None, compact_json({"error": f"tasks[{index}].description cannot be empty"})
            if not prompt:
                return None, compact_json({"error": f"tasks[{index}].prompt cannot be empty"})
            validated.append({"description": description, "prompt": prompt})

        return validated, None
