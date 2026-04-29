"""
title: Memory Curator
author: Local
version: 0.1.0
required_open_webui_version: 0.6.0
description: Local JSON memory curator for reviewable memory proposals, explicit apply/reject, search, archive, delete, pruning, and portable export. Does not call models, networks, embeddings, databases, or vendor-specific memory APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

JsonDict = dict[str, Any]


class Tools:
    """
    Open WebUI tool for local, scoped memory curation.

    Design intent:
    - The tool stores reviewable memory proposals and approved memories.
    - The tool does not decide what is true, call a model, call an embedding API,
      call Open WebUI memory APIs, or perform any network operation.
    - Reads and searches do not mutate state.
    - Long-term state changes are explicit: propose -> apply, reject, archive, delete.
    - State remains local JSON for portability and vendor independence.
    - A future read-only injector/filter can retrieve a bounded subset of active
      memories and insert them as ordinary reference context, not privileged policy.
    """

    TOOL_VERSION = "0.1.0"
    INDEX_FILE = "index.json"
    MEMORIES_DIR = "memories"
    PROPOSALS_DIR = "proposals"
    EVENTS_FILE = "events.jsonl"

    CATEGORIES = {
        "user_preference",
        "project_fact",
        "environment",
        "decision",
        "constraint",
        "workflow",
        "relationship",
        "writing_preference",
        "technical_note",
        "other",
    }
    MEMORY_STATUSES = {"active", "archived"}
    PROPOSAL_STATUSES = {"pending", "applied", "rejected"}
    OPERATIONS = {"create", "update"}
    SENSITIVITIES = {"normal", "sensitive"}

    class Valves(BaseModel):
        data_dir: str = Field(
            default="/app/backend/data/memory_curator",
            description="Base directory containing per-scope memory stores.",
        )
        default_scope: str = Field(
            default="default",
            description="Default scope if no chat, project, or explicit scope is provided.",
        )
        max_memory_chars: int = Field(
            default=8_000,
            description="Maximum memory_text characters accepted per proposal or memory.",
        )
        max_title_chars: int = Field(
            default=160,
            description="Maximum title length retained for each memory or proposal.",
        )
        max_rationale_chars: int = Field(
            default=2_000,
            description="Maximum rationale/change_note characters retained for proposals.",
        )
        max_source_ref_chars: int = Field(
            default=500,
            description="Maximum source_ref length retained for each memory or proposal.",
        )
        max_tags: int = Field(
            default=20,
            description="Maximum number of tags retained per memory or proposal.",
        )
        max_tag_chars: int = Field(
            default=50,
            description="Maximum characters retained per tag.",
        )
        max_duplicate_candidates: int = Field(
            default=5,
            description="Maximum near-duplicate candidates returned during proposal creation.",
        )
        near_duplicate_threshold: float = Field(
            default=0.72,
            description="Lexical score at or above which an existing memory is reported as a near duplicate.",
        )
        redact_secrets_on_save: bool = Field(
            default=True,
            description="Redact common secret patterns before saving proposal/memory text.",
        )
        redact_long_tokens_on_save: bool = Field(
            default=True,
            description="Redact very long token-like strings before saving proposal/memory text.",
        )
        reject_secret_like_memories: bool = Field(
            default=True,
            description="Reject proposal memory_text that appears to contain credentials or private keys.",
        )
        lock_timeout_seconds: float = Field(
            default=5.0,
            description="Maximum time to wait for a local JSON file lock.",
        )
        stale_lock_seconds: float = Field(
            default=120.0,
            description="Age after which a lock file is considered stale and removable.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _json(self, payload: JsonDict) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)

    def _ok(self, data: Any = None, **extra: Any) -> str:
        payload: JsonDict = {"ok": True}
        if data is not None:
            payload["data"] = data
        payload.update(extra)
        return self._json(payload)

    def _err(self, code: str, message: str, **extra: Any) -> str:
        payload: JsonDict = {
            "ok": False,
            "error": code,
            "message": message,
        }
        payload.update(extra)
        return self._json(payload)

    # ------------------------------------------------------------------
    # Path, scope, time, and ID helpers
    # ------------------------------------------------------------------

    def _scope_name(self, scope_id: Optional[str]) -> str:
        raw = (scope_id or self.valves.default_scope or "default").strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
        if not safe or safe in {".", ".."}:
            safe = "default"
        return safe[:80]

    def _base_dir(self) -> Path:
        return Path(str(self.valves.data_dir or "/app/backend/data/memory_curator")).expanduser()

    def _scope_dir(self, scope_id: Optional[str]) -> Path:
        return self._base_dir() / self._scope_name(scope_id)

    def _index_path(self, scope_id: Optional[str]) -> Path:
        return self._scope_dir(scope_id) / self.INDEX_FILE

    def _memories_dir(self, scope_id: Optional[str]) -> Path:
        return self._scope_dir(scope_id) / self.MEMORIES_DIR

    def _proposals_dir(self, scope_id: Optional[str]) -> Path:
        return self._scope_dir(scope_id) / self.PROPOSALS_DIR

    def _events_path(self, scope_id: Optional[str]) -> Path:
        return self._scope_dir(scope_id) / self.EVENTS_FILE

    def _validate_id(self, value: str, field_name: str = "id") -> str:
        safe = (value or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", safe):
            raise ValueError(
                f"{field_name} must contain only letters, numbers, underscores, and hyphens."
            )
        return safe

    def _memory_path(self, memory_id: str, scope_id: Optional[str]) -> Path:
        safe_id = self._validate_id(memory_id, "memory_id")
        return self._memories_dir(scope_id) / f"{safe_id}.json"

    def _proposal_path(self, proposal_id: str, scope_id: Optional[str]) -> Path:
        safe_id = self._validate_id(proposal_id, "proposal_id")
        return self._proposals_dir(scope_id) / f"{safe_id}.json"

    def _utc_now(self) -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _new_id(self, prefix: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}_{stamp}_{secrets.token_hex(4)}"

    def _created_by(self, user: Optional[dict]) -> str:
        if not isinstance(user, dict):
            return "unknown"
        return str(user.get("name") or user.get("email") or user.get("id") or "unknown")[:160]

    def _field_default(self, value: Any, fallback: Any = "") -> Any:
        """
        Open WebUI supplies real argument values when a tool is called.
        Direct Python tests may leave pydantic FieldInfo objects in defaults,
        so normalize them here to keep the tool robust and easy to test.
        """
        if value.__class__.__name__ == "FieldInfo":
            default = getattr(value, "default", fallback)
            if default is Ellipsis or str(default) == "PydanticUndefined":
                return fallback
            return default
        return value

    def _as_bool(self, value: Any, fallback: bool = False) -> bool:
        value = self._field_default(value, fallback)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _as_int(self, value: Any, fallback: int = 0) -> int:
        value = self._field_default(value, fallback)
        try:
            return int(value)
        except Exception:
            return fallback

    def _as_float(self, value: Any, fallback: float = 0.0) -> float:
        value = self._field_default(value, fallback)
        try:
            return float(value)
        except Exception:
            return fallback

    # ------------------------------------------------------------------
    # Local locking and atomic JSON persistence
    # ------------------------------------------------------------------

    @contextmanager
    def _file_lock(self, target: Path):
        """
        Best-effort local lock using an adjacent .lock file.

        This is dependency-free and suitable for a normal local/container Open
        WebUI deployment. It is not a distributed lock for network filesystems.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_name(f"{target.name}.lock")
        start = time.monotonic()
        fd: Optional[int] = None

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = json.dumps(
                    {
                        "pid": os.getpid(),
                        "created_at": self._utc_now(),
                        "target": target.name,
                    },
                    ensure_ascii=False,
                )
                os.write(fd, payload.encode("utf-8"))
                break
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > float(self.valves.stale_lock_seconds):
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue

                if time.monotonic() - start > float(self.valves.lock_timeout_seconds):
                    raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
                time.sleep(0.05)

        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_json_atomic(self, path: Path, data: JsonDict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp_name = tmp.name
                json.dump(data, tmp, ensure_ascii=False, indent=2)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())

            os.replace(tmp_name, path)
        finally:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass

    def _load_json_file(self, path: Path) -> JsonDict:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must contain a JSON object at the top level.")
        return data

    def _append_event_no_lock(self, scope_id: Optional[str], event: JsonDict) -> None:
        path = self._events_path(scope_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------
    # Index, memory, and proposal helpers
    # ------------------------------------------------------------------

    def _empty_index(self, scope_id: Optional[str]) -> JsonDict:
        scope_name = self._scope_name(scope_id)
        return {
            "schema_version": "1.0",
            "scope_id": scope_name,
            "memories": [],
            "proposals": [],
        }

    def _load_index_no_lock(self, scope_id: Optional[str]) -> JsonDict:
        path = self._index_path(scope_id)
        if not path.exists():
            return self._empty_index(scope_id)

        data = self._load_json_file(path)
        data.setdefault("schema_version", "1.0")
        data.setdefault("scope_id", self._scope_name(scope_id))
        data.setdefault("memories", [])
        data.setdefault("proposals", [])

        if not isinstance(data["memories"], list):
            raise ValueError("index.json field 'memories' must be a list.")
        if not isinstance(data["proposals"], list):
            raise ValueError("index.json field 'proposals' must be a list.")

        return data

    def _save_index_no_lock(self, scope_id: Optional[str], index: JsonDict) -> None:
        self._write_json_atomic(self._index_path(scope_id), index)

    def _load_memory_no_lock(self, memory_id: str, scope_id: Optional[str]) -> Optional[JsonDict]:
        path = self._memory_path(memory_id, scope_id)
        if not path.exists():
            return None
        return self._load_json_file(path)

    def _save_memory_no_lock(self, memory_id: str, scope_id: Optional[str], memory: JsonDict) -> None:
        self._write_json_atomic(self._memory_path(memory_id, scope_id), memory)

    def _load_proposal_no_lock(self, proposal_id: str, scope_id: Optional[str]) -> Optional[JsonDict]:
        path = self._proposal_path(proposal_id, scope_id)
        if not path.exists():
            return None
        return self._load_json_file(path)

    def _save_proposal_no_lock(self, proposal_id: str, scope_id: Optional[str], proposal: JsonDict) -> None:
        self._write_json_atomic(self._proposal_path(proposal_id, scope_id), proposal)

    def _memory_meta(self, memory: JsonDict) -> JsonDict:
        memory_text = str(memory.get("memory_text", ""))
        return {
            "memory_id": memory.get("memory_id", ""),
            "title": memory.get("title", ""),
            "status": memory.get("status", "active"),
            "category": memory.get("category", "other"),
            "sensitivity": memory.get("sensitivity", "normal"),
            "priority": memory.get("priority", 3),
            "confidence": memory.get("confidence", 0.8),
            "created_at": memory.get("created_at", ""),
            "updated_at": memory.get("updated_at", ""),
            "expires_at": memory.get("expires_at", ""),
            "source_ref": memory.get("source_ref", ""),
            "tags": memory.get("tags", []),
            "char_count": len(memory_text),
            "content_hash": memory.get("metadata", {}).get("content_hash", ""),
        }

    def _proposal_meta(self, proposal: JsonDict) -> JsonDict:
        memory_text = str(proposal.get("memory_text", ""))
        return {
            "proposal_id": proposal.get("proposal_id", ""),
            "operation": proposal.get("operation", "create"),
            "target_memory_id": proposal.get("target_memory_id", ""),
            "result_memory_id": proposal.get("result_memory_id", ""),
            "title": proposal.get("title", ""),
            "status": proposal.get("status", "pending"),
            "category": proposal.get("category", "other"),
            "sensitivity": proposal.get("sensitivity", "normal"),
            "priority": proposal.get("priority", 3),
            "confidence": proposal.get("confidence", 0.8),
            "created_at": proposal.get("created_at", ""),
            "updated_at": proposal.get("updated_at", ""),
            "applied_at": proposal.get("applied_at", ""),
            "rejected_at": proposal.get("rejected_at", ""),
            "expires_at": proposal.get("expires_at", ""),
            "source_ref": proposal.get("source_ref", ""),
            "tags": proposal.get("tags", []),
            "char_count": len(memory_text),
            "content_hash": proposal.get("metadata", {}).get("content_hash", ""),
            "warnings": proposal.get("warnings", []),
            "duplicate_candidates": proposal.get("duplicate_candidates", []),
        }

    def _find_meta_index(self, index: JsonDict, collection: str, id_key: str, item_id: str) -> Optional[int]:
        items = index.get(collection, [])
        if not isinstance(items, list):
            raise ValueError(f"index.json field '{collection}' must be a list.")
        for i, item in enumerate(items):
            if isinstance(item, dict) and item.get(id_key) == item_id:
                return i
        return None

    def _replace_or_append_meta(self, index: JsonDict, collection: str, id_key: str, meta: JsonDict) -> None:
        items = index.setdefault(collection, [])
        if not isinstance(items, list):
            raise ValueError(f"index.json field '{collection}' must be a list.")
        item_id = str(meta.get(id_key) or "")
        existing = self._find_meta_index(index, collection, id_key, item_id)
        if existing is None:
            items.append(meta)
        else:
            items[existing] = meta

    def _remove_meta(self, index: JsonDict, collection: str, id_key: str, item_id: str) -> bool:
        items = index.get(collection, [])
        if not isinstance(items, list):
            raise ValueError(f"index.json field '{collection}' must be a list.")
        before = len(items)
        index[collection] = [
            item
            for item in items
            if not (isinstance(item, dict) and item.get(id_key) == item_id)
        ]
        return len(index[collection]) != before

    def _event(self, event_type: str, scope_id: str, user: Optional[dict] = None, **data: Any) -> JsonDict:
        return {
            "schema_version": "1.0",
            "event_type": event_type,
            "scope_id": scope_id,
            "created_at": self._utc_now(),
            "created_by": self._created_by(user),
            "data": data,
        }

    # ------------------------------------------------------------------
    # Validation, normalization, scoring, and redaction helpers
    # ------------------------------------------------------------------

    def _clamp_int(self, value: int, minimum: int, maximum: int) -> int:
        value = self._field_default(value, minimum)
        try:
            parsed = int(value)
        except Exception:
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _clamp_float(self, value: float, minimum: float, maximum: float) -> float:
        value = self._field_default(value, minimum)
        try:
            parsed = float(value)
        except Exception:
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _clean_title(self, title: str, fallback: str = "Memory") -> str:
        title = self._field_default(title, "")
        value = re.sub(r"\s+", " ", str(title or "").strip())
        if not value:
            value = fallback
        max_len = max(1, int(self.valves.max_title_chars))
        return value[:max_len]

    def _clean_source_ref(self, source_ref: str) -> str:
        source_ref = self._field_default(source_ref, "")
        value = re.sub(r"\s+", " ", str(source_ref or "").strip())
        max_len = max(0, int(self.valves.max_source_ref_chars))
        return value[:max_len] if max_len else ""

    def _clean_rationale(self, rationale: str) -> str:
        rationale = self._field_default(rationale, "")
        value = str(rationale or "").strip()
        max_len = max(0, int(self.valves.max_rationale_chars))
        return value[:max_len] if max_len else ""

    def _clean_category(self, category: str) -> str:
        category = self._field_default(category, "other")
        value = str(category or "other").strip().lower()
        if value not in self.CATEGORIES:
            return "other"
        return value

    def _clean_status_filter(self, status_filter: str, allowed: set[str], default: str = "active") -> str:
        status_filter = self._field_default(status_filter, default)
        value = str(status_filter or default).strip().lower()
        if value == "all":
            return "all"
        if value not in allowed:
            return default
        return value

    def _clean_operation(self, operation: str) -> str:
        operation = self._field_default(operation, "create")
        value = str(operation or "create").strip().lower()
        return value if value in self.OPERATIONS else "create"

    def _clean_sensitivity(self, sensitivity: str) -> str:
        sensitivity = self._field_default(sensitivity, "normal")
        value = str(sensitivity or "normal").strip().lower()
        return value if value in self.SENSITIVITIES else "normal"

    def _clean_expires_at(self, expires_at: str) -> str:
        expires_at = self._field_default(expires_at, "")
        value = str(expires_at or "").strip()
        if not value:
            return ""
        # Keep this permissive and portable: accept common ISO/RFC3339-like text, reject path/control-like junk.
        if not re.fullmatch(r"[0-9TtZz:+\-\. ]{8,40}", value):
            return ""
        return value[:40]

    def _parse_iso_utc(self, value: str) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    def _is_expired(self, expires_at: str) -> bool:
        parsed = self._parse_iso_utc(expires_at)
        if not parsed:
            return False
        return parsed <= datetime.now(timezone.utc)

    def _parse_tags(self, tags: Any) -> list[str]:
        tags = self._field_default(tags, "")
        if tags is None:
            return []

        raw_items: list[str]
        if isinstance(tags, list):
            raw_items = [str(item) for item in tags]
        else:
            text = str(tags or "").strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        raw_items = [str(item) for item in parsed]
                    else:
                        raw_items = []
                except Exception:
                    raw_items = [part.strip() for part in text.split(",")]
            else:
                raw_items = [part.strip() for part in text.split(",")]

        max_tags = self._clamp_int(int(self.valves.max_tags), 0, 100)
        max_chars = self._clamp_int(int(self.valves.max_tag_chars), 1, 200)
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            tag = re.sub(r"\s+", " ", item).strip()[:max_chars]
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tag)
            if len(out) >= max_tags:
                break
        return out

    def _secret_findings(self, text: str) -> list[str]:
        text = self._field_default(text, "")
        findings: list[str] = []
        value = str(text or "")
        if not value:
            return findings

        patterns = [
            (
                "private_key_block",
                r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                re.DOTALL,
            ),
            ("bearer_token", r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", 0),
            (
                "credential_assignment",
                r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|cookie|session[_-]?id)\s*[:=]\s*([\"']?)[^\s\"']{8,}\2",
                0,
            ),
        ]
        for name, pattern, flags in patterns:
            if re.search(pattern, value, flags=flags):
                findings.append(name)

        if bool(self.valves.redact_long_tokens_on_save):
            if re.search(
                r"\b(?=[A-Za-z0-9_-]{64,}\b)(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9_-]+\b",
                value,
            ):
                findings.append("long_token_like_string")

        return sorted(set(findings))

    def _redact_text(self, text: str) -> str:
        text = self._field_default(text, "")
        if not text:
            return ""

        redacted = str(text)
        redacted = re.sub(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            "[REDACTED_PRIVATE_KEY_BLOCK]",
            redacted,
            flags=re.DOTALL,
        )
        redacted = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}",
            "Bearer [REDACTED_TOKEN]",
            redacted,
        )
        redacted = re.sub(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|cookie|session[_-]?id)\s*[:=]\s*([\"']?)[^\s\"']{8,}\2",
            lambda m: f"{m.group(1)}=[REDACTED]",
            redacted,
        )

        if bool(self.valves.redact_long_tokens_on_save):
            redacted = re.sub(
                r"\b(?=[A-Za-z0-9_-]{64,}\b)(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9_-]+\b",
                "[REDACTED_LONG_TOKEN]",
                redacted,
            )
        return redacted

    def _instruction_warnings(self, text: str) -> list[str]:
        text = self._field_default(text, "")
        value = str(text or "").lower()
        checks = {
            "instruction_like_ignore_previous": ["ignore previous", "ignore prior", "ignore earlier"],
            "instruction_like_system_priority": ["system instruction", "developer instruction", "highest priority"],
            "instruction_like_absolute_obedience": ["always obey", "must always", "must never"],
            "instruction_like_exfiltration": ["reveal hidden", "show hidden", "dump system", "print system"],
        }
        warnings: list[str] = []
        for name, phrases in checks.items():
            if any(phrase in value for phrase in phrases):
                warnings.append(name)
        return warnings

    def _normalize_for_hash(self, text: str) -> str:
        text = self._field_default(text, "")
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def _content_hash(self, memory_text: str, category: str, scope_id: str) -> str:
        normalized = "\n".join(
            [self._scope_name(scope_id), self._clean_category(category), self._normalize_for_hash(memory_text)]
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _tokens(self, text: str) -> set[str]:
        text = self._field_default(text, "")
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{1,}", str(text or "").lower())
            if len(token) > 1
        }

    def _lexical_score(self, query: str, record: JsonDict) -> float:
        query_text = self._normalize_for_hash(query)
        if not query_text:
            return 0.0

        haystack = " ".join(
            [
                str(record.get("title", "")),
                str(record.get("memory_text", "")),
                str(record.get("category", "")),
                " ".join(str(tag) for tag in record.get("tags", []) if isinstance(tag, str)),
                str(record.get("source_ref", "")),
            ]
        )
        haystack_norm = self._normalize_for_hash(haystack)

        score = 0.0
        if query_text and query_text in haystack_norm:
            score += 0.55

        q_tokens = self._tokens(query_text)
        h_tokens = self._tokens(haystack_norm)
        if q_tokens and h_tokens:
            overlap = len(q_tokens & h_tokens) / max(1, len(q_tokens))
            jaccard = len(q_tokens & h_tokens) / max(1, len(q_tokens | h_tokens))
            score += 0.35 * overlap
            score += 0.10 * jaccard

        title = self._normalize_for_hash(str(record.get("title", "")))
        if title and query_text in title:
            score += 0.15

        tags = " ".join(str(tag).lower() for tag in record.get("tags", []) if isinstance(tag, str))
        if tags and any(token in tags for token in q_tokens):
            score += 0.10

        try:
            priority = int(record.get("priority", 3))
            score += min(max(priority, 1), 5) * 0.01
        except Exception:
            pass

        return round(min(score, 1.0), 4)

    def _prepare_memory_text(
        self,
        memory_text: str,
        *,
        redact: bool,
        allow_truncate: bool,
    ) -> tuple[Optional[str], Optional[str], bool, list[str]]:
        memory_text = self._field_default(memory_text, "")
        redact = self._as_bool(redact, True)
        allow_truncate = self._as_bool(allow_truncate, False)
        text = str(memory_text or "").strip()
        if not text:
            return None, "memory_text cannot be empty.", False, []

        secret_findings = self._secret_findings(text)
        if secret_findings and bool(self.valves.reject_secret_like_memories):
            return (
                None,
                "memory_text appears to contain credential-like or secret-like content and was not saved.",
                False,
                secret_findings,
            )

        redacted = False
        if bool(self.valves.redact_secrets_on_save) and redact:
            new_text = self._redact_text(text)
            redacted = new_text != text
            text = new_text

        max_chars = int(self.valves.max_memory_chars)
        if max_chars > 0 and len(text) > max_chars:
            if not allow_truncate:
                return (
                    None,
                    f"memory_text is too large: {len(text)} characters. Limit: {max_chars}.",
                    redacted,
                    secret_findings,
                )
            omitted = len(text) - max_chars
            suffix = f"\n\n[MemoryCurator: truncated {omitted} characters during save]"
            text = text[: max(0, max_chars - len(suffix))] + suffix

        return text, None, redacted, secret_findings

    def _clean_and_redact_aux_text(self, text: str, cleaner: str, redact: bool) -> tuple[str, bool]:
        if cleaner == "title":
            value = self._clean_title(text)
        elif cleaner == "source_ref":
            value = self._clean_source_ref(text)
        elif cleaner == "rationale":
            value = self._clean_rationale(text)
        else:
            value = str(text or "").strip()

        redacted = False
        if bool(self.valves.redact_secrets_on_save) and redact:
            new_value = self._redact_text(value)
            redacted = new_value != value
            value = new_value
        return value, redacted

    def _memory_matches_filters(
        self,
        memory: JsonDict,
        *,
        category_filter: str,
        status_filter: str,
        tag_filter: list[str],
        sensitivity_filter: str,
        include_expired: bool,
    ) -> bool:
        if status_filter != "all" and memory.get("status") != status_filter:
            return False
        if category_filter != "all" and memory.get("category") != category_filter:
            return False
        if sensitivity_filter != "all" and memory.get("sensitivity") != sensitivity_filter:
            return False
        if tag_filter:
            existing = {str(tag).lower() for tag in memory.get("tags", []) if isinstance(tag, str)}
            wanted = {tag.lower() for tag in tag_filter}
            if not wanted.issubset(existing):
                return False
        if not include_expired and self._is_expired(str(memory.get("expires_at", ""))):
            return False
        return True

    def _duplicate_candidates_no_lock(
        self,
        scope_id: str,
        memory_text: str,
        category: str,
        *,
        limit: Optional[int] = None,
    ) -> tuple[list[JsonDict], list[JsonDict]]:
        limit_count = self._clamp_int(
            limit if limit is not None else int(self.valves.max_duplicate_candidates),
            0,
            50,
        )
        index = self._load_index_no_lock(scope_id)
        target_hash = self._content_hash(memory_text, category, scope_id)
        exact: list[JsonDict] = []
        near: list[JsonDict] = []

        for meta in index.get("memories", []):
            if not isinstance(meta, dict):
                continue
            if meta.get("status") != "active":
                continue
            memory_id = str(meta.get("memory_id") or "")
            if not memory_id:
                continue

            if meta.get("content_hash") == target_hash:
                exact.append(meta)
                continue

            memory = self._load_memory_no_lock(memory_id, scope_id)
            if not memory:
                continue
            if memory.get("category") != category:
                continue
            score = self._lexical_score(memory_text, memory)
            if score >= float(self.valves.near_duplicate_threshold):
                candidate = self._memory_meta(memory)
                candidate["score"] = score
                near.append(candidate)

        near.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if limit_count > 0:
            exact = exact[:limit_count]
            near = near[:limit_count]
        else:
            exact = []
            near = []
        return exact, near

    # ------------------------------------------------------------------
    # Public Open WebUI tool methods
    # ------------------------------------------------------------------

    async def initialize_scope(
        self,
        scope_id: str = Field(
            default="",
            description="Scope folder to initialize. Empty uses the default scope.",
        ),
    ) -> str:
        """Create the local JSON store for a memory scope."""
        try:
            scope_name = self._scope_name(scope_id)
            index_path = self._index_path(scope_name)
            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                self._memories_dir(scope_name).mkdir(parents=True, exist_ok=True)
                self._proposals_dir(scope_name).mkdir(parents=True, exist_ok=True)
                self._events_path(scope_name).parent.mkdir(parents=True, exist_ok=True)
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "index_path": str(index_path),
                    "memories_dir": str(self._memories_dir(scope_name)),
                    "proposals_dir": str(self._proposals_dir(scope_name)),
                    "events_path": str(self._events_path(scope_name)),
                }
            )
        except Exception as e:
            return self._err("initialize_failed", str(e))

    async def propose_memory(
        self,
        memory_text: str = Field(
            ...,
            description="Durable memory text to propose. Do not include credentials, raw secrets, or large logs.",
        ),
        category: str = Field(
            default="project_fact",
            description="One of: user_preference, project_fact, environment, decision, constraint, workflow, relationship, writing_preference, technical_note, other.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Use a project/chat/task ID when available. Empty uses the default scope.",
        ),
        title: str = Field(
            default="",
            description="Short human-readable title. Empty derives a generic title.",
        ),
        rationale: str = Field(
            default="",
            description="Why this deserves durable memory. Stored with the proposal, not the active memory.",
        ),
        source_ref: str = Field(
            default="",
            description="Optional source reference, such as chat title, message range, file name, or task name.",
        ),
        tags: str = Field(
            default="",
            description="Optional comma-separated tags, or a JSON list string.",
        ),
        confidence: float = Field(
            default=0.8,
            description="Confidence from 0.0 to 1.0. This is metadata, not proof.",
        ),
        priority: int = Field(
            default=3,
            description="Recall priority from 1 to 5. Higher means more important for future recall.",
        ),
        sensitivity: str = Field(
            default="normal",
            description="One of: normal, sensitive. Secrets should not be stored at all.",
        ),
        expires_at: str = Field(
            default="",
            description="Optional ISO-like expiration timestamp. Empty means no expiry.",
        ),
        target_memory_id: str = Field(
            default="",
            description="For update proposals only: existing memory ID to update.",
        ),
        operation: str = Field(
            default="create",
            description="One of: create, update.",
        ),
        redact: bool = Field(
            default=True,
            description="If true, apply local secret redaction before saving non-rejected content.",
        ),
        allow_truncate: bool = Field(
            default=False,
            description="If true, truncate memory_text larger than max_memory_chars instead of rejecting it.",
        ),
        allow_duplicate: bool = Field(
            default=False,
            description="If false, exact active duplicate memories are rejected during proposal creation.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Create a pending memory proposal. This does not create active memory.
        """
        try:
            scope_name = self._scope_name(scope_id)
            redact = self._as_bool(redact, True)
            allow_truncate = self._as_bool(allow_truncate, False)
            allow_duplicate = self._as_bool(allow_duplicate, False)
            op = self._clean_operation(operation)
            category_value = self._clean_category(category)
            sensitivity_value = self._clean_sensitivity(sensitivity)
            confidence_value = round(self._clamp_float(confidence, 0.0, 1.0), 3)
            priority_value = self._clamp_int(priority, 1, 5)
            expires_value = self._clean_expires_at(expires_at)

            prepared, error, redacted_text, secret_findings = self._prepare_memory_text(
                memory_text,
                redact=redact,
                allow_truncate=allow_truncate,
            )
            if error:
                return self._err(
                    "invalid_memory_text",
                    error,
                    secret_findings=secret_findings,
                )
            assert prepared is not None

            title_value = self._field_default(title, "")
            cleaned_title, redacted_title = self._clean_and_redact_aux_text(
                title_value or category_value.replace("_", " ").title(), "title", redact
            )
            cleaned_source_ref, redacted_source = self._clean_and_redact_aux_text(
                source_ref, "source_ref", redact
            )
            cleaned_rationale, redacted_rationale = self._clean_and_redact_aux_text(
                rationale, "rationale", redact
            )
            parsed_tags = self._parse_tags(tags)

            safe_target = ""
            if op == "update":
                safe_target = self._validate_id(target_memory_id, "target_memory_id")

            now = self._utc_now()
            proposal_id = self._new_id("prop")
            content_hash = self._content_hash(prepared, category_value, scope_name)
            warnings = self._instruction_warnings(prepared)
            if self._is_expired(expires_value):
                warnings.append("expires_at_is_already_past")
            if sensitivity_value == "sensitive":
                warnings.append("sensitive_memory_should_not_be_injected_by_default")
            if redacted_text or redacted_title or redacted_source or redacted_rationale:
                warnings.append("redacted_on_save")

            index_path = self._index_path(scope_name)
            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)

                if op == "update":
                    existing = self._load_memory_no_lock(safe_target, scope_name)
                    if not existing:
                        return self._err(
                            "target_not_found",
                            f"No memory found for target_memory_id {safe_target}.",
                            scope_id=scope_name,
                            target_memory_id=safe_target,
                        )

                exact_duplicates, near_duplicates = self._duplicate_candidates_no_lock(
                    scope_name,
                    prepared,
                    category_value,
                )
                if op == "update" and safe_target:
                    exact_duplicates = [
                        item for item in exact_duplicates if item.get("memory_id") != safe_target
                    ]
                    near_duplicates = [
                        item for item in near_duplicates if item.get("memory_id") != safe_target
                    ]

                if exact_duplicates and not allow_duplicate:
                    return self._err(
                        "duplicate_memory",
                        "An active memory with the same normalized text, category, and scope already exists.",
                        scope_id=scope_name,
                        exact_duplicates=exact_duplicates,
                        near_duplicates=near_duplicates,
                    )

                proposal: JsonDict = {
                    "schema_version": "1.0",
                    "proposal_id": proposal_id,
                    "scope_id": scope_name,
                    "operation": op,
                    "target_memory_id": safe_target,
                    "result_memory_id": "",
                    "status": "pending",
                    "title": cleaned_title,
                    "memory_text": prepared,
                    "category": category_value,
                    "rationale": cleaned_rationale,
                    "source_ref": cleaned_source_ref,
                    "tags": parsed_tags,
                    "confidence": confidence_value,
                    "priority": priority_value,
                    "sensitivity": sensitivity_value,
                    "expires_at": expires_value,
                    "created_at": now,
                    "updated_at": now,
                    "applied_at": "",
                    "rejected_at": "",
                    "rejection_reason": "",
                    "warnings": sorted(set(warnings)),
                    "duplicate_candidates": near_duplicates,
                    "metadata": {
                        "tool_version": self.TOOL_VERSION,
                        "created_by": self._created_by(__user__),
                        "char_count": len(prepared),
                        "content_hash": content_hash,
                        "redacted_on_save": bool(
                            redacted_text or redacted_title or redacted_source or redacted_rationale
                        ),
                        "secret_findings": secret_findings,
                    },
                }

                self._save_proposal_no_lock(proposal_id, scope_name, proposal)
                self._replace_or_append_meta(
                    index, "proposals", "proposal_id", self._proposal_meta(proposal)
                )
                self._append_event_no_lock(
                    scope_name,
                    self._event(
                        "proposal_created",
                        scope_name,
                        __user__,
                        proposal_id=proposal_id,
                        operation=op,
                        target_memory_id=safe_target,
                    ),
                )
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "proposal": proposal,
                    "requires_apply": True,
                    "apply_instruction": 'Call apply_memory_proposal with confirm="APPLY" to create or update active memory.',
                }
            )
        except Exception as e:
            return self._err("proposal_failed", str(e))

    async def propose_memory_update(
        self,
        memory_id: str = Field(
            ...,
            description="Existing memory ID to update through a pending proposal.",
        ),
        new_memory_text: str = Field(
            ...,
            description="Replacement durable memory text. The active memory is not changed until apply_memory_proposal is called.",
        ),
        change_note: str = Field(
            default="",
            description="Why the memory should be changed.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        title: str = Field(
            default="",
            description="Optional replacement title. Empty preserves the existing title.",
        ),
        category: str = Field(
            default="",
            description="Optional replacement category. Empty preserves the existing category.",
        ),
        source_ref: str = Field(
            default="",
            description="Optional source reference for the update.",
        ),
        tags: str = Field(
            default="",
            description="Optional replacement tags. Empty preserves the existing tags.",
        ),
        confidence: float = Field(
            default=-1.0,
            description="Optional replacement confidence from 0.0 to 1.0. Negative preserves existing confidence.",
        ),
        priority: int = Field(
            default=0,
            description="Optional replacement priority from 1 to 5. 0 preserves existing priority.",
        ),
        sensitivity: str = Field(
            default="",
            description="Optional replacement sensitivity: normal or sensitive. Empty preserves existing sensitivity.",
        ),
        expires_at: str = Field(
            default="",
            description="Optional replacement expiry. Empty preserves existing expiry.",
        ),
        redact: bool = Field(
            default=True,
            description="If true, apply local secret redaction before saving non-rejected content.",
        ),
        allow_truncate: bool = Field(
            default=False,
            description="If true, truncate new_memory_text larger than max_memory_chars instead of rejecting it.",
        ),
        allow_duplicate: bool = Field(
            default=False,
            description="If false, exact active duplicate memories are rejected during proposal creation.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """Create a pending update proposal for an existing memory."""
        try:
            scope_name = self._scope_name(scope_id)
            redact = self._as_bool(redact, True)
            allow_truncate = self._as_bool(allow_truncate, False)
            allow_duplicate = self._as_bool(allow_duplicate, False)
            confidence = self._as_float(confidence, -1.0)
            priority = self._as_int(priority, 0)
            safe_id = self._validate_id(memory_id, "memory_id")
            existing = self._load_memory_no_lock(safe_id, scope_name)
            if not existing:
                return self._err(
                    "not_found",
                    f"No memory found for {safe_id}.",
                    scope_id=scope_name,
                    memory_id=safe_id,
                )

            existing_tags = existing.get("tags", [])
            tags_raw = self._field_default(tags, "")
            category_raw = self._field_default(category, "")
            title_raw = self._field_default(title, "")
            source_ref_raw = self._field_default(source_ref, "")
            sensitivity_raw = self._field_default(sensitivity, "")
            expires_raw = self._field_default(expires_at, "")

            tag_value = tags_raw if str(tags_raw or "").strip() else json.dumps(existing_tags, ensure_ascii=False)
            category_value = category_raw or str(existing.get("category") or "other")
            title_value = title_raw or str(existing.get("title") or "Memory")
            source_ref_value = source_ref_raw or str(existing.get("source_ref") or "")
            confidence_value = float(existing.get("confidence", 0.8)) if confidence < 0 else confidence
            priority_value = int(existing.get("priority", 3)) if priority <= 0 else priority
            sensitivity_value = sensitivity_raw or str(existing.get("sensitivity") or "normal")
            expires_value = expires_raw if str(expires_raw or "").strip() else str(existing.get("expires_at") or "")

            return await self.propose_memory(
                memory_text=new_memory_text,
                category=category_value,
                scope_id=scope_name,
                title=title_value,
                rationale=change_note,
                source_ref=source_ref_value,
                tags=tag_value,
                confidence=confidence_value,
                priority=priority_value,
                sensitivity=sensitivity_value,
                expires_at=expires_value,
                target_memory_id=safe_id,
                operation="update",
                redact=redact,
                allow_truncate=allow_truncate,
                allow_duplicate=allow_duplicate,
                __user__=__user__,
            )
        except Exception as e:
            return self._err("proposal_update_failed", str(e), memory_id=memory_id)

    async def list_memory_proposals(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        status_filter: str = Field(
            default="pending",
            description="One of: all, pending, applied, rejected.",
        ),
        limit: int = Field(
            default=20,
            description="Maximum number of proposal metadata entries to return.",
        ),
    ) -> str:
        """List proposal metadata for a scope without returning full memory_text bodies."""
        try:
            scope_name = self._scope_name(scope_id)
            limit_value = self._clamp_int(limit, 1, 500)
            status = self._clean_status_filter(status_filter, self.PROPOSAL_STATUSES, "pending")

            index = self._load_index_no_lock(scope_name)
            proposals = [item for item in index.get("proposals", []) if isinstance(item, dict)]
            if status != "all":
                proposals = [item for item in proposals if item.get("status") == status]

            proposals.sort(
                key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                reverse=True,
            )
            return self._ok(
                {
                    "scope_id": scope_name,
                    "proposals": proposals[:limit_value],
                    "returned_count": min(len(proposals), limit_value),
                    "matched_count": len(proposals),
                }
            )
        except Exception as e:
            return self._err("list_proposals_failed", str(e))

    async def get_memory_proposal(
        self,
        proposal_id: str = Field(
            ...,
            description="Proposal ID returned by propose_memory or list_memory_proposals.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Return one full memory proposal."""
        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_id(proposal_id, "proposal_id")
            proposal = self._load_proposal_no_lock(safe_id, scope_name)
            if not proposal:
                return self._err(
                    "not_found",
                    f"No memory proposal found for {safe_id}.",
                    scope_id=scope_name,
                    proposal_id=safe_id,
                )
            return self._ok({"scope_id": scope_name, "proposal": proposal})
        except Exception as e:
            return self._err("read_proposal_failed", str(e), proposal_id=proposal_id)

    async def apply_memory_proposal(
        self,
        proposal_id: str = Field(
            ...,
            description="Pending proposal ID to apply.",
        ),
        confirm: str = Field(
            ...,
            description='Must be exactly "APPLY" to create or update active memory.',
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """Apply a pending proposal, creating or updating an active memory."""
        if confirm != "APPLY":
            return self._err(
                "confirmation_required",
                'Apply aborted. The confirm value must be exactly "APPLY".',
            )

        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_id(proposal_id, "proposal_id")
            now = self._utc_now()
            index_path = self._index_path(scope_name)

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                proposal = self._load_proposal_no_lock(safe_id, scope_name)
                if not proposal:
                    return self._err(
                        "not_found",
                        f"No memory proposal found for {safe_id}.",
                        scope_id=scope_name,
                        proposal_id=safe_id,
                    )
                if proposal.get("status") != "pending":
                    return self._err(
                        "proposal_not_pending",
                        f"Proposal {safe_id} has status {proposal.get('status')!r} and cannot be applied.",
                        scope_id=scope_name,
                        proposal_id=safe_id,
                        status=proposal.get("status"),
                    )

                op = self._clean_operation(str(proposal.get("operation") or "create"))
                if op == "create":
                    memory_id = self._new_id("mem")
                    created_at = now
                    previous_memory: Optional[JsonDict] = None
                else:
                    memory_id = self._validate_id(
                        str(proposal.get("target_memory_id") or ""), "target_memory_id"
                    )
                    previous_memory = self._load_memory_no_lock(memory_id, scope_name)
                    if not previous_memory:
                        return self._err(
                            "target_not_found",
                            f"No memory found for target_memory_id {memory_id}.",
                            scope_id=scope_name,
                            target_memory_id=memory_id,
                        )
                    created_at = str(previous_memory.get("created_at") or now)

                memory_text = str(proposal.get("memory_text") or "").strip()
                category_value = self._clean_category(str(proposal.get("category") or "other"))
                content_hash = self._content_hash(memory_text, category_value, scope_name)
                memory: JsonDict = {
                    "schema_version": "1.0",
                    "memory_id": memory_id,
                    "scope_id": scope_name,
                    "status": "active",
                    "title": self._clean_title(str(proposal.get("title") or "Memory")),
                    "memory_text": memory_text,
                    "category": category_value,
                    "source_proposal_id": safe_id,
                    "source_ref": self._clean_source_ref(str(proposal.get("source_ref") or "")),
                    "tags": self._parse_tags(proposal.get("tags", [])),
                    "confidence": self._clamp_float(float(proposal.get("confidence", 0.8)), 0.0, 1.0),
                    "priority": self._clamp_int(int(proposal.get("priority", 3)), 1, 5),
                    "sensitivity": self._clean_sensitivity(str(proposal.get("sensitivity") or "normal")),
                    "expires_at": self._clean_expires_at(str(proposal.get("expires_at") or "")),
                    "created_at": created_at,
                    "updated_at": now,
                    "metadata": {
                        "tool_version": self.TOOL_VERSION,
                        "created_by": self._created_by(__user__),
                        "updated_by": self._created_by(__user__),
                        "char_count": len(memory_text),
                        "content_hash": content_hash,
                        "previous_content_hash": ""
                        if not previous_memory
                        else previous_memory.get("metadata", {}).get("content_hash", ""),
                        "applied_from_operation": op,
                    },
                }

                self._save_memory_no_lock(memory_id, scope_name, memory)
                self._replace_or_append_meta(index, "memories", "memory_id", self._memory_meta(memory))

                proposal["status"] = "applied"
                proposal["result_memory_id"] = memory_id
                proposal["applied_at"] = now
                proposal["updated_at"] = now
                self._save_proposal_no_lock(safe_id, scope_name, proposal)
                self._replace_or_append_meta(
                    index, "proposals", "proposal_id", self._proposal_meta(proposal)
                )
                self._append_event_no_lock(
                    scope_name,
                    self._event(
                        "proposal_applied",
                        scope_name,
                        __user__,
                        proposal_id=safe_id,
                        operation=op,
                        memory_id=memory_id,
                    ),
                )
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "proposal_id": safe_id,
                    "operation": op,
                    "memory": memory,
                }
            )
        except Exception as e:
            return self._err("apply_failed", str(e), proposal_id=proposal_id)

    async def reject_memory_proposal(
        self,
        proposal_id: str = Field(
            ...,
            description="Pending proposal ID to reject.",
        ),
        reason: str = Field(
            default="",
            description="Optional reason for rejection.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """Reject a pending proposal without deleting it."""
        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_id(proposal_id, "proposal_id")
            now = self._utc_now()
            index_path = self._index_path(scope_name)

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                proposal = self._load_proposal_no_lock(safe_id, scope_name)
                if not proposal:
                    return self._err(
                        "not_found",
                        f"No memory proposal found for {safe_id}.",
                        scope_id=scope_name,
                        proposal_id=safe_id,
                    )
                if proposal.get("status") != "pending":
                    return self._err(
                        "proposal_not_pending",
                        f"Proposal {safe_id} has status {proposal.get('status')!r} and cannot be rejected.",
                        scope_id=scope_name,
                        proposal_id=safe_id,
                        status=proposal.get("status"),
                    )

                proposal["status"] = "rejected"
                proposal["rejected_at"] = now
                proposal["updated_at"] = now
                proposal["rejection_reason"] = self._clean_rationale(reason)
                self._save_proposal_no_lock(safe_id, scope_name, proposal)
                self._replace_or_append_meta(index, "proposals", "proposal_id", self._proposal_meta(proposal))
                self._append_event_no_lock(
                    scope_name,
                    self._event(
                        "proposal_rejected",
                        scope_name,
                        __user__,
                        proposal_id=safe_id,
                        reason=proposal["rejection_reason"],
                    ),
                )
                self._save_index_no_lock(scope_name, index)

            return self._ok({"scope_id": scope_name, "proposal": proposal})
        except Exception as e:
            return self._err("reject_failed", str(e), proposal_id=proposal_id)

    async def list_memories(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        category_filter: str = Field(
            default="all",
            description="One of the memory categories, or all.",
        ),
        status_filter: str = Field(
            default="active",
            description="One of: all, active, archived.",
        ),
        sensitivity_filter: str = Field(
            default="all",
            description="One of: all, normal, sensitive.",
        ),
        tags: str = Field(
            default="",
            description="Optional comma-separated tags. All listed tags must be present.",
        ),
        include_expired: bool = Field(
            default=False,
            description="If false, omit memories whose expires_at is in the past.",
        ),
        limit: int = Field(
            default=50,
            description="Maximum number of memory metadata entries to return.",
        ),
    ) -> str:
        """List memory metadata for a scope without returning full memory_text bodies."""
        try:
            scope_name = self._scope_name(scope_id)
            include_expired = self._as_bool(include_expired, False)
            limit_value = self._clamp_int(limit, 1, 1000)
            category_raw = str(self._field_default(category_filter, "all")).strip().lower()
            sensitivity_raw = str(self._field_default(sensitivity_filter, "all")).strip().lower()
            category = "all" if category_raw == "all" else self._clean_category(category_raw)
            status = self._clean_status_filter(status_filter, self.MEMORY_STATUSES, "active")
            sensitivity = "all" if sensitivity_raw == "all" else self._clean_sensitivity(sensitivity_raw)
            tag_filter = self._parse_tags(tags)

            index = self._load_index_no_lock(scope_name)
            metas = [item for item in index.get("memories", []) if isinstance(item, dict)]
            filtered: list[JsonDict] = []
            for meta in metas:
                if status != "all" and meta.get("status") != status:
                    continue
                if category != "all" and meta.get("category") != category:
                    continue
                if sensitivity != "all" and meta.get("sensitivity") != sensitivity:
                    continue
                if tag_filter:
                    existing = {str(tag).lower() for tag in meta.get("tags", []) if isinstance(tag, str)}
                    wanted = {tag.lower() for tag in tag_filter}
                    if not wanted.issubset(existing):
                        continue
                if not include_expired and self._is_expired(str(meta.get("expires_at", ""))):
                    continue
                filtered.append(meta)

            filtered.sort(
                key=lambda item: (int(item.get("priority", 3)), str(item.get("updated_at") or item.get("created_at") or "")),
                reverse=True,
            )
            return self._ok(
                {
                    "scope_id": scope_name,
                    "memories": filtered[:limit_value],
                    "returned_count": min(len(filtered), limit_value),
                    "matched_count": len(filtered),
                }
            )
        except Exception as e:
            return self._err("list_memories_failed", str(e))

    async def get_memory(
        self,
        memory_id: str = Field(
            ...,
            description="Memory ID returned by apply_memory_proposal, list_memories, or search_memories.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Return one full memory record."""
        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_id(memory_id, "memory_id")
            memory = self._load_memory_no_lock(safe_id, scope_name)
            if not memory:
                return self._err(
                    "not_found",
                    f"No memory found for {safe_id}.",
                    scope_id=scope_name,
                    memory_id=safe_id,
                )
            return self._ok({"scope_id": scope_name, "memory": memory})
        except Exception as e:
            return self._err("read_memory_failed", str(e), memory_id=memory_id)

    async def search_memories(
        self,
        query: str = Field(
            ...,
            description="Search query for local lexical matching over titles, memory text, tags, category, and source_ref.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        category_filter: str = Field(
            default="all",
            description="One of the memory categories, or all.",
        ),
        status_filter: str = Field(
            default="active",
            description="One of: all, active, archived.",
        ),
        sensitivity_filter: str = Field(
            default="all",
            description="One of: all, normal, sensitive.",
        ),
        tags: str = Field(
            default="",
            description="Optional comma-separated tags. All listed tags must be present.",
        ),
        include_expired: bool = Field(
            default=False,
            description="If false, omit memories whose expires_at is in the past.",
        ),
        limit: int = Field(
            default=10,
            description="Maximum number of full memory records to return.",
        ),
    ) -> str:
        """Search approved memories using bounded dependency-free lexical scoring."""
        try:
            scope_name = self._scope_name(scope_id)
            include_expired = self._as_bool(include_expired, False)
            q = str(self._field_default(query, "") or "").strip()
            if not q:
                return self._err("invalid_query", "query cannot be empty.")

            limit_value = self._clamp_int(limit, 1, 100)
            category_raw = str(self._field_default(category_filter, "all")).strip().lower()
            sensitivity_raw = str(self._field_default(sensitivity_filter, "all")).strip().lower()
            category = "all" if category_raw == "all" else self._clean_category(category_raw)
            status = self._clean_status_filter(status_filter, self.MEMORY_STATUSES, "active")
            sensitivity = "all" if sensitivity_raw == "all" else self._clean_sensitivity(sensitivity_raw)
            tag_filter = self._parse_tags(tags)

            index = self._load_index_no_lock(scope_name)
            results: list[JsonDict] = []
            for meta in index.get("memories", []):
                if not isinstance(meta, dict):
                    continue
                memory_id = str(meta.get("memory_id") or "")
                if not memory_id:
                    continue
                memory = self._load_memory_no_lock(memory_id, scope_name)
                if not memory:
                    continue
                if not self._memory_matches_filters(
                    memory,
                    category_filter=category,
                    status_filter=status,
                    tag_filter=tag_filter,
                    sensitivity_filter=sensitivity,
                    include_expired=include_expired,
                ):
                    continue
                score = self._lexical_score(q, memory)
                if score <= 0:
                    continue
                item = dict(memory)
                item["score"] = score
                results.append(item)

            results.sort(
                key=lambda item: (
                    float(item.get("score", 0.0)),
                    int(item.get("priority", 3)),
                    str(item.get("updated_at") or item.get("created_at") or ""),
                ),
                reverse=True,
            )
            return self._ok(
                {
                    "scope_id": scope_name,
                    "query": q,
                    "results": results[:limit_value],
                    "returned_count": min(len(results), limit_value),
                    "matched_count": len(results),
                    "search_method": "local_lexical",
                }
            )
        except Exception as e:
            return self._err("search_failed", str(e))

    async def archive_memory(
        self,
        memory_id: str = Field(
            ...,
            description="Memory ID to archive.",
        ),
        confirm: str = Field(
            ...,
            description='Must be exactly "ARCHIVE" to archive the memory.',
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        reason: str = Field(
            default="",
            description="Optional reason for archival.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """Soft-remove a memory from normal active recall without deleting it."""
        if confirm != "ARCHIVE":
            return self._err(
                "confirmation_required",
                'Archive aborted. The confirm value must be exactly "ARCHIVE".',
            )

        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_id(memory_id, "memory_id")
            now = self._utc_now()
            index_path = self._index_path(scope_name)

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                memory = self._load_memory_no_lock(safe_id, scope_name)
                if not memory:
                    return self._err(
                        "not_found",
                        f"No memory found for {safe_id}.",
                        scope_id=scope_name,
                        memory_id=safe_id,
                    )
                memory["status"] = "archived"
                memory["updated_at"] = now
                memory.setdefault("metadata", {})["archived_by"] = self._created_by(__user__)
                memory["metadata"]["archived_at"] = now
                memory["metadata"]["archive_reason"] = self._clean_rationale(reason)
                self._save_memory_no_lock(safe_id, scope_name, memory)
                self._replace_or_append_meta(index, "memories", "memory_id", self._memory_meta(memory))
                self._append_event_no_lock(
                    scope_name,
                    self._event(
                        "memory_archived",
                        scope_name,
                        __user__,
                        memory_id=safe_id,
                        reason=memory["metadata"].get("archive_reason", ""),
                    ),
                )
                self._save_index_no_lock(scope_name, index)

            return self._ok({"scope_id": scope_name, "memory": memory})
        except Exception as e:
            return self._err("archive_failed", str(e), memory_id=memory_id)

    async def delete_memory(
        self,
        memory_id: str = Field(
            ...,
            description="Memory ID to hard-delete.",
        ),
        confirm: str = Field(
            ...,
            description='Must be exactly "DELETE" to hard-delete the memory.',
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """Hard-delete one memory after explicit confirmation."""
        if confirm != "DELETE":
            return self._err(
                "confirmation_required",
                'Delete aborted. The confirm value must be exactly "DELETE".',
            )

        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_id(memory_id, "memory_id")
            index_path = self._index_path(scope_name)
            deleted = False

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                removed_from_index = self._remove_meta(index, "memories", "memory_id", safe_id)
                path = self._memory_path(safe_id, scope_name)
                if path.exists():
                    path.unlink()
                    deleted = True
                deleted = deleted or removed_from_index

                self._append_event_no_lock(
                    scope_name,
                    self._event(
                        "memory_deleted",
                        scope_name,
                        __user__,
                        memory_id=safe_id,
                        deleted=deleted,
                    ),
                )
                self._save_index_no_lock(scope_name, index)

            if not deleted:
                return self._err(
                    "not_found",
                    f"No memory found for {safe_id}.",
                    scope_id=scope_name,
                    memory_id=safe_id,
                )

            return self._ok(
                {
                    "scope_id": scope_name,
                    "memory_id": safe_id,
                    "deleted": True,
                }
            )
        except Exception as e:
            return self._err("delete_failed", str(e), memory_id=memory_id)

    async def prune_memory_proposals(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        keep_recent: int = Field(
            default=100,
            description="Number of newest proposals to preserve regardless of status.",
        ),
        delete_rejected: bool = Field(
            default=True,
            description="Delete rejected proposals outside the keep_recent window.",
        ),
        delete_applied: bool = Field(
            default=False,
            description="Delete applied proposals outside the keep_recent window.",
        ),
        confirm: str = Field(
            default="",
            description='Must be exactly "PRUNE" to prune proposals.',
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """Prune old applied/rejected proposal files after explicit confirmation."""
        if confirm != "PRUNE":
            return self._err(
                "confirmation_required",
                'Prune aborted. The confirm value must be exactly "PRUNE".',
            )

        try:
            scope_name = self._scope_name(scope_id)
            delete_rejected = self._as_bool(delete_rejected, True)
            delete_applied = self._as_bool(delete_applied, False)
            keep_count = self._clamp_int(keep_recent, 0, 5000)
            index_path = self._index_path(scope_name)
            removed: list[str] = []
            preserved: list[str] = []

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                metas = [item for item in index.get("proposals", []) if isinstance(item, dict)]
                metas.sort(
                    key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                    reverse=True,
                )
                keep_ids = {
                    str(item.get("proposal_id") or "")
                    for item in metas[:keep_count]
                    if str(item.get("proposal_id") or "")
                }

                new_metas: list[JsonDict] = []
                for item in metas:
                    proposal_id = str(item.get("proposal_id") or "")
                    status = str(item.get("status") or "pending")
                    should_delete = False
                    if proposal_id and proposal_id not in keep_ids:
                        if status == "rejected" and delete_rejected:
                            should_delete = True
                        elif status == "applied" and delete_applied:
                            should_delete = True

                    if should_delete:
                        try:
                            self._proposal_path(proposal_id, scope_name).unlink(missing_ok=True)
                        except OSError:
                            pass
                        removed.append(proposal_id)
                    else:
                        if proposal_id:
                            preserved.append(proposal_id)
                        new_metas.append(item)

                index["proposals"] = new_metas
                self._append_event_no_lock(
                    scope_name,
                    self._event(
                        "proposals_pruned",
                        scope_name,
                        __user__,
                        removed_count=len(removed),
                        delete_rejected=delete_rejected,
                        delete_applied=delete_applied,
                        keep_recent=keep_count,
                    ),
                )
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "removed_proposal_ids": removed,
                    "removed_count": len(removed),
                    "preserved_count": len(preserved),
                }
            )
        except Exception as e:
            return self._err("prune_failed", str(e))

    async def export_scope(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        include_archived: bool = Field(
            default=True,
            description="Include archived memories in export.",
        ),
        include_proposals: bool = Field(
            default=True,
            description="Include proposal records in export.",
        ),
        include_events: bool = Field(
            default=False,
            description="Include events.jsonl entries in export. This can be noisy.",
        ),
    ) -> str:
        """Export a scope as portable JSON for backup or migration."""
        try:
            scope_name = self._scope_name(scope_id)
            include_archived = self._as_bool(include_archived, True)
            include_proposals = self._as_bool(include_proposals, True)
            include_events = self._as_bool(include_events, False)
            index = self._load_index_no_lock(scope_name)
            memories: list[JsonDict] = []
            proposals: list[JsonDict] = []
            events: list[JsonDict] = []

            for meta in index.get("memories", []):
                if not isinstance(meta, dict):
                    continue
                memory_id = str(meta.get("memory_id") or "")
                if not memory_id:
                    continue
                memory = self._load_memory_no_lock(memory_id, scope_name)
                if not memory:
                    continue
                if memory.get("status") == "archived" and not include_archived:
                    continue
                memories.append(memory)

            if include_proposals:
                for meta in index.get("proposals", []):
                    if not isinstance(meta, dict):
                        continue
                    proposal_id = str(meta.get("proposal_id") or "")
                    if not proposal_id:
                        continue
                    proposal = self._load_proposal_no_lock(proposal_id, scope_name)
                    if proposal:
                        proposals.append(proposal)

            if include_events:
                path = self._events_path(scope_name)
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                if isinstance(item, dict):
                                    events.append(item)
                            except Exception:
                                continue

            return self._ok(
                {
                    "schema_version": "1.0",
                    "tool": "memory_curator",
                    "tool_version": self.TOOL_VERSION,
                    "scope_id": scope_name,
                    "exported_at": self._utc_now(),
                    "index": index,
                    "memories": memories,
                    "proposals": proposals,
                    "events": events if include_events else [],
                    "counts": {
                        "memories": len(memories),
                        "proposals": len(proposals),
                        "events": len(events),
                    },
                }
            )
        except Exception as e:
            return self._err("export_failed", str(e))
