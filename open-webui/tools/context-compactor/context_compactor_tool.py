"""
title: Context Compactor
author: Local
version: 0.1.0
required_open_webui_version: 0.6.0
description: Local JSON context snapshot manager for saving, retrieving, activating, deactivating, deleting, and pruning compacted conversation state. Designed to pair with a read-only injection filter.
"""

from __future__ import annotations

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
    Open WebUI tool for local, scoped context-compaction snapshots.

    Design intent:
    - The tool stores compacted context records; it does not call a model.
    - The model or user supplies the summary text; this tool validates and persists it.
    - Reads do not mutate state.
    - Activating a snapshot is explicit.
    - State remains local JSON for portability and vendor independence.
    - A future Context Compactor filter can read only the active snapshot and inject it.
    """

    TOOL_VERSION = "0.1.0"
    INDEX_FILE = "index.json"
    SNAPSHOT_DIR = "snapshots"

    class Valves(BaseModel):
        data_dir: str = Field(
            default="/app/backend/data/context_compactor",
            description="Base directory containing per-scope context snapshot stores.",
        )
        default_scope: str = Field(
            default="default",
            description="Default scope if no chat or explicit scope is provided.",
        )
        max_summary_chars: int = Field(
            default=60_000,
            description="Maximum summary_markdown characters accepted per snapshot.",
        )
        max_title_chars: int = Field(
            default=160,
            description="Maximum title length retained for each snapshot.",
        )
        max_source_ref_chars: int = Field(
            default=500,
            description="Maximum source_ref length retained for each snapshot.",
        )
        max_tags: int = Field(
            default=20,
            description="Maximum number of tags retained per snapshot.",
        )
        max_tag_chars: int = Field(
            default=50,
            description="Maximum characters retained per tag.",
        )
        redact_secrets_on_save: bool = Field(
            default=True,
            description="Redact common secret patterns before saving snapshots.",
        )
        redact_long_tokens_on_save: bool = Field(
            default=True,
            description="Redact very long token-like strings before saving snapshots.",
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
        return Path(self.valves.data_dir).expanduser()

    def _scope_dir(self, scope_id: Optional[str]) -> Path:
        return self._base_dir() / self._scope_name(scope_id)

    def _index_path(self, scope_id: Optional[str]) -> Path:
        return self._scope_dir(scope_id) / self.INDEX_FILE

    def _snapshots_dir(self, scope_id: Optional[str]) -> Path:
        return self._scope_dir(scope_id) / self.SNAPSHOT_DIR

    def _validate_snapshot_id(self, snapshot_id: str) -> str:
        value = (snapshot_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value):
            raise ValueError(
                "snapshot_id must contain only letters, numbers, underscores, and hyphens."
            )
        return value

    def _snapshot_path(self, snapshot_id: str, scope_id: Optional[str]) -> Path:
        safe_id = self._validate_snapshot_id(snapshot_id)
        return self._snapshots_dir(scope_id) / f"{safe_id}.json"

    def _utc_now(self) -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _new_id(self, prefix: str = "ctx") -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}_{stamp}_{secrets.token_hex(4)}"

    # ------------------------------------------------------------------
    # Local locking and atomic JSON persistence
    # ------------------------------------------------------------------

    @contextmanager
    def _file_lock(self, target: Path):
        """
        Best-effort local lock using an adjacent .lock file.

        This is dependency-free and suitable for a normal local/container
        Open WebUI deployment. It is not a distributed lock for network filesystems.
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

    # ------------------------------------------------------------------
    # Index and snapshot helpers
    # ------------------------------------------------------------------

    def _empty_index(self, scope_id: Optional[str]) -> JsonDict:
        scope_name = self._scope_name(scope_id)
        return {
            "schema_version": "1.0",
            "scope_id": scope_name,
            "active_snapshot_id": "",
            "snapshots": [],
        }

    def _load_index_no_lock(self, scope_id: Optional[str]) -> JsonDict:
        path = self._index_path(scope_id)
        if not path.exists():
            return self._empty_index(scope_id)

        data = self._load_json_file(path)
        data.setdefault("schema_version", "1.0")
        data.setdefault("scope_id", self._scope_name(scope_id))
        data.setdefault("active_snapshot_id", "")
        data.setdefault("snapshots", [])

        if not isinstance(data["snapshots"], list):
            raise ValueError("index.json field 'snapshots' must be a list.")

        return data

    def _save_index_no_lock(self, scope_id: Optional[str], index: JsonDict) -> None:
        self._write_json_atomic(self._index_path(scope_id), index)

    def _load_snapshot_no_lock(
        self, snapshot_id: str, scope_id: Optional[str]
    ) -> Optional[JsonDict]:
        path = self._snapshot_path(snapshot_id, scope_id)
        if not path.exists():
            return None
        return self._load_json_file(path)

    def _save_snapshot_no_lock(
        self, snapshot_id: str, scope_id: Optional[str], snapshot: JsonDict
    ) -> None:
        self._write_json_atomic(self._snapshot_path(snapshot_id, scope_id), snapshot)

    def _snapshot_meta(self, snapshot: JsonDict) -> JsonDict:
        summary = str(snapshot.get("summary_markdown", ""))
        return {
            "snapshot_id": snapshot.get("snapshot_id", ""),
            "title": snapshot.get("title", ""),
            "status": snapshot.get("status", "pending"),
            "created_at": snapshot.get("created_at", ""),
            "updated_at": snapshot.get("updated_at", ""),
            "source_ref": snapshot.get("source_ref", ""),
            "tags": snapshot.get("tags", []),
            "char_count": len(summary),
        }

    def _find_meta_index(self, index: JsonDict, snapshot_id: str) -> Optional[int]:
        snapshots = index.get("snapshots", [])
        if not isinstance(snapshots, list):
            raise ValueError("index.json field 'snapshots' must be a list.")

        for i, item in enumerate(snapshots):
            if isinstance(item, dict) and item.get("snapshot_id") == snapshot_id:
                return i
        return None

    def _replace_or_append_meta(self, index: JsonDict, snapshot: JsonDict) -> None:
        meta = self._snapshot_meta(snapshot)
        snapshots = index.setdefault("snapshots", [])
        if not isinstance(snapshots, list):
            raise ValueError("index.json field 'snapshots' must be a list.")

        snapshot_id = str(meta.get("snapshot_id", ""))
        existing = self._find_meta_index(index, snapshot_id)
        if existing is None:
            snapshots.append(meta)
        else:
            snapshots[existing] = meta

    def _set_active_no_lock(
        self, index: JsonDict, snapshot_id: str, scope_id: Optional[str]
    ) -> JsonDict:
        snapshot_id = self._validate_snapshot_id(snapshot_id)

        if self._find_meta_index(index, snapshot_id) is None:
            raise KeyError(f"No snapshot found for {snapshot_id}.")

        target = self._load_snapshot_no_lock(snapshot_id, scope_id)
        if not target:
            raise KeyError(f"No snapshot file found for {snapshot_id}.")

        now = self._utc_now()
        old_active_id = str(index.get("active_snapshot_id") or "")

        if old_active_id and old_active_id != snapshot_id:
            old = self._load_snapshot_no_lock(old_active_id, scope_id)
            if old:
                old["status"] = "inactive"
                old["updated_at"] = now
                self._save_snapshot_no_lock(old_active_id, scope_id, old)
                self._replace_or_append_meta(index, old)

        target["status"] = "active"
        target["updated_at"] = now
        self._save_snapshot_no_lock(snapshot_id, scope_id, target)
        self._replace_or_append_meta(index, target)

        for meta in index.get("snapshots", []):
            if not isinstance(meta, dict):
                continue
            if meta.get("snapshot_id") == snapshot_id:
                meta["status"] = "active"
                meta["updated_at"] = now
            elif meta.get("status") == "active":
                meta["status"] = "inactive"
                meta["updated_at"] = now

        index["active_snapshot_id"] = snapshot_id
        return target

    def _deactivate_no_lock(self, index: JsonDict, scope_id: Optional[str]) -> str:
        active_id = str(index.get("active_snapshot_id") or "")
        if not active_id:
            return ""

        now = self._utc_now()
        active = self._load_snapshot_no_lock(active_id, scope_id)
        if active:
            active["status"] = "inactive"
            active["updated_at"] = now
            self._save_snapshot_no_lock(active_id, scope_id, active)
            self._replace_or_append_meta(index, active)

        for meta in index.get("snapshots", []):
            if isinstance(meta, dict) and meta.get("snapshot_id") == active_id:
                meta["status"] = "inactive"
                meta["updated_at"] = now

        index["active_snapshot_id"] = ""
        return active_id

    # ------------------------------------------------------------------
    # Validation, normalization, and redaction helpers
    # ------------------------------------------------------------------

    def _clamp_int(self, value: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _clean_title(self, title: str) -> str:
        value = str(title or "").strip()
        if not value:
            value = "Compacted context"
        max_len = max(1, int(self.valves.max_title_chars))
        return value[:max_len]

    def _clean_source_ref(self, source_ref: str) -> str:
        value = str(source_ref or "").strip()
        max_len = max(0, int(self.valves.max_source_ref_chars))
        return value[:max_len] if max_len else ""

    def _parse_tags(self, tags: str) -> list[str]:
        if not isinstance(tags, str) or not tags.strip():
            return []

        raw_items: list[str]
        text = tags.strip()

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
            if not tag or tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
            if len(out) >= max_tags:
                break

        return out

    def _redact_text(self, text: str) -> str:
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

    def _prepare_summary(
        self,
        summary_markdown: str,
        *,
        redact: bool,
        allow_truncate: bool,
    ) -> tuple[Optional[str], Optional[str], bool]:
        summary = str(summary_markdown or "").strip()
        if not summary:
            return None, "summary_markdown cannot be empty.", False

        redacted = False
        if bool(self.valves.redact_secrets_on_save) and redact:
            new_summary = self._redact_text(summary)
            redacted = new_summary != summary
            summary = new_summary

        max_chars = int(self.valves.max_summary_chars)
        if max_chars > 0 and len(summary) > max_chars:
            if not allow_truncate:
                return (
                    None,
                    f"summary_markdown is too large: {len(summary)} characters. Limit: {max_chars}.",
                    redacted,
                )

            omitted = len(summary) - max_chars
            suffix = f"\n\n[ContextCompactor: truncated {omitted} characters during save]"
            summary = summary[: max(0, max_chars - len(suffix))] + suffix

        return summary, None, redacted

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
        """Create the local JSON store for a context-compaction scope."""
        try:
            scope_name = self._scope_name(scope_id)
            index_path = self._index_path(scope_name)
            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                self._snapshots_dir(scope_name).mkdir(parents=True, exist_ok=True)
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "index_path": str(index_path),
                    "snapshots_dir": str(self._snapshots_dir(scope_name)),
                }
            )
        except Exception as e:
            return self._err("initialize_failed", str(e))

    async def save_context_snapshot(
        self,
        title: str = Field(
            ...,
            description="Short human-readable title for the compacted context snapshot.",
        ),
        summary_markdown: str = Field(
            ...,
            description="Structured compacted context summary in Markdown.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Use a chat/project/task ID when available. Empty uses the default scope.",
        ),
        source_ref: str = Field(
            default="",
            description="Optional source reference, such as chat title, message range, file name, or task name.",
        ),
        tags: str = Field(
            default="",
            description="Optional comma-separated tags, or a JSON list string.",
        ),
        activate: bool = Field(
            default=False,
            description="If true, immediately make this snapshot the active injected context for its scope.",
        ),
        redact: bool = Field(
            default=True,
            description="If true, apply local secret redaction before saving.",
        ),
        allow_truncate: bool = Field(
            default=False,
            description="If true, truncate summaries larger than max_summary_chars instead of rejecting them.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Save a context-compaction snapshot.

        The tool does not generate the summary. The model/user should provide a
        structured summary, preferably using the Context Compaction skill.
        """
        try:
            scope_name = self._scope_name(scope_id)
            summary, error, redacted = self._prepare_summary(
                summary_markdown,
                redact=redact,
                allow_truncate=allow_truncate,
            )
            if error:
                return self._err("invalid_summary", error)
            assert summary is not None

            cleaned_source_ref = self._clean_source_ref(source_ref)
            if bool(self.valves.redact_secrets_on_save) and redact:
                cleaned_source_ref = self._redact_text(cleaned_source_ref)

            now = self._utc_now()
            snapshot_id = self._new_id("ctx")
            snapshot: JsonDict = {
                "schema_version": "1.0",
                "snapshot_id": snapshot_id,
                "scope_id": scope_name,
                "title": self._clean_title(title),
                "status": "pending",
                "created_at": now,
                "updated_at": now,
                "source_ref": cleaned_source_ref,
                "tags": self._parse_tags(tags),
                "summary_markdown": summary,
                "metadata": {
                    "tool_version": self.TOOL_VERSION,
                    "created_by": (__user__ or {}).get("name", "unknown"),
                    "char_count": len(summary),
                    "redacted_on_save": redacted,
                },
            }

            index_path = self._index_path(scope_name)
            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                self._save_snapshot_no_lock(snapshot_id, scope_name, snapshot)
                self._replace_or_append_meta(index, snapshot)

                if activate:
                    snapshot = self._set_active_no_lock(index, snapshot_id, scope_name)

                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "snapshot": snapshot,
                    "activated": bool(activate),
                }
            )
        except Exception as e:
            return self._err("save_failed", str(e))

    async def get_active_context(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Return the active context snapshot for a scope, if one exists."""
        try:
            scope_name = self._scope_name(scope_id)
            index = self._load_index_no_lock(scope_name)
            active_id = str(index.get("active_snapshot_id") or "")
            if not active_id:
                return self._ok(
                    {
                        "scope_id": scope_name,
                        "active_snapshot": None,
                    }
                )

            snapshot = self._load_snapshot_no_lock(active_id, scope_name)
            if not snapshot:
                return self._err(
                    "active_snapshot_missing",
                    f"Index points to active snapshot {active_id}, but the snapshot file was not found.",
                    scope_id=scope_name,
                    active_snapshot_id=active_id,
                )

            return self._ok(
                {
                    "scope_id": scope_name,
                    "active_snapshot": snapshot,
                }
            )
        except Exception as e:
            return self._err("read_failed", str(e))

    async def list_context_snapshots(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        limit: int = Field(
            default=20,
            description="Maximum number of snapshot metadata entries to return.",
        ),
        status_filter: str = Field(
            default="all",
            description="One of: all, active, pending, inactive.",
        ),
    ) -> str:
        """List snapshot metadata for a scope without returning full summaries."""
        try:
            scope_name = self._scope_name(scope_id)
            limit = self._clamp_int(limit, 1, 200)
            status = (status_filter or "all").strip().lower()
            if status not in {"all", "active", "pending", "inactive"}:
                return self._err(
                    "invalid_status_filter",
                    "status_filter must be one of: all, active, pending, inactive.",
                    status_filter=status_filter,
                )

            index = self._load_index_no_lock(scope_name)
            snapshots = [item for item in index.get("snapshots", []) if isinstance(item, dict)]

            if status != "all":
                snapshots = [item for item in snapshots if item.get("status") == status]

            snapshots.sort(
                key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                reverse=True,
            )

            return self._ok(
                {
                    "scope_id": scope_name,
                    "active_snapshot_id": index.get("active_snapshot_id", ""),
                    "snapshots": snapshots[:limit],
                    "returned_count": min(len(snapshots), limit),
                    "matched_count": len(snapshots),
                }
            )
        except Exception as e:
            return self._err("list_failed", str(e))

    async def get_context_snapshot(
        self,
        snapshot_id: str = Field(
            ...,
            description="Snapshot ID returned by save_context_snapshot or list_context_snapshots.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Return one full context snapshot."""
        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_snapshot_id(snapshot_id)
            snapshot = self._load_snapshot_no_lock(safe_id, scope_name)
            if not snapshot:
                return self._err(
                    "not_found",
                    f"No context snapshot found for {safe_id}.",
                    scope_id=scope_name,
                    snapshot_id=safe_id,
                )

            return self._ok(
                {
                    "scope_id": scope_name,
                    "snapshot": snapshot,
                }
            )
        except Exception as e:
            return self._err("read_failed", str(e), snapshot_id=snapshot_id)

    async def activate_context_snapshot(
        self,
        snapshot_id: str = Field(
            ...,
            description="Snapshot ID to make active for this scope.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Make a saved snapshot the active compacted context for a scope."""
        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_snapshot_id(snapshot_id)
            index_path = self._index_path(scope_name)

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                snapshot = self._set_active_no_lock(index, safe_id, scope_name)
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "active_snapshot_id": safe_id,
                    "active_snapshot": snapshot,
                }
            )
        except KeyError as e:
            return self._err("not_found", str(e), snapshot_id=snapshot_id)
        except Exception as e:
            return self._err("activate_failed", str(e), snapshot_id=snapshot_id)

    async def deactivate_context(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Clear the active snapshot for a scope without deleting any snapshots."""
        try:
            scope_name = self._scope_name(scope_id)
            index_path = self._index_path(scope_name)

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                deactivated_id = self._deactivate_no_lock(index, scope_name)
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "deactivated_snapshot_id": deactivated_id,
                    "active_snapshot_id": "",
                }
            )
        except Exception as e:
            return self._err("deactivate_failed", str(e))

    async def delete_context_snapshot(
        self,
        snapshot_id: str = Field(
            ...,
            description="Snapshot ID to delete.",
        ),
        confirm: str = Field(
            ...,
            description='Must be exactly "DELETE" to delete the snapshot.',
        ),
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
    ) -> str:
        """Delete one context snapshot after explicit confirmation."""
        if confirm != "DELETE":
            return self._err(
                "confirmation_required",
                'Delete aborted. The confirm value must be exactly "DELETE".',
            )

        try:
            scope_name = self._scope_name(scope_id)
            safe_id = self._validate_snapshot_id(snapshot_id)
            index_path = self._index_path(scope_name)
            deleted = False
            was_active = False

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                was_active = index.get("active_snapshot_id") == safe_id

                snapshots = index.get("snapshots", [])
                if not isinstance(snapshots, list):
                    raise ValueError("index.json field 'snapshots' must be a list.")

                before = len(snapshots)
                index["snapshots"] = [
                    item
                    for item in snapshots
                    if not (
                        isinstance(item, dict)
                        and item.get("snapshot_id") == safe_id
                    )
                ]

                if len(index["snapshots"]) != before:
                    deleted = True

                if was_active:
                    index["active_snapshot_id"] = ""

                path = self._snapshot_path(safe_id, scope_name)
                if path.exists():
                    path.unlink()
                    deleted = True

                self._save_index_no_lock(scope_name, index)

            if not deleted:
                return self._err(
                    "not_found",
                    f"No context snapshot found for {safe_id}.",
                    scope_id=scope_name,
                    snapshot_id=safe_id,
                )

            return self._ok(
                {
                    "scope_id": scope_name,
                    "snapshot_id": safe_id,
                    "deleted": True,
                    "was_active": was_active,
                    "active_snapshot_id": "",
                }
            )
        except Exception as e:
            return self._err("delete_failed", str(e), snapshot_id=snapshot_id)

    async def prune_context_snapshots(
        self,
        scope_id: str = Field(
            default="",
            description="Optional scope. Empty uses the default scope.",
        ),
        keep_recent: int = Field(
            default=20,
            description="Number of newest snapshots to preserve regardless of inactive/pending pruning.",
        ),
        delete_inactive: bool = Field(
            default=True,
            description="Delete inactive snapshots outside the keep_recent window.",
        ),
        delete_pending: bool = Field(
            default=False,
            description="Delete pending snapshots outside the keep_recent window.",
        ),
        confirm: str = Field(
            default="",
            description='Must be exactly "PRUNE" to prune snapshots.',
        ),
    ) -> str:
        """Prune old inactive or pending snapshots after explicit confirmation."""
        if confirm != "PRUNE":
            return self._err(
                "confirmation_required",
                'Prune aborted. The confirm value must be exactly "PRUNE".',
            )

        try:
            scope_name = self._scope_name(scope_id)
            keep_recent = self._clamp_int(keep_recent, 0, 500)
            index_path = self._index_path(scope_name)
            removed: list[str] = []
            preserved: list[str] = []

            with self._file_lock(index_path):
                index = self._load_index_no_lock(scope_name)
                active_id = str(index.get("active_snapshot_id") or "")
                metas = [item for item in index.get("snapshots", []) if isinstance(item, dict)]

                metas.sort(
                    key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                    reverse=True,
                )

                keep_ids: set[str] = set()
                if active_id:
                    keep_ids.add(active_id)

                for item in metas[:keep_recent]:
                    sid = str(item.get("snapshot_id") or "")
                    if sid:
                        keep_ids.add(sid)

                new_metas: list[JsonDict] = []
                for item in metas:
                    sid = str(item.get("snapshot_id") or "")
                    status = str(item.get("status") or "pending")
                    should_delete = False

                    if sid and sid not in keep_ids:
                        if status == "inactive" and delete_inactive:
                            should_delete = True
                        elif status == "pending" and delete_pending:
                            should_delete = True

                    if should_delete:
                        path = self._snapshot_path(sid, scope_name)
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            pass
                        removed.append(sid)
                    else:
                        if sid:
                            preserved.append(sid)
                        new_metas.append(item)

                index["snapshots"] = new_metas
                self._save_index_no_lock(scope_name, index)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "removed_snapshot_ids": removed,
                    "removed_count": len(removed),
                    "preserved_count": len(preserved),
                    "active_snapshot_id": self._load_index_no_lock(scope_name).get(
                        "active_snapshot_id", ""
                    ),
                }
            )
        except Exception as e:
            return self._err("prune_failed", str(e))
