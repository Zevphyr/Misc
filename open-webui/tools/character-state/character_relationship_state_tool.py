"""
title: Character & Relationship State
author: Local
version: 0.3.0
required_open_webui_version: 0.6.0
description: Local state lookup and reviewable update tool for character, relationship, and event records stored as scoped JSON files.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


JsonDict = dict[str, Any]


class Tools:
    """
    Open WebUI tool for local, scoped character and relationship state.

    Design intent:
    - Read functions never mutate state.
    - Proposal functions create reviewable pending proposals only.
    - Apply functions explicitly mutate live state, append an event, and mark the proposal applied.
    - State remains local JSON for portability and vendor independence.
    """

    STORE_FILES = {
        "character_profiles.json",
        "character_states.json",
        "relationship_states.json",
        "state_events.json",
        "proposed_updates.json",
    }

    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".rst",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".log",
    }

    class Valves(BaseModel):
        data_dir: str = Field(
            default="/app/backend/data/state_data",
            description="Base directory containing per-scope state folders.",
        )
        default_scope: str = Field(
            default="default",
            description="Default state scope/campaign if none is provided.",
        )
        max_file_bytes: int = Field(
            default=1_500_000,
            description="Maximum attached file size accepted for text extraction.",
        )
        max_text_chars: int = Field(
            default=250_000,
            description="Maximum extracted text characters to inspect from an attached file.",
        )
        max_patch_chars: int = Field(
            default=60_000,
            description="Maximum serialized size of an incoming JSON patch.",
        )
        max_evidence_messages: int = Field(
            default=20,
            description="Maximum recent chat messages to store as proposal evidence.",
        )
        max_message_chars: int = Field(
            default=1500,
            description="Maximum characters retained from each evidence message.",
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

    # ---------------------------------------------------------------------
    # Response helpers
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Path, scope, time, and ID helpers
    # ---------------------------------------------------------------------

    def _scope_name(self, scope_id: Optional[str]) -> str:
        raw = (scope_id or self.valves.default_scope or "default").strip()
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
        if not safe or safe in {".", ".."}:
            safe = "default"
        return safe[:80]

    def _scope_dir(self, scope_id: Optional[str]) -> Path:
        return Path(self.valves.data_dir).expanduser() / self._scope_name(scope_id)

    def _path(self, filename: str, scope_id: Optional[str] = None) -> Path:
        if filename not in self.STORE_FILES:
            raise ValueError(f"Unsupported state filename: {filename}")
        return self._scope_dir(scope_id) / filename

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _new_id(self, prefix: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}_{stamp}_{secrets.token_hex(3)}"

    # ---------------------------------------------------------------------
    # Local locking and atomic JSON persistence
    # ---------------------------------------------------------------------

    @contextmanager
    def _file_lock(self, target: Path):
        """
        Best-effort lock using an adjacent .lock file.

        This avoids adding a platform-specific dependency while protecting the
        common local-container case. It is not a distributed lock and should not
        be used over network filesystems that do not honor atomic create.
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

    def _empty_store_templates(self) -> dict[str, JsonDict]:
        return {
            "character_profiles.json": {},
            "character_states.json": {},
            "relationship_states.json": {},
            "state_events.json": {"events": []},
            "proposed_updates.json": {"items": []},
        }

    def _default_for_file(self, filename: str) -> JsonDict:
        return copy.deepcopy(self._empty_store_templates()[filename])

    def _load_json_no_lock(self, filename: str, scope_id: Optional[str] = None) -> JsonDict:
        path = self._path(filename, scope_id)
        if not path.exists():
            return self._default_for_file(filename)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{filename} must contain a JSON object at the top level.")

        return data

    def _load_json(self, filename: str, scope_id: Optional[str] = None) -> JsonDict:
        return self._load_json_no_lock(filename, scope_id)

    def _write_json_atomic_no_lock(self, filename: str, data: JsonDict, scope_id: Optional[str] = None) -> None:
        path = self._path(filename, scope_id)
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

    def _save_json(self, filename: str, data: JsonDict, scope_id: Optional[str] = None) -> None:
        path = self._path(filename, scope_id)
        with self._file_lock(path):
            self._write_json_atomic_no_lock(filename, data, scope_id)

    def _update_json_file(
        self,
        filename: str,
        scope_id: Optional[str],
        updater: Callable[[JsonDict], tuple[JsonDict, Any]],
    ) -> Any:
        path = self._path(filename, scope_id)
        with self._file_lock(path):
            current = self._load_json_no_lock(filename, scope_id)
            updated, result = updater(current)
            if not isinstance(updated, dict):
                raise ValueError(f"Updater for {filename} did not return a JSON object.")
            self._write_json_atomic_no_lock(filename, updated, scope_id)
            return result

    # ---------------------------------------------------------------------
    # Validation and merge helpers
    # ---------------------------------------------------------------------

    def _clamp_int(self, value: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _parse_patch_json(self, patch_json: str) -> tuple[Optional[JsonDict], Optional[str]]:
        if len(patch_json) > int(self.valves.max_patch_chars):
            return None, f"patch_json is too large. Limit: {self.valves.max_patch_chars} characters."

        try:
            patch = json.loads(patch_json)
        except Exception as e:
            return None, f"Invalid patch_json: {e}"

        if not isinstance(patch, dict):
            return None, "patch_json must decode to a JSON object."

        return patch, None

    def _deep_merge(self, current: Any, patch: Any) -> Any:
        if isinstance(current, dict) and isinstance(patch, dict):
            merged = copy.deepcopy(current)
            for key, value in patch.items():
                merged[key] = self._deep_merge(merged.get(key), value)
            return merged
        return copy.deepcopy(patch)

    # ---------------------------------------------------------------------
    # Proposal and event helpers
    # ---------------------------------------------------------------------

    def _append_proposal(self, proposal: JsonDict, scope_id: Optional[str] = None) -> JsonDict:
        created = copy.deepcopy(proposal)
        created.setdefault("proposal_id", self._new_id("proposal"))
        created.setdefault("scope_id", self._scope_name(scope_id))
        created.setdefault("status", "pending")
        created.setdefault("created_at", self._utc_now())

        def updater(data: JsonDict) -> tuple[JsonDict, JsonDict]:
            items = data.setdefault("items", [])
            if not isinstance(items, list):
                raise ValueError("proposed_updates.json field 'items' must be a list.")
            items.append(copy.deepcopy(created))
            return data, created

        return self._update_json_file("proposed_updates.json", scope_id, updater)

    def _get_proposal(self, proposal_id: str, scope_id: Optional[str]) -> tuple[Optional[JsonDict], Optional[int]]:
        proposals = self._load_json("proposed_updates.json", scope_id)
        items = proposals.get("items", [])
        if not isinstance(items, list):
            raise ValueError("proposed_updates.json field 'items' must be a list.")

        for index, proposal in enumerate(items):
            if isinstance(proposal, dict) and proposal.get("proposal_id") == proposal_id:
                return proposal, index
        return None, None

    def _mark_proposal_applied(self, proposal_id: str, event_id: str, scope_id: Optional[str]) -> JsonDict:
        def updater(data: JsonDict) -> tuple[JsonDict, JsonDict]:
            items = data.setdefault("items", [])
            if not isinstance(items, list):
                raise ValueError("proposed_updates.json field 'items' must be a list.")

            for item in items:
                if isinstance(item, dict) and item.get("proposal_id") == proposal_id:
                    item["status"] = "applied"
                    item["applied_at"] = self._utc_now()
                    item["applied_event_id"] = event_id
                    return data, copy.deepcopy(item)

            raise ValueError(f"No proposal found for {proposal_id}.")

        return self._update_json_file("proposed_updates.json", scope_id, updater)

    def _append_event(self, event: JsonDict, scope_id: Optional[str]) -> JsonDict:
        created = copy.deepcopy(event)
        created.setdefault("event_id", self._new_id("evt"))
        created.setdefault("timestamp", self._utc_now())

        def updater(data: JsonDict) -> tuple[JsonDict, JsonDict]:
            events = data.setdefault("events", [])
            if not isinstance(events, list):
                raise ValueError("state_events.json field 'events' must be a list.")
            events.append(copy.deepcopy(created))
            return data, created

        return self._update_json_file("state_events.json", scope_id, updater)

    def _proposal_target_id(self, proposal: JsonDict) -> str:
        return str(
            proposal.get("target_id")
            or proposal.get("relationship_id")
            or proposal.get("character_id")
            or ""
        ).strip()

    def _apply_state_like_proposal(
        self,
        proposal_id: str,
        scope_id: Optional[str],
        expected_type: Optional[str] = None,
    ) -> str:
        try:
            proposal, _ = self._get_proposal(proposal_id, scope_id)
            if not proposal:
                return self._err("not_found", f"No proposal found for {proposal_id}.")

            status = str(proposal.get("status", "pending"))
            if status != "pending":
                return self._err(
                    "proposal_not_pending",
                    f"Proposal {proposal_id} has status '{status}' and will not be applied again.",
                    proposal_id=proposal_id,
                    status=status,
                )

            proposal_type = str(proposal.get("type", ""))
            if expected_type and proposal_type != expected_type:
                return self._err(
                    "wrong_proposal_type",
                    f"Proposal {proposal_id} has type '{proposal_type}', not '{expected_type}'.",
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                )

            if proposal_type not in {"character_state", "relationship_state"}:
                return self._err(
                    "wrong_proposal_type",
                    f"Proposal {proposal_id} is not a character_state or relationship_state proposal.",
                    proposal_id=proposal_id,
                    proposal_type=proposal_type,
                )

            target_id = self._proposal_target_id(proposal)
            if not target_id:
                return self._err("invalid_proposal", f"Proposal {proposal_id} has no target_id.")

            patch = proposal.get("patch", {})
            if not isinstance(patch, dict):
                return self._err("invalid_proposal", f"Proposal {proposal_id} patch must be a JSON object.")

            if proposal_type == "character_state":
                filename = "character_states.json"
                missing_message = f"No character_state found for {target_id}."
                event_type = "character_state_update"
                relationship_id = ""
            else:
                filename = "relationship_states.json"
                missing_message = f"No relationship_state found for {target_id}."
                event_type = "relationship_state_update"
                relationship_id = target_id

            def state_updater(data: JsonDict) -> tuple[JsonDict, JsonDict]:
                current = data.get(target_id)
                if not isinstance(current, dict):
                    raise KeyError(missing_message)
                merged = self._deep_merge(current, patch)
                data[target_id] = merged
                return data, merged

            try:
                updated_state = self._update_json_file(filename, scope_id, state_updater)
            except KeyError:
                return self._err("not_found", missing_message, target_id=target_id)

            participants = [target_id]
            if proposal_type == "relationship_state" and isinstance(updated_state, dict):
                participants = updated_state.get("participants", []) or []
                if not isinstance(participants, list):
                    participants = []

            event = self._append_event(
                {
                    "event_type": event_type,
                    "participants": participants,
                    "relationship_id": relationship_id,
                    "summary": proposal.get("summary", f"Applied state proposal {proposal_id}."),
                    "source": {
                        "origin_type": "proposal",
                        "proposal_id": proposal_id,
                        "document_or_chat_ref": proposal.get("source_ref", ""),
                        "approved": True,
                    },
                },
                scope_id,
            )

            applied_proposal = self._mark_proposal_applied(proposal_id, event["event_id"], scope_id)

            return self._ok(
                {
                    "scope_id": self._scope_name(scope_id),
                    "proposal": applied_proposal,
                    "event": event,
                    "state": updated_state,
                }
            )
        except Exception as e:
            return self._err("apply_failed", str(e), proposal_id=proposal_id)

    # ---------------------------------------------------------------------
    # Attached-file extraction helpers
    # ---------------------------------------------------------------------

    def _find_existing_path(self, obj: Any) -> Optional[str]:
        """Recursively search an Open WebUI file object for an existing local file path."""
        if isinstance(obj, str):
            p = Path(obj)
            if p.exists() and p.is_file():
                return str(p)
            return None

        if isinstance(obj, dict):
            for value in obj.values():
                found = self._find_existing_path(value)
                if found:
                    return found
            return None

        if isinstance(obj, list):
            for item in obj:
                found = self._find_existing_path(item)
                if found:
                    return found
            return None

        return None

    def _enforce_file_bounds(self, path: Path) -> None:
        size = path.stat().st_size
        if size > int(self.valves.max_file_bytes):
            raise ValueError(
                f"File is too large for extraction: {size} bytes. Limit: {self.valves.max_file_bytes} bytes."
            )

    def _read_docx_text(self, path: Path) -> str:
        """Minimal DOCX text extractor using only the Python standard library."""
        self._enforce_file_bounds(path)
        with zipfile.ZipFile(path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("DOCX does not contain word/document.xml.")
            with zf.open("word/document.xml") as f:
                xml_bytes = f.read()

        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        paragraphs: list[str] = []
        for para in root.findall(".//w:p", ns):
            runs: list[str] = []
            for node in para.iter():
                if node.tag == f"{{{ns['w']}}}t" and node.text:
                    runs.append(node.text)
                elif node.tag == f"{{{ns['w']}}}tab":
                    runs.append("\t")
                elif node.tag == f"{{{ns['w']}}}br":
                    runs.append("\n")
            line = "".join(runs).strip()
            if line:
                paragraphs.append(line)

        text = "\n\n".join(paragraphs)
        return text[: int(self.valves.max_text_chars)]

    def _read_file_text(self, path: Path) -> str:
        self._enforce_file_bounds(path)
        ext = path.suffix.lower()

        if ext == ".docx":
            return self._read_docx_text(path)

        if ext not in self.TEXT_EXTENSIONS:
            allowed = sorted(self.TEXT_EXTENSIONS | {".docx"})
            raise ValueError(f"Unsupported file extension '{ext}'. Allowed extensions: {allowed}")

        text = path.read_text(encoding="utf-8", errors="replace")
        return text[: int(self.valves.max_text_chars)]

    def _extract_snippets(
        self,
        text: str,
        needle: str,
        max_snippets: int = 8,
        context_paragraphs: int = 1,
    ) -> list[str]:
        if not text.strip():
            return []

        max_snippets = self._clamp_int(max_snippets, 1, 20)
        context_paragraphs = self._clamp_int(context_paragraphs, 0, 3)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        if not needle.strip():
            return [p[:4000] for p in paragraphs[:max_snippets]]

        needle_lower = needle.lower()
        hits: list[str] = []

        for i, para in enumerate(paragraphs):
            if needle_lower in para.lower():
                start = max(0, i - context_paragraphs)
                end = min(len(paragraphs), i + context_paragraphs + 1)
                snippet = "\n\n".join(paragraphs[start:end]).strip()
                hits.append(snippet[:4000])

        unique_hits: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            if hit not in seen:
                seen.add(hit)
                unique_hits.append(hit)

        return unique_hits[:max_snippets]

    def _suggest_baseline_fields(self, text: str) -> JsonDict:
        """
        Conservative heuristic extraction for labeled fields.

        This deliberately avoids pretending to infer complex characterization.
        It only captures simple explicit labels from structured notes.
        """

        def first_match(pattern: str) -> str:
            m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            return m.group(1).strip() if m else ""

        traits_raw = first_match(
            r"^\s*(?:traits|core behavioral traits|behavioral traits)\s*:\s*(.+)$"
        )
        traits = [t.strip() for t in re.split(r"[;,]", traits_raw) if t.strip()] if traits_raw else []

        return {
            "role": first_match(r"^\s*(?:role|function|position)\s*:\s*(.+)$"),
            "public_persona": first_match(
                r"^\s*(?:public persona|surface affect|public affect)\s*:\s*(.+)$"
            ),
            "default_register": first_match(
                r"^\s*(?:default register|speech level|register)\s*:\s*(.+)$"
            ),
            "traits": traits,
        }

    def _tail_messages(self, messages: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
        if not messages:
            return []

        limit = self._clamp_int(limit, 1, int(self.valves.max_evidence_messages))
        max_chars = int(self.valves.max_message_chars)
        tail = messages[-limit:]
        compact: list[dict[str, str]] = []

        for msg in tail:
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "unknown"))
            content = msg.get("content", "")

            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                        else:
                            parts.append(str(item))
                    else:
                        parts.append(str(item))
                content_str = "\n".join(parts)
            else:
                content_str = str(content)

            content_str = content_str.strip()
            if len(content_str) > max_chars:
                content_str = content_str[:max_chars] + "\n...[truncated]"

            compact.append({"role": role, "content": content_str})

        return compact

    # ---------------------------------------------------------------------
    # Public Open WebUI tool methods
    # ---------------------------------------------------------------------

    async def initialize_scope(
        self,
        scope_id: str = Field(
            ...,
            description="Scope or campaign folder to initialize, for example rwby-main or rp-01.",
        ),
        overwrite: bool = Field(
            False,
            description="If true, overwrite existing files in the scope with empty templates.",
        ),
    ) -> str:
        """Create the JSON store for a new scope/campaign."""
        try:
            scope_name = self._scope_name(scope_id)
            templates = self._empty_store_templates()
            created: list[str] = []
            skipped: list[str] = []

            for filename, data in templates.items():
                path = self._path(filename, scope_name)
                if path.exists() and not overwrite:
                    skipped.append(filename)
                    continue
                self._save_json(filename, copy.deepcopy(data), scope_name)
                created.append(filename)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "created_files": created,
                    "skipped_files": skipped,
                    "overwrite": overwrite,
                }
            )
        except Exception as e:
            return self._err("initialize_failed", str(e))

    async def reset_scope(
        self,
        scope_id: str = Field(
            ...,
            description="Scope or campaign folder to reset, for example rwby-main or rp-01.",
        ),
        confirm: str = Field(
            ...,
            description='Must be exactly "RESET" to proceed.',
        ),
    ) -> str:
        """Reset one scope/campaign to empty JSON files."""
        if confirm != "RESET":
            return self._err("confirmation_required", 'Reset aborted. The confirm value must be exactly "RESET".')

        try:
            scope_name = self._scope_name(scope_id)
            templates = self._empty_store_templates()
            for filename, data in templates.items():
                self._save_json(filename, copy.deepcopy(data), scope_name)

            return self._ok(
                {
                    "scope_id": scope_name,
                    "status": "reset",
                    "files": list(templates.keys()),
                }
            )
        except Exception as e:
            return self._err("reset_failed", str(e))

    async def list_scopes(self) -> str:
        """List available scope/campaign folders."""
        try:
            base = Path(self.valves.data_dir).expanduser()
            if not base.exists():
                return self._ok({"scopes": []})

            scopes = sorted([p.name for p in base.iterdir() if p.is_dir()])
            return self._ok({"scopes": scopes})
        except Exception as e:
            return self._err("list_scopes_failed", str(e))

    async def extract_character_baseline_from_file(
        self,
        character_id: str = Field(
            ..., description="Character ID, for example char_ruby_rose"
        ),
        character_name: str = Field(
            ...,
            description="Character display name to search for in the attached file, for example Ruby Rose",
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
        file_index: int = Field(
            0, description="Which attached file to use, starting from 0."
        ),
        save_proposal: bool = Field(
            True,
            description="Whether to save the extracted baseline as a proposal in proposed_updates.json.",
        ),
        __files__: Optional[list] = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Read an attached file, extract evidence relevant to one character,
        and optionally create a proposed baseline record.
        """
        try:
            files = list(__files__ or [])
            if not files:
                return self._err("no_files", "No attached files were available to the tool.")

            file_index = int(file_index)
            if file_index < 0 or file_index >= len(files):
                return self._err(
                    "invalid_file_index",
                    f"Invalid file_index {file_index}. Available file count: {len(files)}.",
                    file_count=len(files),
                )

            path_str = self._find_existing_path(files[file_index])
            if not path_str:
                return self._err(
                    "file_path_not_found",
                    "Could not find a readable local file path in the selected attached file object.",
                )

            path = Path(path_str)
            text = self._read_file_text(path)
            if not text.strip():
                return self._err("empty_file_text", "The attached file was readable, but no text could be extracted.")

            snippets = self._extract_snippets(text, character_name)
            suggested = self._suggest_baseline_fields(text)
            proposed_by = (__user__ or {}).get("name", "unknown")

            proposal: JsonDict = {
                "type": "character_baseline_extraction",
                "character_id": character_id,
                "character_name": character_name,
                "source_file": {
                    "name": path.name,
                    "stem": path.stem,
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                },
                "summary": f"Proposed character baseline extracted from attached file for {character_name}.",
                "suggested_profile_patch": {
                    "id": character_id,
                    "name": character_name,
                    "source_documents": [path.stem],
                    "baseline": {
                        "role": suggested.get("role", ""),
                        "public_persona": suggested.get("public_persona", ""),
                        "core_behavioral_traits": suggested.get("traits", []),
                        "speech_profile": {
                            "default_register": suggested.get("default_register", ""),
                            "address_habits": [],
                            "notable_patterns": [],
                        },
                        "knowledge_baseline": {
                            "setting_facts_known": [],
                            "setting_facts_unknown": [],
                        },
                        "baseline_relationship_notes": [],
                        "hard_constraints": [],
                    },
                },
                "evidence": {
                    "snippet_count": len(snippets),
                    "snippets": snippets if snippets else [text[:2500]],
                },
                "proposed_by": proposed_by,
            }

            if save_proposal:
                proposal = self._append_proposal(proposal, scope_id)

            return self._ok({"proposal": proposal, "saved": save_proposal})
        except Exception as e:
            return self._err("baseline_extraction_failed", str(e))

    async def propose_state_update_from_chat(
        self,
        update_type: str = Field(
            ..., description="Either 'character_state' or 'relationship_state'."
        ),
        target_id: str = Field(
            ..., description="Character ID or relationship ID to update."
        ),
        summary: str = Field(..., description="Short summary of what changed and why."),
        patch_json: str = Field(
            ...,
            description='JSON object containing the proposed state patch. Example: {"shared_state": {"emotional_temperature": "tense"}}',
        ),
        lookback_messages: int = Field(
            8, description="How many recent chat messages to include as evidence."
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
        __messages__: Optional[list] = None,
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Create a proposed character_state or relationship_state update from recent chat history.
        This stores the patch plus transcript evidence, but does not apply the update.
        """
        try:
            if update_type not in {"character_state", "relationship_state"}:
                return self._err(
                    "invalid_update_type",
                    "update_type must be either 'character_state' or 'relationship_state'.",
                    update_type=update_type,
                )

            patch, patch_error = self._parse_patch_json(patch_json)
            if patch_error:
                return self._err("invalid_patch_json", patch_error)
            assert patch is not None

            existing_file = "character_states.json" if update_type == "character_state" else "relationship_states.json"
            existing = self._load_json(existing_file, scope_id)
            target_exists = target_id in existing
            evidence_messages = self._tail_messages(list(__messages__ or []), lookback_messages)

            proposal = self._append_proposal(
                {
                    "type": update_type,
                    "target_id": target_id,
                    "summary": summary,
                    "patch": patch,
                    "target_exists": target_exists,
                    "evidence": {
                        "message_count": len(evidence_messages),
                        "messages": evidence_messages,
                    },
                    "proposed_by": (__user__ or {}).get("name", "unknown"),
                },
                scope_id,
            )

            return self._ok({"proposal": proposal})
        except Exception as e:
            return self._err("proposal_failed", str(e))

    async def get_character_profile(
        self,
        character_id: str = Field(
            ..., description="Character ID, for example char_ruby_rose"
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Return the baseline stored profile for one character."""
        try:
            profiles = self._load_json("character_profiles.json", scope_id)
            profile = profiles.get(character_id)
            if profile is None:
                return self._err(
                    "not_found",
                    f"No character_profile found for {character_id} in scope {self._scope_name(scope_id)}.",
                    character_id=character_id,
                    scope_id=self._scope_name(scope_id),
                )
            return self._ok({"character_id": character_id, "profile": profile})
        except Exception as e:
            return self._err("read_failed", str(e), character_id=character_id)

    async def get_character_state(
        self,
        character_id: str = Field(
            ..., description="Character ID, for example char_ruby_rose"
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Return the current stored state for one character."""
        try:
            states = self._load_json("character_states.json", scope_id)
            state = states.get(character_id)
            if state is None:
                return self._err(
                    "not_found",
                    f"No character_state found for {character_id} in scope {self._scope_name(scope_id)}.",
                    character_id=character_id,
                    scope_id=self._scope_name(scope_id),
                )
            return self._ok({"character_id": character_id, "state": state})
        except Exception as e:
            return self._err("read_failed", str(e), character_id=character_id)

    async def get_relationship_state(
        self,
        relationship_id: str = Field(
            ..., description="Relationship ID, for example rel_ruby_jaune"
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Return the current stored state for one relationship."""
        try:
            states = self._load_json("relationship_states.json", scope_id)
            state = states.get(relationship_id)
            if state is None:
                return self._err(
                    "not_found",
                    f"No relationship_state found for {relationship_id} in scope {self._scope_name(scope_id)}.",
                    relationship_id=relationship_id,
                    scope_id=self._scope_name(scope_id),
                )
            return self._ok({"relationship_id": relationship_id, "state": state})
        except Exception as e:
            return self._err("read_failed", str(e), relationship_id=relationship_id)

    async def list_recent_state_events(
        self,
        limit: int = Field(
            5, description="Maximum number of most recent events to return."
        ),
        participant_id: str = Field(
            default="",
            description="Optional character ID to filter events by participant.",
        ),
        relationship_id: str = Field(
            default="",
            description="Optional relationship ID to filter events by relationship.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Return recent state events, optionally filtered by character or relationship."""
        try:
            limit = self._clamp_int(limit, 1, 100)
            data = self._load_json("state_events.json", scope_id)
            events = data.get("events", [])
            if not isinstance(events, list):
                return self._err("invalid_store", "state_events.json field 'events' must be a list.")

            filtered = events
            if participant_id:
                filtered = [evt for evt in filtered if participant_id in evt.get("participants", [])]

            if relationship_id:
                filtered = [evt for evt in filtered if evt.get("relationship_id", "") == relationship_id]

            return self._ok(
                {
                    "scope_id": self._scope_name(scope_id),
                    "events": filtered[-limit:],
                    "returned_count": min(len(filtered), limit),
                    "matched_count": len(filtered),
                }
            )
        except Exception as e:
            return self._err("list_events_failed", str(e))

    async def propose_relationship_update(
        self,
        relationship_id: str = Field(
            ..., description="Relationship ID, for example rel_ruby_jaune"
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
        summary: str = Field(..., description="Short summary of what changed and why."),
        patch_json: str = Field(
            ...,
            description='JSON object containing the proposed patch. Example: {"shared_state": {"emotional_temperature": "tense"}}',
        ),
        source_ref: str = Field(
            default="",
            description="Optional scene, draft, or note reference that caused the update.",
        ),
        __user__: Optional[dict] = None,
    ) -> str:
        """
        Store a proposed relationship-state update for later review and approval.

        Kept for compatibility with the older relationship-specific workflow,
        but internally uses the same proposal shape as generic state updates.
        """
        try:
            patch, patch_error = self._parse_patch_json(patch_json)
            if patch_error:
                return self._err("invalid_patch_json", patch_error)
            assert patch is not None

            existing = self._load_json("relationship_states.json", scope_id)
            proposal = self._append_proposal(
                {
                    "type": "relationship_state",
                    "target_id": relationship_id,
                    "relationship_id": relationship_id,
                    "summary": summary,
                    "patch": patch,
                    "source_ref": source_ref,
                    "target_exists": relationship_id in existing,
                    "proposed_by": (__user__ or {}).get("name", "unknown"),
                },
                scope_id,
            )

            return self._ok({"proposal": proposal})
        except Exception as e:
            return self._err("proposal_failed", str(e), relationship_id=relationship_id)

    async def apply_relationship_update(
        self,
        proposal_id: str = Field(
            ..., description="Proposal ID created by propose_relationship_update."
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Apply a pending relationship-state proposal to the live relationship state."""
        return self._apply_state_like_proposal(proposal_id, scope_id, expected_type="relationship_state")

    async def apply_character_baseline_proposal(
        self,
        proposal_id: str = Field(
            ...,
            description="Proposal ID created by extract_character_baseline_from_file.",
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Apply a pending character baseline extraction proposal into character_profiles.json."""
        try:
            proposal, _ = self._get_proposal(proposal_id, scope_id)
            if not proposal:
                return self._err("not_found", f"No proposal found for {proposal_id}.")

            status = str(proposal.get("status", "pending"))
            if status != "pending":
                return self._err(
                    "proposal_not_pending",
                    f"Proposal {proposal_id} has status '{status}' and will not be applied again.",
                    proposal_id=proposal_id,
                    status=status,
                )

            if proposal.get("type") != "character_baseline_extraction":
                return self._err(
                    "wrong_proposal_type",
                    f"Proposal {proposal_id} is not a character_baseline_extraction proposal.",
                    proposal_id=proposal_id,
                    proposal_type=proposal.get("type", ""),
                )

            patch = proposal.get("suggested_profile_patch", {})
            if not isinstance(patch, dict):
                return self._err("invalid_proposal", f"Proposal {proposal_id} suggested_profile_patch must be an object.")

            character_id = str(patch.get("id", "")).strip()
            if not character_id:
                return self._err("invalid_proposal", f"Proposal {proposal_id} does not contain a valid character ID.")

            def profile_updater(data: JsonDict) -> tuple[JsonDict, JsonDict]:
                current = data.get(character_id, {})
                if current is None:
                    current = {}
                if not isinstance(current, dict):
                    raise ValueError(f"Existing profile for {character_id} must be a JSON object.")
                merged = self._deep_merge(current, patch)
                data[character_id] = merged
                return data, merged

            merged_profile = self._update_json_file("character_profiles.json", scope_id, profile_updater)

            event = self._append_event(
                {
                    "event_type": "character_profile_update",
                    "participants": [character_id],
                    "relationship_id": "",
                    "summary": proposal.get("summary", f"Applied baseline extraction for {character_id}."),
                    "source": {
                        "origin_type": "proposal",
                        "proposal_id": proposal_id,
                        "document_or_chat_ref": proposal.get("source_file", {}).get("name", ""),
                        "approved": True,
                    },
                },
                scope_id,
            )

            applied_proposal = self._mark_proposal_applied(proposal_id, event["event_id"], scope_id)

            return self._ok(
                {
                    "scope_id": self._scope_name(scope_id),
                    "proposal": applied_proposal,
                    "event": event,
                    "profile": merged_profile,
                }
            )
        except Exception as e:
            return self._err("apply_failed", str(e), proposal_id=proposal_id)

    async def apply_state_proposal(
        self,
        proposal_id: str = Field(
            ..., description="Proposal ID created by propose_state_update_from_chat."
        ),
        scope_id: str = Field(
            default="",
            description="Optional state scope or campaign folder, for example rwby-main or rp-01.",
        ),
    ) -> str:
        """Apply a pending character_state or relationship_state proposal to the live state store."""
        return self._apply_state_like_proposal(proposal_id, scope_id)
