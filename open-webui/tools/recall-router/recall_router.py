"""
title: Recall Router
author: Local
version: 0.1.0
required_open_webui_version: 0.6.0
description: Read-only, backend-agnostic recall routing planner. It decides whether stored context should be consulted, which source type should be searched first, and what narrow queries should be used. It does not retrieve, mutate, store, or delete anything.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

JsonDict = dict[str, Any]


SOURCE_CURRENT = "current_context"
SOURCE_ATTACHED = "attached_files"
SOURCE_KNOWLEDGE = "knowledge_base"
SOURCE_MEMORY = "memory"
SOURCE_NOTES = "notes"
SOURCE_PRIOR_CHATS = "prior_chats"
SOURCE_DOMAIN_STATE = "domain_state"
SOURCE_MULTI = "multi_source"
SOURCE_NO_RECALL = "no_recall"

CANONICAL_SOURCES = {
    SOURCE_CURRENT,
    SOURCE_ATTACHED,
    SOURCE_KNOWLEDGE,
    SOURCE_MEMORY,
    SOURCE_NOTES,
    SOURCE_PRIOR_CHATS,
    SOURCE_DOMAIN_STATE,
}

SENSITIVE_SOURCES = {
    SOURCE_MEMORY,
    SOURCE_NOTES,
    SOURCE_PRIOR_CHATS,
}

SOURCE_ALIASES = {
    "current": SOURCE_CURRENT,
    "conversation": SOURCE_CURRENT,
    "visible": SOURCE_CURRENT,
    "visible_context": SOURCE_CURRENT,
    "current_conversation": SOURCE_CURRENT,
    "context": SOURCE_CURRENT,
    "file": SOURCE_ATTACHED,
    "files": SOURCE_ATTACHED,
    "attached": SOURCE_ATTACHED,
    "attachment": SOURCE_ATTACHED,
    "attachments": SOURCE_ATTACHED,
    "attached_file": SOURCE_ATTACHED,
    "attached_files": SOURCE_ATTACHED,
    "uploaded": SOURCE_ATTACHED,
    "upload": SOURCE_ATTACHED,
    "document": SOURCE_ATTACHED,
    "documents": SOURCE_ATTACHED,
    "knowledge": SOURCE_KNOWLEDGE,
    "knowledge_base": SOURCE_KNOWLEDGE,
    "knowledge_bases": SOURCE_KNOWLEDGE,
    "kb": SOURCE_KNOWLEDGE,
    "stored_files": SOURCE_KNOWLEDGE,
    "stored_file": SOURCE_KNOWLEDGE,
    "docs": SOURCE_KNOWLEDGE,
    "documentation": SOURCE_KNOWLEDGE,
    "reference": SOURCE_KNOWLEDGE,
    "memory": SOURCE_MEMORY,
    "memories": SOURCE_MEMORY,
    "remembered": SOURCE_MEMORY,
    "preference": SOURCE_MEMORY,
    "preferences": SOURCE_MEMORY,
    "note": SOURCE_NOTES,
    "notes": SOURCE_NOTES,
    "handoff": SOURCE_NOTES,
    "summary": SOURCE_NOTES,
    "summaries": SOURCE_NOTES,
    "prior_chat": SOURCE_PRIOR_CHATS,
    "prior_chats": SOURCE_PRIOR_CHATS,
    "chat": SOURCE_PRIOR_CHATS,
    "chats": SOURCE_PRIOR_CHATS,
    "history": SOURCE_PRIOR_CHATS,
    "conversation_history": SOURCE_PRIOR_CHATS,
    "previous_chat": SOURCE_PRIOR_CHATS,
    "previous_chats": SOURCE_PRIOR_CHATS,
    "state": SOURCE_DOMAIN_STATE,
    "local_state": SOURCE_DOMAIN_STATE,
    "domain_state": SOURCE_DOMAIN_STATE,
    "character_state": SOURCE_DOMAIN_STATE,
    "relationship_state": SOURCE_DOMAIN_STATE,
    "scene_state": SOURCE_DOMAIN_STATE,
    "campaign_state": SOURCE_DOMAIN_STATE,
    "world_state": SOURCE_DOMAIN_STATE,
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "show",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "to",
    "use",
    "using",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


RECALL_TRIGGER_PATTERNS: list[tuple[str, str, int]] = [
    # Attached/current files.
    (
        SOURCE_ATTACHED,
        r"\b(this|the|attached|uploaded|provided)\s+(file|document|pdf|docx|attachment|upload)\b",
        8,
    ),
    (
        SOURCE_ATTACHED,
        r"\b(attachment|attached file|uploaded file|provided document|this document|this pdf)\b",
        8,
    ),
    (
        SOURCE_ATTACHED,
        r"\b(read|extract|summarize|compare|parse|look at|what does)\s+(this|the)\s+(file|document|pdf|attachment)\b",
        8,
    ),
    (SOURCE_ATTACHED, r"\b(in|from)\s+(the\s+)?(attached|uploaded|provided)\b", 6),
    # Knowledge / source documents.
    (
        SOURCE_KNOWLEDGE,
        r"\b(knowledge base|kb|stored files?|documentation|docs|manual|spec|specification|policy|reference|source[- ]of[- ]truth)\b",
        7,
    ),
    (
        SOURCE_KNOWLEDGE,
        r"\b(what does|according to|find in|look up in)\s+(my|the|our)?\s*(docs|documentation|knowledge|manual|spec|reference)\b",
        8,
    ),
    (
        SOURCE_KNOWLEDGE,
        r"\b(project doc|design doc|technical note|reference material|stored document)\b",
        6,
    ),
    # Memory.
    (
        SOURCE_MEMORY,
        r"\b(my usual|usual preferences?|my preferences?|remember(?:ed)?|you remember|as before|my setup|my environment|my constraints)\b",
        7,
    ),
    (SOURCE_MEMORY, r"\b(always|normally|typically)\s+(?:use|prefer|want|do)\b", 5),
    (
        SOURCE_MEMORY,
        r"\b(durable|persistent|saved)\s+(memory|preference|setting|context)\b",
        7,
    ),
    # Notes.
    (
        SOURCE_NOTES,
        r"\b(notes?|handoff|working summary|project summary|status summary|recap|current plan|standing plan)\b",
        7,
    ),
    (
        SOURCE_NOTES,
        r"\b(where are we|project status|where things stand|current state of the project|what remains)\b",
        8,
    ),
    (SOURCE_NOTES, r"\b(next steps|open tasks|unresolved tasks|todo|to-do)\b", 5),
    # Prior chats.
    (
        SOURCE_PRIOR_CHATS,
        r"\b(earlier|previously|last time|the other day|before|prior chat|previous chat|chat history|conversation history)\b",
        7,
    ),
    (
        SOURCE_PRIOR_CHATS,
        r"\b(what did we decide|we decided|we discussed|we talked about|where did we leave off|continue from before)\b",
        9,
    ),
    (
        SOURCE_PRIOR_CHATS,
        r"\b(the command we used|that command|the issue we made|the plan we made|the patch we wrote)\b",
        8,
    ),
    (
        SOURCE_PRIOR_CHATS,
        r"\b(find|recover|pull up|look for)\s+(that|the|our)\s+(command|discussion|decision|plan|issue|thread|chat)\b",
        8,
    ),
    # Domain-specific local state.
    (
        SOURCE_DOMAIN_STATE,
        r"\b(character state|relationship state|scene state|campaign state|world state|local state|scope)\b",
        8,
    ),
    (
        SOURCE_DOMAIN_STATE,
        r"\b(continue the scene|continue roleplay|continue the rp|where the scene left off|relationship between|character profile)\b",
        8,
    ),
    (
        SOURCE_DOMAIN_STATE,
        r"\b(scene|character|relationship|campaign|worldbuilding)\s+(continuity|state|history)\b",
        7,
    ),
    # Explicit multi-source signals.
    (
        SOURCE_MULTI,
        r"\b(across|compare|reconcile|cross[- ]check)\s+(memory|notes|docs|files|chats|history|knowledge)\b",
        8,
    ),
    (
        SOURCE_MULTI,
        r"\b(memory|notes|docs|files|chats|history|knowledge).*\b(memory|notes|docs|files|chats|history|knowledge)\b",
        6,
    ),
]


NO_RECALL_PATTERNS = [
    r"\b(do not|don't|dont|without)\s+(use|search|look up|check)\s+(memory|notes|chats|history|stored context|prior context|files|knowledge)\b",
    r"\b(no memory|no recall|current context only|visible context only|from this message only)\b",
]


CURRENT_CONTEXT_PATTERNS = [
    r"\b(this answer|your last answer|the last paragraph|that sentence|the above|this snippet|this code|rewrite this|rephrase this)\b",
    r"\b(explain|define|teach|summarize)\s+(this|the above|that)\b",
]


class Tools:
    """
    Backend-agnostic recall routing planner.

    Design intent:
    - Read-only: never retrieves, writes, edits, deletes, or stores context.
    - Portable: no Open WebUI internal imports, no database calls, no HTTP calls.
    - Deterministic: returns a structured plan based on stable routing rules.
    - Safe: flags sensitive source types and avoids broad retrieval by default.
    """

    class Valves(BaseModel):
        max_request_chars: int = Field(
            default=8000,
            ge=500,
            le=100000,
            description="Maximum request characters inspected by the router.",
        )
        max_visible_context_chars: int = Field(
            default=4000,
            ge=0,
            le=50000,
            description="Maximum visible context summary characters inspected by the router.",
        )
        default_max_queries: int = Field(
            default=3,
            ge=1,
            le=8,
            description="Default maximum number of recall search queries to produce.",
        )
        enable_sub_agent_prompt: bool = Field(
            default=True,
            description="Include a self-contained sub-agent prompt when multi-source retrieval is recommended.",
        )
        strict_privacy_by_default: bool = Field(
            default=False,
            description="If true, use strict privacy behavior unless privacy_level is explicitly set.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    # JSON response helpers
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
    # Generic helpers
    # ------------------------------------------------------------------

    def _clamp_int(self, value: Any, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = minimum
        return max(minimum, min(maximum, parsed))

    def _truncate(self, value: Any, limit: int) -> str:
        text = "" if value is None else str(value)
        if limit > 0 and len(text) > limit:
            return text[:limit]
        return text

    def _normalize_text(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\u2019", "'")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _lower(self, value: Any) -> str:
        return self._normalize_text(value).lower()

    def _normalize_source(self, source: Any) -> str:
        raw = self._lower(source)
        if not raw:
            return ""
        raw = raw.replace("-", "_").replace(" ", "_")
        if raw in CANONICAL_SOURCES:
            return raw
        return SOURCE_ALIASES.get(raw, "")

    def _parse_available_sources(
        self, available_sources_json: str
    ) -> tuple[set[str], list[str]]:
        warnings: list[str] = []
        if not available_sources_json.strip():
            return set(CANONICAL_SOURCES), warnings

        try:
            parsed = json.loads(available_sources_json)
        except Exception as exc:
            warnings.append(
                f"available_sources_json could not be parsed as JSON; assuming all source types are available. Error: {exc}"
            )
            return set(CANONICAL_SOURCES), warnings

        raw_sources: list[Any]
        if isinstance(parsed, list):
            raw_sources = parsed
        elif isinstance(parsed, dict):
            value = parsed.get("sources", parsed.get("available_sources", []))
            raw_sources = value if isinstance(value, list) else []
        else:
            warnings.append(
                "available_sources_json must be a list or an object with a 'sources' field; assuming all source types are available."
            )
            return set(CANONICAL_SOURCES), warnings

        normalized: set[str] = {SOURCE_CURRENT}
        unknown: list[str] = []
        for item in raw_sources:
            source = self._normalize_source(item)
            if source:
                normalized.add(source)
            else:
                unknown.append(str(item))

        if unknown:
            warnings.append(f"Ignored unknown source types: {unknown}")

        return normalized, warnings

    def _privacy_level(self, privacy_level: str) -> str:
        raw = self._lower(privacy_level)
        if raw in {"normal", "sensitive", "strict"}:
            return raw
        if self.valves.strict_privacy_by_default:
            return "strict"
        return "normal"

    # ------------------------------------------------------------------
    # Recall classification
    # ------------------------------------------------------------------

    def _matches_any(self, text: str, patterns: list[str]) -> bool:
        return any(
            re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns
        )

    def _score_sources(
        self, request: str, visible_context_summary: str
    ) -> dict[str, int]:
        text = self._lower(f"{request}\n{visible_context_summary}")
        scores = {source: 0 for source in CANONICAL_SOURCES}
        scores[SOURCE_MULTI] = 0
        scores[SOURCE_NO_RECALL] = 0

        for source, pattern, weight in RECALL_TRIGGER_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                scores[source] = scores.get(source, 0) + weight

        if self._matches_any(text, NO_RECALL_PATTERNS):
            scores[SOURCE_NO_RECALL] += 100

        if self._matches_any(text, CURRENT_CONTEXT_PATTERNS):
            scores[SOURCE_CURRENT] += 5

        # If the request is explicitly about "this" and visible context exists,
        # prefer current context unless another source is strongly indicated.
        if visible_context_summary.strip() and re.search(
            r"\b(this|that|above|previous answer|last answer|following)\b",
            text,
            flags=re.IGNORECASE,
        ):
            scores[SOURCE_CURRENT] += 3

        # Generic "file/document" wording is ambiguous. Give both a small score,
        # but let explicit attached/uploaded wording dominate.
        if re.search(r"\b(file|document|pdf|docx)\b", text, flags=re.IGNORECASE):
            scores[SOURCE_ATTACHED] += 2
            scores[SOURCE_KNOWLEDGE] += 1

        # Long-running technical/project work often benefits from notes or prior chats
        # when the user asks to resume without giving visible context.
        if re.search(
            r"\b(resume|continue|pick up|finish|proceed)\b", text, flags=re.IGNORECASE
        ):
            scores[SOURCE_NOTES] += 2
            scores[SOURCE_PRIOR_CHATS] += 2

        return scores

    def _rank_sources(
        self,
        scores: dict[str, int],
        available_sources: set[str],
    ) -> list[str]:
        ranked = [
            source
            for source, score in sorted(
                scores.items(),
                key=lambda item: (-item[1], self._source_tiebreak_order(item[0])),
            )
            if source in CANONICAL_SOURCES and source in available_sources and score > 0
        ]
        return ranked

    def _source_tiebreak_order(self, source: str) -> int:
        order = {
            SOURCE_CURRENT: 0,
            SOURCE_ATTACHED: 1,
            SOURCE_KNOWLEDGE: 2,
            SOURCE_NOTES: 3,
            SOURCE_MEMORY: 4,
            SOURCE_PRIOR_CHATS: 5,
            SOURCE_DOMAIN_STATE: 6,
        }
        return order.get(source, 99)

    def _classify(
        self,
        request: str,
        visible_context_summary: str,
        available_sources: set[str],
        force_recall: bool,
        forbid_recall: bool,
    ) -> JsonDict:
        request = self._truncate(request, int(self.valves.max_request_chars))
        visible_context_summary = self._truncate(
            visible_context_summary, int(self.valves.max_visible_context_chars)
        )

        scores = self._score_sources(request, visible_context_summary)
        ranked = self._rank_sources(scores, available_sources)

        if forbid_recall or scores.get(SOURCE_NO_RECALL, 0) >= 100:
            return {
                "classification": SOURCE_NO_RECALL,
                "needs_recall": False,
                "ranked_sources": [SOURCE_CURRENT],
                "scores": scores,
                "confidence": "high",
                "reason": "The request forbids recall or asks to use only visible/current context.",
            }

        if force_recall and not ranked:
            fallback = self._first_available(
                [SOURCE_MEMORY, SOURCE_NOTES, SOURCE_PRIOR_CHATS, SOURCE_KNOWLEDGE],
                available_sources,
                default=SOURCE_CURRENT,
            )
            return {
                "classification": fallback,
                "needs_recall": fallback != SOURCE_CURRENT,
                "ranked_sources": [fallback],
                "scores": scores,
                "confidence": "low",
                "reason": "Recall was forced, but the request did not clearly identify a source type.",
            }

        if not ranked:
            return {
                "classification": SOURCE_CURRENT,
                "needs_recall": False,
                "ranked_sources": [SOURCE_CURRENT],
                "scores": scores,
                "confidence": "medium",
                "reason": "No strong stored-context trigger was detected; visible/current context is the default.",
            }

        top = ranked[0]
        top_score = scores.get(top, 0)
        meaningful = [
            source
            for source in ranked
            if scores.get(source, 0) >= max(4, top_score - 2)
        ]

        explicit_multi = scores.get(SOURCE_MULTI, 0) >= 6
        if explicit_multi and len(meaningful) >= 2:
            return {
                "classification": SOURCE_MULTI,
                "needs_recall": True,
                "ranked_sources": meaningful,
                "scores": scores,
                "confidence": "medium",
                "reason": "The request appears to require cross-source recall or reconciliation.",
            }

        if len(meaningful) >= 3 and top_score < 9:
            return {
                "classification": SOURCE_MULTI,
                "needs_recall": True,
                "ranked_sources": meaningful,
                "scores": scores,
                "confidence": "low",
                "reason": "Several source types appear plausible, but none is dominant.",
            }

        return {
            "classification": top,
            "needs_recall": top != SOURCE_CURRENT,
            "ranked_sources": ranked,
            "scores": scores,
            "confidence": "high" if top_score >= 8 else "medium",
            "reason": f"The request most strongly indicates {top}.",
        }

    def _first_available(
        self,
        candidates: list[str],
        available_sources: set[str],
        default: str,
    ) -> str:
        for source in candidates:
            if source in available_sources:
                return source
        return default

    # ------------------------------------------------------------------
    # Route construction
    # ------------------------------------------------------------------

    def _fallback_chain(self, source: str) -> list[str]:
        chains = {
            SOURCE_CURRENT: [],
            SOURCE_ATTACHED: [SOURCE_KNOWLEDGE, SOURCE_NOTES, SOURCE_PRIOR_CHATS],
            SOURCE_KNOWLEDGE: [SOURCE_ATTACHED, SOURCE_NOTES, SOURCE_PRIOR_CHATS],
            SOURCE_NOTES: [SOURCE_PRIOR_CHATS, SOURCE_MEMORY, SOURCE_KNOWLEDGE],
            SOURCE_MEMORY: [SOURCE_NOTES, SOURCE_PRIOR_CHATS, SOURCE_KNOWLEDGE],
            SOURCE_PRIOR_CHATS: [SOURCE_NOTES, SOURCE_MEMORY, SOURCE_KNOWLEDGE],
            SOURCE_DOMAIN_STATE: [SOURCE_NOTES, SOURCE_PRIOR_CHATS, SOURCE_MEMORY],
        }
        return chains.get(source, [])

    def _build_route(
        self,
        classification: JsonDict,
        available_sources: set[str],
    ) -> tuple[str, list[str], list[str]]:
        kind = classification.get("classification")
        ranked = [
            source
            for source in classification.get("ranked_sources", [])
            if source in CANONICAL_SOURCES and source in available_sources
        ]

        if kind in {SOURCE_NO_RECALL, SOURCE_CURRENT} or not classification.get(
            "needs_recall"
        ):
            return SOURCE_CURRENT, [], [SOURCE_CURRENT]

        if kind == SOURCE_MULTI:
            route = []
            for source in ranked:
                if source not in route:
                    route.append(source)
            if not route:
                route = [
                    source
                    for source in [
                        SOURCE_NOTES,
                        SOURCE_KNOWLEDGE,
                        SOURCE_PRIOR_CHATS,
                        SOURCE_MEMORY,
                    ]
                    if source in available_sources
                ]
            primary = route[0] if route else SOURCE_CURRENT
            return primary, route[1:], route

        primary = ranked[0] if ranked else SOURCE_CURRENT
        route = [primary]
        for source in self._fallback_chain(primary):
            if source in available_sources and source not in route:
                route.append(source)

        return primary, route[1:], route

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def _extract_quoted_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        patterns = [
            r'"([^"]{2,120})"',
            r"'([^']{2,120})'",
            r"`([^`]{2,160})`",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = self._normalize_text(match.group(1))
                if value and value not in terms:
                    terms.append(value)
        return terms[:8]

    def _extract_codeish_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        pattern = r"(?<!\w)([A-Za-z0-9_./:-]{3,80})(?!\w)"
        for match in re.finditer(pattern, text):
            token = match.group(1).strip(".,;:()[]{}<>")
            if not token:
                continue
            if (
                "/" in token
                or "." in token
                or "_" in token
                or "-" in token
                or ":" in token
            ):
                lowered = token.lower()
                if lowered not in STOPWORDS and token not in terms:
                    terms.append(token)
        return terms[:10]

    def _extract_keywords(self, text: str) -> list[str]:
        cleaned = self._normalize_text(text)
        cleaned = re.sub(r"[^\w\s./:#@+-]", " ", cleaned)
        tokens = re.findall(r"[A-Za-z0-9_./:#@+-]{3,80}", cleaned)

        out: list[str] = []
        for token in tokens:
            stripped = token.strip(".,;:()[]{}<>").lower()
            if not stripped or stripped in STOPWORDS:
                continue
            if stripped in {
                "earlier",
                "previously",
                "before",
                "attached",
                "uploaded",
                "provided",
                "document",
                "file",
                "files",
                "memory",
                "notes",
                "chat",
                "chats",
                "history",
                "recall",
                "remember",
            }:
                continue
            if stripped not in out:
                out.append(stripped)

        return out[:16]

    def _substantive_query_base(self, request: str) -> str:
        quoted = self._extract_quoted_terms(request)
        codeish = self._extract_codeish_terms(request)
        keywords = self._extract_keywords(request)

        parts: list[str] = []
        for group in (quoted, codeish, keywords):
            for item in group:
                if item not in parts:
                    parts.append(item)

        if not parts:
            return self._normalize_text(request)[:200]

        return " ".join(parts[:10]).strip()

    def _source_query_variants(
        self, request: str, source: str, max_queries: int
    ) -> list[str]:
        base = self._substantive_query_base(request)
        if not base:
            base = self._normalize_text(request)[:200]

        variants: list[str] = []

        def add(value: str) -> None:
            value = self._normalize_text(value)
            if value and value not in variants:
                variants.append(value[:240])

        add(base)

        if source == SOURCE_ATTACHED:
            add(f"{base} section heading table exact wording")
            add(f"{base} requirement command config")
        elif source == SOURCE_KNOWLEDGE:
            add(f"{base} documentation spec reference")
            add(f"{base} design note policy")
        elif source == SOURCE_MEMORY:
            add(f"{base} user preference setup constraint")
            add(f"{base} usual environment recurring")
        elif source == SOURCE_NOTES:
            add(f"{base} project status plan next steps")
            add(f"{base} summary handoff open tasks")
        elif source == SOURCE_PRIOR_CHATS:
            add(f"{base} decision earlier previous discussion")
            add(f"{base} command issue plan last time")
        elif source == SOURCE_DOMAIN_STATE:
            add(f"{base} character relationship scene state")
            add(f"{base} continuity current state recent event")
        else:
            add(base)

        return variants[:max_queries]

    def _build_queries_for_route(
        self,
        request: str,
        route: list[str],
        max_queries: int,
    ) -> dict[str, list[str]]:
        queries: dict[str, list[str]] = {}
        for source in route:
            if source == SOURCE_CURRENT:
                continue
            queries[source] = self._source_query_variants(request, source, max_queries)
        return queries

    # ------------------------------------------------------------------
    # Notes and prompt construction
    # ------------------------------------------------------------------

    def _security_notes(self, route: list[str], privacy_level: str) -> list[str]:
        notes: list[str] = []

        if any(source in SENSITIVE_SOURCES for source in route):
            notes.append(
                "Retrieve only entries directly relevant to the request; do not expose unrelated memories, notes, or prior chat contents."
            )

        if SOURCE_MEMORY in route:
            notes.append(
                "Treat memory as durable context, not as source-of-truth evidence. Prefer source documents when they conflict with memory."
            )

        if SOURCE_PRIOR_CHATS in route:
            notes.append(
                "Extract only decisions, commands, plans, or unresolved issues needed for the current request."
            )

        if SOURCE_NOTES in route:
            notes.append(
                "Treat notes as curated working context; verify against source documents when exact wording or technical details matter."
            )

        if privacy_level in {"sensitive", "strict"}:
            notes.append(
                "Use a narrow query and summarize only the minimum necessary findings."
            )

        if privacy_level == "strict":
            notes.append(
                "Do not reveal raw stored entries unless the user explicitly asks for them and they are directly relevant."
            )

        if not notes:
            notes.append(
                "No sensitive stored source is required by the proposed route."
            )

        return notes

    def _performance_notes(self, route: list[str], max_queries: int) -> list[str]:
        notes = [
            f"Use at most {max_queries} narrow query or query variants per source.",
            "Stop after enough relevant evidence is found.",
        ]

        if SOURCE_KNOWLEDGE in route or SOURCE_ATTACHED in route:
            notes.append(
                "Prefer targeted or chunked retrieval for large documents; do not force full-context loading unless the user explicitly asks."
            )

        if len(route) > 2:
            notes.append(
                "Search sources in order and avoid querying fallback sources if the primary source answers the request."
            )

        return notes

    def _stop_condition(self, primary_source: str, route: list[str]) -> str:
        if primary_source == SOURCE_CURRENT:
            return "No stored recall is needed; answer from the visible current conversation."
        if primary_source == SOURCE_ATTACHED:
            return "Stop after finding the relevant section, table, heading, command, or passage in the attached/current file."
        if primary_source == SOURCE_KNOWLEDGE:
            return "Stop after finding source-document support that directly answers the request."
        if primary_source == SOURCE_MEMORY:
            return "Stop after finding the smallest durable preference, setup fact, or recurring constraint relevant to the request."
        if primary_source == SOURCE_NOTES:
            return "Stop after finding the relevant curated status, plan, summary, or handoff note."
        if primary_source == SOURCE_PRIOR_CHATS:
            return "Stop after finding the most recent relevant prior discussion containing the decision, command, plan, or unresolved issue."
        if primary_source == SOURCE_DOMAIN_STATE:
            return "Stop after finding the relevant local state record for the scene, character, relationship, campaign, or domain."
        return "Stop after enough directly relevant context is found."

    def _sub_agent_recommended(self, classification: str, route: list[str]) -> bool:
        if classification == SOURCE_MULTI:
            return len(route) >= 2
        return len([source for source in route if source != SOURCE_CURRENT]) >= 3

    def _build_sub_agent_prompt(
        self,
        request: str,
        route: list[str],
        queries_by_source: dict[str, list[str]],
        privacy_level: str,
    ) -> str:
        if not self.valves.enable_sub_agent_prompt:
            return ""

        source_lines = []
        for source in route:
            if source == SOURCE_CURRENT:
                continue
            queries = queries_by_source.get(source, [])
            source_lines.append(
                f"- {source}: {', '.join(queries) if queries else '(use the user request as the query)'}"
            )

        return (
            "Task: perform narrow stored-context recall for the user's request.\n\n"
            f"User request:\n{request.strip()}\n\n"
            "Search sources in this order:\n" + "\n".join(source_lines) + "\n\n"
            "Rules:\n"
            "- Search only the listed sources.\n"
            "- Stop once enough directly relevant evidence is found.\n"
            "- Return only relevant findings, source labels, and uncertainty.\n"
            "- Do not expose unrelated memories, notes, chats, or file contents.\n"
            "- Do not write, replace, delete, or mutate any stored state.\n"
            f"- Privacy level: {privacy_level}."
        ).strip()

    # ------------------------------------------------------------------
    # Public Open WebUI tool methods
    # ------------------------------------------------------------------

    async def classify_recall_need(
        self,
        request: str = Field(
            ...,
            description="The user's request to classify for recall need.",
        ),
        visible_context_summary: str = Field(
            default="",
            description="Optional short summary of what is already visible in the current conversation.",
        ),
        available_sources_json: str = Field(
            default="",
            description=(
                "Optional JSON list of available source types, or object with a 'sources' field. "
                'Examples: ["memory", "notes", "prior_chats"] or {"sources": ["knowledge_base"]}. '
                "Empty means all source types are assumed available."
            ),
        ),
        force_recall: bool = Field(
            default=False,
            description="Force the router to choose a stored recall source even if the request does not strongly require one.",
        ),
        forbid_recall: bool = Field(
            default=False,
            description="Force current-context-only behavior.",
        ),
    ) -> str:
        """
        Classify whether the request needs stored recall and identify the most likely source type.
        This does not retrieve anything.
        """
        try:
            available_sources, warnings = self._parse_available_sources(
                available_sources_json
            )
            classification = self._classify(
                request=request,
                visible_context_summary=visible_context_summary,
                available_sources=available_sources,
                force_recall=force_recall,
                forbid_recall=forbid_recall,
            )
            return self._ok(
                {
                    **classification,
                    "available_sources": sorted(available_sources),
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            return self._err("classification_failed", str(exc))

    async def build_recall_queries(
        self,
        request: str = Field(
            ...,
            description="The user's request to convert into narrow recall search queries.",
        ),
        target_source_type: str = Field(
            ...,
            description=(
                "Target source type. Supported values: current_context, attached_files, "
                "knowledge_base, memory, notes, prior_chats, domain_state."
            ),
        ),
        max_queries: int = Field(
            default=3,
            ge=1,
            le=8,
            description="Maximum query variants to return.",
        ),
    ) -> str:
        """
        Build narrow search queries for one target source type.
        This does not retrieve anything.
        """
        try:
            source = self._normalize_source(target_source_type)
            if source not in CANONICAL_SOURCES:
                return self._err(
                    "invalid_source_type",
                    f"Unsupported target_source_type: {target_source_type}",
                    supported_sources=sorted(CANONICAL_SOURCES),
                )

            max_queries = self._clamp_int(
                max_queries, 1, int(self.valves.default_max_queries)
            )
            queries = (
                []
                if source == SOURCE_CURRENT
                else self._source_query_variants(request, source, max_queries)
            )

            return self._ok(
                {
                    "source": source,
                    "queries": queries,
                    "query_count": len(queries),
                    "note": "This function only builds query strings; it does not retrieve stored context.",
                }
            )
        except Exception as exc:
            return self._err("query_build_failed", str(exc))

    async def route_recall_request(
        self,
        request: str = Field(
            ...,
            description="The user's actual request.",
        ),
        visible_context_summary: str = Field(
            default="",
            description="Optional short summary of what is already visible in the current conversation.",
        ),
        available_sources_json: str = Field(
            default="",
            description=(
                "Optional JSON list of available source types, or object with a 'sources' field. "
                "Supported source types: current_context, attached_files, knowledge_base, memory, notes, prior_chats, domain_state. "
                "Empty means all source types are assumed available."
            ),
        ),
        force_recall: bool = Field(
            default=False,
            description="Force the router to choose a stored recall route even if no strong recall trigger is detected.",
        ),
        forbid_recall: bool = Field(
            default=False,
            description="Force current-context-only behavior.",
        ),
        max_queries: int = Field(
            default=3,
            ge=1,
            le=8,
            description="Maximum query variants to produce per source.",
        ),
        privacy_level: str = Field(
            default="normal",
            description="Privacy posture for the route: normal, sensitive, or strict.",
        ),
    ) -> str:
        """
        Return a full backend-agnostic recall routing plan.

        The plan tells the model or caller where to look first, what query strings
        to use, when to stop, and what privacy/performance limits apply.

        This function is read-only and does not retrieve or mutate any stored context.
        """
        try:
            request = self._truncate(request, int(self.valves.max_request_chars))
            visible_context_summary = self._truncate(
                visible_context_summary, int(self.valves.max_visible_context_chars)
            )
            max_queries = self._clamp_int(max_queries, 1, 8)
            privacy = self._privacy_level(privacy_level)

            available_sources, warnings = self._parse_available_sources(
                available_sources_json
            )
            classification = self._classify(
                request=request,
                visible_context_summary=visible_context_summary,
                available_sources=available_sources,
                force_recall=force_recall,
                forbid_recall=forbid_recall,
            )

            primary_source, fallback_sources, route = self._build_route(
                classification, available_sources
            )
            queries_by_source = self._build_queries_for_route(
                request=request,
                route=route,
                max_queries=max_queries,
            )

            sub_agent_recommended = self._sub_agent_recommended(
                str(classification.get("classification", "")),
                route,
            )
            sub_agent_prompt = (
                self._build_sub_agent_prompt(
                    request=request,
                    route=route,
                    queries_by_source=queries_by_source,
                    privacy_level=privacy,
                )
                if sub_agent_recommended
                else ""
            )

            plan = {
                "needs_recall": bool(classification.get("needs_recall")),
                "classification": classification.get("classification"),
                "primary_source": primary_source,
                "fallback_sources": fallback_sources,
                "recommended_route": route,
                "queries_by_source": queries_by_source,
                "stop_condition": self._stop_condition(primary_source, route),
                "sub_agent_recommended": sub_agent_recommended,
                "sub_agent_prompt": sub_agent_prompt,
                "confidence": classification.get("confidence"),
                "reason": classification.get("reason"),
                "security_notes": self._security_notes(route, privacy),
                "performance_notes": self._performance_notes(route, max_queries),
                "available_sources": sorted(available_sources),
                "warnings": warnings,
                "read_only": True,
                "mutation_allowed": False,
            }

            return self._ok(plan)
        except Exception as exc:
            return self._err("routing_failed", str(exc))
