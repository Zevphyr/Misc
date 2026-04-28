"""
title: Image Search & Inline Display
author: Local
version: 0.5.0
requirements: requests
description: Search for an image through SearXNG or DDGS, return inline Markdown, optionally cache it in Open WebUI temporarily, and clean expired cached files.
"""

from __future__ import annotations

import html
import ipaddress
import json
import mimetypes
import os
import re
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import (
    parse_qsl,
    quote,
    quote_plus,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunsplit,
)

import requests
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS as _DDGS
    _DDGS_IMPORT_ERROR = ""
except Exception as ddgs_error:  # pragma: no cover - compatibility fallback
    try:
        from duckduckgo_search import DDGS as _DDGS  # type: ignore
        _DDGS_IMPORT_ERROR = ""
    except Exception as legacy_error:  # pragma: no cover
        _DDGS = None  # type: ignore[assignment]
        _DDGS_IMPORT_ERROR = (
            "Could not import DDGS from either 'ddgs' or 'duckduckgo_search'. "
            f"ddgs error: {type(ddgs_error).__name__}: {ddgs_error}; "
            f"duckduckgo_search error: {type(legacy_error).__name__}: {legacy_error}"
        )


@dataclass(frozen=True)
class ImageCandidate:
    title: str
    image_url: str
    thumbnail_url: str
    page_url: str
    source: str
    width: int
    height: int
    score: int


class Tools:
    class Valves(BaseModel):
        # Open WebUI / delivery behavior
        openwebui_base_url: str = Field(
            default="",
            description=(
                "Leave blank to infer from the current request. Set this only when "
                "the inferred base URL cannot reach the Open WebUI file API."
            ),
        )
        default_mode: str = Field(
            default="remote",
            description="Default image delivery mode.",
            json_schema_extra={
                "input": {"type": "select", "options": ["remote", "cache"]}
            },
        )
        cache_ttl_hours: int = Field(
            default=24,
            description="How long cached Open WebUI image files should be kept.",
        )
        cache_registry_dir: str = Field(
            default="/app/backend/data/cache/image_tool",
            description="Where registry metadata for cached files is stored.",
        )
        cleanup_expired_on_search: bool = Field(
            default=True,
            description="Delete expired cached files at the start of image searches.",
        )
        max_download_mb: int = Field(
            default=12,
            description="Maximum image size to download in cache mode.",
        )
        verify_ssl: bool = Field(
            default=True,
            description="Verify TLS certificates for outbound HTTP requests.",
        )
        user_agent: str = Field(
            default="OpenWebUI-ImageTool/0.5",
            description="HTTP User-Agent header for remote fetches.",
        )

        # Search provider behavior
        search_provider: str = Field(
            default="auto",
            description=(
                "Search backend to use. 'auto' uses SearXNG when a SearXNG URL is configured, "
                "then falls back to DDGS. 'searxng' keeps search traffic centralized through "
                "your own SearXNG instance. 'ddgs' uses DDGS directly from the Open WebUI process."
            ),
            json_schema_extra={
                "input": {"type": "select", "options": ["auto", "searxng", "ddgs"]}
            },
        )
        fallback_to_ddgs_on_searxng_failure: bool = Field(
            default=False,
            description=(
                "When search_provider='searxng', fall back to DDGS if SearXNG fails. "
                "Keep false if you want to avoid direct DDGS traffic from this tool."
            ),
        )

        # SearXNG behavior
        searxng_query_url: str = Field(
            default="",
            description=(
                "SearXNG search URL. Supports Open WebUI-style '<query>' placeholders, "
                "for example 'http://searxng:8080/search?q=<query>'. If blank, the tool "
                "checks SEARXNG_QUERY_URL and then searxng_base_url."
            ),
        )
        searxng_base_url: str = Field(
            default="",
            description=(
                "SearXNG base URL used when searxng_query_url is blank, e.g. "
                "'http://searxng:8080'. The tool appends '/search'."
            ),
        )
        searxng_categories: str = Field(
            default="images",
            description="SearXNG categories to request. Use 'images' for image search.",
        )
        searxng_engines: str = Field(
            default="",
            description="Optional comma-separated SearXNG engines. Blank uses the instance defaults.",
        )
        searxng_language: str = Field(
            default="all",
            description="SearXNG search language, e.g. all, en, de, fr.",
        )
        searxng_safesearch: str = Field(
            default="1",
            description="SearXNG safesearch: 0=off, 1=moderate, 2=strict.",
            json_schema_extra={
                "input": {"type": "select", "options": ["0", "1", "2"]}
            },
        )
        searxng_time_range: str = Field(
            default="",
            description="Optional SearXNG time range: day, month, year, or blank.",
        )
        searxng_image_proxy: bool = Field(
            default=True,
            description=(
                "Ask SearXNG to proxy image result URLs when the instance supports it. "
                "This can reduce direct client exposure to third-party image hosts."
            ),
        )
        searxng_timeout_seconds: int = Field(
            default=15,
            description="Timeout for SearXNG API requests.",
        )
        allow_internal_search_service_urls: bool = Field(
            default=True,
            description=(
                "Allow the configured search provider URL to point at a private/container "
                "network address. This is needed for local SearXNG, but only admins should "
                "control the configured URL."
            ),
        )

        # DDGS behavior
        ddgs_backend: str = Field(
            default="auto",
            description=(
                "DDGS image backend. Current DDGS supports auto, duckduckgo, bing, "
                "or a comma-delimited backend list."
            ),
            json_schema_extra={
                "input": {"type": "select", "options": ["auto", "duckduckgo", "bing"]}
            },
        )
        ddgs_region: str = Field(
            default="us-en",
            description="DDGS region, e.g. us-en, wt-wt, uk-en.",
        )
        ddgs_safesearch: str = Field(
            default="moderate",
            description="DDGS safesearch level.",
            json_schema_extra={
                "input": {"type": "select", "options": ["on", "moderate", "off"]}
            },
        )
        ddgs_timelimit: str = Field(
            default="",
            description="Optional DDGS timelimit: d, w, m, y, or blank.",
        )
        ddgs_timeout_seconds: int = Field(
            default=15,
            description="DDGS client timeout in seconds.",
        )
        ddgs_proxy: str = Field(
            default="",
            description="Optional proxy URL for DDGS and requests.",
        )
        search_result_limit: int = Field(
            default=12,
            description="How many image results to retrieve before ranking.",
        )
        image_size: str = Field(
            default="",
            description="Optional DDGS image size filter: Small, Medium, Large, Wallpaper, or blank.",
        )
        image_color: str = Field(
            default="",
            description="Optional DDGS image color filter.",
        )
        image_type: str = Field(
            default="",
            description="Optional DDGS image type filter: photo, clipart, gif, transparent, line.",
        )
        image_layout: str = Field(
            default="",
            description="Optional DDGS image layout filter: Square, Tall, Wide.",
        )
        image_license: str = Field(
            default="",
            description="Optional DDGS image license filter.",
        )

        # Candidate selection and resilience
        use_page_meta_image_fallback: bool = Field(
            default=True,
            description="If direct image URLs fail, try og:image/twitter:image from the source page.",
        )
        allow_thumbnail_fallback: bool = Field(
            default=True,
            description="Allow thumbnail URLs when no full image URL is usable.",
        )
        skip_non_image_candidates: bool = Field(
            default=True,
            description="Skip candidates whose URLs do not return an allowed image Content-Type.",
        )
        max_candidate_attempts: int = Field(
            default=5,
            description="How many ranked candidates to test before giving up.",
        )
        probe_candidates_during_selection: bool = Field(
            default=True,
            description="Probe candidates during selection so HTML pages and dead links are skipped.",
        )
        remote_mode_validate_url: bool = Field(
            default=False,
            description="In remote mode, re-probe the chosen image URL before returning it.",
        )
        remote_mode_probe_timeout_seconds: int = Field(
            default=8,
            description="Timeout for remote-mode URL validation.",
        )
        page_meta_timeout_seconds: int = Field(
            default=12,
            description="Timeout for fetching source HTML pages to inspect og:image/twitter:image.",
        )
        page_meta_max_bytes: int = Field(
            default=524288,
            description="Maximum HTML bytes to read while looking for page meta images.",
        )

        # Fallback behavior
        fallback_to_cache_on_remote_validation_failure: bool = Field(
            default=False,
            description="If remote validation fails, fall back to cache mode instead of failing.",
        )
        fallback_to_remote_on_cache_failure: bool = Field(
            default=True,
            description="If cache mode fails to upload/store, fall back to remote mode.",
        )

        # Query variation behavior
        query_variant_retry: bool = Field(
            default=True,
            description="Try lightweight query variants when the first search returns no usable candidates.",
        )
        max_query_variants: int = Field(
            default=4,
            description="Maximum number of query variants to try.",
        )
        query_phrase_variants_json: str = Field(
            default=json.dumps(
                {
                    "official art": ["official artwork", "character art", "wallpaper"],
                    "character art": ["official art", "wallpaper"],
                },
                ensure_ascii=False,
            ),
            description=(
                "JSON object mapping trigger phrases to replacement phrases for query variants. "
                'Example: {"official art": ["official artwork", "character art", "wallpaper"]}'
            ),
        )
        query_disambiguation_suffixes_json: str = Field(
            default="{}",
            description=(
                "Optional JSON object mapping ambiguous terms to suffixes appended as extra variants "
                'when the suffix is not already present. Example: {"nahida": "Genshin Impact", "2b": "NieR Automata"}'
            ),
        )

        # Optional domain steering
        preferred_source_domains: str = Field(
            default="",
            description="Comma-separated preferred domains for source pages or image hosts.",
        )
        blocked_source_domains: str = Field(
            default="",
            description="Comma-separated blocked domains for source pages or image hosts.",
        )

        # Security controls for outbound fetches
        allowed_url_schemes: str = Field(
            default="https,http",
            description="Comma-separated URL schemes allowed for fetched image/source URLs.",
        )
        allow_private_network_urls: bool = Field(
            default=False,
            description=(
                "Allow image/source URLs that resolve to loopback, private, link-local, "
                "reserved, multicast, or unspecified IP addresses. Keep false unless you trust all users."
            ),
        )
        allowed_image_content_types: str = Field(
            default="image/jpeg,image/png,image/webp,image/gif",
            description=(
                "Comma-separated image Content-Types allowed for rendered/cached images. "
                "Blank allows any image/* type. SVG is intentionally not included by default."
            ),
        )
        forward_auth_headers_to_openwebui: bool = Field(
            default=True,
            description=(
                "Forward the current request Authorization/Cookie headers to the configured "
                "Open WebUI file API. Disable if openwebui_base_url points outside your trusted deployment."
            ),
        )

        # Response / debug behavior
        include_source_link: bool = Field(
            default=True,
            description="Include the source page link beneath the image markdown.",
        )
        include_source_name: bool = Field(
            default=True,
            description="Include the source/provider name if available.",
        )
        debug_mode: bool = Field(
            default=True,
            description="Collect diagnostic information while the tool runs.",
        )
        include_debug_in_response: bool = Field(
            default=False,
            description="Append a readable debug report to normal tool responses.",
        )
        max_debug_chars: int = Field(
            default=12000,
            description="Maximum characters of debug JSON to include in responses.",
        )

    class UserValves(BaseModel):
        preferred_mode: str = Field(
            default="",
            description="Optional per-user override.",
            json_schema_extra={
                "input": {"type": "select", "options": ["", "remote", "cache"]}
            },
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.citation = False

    # ---------- path / registry ----------

    def _registry_path(self) -> Path:
        base = Path(self.valves.cache_registry_dir).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        return base / "registry.json"

    def _load_registry(self) -> dict[str, Any]:
        path = self._registry_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        path = self._registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True)

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp_name = tmp.name

        Path(tmp_name).replace(path)

    # ---------- generic helpers ----------

    def _headers(self, accept: str = "*/*") -> dict[str, str]:
        return {
            "User-Agent": self.valves.user_agent.strip() or "OpenWebUI-ImageTool/0.5",
            "Accept": accept,
        }

    def _proxy(self) -> Optional[str]:
        proxy = self.valves.ddgs_proxy.strip()
        return proxy or None

    def _requests_proxies(self) -> Optional[dict[str, str]]:
        proxy = self._proxy()
        if not proxy:
            return None
        return {"http": proxy, "https": proxy}

    def _safe_search_provider(self) -> str:
        provider = str(self.valves.search_provider or "auto").strip().lower()
        return provider if provider in {"auto", "searxng", "ddgs"} else "auto"

    def _searxng_configured(self) -> bool:
        return bool(
            str(self.valves.searxng_query_url or "").strip()
            or str(self.valves.searxng_base_url or "").strip()
            or os.environ.get("SEARXNG_QUERY_URL", "").strip()
            or os.environ.get("SEARXNG_BASE_URL", "").strip()
        )

    def _searxng_endpoint_template(self) -> str:
        configured = str(self.valves.searxng_query_url or "").strip()
        configured = configured or os.environ.get("SEARXNG_QUERY_URL", "").strip()
        if configured:
            return configured

        base = str(self.valves.searxng_base_url or "").strip()
        base = base or os.environ.get("SEARXNG_BASE_URL", "").strip()
        if not base:
            return ""

        return base.rstrip("/") + "/search"

    def _searxng_public_base_url(self) -> str:
        # Optional escape hatch for deployments where the search URL is an
        # internal container name but returned image-proxy URLs need a browser-
        # reachable base. This intentionally reads only an environment variable
        # to avoid adding another Open WebUI valve unless needed.
        return os.environ.get("SEARXNG_PUBLIC_BASE_URL", "").strip().rstrip("/")

    def _search_service_url_allowed(self, url: str) -> tuple[bool, str]:
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Search provider URL parse failed: {type(e).__name__}: {e}"

        if parsed.scheme.lower() not in {"http", "https"}:
            return False, f"Search provider URL scheme is not allowed: {parsed.scheme or '(missing)'}"
        if not parsed.hostname:
            return False, "Search provider URL is missing a host."
        if parsed.username or parsed.password:
            return False, "Search provider URL userinfo is not allowed."

        if self.valves.allow_internal_search_service_urls:
            return True, "ok"

        privateish, detail = self._host_resolves_to_private_address(parsed.hostname)
        if privateish:
            return False, f"Search provider URL host resolves to a blocked address ({detail})."
        return True, "ok"

    def _append_missing_query_params(
        self,
        url: str,
        params: dict[str, str],
    ) -> str:
        parts = urlsplit(url)
        existing = parse_qsl(parts.query, keep_blank_values=True)
        existing_names = {key for key, _ in existing}
        combined = list(existing)

        for key, value in params.items():
            if value == "" or key in existing_names:
                continue
            combined.append((key, value))

        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(combined, doseq=True),
                parts.fragment,
            )
        )

    def _searxng_search_url(self, query: str) -> str:
        template = self._searxng_endpoint_template()
        if not template:
            raise RuntimeError(
                "SearXNG is selected but no SearXNG URL is configured. Set "
                "searxng_query_url, searxng_base_url, SEARXNG_QUERY_URL, or SEARXNG_BASE_URL."
            )

        if "<query>" in template:
            url = template.replace("<query>", quote_plus(query))
            has_query_placeholder = True
        else:
            url = template
            has_query_placeholder = False

        safe = str(self.valves.searxng_safesearch or "1").strip()
        if safe not in {"0", "1", "2"}:
            safe = "1"

        params: dict[str, str] = {
            "format": "json",
            "categories": str(self.valves.searxng_categories or "images").strip() or "images",
            "language": str(self.valves.searxng_language or "all").strip() or "all",
            "safesearch": safe,
            "image_proxy": "true" if self.valves.searxng_image_proxy else "false",
        }

        if not has_query_placeholder:
            params["q"] = query

        engines = str(self.valves.searxng_engines or "").strip()
        if engines:
            params["engines"] = engines

        time_range = str(self.valves.searxng_time_range or "").strip()
        if time_range:
            params["time_range"] = time_range

        return self._append_missing_query_params(url, params)

    def _url_origin(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            pass
        return ""

    def _make_absolute_result_url(self, value: str, response_url: str) -> str:
        value = str(value or "").strip()
        if not value:
            return ""

        public_base = self._searxng_public_base_url()
        if public_base and value.startswith("/"):
            return urljoin(public_base + "/", value.lstrip("/"))

        base = self._url_origin(response_url) or response_url
        return urljoin(base.rstrip("/") + "/", value)

    def _configured_searxng_origins(self) -> set[str]:
        origins: set[str] = set()
        for value in [
            self._searxng_endpoint_template(),
            str(self.valves.searxng_base_url or "").strip(),
            os.environ.get("SEARXNG_BASE_URL", "").strip(),
            os.environ.get("SEARXNG_PUBLIC_BASE_URL", "").strip(),
        ]:
            if value:
                origin = self._url_origin(value)
                if origin:
                    origins.add(origin.lower())
        return origins

    def _is_configured_searxng_origin(self, url: str) -> bool:
        origin = self._url_origin(url).lower()
        return bool(origin and origin in self._configured_searxng_origins())

    def _base_url_from_request(self, __request__=None) -> str:
        if __request__ is None:
            return ""
        try:
            return str(__request__.base_url).rstrip("/")
        except Exception:
            return ""

    def _owui_base_url(self, __request__=None) -> str:
        configured = self.valves.openwebui_base_url.strip()
        if configured:
            return configured.rstrip("/")

        inferred = self._base_url_from_request(__request__)
        if inferred:
            return inferred

        return "http://127.0.0.1:8080"

    def _auth_headers(self, __request__=None) -> dict[str, str]:
        if not self.valves.forward_auth_headers_to_openwebui or __request__ is None:
            return {}

        headers: dict[str, str] = {}

        try:
            auth = __request__.headers.get("authorization")
            cookie = __request__.headers.get("cookie")
        except Exception:
            return headers

        if auth:
            headers["Authorization"] = auth
        if cookie:
            headers["Cookie"] = cookie

        return headers

    def _load_json_dict(self, raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _dedupe_keep_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []

        for item in items:
            value = str(item).strip()
            key = value.lower()
            if key and key not in seen:
                seen.add(key)
                output.append(value)

        return output

    async def _emit(
        self,
        __event_emitter__=None,
        description: str = "",
        done: bool = False,
    ) -> None:
        if not __event_emitter__:
            return

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                    "hidden": False,
                },
            }
        )

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _bounded_positive_int(self, value: Any, default: int, minimum: int = 1) -> int:
        parsed = self._safe_int(value, default=default)
        return max(minimum, parsed)

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9]+", "-", text)
        return text.strip("-")[:96] or "image"

    def _parse_csv(self, raw: str) -> list[str]:
        return [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]

    def _host(self, value: str) -> str:
        try:
            parsed = urlparse(value)
            host = parsed.hostname or ""
            return host.lower().strip(".")
        except Exception:
            return ""

    def _domain_matches(self, host: str, domains: list[str]) -> bool:
        host = host.lower().strip(".")
        normalized = [d.lower().strip().lstrip(".") for d in domains if d.strip()]
        return any(host == d or host.endswith(f".{d}") for d in normalized)

    def _is_blocked(self, page_url: str, image_url: str) -> bool:
        blocked = self._parse_csv(self.valves.blocked_source_domains)
        if not blocked:
            return False

        return self._domain_matches(self._host(page_url), blocked) or self._domain_matches(
            self._host(image_url), blocked
        )

    def _preference_bonus(self, page_url: str, image_url: str) -> int:
        preferred = self._parse_csv(self.valves.preferred_source_domains)
        if not preferred:
            return 0

        bonus = 0
        if self._domain_matches(self._host(page_url), preferred):
            bonus += 25
        if self._domain_matches(self._host(image_url), preferred):
            bonus += 15
        return bonus

    def _safe_ddgs_backend(self) -> str:
        backend = self.valves.ddgs_backend.strip().lower()
        if not backend:
            return "auto"

        # Permit simple comma-delimited backend lists while avoiding accidental
        # shell-like or URL-like input in a setting that should be just names.
        if re.fullmatch(r"[a-z0-9_, -]+", backend):
            return ",".join(part.strip() for part in backend.split(",") if part.strip())

        return "auto"

    def _ddgs_client(self) -> Any:
        if _DDGS is None:
            raise RuntimeError(_DDGS_IMPORT_ERROR)
        return _DDGS(
            proxy=self._proxy(),
            timeout=self._bounded_positive_int(self.valves.ddgs_timeout_seconds, 15),
            verify=self.valves.verify_ssl,
        )

    def _new_debug_report(
        self,
        query: str,
        mode: str,
        result_index: int,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "mode_requested": mode,
            "result_index_requested": result_index,
            "selected_mode": None,
            "query_variants_tried": [],
            "cleanup": {},
            "search": [],
            "candidate_attempts": [],
            "remote_validation": None,
            "selected_candidate": None,
            "cache": {},
            "final_stage": None,
            "final_status": None,
            "error": None,
        }

    def _sanitize_url_for_debug(self, value: str) -> str:
        if not value:
            return value

        sensitive_names = {
            "access_token",
            "api_key",
            "apikey",
            "auth",
            "authorization",
            "code",
            "credential",
            "key",
            "password",
            "sig",
            "signature",
            "signed",
            "token",
        }

        try:
            parts = urlsplit(value)
            pairs = parse_qsl(parts.query, keep_blank_values=True)
            clean_pairs = [
                (key, "REDACTED" if key.lower() in sensitive_names else val)
                for key, val in pairs
            ]
            return urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    parts.path,
                    urlencode(clean_pairs, doseq=True),
                    parts.fragment,
                )
            )
        except Exception:
            return value

    def _candidate_for_debug(self, candidate: ImageCandidate) -> dict[str, Any]:
        return {
            "title": candidate.title,
            "source": candidate.source,
            "page_url": self._sanitize_url_for_debug(candidate.page_url),
            "image_url": self._sanitize_url_for_debug(candidate.image_url),
            "thumbnail_url": self._sanitize_url_for_debug(candidate.thumbnail_url),
            "score": candidate.score,
            "width": candidate.width,
            "height": candidate.height,
        }

    def _debug_text(self, debug: dict[str, Any]) -> str:
        text = "DEBUG REPORT\n" + json.dumps(debug, indent=2, ensure_ascii=False)
        max_chars = max(1000, self._safe_int(self.valves.max_debug_chars, 12000))
        if len(text) > max_chars:
            return text[:max_chars] + "\n... debug output truncated ..."
        return text

    def _append_debug_note(self, note: str, debug: dict[str, Any]) -> str:
        if self.valves.debug_mode and self.valves.include_debug_in_response:
            return f"{note}\n\n{self._debug_text(debug)}"
        return note

    # ---------- URL / network safety ----------

    def _allowed_schemes(self) -> set[str]:
        schemes = set(self._parse_csv(self.valves.allowed_url_schemes))
        return schemes or {"https"}

    def _allowed_content_types(self) -> set[str]:
        return set(self._parse_csv(self.valves.allowed_image_content_types))

    def _content_type_allowed(self, content_type: str) -> bool:
        normalized = (content_type or "").split(";")[0].strip().lower()
        if not normalized.startswith("image/"):
            return False

        allowed = self._allowed_content_types()
        if not allowed:
            return True

        return normalized in allowed

    def _host_resolves_to_private_address(self, host: str) -> tuple[bool, str]:
        try:
            ip = ipaddress.ip_address(host)
            return self._is_privateish_ip(ip), str(ip)
        except ValueError:
            pass

        try:
            infos = socket.getaddrinfo(host, None)
        except Exception as e:
            return True, f"DNS resolution failed: {type(e).__name__}: {e}"

        addresses: set[str] = set()
        for info in infos:
            sockaddr = info[4]
            if not sockaddr:
                continue
            addresses.add(str(sockaddr[0]))

        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                return True, f"Unparseable resolved address: {address}"
            if self._is_privateish_ip(ip):
                return True, str(ip)

        return False, ",".join(sorted(addresses))

    def _is_privateish_ip(self, ip: ipaddress._BaseAddress) -> bool:
        return any(
            [
                ip.is_loopback,
                ip.is_private,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            ]
        )

    def _validate_fetch_url(self, url: str) -> tuple[bool, str]:
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"URL parse failed: {type(e).__name__}: {e}"

        if parsed.scheme.lower() not in self._allowed_schemes():
            return False, f"URL scheme is not allowed: {parsed.scheme or '(missing)'}"

        if not parsed.hostname:
            return False, "URL is missing a host."

        if parsed.username or parsed.password:
            return False, "URL userinfo is not allowed."

        if self.valves.allow_private_network_urls:
            return True, "ok"

        privateish, detail = self._host_resolves_to_private_address(parsed.hostname)
        if privateish:
            # SearXNG image_proxy URLs may legitimately point to an internal
            # container-only hostname. Allow only the explicitly configured
            # SearXNG origin; keep all other private-network fetch targets blocked.
            if (
                self.valves.allow_internal_search_service_urls
                and self._is_configured_searxng_origin(url)
            ):
                return True, "ok"
            return False, f"URL host resolves to a blocked address ({detail})."

        return True, "ok"

    # ---------- search / ranking ----------

    def _score_result(
        self,
        *,
        title: str,
        image_url: str,
        thumbnail_url: str,
        page_url: str,
        source: str,
        width: int,
        height: int,
        query: str,
    ) -> int:
        if not image_url or not page_url:
            return -10_000

        if self._is_blocked(page_url, image_url):
            return -10_000

        score = 0
        area = max(0, width) * max(0, height)
        if area >= 1_000_000:
            score += 40
        elif area >= 300_000:
            score += 25
        elif area > 0:
            score += 10

        title_l = title.lower()
        query_terms = [
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 2
        ]
        score += sum(3 for term in query_terms if term in title_l)

        if source:
            score += 2

        image_url_l = image_url.lower()
        if image_url_l.startswith("https://"):
            score += 3
        elif image_url_l.startswith("http://"):
            score += 1

        if re.search(r"\.(jpg|jpeg|png|webp|gif)(?:[?#]|$)", image_url_l):
            score += 4

        if thumbnail_url and thumbnail_url == image_url:
            score -= 4

        score += self._preference_bonus(page_url, image_url)
        return score

    def _query_variants(self, query: str) -> list[str]:
        base = query.strip()
        if not base:
            return []

        if not self.valves.query_variant_retry:
            return [base]

        lowered = base.lower()
        variants: list[str] = [base]

        phrase_rules = self._load_json_dict(self.valves.query_phrase_variants_json)
        for trigger, replacements in phrase_rules.items():
            if not isinstance(trigger, str) or not trigger.strip():
                continue
            if trigger.lower() not in lowered:
                continue
            if not isinstance(replacements, list):
                continue

            for replacement in replacements:
                if not isinstance(replacement, str) or not replacement.strip():
                    continue
                variants.append(re.sub(re.escape(trigger), replacement, base, flags=re.I))

        disambiguation_rules = self._load_json_dict(
            self.valves.query_disambiguation_suffixes_json
        )
        for trigger, suffix in disambiguation_rules.items():
            if not isinstance(trigger, str) or not trigger.strip():
                continue
            if not isinstance(suffix, str) or not suffix.strip():
                continue

            trigger_l = trigger.lower()
            suffix_l = suffix.lower()
            if trigger_l in lowered and suffix_l not in lowered:
                variants.append(f"{base} {suffix}")

        limit = self._bounded_positive_int(self.valves.max_query_variants, 4)
        return self._dedupe_keep_order(variants)[:limit]

    def _ddgs_kwargs(self, query: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query": query,
            "region": self.valves.ddgs_region.strip() or "us-en",
            "safesearch": self.valves.ddgs_safesearch.strip() or "moderate",
            "max_results": self._bounded_positive_int(
                self.valves.search_result_limit,
                12,
            ),
            "page": 1,
            "backend": self._safe_ddgs_backend(),
        }

        optional_map = {
            "timelimit": self.valves.ddgs_timelimit,
            "size": self.valves.image_size,
            "color": self.valves.image_color,
            "type_image": self.valves.image_type,
            "layout": self.valves.image_layout,
            "license_image": self.valves.image_license,
        }

        for key, value in optional_map.items():
            value = str(value or "").strip()
            if value:
                kwargs[key] = value

        return kwargs

    def _search_image_candidates(
        self,
        query: str,
        debug: Optional[dict[str, Any]] = None,
    ) -> list[ImageCandidate]:
        provider = self._safe_search_provider()

        if provider == "searxng":
            try:
                return self._search_searxng_image_candidates(query, debug=debug)
            except Exception:
                if self.valves.fallback_to_ddgs_on_searxng_failure:
                    return self._search_ddgs_image_candidates(query, debug=debug)
                raise

        if provider == "ddgs":
            return self._search_ddgs_image_candidates(query, debug=debug)

        # auto: prefer SearXNG only when it is explicitly configured. This avoids
        # silently trying a guessed container hostname on installations without
        # SearXNG while still letting Open WebUI/SearXNG deployments opt in cleanly.
        if self._searxng_configured():
            try:
                return self._search_searxng_image_candidates(query, debug=debug)
            except Exception:
                return self._search_ddgs_image_candidates(query, debug=debug)

        return self._search_ddgs_image_candidates(query, debug=debug)

    def _search_ddgs_image_candidates(
        self,
        query: str,
        debug: Optional[dict[str, Any]] = None,
    ) -> list[ImageCandidate]:
        kwargs = self._ddgs_kwargs(query)
        search_debug: dict[str, Any] = {
            "provider": "ddgs",
            "query": query,
            "kwargs": kwargs,
            "raw_result_count": None,
            "normalized_result_count": None,
            "exception": None,
            "top_candidates": [],
        }

        try:
            results = list(self._ddgs_client().images(**kwargs) or [])
            search_debug["raw_result_count"] = len(results)
        except Exception as e:
            search_debug["exception"] = f"{type(e).__name__}: {e}"
            if debug is not None and self.valves.debug_mode:
                debug["search"].append(search_debug)
            raise RuntimeError(
                f"DDGS image search failed for query={query!r}: {type(e).__name__}: {e}"
            ) from e

        normalized: list[ImageCandidate] = []

        for item in results:
            if not isinstance(item, dict):
                continue

            image_url = str(item.get("image") or "").strip()
            thumbnail_url = str(item.get("thumbnail") or "").strip()
            page_url = str(item.get("url") or item.get("href") or "").strip()

            if not image_url and self.valves.allow_thumbnail_fallback:
                image_url = thumbnail_url

            if not image_url or not page_url:
                continue

            title = str(item.get("title") or query).strip() or query
            source = str(item.get("source") or "").strip()
            width = self._safe_int(item.get("width"), 0)
            height = self._safe_int(item.get("height"), 0)

            score = self._score_result(
                title=title,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                page_url=page_url,
                source=source,
                width=width,
                height=height,
                query=query,
            )

            if score <= -10_000:
                continue

            normalized.append(
                ImageCandidate(
                    title=title,
                    image_url=image_url,
                    thumbnail_url=thumbnail_url,
                    page_url=page_url,
                    source=source,
                    width=width,
                    height=height,
                    score=score,
                )
            )

        normalized.sort(key=lambda candidate: candidate.score, reverse=True)

        search_debug["normalized_result_count"] = len(normalized)
        search_debug["top_candidates"] = [
            self._candidate_for_debug(candidate) for candidate in normalized[:5]
        ]

        if debug is not None and self.valves.debug_mode:
            debug["search"].append(search_debug)

        return normalized

    def _parse_resolution(self, value: Any) -> tuple[int, int]:
        raw = str(value or "")
        match = re.search(r"(\d{2,5})\s*[x×]\s*(\d{2,5})", raw, flags=re.I)
        if not match:
            return 0, 0
        return self._safe_int(match.group(1), 0), self._safe_int(match.group(2), 0)

    def _candidate_from_searxng_item(
        self,
        item: dict[str, Any],
        query: str,
        response_url: str,
    ) -> Optional[ImageCandidate]:
        image_url = (
            str(
                item.get("img_src")
                or item.get("image")
                or item.get("image_url")
                or item.get("thumbnail_src")
                or item.get("thumbnail")
                or ""
            )
            .strip()
        )
        thumbnail_url = (
            str(
                item.get("thumbnail")
                or item.get("thumbnail_src")
                or item.get("thumb")
                or ""
            )
            .strip()
        )
        page_url = str(item.get("url") or item.get("href") or item.get("source_url") or "").strip()

        image_url = self._make_absolute_result_url(image_url, response_url)
        thumbnail_url = self._make_absolute_result_url(thumbnail_url, response_url)
        page_url = self._make_absolute_result_url(page_url, response_url)

        if not image_url and self.valves.allow_thumbnail_fallback:
            image_url = thumbnail_url

        if not page_url:
            page_url = image_url

        if not image_url or not page_url:
            return None

        title = str(item.get("title") or item.get("content") or query).strip() or query

        engines = item.get("engines")
        if isinstance(engines, list):
            source = ", ".join(str(engine) for engine in engines if str(engine).strip())
        else:
            source = str(item.get("engine") or item.get("source") or "SearXNG").strip()

        width = self._safe_int(item.get("width"), 0)
        height = self._safe_int(item.get("height"), 0)

        if not width or not height:
            width, height = self._parse_resolution(
                item.get("resolution")
                or item.get("img_format")
                or item.get("metadata")
                or ""
            )

        score = self._score_result(
            title=title,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            page_url=page_url,
            source=source,
            width=width,
            height=height,
            query=query,
        )

        if score <= -10_000:
            return None

        return ImageCandidate(
            title=title,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            page_url=page_url,
            source=source,
            width=width,
            height=height,
            score=score,
        )

    def _search_searxng_image_candidates(
        self,
        query: str,
        debug: Optional[dict[str, Any]] = None,
    ) -> list[ImageCandidate]:
        url = self._searxng_search_url(query)
        ok, reason = self._search_service_url_allowed(url)
        search_debug: dict[str, Any] = {
            "provider": "searxng",
            "query": query,
            "url": self._sanitize_url_for_debug(url),
            "raw_result_count": None,
            "normalized_result_count": None,
            "exception": None,
            "top_candidates": [],
        }

        if not ok:
            search_debug["exception"] = reason
            if debug is not None and self.valves.debug_mode:
                debug["search"].append(search_debug)
            raise RuntimeError(reason)

        try:
            with requests.get(
                url,
                headers=self._headers(accept="application/json,*/*;q=0.8"),
                timeout=self._bounded_positive_int(self.valves.searxng_timeout_seconds, 15),
                allow_redirects=True,
                verify=self.valves.verify_ssl,
            ) as resp:
                resp.raise_for_status()
                data = resp.json()
                response_url = resp.url
        except Exception as e:
            search_debug["exception"] = f"{type(e).__name__}: {e}"
            if debug is not None and self.valves.debug_mode:
                debug["search"].append(search_debug)
            raise RuntimeError(
                f"SearXNG image search failed for query={query!r}: {type(e).__name__}: {e}"
            ) from e

        if not isinstance(data, dict):
            search_debug["exception"] = "SearXNG response was not a JSON object."
            if debug is not None and self.valves.debug_mode:
                debug["search"].append(search_debug)
            raise RuntimeError("SearXNG response was not a JSON object.")

        results = data.get("results") or []
        if not isinstance(results, list):
            results = []

        search_debug["raw_result_count"] = len(results)
        normalized: list[ImageCandidate] = []

        for item in results:
            if not isinstance(item, dict):
                continue
            candidate = self._candidate_from_searxng_item(
                item=item,
                query=query,
                response_url=response_url,
            )
            if candidate is not None:
                normalized.append(candidate)

        normalized.sort(key=lambda candidate: candidate.score, reverse=True)

        limit = self._bounded_positive_int(self.valves.search_result_limit, 12)
        if len(normalized) > limit:
            normalized = normalized[:limit]

        search_debug["normalized_result_count"] = len(normalized)
        search_debug["top_candidates"] = [
            self._candidate_for_debug(candidate) for candidate in normalized[:5]
        ]

        if debug is not None and self.valves.debug_mode:
            debug["search"].append(search_debug)

        return normalized

    # ---------- probing / fetching ----------

    def _probe_remote_image_url(self, image_url: str) -> tuple[bool, str]:
        ok, reason = self._validate_fetch_url(image_url)
        if not ok:
            return False, reason

        timeout = self._bounded_positive_int(
            self.valves.remote_mode_probe_timeout_seconds,
            8,
        )

        head_error = "HEAD not attempted"

        try:
            resp = requests.head(
                image_url,
                headers=self._headers(accept="image/*,*/*;q=0.8"),
                timeout=timeout,
                allow_redirects=True,
                verify=self.valves.verify_ssl,
                proxies=self._requests_proxies(),
            )
            ctype = (resp.headers.get("content-type") or "").lower()
            if resp.ok and self._content_type_allowed(ctype):
                resp.close()
                return True, f"HEAD ok: {ctype}"
            head_error = (
                f"HEAD non-image/disallowed or bad status: "
                f"status={resp.status_code}, content_type={ctype or 'unknown'}"
            )
            resp.close()
        except Exception as e:
            head_error = f"HEAD {type(e).__name__}: {e}"

        try:
            with requests.get(
                image_url,
                headers=self._headers(accept="image/*,*/*;q=0.8"),
                timeout=timeout,
                stream=True,
                allow_redirects=True,
                verify=self.valves.verify_ssl,
                proxies=self._requests_proxies(),
            ) as resp:
                ctype = (resp.headers.get("content-type") or "").lower()
                if resp.ok and self._content_type_allowed(ctype):
                    return True, f"GET ok: {ctype}"

                detail = (
                    f"{head_error}; GET status={resp.status_code}, "
                    f"content_type={ctype or 'unknown'}"
                )
                if not self._content_type_allowed(ctype):
                    detail += ", content_type_allowed=false"
                return False, detail
        except Exception as e:
            return False, f"{head_error}; GET {type(e).__name__}: {e}"

    def _download_image(self, image_url: str) -> tuple[bytes, str, str]:
        ok, reason = self._validate_fetch_url(image_url)
        if not ok:
            raise ValueError(reason)

        max_mb = self._bounded_positive_int(self.valves.max_download_mb, 12)
        max_bytes = max_mb * 1024 * 1024

        with requests.get(
            image_url,
            headers=self._headers(accept="image/*,*/*;q=0.8"),
            timeout=60,
            stream=True,
            allow_redirects=True,
            verify=self.valves.verify_ssl,
            proxies=self._requests_proxies(),
        ) as resp:
            resp.raise_for_status()

            content_type = (
                (resp.headers.get("content-type") or "")
                .split(";")[0]
                .strip()
                .lower()
            )
            if not self._content_type_allowed(content_type):
                raise ValueError(
                    "URL did not return an allowed image Content-Type: "
                    f"{content_type or 'unknown'}"
                )

            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise ValueError("Image exceeds configured size limit.")
                except ValueError:
                    raise
                except Exception:
                    pass

            chunks: list[bytes] = []
            total = 0

            for chunk in resp.iter_content(1024 * 64):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Image exceeds configured size limit.")
                chunks.append(chunk)

        image_bytes = b"".join(chunks)
        ext = mimetypes.guess_extension(content_type) or ".img"
        if ext == ".jpe":
            ext = ".jpg"

        return image_bytes, content_type, ext

    def _extract_page_meta_image(self, page_url: str) -> Optional[str]:
        ok, reason = self._validate_fetch_url(page_url)
        if not ok:
            raise ValueError(reason)

        max_bytes = self._bounded_positive_int(self.valves.page_meta_max_bytes, 524288)

        with requests.get(
            page_url,
            headers=self._headers(accept="text/html,*/*;q=0.8"),
            timeout=self._bounded_positive_int(self.valves.page_meta_timeout_seconds, 12),
            stream=True,
            allow_redirects=True,
            verify=self.valves.verify_ssl,
            proxies=self._requests_proxies(),
        ) as resp:
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").lower()
            if "html" not in ctype:
                return None

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(8192):
                if not chunk:
                    continue
                total += len(chunk)
                chunks.append(chunk)
                if total >= max_bytes:
                    break

            encoding = resp.encoding or "utf-8"

        text = b"".join(chunks).decode(encoding, errors="replace")

        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return urljoin(page_url, html.unescape(match.group(1).strip()))

        return None

    def _pick_working_image_url(
        self,
        candidate: ImageCandidate,
        debug: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], str]:
        attempt_debug: dict[str, Any] = {
            **self._candidate_for_debug(candidate),
            "trials": [],
            "selected_via": None,
            "selected_url": None,
            "failure_reason": None,
        }

        trial_urls: list[tuple[str, str]] = []
        if candidate.image_url:
            trial_urls.append(("image_url", candidate.image_url))
        if candidate.thumbnail_url and candidate.thumbnail_url != candidate.image_url:
            trial_urls.append(("thumbnail_url", candidate.thumbnail_url))

        for label, trial_url in trial_urls:
            if not self.valves.probe_candidates_during_selection:
                attempt_debug["trials"].append(
                    {
                        "via": label,
                        "url": self._sanitize_url_for_debug(trial_url),
                        "ok": True,
                        "detail": "probe skipped",
                    }
                )
                attempt_debug["selected_via"] = label
                attempt_debug["selected_url"] = self._sanitize_url_for_debug(trial_url)
                if debug is not None and self.valves.debug_mode:
                    debug["candidate_attempts"].append(attempt_debug)
                return trial_url, label

            ok, detail = self._probe_remote_image_url(trial_url)
            attempt_debug["trials"].append(
                {
                    "via": label,
                    "url": self._sanitize_url_for_debug(trial_url),
                    "ok": ok,
                    "detail": detail,
                }
            )

            if ok:
                attempt_debug["selected_via"] = label
                attempt_debug["selected_url"] = self._sanitize_url_for_debug(trial_url)
                if debug is not None and self.valves.debug_mode:
                    debug["candidate_attempts"].append(attempt_debug)
                return trial_url, label

            if not self.valves.skip_non_image_candidates and re.search(
                r"\.(jpg|jpeg|png|webp|gif)(?:[?#]|$)",
                trial_url.lower(),
            ):
                attempt_debug["selected_via"] = f"{label}_unverified"
                attempt_debug["selected_url"] = self._sanitize_url_for_debug(trial_url)
                if debug is not None and self.valves.debug_mode:
                    debug["candidate_attempts"].append(attempt_debug)
                return trial_url, f"{label}_unverified"

        if self.valves.use_page_meta_image_fallback and candidate.page_url:
            try:
                meta_image = self._extract_page_meta_image(candidate.page_url)
                attempt_debug["meta_image_url"] = self._sanitize_url_for_debug(
                    meta_image or ""
                )
                if meta_image:
                    ok, detail = self._probe_remote_image_url(meta_image)
                    attempt_debug["trials"].append(
                        {
                            "via": "page_meta_image",
                            "url": self._sanitize_url_for_debug(meta_image),
                            "ok": ok,
                            "detail": detail,
                        }
                    )
                    if ok:
                        attempt_debug["selected_via"] = "page_meta_image"
                        attempt_debug["selected_url"] = self._sanitize_url_for_debug(
                            meta_image
                        )
                        if debug is not None and self.valves.debug_mode:
                            debug["candidate_attempts"].append(attempt_debug)
                        return meta_image, "page_meta_image"
            except Exception as e:
                attempt_debug["meta_image_exception"] = f"{type(e).__name__}: {e}"

        attempt_debug["failure_reason"] = "no_working_image"
        if debug is not None and self.valves.debug_mode:
            debug["candidate_attempts"].append(attempt_debug)

        return None, "no_working_image"

    # ---------- Open WebUI file API / cache ----------

    def _upload_to_openwebui(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        __request__=None,
    ) -> str:
        base_url = self._owui_base_url(__request__)
        url = f"{base_url}/api/v1/files/"

        resp = requests.post(
            url,
            params={"process": "false"},
            files={"file": (filename, image_bytes, content_type)},
            headers=self._auth_headers(__request__),
            timeout=60,
            verify=self.valves.verify_ssl,
        )
        resp.raise_for_status()

        data = resp.json()
        file_id: Optional[str] = None

        if isinstance(data, dict):
            file_id = data.get("id")
            nested = data.get("file")
            if not file_id and isinstance(nested, dict):
                file_id = nested.get("id")
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                file_id = first.get("id")
                nested = first.get("file")
                if not file_id and isinstance(nested, dict):
                    file_id = nested.get("id")

        if not file_id:
            raise ValueError("Could not extract uploaded file id from Open WebUI response.")

        return str(file_id)

    def _delete_openwebui_file(self, file_id: str, __request__=None) -> bool:
        base_url = self._owui_base_url(__request__)
        url = f"{base_url}/api/v1/files/{quote(str(file_id), safe='')}"

        try:
            resp = requests.delete(
                url,
                headers=self._auth_headers(__request__),
                timeout=30,
                verify=self.valves.verify_ssl,
            )
            return resp.ok
        except Exception:
            return False

    def _register_cached_file(
        self,
        *,
        file_id: str,
        query: str,
        source_url: str,
        filename: str,
        content_type: str,
    ) -> None:
        registry = self._load_registry()
        registry[str(file_id)] = {
            "query": query,
            "source_url": source_url,
            "filename": filename,
            "content_type": content_type,
            "created_at": int(time.time()),
            "expires_at": int(time.time()) + (self.valves.cache_ttl_hours * 3600),
        }
        self._save_registry(registry)

    def _sweep_expired_cache(self, __request__=None) -> dict[str, int]:
        registry = self._load_registry()
        if not registry:
            return {"deleted": 0, "failed": 0, "remaining": 0}

        now = int(time.time())
        kept: dict[str, Any] = {}
        deleted = 0
        failed = 0

        for file_id, meta in registry.items():
            if not isinstance(meta, dict):
                failed += 1
                continue

            expires_at = self._safe_int(meta.get("expires_at"), 0)
            if expires_at > now:
                kept[file_id] = meta
                continue

            if self._delete_openwebui_file(file_id, __request__=__request__):
                deleted += 1
            else:
                failed += 1
                kept[file_id] = meta

        self._save_registry(kept)
        return {"deleted": deleted, "failed": failed, "remaining": len(kept)}

    # ---------- rendering / mode selection ----------

    def _preferred_mode(self, explicit_mode: str, __user__=None) -> str:
        mode = (explicit_mode or "").strip().lower()
        if mode in {"remote", "cache"}:
            return mode

        if __user__ and isinstance(__user__, dict):
            user_valves = __user__.get("valves")
            user_mode = ""

            if isinstance(user_valves, dict):
                user_mode = str(user_valves.get("preferred_mode") or "").strip().lower()
            else:
                user_mode = str(getattr(user_valves, "preferred_mode", "") or "").strip().lower()

            if user_mode in {"remote", "cache"}:
                return user_mode

        mode = self.valves.default_mode.strip().lower()
        return mode if mode in {"remote", "cache"} else "remote"

    def _markdown_escape_alt(self, text: str) -> str:
        return (
            str(text or "Image")
            .replace("\\", "\\\\")
            .replace("[", "\\[")
            .replace("]", "\\]")
            .replace("\n", " ")
            .strip()
            or "Image"
        )

    def _markdown_url(self, url: str) -> str:
        clean = str(url or "").replace("<", "%3C").replace(">", "%3E").strip()
        return f"<{clean}>"

    def _render_markdown(
        self,
        *,
        title: str,
        image_url: str,
        page_url: str,
        source_name: str = "",
        note: str = "",
    ) -> str:
        lines = [f"![{self._markdown_escape_alt(title)}]({self._markdown_url(image_url)})"]

        if self.valves.include_source_link and page_url:
            label = "Source"
            if self.valves.include_source_name and source_name:
                label = f"Source: {self._markdown_escape_alt(source_name)}"
            lines.append(f"\n{label} — {self._markdown_url(page_url)}")

        if note:
            lines.append(f"\n{note}")

        return "\n".join(lines).strip()

    def _select_candidate(
        self,
        *,
        query: str,
        result_index: int,
        debug: dict[str, Any],
    ) -> tuple[Optional[ImageCandidate], Optional[str], Optional[str], Optional[str]]:
        search_exception = None
        query_variants = self._query_variants(query)
        debug["query_variants_tried"] = query_variants

        for variant in query_variants:
            try:
                candidates = self._search_image_candidates(variant, debug=debug)
            except Exception as e:
                search_exception = f"{type(e).__name__}: {e}"
                continue

            if not candidates:
                continue

            start_idx = max(1, int(result_index)) - 1
            ordered = candidates[start_idx:] + candidates[:start_idx]

            max_attempts = self._bounded_positive_int(
                self.valves.max_candidate_attempts,
                5,
            )
            for candidate in ordered[:max_attempts]:
                working_url, via = self._pick_working_image_url(candidate, debug=debug)
                if not working_url:
                    continue

                debug["selected_candidate"] = {
                    **self._candidate_for_debug(candidate),
                    "query_variant": variant,
                    "selected_url": self._sanitize_url_for_debug(working_url),
                    "selected_via": via,
                }
                return candidate, working_url, via, search_exception

        return None, None, None, search_exception

    # ---------- public tool methods ----------

    async def find_and_show_image(
        self,
        query: str = Field(..., description="What image to search for and display."),
        mode: str = Field(
            default="",
            description="Use 'remote' to hotlink with zero local storage, or 'cache' to upload the image into Open WebUI temporarily.",
        ),
        result_index: int = Field(
            default=1,
            description="1-based index into the ranked image results after filtering. Use 1 for the best match, 2 for the next one, and so on.",
        ),
        __request__=None,
        __user__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Search for an image and return Markdown for inline display.

        Important model behavior:
        If this tool returns Markdown image syntax, reproduce it exactly in the final
        assistant reply and do not wrap it in code fences.
        """

        query = str(query or "").strip()
        if not query:
            return "Please provide a non-empty image search query."

        result_index = self._bounded_positive_int(result_index, 1)
        debug = self._new_debug_report(query=query, mode=mode, result_index=result_index)

        selected_mode = self._preferred_mode(mode, __user__=__user__)
        debug["selected_mode"] = selected_mode

        if self.valves.cleanup_expired_on_search:
            try:
                debug["cleanup"] = self._sweep_expired_cache(__request__=__request__)
            except Exception as e:
                debug["cleanup"] = {
                    "deleted": 0,
                    "failed": 0,
                    "remaining": -1,
                    "error": f"{type(e).__name__}: {e}",
                }
        else:
            debug["cleanup"] = {"deleted": 0, "failed": 0, "remaining": -1, "skipped": 1}

        await self._emit(__event_emitter__, "Searching for image results...", False)

        candidate, image_url, selected_via, search_exception = self._select_candidate(
            query=query,
            result_index=result_index,
            debug=debug,
        )

        if not candidate or not image_url:
            debug["final_stage"] = "candidate_selection"
            if search_exception and not debug.get("search"):
                debug["error"] = search_exception
                debug["final_status"] = "search_exception"
                await self._emit(__event_emitter__, "Image search failed.", True)
                return self._append_debug_note(
                    f"Image search failed: {search_exception}",
                    debug,
                )

            debug["final_status"] = "no_usable_candidate"
            await self._emit(
                __event_emitter__,
                "No candidate returned a usable image payload.",
                True,
            )
            return self._append_debug_note(
                "Image search completed, but none of the tested candidates produced a usable direct image payload.",
                debug,
            )

        if selected_mode == "remote":
            if self.valves.remote_mode_validate_url:
                await self._emit(__event_emitter__, "Validating remote image URL...", False)
                ok, detail = self._probe_remote_image_url(image_url)
                debug["remote_validation"] = {"ok": ok, "detail": detail}
                if not ok:
                    if self.valves.fallback_to_cache_on_remote_validation_failure:
                        selected_mode = "cache"
                        debug["selected_mode"] = "cache"
                    else:
                        debug["final_stage"] = "remote_validation"
                        debug["final_status"] = "remote_validation_failed"
                        await self._emit(
                            __event_emitter__,
                            "Remote image URL validation failed.",
                            True,
                        )
                        return self._append_debug_note(
                            (
                                "Remote image validation failed for the selected result: "
                                f"{detail}\n\nTry again with mode='cache' or choose a different result_index."
                            ),
                            debug,
                        )

            if selected_mode == "remote":
                debug["final_stage"] = "remote_render"
                debug["final_status"] = "success"
                await self._emit(__event_emitter__, "Done.", True)

                note = (
                    f"(Cleanup: deleted={debug['cleanup'].get('deleted', 0)}, "
                    f"failed={debug['cleanup'].get('failed', 0)}, "
                    f"remaining={debug['cleanup'].get('remaining', 0)})"
                )
                note = self._append_debug_note(note, debug)

                return self._render_markdown(
                    title=candidate.title,
                    image_url=image_url,
                    page_url=candidate.page_url,
                    source_name=candidate.source,
                    note=note,
                )

        await self._emit(__event_emitter__, "Downloading and caching image...", False)

        try:
            image_bytes, content_type, ext = self._download_image(image_url)
            debug["cache"]["download"] = {
                "ok": True,
                "content_type": content_type,
                "byte_count": len(image_bytes),
                "extension": ext,
            }

            filename = f"{self._slugify(query)}-{uuid.uuid4().hex[:8]}{ext}"
            file_id = self._upload_to_openwebui(
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
                __request__=__request__,
            )

            debug["cache"]["upload"] = {
                "ok": True,
                "filename": filename,
                "file_id": file_id,
            }

            self._register_cached_file(
                file_id=file_id,
                query=query,
                source_url=candidate.page_url,
                filename=filename,
                content_type=content_type,
            )

            local_url = (
                f"{self._owui_base_url(__request__)}/api/v1/files/"
                f"{quote(file_id, safe='')}/content"
            )

            debug["final_stage"] = "cache_render"
            debug["final_status"] = "success"

            await self._emit(__event_emitter__, "Done.", True)

            note = (
                f"Cached temporarily for {self.valves.cache_ttl_hours} hours.\n"
                f"(Cleanup: deleted={debug['cleanup'].get('deleted', 0)}, "
                f"failed={debug['cleanup'].get('failed', 0)}, "
                f"remaining={debug['cleanup'].get('remaining', 0)})"
            )
            note = self._append_debug_note(note, debug)

            return self._render_markdown(
                title=candidate.title,
                image_url=local_url,
                page_url=candidate.page_url,
                source_name=candidate.source,
                note=note,
            )

        except Exception as e:
            debug["cache"]["exception"] = f"{type(e).__name__}: {e}"

            if self.valves.fallback_to_remote_on_cache_failure:
                debug["final_stage"] = "cache_render"
                debug["final_status"] = "cache_failed_remote_fallback"

                await self._emit(
                    __event_emitter__,
                    "Cache failed; falling back to remote image URL...",
                    False,
                )
                await self._emit(__event_emitter__, "Done.", True)

                note = (
                    "Cache attempt failed and remote fallback was used instead: "
                    f"{type(e).__name__}: {e}"
                )
                note = self._append_debug_note(note, debug)

                return self._render_markdown(
                    title=candidate.title,
                    image_url=image_url,
                    page_url=candidate.page_url,
                    source_name=candidate.source,
                    note=note,
                )

            debug["final_stage"] = "cache_render"
            debug["final_status"] = "cache_failed"
            await self._emit(__event_emitter__, "Cache failed.", True)
            return self._append_debug_note(
                f"Cache mode failed: {type(e).__name__}: {e}",
                debug,
            )

    async def cleanup_expired_cached_images(
        self,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Clean up expired image files created by the image search tool.

        This does not search for new images. It only deletes expired cached image
        files recorded by this tool's cache registry.
        """

        await self._emit(__event_emitter__, "Cleaning expired cached images...", False)

        try:
            result = self._sweep_expired_cache(__request__=__request__)
            await self._emit(__event_emitter__, "Done.", True)
            return (
                "Expired cached images cleanup complete. "
                f"Deleted={result['deleted']}, failed={result['failed']}, "
                f"remaining={result['remaining']}."
            )
        except Exception as e:
            await self._emit(__event_emitter__, "Cached image cleanup failed.", True)
            return f"Expired cached images cleanup failed: {type(e).__name__}: {e}"

    async def cleanup_cached_images(
        self,
        __request__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Alias for cleanup_expired_cached_images.
        """

        return await self.cleanup_expired_cached_images(
            __request__=__request__,
            __event_emitter__=__event_emitter__,
        )

    async def debug_candidates(
        self,
        query: str = Field(..., description="What image query to inspect."),
        __event_emitter__=None,
    ) -> str:
        """
        Inspect image candidates and return a debug report without rendering or caching.
        """

        query = str(query or "").strip()
        if not query:
            return "Please provide a non-empty image search query."

        debug = self._new_debug_report(query=query, mode="debug", result_index=1)
        await self._emit(__event_emitter__, "Inspecting image candidates...", False)

        try:
            variants = self._query_variants(query)
            debug["query_variants_tried"] = variants
            for variant in variants:
                self._search_image_candidates(variant, debug=debug)

            debug["final_stage"] = "debug_candidates"
            debug["final_status"] = "success"
            await self._emit(__event_emitter__, "Done.", True)
            return self._debug_text(debug)
        except Exception as e:
            debug["error"] = f"{type(e).__name__}: {e}"
            debug["final_stage"] = "debug_candidates"
            debug["final_status"] = "failed"
            await self._emit(__event_emitter__, "Candidate inspection failed.", True)
            return self._debug_text(debug)
