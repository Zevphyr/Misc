# Image Search & Inline Display (DDGS)

Open WebUI tool for searching DDGS image results and returning inline Markdown, with optional temporary caching through Open WebUI's file API.

## Requirements

- `ddgs`
- `requests`
- Open WebUI's existing Python environment normally provides `pydantic`.

## Notes

Review the valves before enabling remote image fetching or caching. Avoid disabling TLS verification except while debugging a trusted local setup.
