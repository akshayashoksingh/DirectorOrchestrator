#!/usr/bin/env python3
"""
lore_mcp_server.py — Phase 1 MCP research server (stdio transport).

Exposes three read-only research tools over MCP:
    1. fetch_anime_episodes_and_reviews  -> Jikan v4 (MyAnimeList mirror)
    2. scrape_reddit_character_lore      -> Reddit public search JSON
    3. fetch_wiki_character_summary      -> Wikipedia MediaWiki API

Design constraint (6–8 GB VRAM profile):
    The consuming model runs with a small context window (~8k tokens), so
    every tool trims its payload to a fixed character budget BEFORE returning.
    Budgets are module-level constants below; raise them if you raise
    --num-ctx in the orchestrator.

IMPORTANT: stdout is the MCP JSON-RPC channel. Never print() here;
all diagnostics go to stderr via `logging`.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging — stderr only (stdout belongs to the MCP protocol stream).
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[lore-mcp] %(levelname)s %(message)s",
)
log = logging.getLogger("lore_mcp_server")

# ---------------------------------------------------------------------------
# Server instance. The name is what MCP clients see during initialisation.
# ---------------------------------------------------------------------------
mcp = FastMCP("anime-lore-engine")

# ---------------------------------------------------------------------------
# Endpoints and HTTP configuration.
# ---------------------------------------------------------------------------
JIKAN_BASE = "https://api.jikan.moe/v4"
REDDIT_SEARCH = "https://www.reddit.com/search.json"
REDDIT_SEARCH_FALLBACK = "https://old.reddit.com/search.json"  # often less strict
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Wikimedia policy requires contact info in the UA; Reddit wants a descriptive one.
# Set it in cmd before running:  set LORE_CONTACT=you@example.com
LORE_CONTACT = os.environ.get("LORE_CONTACT", "").strip()
USER_AGENT = "AnimeLoreEngine/1.0" + (f" (contact: {LORE_CONTACT})" if LORE_CONTACT else "")
HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
JIKAN_DELAY_S = 1.0  # mandated 1 s spacing between Jikan calls

# ---------------------------------------------------------------------------
# Payload budgets (characters). Sized so that all three tool results together
# stay around ~4k tokens, leaving room in an 8k window for prompts + output.
# ---------------------------------------------------------------------------
MAX_EPISODES_LISTED = 60        # episode titles listed
MAX_EPISODE_SYNOPSES = 26       # per-episode synopsis lookups (1 req each)
EPISODE_SYNOPSIS_CHARS = 240
MAX_REVIEWS = 3
REVIEW_CHARS = 500
REDDIT_LIMIT = 8
REDDIT_SELFTEXT_CHARS = 450
WIKI_MAX_PAGES = 2
WIKI_CHARS_PER_PAGE = 1800


# ===========================================================================
# Shared helpers
# ===========================================================================
def _clean(text: str | None, limit: int) -> str:
    """Collapse whitespace and truncate at a sentence or word boundary.

    Prefers cutting at the last sentence terminator in the final 40 % of the
    window; otherwise cuts at the last space. Appends an ellipsis marker so
    the model knows the text was shortened.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if sentence_end >= int(limit * 0.6):
        return cut[: sentence_end + 1] + " …"
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut) + " …"


def _norm(s: str | None) -> str:
    """Lower-case, alphanumeric-only form used for title comparison."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _dump(obj: Any) -> str:
    """Compact JSON (no indentation: whitespace costs context tokens)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _client() -> httpx.AsyncClient:
    """One configured async client per tool invocation."""
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )


# ===========================================================================
# Season resolution (shared by Jikan, Kitsu and AniList)
# ===========================================================================
# Those catalogues list every season as its own entry ("My Hero Academia",
# "My Hero Academia Season 2", ...), so choosing a season means choosing an entry.
# Entries whose title matches the query are ordered by air date: the Nth TV
# entry is season N. This is a heuristic: it assumes the catalogue has no gaps
# and that other TV series do not share the title.
#
# The `season` argument understood by the tools:
#     ""            best title match (normally season 1)
#     "2", "Season 2", "2nd season"       one season
#     "final", or text from a season's title ("Culling Game")
#     "all", "1-7", "1,3,5", "1-3,5"      several seasons -> multi-season result
MAX_SEASONS = 12  # safety cap for "all"

_SEASON_RE = re.compile(r"(?:season|series|s)?\s*(\d{1,2})(?:st|nd|rd|th)?(?:\s*season)?")
_LIST_RE = re.compile(r"\d{1,2}(?:\s*[-–]\s*\d{1,2})?(?:\s*,\s*\d{1,2}(?:\s*[-–]\s*\d{1,2})?)*")


def season_number(spec: str) -> int | None:
    """'2', 'Season 2', '2nd season', 's2' -> 2; anything else (a name) -> None."""
    m = _SEASON_RE.fullmatch((spec or "").strip().lower())
    return int(m.group(1)) if m else None


def parse_season_spec(spec: str) -> tuple[str, list[int] | None]:
    """Classify a season argument.

    Returns (kind, numbers):
      ("default", None)  blank
      ("single", [n])    one season number
      ("multi", None)    "all"
      ("multi", [..])    a list or range of several seasons
      ("text", None)     a name or "final" (single season, resolved by title)
    """
    s = (spec or "").strip().lower()
    if not s:
        return "default", None
    n = season_number(s)
    if n is not None:
        return "single", [n]
    if s == "all":
        return "multi", None
    if _LIST_RE.fullmatch(s):
        nums: list[int] = []
        for part in s.split(","):
            part = part.strip()
            m = re.fullmatch(r"(\d{1,2})\s*[-–]\s*(\d{1,2})", part)
            if m:
                a, b = sorted((int(m.group(1)), int(m.group(2))))
                nums.extend(range(a, b + 1))
            else:
                nums.append(int(part))
        nums = sorted(set(nums))
        return ("single" if len(nums) == 1 else "multi"), nums
    return "text", None


def _titles_of(r: dict[str, Any]) -> list[str]:
    return [_norm(t) for t in r["titles"] if t]


def _is_tv(r: dict[str, Any]) -> bool:
    return str(r.get("type") or "").upper() == "TV"


def _first_title(r: dict[str, Any]) -> str:
    return next((t for t in r["titles"] if t), "?")


def _tv_seasons(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """TV entries whose title matches the query, in air-date order (index 0 = season 1)."""
    q = _norm(query)
    tv = [r for r in records if _is_tv(r)
          and any(len(t) >= 4 and (q in t or t in q) for t in _titles_of(r))]
    tv.sort(key=lambda r: r.get("start") or "9999")   # stable: ties keep source order
    return tv


def _season_list(tv: list[dict[str, Any]]) -> str:
    return "; ".join(f"{i}: {_first_title(r)}" for i, r in enumerate(tv, 1))


def resolve_season(records: list[dict[str, Any]], query: str, season: str) -> tuple[dict[str, Any], int | None]:
    """Pick the catalogue entry for ONE requested season.

    records: [{"obj": <source entry>, "titles": [str|None, ...], "type": str|None,
               "start": "YYYY-MM-DD" or ""}] in the source's relevance order.

    Returns (record, season_number). The number is the entry's position among
    the matching TV entries by air date, or None when that cannot be determined.
    Raises RuntimeError listing the seasons found if the request cannot be met.
    """
    spec = (season or "").strip()
    q = _norm(query)
    tv = _tv_seasons(records, query)

    def number_of(rec: dict[str, Any]) -> int | None:
        return next((i for i, r in enumerate(tv, 1) if r is rec), None)

    if not spec:  # best title match, as before; report which season that turned out to be
        def score(r: dict[str, Any]) -> int:
            t = _titles_of(r)
            if q in t:
                return 3
            if any(q in x or x in q for x in t if x):
                return 2
            return 1 if _is_tv(r) else 0
        best = max(records, key=score)
        return best, number_of(best)

    n = season_number(spec)
    if n is not None:
        if 1 <= n <= len(tv):
            return tv[n - 1], n
    elif "final" in spec.lower() and tv:
        return tv[-1], len(tv)
    elif tv:
        want = _norm(spec)
        for r in tv:
            if any(want in t for t in _titles_of(r)):
                return r, number_of(r)

    raise RuntimeError(f"Season {spec!r} not found for {query!r}. "
                       + (f"Seasons found: {_season_list(tv)}" if tv else "No TV seasons matched the title."))


def resolve_seasons(records: list[dict[str, Any]], query: str,
                    numbers: list[int] | None) -> list[tuple[dict[str, Any], int]]:
    """Pick the catalogue entries for SEVERAL seasons. numbers=None means all of them."""
    tv = _tv_seasons(records, query)
    if not tv:
        raise RuntimeError(f"No TV seasons matched the title {query!r}.")
    wanted = list(range(1, len(tv) + 1)) if numbers is None else numbers
    missing = [n for n in wanted if not 1 <= n <= len(tv)]
    if missing:
        raise RuntimeError(f"Season(s) {missing} not found for {query!r}. Seasons found: {_season_list(tv)}")
    return [(tv[n - 1], n) for n in wanted[:MAX_SEASONS]]


# ===========================================================================
# Jikan v4
# ===========================================================================
class _Throttle:
    """Enforces a minimum interval between successive calls (async-safe)."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            remaining = self.delay_s - (time.monotonic() - self._last)
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last = time.monotonic()


_jikan_throttle = _Throttle(JIKAN_DELAY_S)


async def _jikan_get(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """GET a Jikan endpoint with 1 s spacing and backoff on 429/5xx."""
    for attempt in range(1, retries + 1):
        await _jikan_throttle.wait()
        r = await client.get(f"{JIKAN_BASE}{path}", params=params)
        if r.status_code == 429 or r.status_code >= 500:
            backoff = 2.0 * attempt
            log.warning("Jikan %s -> HTTP %s; retry in %.0fs", path, r.status_code, backoff)
            await asyncio.sleep(backoff)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Jikan {path} failed after {retries} attempts")


def _pick_best_anime(candidates: list[dict[str, Any]], query: str) -> dict[str, Any]:
    """Choose the search hit whose titles best match the query.

    Score: 3 = exact normalised title match, 2 = substring match either way,
    1 = TV series, 0 = otherwise. `max` returns the first maximum, so Jikan's
    own relevance order breaks ties.
    """
    q = _norm(query)

    def score(a: dict[str, Any]) -> int:
        titles = [a.get("title"), a.get("title_english"), a.get("title_japanese")]
        titles += [t.get("title") for t in a.get("titles", []) if isinstance(t, dict)]
        normed = [_norm(t) for t in titles if t]
        if q in normed:
            return 3
        if any(q in t or t in q for t in normed if t):
            return 2
        return 1 if a.get("type") == "TV" else 0

    return max(candidates, key=score)


# ---------------------------------------------------------------------------
# Episode sources. Each takes the `season` argument; a multi-season request
# ("all", "1-7", "1,3") returns the compact multi-season shape instead:
#   {"multi_season": true, "seasons": [{"season_number", "season_title",
#     "episode_count", "synopsis", "episodes": ["1. Title", ...]}], "reviews": [...]}
# Episode lines are plain strings ("11. Title") to keep seven seasons small
# enough for a local model's context window.
# ---------------------------------------------------------------------------
MAX_EPISODES_PER_SEASON = 60   # multi-season: episodes listed per season
EPISODE_TITLE_CHARS = 40       # multi-season: title length in the compact lines
SEASON_SYNOPSIS_CHARS = 200    # multi-season: one-line synopsis per season


def _season_block(number: int | None, title: str | None, count: Any, synopsis: str | None,
                  episodes: list[tuple[Any, str | None]]) -> dict[str, Any]:
    """One season of a multi-season result, episodes as compact "N. Title" lines."""
    return {
        "season_number": number,
        "season_title": title,
        "episode_count": count,
        "synopsis": _clean(synopsis, SEASON_SYNOPSIS_CHARS),
        "episodes": [f"{n}. {_clean(t, EPISODE_TITLE_CHARS) or '?'}" for n, t in episodes if n is not None],
    }


def _require_episodes(blocks: list[dict[str, Any]], source: str) -> None:
    if not any(b["episodes"] for b in blocks):
        raise RuntimeError(f"{source} returned no episode data for the requested seasons")


# ---------------------------------------------------------------------------
# Key episodes: rank episodes by how many character/arc words their title and
# synopsis contain, so the model is pointed at scenes the sources actually
# describe instead of guessing from titles.
# ---------------------------------------------------------------------------
KEY_EPISODES_SINGLE = 6      # single season: synopses are already in the list; just flag the best
KEY_EPISODES_MULTI = 12      # several seasons: these get their synopses included
KEY_SYNOPSIS_CHARS = 240
_FOCUS_STOP = {"with", "arc", "season", "seasons", "episode", "episodes", "from", "into", "the", "and",
               "their", "after", "before", "about", "story", "relationship", "character"}


def focus_tokens(character: str, arc_theme: str) -> frozenset[str]:
    """Words that mark an episode as relevant: the character's name parts and the arc's key words."""
    words = _norm(f"{character} {arc_theme}").split()
    return frozenset(w for w in words if len(w) >= 4 and w not in _FOCUS_STOP)


def key_episodes(data: dict[str, Any], focus: frozenset[str]) -> list[dict[str, Any]]:
    """Top episodes by focus-word hits (title hits count double), in story order."""
    if not focus:
        return []
    multi = bool(data.get("multi_season"))
    if multi:
        rows = data.get("_synopses") or []           # (season, ep, title, synopsis)
    else:
        rows = [(data.get("season_number"), e.get("ep"), e.get("title"), e.get("synopsis"))
                for e in data.get("episodes", []) if isinstance(e, dict)]
    scored = []
    for sn, ep, title, syn in rows:
        if ep is None:
            continue
        hit_t = focus & set(_norm(title or "").split())
        hit_s = focus & set(_norm(syn or "").split())
        score = 2 * len(hit_t) + len(hit_s)
        if score:
            scored.append((score, sn or 0, ep, title, syn, sorted(hit_t | hit_s)))
    scored.sort(key=lambda r: (-r[0], r[1], r[2]))
    out: list[dict[str, Any]] = []
    for _, sn, ep, title, syn, hits in scored[:KEY_EPISODES_MULTI if multi else KEY_EPISODES_SINGLE]:
        item: dict[str, Any] = {"ep": ep, "title": title, "mentions": hits}
        if multi:
            item = {"season": sn, **item, "synopsis": _clean(syn, KEY_SYNOPSIS_CHARS)}
        out.append(item)
    out.sort(key=lambda x: (x.get("season") or 0, x["ep"]))
    return out


# ------------------------------- Jikan v4 -----------------------------------
async def _jikan_records(client: httpx.AsyncClient, series_title: str) -> list[dict[str, Any]]:
    """Search MAL; every season is its own entry."""
    search = await _jikan_get(client, "/anime", {"q": series_title, "limit": 25})
    hits = search.get("data") or []
    if not hits:
        raise RuntimeError(f"No Jikan results for {series_title!r}")
    return [{
        "obj": a,
        "titles": [a.get("title"), a.get("title_english"), a.get("title_japanese")]
                  + [t.get("title") for t in a.get("titles", []) if isinstance(t, dict)],
        "type": a.get("type"),
        "start": (a.get("aired") or {}).get("from") or "",
    } for a in hits]


async def _jikan_episode_list(client: httpx.AsyncClient, mal_id: int, cap: int) -> list[dict[str, Any]]:
    """Paginated episode list (titles + flags; no synopses)."""
    episodes: list[dict[str, Any]] = []
    page = 1
    while len(episodes) < cap:
        try:
            data = await _jikan_get(client, f"/anime/{mal_id}/episodes", {"page": page})
        except httpx.HTTPStatusError:
            break  # films / OVAs may have no episode resource
        episodes.extend(data.get("data") or [])
        if not (data.get("pagination") or {}).get("has_next_page"):
            break
        page += 1
    return episodes[:cap]


async def _jikan_reviews(client: httpx.AsyncClient, mal_id: int) -> list[dict[str, Any]]:
    """Top reviews ranked by total reactions (community weight). Best effort."""
    out: list[dict[str, Any]] = []
    try:
        rv = await _jikan_get(client, f"/anime/{mal_id}/reviews", {"spoilers": "true"})
        reviews = rv.get("data") or []
        reviews.sort(key=lambda r: (r.get("reactions") or {}).get("overall", 0), reverse=True)
        for r in reviews[:MAX_REVIEWS]:
            out.append({"score": r.get("score"), "tags": r.get("tags"),
                        "excerpt": _clean(r.get("review"), REVIEW_CHARS)})
    except (httpx.HTTPError, RuntimeError) as exc:
        log.warning("Reviews unavailable: %s", exc)
    return out


async def _multi_via_jikan(series_title: str, numbers: list[int] | None,
                           focus: frozenset[str] = frozenset()) -> dict[str, Any]:
    async with _client() as client:
        chosen = resolve_seasons(await _jikan_records(client, series_title), series_title, numbers)
        blocks = []
        rows: list[list[Any]] = []          # [season, ep, title, synopsis, mal_id]
        for rec, n in chosen:
            a = rec["obj"]
            raw = await _jikan_episode_list(client, a["mal_id"], MAX_EPISODES_PER_SEASON)
            log.info("Jikan: season %s -> MAL %s (%s), %d episodes", n, a["mal_id"], a.get("title"), len(raw))
            blocks.append(_season_block(n, a.get("title_english") or a.get("title"), a.get("episodes"),
                                        a.get("synopsis"), [(e.get("mal_id"), e.get("title")) for e in raw]))
            rows += [[n, e.get("mal_id"), e.get("title"), None, a["mal_id"]] for e in raw if e.get("mal_id") is not None]
        _require_episodes(blocks, "Jikan")

        # The list endpoint has no synopses (one request per episode), so fetch them
        # only for the episodes whose titles mention the character or arc.
        if focus:
            ranked = sorted((r for r in rows if focus & set(_norm(r[2] or "").split())),
                            key=lambda r: -len(focus & set(_norm(r[2] or "").split())))
            for r in ranked[:KEY_EPISODES_MULTI]:
                try:
                    detail = await _jikan_get(client, f"/anime/{r[4]}/episodes/{r[1]}")
                    r[3] = (detail.get("data") or {}).get("synopsis")
                except (httpx.HTTPError, RuntimeError) as exc:
                    log.warning("Synopsis for S%s E%s unavailable: %s", r[0], r[1], exc)
        return {"multi_season": True, "seasons": blocks,
                "_synopses": [tuple(r[:4]) for r in rows],
                "reviews": await _jikan_reviews(client, chosen[0][0]["obj"]["mal_id"])}


async def _episodes_via_jikan(series_title: str, season: str,
                              focus: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Primary source (episodes + synopses + reviews). Raises on failure so
    the caller can fall back to another source."""
    kind, nums = parse_season_spec(season)
    if kind == "multi":
        return await _multi_via_jikan(series_title, nums, focus)

    async with _client() as client:
        rec, season_no = resolve_season(await _jikan_records(client, series_title), series_title, season)
        anime = rec["obj"]
        mal_id = anime["mal_id"]
        log.info("Jikan: %r (season %r) -> MAL %s (%s)", series_title, season, mal_id, anime.get("title"))

        episodes = await _jikan_episode_list(client, mal_id, MAX_EPISODES_LISTED)

        # Per-episode synopses (one call each, throttled, capped).
        ep_out: list[dict[str, Any]] = []
        for i, ep in enumerate(episodes):
            number = ep.get("mal_id")
            synopsis = ""
            if i < MAX_EPISODE_SYNOPSES:
                try:
                    detail = await _jikan_get(client, f"/anime/{mal_id}/episodes/{number}")
                    synopsis = _clean((detail.get("data") or {}).get("synopsis"), EPISODE_SYNOPSIS_CHARS)
                except (httpx.HTTPError, RuntimeError) as exc:
                    log.warning("Synopsis for ep %s unavailable: %s", number, exc)
            entry: dict[str, Any] = {"ep": number, "title": ep.get("title")}
            if synopsis:
                entry["synopsis"] = synopsis
            if ep.get("filler"):
                entry["filler"] = True
            if ep.get("recap"):
                entry["recap"] = True
            ep_out.append(entry)

        return {
            "season_number": season_no,
            "season_title": anime.get("title_english") or anime.get("title"),
            "mal_id": mal_id,
            "title": anime.get("title"),
            "title_english": anime.get("title_english"),
            "type": anime.get("type"),
            "episode_count": anime.get("episodes"),
            "series_synopsis": _clean(anime.get("synopsis"), 600),
            "episodes": ep_out,
            "reviews": await _jikan_reviews(client, mal_id),
        }


# --------------------- Fallback 1: Kitsu (no reviews) -----------------------
KITSU_BASE = "https://kitsu.io/api/edge"
KITSU_HEADERS = {"Accept": "application/vnd.api+json"}


async def _kitsu_records(client: httpx.AsyncClient, series_title: str) -> list[dict[str, Any]]:
    """Search Kitsu; each season is its own entry."""
    r = await client.get(f"{KITSU_BASE}/anime", headers=KITSU_HEADERS,
                         params={"filter[text]": series_title, "page[limit]": 20})
    r.raise_for_status()
    hits = r.json().get("data") or []
    if not hits:
        raise RuntimeError(f"No Kitsu results for {series_title!r}")

    def attrs_of(h: dict[str, Any]) -> dict[str, Any]:
        return h.get("attributes") or {}

    return [{
        "obj": h,
        "titles": [attrs_of(h).get("canonicalTitle")] + list((attrs_of(h).get("titles") or {}).values()),
        "type": attrs_of(h).get("subtype"),
        "start": attrs_of(h).get("startDate") or "",
    } for h in hits]


async def _kitsu_episode_list(client: httpx.AsyncClient, kid: str, cap: int) -> list[dict[str, Any]]:
    """Kitsu pages episodes 20 at a time; entries without a number are dropped."""
    raw: list[dict[str, Any]] = []
    offset = 0
    while len(raw) < cap:
        r = await client.get(f"{KITSU_BASE}/anime/{kid}/episodes", headers=KITSU_HEADERS,
                             params={"page[limit]": 20, "page[offset]": offset, "sort": "number"})
        r.raise_for_status()
        data = r.json().get("data") or []
        raw.extend(data)
        if len(data) < 20:
            break
        offset += 20
    return [e for e in raw[:cap] if (e.get("attributes") or {}).get("number") is not None]


def _kitsu_title(attrs: dict[str, Any]) -> str | None:
    return (attrs.get("titles") or {}).get("en") or attrs.get("canonicalTitle")


async def _multi_via_kitsu(series_title: str, numbers: list[int] | None,
                           focus: frozenset[str] = frozenset()) -> dict[str, Any]:
    async with _client() as client:
        chosen = resolve_seasons(await _kitsu_records(client, series_title), series_title, numbers)
        blocks = []
        synopses: list[tuple[Any, ...]] = []    # (season, ep, title, synopsis): Kitsu lists include synopses
        for rec, n in chosen:
            attrs = (rec["obj"].get("attributes") or {})
            raw = await _kitsu_episode_list(client, rec["obj"]["id"], MAX_EPISODES_PER_SEASON)
            log.info("Kitsu: season %s -> id %s (%s), %d episodes", n, rec["obj"]["id"], attrs.get("canonicalTitle"), len(raw))
            pairs = [(e["attributes"].get("number"),
                      e["attributes"].get("canonicalTitle") or (e["attributes"].get("titles") or {}).get("en_us"))
                     for e in raw]
            blocks.append(_season_block(n, _kitsu_title(attrs), attrs.get("episodeCount"), attrs.get("synopsis"), pairs))
            synopses += [(n, num, title, e["attributes"].get("synopsis")) for (num, title), e in zip(pairs, raw)]
        _require_episodes(blocks, "Kitsu")
        return {"multi_season": True, "seasons": blocks, "_synopses": synopses, "reviews": []}


async def _episodes_via_kitsu(series_title: str, season: str,
                              focus: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Raises on failure so the caller can try the next source."""
    kind, nums = parse_season_spec(season)
    if kind == "multi":
        return await _multi_via_kitsu(series_title, nums, focus)

    async with _client() as client:
        rec, season_no = resolve_season(await _kitsu_records(client, series_title), series_title, season)
        anime = rec["obj"]
        kid = anime["id"]
        attrs = anime.get("attributes") or {}
        log.info("Kitsu: %r (season %r) -> id %s (%s)", series_title, season, kid, attrs.get("canonicalTitle"))

        ep_out: list[dict[str, Any]] = []
        for i, e in enumerate(await _kitsu_episode_list(client, kid, MAX_EPISODES_LISTED)):
            a = e["attributes"]
            entry: dict[str, Any] = {"ep": a["number"],
                                     "title": a.get("canonicalTitle") or (a.get("titles") or {}).get("en_us")}
            syn = _clean(a.get("synopsis"), EPISODE_SYNOPSIS_CHARS) if i < MAX_EPISODE_SYNOPSES else ""
            if syn:
                entry["synopsis"] = syn
            ep_out.append(entry)
        if not ep_out:
            raise RuntimeError("Kitsu returned no episode data")

        return {
            "season_number": season_no,
            "season_title": _kitsu_title(attrs),
            "title": attrs.get("canonicalTitle"),
            "type": attrs.get("subtype"),
            "episode_count": attrs.get("episodeCount"),
            "series_synopsis": _clean(attrs.get("synopsis"), 600),
            "episodes": ep_out,
            "reviews": [],  # Kitsu has no usable public review endpoint
        }


# --------------- Fallback 2: TVMaze (all seasons in one request) ------------
TVMAZE_SINGLESEARCH = "https://api.tvmaze.com/singlesearch/shows"


async def _episodes_via_tvmaze(series_title: str, season: str,
                               focus: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Raises on failure (a 404 means no match). TVMaze keeps all seasons in one
    show, so the season filters the embedded episode list; blank means season 1."""
    async with _client() as client:
        r = await client.get(TVMAZE_SINGLESEARCH,
                             params=[("q", series_title), ("embed[]", "episodes"), ("embed[]", "seasons")])
        r.raise_for_status()
        show = r.json()
        embedded = show.get("_embedded") or {}
        eps = embedded.get("episodes") or []
        available = sorted({e["season"] for e in eps if e.get("season") is not None})
        if not available:
            raise RuntimeError("TVMaze returned no episodes")
        names = {s.get("number"): s.get("name") for s in embedded.get("seasons") or []}

        def title_of(n: int) -> str:
            return names.get(n) or f"{show.get('name')} season {n}"

        kind, nums = parse_season_spec(season)
        if kind == "multi":
            wanted = available if nums is None else nums
            missing = [n for n in wanted if n not in available]
            if missing:
                raise RuntimeError(f"Season(s) {missing} not found on TVMaze (seasons found: {available})")
            blocks = []
            synopses: list[tuple[Any, ...]] = []
            for n in wanted[:MAX_SEASONS]:
                in_season = [e for e in eps if e.get("season") == n and e.get("number") is not None][:MAX_EPISODES_PER_SEASON]
                chosen = [(e["number"], e.get("name")) for e in in_season]
                blocks.append(_season_block(n, title_of(n), len(chosen), None, chosen))
                synopses += [(n, e["number"], e.get("name"), re.sub(r"<[^>]+>", " ", e.get("summary") or ""))
                             for e in in_season]
            _require_episodes(blocks, "TVMaze")
            log.info("TVMaze: %r -> seasons %s", series_title, wanted)
            return {"multi_season": True, "seasons": blocks, "_synopses": synopses, "reviews": []}

        if kind == "default":
            want = available[0]
        elif kind == "single":
            want = nums[0]
        elif "final" in season.lower():
            want = available[-1]
        else:
            raise RuntimeError(f"TVMaze lists seasons by number only (found {available}); "
                               f"use a number such as 2, not {season!r}")
        if want not in available:
            raise RuntimeError(f"Season {want} not found on TVMaze (seasons found: {available})")

        log.info("TVMaze: %r -> %s, season %s", series_title, show.get("name"), want)
        chosen_eps = [e for e in eps if e.get("season") == want and e.get("number") is not None]
        ep_out: list[dict[str, Any]] = []
        for e in chosen_eps[:MAX_EPISODES_LISTED]:
            entry: dict[str, Any] = {"ep": e["number"], "title": e.get("name")}
            summary = re.sub(r"<[^>]+>", " ", e.get("summary") or "")  # strip HTML tags
            syn = _clean(summary, EPISODE_SYNOPSIS_CHARS) if len(ep_out) < MAX_EPISODE_SYNOPSES else ""
            if syn:
                entry["synopsis"] = syn
            ep_out.append(entry)
        if not ep_out:
            raise RuntimeError(f"TVMaze season {want} has no numbered episodes")

        return {
            "season_number": want,
            "season_title": title_of(want),
            "title": show.get("name"),
            "type": show.get("type"),
            "series_synopsis": _clean(re.sub(r"<[^>]+>", " ", show.get("summary") or ""), 600),
            "episodes": ep_out,
            "reviews": [],
        }


# ---------------------------------------------------------------------------
# MCP tool: tries each source in order, returns the first that works.
# ---------------------------------------------------------------------------
@mcp.tool()
async def fetch_anime_episodes_and_reviews(series_title: str, season: str = "",
                                           character: str = "", arc_theme: str = "") -> str:
    """Resolve an anime's season(s) and return the episode list (numbers within
    each season, titles, short synopses) plus top user review excerpts where
    available, as a JSON string. A single season returns "season_number" and
    "season_title"; a multi-season request returns "multi_season": true with a
    "seasons" list. Uses Jikan v4 (MyAnimeList) first; falls back to Kitsu, then
    TVMaze, if Jikan is unreachable. "source" says which one answered.

    Args:
        series_title: The anime's title, e.g. "My Hero Academia".
        season: Optional. A season number or name ("2", "Season 2", "Final Season"),
            or several seasons: "all", a range "1-7", or a list "1,3,5".
            Blank means the best title match, normally the first season.
        character: Optional character name; ranks "key_episodes" that mention them.
        arc_theme: Optional arc or theme; its key words also rank "key_episodes".
    """
    focus = focus_tokens(character, arc_theme)
    failures: list[str] = []
    for name, fetch in (("jikan", _episodes_via_jikan),
                        ("kitsu", _episodes_via_kitsu),
                        ("tvmaze", _episodes_via_tvmaze)):
        try:
            data = await fetch(series_title, season, focus)
            # Key episodes go FIRST: if the dossier has to be trimmed, the end of
            # this JSON (later seasons' title lists) is cut, not the best evidence.
            data = {"key_episodes": key_episodes(data, focus), **data}
            data.pop("_synopses", None)
            data["source"] = name
            if failures:
                data["fallback_note"] = "earlier sources failed: " + "; ".join(failures)
            return _dump(data)
        except Exception as exc:  # any failure -> try the next source
            # str(exc) is empty for some httpx errors (e.g. ConnectTimeout), so use repr.
            reason = f"{name}: {type(exc).__name__} {exc}".strip()
            log.warning("Episode source failed -> %s", reason)
            failures.append(reason)
    # Single "error" key so the orchestrator's source_failed() recognises it.
    return _dump({"error": "All episode sources failed. " + " | ".join(failures)})


# ===========================================================================
# AniList GraphQL (character bio + arc-relevant top reviews; no API key)
# ===========================================================================
ANILIST_URL = "https://graphql.anilist.co"
ANILIST_SYNOPSIS_CHARS = 500
ANILIST_CHARACTER_CHARS = 900
ANILIST_REVIEWS = 3
ANILIST_REVIEWS_MULTI = 4       # more reviews when several seasons are covered
ANILIST_REVIEW_CHARS = 600
ANILIST_REVIEW_POOL = 8         # top-rated reviews (per season) examined for arc relevance
MAX_ANILIST_SEASONS = 8         # request cap in multi-season mode (one query per season)

# Step 1: find the entries for the title. AniList lists every season as its
# own entry, so the requested season decides which id(s) to load in step 2.
_ANILIST_SEARCH = """
query ($q: String) {
  Page(perPage: 25) {
    media(search: $q, type: ANIME) {
      id
      format
      title { romaji english }
      startDate { year month day }
    }
  }
}
"""

# Step 2: one chosen entry. NOTE: the API ignored `limit` on reviews in testing
# (returned 25), so reviews are ranked and sliced client-side instead.
_ANILIST_QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    title { romaji english }
    episodes
    description(asHtml: false)
    characters(sort: [ROLE, RELEVANCE], perPage: 25) {
      nodes { name { full alternative } description(asHtml: false) }
    }
    reviews(sort: [RATING_DESC]) {
      nodes { score summary rating ratingAmount body(asHtml: false) }
    }
  }
}
"""


def _anilist_text(raw: str | None, limit: int) -> str:
    """Strip AniList markup (spoiler tags, image embeds, links, HTML) then truncate."""
    if not raw:
        return ""
    t = raw.replace("~!", " ").replace("!~", " ")  # keep spoiler text; drop the tags
    t = re.sub(r"img\d*\([^)]*\)", " ", t)          # img###(url) embeds
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[*_#>]+", " ", t)                   # markdown emphasis / headings / quotes
    return _clean(t, limit)


def _tokens(text: str, min_len: int) -> set[str]:
    """Distinct lower-case word tokens of at least `min_len` characters."""
    return {t for t in _norm(text).split() if len(t) >= min_len}


def _best_paragraph(body: str, tokens: set[str]) -> tuple[int, str]:
    """Return (score, paragraph) for the paragraph that mentions the most
    distinct arc/character tokens. Score 0 -> falls back to the opening text."""
    best_score, best = 0, ""
    for para in re.split(r"\n+", body or ""):
        if len(para.strip()) < 40:
            continue
        words = set(_norm(para).split())
        score = len(tokens & words)
        if score > best_score:
            best_score, best = score, para
    return best_score, (best or (body or ""))


async def _anilist_post(client: httpx.AsyncClient, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST one GraphQL query; honours Retry-After once on HTTP 429."""
    payload = {"query": query, "variables": variables}
    r = await client.post(ANILIST_URL, json=payload)
    if r.status_code == 429:  # AniList rate limit
        try:
            wait = min(30.0, float(r.headers.get("Retry-After", 5)))
        except ValueError:
            wait = 5.0
        log.warning("AniList 429; waiting %.0fs", wait)
        await asyncio.sleep(wait)
        r = await client.post(ANILIST_URL, json=payload)
    r.raise_for_status()
    return r.json()


def _pick_character(nodes: list[dict[str, Any]], character: str) -> dict[str, Any] | None:
    """Match a character by name tokens (both tokens of a two-part name must match)."""
    c_tokens = _tokens(character, 3)

    def name_hits(n: dict[str, Any]) -> int:
        nm = n.get("name") or {}
        names = [nm.get("full")] + list(nm.get("alternative") or [])
        return len(c_tokens & set(_norm(" ".join(x for x in names if x)).split()))

    match = max(nodes, key=name_hits, default=None)
    if match and c_tokens and name_hits(match) >= min(2, len(c_tokens)):
        return {"name": match["name"]["full"],
                "bio": _anilist_text(match.get("description"), ANILIST_CHARACTER_CHARS)}
    return None


def _rank_reviews(pool: list[tuple[int | None, dict[str, Any]]], tokens: set[str], limit: int) -> list[dict[str, Any]]:
    """pool: [(season_number or None, review)]. Prefer reviews whose text discusses
    the arc (most keyword hits), then the most helpful; excerpt = best paragraph."""
    scored = []
    for sn, rv in pool:
        score, para = _best_paragraph(rv.get("body") or "", tokens)
        scored.append((score, rv.get("rating") or 0, sn, rv, para))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    out = []
    for _, _, sn, rv, para in scored[:limit]:
        item = {"score": rv.get("score"),
                "helpful": f"{rv.get('rating')}/{rv.get('ratingAmount')}",
                "summary": _clean(rv.get("summary"), 160),
                "excerpt": _anilist_text(para, ANILIST_REVIEW_CHARS)}
        out.append({"season": sn, **item} if sn is not None else item)
    return out


def _top_reviews(media: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = (media.get("reviews") or {}).get("nodes") or []
    return sorted(nodes, key=lambda x: x.get("rating") or 0, reverse=True)[:ANILIST_REVIEW_POOL]


@mcp.tool()
async def fetch_anilist_character_and_reviews(series_title: str, character: str,
                                              arc_theme: str, season: str = "") -> str:
    """Fetch the series synopsis, the character's bio, and the most
    arc-relevant excerpts from top-rated user reviews from AniList, as a JSON
    string. Reviews are ranked by how many arc/character keywords they mention.
    A single season returns "season_number"/"season_title"; several seasons
    return "multi_season": true with a "seasons" list and per-review "season".

    Args:
        series_title: The anime's title.
        character: Character name, e.g. "Sakuta Azusagawa".
        arc_theme: The arc or theme, e.g. "Kaede's Memory Loss & Grief".
        season: Optional season number or name ("2", "Final Season"), or several:
            "all", "1-7", "1,3,5". Blank = best title match.
    """
    try:
        async with _client() as client:
            # --- step 1: which entries are the requested season(s)? ---
            found = await _anilist_post(client, _ANILIST_SEARCH, {"q": series_title})
            hits = ((found.get("data") or {}).get("Page") or {}).get("media") or []
            if not hits:
                return _dump({"error": f"No AniList match for {series_title!r}"})

            def start_of(m: dict[str, Any]) -> str:
                d = m.get("startDate") or {}
                return f"{d['year']:04d}-{(d.get('month') or 1):02d}-{(d.get('day') or 1):02d}" if d.get("year") else ""

            records = [{"obj": m,
                        "titles": [(m.get("title") or {}).get("english"), (m.get("title") or {}).get("romaji")],
                        "type": m.get("format"), "start": start_of(m)} for m in hits]
            r_tokens = _tokens(arc_theme, 4) | _tokens(character, 4)
            kind, nums = parse_season_spec(season)

            # --- step 2a: several seasons -> pool characters and reviews across them ---
            if kind == "multi":
                chosen = resolve_seasons(records, series_title, nums)[:MAX_ANILIST_SEASONS]
                loaded: list[tuple[int, dict[str, Any]]] = []
                for rec, n in chosen:
                    data = await _anilist_post(client, _ANILIST_QUERY, {"id": rec["obj"]["id"]})
                    media = (data.get("data") or {}).get("Media")
                    if media:
                        loaded.append((n, media))
                if not loaded:
                    return _dump({"error": "AniList returned no data for the requested seasons"})

                char_out = None
                for _, media in loaded:
                    char_out = _pick_character((media.get("characters") or {}).get("nodes") or [], character)
                    if char_out:
                        break
                if not char_out:
                    first_nodes = (loaded[0][1].get("characters") or {}).get("nodes") or []
                    char_out = {"note": f"{character!r} not among AniList's top characters",
                                "main_characters": [n["name"]["full"] for n in first_nodes[:10]]}
                pool = [(n, rv) for n, media in loaded for rv in _top_reviews(media)]
                seasons = [{"season_number": n,
                            "season_title": (m.get("title") or {}).get("english") or (m.get("title") or {}).get("romaji")}
                           for n, m in loaded]
                log.info("AniList: %r -> seasons %s", series_title, [s["season_number"] for s in seasons])
                return _dump({"multi_season": True, "seasons": seasons, "character": char_out,
                              "reviews": _rank_reviews(pool, r_tokens, ANILIST_REVIEWS_MULTI)})

            # --- step 2b: one season ---
            rec, season_no = resolve_season(records, series_title, season)
            data = await _anilist_post(client, _ANILIST_QUERY, {"id": rec["obj"]["id"]})
            media = (data.get("data") or {}).get("Media")
            if not media:
                return _dump({"error": f"AniList returned no data for id {rec['obj']['id']}"})
            title = media.get("title") or {}
            season_title = title.get("english") or title.get("romaji")
            log.info("AniList: %r (season %r) -> %s", series_title, season, season_title)

            nodes = (media.get("characters") or {}).get("nodes") or []
            char_out = _pick_character(nodes, character) or {
                "note": f"{character!r} not among AniList's top characters",
                "main_characters": [n["name"]["full"] for n in nodes[:10]]}
            return _dump({
                "season_number": season_no,
                "season_title": season_title,
                "title": season_title,
                "episodes": media.get("episodes"),
                "synopsis": _anilist_text(media.get("description"), ANILIST_SYNOPSIS_CHARS),
                "character": char_out,
                "reviews": _rank_reviews([(None, rv) for rv in _top_reviews(media)], r_tokens, ANILIST_REVIEWS),
            })
    except Exception as exc:
        log.warning("fetch_anilist_character_and_reviews failed: %s", exc)
        # str(exc) is empty for some httpx errors (e.g. ConnectTimeout), so fall back to repr.
        return _dump({"error": f"{type(exc).__name__}: {str(exc) or repr(exc)}"})


# ===========================================================================
# Reddit public JSON
# ===========================================================================
async def _reddit_search(client: httpx.AsyncClient, query: str) -> dict[str, Any]:
    """Query Reddit search; fall back to old.reddit.com on block/failure."""
    params = {"q": query, "sort": "relevance", "limit": REDDIT_LIMIT,
              "type": "link", "raw_json": 1}
    last_err = "unknown error"
    for base in (REDDIT_SEARCH, REDDIT_SEARCH_FALLBACK):
        try:
            r = await client.get(base, params=params)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code} from {base}"
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__} from {base}: {exc}"
        log.warning("Reddit search failed: %s", last_err)
    raise RuntimeError(last_err)


def _reddit_posts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten Reddit listing JSON into compact post records."""
    posts = []
    for child in (payload.get("data") or {}).get("children", []):
        d = child.get("data") or {}
        post: dict[str, Any] = {
            "title": d.get("title"),
            "subreddit": d.get("subreddit_name_prefixed"),
            "score": d.get("score"),
            "comments": d.get("num_comments"),
            "url": f"https://www.reddit.com{d.get('permalink', '')}",
        }
        body = _clean(d.get("selftext"), REDDIT_SELFTEXT_CHARS)
        if body:
            post["selftext"] = body
        if not d.get("is_self") and d.get("url"):
            post["linked_url"] = d["url"]
        posts.append(post)
    return posts


_ATOM = "{http://www.w3.org/2005/Atom}"


async def _reddit_rss(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    """Fallback: Reddit's public Atom search feed (a separate endpoint that is
    sometimes reachable when search.json returns 403)."""
    r = await client.get("https://www.reddit.com/search.rss",
                         params={"q": query, "sort": "relevance", "limit": REDDIT_LIMIT, "type": "link"},
                         headers={"Accept": "application/atom+xml, application/xml"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from search.rss")
    posts: list[dict[str, Any]] = []
    for entry in ET.fromstring(r.text).findall(f"{_ATOM}entry"):
        link, cat = entry.find(f"{_ATOM}link"), entry.find(f"{_ATOM}category")
        body = html.unescape(re.sub(r"<[^>]+>", " ", entry.findtext(f"{_ATOM}content") or ""))
        body = re.sub(r"submitted by.*$", "", re.sub(r"\s+", " ", body)).strip()  # feed boilerplate
        post: dict[str, Any] = {
            "title": entry.findtext(f"{_ATOM}title"),
            "subreddit": cat.get("label") if cat is not None else None,
            "url": link.get("href") if link is not None else None,
        }
        if body:
            post["selftext"] = _clean(body, REDDIT_SELFTEXT_CHARS)
        posts.append(post)
    return posts


async def _reddit_find(client: httpx.AsyncClient, query: str) -> list[dict[str, Any]]:
    """JSON search first (as specified); on failure fall back to the RSS feed."""
    try:
        return _reddit_posts(await _reddit_search(client, query))
    except RuntimeError as json_err:
        log.warning("Reddit JSON failed (%s); trying RSS", json_err)
        try:
            return await _reddit_rss(client, query)
        except Exception as rss_err:
            raise RuntimeError(f"JSON: {json_err} | RSS: {rss_err}") from rss_err


@mcp.tool()
async def scrape_reddit_character_lore(series_title: str, character: str, arc_theme: str) -> str:
    """Search Reddit for fan analyses and episode-discussion threads about a
    character arc. Runs several queries, removes duplicates, prefers posts with
    text that mention the character or arc, and returns titles, subreddits,
    scores, truncated selftext and URLs as a JSON string.

    Args:
        series_title: The anime's title.
        character: Character name, e.g. "Sakuta Azusagawa".
        arc_theme: The arc or theme, e.g. "Kaede's Memory Loss & Grief".
    """
    focus = focus_tokens(character, arc_theme)
    queries = [f"{series_title} {character} {arc_theme} episode",   # most specific first
               f"{series_title} {character} {arc_theme}",
               f"{series_title} {character} character development",
               f"{series_title} {character}"]
    try:
        async with _client() as client:
            found: dict[str, dict[str, Any]] = {}
            used: list[str] = []
            errors: list[str] = []
            for q in queries:
                try:
                    posts = await _reddit_find(client, q)
                except RuntimeError as exc:      # blocked: no point trying the other queries
                    errors.append(str(exc))
                    break
                used.append(q)
                for post in posts:
                    found.setdefault(post.get("url") or post.get("title") or str(len(found)), post)
                if sum(1 for p in found.values() if p.get("selftext")) >= REDDIT_LIMIT:
                    break
            if not found and errors:
                raise RuntimeError(errors[0])

            def relevance(post: dict[str, Any]) -> tuple[int, int, int]:
                text = f"{post.get('title', '')} {post.get('selftext', '')}"
                return (len(focus & set(_norm(text).split())), int(bool(post.get("selftext"))), post.get("score") or 0)

            ranked = sorted(found.values(), key=relevance, reverse=True)[:REDDIT_LIMIT]
            return _dump({"queries": used, "posts": ranked})
    except Exception as exc:
        log.warning("scrape_reddit_character_lore failed: %s", exc)
        return _dump({"error": f"{type(exc).__name__}: {str(exc) or repr(exc)}"})


# ===========================================================================
# Wikipedia / MediaWiki
# ===========================================================================
_KEEP_HEADINGS = re.compile(r"plot|synopsis|story|premise|summary|characters?", re.I)
_DROP_HEADINGS = re.compile(
    r"reception|references|external links|see also|notes|release|music|"
    r"production|sales|broadcast|home media|bibliography|further reading", re.I)


def _split_sections(extract: str) -> list[tuple[str, str]]:
    """Split a plaintext extract (exsectionformat=wiki) into (heading, body)."""
    sections: list[tuple[str, str]] = []
    heading, buf = "Introduction", []
    for line in extract.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", line)
        if m:
            sections.append((heading, "\n".join(buf)))
            heading, buf = m.group(2), []
        else:
            buf.append(line)
    sections.append((heading, "\n".join(buf)))
    return [(h, b.strip()) for h, b in sections if b.strip()]


def _relevant_text(extract: str, character: str, limit: int) -> str:
    """Assemble the most arc-relevant prose from an article within `limit`.

    Priority: (1) short lead, (2) plot/character sections, (3) paragraphs
    elsewhere that mention any token of the character's name.
    """
    tokens = {t.lower() for t in re.split(r"\W+", character) if len(t) >= 3}
    sections = _split_sections(extract)
    pieces: list[str] = []

    if sections and sections[0][0] == "Introduction":
        pieces.append(_clean(sections[0][1], 400))
    for head, body in sections[1:]:
        if _KEEP_HEADINGS.search(head) and not _DROP_HEADINGS.search(head):
            pieces.append(f"[{head}] " + _clean(body, limit))
    for head, body in sections[1:]:
        if _KEEP_HEADINGS.search(head) or _DROP_HEADINGS.search(head):
            continue
        for para in body.split("\n"):
            if tokens and any(t in para.lower() for t in tokens):
                pieces.append(f"[{head}] " + _clean(para, 500))

    return _clean(" ".join(pieces), limit)


async def _wiki_get(client: httpx.AsyncClient, params: dict[str, Any]) -> httpx.Response:
    """GET the MediaWiki API, retrying 403/429/5xx with backoff (Wikimedia
    throttles generic clients). Returns the last response; callers raise_for_status."""
    r = await client.get(WIKI_API, params=params)
    for attempt in range(1, 3):
        if r.status_code not in (403, 429) and r.status_code < 500:
            break
        log.warning("Wikipedia HTTP %s; retry %d/2", r.status_code, attempt)
        await asyncio.sleep(2.0 * attempt)
        r = await client.get(WIKI_API, params=params)
    return r


async def _wiki_search(client: httpx.AsyncClient, query: str, n: int) -> list[str]:
    """Return up to `n` article titles for a full-text search."""
    r = await _wiki_get(client, {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": n, "format": "json", "formatversion": 2,
    })
    r.raise_for_status()
    return [h["title"] for h in (r.json().get("query") or {}).get("search", [])]


async def _wiki_extract(client: httpx.AsyncClient, title: str) -> str:
    """Full plaintext extract of one article (full extracts are 1 page/request)."""
    r = await _wiki_get(client, {
        "action": "query", "prop": "extracts", "titles": title,
        "explaintext": 1, "exsectionformat": "wiki", "redirects": 1,
        "format": "json", "formatversion": 2,
    })
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages", [])
    return (pages[0].get("extract") or "") if pages else ""


@mcp.tool()
async def fetch_wiki_character_summary(series_title: str, character: str) -> str:
    """Fetch plot-progression and character details from English Wikipedia.
    Searches for the series and the character, then returns the lead, plot /
    character sections and character-mentioning paragraphs as a JSON string.

    Args:
        series_title: The anime's title.
        character: Character name.
    """
    try:
        async with _client() as client:
            # Series page first (usually holds Plot/Characters), then the
            # character-specific search (may hit a "List of characters" page).
            titles: list[str] = []
            for q, n in ((series_title, 1), (f"{series_title} {character}", 3)):
                for t in await _wiki_search(client, q, n):
                    if t not in titles:
                        titles.append(t)
            titles = titles[:WIKI_MAX_PAGES]
            if not titles:
                return _dump({"error": f"No Wikipedia results for {series_title!r}"})

            pages = []
            for t in titles:
                text = _relevant_text(await _wiki_extract(client, t), character, WIKI_CHARS_PER_PAGE)
                if text:
                    pages.append({
                        "title": t,
                        "url": f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}",
                        "content": text,
                    })
            return _dump({"pages": pages})
    except Exception as exc:
        log.exception("fetch_wiki_character_summary failed")
        return _dump({"error": f"{type(exc).__name__}: {exc}"})


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")
