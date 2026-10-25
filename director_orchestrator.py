#!/usr/bin/env python3
"""
director_orchestrator.py — Phase 1 orchestrator (6–8 GB VRAM profile).

Pipeline:
    1. Spawn lore_mcp_server.py over stdio and list its MCP tools.
    2. Research phase: bridge MCP tools into an ollama.chat() tool-calling
       loop. Any tool the model fails to call is invoked directly (small
       models frequently stop early).
    3. Synthesis phase: one fresh, compact call constrained by
       format=TimelineBlueprint.model_json_schema(), temperature 0.2.
    4. Validate with Pydantic (one corrective retry), post-process, write JSON.

VRAM budget (Q4_K_M weights + KV cache at num_ctx = 8192):
    qwen2.5:7b   ~4.7 GB + ~0.45 GB KV  -> fits 6 GB   (default)
    qwen3:8b     ~5.2 GB + ~0.6  GB KV  -> fits 8 GB
    llama3.1:8b  ~4.9 GB + ~1.0  GB KV  -> fits 8 GB, spills on 6 GB
    qwen2.5:3b   ~1.9 GB + ~0.3  GB KV  -> 6 GB with headroom, weaker output
Context is the main lever: KV cache grows linearly with --num-ctx.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# Defaults tuned for 6–8 GB VRAM.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:7b"     # strongest tool-caller that fits 6 GB at 8k ctx
DEFAULT_NUM_CTX = 8192           # KV-cache size; the dominant VRAM knob
DEFAULT_MAX_BEATS = 8            # bounds output tokens for the synthesis call
MAX_TOOL_STEPS = 6               # hard cap on research-loop model turns
RESEARCH_TOOL_RESULT_CHARS = 600   # preview fed back during research (full text kept for synthesis)
CHARS_PER_TOKEN = 3.0            # conservative estimate for budget arithmetic
PROMPT_RESERVE_TOKENS = 1500     # system prompt + instructions + slack

# Models that emit reasoning traces; thinking is disabled to save context.
THINKING_PREFIXES = ("qwen3", "deepseek-r1", "magistral")

SILENT_BEAT = "None (Silent Visual Beat)"

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[director] %(levelname)s %(message)s")
log = logging.getLogger("director_orchestrator")


# ===========================================================================
# Output schema
# ---------------------------------------------------------------------------
# Constraints are enforced by validators rather than JSON-schema keywords
# (pattern / minimum / minItems), because grammar-constrained decoding does
# not reliably honour every keyword. The schema stays exactly as specified;
# Pydantic is the authority on validity.
# ===========================================================================
# Tolerates approximation markers the model likes to add ("~02:30 - ~03:00",
# "≈2:30–3:00") and a range separator of -, –, —, or "to". The marker is
# discarded; the normalised output is always "MM:SS - MM:SS".
_TS_RE = re.compile(
    r"^\s*[~≈]?\s*(\d{1,2}):(\d{2})\s*(?:-|–|—|to)\s*[~≈]?\s*(\d{1,2}):(\d{2})\s*$"
)


class ClipBeat(BaseModel):
    clip_index: int = Field(description="1-based position of the clip in the edit")
    episode_number: int = Field(description="Episode number from the episode list (counts within its season)")
    # Single-season runs: set by the orchestrator. Multi-season runs: chosen by the model
    # (required in the schema it sees, see llm_schema) and verified in postprocess.
    season_number: int | None = Field(default=None, description="Season the clip's episode belongs to")
    episode_title: str = Field(description="Episode title exactly as in the episode list")
    # Grounding: words copied from the research dossier that justify the clip. Placed
    # before the creative fields so the model commits to a source first, then writes.
    evidence: str = Field(default="", description="8-25 words copied exactly from the research dossier that support this clip")
    timestamp_estimate: str = Field(description='Estimated in-episode range, format "MM:SS - MM:SS"')
    emotional_beat: str = Field(description="The emotional shift this clip carries")
    dialogue_anchor: str = Field(description=f'Exact memorable quote, or "{SILENT_BEAT}"')
    community_consensus: str = Field(description="Why fans consider this moment pivotal, from sources")
    director_notes: str = Field(description="Framing, pacing speed, J-cut/L-cut audio instructions")
    # Set by the orchestrator (hidden from the model): was the evidence found in the dossier?
    evidence_verified: bool | None = Field(default=None, description="Evidence found in the research dossier")

    @field_validator("timestamp_estimate")
    @classmethod
    def _normalise_timestamp(cls, v: str) -> str:
        """Accept loose variants (e.g. '3:05-4:10') and emit 'MM:SS - MM:SS'."""
        m = _TS_RE.match(v)
        if not m:
            raise ValueError(f'must match "MM:SS - MM:SS", got {v!r}')
        m1, s1, m2, s2 = map(int, m.groups())
        # Work in total seconds (so "12:75" still means 13:15).
        start, end = m1 * 60 + s1, m2 * 60 + s2
        # Small models often emit zero-length ("24:00 - 24:00") or inverted
        # ranges. These are estimates anyway, so repair rather than reject
        # (a rejection just makes the model repeat itself on retry):
        if end < start:                      # swapped -> reorder
            start, end = end, start
        if end == start:                     # zero-length -> nominal 30 s clip ending here
            start = max(0, end - 30)
        if end <= start:                     # only possible at 00:00 -> 30 s clip from the start
            end = start + 30
        return f"{start // 60:02d}:{start % 60:02d} - {end // 60:02d}:{end % 60:02d}"

    @field_validator("dialogue_anchor")
    @classmethod
    def _default_silent(cls, v: str) -> str:
        """An empty quote is coerced to the silent-beat sentinel."""
        v = v.strip()
        return v if v else SILENT_BEAT

    @field_validator("episode_number")
    @classmethod
    def _positive_episode(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"episode_number must be ≥ 1, got {v}")
        return v


class SeasonRef(BaseModel):
    season_number: int
    season_title: str = ""


class TimelineBlueprint(BaseModel):
    series_title: str
    character: str
    arc_theme: str
    narrative_thesis: str = Field(description="2–3 sentence thesis of the arc as an edit")
    timeline: list[ClipBeat] = Field(description="Beats in arc order")
    # Filled in by the orchestrator from the research data (see postprocess); hidden
    # from the model via llm_schema(). Episode numbers refer to THIS season.
    season_number: int | None = Field(default=None, description="Season the episodes belong to")
    season_title: str = Field(default="", description="Catalogue title of that season (or the range, e.g. Seasons 1-7)")
    seasons: list[SeasonRef] = Field(default_factory=list, description="All seasons covered (multi-season runs)")

    @model_validator(mode="after")
    def _non_empty(self) -> "TimelineBlueprint":
        if not self.timeline:
            raise ValueError("timeline must contain at least one ClipBeat")
        return self


def llm_schema(multi: bool = False) -> dict[str, Any]:
    """TimelineBlueprint JSON schema for the model.

    Season fields the orchestrator fills itself are hidden. In a multi-season run
    the model must say which season each clip's episode belongs to, so ClipBeat's
    season_number is shown and required there; otherwise it is hidden too.
    """
    schema = TimelineBlueprint.model_json_schema()
    for key in ("season_number", "season_title", "seasons"):
        schema.get("properties", {}).pop(key, None)
    clip = schema.get("$defs", {}).get("ClipBeat", {})
    clip.get("properties", {}).pop("evidence_verified", None)
    if "evidence" not in clip.setdefault("required", []):
        clip["required"].append("evidence")
    if multi:
        clip.setdefault("properties", {})["season_number"] = {
            "type": "integer", "minimum": 1,
            "description": "Season of the clip's episode, as listed in the dossier's seasons"}
        if "season_number" not in clip.setdefault("required", []):
            clip["required"].append("season_number")
    else:
        clip.get("properties", {}).pop("season_number", None)
    return schema


# ===========================================================================
# Prompts
# ===========================================================================
RESEARCH_SYSTEM = (
    "You are a lore research agent for anime video editors.\n"
    "You have several research tools. Call EACH tool exactly once, using the "
    "series, character and arc supplied by the user. Do not answer from "
    "memory. When all results have been returned, reply with the "
    "single line: RESEARCH COMPLETE"
)

SYNTH_SYSTEM = (
    "You are a film editor and narrative director. You convert research "
    "dossiers into precise clip timelines for a character-arc edit. You "
    "state only what the dossier supports, never invent quotes or episodes, "
    "and output only JSON matching the supplied schema."
)


def synthesis_prompt(ctx: dict[str, str], dossier: str, max_beats: int, beats: int | None = None) -> str:
    """User prompt for the structured synthesis call."""
    multi = bool(ctx.get("multi_season"))
    season_line = (f"\n{'SEASONS' if multi else 'SEASON'}: {ctx['season_label']}"
                   if ctx.get("season_label") else "")
    order_note = " (earlier seasons first, then by episode; spread the clips over the seasons where the arc develops)" if multi else ""
    episode_rule = (
        "season_number, episode_number and episode_title must match an entry in the dossier's seasons list; "
        "episode_number counts within that season (every season restarts at 1)."
        if multi else
        "episode_number and episode_title must come from the dossier's episode list (numbers count within the stated season).")
    count_rule = (f"EXACTLY {beats} ClipBeats (no more, no fewer)" if beats
                  else f"4 to {max_beats} ClipBeats")
    return f"""SERIES: {ctx['series_title']}{season_line}
CHARACTER: {ctx['character']}
ARC: {ctx['arc_theme']}

=== RESEARCH DOSSIER ===
{dossier}
=== END DOSSIER ===

Produce a TimelineBlueprint for this arc. Rules:
1. timeline: {count_rule} in the arc's narrative order{order_note}; clip_index starts at 1. Every clip is a different scene.
2. {episode_rule} Prefer the dossier's key_episodes: their titles or synopses mention the character or the arc.
3. evidence: copy 8 to 25 consecutive words exactly from the dossier (an episode synopsis, review or post) that show why this scene belongs in the arc. It is checked against the dossier.
4. timestamp_estimate: "MM:SS - MM:SS" inside a ~24-minute episode (00:00–24:00), clips of 10–60 s; place it using the synopsis and avoid the opening and ending songs.
5. dialogue_anchor: copy a line exactly ONLY if it appears in the dossier; otherwise exactly "{SILENT_BEAT}". Lines are checked and invented ones are removed. Never use the same line twice.
6. emotional_beat and community_consensus: write them about THIS clip's scene; never reuse a sentence from another clip. community_consensus names its source type (Reddit, AniList review, Wikipedia, synopsis).
7. director_notes: shot framing, pacing (slow/medium/fast), and a concrete J-cut or L-cut audio instruction.
8. narrative_thesis: 2–3 sentences stating what the edit argues about {ctx['character']}.
9. series_title, character, arc_theme: copy exactly as given above."""


# ===========================================================================
# MCP <-> Ollama bridging
# ===========================================================================
def _strip_schema_titles(schema: Any) -> Any:
    """Remove Pydantic-generated 'title' strings from a JSON schema.

    They add tokens without information. Only string-valued 'title' keys are
    dropped, so a *property* literally named 'title' (a dict) survives.
    """
    if isinstance(schema, dict):
        return {k: _strip_schema_titles(v) for k, v in schema.items()
                if not (k == "title" and isinstance(v, str))}
    if isinstance(schema, list):
        return [_strip_schema_titles(v) for v in schema]
    return schema


def mcp_tool_to_ollama(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool definition to Ollama's function-tool format."""
    params = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": _strip_schema_titles(params),
        },
    }


def sanitise_args(raw: Any, schema: dict[str, Any], ctx: dict[str, str]) -> dict[str, Any]:
    """Build tool arguments from the model's call, pinned to CLI context.

    Small models paraphrase or misspell series/character names, which wrecks
    search recall. Any parameter whose name exists in `ctx` is therefore
    pinned to the CLI value; other parameters are taken from the model.
    Unknown keys are discarded.
    """
    if isinstance(raw, str):  # some models emit arguments as a JSON string
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    args: dict[str, Any] = {}
    for key in (schema.get("properties") or {}):
        if key in ctx:
            args[key] = ctx[key]
        elif key in raw:
            args[key] = raw[key]
    return args


async def call_mcp_tool(session: ClientSession, name: str, args: dict[str, Any]) -> str:
    """Invoke an MCP tool and return its concatenated text content."""
    result = await session.call_tool(name, args)
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    if getattr(result, "isError", False):
        text = json.dumps({"error": text or "tool reported an error"})
    # Diagnostic: size + head of every result, so failed scrapes are obvious.
    log.info("<- %s: %d chars | %s", name, len(text), text[:160].replace("\n", " "))
    return text


def source_failed(text: str) -> bool:
    """True if a tool result is an error payload (server returns {"error": ...})."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and set(data) == {"error"}


def truncate(text: str, limit: int) -> str:
    """Hard character cap with a visible marker."""
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


# ===========================================================================
# Phase A — research
# ===========================================================================
async def research_phase(
    llm: AsyncClient,
    session: ClientSession,
    model: str,
    ctx: dict[str, str],
    options: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, str]:
    """Run the tool-calling loop; return {tool_name: full_result_text}."""
    listed = await session.list_tools()
    tools = [mcp_tool_to_ollama(t) for t in listed.tools]
    schemas = {t.name: (t.inputSchema or {}) for t in listed.tools}
    log.info("MCP tools: %s", ", ".join(schemas))

    lore: dict[str, str] = {}  # full, untruncated results (used later)
    messages: list[Any] = [
        {"role": "system", "content": RESEARCH_SYSTEM},
        {"role": "user", "content": (
            f'Series: "{ctx["series_title"]}"\n'
            f'Character: "{ctx["character"]}"\n'
            f'Arc: "{ctx["arc_theme"]}"\n'
            + (f'Season: "{ctx["season"]}"\n' if ctx.get("season") else "")
            + "Gather the lore now."
        )},
    ]

    for step in range(1, MAX_TOOL_STEPS + 1):
        resp = await llm.chat(model=model, messages=messages, tools=tools,
                              options=options, **extra)
        msg = resp.message
        messages.append(msg)  # Message objects are accepted back by ollama.chat
        calls = msg.tool_calls or []
        log.info("Research step %d: %d tool call(s)", step, len(calls))
        if not calls:
            break

        for call in calls:
            name = call.function.name
            if name not in schemas:
                content = json.dumps({"error": f"unknown tool {name!r}"})
            elif name in lore:
                content = "Already fetched; use the earlier result."  # no duplicate HTTP work
            else:
                args = sanitise_args(call.function.arguments, schemas[name], ctx)
                log.info("-> %s(%s)", name, args)
                lore[name] = await call_mcp_tool(session, name, args)
                content = truncate(lore[name], RESEARCH_TOOL_RESULT_CHARS)
            messages.append({"role": "tool", "content": content, "tool_name": name})

        if all(n in lore for n in schemas):
            break  # everything gathered; skip a wasted model turn

    # Coverage guarantee: call anything the model skipped.
    for name, schema in schemas.items():
        if name not in lore:
            args = sanitise_args({}, schema, ctx)
            log.warning("Model skipped %s; calling directly with %s", name, args)
            lore[name] = await call_mcp_tool(session, name, args)

    return lore


# ===========================================================================
# Phase B — structured synthesis
# ===========================================================================
def build_dossier(lore: dict[str, str], budget_chars: int) -> str:
    """Concatenate tool results within a character budget (water-filling).

    Sources shorter than their fair share keep everything; the surplus is
    redistributed to longer sources, so truncation only hits what must be cut.
    """
    remaining = dict(lore)
    alloc: dict[str, int] = {}
    budget = budget_chars
    while remaining:
        share = max(1000, budget // len(remaining))
        short = {n: t for n, t in remaining.items() if len(t) <= share}
        if not short:                      # everyone left is over the share
            alloc.update({n: share for n in remaining})
            break
        for n, t in short.items():
            alloc[n] = len(t)
            budget -= len(t)
            del remaining[n]
    return "\n\n".join(f"### SOURCE: {n}\n{truncate(t, alloc[n])}" for n, t in lore.items())


EPISODES_TOOL = "fetch_anime_episodes_and_reviews"
ANILIST_TOOL = "fetch_anilist_character_and_reviews"


def _tool_json(lore: dict[str, str], tool: str) -> dict[str, Any]:
    """A tool's result as a dict ({} if absent or not JSON)."""
    try:
        data = json.loads(lore.get(tool, "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def known_episodes(lore: dict[str, str]) -> dict[tuple[int | None, int], str]:
    """(season, episode number) -> canonical title from the episode source.

    Single-season results key on (season_number, ep); multi-season results are
    parsed from their compact "N. Title" lines. Episode numbers restart every
    season, so the season is always part of the key.
    """
    data = _tool_json(lore, EPISODES_TOOL)
    out: dict[tuple[int | None, int], str] = {}
    try:
        if data.get("multi_season"):
            for s in data.get("seasons", []):
                for line in s.get("episodes", []):
                    m = re.match(r"\s*(\d+)\.\s*(.*)", str(line))
                    if m:
                        out.setdefault((s.get("season_number"), int(m.group(1))),
                                       m.group(2).removesuffix(" …").strip())
        else:
            for e in data.get("episodes", []):
                if e.get("ep") is not None:
                    out.setdefault((data.get("season_number"), int(e["ep"])), e.get("title") or "")
    except (TypeError, ValueError, AttributeError):
        return {}
    return out


def is_multi_season(lore: dict[str, str]) -> bool:
    """True if the research covered several seasons (a --season of all / 1-7 / 1,3)."""
    return any(_tool_json(lore, t).get("multi_season") for t in (EPISODES_TOOL, ANILIST_TOOL))


def resolved_seasons(lore: dict[str, str], spec: str = "") -> list[dict[str, Any]]:
    """[{"season_number", "season_title"}] the research sources resolved, so the
    timeline states which season(s) its episode numbers belong to. Falls back to a
    number the user typed (--season 2) if no source reported one."""
    for tool in (EPISODES_TOOL, ANILIST_TOOL):
        data = _tool_json(lore, tool)
        if data.get("multi_season") and data.get("seasons"):
            return [{"season_number": s.get("season_number"), "season_title": s.get("season_title") or ""}
                    for s in data["seasons"]]
        if data.get("season_number") or data.get("season_title"):
            return [{"season_number": data.get("season_number"), "season_title": data.get("season_title") or ""}]
    m = re.fullmatch(r"\D*(\d{1,2})\D*", spec or "")
    return [{"season_number": int(m.group(1)), "season_title": ""}] if m else []


def season_label(season: tuple[int | None, str]) -> str:
    """Human-readable season, e.g. 'My Hero Academia 2nd Season (season 2)'."""
    n, title = season
    if n and title:
        return f"{title} (season {n})"
    return f"Season {n}" if n else title


def season_range_label(numbers: list[int]) -> str:
    """'Seasons 1-7' for a contiguous run, else 'Seasons 1, 3, 5'."""
    nums = sorted({n for n in numbers if n})
    if not nums:
        return ""
    if len(nums) > 1 and nums == list(range(nums[0], nums[-1] + 1)):
        return f"Seasons {nums[0]}-{nums[-1]}"
    return ("Seasons " if len(nums) > 1 else "Season ") + ", ".join(map(str, nums))


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def postprocess(bp: TimelineBlueprint, ctx: dict[str, str],
                episodes: dict[tuple[int | None, int], str],
                seasons: list[dict[str, Any]] | None = None, multi: bool = False) -> TimelineBlueprint:
    """Pin header fields, record the season(s), renumber clips, canonicalise episode titles.

    Single season: every clip gets that season. Multi-season: the model's per-clip
    season is checked against the episode list; a wrong or missing one is recovered
    from the episode title (or the only season that has that episode number).
    """
    seasons = seasons or []
    bp.series_title = ctx["series_title"]
    bp.character = ctx["character"]
    bp.arc_theme = ctx["arc_theme"]

    if multi:
        bp.seasons = [SeasonRef(season_number=s["season_number"], season_title=s["season_title"])
                      for s in seasons if s.get("season_number")]
        bp.season_number = None
        bp.season_title = season_range_label([s.season_number for s in bp.seasons])
        by_title: dict[str, list[tuple[int | None, int]]] = {}
        for key, title in episodes.items():
            by_title.setdefault(_norm_title(title), []).append(key)
    else:
        first = seasons[0] if seasons else {"season_number": None, "season_title": ""}
        bp.season_number, bp.season_title = first["season_number"], first["season_title"]

    for i, beat in enumerate(bp.timeline, start=1):
        beat.clip_index = i
        if not multi:
            beat.season_number = bp.season_number
        elif episodes and (beat.season_number, beat.episode_number) not in episodes:
            # Recover: the title usually identifies the right (season, episode).
            cands = by_title.get(_norm_title(beat.episode_title), [])
            if len(cands) > 1:
                cands = [k for k in cands if k[1] == beat.episode_number] or cands
            owners = [sn for (sn, ep) in episodes if ep == beat.episode_number]
            if len(cands) == 1:
                log.warning("Clip %d: season/episode corrected to S%s E%s from its title", i, *cands[0])
                beat.season_number, beat.episode_number = cands[0]
            elif len(owners) == 1:
                log.warning("Clip %d: season set to %s (only season with episode %d)", i, owners[0], beat.episode_number)
                beat.season_number = owners[0]
            else:
                log.warning("Clip %d cites season %s episode %d, which is not in the episode list",
                            i, beat.season_number, beat.episode_number)
        title = episodes.get((beat.season_number, beat.episode_number))
        if title:
            beat.episode_title = title
        elif episodes and not multi:
            log.warning("Clip %d cites episode %d, absent from the episode list", i, beat.episode_number)
    return bp


# ===========================================================================
# Grounding and quality review (anti-hallucination)
# ===========================================================================
# A small local model will invent quotes, cite episodes that do not exist,
# reuse one line in several clips, or write the same sentence for every clip.
# These checks catch that deterministically, against the research dossier the
# model was given and the episode list. The first time problems are found the
# model receives them as a correction request; whatever is still wrong after
# that is fixed in code (repair_timeline).
GROUNDING_THRESHOLD = 0.8      # share of a phrase's words that must occur together in the dossier
NEAR_DUPLICATE = 0.75          # word-set similarity above which two clips' texts count as copies
MAX_CLIP_SECONDS = 90          # longer "clips" are trimmed to TRIM_TO_SECONDS
TRIM_TO_SECONDS = 60
EPISODE_END_SECONDS = 24 * 60
MAX_REPORTED_ISSUES = 12
_STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "into", "their", "his", "her",
              "its", "was", "are", "has", "have", "him", "they", "them", "who", "while", "when"}
_TS_PAIR = re.compile(r"^(\d{1,3}):(\d{2}) - (\d{1,3}):(\d{2})$")


def _words(text: str) -> list[str]:
    """Lower-case word tokens (apostrophes kept, so "won't" stays one word)."""
    return re.sub(r"[^a-z0-9']+", " ", (text or "").lower()).split()


def is_silent(anchor: str) -> bool:
    return not anchor or anchor.strip().startswith("None (Silent")


class Grounder:
    """Answers "does this phrase occur (nearly) word for word in the dossier?".

    Exact match on normalised text, or, for phrases of 4+ words, at least
    GROUNDING_THRESHOLD of the phrase's distinct words inside one window of the
    dossier barely longer than the phrase (tolerates small copying slips, not
    paraphrase).
    """

    def __init__(self, corpus: str) -> None:
        self.tokens = _words(corpus)
        self.joined = " " + " ".join(self.tokens) + " "
        self.vocab = set(self.tokens)

    def __call__(self, phrase: str) -> bool:
        toks = _words(phrase)
        if not toks:
            return False
        if f" {' '.join(toks)} " in self.joined:
            return True
        if len(toks) < 4:                       # short phrases must match exactly
            return False
        need = set(toks)
        if len(need & self.vocab) / len(need) < GROUNDING_THRESHOLD:
            return False                        # cheap reject before the window scan
        window = len(toks) + max(2, len(toks) // 4)
        for i in range(max(1, len(self.tokens) - window + 1)):
            if len(need.intersection(self.tokens[i:i + window])) / len(need) >= GROUNDING_THRESHOLD:
                return True
        return False


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity of content words (stopwords and 1-2 letter words ignored)."""
    sa = {w for w in _words(a) if len(w) > 2 and w not in _STOPWORDS}
    sb = {w for w in _words(b) if len(w) > 2 and w not in _STOPWORDS}
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def _seconds(ts: str) -> tuple[int, int]:
    m = _TS_PAIR.match(ts)
    if not m:                                   # cannot happen after validation; be safe
        return 0, 0
    a, b, c, d = map(int, m.groups())
    return a * 60 + b, c * 60 + d


def _where(beat: "ClipBeat") -> str:
    return f"S{beat.season_number}E{beat.episode_number}" if beat.season_number else f"episode {beat.episode_number}"


def review_timeline(bp: TimelineBlueprint, episodes: dict[tuple[int | None, int], str],
                    ground: Grounder) -> list[str]:
    """Problems to send back to the model, most important first (empty = clean)."""
    issues: list[str] = []
    seen_lines: dict[str, int] = {}
    spans: list[tuple[int, tuple[int | None, int], int, int]] = []
    for b in bp.timeline:
        n, key = b.clip_index, (b.season_number, b.episode_number)
        if episodes and key not in episodes:
            issues.append(f"clip {n}: {_where(b)} is not in the episode list; use a listed episode")
        start, end = _seconds(b.timestamp_estimate)
        if end - start > MAX_CLIP_SECONDS or end > EPISODE_END_SECONDS:
            issues.append(f"clip {n}: {b.timestamp_estimate} is implausible; clips are 10-60 s inside a 24-minute episode")
        for m, other, s2, e2 in spans:
            if other == key and start < e2 and s2 < end:
                issues.append(f"clips {m} and {n} are the same scene ({_where(b)}, overlapping times); "
                              f"choose a different moment for clip {n}")
        spans.append((n, key, start, end))
        if not is_silent(b.dialogue_anchor):
            line = " ".join(_words(b.dialogue_anchor))
            if line in seen_lines:
                issues.append(f"clip {n} repeats the dialogue of clip {seen_lines[line]}; a line may be used once")
            else:
                seen_lines[line] = n
                if not ground(b.dialogue_anchor):
                    issues.append(f'clip {n}: the line "{b.dialogue_anchor[:70]}" is not in the dossier; '
                                  f'copy a line that is, or use "{SILENT_BEAT}"')
    # Grouped issues (one line each, so a templated timeline cannot crowd out the rest):
    def clips(ns: set[int] | list[int]) -> str:
        return ", ".join(map(str, sorted(ns)))

    for field in ("emotional_beat", "community_consensus"):
        copied: set[int] = set()
        beats = bp.timeline
        for i, a in enumerate(beats):
            for b in beats[i + 1:]:
                if _similarity(getattr(a, field), getattr(b, field)) >= NEAR_DUPLICATE:
                    copied |= {a.clip_index, b.clip_index}
        if copied:
            issues.append(f"clips {clips(copied)} have near-identical {field}; write each one about its own scene")
    ungrounded = [b.clip_index for b in bp.timeline if not ground(b.evidence)]
    if ungrounded:
        issues.append(f"clips {clips(ungrounded)}: evidence must be words copied exactly from the dossier "
                      "(an episode synopsis, review or post)")
    unique = list(dict.fromkeys(issues))       # de-duplicate, keep order
    return unique[:MAX_REPORTED_ISSUES]


def timeline_summary(bp: TimelineBlueprint) -> str:
    """One line per clip, so a correction request can say what to keep without
    resending the whole previous answer (the context window is small)."""
    return "\n".join(f"clip {b.clip_index}: {_where(b)} {b.timestamp_estimate} | {b.emotional_beat[:80]}"
                     for b in bp.timeline)


def repair_timeline(bp: TimelineBlueprint, episodes: dict[tuple[int | None, int], str],
                    ground: Grounder) -> TimelineBlueprint:
    """Fix in code whatever the correction round did not:
    drop clips of episodes that do not exist and duplicate scenes, trim overlong
    clips, blank invented or repeated dialogue, and record whether each clip's
    evidence was found (evidence_verified)."""
    invalid = {b.clip_index for b in bp.timeline if episodes and (b.season_number, b.episode_number) not in episodes}
    if invalid and len(invalid) == len(bp.timeline):
        # Every clip "invalid" means the episode list and the model disagree wholesale
        # (e.g. a season-numbering mismatch); dropping everything would help nobody.
        log.warning("No clip matches the episode list; keeping them all, marked unverified")
        invalid = set()
    kept: list[ClipBeat] = []
    seen_lines: set[str] = set()
    spans: list[tuple[tuple[int | None, int], int, int]] = []
    for b in bp.timeline:
        key = (b.season_number, b.episode_number)
        if b.clip_index in invalid:
            log.warning("Dropped clip %d: %s is not in the episode list", b.clip_index, _where(b))
            continue
        start, end = _seconds(b.timestamp_estimate)
        if any(k == key and start < e and s < end for k, s, e in spans):
            log.warning("Dropped clip %d: same scene as an earlier clip (%s)", b.clip_index, _where(b))
            continue
        if end - start > MAX_CLIP_SECONDS:
            end = start + TRIM_TO_SECONDS
            b.timestamp_estimate = f"{start // 60:02d}:{start % 60:02d} - {end // 60:02d}:{end % 60:02d}"
            log.warning("Clip %d trimmed to %d s", b.clip_index, TRIM_TO_SECONDS)
        spans.append((key, start, end))
        if not is_silent(b.dialogue_anchor):
            line = " ".join(_words(b.dialogue_anchor))
            if line in seen_lines or not ground(b.dialogue_anchor):
                log.warning("Clip %d: dialogue removed (%s)", b.clip_index,
                            "repeated" if line in seen_lines else "not found in the research")
                b.dialogue_anchor = SILENT_BEAT
            else:
                seen_lines.add(line)
        b.evidence_verified = ground(b.evidence)
        kept.append(b)
    for i, b in enumerate(kept, start=1):
        b.clip_index = i
    bp.timeline = kept
    unverified = sum(1 for b in kept if not b.evidence_verified)
    if unverified:
        log.warning("%d of %d clips have evidence that is not in the research; check them first", unverified, len(kept))
    return bp


def _compact_errors(exc: ValidationError) -> str:
    """Short error list for the corrective retry (keeps context small)."""
    return "\n".join(f"- {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:12])


async def synthesis_phase(
    llm: AsyncClient,
    model: str,
    ctx: dict[str, str],
    lore: dict[str, str],
    num_ctx: int,
    max_beats: int,
    extra: dict[str, Any],
    beats: int | None = None,
) -> TimelineBlueprint:
    """Single structured call (plus one corrective retry) -> validated blueprint."""
    # Output budget scales with beat count; input budget is what remains.
    num_predict = min(4096, 400 + 260 * max(max_beats, beats or 0))
    input_tokens = num_ctx - num_predict - PROMPT_RESERVE_TOKENS
    budget_chars = max(4000, int(input_tokens * CHARS_PER_TOKEN))
    dossier = build_dossier(lore, budget_chars)
    seasons = resolved_seasons(lore, ctx.get("season", ""))
    multi = is_multi_season(lore)
    label = (season_range_label([s["season_number"] for s in seasons if s["season_number"]]) if multi
             else season_label((seasons[0]["season_number"], seasons[0]["season_title"])) if seasons else "")
    ctx = {**ctx, "season_label": label, "multi_season": "1" if multi else ""}
    log.info("Season%s: %s", "s" if multi else "", label or "not identified")
    log.info("Synthesis: dossier %d chars, num_predict %d", len(dossier), num_predict)

    base_messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": synthesis_prompt(ctx, dossier, max_beats, beats)},
    ]
    options = {"temperature": 0.2, "num_ctx": num_ctx, "num_predict": num_predict}
    schema = llm_schema(multi)

    episodes = known_episodes(lore)
    ground = Grounder(dossier)          # the model may only cite what it was shown

    messages = base_messages
    last_exc: ValidationError | None = None
    for attempt in (1, 2):
        resp = await llm.chat(model=model, messages=messages, format=schema,
                              options=options, **extra)
        raw = resp.message.content or ""
        try:
            bp = TimelineBlueprint.model_validate_json(raw)
        except ValidationError as exc:
            last_exc = exc
            log.warning("Attempt %d failed validation:\n%s", attempt, _compact_errors(exc))
            # Retry from the clean base prompt + error list (not the bad output),
            # so the context does not grow.
            messages = base_messages + [{"role": "user", "content": (
                "Your previous JSON failed validation:\n" + _compact_errors(exc) +
                "\nReturn the full corrected JSON."
            )}]
            continue

        bp = postprocess(bp, ctx, episodes, seasons, multi)
        problems = review_timeline(bp, episodes, ground)
        if beats and len(bp.timeline) != beats:
            problems.insert(0, f"the timeline has {len(bp.timeline)} clips; it must have exactly {beats}")
        if problems and attempt == 1:
            log.warning("Attempt 1 has %d problem(s); asking for a corrected timeline:\n%s",
                        len(problems), "\n".join(f"- {p}" for p in problems))
            # A one-line-per-clip summary instead of the full previous answer keeps the
            # correction request small enough for an 8k context.
            messages = base_messages + [{"role": "user", "content": (
                "Your previous timeline was:\n" + timeline_summary(bp) +
                "\n\nIt has these problems:\n" + "\n".join(f"- {p}" for p in problems) +
                "\n\nReturn the full corrected JSON. Keep the clips that had no problems.")}]
            continue
        if problems:
            log.warning("After the correction round %d problem(s) remain; fixing what can be fixed in code",
                        len(problems))
        if beats and len(bp.timeline) != beats:
            log.warning("Model returned %d beats instead of %d; keeping it", len(bp.timeline), beats)
        return repair_timeline(bp, episodes, ground)
    raise RuntimeError(f"Structured output failed validation twice:\n{last_exc}")


# ===========================================================================
# Wiring
# ===========================================================================
async def preflight(llm: AsyncClient, model: str) -> None:
    """Fail fast if Ollama is down or the model is not pulled."""
    try:
        await llm.show(model)
    except ResponseError as exc:
        sys.exit(f"Model {model!r} unavailable ({exc}). Run: ollama pull {model}")
    except (httpx.ConnectError, ConnectionError) as exc:
        sys.exit(f"Cannot reach Ollama ({exc}). Is `ollama serve` running?")


async def run(args: argparse.Namespace) -> None:
    ctx = {"series_title": args.series, "character": args.character, "arc_theme": args.arc,
           "season": (getattr(args, "season", "") or "").strip()}
    server_path = Path(args.server).resolve()
    if not server_path.is_file():
        sys.exit(f"MCP server not found: {server_path}")

    if args.cut:  # fail before the slow LLM work, not after it
        if not shutil.which("node"):
            sys.exit("--cut needs Node.js on PATH (node --version).")
        if not Path(args.clip_script).is_file():
            sys.exit(f"--cut: clipfinal.mjs not found at {Path(args.clip_script).resolve()} (see --clip-script)")

    llm = AsyncClient(host=args.host) if args.host else AsyncClient()
    await preflight(llm, args.model)

    # Per-call kwargs shared by both phases.
    extra: dict[str, Any] = {"keep_alive": args.keep_alive}
    if args.model.lower().startswith(THINKING_PREFIXES):
        extra["think"] = False  # reasoning traces would consume the small context
    research_options = {"temperature": 0.3, "num_ctx": args.num_ctx}

    # Server inherits the environment (proxies, PATH) and runs under this interpreter.
    params = StdioServerParameters(command=sys.executable, args=[str(server_path)],
                                   env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            lore = await research_phase(llm, session, args.model, ctx, research_options, extra)

    # Never synthesise from nothing: a model with no dossier will invent a timeline.
    failed = [n for n, t in lore.items() if source_failed(t)]
    if len(failed) == len(lore):
        sys.exit("All research sources failed (see '<-' log lines above); aborting "
                 "instead of fabricating a timeline. Likely a network/proxy/SSL problem.")
    if failed:
        log.warning("Sources failed and are excluded from the dossier: %s", ", ".join(failed))
        lore = {n: t for n, t in lore.items() if n not in failed}

    blueprint = await synthesis_phase(llm, args.model, ctx, lore,
                                      args.num_ctx, args.max_beats, extra, beats=args.beats)

    out = Path(args.output)
    out.write_text(blueprint.model_dump_json(indent=2), encoding="utf-8")
    log.info("Wrote %s (%d beats)", out.resolve(), len(blueprint.timeline))

    if args.cut:
        code = await cut_clips(out, args)
        if code == 2:
            log.warning("Some clips could not be cut; the DaVinci timeline holds the rest (see edit_report.json)")
        elif code != 0:
            sys.exit(code)


async def cut_clips(timeline: Path, args: argparse.Namespace) -> int:
    """Run clipfinal.mjs on the blueprint: download each clip, write a DaVinci .fcpxml.

    Inherits stdio so its progress shows live. Exit code: 0 all clips cut, 2 some clips
    could not be cut (the timeline still contains the rest), anything else = failure.
    """
    cmd = [shutil.which("node"), str(Path(args.clip_script).resolve()),
           "--timeline", str(timeline.resolve()), "--pad", str(args.pad)]
    if args.cut_dir:
        cmd += ["--out-dir", str(Path(args.cut_dir).resolve())]
    if args.dub:
        cmd.append("--dub")
    if args.accurate:
        cmd.append("--accurate")
    log.info("Cutting clips: %s", " ".join(cmd[1:]))
    proc = await asyncio.create_subprocess_exec(*cmd)
    return await proc.wait()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anime arc lore -> JSON timeline (local Ollama, 6–8 GB VRAM).")
    p.add_argument("--series", required=True, help="Anime title")
    p.add_argument("--character", required=True, help="Character name")
    p.add_argument("--arc", required=True, help="Arc / theme description")
    p.add_argument("--season", default="",
                   help='Season number or name, e.g. 2, "Season 2", "Final Season" '
                        'or several seasons: all, a range 1-7, a list 1,3,5 (default: best title match, normally the first season)')
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model tag (default {DEFAULT_MODEL}; alt: qwen3:8b, llama3.1:8b, qwen2.5:3b)")
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                   help="Context window; raise to 12288 on 8 GB cards for richer dossiers")
    p.add_argument("--max-beats", type=int, default=DEFAULT_MAX_BEATS, help="Upper bound on clips")
    p.add_argument("--beats", type=int, default=None,
                   help="Request EXACTLY this many clips (max ~14 at this output budget); "
                        "default: the model picks between 4 and --max-beats")
    p.add_argument("--output", default="timeline_blueprint.json", help="Output path")
    p.add_argument("--server", default=str(Path(__file__).with_name("lore_mcp_server.py")),
                   help="Path to lore_mcp_server.py")
    p.add_argument("--host", default=None, help="Ollama host (default: OLLAMA_HOST or localhost:11434)")
    p.add_argument("--keep-alive", default="5m", help="How long Ollama keeps the model loaded")
    g = p.add_argument_group("clip cutting + DaVinci Resolve export (runs clipfinal.mjs on the JSON)")
    g.add_argument("--cut", action="store_true",
                   help="After writing the JSON, download every clip and build a DaVinci Resolve .fcpxml timeline")
    g.add_argument("--cut-dir", default=None, help="Folder for clips + .fcpxml (default: <series>_<character>_edit)")
    g.add_argument("--clip-script", default=str(Path(__file__).with_name("clipfinal.mjs")), help="Path to clipfinal.mjs")
    g.add_argument("--pad", type=float, default=4, help="Seconds of handles added before/after each clip (timestamps are estimates)")
    g.add_argument("--dub", action="store_true", help="Cut from the dub instead of the sub")
    g.add_argument("--accurate", action="store_true", help="Frame-accurate cuts (re-encode; slower)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
