# Anime Arc Timeline Generator

Turn a character arc into an edit plan. Give it a series, a character and an arc ("Bakugo, redemption with Deku, all seasons") and a **local LLM** researches the arc from public sources, then writes a clip-by-clip timeline as JSON: which season and episode, a time range, the emotional beat, direction notes, and the source text each clip is based on.

Everything runs on your machine with [Ollama](https://ollama.com). No paid APIs, no API keys. It is tuned for a GPU with **6-8 GB of VRAM**.

> **Bring your own footage.** This project plans and exports edits; it does not download or stream video. Use episode files you own or are licensed to use.

## How it works

```
 series + character + arc (+ season)
              |
              v
   director_orchestrator.py ----- Ollama (local LLM, tool calling)
              |  MCP over stdio
              v
     lore_mcp_server.py
       |- episode lists + synopses   Jikan -> Kitsu -> TVMaze (first that answers)
       |- character bio + reviews    AniList
       |- fan discussion             Reddit (JSON, RSS fallback)
       `- plot background            Wikipedia (optional)
              |
              v
   structured JSON synthesis + grounding checks
              |
              v
      timeline_blueprint.json
              |
              v
   resolve-export.mjs  ->  DaVinci Resolve timeline (.fcpxml)
```

1. The **MCP server** (`lore_mcp_server.py`) exposes four research tools over stdio.
2. The **orchestrator** (`director_orchestrator.py`) lets the model call those tools, then makes one schema-constrained call (`format=<JSON schema>`, temperature 0.2) to produce the timeline.
3. The result is **checked against the research** (see [Hallucination controls](#hallucination-controls)), corrected once by the model if needed, and repaired in code if not.
4. `resolve-export.mjs` turns clip files cut from your own footage into a Resolve-importable timeline.

## Requirements

- **Python 3.11+**
- **Ollama** with a tool-calling model (default `qwen2.5:7b`, about 4.7 GB)
- **Node.js 18+** and **FFmpeg/ffprobe**, only for the Resolve export
- Internet access to the research sources (Kitsu, AniList, Reddit, ...)

## Install

```bash
pip install "mcp[cli]>=1.2,<2" "ollama>=0.5" httpx "pydantic>=2.6"
ollama pull qwen2.5:7b
```

Keep `mcp` below version 2: the server uses the 1.x `FastMCP` API.

Windows (cmd), if you need the tools themselves:

```bat
winget install Python.Python.3.12
winget install Ollama.Ollama
winget install Gyan.FFmpeg
```

## Quick start

```bash
# Windows cmd: set LORE_CONTACT=you@example.com     macOS/Linux: export LORE_CONTACT=you@example.com
python director_orchestrator.py \
  --series "My Hero Academia" \
  --character "Katsuki Bakugo" \
  --arc "Redemption with Deku" \
  --season all --beats 12 --num-ctx 12288
```

This writes `timeline_blueprint.json` in the current folder. A run takes a few minutes, mostly the model.

`LORE_CONTACT` is placed in the HTTP User-Agent. Wikipedia and some other sites reject anonymous clients, so set it to your email address or project URL.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--series` | required | Anime title |
| `--character` | required | Character name |
| `--arc` | required | Arc or theme, in your own words |
| `--season` | blank | Which season(s): see [Seasons](#seasons) |
| `--model` | `qwen2.5:7b` | Any Ollama model with tool calling (`qwen3:8b`, `llama3.1:8b`, `qwen2.5:3b`) |
| `--num-ctx` | `8192` | Context window. The main VRAM knob: use `12288` on 8 GB cards |
| `--beats` | model decides | Ask for exactly this many clips (about 14 at most) |
| `--max-beats` | `8` | Upper bound on clips when `--beats` is not set |
| `--output` | `timeline_blueprint.json` | Output path |
| `--host` | Ollama default | Ollama server address |
| `--keep-alive` | `5m` | How long Ollama keeps the model loaded |
| `--server` | next to the script | Path to `lore_mcp_server.py` |

### Choosing a model for your GPU

| Model | Approx. VRAM at 8k context | Notes |
|---|---|---|
| `qwen2.5:7b` | about 5.2 GB | Default; fits 6 GB cards |
| `qwen3:8b` | about 5.8 GB | Fits 8 GB |
| `llama3.1:8b` | about 5.9 GB | Fits 8 GB; spills to CPU on 6 GB |
| `qwen2.5:3b` | about 2.2 GB | Fast, noticeably weaker output |

## Seasons

Anime catalogues list every season as its own entry, so the tool resolves "season N" to the Nth TV entry of the title, ordered by air date.

| `--season` | Result |
|---|---|
| *(blank)* | Best title match, normally season 1. The result says which season it used |
| `2`, `Season 2`, `2nd season` | One season |
| `final` or text from a season's title | One season, matched by title |
| `1-7`, `1,3,5`, `all` | Several seasons: every clip carries its own `season_number` |

For several seasons use `--num-ctx 12288` and 10-14 clips. At 8k context the episode lists for a long series do not fit, and later seasons drop out of the model's view; a smaller range such as `1-4` is the alternative on a 6 GB card.

## Output

`timeline_blueprint.json` (abbreviated):

```json
{
  "series_title": "My Hero Academia",
  "character": "Katsuki Bakugo",
  "arc_theme": "Redemption with Deku",
  "narrative_thesis": "...",
  "season_number": null,
  "season_title": "Seasons 1-4",
  "seasons": [{ "season_number": 1, "season_title": "..." }],
  "timeline": [
    {
      "clip_index": 1,
      "season_number": 3,
      "episode_number": 23,
      "episode_title": "...",
      "evidence": "words copied from the research that justify this clip",
      "timestamp_estimate": "10:00 - 10:40",
      "emotional_beat": "...",
      "dialogue_anchor": "None (Silent Visual Beat)",
      "community_consensus": "...",
      "director_notes": "framing, pacing, J-cut / L-cut instruction",
      "evidence_verified": true
    }
  ]
}
```

- `episode_number` counts **within its season**; every season restarts at 1.
- `dialogue_anchor` is a line found in the research, or `"None (Silent Visual Beat)"`. An empty field is preferred to an invented quote.
- `timestamp_estimate` is `MM:SS - MM:SS` and is **an estimate** (see [Limitations](#limitations)).
- `evidence_verified: false` means the clip's source text was not found in the research. Check those clips first.

## Hallucination controls

A small local model will invent quotes, cite episodes that do not exist, reuse one line in several clips, or write the same sentence for every clip. The orchestrator guards against this:

- **Evidence per clip.** Each clip must carry 8-25 words copied from the research; the code verifies they occur there.
- **Key episodes first.** The research tool ranks episodes by how often their title and synopsis mention the character and the arc, and hands the best ones to the model with their synopses.
- **Checks on the result.** Dialogue not found in the research, a line used twice, two clips of the same scene, episodes that are not in the episode list, near-identical wording across clips, and implausible time ranges are all detected.
- **One correction round.** The model gets the list of problems and one chance to fix them.
- **Repair in code.** Whatever is still wrong is fixed deterministically: invented or repeated dialogue becomes a silent beat, duplicate scenes and non-existent episodes are dropped, overlong clips are trimmed to 60 s.

## Research sources

| Tool | Source | Provides | If it fails |
|---|---|---|---|
| `fetch_anime_episodes_and_reviews` | Jikan, then Kitsu, then TVMaze | Episode lists, synopses, reviews (Jikan only), key episodes | Next source in the chain |
| `fetch_anilist_character_and_reviews` | AniList GraphQL | Character bio, arc-relevant review excerpts | Run continues without it |
| `scrape_reddit_character_lore` | Reddit JSON, then RSS | Fan analysis and discussion threads | Run continues without it |
| `fetch_wiki_character_summary` | Wikipedia (MediaWiki API) | Plot and character background | Run continues without it |

A failed source is logged and left out of the research; the run only stops if **all** sources fail, so a timeline is never written from an empty dossier. Every source is queried unauthenticated with a descriptive User-Agent.

## Exporting to DaVinci Resolve

`resolve-export.mjs` builds an FCPXML 1.9 timeline from clip files you have already cut from your own footage. Clips are placed back to back on V1/A1 in order, and each can carry markers (for example the emotional beat and the director notes from the JSON).

```js
import { writeResolveProject } from './resolve-export.mjs';

await writeResolveProject('timeline.fcpxml', {
  projectName: 'Bakugo redemption',
  clips: [
    {
      file: 'clips/01_S03E23.mp4',
      name: '01 S03E23',
      markers: [{ atSec: 0, value: 'Bakugo admits his envy', note: 'slow push-in, L-cut into silence' }],
    },
    // ... in timeline order
  ],
});
```

In Resolve: **File > Import > Timeline...** and pick the `.fcpxml`. It needs `ffprobe` on `PATH` to read each clip's real frame rate, size and duration.

To cut the clips, use the `season_number`, `episode_number` and `timestamp_estimate` from the JSON with any tool (FFmpeg, your editor), and add handles of a few seconds because the timestamps are estimates:

```bash
ffmpeg -ss 00:09:56 -i "your-episode-file.mkv" -t 48 -c:v libx264 -crf 18 -c:a aac clip_01.mp4
```

## Project layout

```
director_orchestrator.py   CLI: runs the research loop and writes the timeline JSON
lore_mcp_server.py         MCP server (stdio) with the four research tools
resolve-export.mjs         FCPXML 1.9 writer for DaVinci Resolve
```

## Limitations

- **Timestamps are estimates.** The research sources contain no timing data, so the model places each scene from synopses. Verify them against the episode, or match dialogue against the episode's own subtitle file for exact times.
- **Dialogue is mostly silent beats**, by design: real quotes rarely appear in the research, and invented ones are removed.
- **Season order is inferred from air dates** among catalogue entries with matching titles. Series that split a season into parts, or list specials as TV, can shift the numbering. Check the season titles in the output.
- **A bare title can match the wrong show.** Use the title as the catalogues spell it.
- **Quality depends on the local model.** A 7B model follows the schema well but can still write generic text; the checks above catch invention, not blandness.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `model '...' not found` | `ollama pull qwen2.5:7b`, or pass `--model` with a model you have (`ollama list`) |
| `Connection closed` right after start, or `No module named 'mcp.server.fastmcp'` | `mcp` 2.x is installed: `pip install "mcp[cli]>=1.2,<2"` |
| Wikipedia returns 403 | Set `LORE_CONTACT`. If it still fails, Wikipedia is optional: the run continues without it |
| Reddit returns 403 | Reddit blocks many unauthenticated clients. The run continues on the other sources |
| `ConnectTimeout` for Jikan | Your network blocks it; the episode tool falls back to Kitsu, then TVMaze automatically |
| `All research sources failed` | Network, VPN, proxy or firewall problem; the run stops instead of inventing a timeline |
| Later seasons missing from the result | Context too small for the episode lists: use `--num-ctx 12288` or a smaller `--season` range |
| Fewer clips than asked | Use `--beats N` for an exact count; `--max-beats` is only an upper bound |
