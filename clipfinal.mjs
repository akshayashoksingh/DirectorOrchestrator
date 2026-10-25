#!/usr/bin/env node
// clip.mjs — search an anime by name, pick an episode, and cut a clip from it.
//
//   node clip.mjs                          fully interactive
//   node clip.mjs "one piece"              start from a name
//   node clip.mjs --anime "one piece" --episode 1 --start 1:30 --duration 45 [--dub] [--accurate]
//                 [--slow-seek] [--pick 1] [--out clip.mp4]
//   node clip.mjs --timeline timeline_blueprint.json [--out-dir DIR] [--pad 4] [--dub] [--accurate]
//                 [--pick 1] [--dry-run]      cut every clip of a director_orchestrator.py timeline
//                                             and write a DaVinci Resolve timeline (.fcpxml)
//
// Needs: anime-clip.mjs in the same folder, playwright + ffmpeg installed, and your hianime-api
// running (default http://localhost:5000, start it with `bun run dev`). Override with API=http://host:port.
// Env: SHOW=1 (visible browser)  WAIT=30 (seconds to wait for the playlist per server)

import readline from 'node:readline/promises';
import { stdin, stdout } from 'node:process';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs/promises';
import path from 'node:path';
import { capturePlaylist, cutClip, toSeconds, normalizeOutFile } from './anime-clip.mjs';
import * as animeClip from './anime-clip.mjs';
import { probeMedia, writeResolveProject } from './resolve-export.mjs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const API = process.env.API || 'http://localhost:5000';
const TIME_HINT = 'Use seconds (90), MM:SS (1:30) or HH:MM:SS (00:01:30).';

const API_HELP = `
Could not talk to the hianime-api at ${API}.
  - Is it running?  cd hianime-api && bun run dev   (it listens on port 5000)
  - If curl says "empty reply from server": in that repo's index.ts add  idleTimeout: 60,
    inside Bun.serve({ ... }) and restart it (the default 10s timeout is too short for the site).
  - Different port or host? run with  API=http://localhost:PORT`;

// ------------------------------------------------------------------ API helpers

class ApiError extends Error {}

async function getJson(path, { timeoutMs = 45000, retries = 1 } = {}) {
  let last;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`${API}${path}`, { signal: controller.signal });
      const text = await res.text();
      let body;
      try {
        body = JSON.parse(text);
      } catch {
        throw new ApiError(`unexpected reply from the API (HTTP ${res.status})`);
      }
      if (!res.ok || body.success === false) throw new ApiError(body.message || `HTTP ${res.status}`);
      return body;
    } catch (err) {
      if (err instanceof ApiError) throw err; // a real answer from the API: don't retry
      last = err; // network error or timeout: retry once
    } finally {
      clearTimeout(timer);
    }
  }
  const reason = last?.cause?.code || last?.cause?.errors?.[0]?.code || last?.message || 'request failed';
  const error = new ApiError(`could not connect (${reason})`);
  error.network = true;
  throw error;
}

export async function searchAnime(query) {
  const body = await getJson(`/api/v2/search?keyword=${encodeURIComponent(query)}`);
  return (body.data?.response ?? []).filter(a => a.id);
}

export async function getEpisodes(animeId) {
  const body = await getJson(`/api/v2/episodes/${encodeURIComponent(animeId)}`);
  const list = Array.isArray(body.data) ? body.data : [];
  return list.map((e, i) => ({ ...e, number: Number(e.episodeNumber ?? i + 1) }));
}

/** Player URLs for an episode, best first (hd-1 is the one that worked in testing). */
export async function getEmbedUrls(episodeId, type) {
  const body = await getJson(`/api/v2/servers/${encodeURIComponent(episodeId)}?type=${type}`);
  const urls = (body.data?.servers ?? []).map(s => s.iframe).filter(Boolean);
  urls.sort((a, b) => Number(b.includes('/embed/hd-1/')) - Number(a.includes('/embed/hd-1/')));
  return urls;
}

// ------------------------------------------------------------------ prompts

const describe = (a, i) => {
  const eps = a.episodes?.eps ?? a.episodes?.sub;
  const bits = [a.type, eps ? `${eps} eps` : null].filter(Boolean).join(', ');
  return `  ${i + 1}) ${a.title}${bits ? ` (${bits})` : ''}`;
};

const slug = s =>
  String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'clip';

async function pickAnime(ask, opts) {
  let query = opts.anime;
  for (;;) {
    if (!query) query = await ask('Anime name: ', v => v.length > 0, 'Type a name to search for.');
    console.log(`Searching for "${query}"...`);
    const results = await searchAnime(query);

    if (!results.length) {
      if (opts.anime) throw new Error(`No results for "${query}".`);
      console.log(`No results for "${query}". Try another name.\n`);
      query = null;
      continue;
    }

    const shown = results.slice(0, 10);
    console.log(`\n${shown.map(describe).join('\n')}\n`);

    if (opts.pick) {
      const chosen = shown[Number(opts.pick) - 1];
      if (!chosen) throw new Error(`--pick ${opts.pick} is out of range (1-${shown.length}).`);
      return chosen;
    }
    if (opts.anime) return shown[0]; // name given on the command line: take the top hit

    const answer = await ask(
      'Pick a number (Enter = 1), or type a new name to search again: ',
      () => true
    );
    if (answer === '') return shown[0];
    const n = Number(answer);
    if (Number.isInteger(n) && n >= 1 && n <= shown.length) return shown[n - 1];
    query = answer; // not a number: treat as a new search
  }
}

/** Seasons are separate entries on this site. Offer the others if the detail page lists any. */
async function pickSeason(ask, anime, opts) {
  if (opts.start && opts.duration) return anime; // non-interactive run
  try {
    const body = await getJson(`/api/v2/anime/${encodeURIComponent(anime.id)}`, { retries: 0 });
    const seasons = (body.data?.moreSeasons ?? []).filter(s => s.id);
    if (seasons.length < 2) return anime;

    console.log('This title has several seasons:');
    seasons.forEach((s, i) => {
      console.log(`  ${i + 1}) ${s.title || s.alternativeTitle || s.id}${s.isActive ? '   <- current' : ''}`);
    });
    const answer = await ask(
      'Season number (Enter = keep current): ',
      v => v === '' || (Number.isInteger(Number(v)) && Number(v) >= 1 && Number(v) <= seasons.length),
      `Enter a number from 1 to ${seasons.length}, or press Enter.`
    );
    if (answer === '') return anime;
    const s = seasons[Number(answer) - 1];
    return { ...anime, id: s.id, title: s.title || s.alternativeTitle || anime.title };
  } catch {
    return anime; // season info is a nicety; never block on it
  }
}

async function pickEpisode(ask, anime, opts) {
  console.log(`Loading episodes for "${anime.title}"...`);
  const eps = await getEpisodes(anime.id);
  if (!eps.length) throw new Error(`No episodes found for "${anime.title}".`);

  const numbers = eps.map(e => e.number);
  const find = v => eps.find(e => e.number === Number(v));
  const range = `${Math.min(...numbers)}-${Math.max(...numbers)}`;
  const answer =
    opts.episode ??
    (await ask(`Episode number (${range}): `, v => v !== '' && Boolean(find(v)), `No such episode. Available: ${range}.`));
  const ep = find(answer);
  if (!ep) throw new Error(`Episode ${answer} is not available (${range}).`);
  return ep;
}

/** Try each player until one yields a playlist (short-lived token: cut right after calling this). */
async function captureFromAny(urls) {
  for (const [i, url] of urls.entries()) {
    console.log(`Opening player ${i + 1}/${urls.length}...`);
    try {
      return await capturePlaylist(url, {
        show: process.env.SHOW === '1',
        waitMs: Number(process.env.WAIT || 30) * 1000,
      });
    } catch (err) {
      console.warn(`  no playlist from this server (${err.message.split('.')[0]}).`);
    }
  }
  return null;
}

// ------------------------------------------------------------------ timeline mode
// director_orchestrator.py writes a JSON blueprint; this section turns it into real clips on disk
// plus a DaVinci Resolve timeline. Per clip it uses: season_number, episode_number, episode_title,
// timestamp_estimate ("MM:SS - MM:SS"), and the creative fields (as timeline markers).

const norm = s => String(s ?? '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
const wordSet = s => new Set(norm(s).split(' ').filter(w => w.length > 1));

/** |A∩B| / |A∪B| — strict; used to tell seasons apart ("Foo" vs "Foo 2nd Season"). */
function jaccard(a, b) {
  const A = wordSet(a), B = wordSet(b);
  if (!A.size || !B.size) return 0;
  let both = 0;
  for (const w of A) if (B.has(w)) both++;
  return both / (A.size + B.size - both);
}

/** |A∩B| / min(|A|,|B|) — lenient; used for episode titles (sites add or drop subtitles). */
function overlap(a, b) {
  const A = wordSet(a), B = wordSet(b);
  if (!A.size || !B.size) return 0;
  let both = 0;
  for (const w of A) if (B.has(w)) both++;
  return both / Math.min(A.size, B.size);
}

/** "My Hero Academia 2nd Season" / "... Season 3" -> 2 / 3; no marker -> null. */
function seasonOrdinal(title) {
  const t = String(title ?? '');
  const m = t.match(/\b(\d{1,2})(?:st|nd|rd|th)\s+season\b/i) || t.match(/\bseason\s+(\d{1,2})\b/i);
  return m ? Number(m[1]) : null;
}

const RANGE_RE = /^\s*(\d{1,3}):(\d{2})\s*-\s*(\d{1,3}):(\d{2})\s*$/;

function parseRange(text) {
  const m = RANGE_RE.exec(String(text ?? ''));
  if (!m) return null;
  const start = Number(m[1]) * 60 + Number(m[2]);
  const end = Number(m[3]) * 60 + Number(m[4]);
  return end > start ? { start, end } : null;
}

const mmss = sec => `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(Math.round(sec % 60)).padStart(2, '0')}`;
const tag = c => `S${String(c.season).padStart(2, '0')}E${String(c.episode).padStart(2, '0')}`;
const clip2 = n => String(n).padStart(2, '0');

async function loadTimeline(file) {
  let bp;
  try {
    bp = JSON.parse(await fs.readFile(file, 'utf8'));
  } catch (err) {
    throw new Error(`Cannot read timeline JSON "${file}": ${err.message}`);
  }
  if (!Array.isArray(bp.timeline) || !bp.timeline.length) throw new Error(`"${file}" has no timeline[] entries.`);
  if (!bp.series_title) throw new Error(`"${file}" has no series_title.`);

  const seasonTitles = new Map((bp.seasons ?? []).map(s => [Number(s.season_number), s.season_title || '']));
  if (bp.season_number) seasonTitles.set(Number(bp.season_number), bp.season_title || '');

  const clips = bp.timeline.map((b, i) => {
    const index = Number(b.clip_index) || i + 1;
    const range = parseRange(b.timestamp_estimate);
    if (!range) throw new Error(`clip ${index}: timestamp_estimate "${b.timestamp_estimate}" is not "MM:SS - MM:SS".`);
    const episode = Number(b.episode_number);
    if (!Number.isInteger(episode) || episode < 1) throw new Error(`clip ${index}: bad episode_number "${b.episode_number}".`);
    let season = Number(b.season_number ?? bp.season_number);
    if (!Number.isInteger(season) || season < 1) {
      console.warn(`clip ${index}: JSON names no season; assuming season 1.`);
      season = 1;
    }
    return {
      index, season, episode, range,
      seasonTitle: seasonTitles.get(season) || '',
      episodeTitle: b.episode_title || '',
      beat: b,
    };
  });
  clips.sort((a, b) => a.index - b.index);
  return { bp, clips };
}

async function pickBaseAnime(bp, opts) {
  console.log(`Searching for "${bp.series_title}"...`);
  const shown = (await searchAnime(bp.series_title)).slice(0, 10);
  if (!shown.length) throw new Error(`No results for "${bp.series_title}".`);
  let chosen;
  if (opts.pick) {
    chosen = shown[Number(opts.pick) - 1];
    if (!chosen) throw new Error(`--pick ${opts.pick} is out of range (1-${shown.length}).`);
  } else {
    chosen = shown.find(a => norm(a.title) === norm(bp.series_title)) ?? shown[0];
  }
  console.log(`Using: ${chosen.title}${opts.pick ? '' : '  (override with --pick N if this is wrong)'}`);
  return chosen;
}

/** Every season entry of a title (hianime lists each season as its own anime). */
async function loadSeasonEntries(base) {
  try {
    const body = await getJson(`/api/v2/anime/${encodeURIComponent(base.id)}`, { retries: 0 });
    const list = (body.data?.moreSeasons ?? []).filter(s => s.id);
    if (list.length) return list.map(s => ({ id: s.id, title: s.title || s.alternativeTitle || s.id }));
  } catch {
    /* fall through: treat the title as a single entry */
  }
  return [{ id: base.id, title: base.title }];
}

/** Find the site entry for "season N". Returns { entry, how } or null. */
async function findSeasonEntry(entries, n, seasonTitle, bp) {
  const byOrdinal = entries.find(e => seasonOrdinal(e.title) === n);
  if (byOrdinal) return { entry: byOrdinal, how: `title says season ${n}` };

  if (seasonTitle) {
    const scored = entries.map(e => ({ e, score: jaccard(e.title, seasonTitle) })).sort((a, b) => b.score - a.score);
    if (scored[0]?.score >= 0.8) return { entry: scored[0].e, how: `title matches "${seasonTitle}"` };
  }
  if (n === 1) {
    const plain = entries.find(e => seasonOrdinal(e.title) === null);
    if (plain) return { entry: plain, how: 'first season (no season number in title)' };
  }
  if (entries.length > 1 && entries[n - 1]) return { entry: entries[n - 1], how: `position ${n} in the site's season list` };

  if (n > 1) {
    // The site may not link its seasons together: search for "<series> season N" directly.
    for (const q of [`${bp.series_title} season ${n}`, seasonTitle].filter(Boolean)) {
      const hits = (await searchAnime(q)).slice(0, 10);
      const hit = hits.find(h => seasonOrdinal(h.title) === n) ?? hits.find(h => seasonTitle && jaccard(h.title, seasonTitle) >= 0.8);
      if (hit) return { entry: { id: hit.id, title: hit.title }, how: `search "${q}"` };
    }
  }
  return null;
}

/** Match the JSON's episode (number + title) to the site's episode list. */
function findEpisode(eps, clip) {
  const nameOf = e => [e.title, e.alternativeTitle, e.name].filter(Boolean);
  const sim = e => Math.max(0, ...nameOf(e).map(t => overlap(t, clip.episodeTitle)));
  const byNumber = eps.find(e => e.number === clip.episode);

  if (byNumber && (!clip.episodeTitle || !nameOf(byNumber).length || sim(byNumber) >= 0.5)) {
    return { ep: byNumber, note: '' };
  }
  const byTitle = clip.episodeTitle
    ? eps.map(e => ({ e, score: sim(e) })).sort((a, b) => b.score - a.score)[0]
    : null;
  if (byTitle && byTitle.score >= 0.8) {
    return { ep: byTitle.e, note: `found by title: the JSON says episode ${clip.episode}, the site lists it as ${byTitle.e.number}` };
  }
  if (byNumber) {
    return { ep: byNumber, note: `title differs (site: "${nameOf(byNumber)[0]}", JSON: "${clip.episodeTitle}"); kept the episode number` };
  }
  return null;
}

async function resolveClips(clips, bp, opts) {
  const base = await pickBaseAnime(bp, opts);
  const entries = await loadSeasonEntries(base);
  const seasonCache = new Map();   // season -> { entry, how } | null
  const episodeCache = new Map();  // entry id -> episodes

  for (const clip of clips) {
    if (!seasonCache.has(clip.season)) {
      const found = await findSeasonEntry(entries, clip.season, clip.seasonTitle, bp);
      seasonCache.set(clip.season, found);
      console.log(found
        ? `Season ${clip.season}: "${found.entry.title}"  [${found.how}]`
        : `Season ${clip.season}: no matching entry on the site.`);
    }
    const season = seasonCache.get(clip.season);
    if (!season) {
      clip.error = `season ${clip.season} not found on the site`;
      continue;
    }
    if (!episodeCache.has(season.entry.id)) {
      console.log(`Loading episodes for "${season.entry.title}"...`);
      episodeCache.set(season.entry.id, await getEpisodes(season.entry.id));
    }
    const hit = findEpisode(episodeCache.get(season.entry.id), clip);
    if (!hit) {
      clip.error = `episode ${clip.episode} ("${clip.episodeTitle}") not found in "${season.entry.title}"`;
      continue;
    }
    clip.ep = hit.ep;
    clip.siteSeason = season.entry.title;
    if (hit.ep.number !== clip.episode) {
      clip.jsonEpisode = clip.episode; // keep what the JSON said; labels and filenames use the real episode
      clip.episode = hit.ep.number;
    }
    if (hit.note) {
      clip.note = hit.note;
      console.warn(`  clip ${clip.index} ${tag(clip)}: ${hit.note}`);
    }
  }
}

function printPlan(clips) {
  console.log('\nPlan:');
  for (const c of clips) {
    const r = `${mmss(c.range.start)}-${mmss(c.range.end)}`;
    console.log(c.ep
      ? `  ${clip2(c.index)}  ${tag(c)}  ${r}  "${c.ep.title ?? c.episodeTitle}"`
      : `  ${clip2(c.index)}  ${tag(c)}  ${r}  !! ${c.error}`);
  }
  console.log('');
}

/**
 * Length of the captured stream in seconds (null if it cannot be read).
 * Needed because ffmpeg, asked to seek past the end of an HLS stream, does NOT fail: it prints
 * "could not seek" and cuts from the very start of the episode instead.
 */
async function streamDuration(found) {
  const headers = `Referer: ${found.referer}\r\nUser-Agent: ${found.userAgent}\r\n`;
  const extra = typeof animeClip.hlsExtraOptions === 'function' ? animeClip.hlsExtraOptions() : [];
  try {
    const { stdout } = await promisify(execFile)(
      'ffprobe',
      ['-v', 'error', '-headers', headers, '-allowed_extensions', 'ALL', ...extra,
       '-show_entries', 'format=duration', '-of', 'csv=p=0', found.playlistUrl],
      { timeout: 45000 }
    );
    const d = Number(stdout.trim());
    return Number.isFinite(d) && d > 0 ? d : null;
  } catch {
    return null;
  }
}

function markerFor(clip, padBefore) {
  const b = clip.beat;
  const silent = !b.dialogue_anchor || String(b.dialogue_anchor).startsWith('None (Silent');
  const parts = [
    silent ? 'Dialogue: (silent beat)' : `Dialogue: "${b.dialogue_anchor}"`,
    `Director: ${b.director_notes}`,
    b.community_consensus ? `Why it matters: ${b.community_consensus}` : '',
    `Estimated ${b.timestamp_estimate} in ${tag(clip)}${b.evidence_verified === false ? ' (evidence unverified: check this clip first)' : ''}`,
  ].filter(Boolean);
  return { atSec: padBefore, value: `${clip.index}. ${b.emotional_beat}`, note: parts.join(' | ') };
}

async function runTimeline(opts) {
  const { bp, clips } = await loadTimeline(opts.timeline);
  const pad = opts.pad === undefined ? 4 : Math.round(Number(opts.pad));
  if (!Number.isFinite(pad) || pad < 0) throw new Error('--pad must be a number of seconds, 0 or more.');
  const accurate = Boolean(opts.accurate);
  const outDir = path.resolve(opts.outDir || `${slug(bp.series_title)}_${slug(bp.character || 'edit')}_edit`);
  const clipDir = path.join(outDir, 'clips');

  console.log(`Timeline: ${bp.series_title} / ${bp.character ?? '?'} / ${bp.arc_theme ?? '?'}  (${clips.length} clips)`);
  await resolveClips(clips, bp, opts);
  printPlan(clips);
  if (opts.dryRun) return console.log('Dry run: nothing downloaded.');

  const resolved = clips.filter(c => c.ep);
  if (!resolved.length) throw new Error('None of the clips could be matched to an episode on the site.');
  await fs.mkdir(clipDir, { recursive: true });

  // One player capture serves every clip of the same episode.
  const byEpisode = new Map();
  for (const c of resolved) {
    if (!byEpisode.has(c.ep.id)) byEpisode.set(c.ep.id, []);
    byEpisode.get(c.ep.id).push(c);
  }

  const cutOne = async (found, c, epLen) => {
    const startSec = Math.max(0, c.range.start - pad);
    c.padBefore = c.range.start - startSec;
    let durSec = c.range.end - c.range.start + c.padBefore + pad;
    if (epLen) {
      if (c.range.start >= epLen - 1) {
        return { ok: false, final: true, why: `starts at ${mmss(c.range.start)} but the episode is only ${mmss(epLen)} long` };
      }
      durSec = Math.min(durSec, Math.floor(epLen - startSec)); // a clip running past the end is shortened
    }
    const name = `${tag(c)}_clip${clip2(c.index)}_${slug(c.ep.title || c.episodeTitle)}.mp4`;
    const outFile = normalizeOutFile(path.join(clipDir, name));
    const reject = async why => {
      await fs.rm(outFile, { force: true }); // never leave a partial file next to the good ones
      return { ok: false, why };
    };
    console.log(`Cutting clip ${c.index}: ${tag(c)} ${mmss(startSec)} +${durSec}s -> ${path.basename(outFile)} ${accurate ? '(re-encode)' : '(stream copy)'}`);
    const code = await cutClip({ ...found, start: String(startSec), duration: String(durSec), accurate, outFile, slowSeek: opts.slowSeek });
    if (code !== 0) return reject(`ffmpeg exited with code ${code}`);
    try {
      const probe = await probeMedia(outFile);
      // Stream copy can start a few seconds early, never half a clip short: a big shortfall means the
      // start time lies at or past the end of the episode (or the stream ended early).
      if (probe.duration < Math.max(0.5, durSec * 0.5)) {
        return reject(`cut is only ${probe.duration.toFixed(1)}s of the ${durSec}s requested (timestamp past the end of the episode?)`);
      }
      return { ok: true, outFile, probe };
    } catch (err) {
      return reject(err.message);
    }
  };

  for (const group of byEpisode.values()) {
    const first = group[0];
    const ep = first.ep;
    console.log(`\n== ${tag(first)} "${ep.title ?? first.episodeTitle}" (${group.length} clip${group.length > 1 ? 's' : ''}) ==`);
    let type = 'sub';
    if (opts.dub) type = 'dub';
    else if (ep.sub === false && ep.dub) type = 'dub';

    let urls;
    try {
      urls = await getEmbedUrls(ep.id, type);
      if (!urls.length) throw new Error(`the API returned no ${type} players`);
    } catch (err) {
      for (const c of group) c.error = `no player: ${err.message}`;
      continue;
    }

    let found = await captureFromAny(urls);
    const epLen = found ? await streamDuration(found) : null;
    if (found && !epLen) console.warn('  could not read the episode length; clips past the end will not be caught.');
    else if (epLen) console.log(`  episode length ${mmss(epLen)}`);
    for (const c of group) {
      let result = found ? await cutOne(found, c, epLen) : { ok: false, why: 'no playlist from any server (try SHOW=1 WAIT=60)' };
      if (!result.ok && !result.final) {
        // The playlist token is short-lived: capture a fresh one and retry once.
        console.warn(`  clip ${c.index} failed (${result.why}); retrying with a fresh playlist...`);
        found = await captureFromAny(urls);
        result = found ? await cutOne(found, c, epLen) : result;
      }
      if (result.ok) Object.assign(c, { file: result.outFile, probe: result.probe });
      else c.error = result.why;
    }
  }

  const done = clips.filter(c => c.file);
  const failed = clips.filter(c => !c.file);

  // Report first, so it exists even if the export step fails.
  const report = {
    source: path.resolve(opts.timeline), series: bp.series_title, pad, accurate,
    clips: clips.map(c => ({
      clip_index: c.index, season: c.season, episode: c.episode, json_episode: c.jsonEpisode ?? null,
      episode_title: c.ep?.title ?? c.episodeTitle,
      estimated: c.beat.timestamp_estimate, file: c.file ?? null, error: c.error ?? null, note: c.note ?? null,
    })),
  };
  await fs.writeFile(path.join(outDir, 'edit_report.json'), JSON.stringify(report, null, 2), 'utf8');

  if (!done.length) throw new Error(`No clip could be cut. See ${path.join(outDir, 'edit_report.json')}.`);

  const projectName = `${bp.series_title} - ${bp.character ?? 'edit'}`.slice(0, 120);
  const fcpxml = path.join(outDir, `${slug(bp.series_title)}_${slug(bp.character || 'edit')}.fcpxml`);
  const { totalSeconds } = await writeResolveProject(fcpxml, {
    projectName,
    clips: done.map(c => ({
      file: c.file,
      probe: c.probe,
      name: `${c.index}. ${tag(c)} ${c.ep.title ?? c.episodeTitle}`,
      markers: [markerFor(c, c.padBefore)],
    })),
  });

  console.log(`\nDone: ${done.length}/${clips.length} clips, ${totalSeconds.toFixed(0)} s on the timeline.`);
  console.log(`  Clips:    ${clipDir}`);
  console.log(`  DaVinci:  ${fcpxml}   (Resolve: File > Import > Timeline)`);
  console.log(`  Report:   ${path.join(outDir, 'edit_report.json')}`);
  if (failed.length) {
    console.warn(`\n${failed.length} clip(s) missing from the timeline:`);
    for (const c of failed) console.warn(`  clip ${c.index} ${tag(c)}: ${c.error}`);
    process.exitCode = 2;
  }
  if (!accurate) console.log('\nStream-copy cuts snap to keyframes; the handles (--pad) absorb that. Use --accurate for frame-exact cuts.');
  console.log('Keep the clips folder where it is: the timeline links to those files (Resolve can relink if moved).');
}

// ------------------------------------------------------------------ CLI

function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--anime' || a === '-a') opts.anime = argv[++i];
    else if (a === '--episode' || a === '-e') opts.episode = argv[++i];
    else if (a === '--pick') opts.pick = argv[++i];
    else if (a === '--start') opts.start = argv[++i];
    else if (a === '--duration' || a === '-t') opts.duration = argv[++i];
    else if (a === '--out' || a === '-o') opts.out = argv[++i];
    else if (a === '--timeline' || a === '-j') opts.timeline = argv[++i];
    else if (a === '--out-dir') opts.outDir = argv[++i];
    else if (a === '--pad') opts.pad = argv[++i];
    else if (a === '--dry-run') opts.dryRun = true;
    else if (a === '--dub') opts.dub = true;
    else if (a === '--accurate') opts.accurate = true;
    else if (a === '--slow-seek') opts.slowSeek = true;
    else if (a === '--help' || a === '-h') opts.help = true;
    else if (!a.startsWith('-') && !opts.anime) opts.anime = a;
    else throw new Error(`Unknown argument: ${a}`);
  }
  return opts;
}

const USAGE = `Usage:
  node clip.mjs                          interactive: asks for name, episode, start, duration
  node clip.mjs "one piece"              start from a name
  node clip.mjs --anime "one piece" --episode 1 --start 1:30 --duration 45 [--dub] [--accurate]
                [--slow-seek] [--pick N] [--out clip.mp4]

  node clip.mjs --timeline timeline_blueprint.json [--out-dir DIR] [--pad 4] [--dub] [--accurate]
                [--slow-seek] [--pick N] [--dry-run]

Timeline mode reads the JSON written by director_orchestrator.py: for every clip it finds the season,
episode (number, checked against the title) and time range, cuts it, and writes <out-dir>/<series>.fcpxml
for DaVinci Resolve (File > Import > Timeline). --pad adds that many seconds of handles before and after
each clip (default 4) because the timestamps are estimates. --dry-run only resolves episodes.

Times: seconds (90), MM:SS (1:30) or HH:MM:SS (00:01:30).
  --pick N      choose the Nth search result instead of the first
  --slow-seek   read from the start instead of jumping to the start point (use if ffmpeg stalls)
Env: API=http://localhost:5000  SHOW=1 (visible browser)  WAIT=30 (seconds per server)`;

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) return console.log(USAGE);
  if (opts.timeline) return runTimeline(opts);

  const rl = readline.createInterface({ input: stdin, output: stdout });
  const ask = async (question, valid, hint) => {
    for (;;) {
      const answer = (await rl.question(question)).trim();
      if (valid(answer)) return answer;
      console.log(hint);
    }
  };

  let anime, ep, type, urls, start, duration, accurate, outFile;
  try {
    anime = await pickAnime(ask, opts);
    anime = await pickSeason(ask, anime, opts);
    ep = await pickEpisode(ask, anime, opts);

    type = 'sub';
    if (opts.dub) type = 'dub';
    else if (ep.sub === false && ep.dub) type = 'dub';
    else if (ep.sub !== false && ep.dub && !(opts.start && opts.duration)) {
      const answer = await ask('Dub instead of sub? (y/N): ', () => true);
      if (answer.toLowerCase() === 'y') type = 'dub';
    }

    console.log(`\nLooking up the ${type} player for episode ${ep.number}...`);
    urls = await getEmbedUrls(ep.id, type);
    if (!urls.length) throw new Error(`The API returned no ${type} players for episode ${ep.number}.`);

    start = opts.start ?? (await ask('Start time of the clip: ', v => toSeconds(v) !== null, TIME_HINT));
    duration = opts.duration ?? (await ask('Duration of the clip: ', v => toSeconds(v) > 0, TIME_HINT));
    if (toSeconds(start) === null || !(toSeconds(duration) > 0)) {
      throw new Error(`Invalid start or duration. ${TIME_HINT}`);
    }
    accurate =
      opts.accurate ??
      (opts.start && opts.duration
        ? false
        : (await rl.question('Frame-accurate cut? Re-encodes, slower (y/N): ')).trim().toLowerCase() === 'y');

    const defaultName = `${slug(anime.title)}_ep${ep.number}_${start.replace(/[:.]/g, '-')}_${duration.replace(/[:.]/g, '-')}.mp4`;
    const fullyScripted = Boolean(opts.start && opts.duration);
    outFile = normalizeOutFile(
      opts.out ??
        (fullyScripted
          ? defaultName
          : (await rl.question(`Output file [${defaultName}]: `)).trim() || defaultName)
    );
  } finally {
    rl.close();
  }

  // Capture the playlist, trying each server until one works. Cut immediately after (token is short-lived).
  const found = await captureFromAny(urls);
  if (!found) throw new Error('Could not capture a playlist from any server. Try SHOW=1 WAIT=60 to watch the player.');

  console.log('Playlist found.');
  console.log(`Cutting ${duration} from ${start} of "${anime.title}" episode ${ep.number} -> ${outFile} ${accurate ? '(re-encode)' : '(stream copy)'}\n`);
  const code = await cutClip({ ...found, start, duration, accurate, outFile, slowSeek: opts.slowSeek });

  if (code === 0) {
    console.log(`\nDone: ${outFile}`);
    if (!accurate) console.log('Note: stream-copy cuts snap to the nearest keyframe, so the start can be a few seconds off.');
  } else {
    console.error(`\nffmpeg exited with code ${code}. If it reports a 403 or can't open the playlist, the token probably expired; run again.`);
    process.exitCode = code;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(err => {
    console.error(`\nError: ${err.message}`);
    if (err instanceof ApiError && err.network) console.error(API_HELP);
    process.exit(1);
  });
}
