#!/usr/bin/env node
// anime-clip.mjs — capture the HLS playlist from an embed player and cut a clip with ffmpeg.
//
// Interactive:   node anime-clip.mjs
// With args:     node anime-clip.mjs "<embed-url>" --start 1:30 --duration 45 [--accurate] [--out clip.mp4]
// Link only:     node anime-clip.mjs "<embed-url>" --capture-only [--json]
//
// Env: SHOW=1 (visible browser, for debugging), WAIT=45 (seconds to wait for the playlist, default 30)
//
// Also usable as a module from another project:
//   import { capturePlaylist, cutClip } from './anime-clip.mjs';

import { spawn, spawnSync } from 'node:child_process';
import readline from 'node:readline/promises';
import { stdin, stdout } from 'node:process';
import { pathToFileURL } from 'node:url';

// ---------------------------------------------------------------- time helpers

/** "90" | "1:30" | "01:02:03" | "12.5"  ->  seconds, or null if invalid */
export function toSeconds(text) {
  const parts = String(text).trim().split(':');
  if (parts.length < 1 || parts.length > 3) return null;
  if (!parts.every(p => /^\d+(\.\d+)?$/.test(p))) return null;
  return parts.map(Number).reduce((total, n) => total * 60 + n, 0);
}

const TIME_HINT = 'Use seconds (90), MM:SS (1:30) or HH:MM:SS (00:01:30).';

// ---------------------------------------------------------------- 1. capture

/**
 * Loads the embed inside an iframe (the player refuses to run as a top-level page)
 * and returns the HLS playlist request the player makes.
 * @returns {{ playlistUrl: string, referer: string, origin: string, userAgent: string }}
 */
export async function capturePlaylist(embedUrl, { show = false, waitMs = 30000 } = {}) {
  const { chromium } = await import('playwright');

  const browser = await chromium.launch({
    headless: !show,
    args: ['--autoplay-policy=no-user-gesture-required'],
  });

  try {
    const context = await browser.newContext();
    const page = await context.newPage();

    // Give the wrapper page a real http origin instead of about:blank / file://
    const HOST = 'http://local.test/';
    const playerUrl = embedUrl.replace('autoPlay=0', 'autoPlay=1');
    await page.route(HOST, route =>
      route.fulfill({
        contentType: 'text/html',
        body: `<!doctype html><iframe src="${playerUrl}" width="960" height="540"
          allow="autoplay; fullscreen" allowfullscreen></iframe>`,
      })
    );

    let hit = null;
    // context-level so requests from iframes and workers are included
    context.on('request', req => {
      if (!hit && /\.m3u8(\?|$)/i.test(req.url())) hit = req;
    });
    context.on('response', res => {
      const type = res.headers()['content-type'] || '';
      if (!hit && /mpegurl|m3u8/i.test(type)) hit = res.request();
    });

    await page.goto(HOST);

    // Nudge playback in case autoplay is blocked
    setTimeout(async () => {
      try {
        await page.frameLocator('iframe').locator('body').click({ timeout: 2000 });
      } catch {}
    }, 5000);

    const started = Date.now();
    while (!hit && Date.now() - started < waitMs) await page.waitForTimeout(500);

    if (!hit) {
      throw new Error(
        `No HLS playlist seen in ${waitMs / 1000}s. Try SHOW=1 WAIT=60 to watch what the player does.`
      );
    }

    const headers = hit.headers();
    return {
      playlistUrl: hit.url(),
      referer: headers['referer'] || headers['origin'] || '',
      origin: headers['origin'] || '',
      userAgent: headers['user-agent'] || 'Mozilla/5.0',
    };
  } finally {
    await browser.close();
  }
}

// ---------------------------------------------------------------- 2. cut

/**
 * The playlist/segment URLs have no file extension (e.g. /p?t=...). Newer ffmpeg builds refuse
 * those unless told otherwise, but older builds don't know the options and fail if they're passed.
 * So ask the installed ffmpeg which of them it supports.
 */
let hlsOptionCache;
export function hlsExtraOptions() {
  if (hlsOptionCache) return hlsOptionCache;
  const out = spawnSync('ffmpeg', ['-hide_banner', '-h', 'demuxer=hls'], { encoding: 'utf8' });
  const text = `${out.stdout ?? ''}${out.stderr ?? ''}`;
  const opts = [];
  if (/-allowed_segment_extensions\b/.test(text)) opts.push('-allowed_segment_extensions', 'ALL');
  if (/-extension_picky\b/.test(text)) opts.push('-extension_picky', '0');
  return (hlsOptionCache = opts);
}

const OUT_EXT = /\.(mp4|mkv|mov|ts|webm)$/i;

/** Fixes typos like "clip.pp4" or a missing extension so ffmpeg can pick an output format. */
export function normalizeOutFile(name) {
  if (OUT_EXT.test(name)) return name;
  const fixed = `${name.replace(/\.[^./\\]*$/, '')}.mp4`;
  console.warn(`Output name "${name}" has no usable extension; using "${fixed}" instead.`);
  return fixed;
}

export function buildFfmpegArgs({
  playlistUrl, referer, userAgent, start, duration, accurate = false, outFile, hlsOptions = [],
  slowSeek = false,
}) {
  return [
    '-hide_banner',
    '-headers', `Referer: ${referer}\r\nUser-Agent: ${userAgent}\r\n`,
    '-allowed_extensions', 'ALL',
    ...hlsOptions,
    '-rw_timeout', '20000000', // give up after 20s without data instead of hanging forever
    // Fast seek (default): jump straight to the right segment.
    // Slow seek: read from the beginning up to the start point (only the clip is written). More robust.
    ...(slowSeek ? [] : ['-ss', String(start)]),
    '-i', playlistUrl,
    ...(slowSeek ? ['-ss', String(start)] : []),
    '-t', String(duration),
    ...(accurate
      ? ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-c:a', 'aac']
      : ['-c', 'copy']),
    '-movflags', '+faststart',
    '-y', outFile,
  ];
}

/** Runs ffmpeg; resolves with its exit code. Rejects if ffmpeg can't be started. */
export function cutClip(options) {
  return new Promise((resolve, reject) => {
    const args = buildFfmpegArgs({ hlsOptions: hlsExtraOptions(), ...options });
    const ff = spawn('ffmpeg', args, { stdio: 'inherit' });
    ff.on('error', err =>
      reject(
        err.code === 'ENOENT'
          ? new Error('ffmpeg was not found. Install it and make sure `ffmpeg -version` works.')
          : err
      )
    );
    ff.on('close', code => resolve(code ?? 1));
  });
}

// ---------------------------------------------------------------- CLI

function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--start') opts.start = argv[++i];
    else if (a === '--duration' || a === '-t') opts.duration = argv[++i];
    else if (a === '--out' || a === '-o') opts.out = argv[++i];
    else if (a === '--accurate') opts.accurate = true;
    else if (a === '--slow-seek') opts.slowSeek = true;
    else if (a === '--capture-only') opts.captureOnly = true;
    else if (a === '--json') opts.json = true;
    else if (a === '--help' || a === '-h') opts.help = true;
    else if (!a.startsWith('-') && !opts.url) opts.url = a;
    else throw new Error(`Unknown argument: ${a}`);
  }
  return opts;
}

const USAGE = `Usage:
  node anime-clip.mjs                      interactive
  node anime-clip.mjs "<embed-url>" --start 1:30 --duration 45 [--accurate] [--out clip.mp4]
  node anime-clip.mjs "<embed-url>" --capture-only [--json]

Times: seconds (90), MM:SS (1:30) or HH:MM:SS (00:01:30).
Flags: --slow-seek  read from the start instead of jumping to the start point (use if ffmpeg stalls).
Env:   SHOW=1 to watch the browser, WAIT=60 to wait longer for the playlist.`;

const shQuote = s => String(s).replace(/'/g, "\\'");

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) return console.log(USAGE);
  if (opts.url && !/^https?:\/\/\S+$/.test(opts.url)) {
    throw new Error(
      `"${opts.url}" is not a valid embed URL. Replace the placeholder with the real URL ` +
        `from your /api/v2/servers/<episodeId> response (it starts with https://).`
    );
  }

  const log = opts.json ? console.error : console.log;
  const rl = readline.createInterface({ input: stdin, output: opts.json ? process.stderr : stdout });
  const ask = async (question, valid, hint) => {
    for (;;) {
      const answer = (await rl.question(question)).trim();
      if (valid(answer)) return answer;
      console.error(hint);
    }
  };

  let job;
  try {
    // Gather everything first: the playlist token expires quickly, so capture -> cut must be back to back.
    const url = opts.url ?? (await ask('Embed URL: ', v => /^https?:\/\//.test(v), 'Paste the full embed URL (starts with http).'));

    let mode = opts.captureOnly ? 'capture' : opts.start || opts.duration ? 'clip' : null;
    if (!mode) {
      const choice = await ask(
        '\nWhat do you want to do?\n  1) Cut a clip\n  2) Just get the playlist link\n> ',
        v => v === '1' || v === '2',
        'Enter 1 or 2.'
      );
      mode = choice === '1' ? 'clip' : 'capture';
    }

    if (mode === 'clip') {
      const valid = v => toSeconds(v) !== null;
      const start = opts.start ?? (await ask('Start time of the clip: ', valid, TIME_HINT));
      const duration = opts.duration ?? (await ask('Duration of the clip: ', v => valid(v) && toSeconds(v) > 0, TIME_HINT));
      if (toSeconds(start) === null || !(toSeconds(duration) > 0)) throw new Error(`Invalid start/duration. ${TIME_HINT}`);

      const accurate = opts.accurate ?? (opts.start && opts.duration
        ? false
        : (await rl.question('Frame-accurate cut? Re-encodes, slower (y/N): ')).trim().toLowerCase() === 'y');

      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const defaultName = `clip_${stamp}.mp4`;
      const outFile = normalizeOutFile(
        opts.out ?? ((await rl.question(`Output file [${defaultName}]: `)).trim() || defaultName)
      );
      job = { mode, url, start, duration, accurate, outFile };
    } else {
      job = { mode, url };
    }
  } finally {
    rl.close();
  }

  log('\nOpening the player and waiting for the playlist...');
  const found = await capturePlaylist(job.url, {
    show: process.env.SHOW === '1',
    waitMs: Number(process.env.WAIT || 30) * 1000,
  });

  if (job.mode === 'capture') {
    if (opts.json) {
      return console.log(JSON.stringify(found, null, 2));
    }
    console.log('\nPlaylist :', found.playlistUrl);
    console.log('Referer  :', found.referer);
    console.log('Origin   :', found.origin || '(none)');
    console.log('\nffmpeg command (bash):\n');
    console.log(
      `ffmpeg -headers $'Referer: ${shQuote(found.referer)}\\r\\nUser-Agent: ${shQuote(found.userAgent)}\\r\\n' ` +
        `-allowed_extensions ALL -i "${found.playlistUrl}" -c copy out.mp4`
    );
    console.log('\nThe link is short-lived; use it right away.');
    return;
  }

  console.log('Playlist found.');
  console.log(
    `Cutting ${job.duration} from ${job.start} -> ${job.outFile} ${job.accurate ? '(re-encode)' : '(stream copy)'}\n`
  );
  const code = await cutClip({ ...found, ...job, slowSeek: opts.slowSeek });
  if (code === 0) {
    console.log(`\nDone: ${job.outFile}`);
    if (!job.accurate) {
      console.log('Note: stream-copy cuts snap to the nearest keyframe, so the start can be a few seconds off.');
    }
  } else {
    console.error(`\nffmpeg exited with code ${code}. If it reports a 403 or can't open the playlist, the token probably expired; run again.`);
    process.exitCode = code;
  }
}

// Run only when executed directly, not when imported as a module.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(err => {
    console.error(`\nError: ${err.message}`);
    process.exit(1);
  });
}
