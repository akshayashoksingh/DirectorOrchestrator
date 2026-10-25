// resolve-export.mjs — turn a list of cut clips into a DaVinci Resolve timeline (FCPXML 1.9).
//
// Import in Resolve:  File > Import > Timeline... > pick the .fcpxml
// Every clip is placed back to back on V1/A1 in arc order. Each clip carries a marker with the
// emotional beat, and a second marker with the dialogue anchor and director's notes, so the
// edit instructions sit on the timeline next to the footage.
//
// Needs ffprobe (ships with ffmpeg) on PATH to read each clip's real fps, size and duration.

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { writeFile } from 'node:fs/promises';

const run = promisify(execFile);

// ------------------------------------------------------------------ probing

function parseRate(text) {
  const [n, d = '1'] = String(text ?? '').split('/');
  const num = Number(n);
  const den = Number(d);
  if (!Number.isFinite(num) || !Number.isFinite(den) || num <= 0 || den <= 0) return null;
  const fps = num / den;
  return fps >= 1 && fps <= 120 ? { num, den } : null;
}

/** Read fps, size, duration and audio info of a media file. Throws if it has no video stream. */
export async function probeMedia(file) {
  let stdout;
  try {
    ({ stdout } = await run(
      'ffprobe',
      ['-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', file],
      { maxBuffer: 16 * 1024 * 1024 }
    ));
  } catch (err) {
    if (err.code === 'ENOENT') throw new Error('ffprobe not found on PATH (it installs together with ffmpeg).');
    throw new Error(`ffprobe could not read ${file}: ${err.message.split('\n')[0]}`);
  }
  const info = JSON.parse(stdout);
  const video = (info.streams ?? []).find(s => s.codec_type === 'video');
  if (!video) throw new Error(`${file} has no video stream`);
  const audio = (info.streams ?? []).find(s => s.codec_type === 'audio');
  const rate = parseRate(video.r_frame_rate) ?? parseRate(video.avg_frame_rate) ?? { num: 24000, den: 1001 };
  const duration = Number(video.duration) || Number(info.format?.duration) || 0;
  return {
    width: Number(video.width) || 1920,
    height: Number(video.height) || 1080,
    rate,
    duration,
    hasAudio: Boolean(audio),
    audioChannels: Number(audio?.channels) || 2,
    audioRate: Number(audio?.sample_rate) || 48000,
  };
}

// ------------------------------------------------------------------ FCPXML helpers

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/** Frame count at a frame rate -> FCPXML rational time, e.g. 48 frames @ 24000/1001 -> "1001/500s". */
function frames(count, { num, den }) {
  const n = Math.round(count) * den;
  if (n === 0) return '0s';
  const g = gcd(n, num);
  return `${n / g}/${num / g}s`;
}

const secondsToFrames = (sec, { num, den }) => Math.round((sec * num) / den);

const esc = s =>
  String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, ''); // illegal in XML 1.0

/**
 * Build the FCPXML text.
 *
 * clips: [{ file, name, markers: [{ atSec, value, note }] , probe? }]  (in timeline order)
 * Each clip's probe result is looked up if not supplied.
 */
export async function buildFcpxml({ projectName, eventName = projectName, clips }) {
  if (!clips.length) throw new Error('No clips to put on the timeline.');
  for (const c of clips) c.probe ??= await probeMedia(c.file);

  // One <format> per distinct size + frame rate; the timeline uses the first clip's.
  const formats = new Map();
  const formatFor = p => {
    const key = `${p.width}x${p.height}@${p.rate.num}/${p.rate.den}`;
    if (!formats.has(key)) formats.set(key, { id: `r${formats.size + 1}`, p });
    return formats.get(key).id;
  };
  const seqProbe = clips[0].probe;
  const seqFormat = formatFor(seqProbe);
  const seqRate = seqProbe.rate;

  const assets = clips.map((c, i) => ({ id: `a${i + 1}`, format: formatFor(c.probe), clip: c }));

  let offsetFrames = 0;
  const spine = [];
  for (const { id, format, clip } of assets) {
    const durFrames = Math.max(1, secondsToFrames(clip.probe.duration, seqRate));
    const markers = (clip.markers ?? [])
      .map(m => {
        const at = Math.min(Math.max(0, m.atSec ?? 0), Math.max(0, clip.probe.duration - 0.05));
        const attrs =
          `start="${frames(secondsToFrames(at, clip.probe.rate), clip.probe.rate)}" ` +
          `duration="${frames(1, clip.probe.rate)}" value="${esc(m.value)}"` +
          (m.note ? ` note="${esc(m.note)}"` : '');
        return `            <marker ${attrs}/>`;
      })
      .join('\n');
    spine.push(
      `          <asset-clip ref="${id}" offset="${frames(offsetFrames, seqRate)}" name="${esc(clip.name)}" ` +
        `start="0s" duration="${frames(durFrames, seqRate)}" format="${format}" tcFormat="NDF">` +
        (markers ? `\n${markers}\n          </asset-clip>` : '</asset-clip>')
    );
    offsetFrames += durFrames;
  }

  const formatXml = [...formats.values()]
    .map(({ id, p }) =>
      `    <format id="${id}" frameDuration="${frames(1, p.rate)}" width="${p.width}" height="${p.height}" colorSpace="1-1-1 (Rec. 709)"/>`
    )
    .join('\n');

  const assetXml = assets
    .map(({ id, format, clip }) => {
      const p = clip.probe;
      const audio = p.hasAudio
        ? ` hasAudio="1" audioSources="1" audioChannels="${p.audioChannels}" audioRate="${p.audioRate}"`
        : '';
      const durFrames = Math.max(1, secondsToFrames(p.duration, p.rate));
      return (
        `    <asset id="${id}" name="${esc(path.basename(clip.file))}" start="0s" ` +
        `duration="${frames(durFrames, p.rate)}" hasVideo="1" format="${format}"${audio} ` +
        `src="${esc(pathToFileURL(path.resolve(clip.file)).href)}"/>`
      );
    })
    .join('\n');

  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
${formatXml}
${assetXml}
  </resources>
  <library>
    <event name="${esc(eventName)}">
      <project name="${esc(projectName)}">
        <sequence format="${seqFormat}" duration="${frames(offsetFrames, seqRate)}" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
${spine.join('\n')}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
`;
}

/** Build the FCPXML and write it. Returns { file, totalSeconds }. */
export async function writeResolveProject(outFile, opts) {
  const xml = await buildFcpxml(opts);
  await writeFile(outFile, xml, 'utf8');
  const total = opts.clips.reduce((sum, c) => sum + (c.probe?.duration ?? 0), 0);
  return { file: outFile, totalSeconds: total };
}
