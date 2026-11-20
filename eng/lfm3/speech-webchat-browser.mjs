#!/usr/bin/env node
// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Drive the bundled Web UI's spoken replies in headless Chrome against a running TensorSharp.Server.Host with
// LFM3-Omni and its mmproj loaded, over the Chrome DevTools Protocol (Node's built-in WebSocket; no packages): the
// "Speak replies" switch is offered, a spoken turn schedules its audio chunks on the page's AudioContext as they
// arrive, and the finished bubble holds an <audio> player whose WAV decodes and a Download link; a second spoken turn
// sends the first back with its audioId. Screenshots and a JSON report go to --output (under artifacts/ or
// docs/validation/).
//
//   node eng/lfm3/speech-webchat-browser.mjs --url http://127.0.0.1:18765 --output artifacts/lfm3-speech-webchat/browser \
//     [--first "Use Python to compute 37 times 43."] [--second "Now write it into answer.txt."]
// --first / --second replace the two questions (e.g. turns that run the host's code tools, on a server with --code-exec).
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, arg, i, all) => {
  if (arg.startsWith('--')) pairs.push([arg.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return pairs;
}, []));
const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const output = path.resolve(args.output || '');
if (!args.output || !['artifacts', 'docs/validation'].some(folder => output.startsWith(path.join(repository, folder) + path.sep)))
  throw new Error('--output must be below the ignored artifacts/ or docs/validation/ folders.');
mkdirSync(output, { recursive: true });
const base = String(args.url || 'http://127.0.0.1:18765').replace(/\/$/, '');
const report = { url: base, steps: [], screenshots: [] };
const step = (name, detail) => { report.steps.push({ name, ...detail }); console.log('PASS', name, detail ? JSON.stringify(detail) : ''); };

function chromePath() {
  if (args.chrome) return args.chrome;
  const candidates = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'];
  const cache = path.join(os.homedir(), 'Library/Caches/ms-playwright');
  if (existsSync(cache)) for (const dir of readdirSync(cache).filter(d => d.startsWith('chromium-')).sort().reverse())
    candidates.push(path.join(cache, dir, 'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'));
  const found = candidates.find(existsSync); if (!found) throw new Error('No Chrome found; pass --chrome.'); return found;
}

const profile = mkdtempSync(path.join(os.tmpdir(), 'speech-ui-'));
const chrome = spawn(chromePath(), ['--headless=new', '--remote-debugging-port=0', '--user-data-dir=' + profile, '--no-first-run',
  '--no-default-browser-check', '--window-size=1280,900', '--autoplay-policy=no-user-gesture-required', 'about:blank'],
{ stdio: ['ignore', 'ignore', 'pipe'] });
let socket; let nextId = 0; const pending = new Map();
async function cdp(method, params = {}) {
  const id = ++nextId; socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject, method }));
}
async function evaluate(expression) {
  const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true });
  if (r.exceptionDetails) throw new Error('In page: ' + (r.exceptionDetails.exception && r.exceptionDetails.exception.description || r.exceptionDetails.text));
  return r.result.value;
}
async function until(expression, what, timeout = 180000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await evaluate(expression)) return; await new Promise(r => setTimeout(r, 250)); }
  throw new Error('Timed out waiting for ' + what);
}
async function shot(name) {
  const r = await cdp('Page.captureScreenshot', { format: 'png' });
  const file = path.join(output, name + '.png'); writeFileSync(file, Buffer.from(r.data, 'base64')); report.screenshots.push(path.basename(file));
}

async function send(text) {
  await evaluate(`(() => { const t = document.getElementById('message-input'); t.value = ${JSON.stringify(text)};
    t.dispatchEvent(new Event('input', { bubbles: true })); document.getElementById('btn-send').click(); return true; })()`);
}

async function scenario() {
  await cdp('Page.enable'); await cdp('Runtime.enable');
  // Count what the page schedules on its AudioContext, before the page's script runs.
  await cdp('Page.addScriptToEvaluateOnNewDocument', { source: `
    window.__speech = { started: 0, rates: [] };
    const start = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function (...a) { window.__speech.started++;
      if (this.buffer) window.__speech.rates.push(this.buffer.sampleRate); return start.apply(this, a); };` });
  await cdp('Page.navigate', { url: base + '/' });
  await until(`document.getElementById('speech-toggle-label') && getComputedStyle(document.getElementById('speech-toggle-label')).display !== 'none'`,
    'the Speak replies switch', 60000);
  step('Speak replies switch offered', { voices: await evaluate(`Array.from(document.querySelectorAll('#speech-voice option')).map(o => o.value)`) });
  await evaluate(`(() => { const t = document.getElementById('speech-toggle'); t.checked = true; t.dispatchEvent(new Event('change')); return true; })()`);
  await shot('01-switch-on');

  await send(args.first || 'What is the capital of France? Answer in one short sentence.');
  await until(`window.__speech.started > 0`, 'the first audio chunk to be scheduled', 120000);
  const live = await evaluate(`({ started: window.__speech.started, stop: !!document.querySelector('.spoken-reply-live') })`);
  step('audio chunks play as they arrive', live);
  await until(`!!document.querySelector('.message.assistant .spoken-reply audio[src]') && !isGenerating`, 'the finished spoken reply', 180000);
  const finished = await evaluate(`(async () => {
    const audio = document.querySelector('.message.assistant .spoken-reply audio');
    const link = document.querySelector('.message.assistant .spoken-reply-download');
    if (audio.readyState < 1) await new Promise(r => { audio.onloadedmetadata = r; audio.onerror = r; setTimeout(r, 10000); });
    const bytes = new Uint8Array(await (await fetch(link.href)).arrayBuffer());
    const header = String.fromCharCode(...bytes.slice(0, 4)) + '/' + String.fromCharCode(...bytes.slice(8, 12));
    return { src: audio.getAttribute('src'), duration: audio.duration, error: audio.error && audio.error.code,
      download: link.getAttribute('download'), label: link.textContent, header, bytes: bytes.length,
      chunks: window.__speech.started, rates: Array.from(new Set(window.__speech.rates)),
      history: chatHistory.map(m => ({ role: m.role, audioId: m.audioId || null, audioUrl: m.audioUrl || null })),
      answer: document.querySelector('.message.assistant .bubble-text').textContent.slice(0, 400),
      notes: Array.from(document.querySelectorAll('.message.assistant .turn-notice, .message.assistant .turn-note')).map(n => n.textContent),
      stats: document.querySelector('.message.assistant .stats').textContent };
  })()`);
  if (!(finished.duration > 0) || finished.header !== 'RIFF/WAVE' || !finished.download)
    throw new Error('The finished reply does not play or download: ' + JSON.stringify(finished));
  step('player and download of the finished reply', finished);
  await shot('02-finished-reply');

  // The second spoken turn sends the first back by its audioId.
  let sent = null;
  await cdp('Network.enable');
  const requestSeen = new Promise(resolve => { onRequest = resolve; });
  await send(args.second || 'And what is the capital of Germany? Answer in one short sentence.');
  sent = await Promise.race([requestSeen, new Promise(r => setTimeout(() => r(null), 30000))]);
  await until(`document.querySelectorAll('.message.assistant .spoken-reply audio[src]').length >= 2 && !isGenerating`, 'the second spoken reply', 180000);
  const second = await evaluate(`({ stats: Array.from(document.querySelectorAll('.message.assistant .stats')).map(s => s.textContent),
    players: document.querySelectorAll('.message.assistant .spoken-reply audio').length,
    answers: Array.from(document.querySelectorAll('.message.assistant .bubble-text')).map(b => b.textContent.slice(0, 300)) })`);
  const body = sent && JSON.parse(sent);
  const back = body && body.messages && body.messages[1];
  if (!back || !back.audioId) throw new Error('The second request did not carry the first reply\'s audioId: ' + sent);
  step('second turn carries the first back by its audioId', { audioId: back.audioId, speech: body.speech, ...second });
  await shot('03-second-reply');
}

let onRequest = null;
(async () => {
  let wsUrl = null; let errText = '';
  chrome.stderr.on('data', d => { errText += d; const m = /DevTools listening on (ws:\/\/\S+)/.exec(errText); if (m) wsUrl = m[1]; });
  for (let i = 0; i < 100 && !wsUrl; i++) await new Promise(r => setTimeout(r, 100));
  if (!wsUrl) throw new Error('Chrome did not start: ' + errText);
  const targets = await (await fetch(wsUrl.replace('ws://', 'http://').replace(/\/devtools\/browser\/.*/, '/json/list'))).json();
  socket = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject, method } = pending.get(message.id); pending.delete(message.id);
      if (message.error) reject(new Error(method + ': ' + message.error.message)); else resolve(message.result);
    } else if (message.method === 'Network.requestWillBeSent' && message.params.request.url.endsWith('/api/chat')
      && message.params.request.postData && onRequest) {
      const done = onRequest; onRequest = null; done(message.params.request.postData);
    }
  };
  try {
    await scenario();
    report.passed = true;
  } catch (e) {
    report.passed = false; report.error = String(e && e.stack || e);
    try { await shot('99-failure'); } catch { }
    console.error('FAIL', e);
  } finally {
    writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
    chrome.kill();
    process.exit(report.passed ? 0 : 1);
  }
})();
