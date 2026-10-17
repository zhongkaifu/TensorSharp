#!/usr/bin/env node
// Drive the real Video Studio page in headless Chrome against a running host, over the Chrome DevTools
// Protocol (Node's built-in WebSocket; no packages). It imports two stills through the page's own file
// input, makes a take that only ENDS on a photo from the library ("As last frame"), opens and Escapes the
// frame picker, checks the Assistant's video-model state, and screenshots desktop and phone layouts.
// With --submit it also generates that last-frame-only take through the form and waits for it.
//
// Requires a host with a video model loaded, ffmpeg (for the stills) and Chrome or Playwright's Chromium.
// Evidence belongs below ignored artifacts/ or docs/validation/. Example:
//
//   node eng/studio/validate_ui_browser.mjs --url http://127.0.0.1:8187 --output artifacts/studio-optional-frames/ui
//   node eng/studio/validate_ui_browser.mjs --url ... --submit --size 640x384 --output ...
//   node eng/studio/validate_ui_browser.mjs --entry "<TensorAgentHost entryUrl>" --screens-only --output ...   (TensorAgent's page)
//   node eng/studio/validate_ui_browser.mjs --url ... --assistant --output ...   (a tool-capable chat model loaded)
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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
const base = String(args.url || 'http://127.0.0.1:8187').replace(/\/$/, '');
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
function still(name, source, size) {
  const file = path.join(output, name + '.png');
  const run = spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', `${source}=s=${size}`, '-frames:v', '1', file]);
  if (run.status !== 0) throw new Error('ffmpeg failed: ' + run.stderr); return file;
}

const profile = mkdtempSync(path.join(os.tmpdir(), 'studio-ui-'));
const chrome = spawn(chromePath(), ['--headless=new', '--remote-debugging-port=0', '--user-data-dir=' + profile, '--no-first-run',
  '--no-default-browser-check', '--window-size=1440,900', 'about:blank'], { stdio: 'ignore' });
let socket; let nextId = 0; const pending = new Map();
async function cdp(method, params = {}) {
  const id = ++nextId; socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject, method }));
}
async function evaluate(expression) {
  const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error('In page: ' + (r.exceptionDetails.exception && r.exceptionDetails.exception.description || r.exceptionDetails.text));
  return r.result.value;
}
async function until(expression, what, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await evaluate(expression)) return; await new Promise(r => setTimeout(r, 250)); }
  throw new Error('Timed out waiting for ' + what);
}
async function shot(name) {
  const r = await cdp('Page.captureScreenshot', { format: 'png' });
  const file = path.join(output, name + '.png'); writeFileSync(file, Buffer.from(r.data, 'base64')); report.screenshots.push(path.basename(file));
}
async function viewport(width, height, mobile) {
  await cdp('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: mobile ? 3 : 1, mobile });
  await cdp('Emulation.setTouchEmulationEnabled', { enabled: mobile }); await new Promise(r => setTimeout(r, 400));
}
// Clicks the first visible button in the Studio whose visible text or accessible name matches.
const click = (label, scope = '.ts-studio') => evaluate(`(() => {
  const b = Array.from(document.querySelectorAll(${JSON.stringify(scope)} + ' button')).find(n => n.getClientRects().length && (n.textContent.trim() === ${JSON.stringify(label)} || n.getAttribute('aria-label') === ${JSON.stringify(label)}));
  if (!b) throw new Error('No visible button: ' + ${JSON.stringify(label)}); b.focus(); b.click(); return true; })()`);
const texts = scope => evaluate(`Array.from(document.querySelectorAll(${JSON.stringify(scope)})).filter(n => n.getClientRects().length).map(n => n.textContent.trim())`);

// The default scenario: optional frames from the library, the picker, the Assistant's state and the phone layout.
async function uiScenario() {
  const guide = await texts('.ts-studio .ts-guide-step strong');
  if (guide.join('|') !== 'Add footage|Arrange the timeline|Preview and export') throw new Error('Guide steps: ' + guide.join('|'));
  step('guide shown with no project', { guide });
  await shot('01-welcome-guide');

  // Import through the page's own file input; with no project open, the import starts an untitled one.
  const size = String(args['still-size'] || '640x384');
  const stills = [still('studio-first', 'testsrc2', size), still('studio-last', 'smptebars', size)];
  const input = await cdp('Runtime.evaluate', { expression: `document.querySelector('.ts-studio > input[type=file]')` });
  await cdp('DOM.setFileInputFiles', { files: stills, objectId: input.result.objectId });
  await until(`document.querySelectorAll('.ts-studio-library .ts-asset').length >= 2`, 'two imported library assets', 60000);
  step('imported two stills into a new project', { assets: await texts('.ts-studio-library .ts-asset strong') });

  // "As last frame" on a library photo opens Generate with that photo as the take's last frame.
  await click('As last frame: studio-last.png', '.ts-studio-library');
  await until(`!!document.querySelector('.ts-studio-inspector .ts-slot[data-slot=last][data-state=filled]')`, 'the filled last-frame slot');
  const result = await texts('.ts-studio-inspector .ts-gen-result');
  if (!result.includes('The take ends on the last frame.')) throw new Error('Result line: ' + result);
  const focused = await evaluate(`document.activeElement && document.activeElement.dataset.focusKey`);
  step('library photo set as the last frame', { result: result[0], focus: focused });
  await evaluate(`(() => { const t = document.querySelector('.ts-studio-inspector textarea'); t.focus(); t.value = 'The camera drifts slowly and settles on the colour bars.'; t.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
  const summary = await texts('.ts-studio-inspector .ts-gen-summary');
  if (!/^Ends on studio-last\.png · /.test(summary[0] || '')) throw new Error('Summary: ' + summary);
  step('summary names the last frame', { summary: summary[0] });
  await shot('02-generate-last-frame-desktop');

  // The picker opens inline, is reachable by keyboard, and Escape closes it, not Studio.
  await click('Choose photo for the first frame', '.ts-studio-inspector');
  await until(`!!document.querySelector('.ts-studio-inspector .ts-picker')`, 'the inline picker');
  const tiles = await texts('.ts-studio-inspector .ts-picker .ts-picker-name');
  await shot('03-picker-open');
  await cdp('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await cdp('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await until(`!document.querySelector('.ts-studio-inspector .ts-picker')`, 'the picker to close');
  if (!await evaluate(`!document.querySelector('.ts-studio').hidden`)) throw new Error('Escape closed Studio instead of the picker.');
  step('picker lists photos and Escape closes only the picker', { tiles, focusAfter: await evaluate(`document.activeElement && document.activeElement.dataset.focusKey`) });

  // A loaded video model means no chat model: the Assistant says so and does not offer Send.
  await click('Assistant', '.ts-studio-inspector .ts-tabbar');
  await until(`!!document.querySelector('.ts-studio-inspector .ts-chips')`, 'the Assistant tab');
  const send = await evaluate(`(() => { const b = Array.from(document.querySelectorAll('.ts-studio-inspector button')).find(n => n.textContent === 'Send'); return b && b.disabled; })()`);
  const notice = await texts('.ts-studio-inspector .ts-notice strong');
  step('assistant with a video model loaded', { sendDisabled: send, notice: notice[0], chips: await texts('.ts-studio-inspector .ts-chip') });
  await shot('04-assistant-video-model-loaded');

  // Phone layout: the panels are overlays switched by the tab row; the slots stack.
  await viewport(390, 844, true);
  await click('Generate', '.ts-mobile-tabs');
  await until(`!!document.querySelector('.ts-studio-inspector:not([data-panel-hidden=true]) .ts-slots')`, 'Generate on a phone');
  const columns = await evaluate(`getComputedStyle(document.querySelector('.ts-studio-inspector .ts-slots')).gridTemplateColumns.split(' ').length`);
  const smallest = await evaluate(`Math.min(...Array.from(document.querySelectorAll('.ts-studio-inspector button')).filter(n => n.getClientRects().length).map(n => n.getBoundingClientRect().height))`);
  step('phone layout', { slotColumns: columns, smallestButtonHeight: Math.round(smallest) });
  await shot('05-generate-phone');
  await viewport(1440, 900, false);

  if (args.submit) {
    // Generate the last-frame-only take through the form. --size WxH chooses Custom for a quicker preview.
    if (args.size) {
      const [w, h] = String(args.size).split('x');
      await evaluate(`(() => { const s = Array.from(document.querySelectorAll('.ts-studio-inspector label')).find(l => l.firstChild && l.firstChild.textContent === 'Size').querySelector('select'); s.value = 'custom'; s.dispatchEvent(new Event('change', { bubbles: true })); return true; })()`);
      await evaluate(`(() => { const field = name => Array.from(document.querySelectorAll('.ts-studio-inspector label')).find(l => l.firstChild && l.firstChild.textContent === name).querySelector('input');
        for (const [name, value] of [['Width', ${JSON.stringify(w)}], ['Height', ${JSON.stringify(h)}]]) { const n = field(name); n.value = value; n.dispatchEvent(new Event('input', { bubbles: true })); } return true; })()`);
    }
    await click('Generate', '.ts-studio-inspector .ts-tabbar');
    await click('Generate take', '.ts-studio-inspector');
    await until(`!!document.querySelector('.ts-studio-inspector .ts-job')`, 'the queued job', 60000);
    step('submitted from the form', { job: (await texts('.ts-studio-inspector .ts-job .ts-hint'))[0] });
    await shot('06-job-running');
    await until(`!!document.querySelector('.ts-studio-library .ts-asset[data-highlight=true]')`, 'the finished take in the library', Number(args.timeout || 3600) * 1000);
    const take = await texts('.ts-studio-library .ts-asset[data-highlight=true] strong');
    step('finished take highlighted in the library', { take: take[0], status: (await texts('.ts-status'))[0] });
    await shot('07-take-ready');
  }
}

// Any host, any model (or none), any language: the guide, an import, a library photo set as the last frame (Generate
// then says what it needs and keeps the photo), and the phone layout. Text is reported as shown, so a locale can be read.
async function screensScenario() {
  report.guide = await texts('.ts-studio .ts-guide-step strong'); await shot('s1-guide');
  const stills = [still('studio-first', 'testsrc2', '640x384'), still('studio-last', 'smptebars', '640x384')];
  const input = await cdp('Runtime.evaluate', { expression: `document.querySelector('.ts-studio > input[type=file]')` });
  await cdp('DOM.setFileInputFiles', { files: stills, objectId: input.result.objectId });
  await until(`document.querySelectorAll('.ts-studio-library .ts-asset').length >= 2`, 'two imported library assets', 60000);
  report.library = await texts('.ts-studio-library .ts-asset button');
  const lastFrame = await evaluate(`(() => { const b = Array.from(document.querySelectorAll('.ts-studio-library .ts-asset-frames button')).filter(n => n.getClientRects().length)[3]; if (b) b.click(); return b ? b.getAttribute('aria-label') : null; })()`);
  await new Promise(r => setTimeout(r, 800));
  report.generate = await texts('.ts-studio-inspector .ts-stack > *'); report.lastFrameAction = lastFrame;
  step('generate after a library photo was set as the last frame', { lastFrame, generate: report.generate.slice(0, 8) });
  await shot('s2-generate'); await viewport(390, 844, true); await shot('s3-phone'); await viewport(1440, 900, false);
}

// With a tool-capable CHAT model loaded: put a clip on the timeline, select it, ask the Assistant for an edit,
// and apply the proposal it saves. This is the worked example the user guide describes.
async function assistantScenario() {
  const clipFile = path.join(output, 'assistant-clip.mp4');
  const make = spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=24:d=4', '-f', 'lavfi', '-i', 'sine=f=440:d=4',
    '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', clipFile]);
  if (make.status !== 0) throw new Error('ffmpeg failed: ' + make.stderr);
  const input = await cdp('Runtime.evaluate', { expression: `document.querySelector('.ts-studio > input[type=file]')` });
  await cdp('DOM.setFileInputFiles', { files: [clipFile], objectId: input.result.objectId });
  await until(`document.querySelectorAll('.ts-studio-library .ts-asset').length >= 1`, 'the imported clip', 60000);
  await click('Add to timeline: assistant-clip.mp4', '.ts-studio-library');
  await until(`!!document.querySelector('.ts-studio-timeline .ts-clip[aria-pressed=true]')`, 'the selected timeline clip');
  const before = await evaluate(`document.querySelector('.ts-studio-timeline .ts-clip[aria-pressed=true] span').textContent`);
  await click('Assistant', '.ts-studio-inspector .ts-tabbar');
  await until(`!!document.querySelector('.ts-studio-inspector .ts-chips')`, 'the Assistant tab');
  const context = await texts('.ts-studio-inspector > .ts-stack > p.ts-hint');
  const instruction = String(args.instruction || 'Trim the selected clip to its first 2 seconds and lower its volume to 25 percent. Keep its position.');
  await evaluate(`(() => { const t = document.querySelector('.ts-studio-inspector textarea'); t.value = ${JSON.stringify(instruction)}; t.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
  await shot('a1-assistant-ready');
  await click('Send', '.ts-studio-inspector');
  await until(`!!document.querySelector('.ts-studio-inspector .ts-proposal-list') || !!document.querySelector('.ts-studio-inspector .ts-form-error')`, 'a proposal or an error', Number(args.timeout || 600) * 1000);
  const error = await texts('.ts-studio-inspector .ts-form-error');
  if (error.length) throw new Error('Assistant error: ' + error.join(' | '));
  const proposal = { summary: (await texts('.ts-studio-inspector .ts-proposal > strong'))[0], changes: await texts('.ts-studio-inspector .ts-proposal-list li'),
    reply: (await texts('.ts-studio-inspector .ts-bubble[data-role=assistant]')).pop() };
  step('assistant proposed an edit', { instruction, context, proposal });
  await shot('a2-assistant-proposal');
  await click('Apply proposal', '.ts-studio-inspector');
  await until(`!document.querySelector('.ts-studio-inspector .ts-proposal-list')`, 'the proposal to apply');
  const after = await evaluate(`(document.querySelector('.ts-studio-timeline .ts-clip span') || {}).textContent`);
  step('proposal applied as one undoable edit', { clipLengthBefore: before, clipLengthAfter: after, status: (await texts('.ts-status'))[0] });
  await shot('a3-assistant-applied');
  writeFileSync(path.join(output, 'ui.json'), JSON.stringify(report, null, 2));
}

try {
  let port;
  for (let i = 0; i < 100 && !port; i++) {
    const file = path.join(profile, 'DevToolsActivePort');
    if (existsSync(file)) port = readFileSync(file, 'utf8').split('\n')[0]; else await new Promise(r => setTimeout(r, 100));
  }
  if (!port) throw new Error('Chrome did not open a DevTools port.');
  const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => { const m = JSON.parse(event.data); const p = m.id && pending.get(m.id); if (p) { pending.delete(m.id); if (m.error) p.reject(new Error(p.method + ': ' + m.error.message)); else p.resolve(m.result); } };
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('DOM.enable');
  await viewport(1440, 900, false);
  // A fresh browser has no remembered project, so Studio opens on its getting-started guide.
  // --entry opens a host page that signs in by URL first (TensorAgentHost's entryUrl with its token); --lang sets the UI language.
  if (args.lang) await cdp('Emulation.setLocaleOverride', { locale: String(args.lang) });
  await cdp('Page.navigate', { url: (args.entry ? String(args.entry).split('#')[0] : base + '/') + '#studio' });
  await until(`!!document.querySelector('.ts-studio:not([hidden]) .ts-guide')`, 'the Studio guide');
  if (args.assistant) await assistantScenario(); else if (args['screens-only']) await screensScenario(); else await uiScenario();
  writeFileSync(path.join(output, 'ui.json'), JSON.stringify(report, null, 2));
  console.log('UI PASSED', report.steps.length, 'steps; screenshots in', output);
} catch (error) {
  report.error = String(error && error.stack || error); writeFileSync(path.join(output, 'ui.json'), JSON.stringify(report, null, 2));
  try { await shot('zz-failure'); } catch (_) { /* The page may be gone. */ }
  console.error('UI FAILED', error); process.exitCode = 1;
} finally {
  try { socket && socket.close(); } catch (_) { }
  chrome.kill(); setTimeout(() => rmSync(profile, { recursive: true, force: true }), 500);
}
