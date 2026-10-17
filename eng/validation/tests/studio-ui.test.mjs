// Shared Studio client contract tests. These simulate the host contract; they do not
// constitute a browser, real-device, media-rendering, or inference validation.
import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const source = readFileSync(fileURLToPath(new URL('../../../TensorSharp.Chat/WebUi/studio.js', import.meta.url)), 'utf8');
function simulatedDocument() {
  const document = { readyState: 'loading', addEventListener: () => {}, activeElement: null };
  class Element {
    constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.listeners = {}; this.attributes = {}; this.dataset = {}; this.style = {}; this.className = ''; this.textContent = ''; this.inert = false; }
    append(...nodes) { for (const node of nodes) { node.parentNode = this; this.children.push(node); } }
    appendChild(node) { this.append(node); return node; }
    replaceChildren(...nodes) { for (const node of this.children) node.parentNode = null; this.children = []; this.append(...nodes); }
    remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(node => node !== this); this.parentNode = null; }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(event, handler) { (this.listeners[event] ||= []).push(handler); }
    dispatch(event, properties = {}) { const e = { target: this, preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.propagationStopped = true; }, ...properties }; return { event: e, results: (this.listeners[event] || []).map(handler => handler(e)) }; }
    click() { return Promise.all(this.dispatch('click').results); }
    checkValidity() { return true; }
    focus() { document.activeElement = this; }
    get isConnected() { return this === document.body || !!this.parentNode?.isConnected; }
    querySelectorAll(selector) { const matches = node => selector.startsWith('.') ? node.className.split(' ').includes(selector.slice(1)) : node.tagName === selector.toUpperCase(); return this.children.flatMap(node => [...(matches(node) ? [node] : []), ...node.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  }
  document.createElement = tag => new Element(tag); document.createTextNode = value => Object.assign(new Element('#text'), { textContent: value }); document.body = new Element('body');
  return document;
}
function harness() {
  const calls = [], handlers = [];
  const document = simulatedDocument();
  const context = vm.createContext({
    console, URL, Map, Set, Date, Math, JSON, Promise, Error, Object, Array, String, Number, TextDecoder, AbortController,
    setTimeout, clearTimeout,
    window: { location: { href: 'http://127.0.0.1:5000/', origin: 'http://127.0.0.1:5000' }, crypto: { randomUUID: () => '11111111-2222-4333-8444-555555555555' } },
    document,
    fetch: async (url, options) => {
      calls.push({ url, options, body: typeof options?.body === 'string' ? JSON.parse(options.body) : options?.body });
      const next = handlers.shift(); if (!next) throw new Error('Unexpected network call: ' + url);
      return next;
    }
  });
  // Expose private helpers only in this VM, leaving the production public API small.
  const instrumented = source.replace('  function renderProject() {', '  function renderProject() { if (window.__skipRender) return;').replace('  window.TensorSharpStudio = Object.freeze', '  window.__test = { S, ui, api, change, receiveProject, localMediaUrl, frames, seconds, snapped, renderAllowed, history, response, boundedMessages, restoreAssistantHistory, deleteAsset, confirmAssetDeletion, renderLibrary, generationForm, keydown, removeTrack, renderTimeline, jobsPanel, jobsTabText, renderInspector, renderGuide };\n  window.TensorSharpStudio = Object.freeze');
  vm.runInContext(instrumented, context, { filename: 'studio.js' });
  const api = context.window.__test;
  api.ui.save = { textContent: '' }; api.ui.status = { textContent: '', dataset: {} };
  const project = { id: 'a'.repeat(32), name: 'Test', revision: 4, frameRate: { numerator: 30000, denominator: 1001 }, tracks: [], clips: [], assets: [], captions: [] };
  api.S.project = project;
  return { api, calls, handlers, project, context, document };
}
function response(value, status = 200) { return { ok: status < 400, status, statusText: 'Failure', text: async () => JSON.stringify(value) }; }
function deletionHarness(usageCount = 0) {
  const h = harness(), { api, project, document, context } = h;
  context.window.__skipRender = true;
  api.S.root = document.createElement('section'); document.body.appendChild(api.S.root);
  api.ui.library = document.createElement('aside'); api.S.root.appendChild(api.ui.library);
  const trigger = document.createElement('button'); api.ui.library.appendChild(trigger); trigger.focus();
  project.assets.push({ id: 'asset-a', name: '<image & video>', kind: 'video', durationSeconds: 1 });
  project.clips = Array.from({ length: usageCount }, (_, i) => ({ id: 'clip-' + i, assetId: 'asset-a' }));
  return { ...h, trigger, dialog: () => api.S.root.querySelector('.ts-delete-dialog'), dialogButtons: () => api.S.root.querySelector('.ts-delete-dialog').querySelectorAll('button') };
}

function trackHarness(audioClips = 0) {
  const h = harness(), { api, project, document, context } = h;
  context.window.__skipRender = true;
  api.S.root = document.createElement('section'); document.body.appendChild(api.S.root);
  api.ui.timeline = document.createElement('section'); api.S.root.appendChild(api.ui.timeline);
  const trigger = document.createElement('button'); api.ui.timeline.appendChild(trigger); trigger.focus();
  project.durationFrames = 120;
  project.tracks.push({ id: 'track-v', name: 'Video 1', kind: 'video', muted: false, visible: true }, { id: 'track-a', name: '<Music & FX>', kind: 'audio', muted: false, visible: true });
  project.assets.push({ id: 'asset-v', name: 'shot.mp4', kind: 'video', durationSeconds: 10, hasAudio: true }, { id: 'asset-a', name: 'song.wav', kind: 'audio', durationSeconds: 10, hasAudio: true });
  // One clip always sits on the video track, so every count shown must be the removed track's own.
  project.clips = [{ id: 'clip-v', assetId: 'asset-v', trackId: 'track-v', startFrame: 0, durationFrames: 10 },
    ...Array.from({ length: audioClips }, (_, i) => ({ id: 'clip-' + i, assetId: 'asset-a', trackId: 'track-a', startFrame: i * 10, durationFrames: 10 }))];
  return { ...h, trigger, dialog: () => api.S.root.querySelector('.ts-delete-dialog'), dialogButtons: () => api.S.root.querySelector('.ts-delete-dialog').querySelectorAll('button') };
}

function generationHarness(previous = null, overrides = {}, effectiveParameters = undefined) {
  const h = harness(), { api, project, document, handlers } = h;
  const model = { modelId: 'ltx-test', displayName: 'LTX 2.5', family: 'ltx-2.5', available: true, textToVideo: true,
    imageConditioning: true, lastFrameConditioning: true, audioOutput: true, temporalGuidance: true,
    spatialMultiple: 64, temporalMultiple: 8, temporalOffset: 1, maxDimension: 2048, maxFrames: 257,
    defaultWidth: 1024, defaultHeight: 1024, defaultFrameCount: 145, defaultFps: 24,
    defaultSteps: 8, fixedSteps: 8, defaultCfgScale: 1, fixedCfgScale: 1, defaultGenerateAudio: true,
    modes: ['t2v', 'i2v', 'fl2v', 'keyframes'], unavailableModes: { ref: 'Needs the Ingredients IC-LoRA.' }, ...overrides };
  api.S.capabilities = { models: [model] }; api.S.capabilitiesAt = Date.now();
  project.width = 1280; project.height = 720;
  project.assets.push({ id: 'banner', kind: 'image', name: 'banner_1.png', width: 1536, height: 1024 },
    { id: 'cat', kind: 'image', name: 'cat.png', width: 1024, height: 1024 });
  if (previous) {
    project.assets.push({ id: 'old-take', kind: 'video', name: 'Old take', generation: { request: previous, effectiveParameters } });
    api.S.regenerateAsset = 'old-take';
  }
  // The form renders where the page renders it, so a pick (which redraws the inspector) is seen as the page shows it.
  api.ui.inspector = document.createElement('aside'); api.S.tab = 'generate'; api.renderInspector();
  const view = () => api.ui.inspector.children[1] || api.ui.inspector;
  const labels = () => view().querySelectorAll('label');
  const field = label => labels().find(node => node.textContent === label)?.children[0];
  const audio = () => labels().find(node => node.children[1]?.textContent === 'Generate audio')?.children[0];
  const named = text => view().querySelectorAll('button').find(b => b.textContent === text || b.attributes['aria-label'] === text);
  const generate = () => named('Generate take');
  const reset = () => named('Reset to model defaults');
  const slot = name => view().querySelectorAll('.ts-slot').find(s => s.dataset.slot === name);
  const slotButton = (name, kind) => slot(name)?.querySelectorAll('button').find(b => b.dataset.focusKey === 'slot-' + name + '-' + kind);
  // Choose a library photo for a slot the way a person does: open the slot's picker, then tap the photo's tile.
  const pick = async (name, assetId) => {
    await (slotButton(name, 'choose') || slotButton(name, 'change')).click();
    const tile = view().querySelectorAll('.ts-picker-tile').find(t => t.dataset.assetId === assetId);
    assert.ok(tile, 'picker tile for ' + assetId); await tile.click();
  };
  const clear = async name => { await slot(name).querySelectorAll('button').find(b => b.textContent === 'Remove').click(); };
  const use = value => { const radio = view().querySelectorAll('input').find(i => i.type === 'radio' && i.value === value); assert.ok(radio, 'use ' + value); radio.checked = true; radio.dispatch('change'); };
  const radios = () => view().querySelectorAll('input').filter(i => i.type === 'radio').map(i => ({ value: i.value, disabled: !!i.disabled }));
  const hints = () => view().querySelectorAll('p').map(node => node.textContent);
  const text = node => [node.textContent, ...node.children.flatMap(text)].filter(Boolean);
  const submit = async () => { handlers.push(response({ id: 'job' }), response([])); await generate().click(); return h.calls.find(c => c.body && c.body.action === 'jobs/generate')?.body.payload; };
  const describe = value => { field('Describe this shot').value = value; field('Describe this shot').dispatch('input'); };
  return { ...h, get body() { return view(); }, view, field, audio, named, generate, reset, slot, slotButton, pick, clear, use, radios, hints, text, submit, describe };
}

test('a new Studio take submits the host quality defaults without reducing them to timeline or preview size', async () => {
  const { field, audio, pick, describe, submit } = generationHarness();
  assert.equal(Number(field('Width').value), 1024); assert.equal(Number(field('Height').value), 1024);
  assert.equal(Number(field('Length').value), 145); assert.equal(audio().checked, true);
  describe('保持人物全身入镜，让角色移动双手。'); await pick('first', 'banner');
  field('Seed (−1 = random)').value = '42';
  const request = await submit();
  assert.equal(request.width, 1024); assert.equal(request.height, 1024); assert.equal(request.frameCount, 145);
  assert.equal(request.fps, 24); assert.equal(request.steps, 8); assert.equal(request.cfgScale, 1); assert.equal(request.generateAudio, true);
  assert.equal(request.prompt, '保持人物全身入镜，让角色移动双手。'); assert.equal(request.conditioningAssetId, 'banner'); assert.equal(request.seed, 42);
  assert.equal(request.mode, 'i2v'); assert.equal(request.endConditioningAssetId, null);
});

test('the description comes first and the photos are optional, with a line that says what the take will do', async () => {
  const { view, hints, slot, pick, clear } = generationHarness();
  const order = view().children.map(n => n.className || n.tagName);
  assert.ok(order.indexOf('ts-field') < order.indexOf('ts-gen-use'), 'Describe the shot sits above the photos: ' + order.join(','));
  assert.ok(hints().includes('No photos: the take is made from your description alone.'));
  assert.ok(view().querySelectorAll('h4').some(n => n.textContent === 'Start or end on a photo (optional)'));
  assert.equal(slot('first').dataset.state, 'empty'); assert.equal(slot('last').dataset.state, 'empty');
  await pick('last', 'cat'); assert.ok(hints().includes('The take ends on the last frame.'));
  await pick('first', 'banner'); assert.ok(hints().includes('The take goes from the first frame to the last frame.'));
  await clear('last'); assert.ok(hints().includes('The take starts on the first frame and brings it to life.'));
});

test('Studio uses advertised defaults for other host geometry and frame rate', () => {
  const { field, audio, reset } = generationHarness(null, { defaultWidth: 1280, defaultHeight: 768, defaultFps: 30, defaultGenerateAudio: false });
  assert.equal(Number(field('Width').value), 1280); assert.equal(Number(field('Height').value), 768);
  assert.equal(Number(field('Frames per second').value), 30); assert.equal(audio().checked, false);
  assert.ok(reset(), 'One reset button replaces "Refresh model capabilities" and "Use model defaults".');
});

test('explicit Studio size and audio choices remain in the submitted take', async () => {
  const { field, audio, describe, submit } = generationHarness();
  describe('A dancer raises a hand.'); field('Width').value = '768'; field('Height').value = '512'; audio().checked = false;
  const request = await submit();
  assert.equal(request.width, 768); assert.equal(request.height, 512); assert.equal(request.generateAudio, false);
});

test('an old take keeps its settings until Reset re-reads the model and applies its current defaults', async () => {
  const previous = { modelId: 'ltx-test', prompt: 'A dancer raises a hand.', width: 512, height: 512, frameCount: 81,
    fps: 24, steps: 8, cfgScale: 1, generateAudio: false, seed: 42, conditioningAssetId: 'banner',
    temporalConditions: [{ assetId: 'banner', frameIndex: 80, strength: .75 }], shotId: 'same-shot' };
  const { api, field, audio, reset, calls, handlers, submit } = generationHarness(previous);
  assert.equal(Number(field('Width').value), 512); assert.equal(Number(field('Length').value), 81); assert.equal(audio().checked, false);
  // The host now advertises a longer default: Reset applies the fresh one, never the one the form was drawn with.
  handlers.push(response({ models: [{ ...api.S.capabilities.models[0], defaultFrameCount: 153 }] }));
  await reset().click();
  assert.equal(calls[0].body.action, 'capabilities');
  assert.equal(Number(field('Width').value), 1024); assert.equal(Number(field('Height').value), 1024);
  assert.equal(Number(field('Length').value), 153); assert.equal(audio().checked, true);
  const request = await submit();
  assert.equal(request.width, 1024); assert.equal(request.height, 1024); assert.equal(request.frameCount, 153);
  assert.equal(request.prompt, previous.prompt); assert.equal(request.seed, 42); assert.equal(request.shotId, 'same-shot');
  assert.equal(request.conditioningAssetId, 'banner'); assert.deepEqual(request.temporalConditions, previous.temporalConditions);
  assert.equal(request.mode, 'keyframes', 'A take saved before explicit modes reopens in the mode its inputs ran as.');
  assert.equal(previous.width, 512, 'The saved provenance must remain unchanged.');
});

test('the other ways to use photos list only what the family offers and say why one is unavailable', () => {
  const { radios, body, text } = generationHarness(null, { modes: ['t2v', 'i2v', 'fl2v'], temporalGuidance: false,
    unavailableModes: { ref: 'MiniMax-H3 identity references need the Ref2VA checkpoint.' } });
  assert.deepEqual(radios(), [{ value: 'frames', disabled: false }, { value: 'ref', disabled: true }]);
  const blocked = body.querySelectorAll('label').find(l => l.children[0]?.value === 'ref');
  assert.ok(text(blocked).includes('MiniMax-H3 identity references need the Ref2VA checkpoint.'), 'The disabled use shows its reason.');
  const only = generationHarness(null, { modes: ['t2v', 'i2v'], lastFrameConditioning: false, temporalGuidance: false, unavailableModes: {} });
  assert.deepEqual(only.radios(), [], 'A family with one use shows no chooser.');
});

test('a take with no photos is text-to-video, even after photos were chosen and removed', async () => {
  const { pick, clear, describe, submit } = generationHarness();
  describe('A lighthouse at dusk.');
  await pick('first', 'banner'); await pick('last', 'cat'); await clear('first'); await clear('last');
  const request = await submit();
  assert.equal(request.mode, 't2v'); assert.equal(request.conditioningAssetId, null); assert.equal(request.endConditioningAssetId, null);
  assert.deepEqual(request.temporalConditions, []); assert.deepEqual(request.referenceAssetIds, []);
});

test('first and last frames pin both photos, and Swap exchanges them', async () => {
  const { pick, named, describe, submit } = generationHarness();
  describe('The cat walks across the banner.');
  await pick('first', 'banner'); await pick('last', 'cat');
  await named('⇄ Swap first and last frames').click();
  const request = await submit();
  assert.equal(request.mode, 'fl2v'); assert.equal(request.conditioningAssetId, 'cat'); assert.equal(request.endConditioningAssetId, 'banner');
});

test('a last frame alone is fl2v with no first frame: the take only ends on the photo', async () => {
  const { pick, describe, submit, text, view } = generationHarness();
  describe('The camera drifts down to the cat.'); await pick('last', 'cat');
  assert.ok(text(view()).includes('Ends on cat.png · 6.0 s (145 frames) · 1024 × 1024 · with sound'), text(view()).join(' | '));
  const request = await submit();
  assert.equal(request.mode, 'fl2v'); assert.equal(request.conditioningAssetId, null); assert.equal(request.endConditioningAssetId, 'cat');
});

test('a shot is refused before submission when it lacks its description, keyframes or the photo a model needs', async () => {
  for (const [overrides, prepare, message] of [
    [{}, async () => {}, 'Describe the shot before generating.'],
    [{}, async g => { g.describe('A shot.'); g.use('keyframes'); }, 'Add at least one timed image keyframe.'],
    [{ modes: ['i2v'], textToVideo: false, lastFrameConditioning: false, temporalGuidance: false, unavailableModes: { t2v: 'Always animates a first frame.', fl2v: 'Wan takes at most a first-frame image.' } },
      async g => g.describe('A shot.'), 'Choose the photo this take starts on: this model always starts on one.']]) {
    const g = generationHarness(null, overrides); await prepare(g);
    if (g.body.querySelectorAll('.ts-picker').length) await g.named('Cancel').click();
    await g.generate().click();
    assert.equal(g.calls.filter(c => c.body?.action === 'jobs/generate').length, 0, message); assert.equal(g.api.ui.status.textContent, message);
    assert.ok(g.text(g.body).includes(message), 'The refusal shows beside Generate: ' + message);
  }
});

test('a slot the model cannot use stays visible, says why, and still shows a photo set from the library', async () => {
  const wan = { family: 'wan', modes: ['t2v', 'i2v'], lastFrameConditioning: false, temporalGuidance: false,
    unavailableModes: { fl2v: 'Wan takes at most a first-frame image; it cannot pin a last frame, alone or with a first.' } };
  const g = generationHarness(null, wan);
  assert.equal(g.slot('last').dataset.state, 'disabled');
  assert.ok(g.text(g.slot('last')).includes(wan.unavailableModes.fl2v));
  // Set from the library while Wan is loaded: kept, shown with Remove, and the shot is refused with Wan's reason.
  g.api.S.shotForms.get(g.project.id).last = 'cat'; g.body.replaceChildren(); g.api.generationForm(g.body);
  assert.ok(g.text(g.slot('last')).includes('cat.png'));
  g.describe('A shot.'); await g.generate().click();
  assert.equal(g.api.ui.status.textContent, wan.unavailableModes.fl2v);
});

const H3 = { modelId: 'h3', displayName: 'minimax_h3_fl2va_pruned-Q4_K.gguf', family: 'minimax-h3', imageConditioning: true, lastFrameConditioning: true, temporalGuidance: false,
  spatialMultiple: 32, temporalMultiple: 17, temporalOffset: 5, maxDimension: 2048, defaultWidth: 640, defaultHeight: 384,
  defaultFrameCount: 22, fixedFps: 24, defaultSteps: 20, fixedSteps: null, defaultCfgScale: 1, fixedCfgScale: 1,
  modes: ['t2v', 'i2v', 'fl2v'], unavailableModes: { ref: 'MiniMax-H3 identity references need the Ref2VA checkpoint.' },
  imageCanvasModes: ['i2v', 'fl2v'], imageCanvasShortEdge: 768, imageCanvasMaxPixels: 768 * 1344, imageCanvasMaxAspect: 4 };

test('MiniMax-H3 names its family, not its file, and lists what it can do', () => {
  const { body, text } = generationHarness(null, H3);
  assert.ok(text(body).includes('Video model: MiniMax-H3'), text(body).join(' | '));
  assert.ok(text(body).includes('Makes takes from a description, and can start and/or end on a photo, make sound.'), text(body).join(' | '));
  assert.ok(text(body).includes('minimax_h3_fl2va_pruned-Q4_K.gguf') === false && text(body).includes('h3'), 'The checkpoint id sits on a muted line of its own.');
});

test('MiniMax-H3 sizes the canvas from the first frame by default and leaves the exact size to the host', async () => {
  const { field, pick, hints, describe, submit } = generationHarness(null, H3);
  assert.equal(field('Size'), undefined, 'Text-to-video has no photo to size from.');
  assert.equal(Number(field('Width').value), 640); assert.equal(field('Width').disabled, false);
  await pick('first', 'banner'); await pick('last', 'cat');
  assert.equal(field('Size').value, 'image'); assert.equal(field('Width').disabled, true); assert.equal(field('Height').disabled, true);
  // A 1536x1024 still at a 768-pixel short edge is shown before submission.
  assert.equal(Number(field('Width').value), 1152); assert.equal(Number(field('Height').value), 768);
  assert.ok(hints().some(t => t.startsWith('1152 × 768, matched to banner_1.png') && t.includes('3.6× the work per frame of the 640 × 384 default')), hints().join(' | '));
  describe('The banner turns into the photo.');
  const request = await submit();
  assert.equal(request.width, 0); assert.equal(request.height, 0, 'The host reads the image upright and records the size it used.');
  assert.equal(request.mode, 'fl2v'); assert.equal(request.conditioningAssetId, 'banner'); assert.equal(request.endConditioningAssetId, 'cat');
});

test('MiniMax-H3 with only a last frame sizes the canvas from that photo', async () => {
  const { field, pick, project, hints, describe, submit } = generationHarness(null, H3);
  project.assets.push({ id: 'portrait', kind: 'image', name: 'portrait.jpg', width: 1080, height: 1620 });
  await pick('last', 'portrait');
  assert.equal(Number(field('Width').value), 768); assert.equal(Number(field('Height').value), 1152);
  assert.ok(hints().some(t => t.startsWith('768 × 1152, matched to portrait.jpg')), hints().join(' | '));
  describe('A walk that ends on the portrait.');
  const request = await submit();
  assert.equal(request.mode, 'fl2v'); assert.equal(request.conditioningAssetId, null); assert.equal(request.endConditioningAssetId, 'portrait');
  assert.equal(request.width, 0); assert.equal(request.height, 0);
});

test('a custom MiniMax-H3 keyframe size is sent as typed and reopens as custom', async () => {
  const { field, pick, describe, submit } = generationHarness(null, H3);
  describe('The picture comes alive.'); await pick('first', 'banner');
  field('Size').value = 'custom'; field('Size').dispatch('change');
  assert.equal(field('Width').disabled, false); assert.equal(Number(field('Width').value), 640);
  field('Width').value = '960'; field('Width').dispatch('input'); field('Height').value = '640'; field('Height').dispatch('input');
  const request = await submit();
  assert.equal(request.width, 960); assert.equal(request.height, 640);
  const take = size => ({ modelId: 'h3', prompt: 'p', mode: 'i2v', conditioningAssetId: 'banner', width: size[0], height: size[1], frameCount: 22, fps: 24, steps: 20, cfgScale: 1, seed: 7, temporalConditions: [], referenceAssetIds: [] });
  assert.equal(generationHarness(take([960, 640]), H3).field('Size').value, 'custom');
  const recipe = generationHarness(take([1152, 768]), H3);
  assert.equal(recipe.field('Size').value, 'image'); assert.equal(recipe.field('Width').disabled, true);
});

test('the shown MiniMax-H3 size follows the chosen photo and returns to the typed size without one', async () => {
  const { field, pick, clear, project } = generationHarness(null, H3);
  project.assets.push({ id: 'pano', kind: 'image', name: 'pano.jpg', width: 5000, height: 1000 });
  await pick('first', 'pano');
  assert.equal(Number(field('Width').value), 2016); assert.equal(Number(field('Height').value), 512, 'Beyond 4:1 the recipe clamps and the host crops.');
  await clear('first');
  assert.equal(Number(field('Width').value), 640); assert.equal(Number(field('Height').value), 384);
});

test('a MiniMax-H3 take shorter than its trained 4–15 s says what tends to happen, most of all when it ends on a photo', async () => {
  const g = generationHarness(null, { ...H3, recommendedMinSeconds: 4, recommendedMaxSeconds: 15 });
  assert.ok(g.hints().includes('MiniMax-H3 is trained on 4–15 s clips; a 0.9 s take is a quick preview.'), g.hints().join(' | '));
  await g.pick('last', 'cat');
  assert.ok(g.hints().includes('MiniMax-H3 is trained on 4–15 s clips. At 0.9 s a take has little time to reach its last photo and often cuts or dissolves to it.'), g.hints().join(' | '));
  assert.ok(g.hints().includes('Describe how the scene arrives at the last photo. A photo unlike the scene is usually reached by a cut or dissolve rather than a move.'));
  g.field('Length').value = '107'; g.field('Length').dispatch('change');
  assert.ok(!g.hints().some(t => t.includes('trained on')), 'A 4.5 s take is inside the range.');
});

test('MiniMax-H3 lengths are offered in seconds on its 17n+5 grid', () => {
  const { field } = generationHarness(null, H3);
  const options = field('Length').children.slice(0, 3).map(o => [o.value, o.textContent]);
  assert.deepEqual(options, [['22', '0.9 s · 22 frames'], ['39', '1.6 s · 39 frames'], ['56', '2.3 s · 56 frames']]);
});

// The same FL2VA checkpoint with the LightX2V Turbo 8-step plug-in: the host advertises the recipe and fixes its steps.
const H3_TURBO = { ...H3, displayName: 'h3 · LightX2V Turbo 8-step (FL2V v1.0, 768p)', defaultSteps: 8, fixedSteps: 8,
  samplingRecipe: 'minimax-h3-lora:minimax-h3-fl2v-turbo-8step', samplingRecipeName: 'LightX2V Turbo 8-step (FL2V v1.0, 768p)' };
const H3_BASE = { ...H3, samplingRecipe: 'native' };
const BASE_RECIPE = 'MiniMax-H3: 20 steps, sd.cpp discrete grid, video shift 12 / audio 3, euler, cfg 1';
const h3Take = (steps, samplingRecipe) => ({ modelId: 'h3', prompt: 'A kite rises over the dunes.', mode: 't2v', width: 640, height: 384, frameCount: 22,
  fps: 24, steps, cfgScale: 1, seed: 7, temporalConditions: [], referenceAssetIds: [], shotId: 'kite', ...(samplingRecipe ? { samplingRecipe } : {}) });
const recipeChangedNote = (previous, steps) => 'This take was sampled with ' + previous + '; the loaded model samples differently, so its steps start from the model default (' + steps + ').';

test('a MiniMax-H3 Turbo model names its fixed recipe and plans the shot for it', async () => {
  const { field, hints, describe, submit } = generationHarness(null, H3_TURBO);
  assert.ok(hints().includes('LightX2V Turbo 8-step (FL2V v1.0, 768p) fixes the sampling: exactly 8 steps on its own schedule.'), hints().join(' | '));
  assert.ok(!hints().some(t => t.startsWith('This take was sampled')), 'A new shot has no earlier recipe.');
  assert.equal(Number(field('Sampling steps').value), 8); assert.equal(field('Sampling steps').disabled, true);
  describe('A kite rises over the dunes.');
  const request = await submit();
  assert.equal(request.steps, 8); assert.equal(request.samplingRecipe, H3_TURBO.samplingRecipe);
});

test('a base MiniMax-H3 take reopened under Turbo keeps its shot, takes the recipe steps and says why', async () => {
  const { field, hints, submit } = generationHarness(h3Take(20, 'native'), H3_TURBO, { recipe: BASE_RECIPE, samplingRecipe: 'native' });
  assert.ok(hints().includes(recipeChangedNote(BASE_RECIPE, 8)), hints().join(' | '));
  assert.equal(Number(field('Sampling steps').value), 8);
  const request = await submit();
  assert.equal(request.steps, 8); assert.equal(request.samplingRecipe, H3_TURBO.samplingRecipe, 'The loaded recipe, never the reused take’s.');
  assert.equal(request.prompt, 'A kite rises over the dunes.'); assert.equal(request.seed, 7); assert.equal(request.shotId, 'kite');
});

test('a Turbo take reopened on the base checkpoint drops its 8 steps for the base default', async () => {
  const sampled = { recipe: 'LightX2V Turbo 8-step (FL2V v1.0, 768p): 8 steps, endpoint-inclusive linspace, video shift 6 / audio 3, euler, cfg 1',
    samplingRecipe: H3_TURBO.samplingRecipe, samplingRecipeName: H3_TURBO.samplingRecipeName };
  const { field, hints, submit } = generationHarness(h3Take(8, H3_TURBO.samplingRecipe), H3_BASE, sampled);
  assert.ok(hints().includes(recipeChangedNote(H3_TURBO.samplingRecipeName, 20)), hints().join(' | '));
  assert.ok(!hints().some(t => t.includes('fixes the sampling')), 'The base checkpoint fixes no recipe.');
  assert.equal(Number(field('Sampling steps').value), 20); assert.ok(!field('Sampling steps').disabled, 'The base step count is editable.');
  const request = await submit();
  assert.equal(request.steps, 20); assert.equal(request.samplingRecipe, 'native');
});

test('a take reopened under its own recipe, or saved before recipes on the base checkpoint, keeps its steps without a note', () => {
  const turbo = generationHarness(h3Take(8, H3_TURBO.samplingRecipe), H3_TURBO);
  assert.ok(!turbo.hints().some(t => t.startsWith('This take was sampled')), turbo.hints().join(' | '));
  assert.equal(Number(turbo.field('Sampling steps').value), 8);
  for (const capability of [H3, H3_BASE]) {
    const legacy = generationHarness(h3Take(30), capability);
    assert.ok(!legacy.hints().some(t => t.startsWith('This take was sampled')), legacy.hints().join(' | '));
    assert.equal(Number(legacy.field('Sampling steps').value), 30, 'A take saved before recipes was sampled natively.');
  }
  const reopened = generationHarness(h3Take(20), H3_TURBO, { recipe: 'native model sampling' });
  assert.ok(reopened.hints().includes(recipeChangedNote('native model sampling', 8)), reopened.hints().join(' | '));
});

test('a kept form whose model gained a speed plug-in takes the recipe’s fixed steps and says so', async () => {
  const g = generationHarness(null, H3); g.describe('A kite rises over the dunes.');
  g.field('Sampling steps').value = '30'; g.field('Sampling steps').dispatch('input');
  g.api.S.capabilities = { models: [{ ...g.api.S.capabilities.models[0], ...H3_TURBO }] }; g.body.replaceChildren(); g.api.generationForm(g.body);
  assert.equal(Number(g.field('Sampling steps').value), 8);
  assert.ok(g.hints().includes('LightX2V Turbo 8-step (FL2V v1.0, 768p) is now loaded, so the steps follow it (8).'), g.hints().join(' | '));
  assert.equal(g.field('Describe this shot').value, 'A kite rises over the dunes.');
  const request = await g.submit(); assert.equal(request.steps, 8); assert.equal(request.samplingRecipe, H3_TURBO.samplingRecipe);
});

const REF2VA = { modes: ['t2v', 'ref'], imageConditioning: false, lastFrameConditioning: false, temporalGuidance: false, identityReferences: true, maxReferences: 9,
  referenceMinFrames: 121, defaultFrameCount: 17, unavailableModes: { i2v: 'Needs the FL2VA checkpoint.', fl2v: 'Needs the FL2VA checkpoint.' } };

test('a reference checkpoint opens on reference photos and sends them in the order they were chosen, never as frames', async () => {
  const { api, view, use, describe, field, hints, submit } = generationHarness(null, REF2VA);
  const body = { querySelectorAll: selector => view().querySelectorAll(selector) };
  assert.equal(Number(field('Length').value), 121, 'Reference use raises the length to the model minimum.');
  assert.ok(hints().includes('Length raised to 5.0 s: this model needs at least that for reference photos.'), hints().join(' | '));
  const tile = id => body.querySelectorAll('.ts-picker-tile').find(t => t.dataset.assetId === id);
  await tile('cat').click(); await tile('banner').click();
  assert.equal(tile('cat').attributes['aria-pressed'], 'true');
  assert.ok(hints().includes('Order sent to the model: 1. cat.png · 2. banner_1.png'));
  describe('The person from Picture 1 holds the cat from Picture 2 in a garden.');
  const request = await submit();
  assert.equal(request.mode, 'ref'); assert.deepEqual(request.referenceAssetIds, ['cat', 'banner']);
  assert.equal(request.conditioningAssetId, null); assert.equal(request.endConditioningAssetId, null); assert.equal(request.frameCount, 121);
  api.S.tab = 'generate'; api.renderInspector(); use('frames'); assert.ok(body.querySelectorAll('.ts-slot').every(s => s.dataset.state === 'disabled'), 'Ref2VA pins no frames and says why.');
});

test('reference use requires at least one reference photo', async () => {
  const { api, describe, generate, calls } = generationHarness(null, REF2VA);
  describe('A new scene.'); await generate().click();
  assert.equal(calls.length, 0); assert.equal(api.ui.status.textContent, 'Select at least one reference image.');
});

test('a reference take reopens on reference photos with its references in their saved order', async () => {
  const previous = { modelId: 'ltx-test', prompt: 'Same person, new city.', width: 1024, height: 1024, frameCount: 121, fps: 24, steps: 8, cfgScale: 1,
    generateAudio: true, seed: 7, referenceAssetIds: ['cat', 'banner'], temporalConditions: [], shotId: 'ref-shot' };
  const { body, submit } = generationHarness(previous, { modes: ['t2v', 'i2v', 'fl2v', 'keyframes', 'ref'], identityReferences: true, maxReferences: 16, referenceMinFrames: 121, unavailableModes: {} });
  assert.equal(body.querySelectorAll('input').find(i => i.type === 'radio' && i.checked).value, 'ref');
  const request = await submit();
  assert.deepEqual(request.referenceAssetIds, ['cat', 'banner']); assert.equal(request.mode, 'ref');
});

test('the form survives a re-render: description, photos and settings stay', async () => {
  const g = generationHarness();
  g.describe('Waves at sunrise.'); await g.pick('last', 'cat'); g.field('Seed (−1 = random)').value = '9'; g.field('Seed (−1 = random)').dispatch('input');
  const again = g.document.createElement('div'); g.api.generationForm(again);
  const field = label => again.querySelectorAll('label').find(n => n.textContent === label)?.children[0];
  assert.equal(field('Describe this shot').value, 'Waves at sunrise.'); assert.equal(Number(field('Seed (−1 = random)').value), 9);
  assert.equal(again.querySelectorAll('.ts-slot').find(s => s.dataset.slot === 'last').dataset.state, 'filled');
});

test('Escape closes the open picker first, never Studio from inside a field, and the global keys leave panel buttons alone', async () => {
  const g = generationHarness(); g.api.S.root = g.document.createElement('section'); g.api.S.root.hidden = false;
  await g.slotButton('first', 'choose').click();
  assert.equal(g.body.querySelectorAll('.ts-picker').length, 1); assert.equal(g.api.S.layers.length, 1);
  const tile = g.body.querySelectorAll('.ts-picker-tile').find(t => t.dataset.assetId === 'cat');
  // Space and Delete on a picker tile belong to the browser (activate the tile), not to the timeline's shortcuts.
  g.api.ui.video = { src: 'x', paused: true, play() { throw new Error('Space must not reach the monitor.'); } };
  for (const key of [{ key: ' ', code: 'Space' }, { key: 'Delete' }]) {
    let prevented = false; g.api.keydown({ ...key, target: tile, preventDefault() { prevented = true; } }); assert.equal(prevented, false, key.key);
  }
  g.api.keydown({ key: 'Escape', target: tile, preventDefault() {} });
  assert.equal(g.body.querySelectorAll('.ts-picker').length, 0); assert.equal(g.api.S.layers.length, 0); assert.equal(g.api.S.root.hidden, false);
  assert.equal(g.document.activeElement, g.slotButton('first', 'choose'), 'Focus returns to the slot that opened the picker.');
  g.api.keydown({ key: 'Escape', target: g.field('Describe this shot'), preventDefault() {} });
  assert.equal(g.api.S.root.hidden, false, 'Escape inside the description does not throw Studio away.');
});

test('a refusal about the model re-reads its capabilities and explains itself in plain words, with the host text under Details', async () => {
  const g = generationHarness(null, H3); g.describe('A kite.');
  const message = 'This shot was planned for the model’s own schedule but the loaded model now samples with LightX2V Turbo. Refresh the model settings and review the steps.';
  // The host has the Turbo plug-in loaded meanwhile: the re-read differs, so the inspector is redrawn, and the refusal stays.
  g.handlers.push(response({ code: 'sampling_recipe_conflict', message }, 409), response({ models: [{ ...g.api.S.capabilities.models[0], ...H3_TURBO }] }));
  g.api.S.project.id = g.project.id; await g.generate().click(); await new Promise(resolve => setTimeout(resolve, 0));
  const lines = g.text(g.api.ui.inspector);
  assert.ok(lines.includes('The loaded model now samples differently. Studio re-read its settings: review the steps and generate again.'), lines.join(' | '));
  assert.ok(lines.includes('Details') && lines.includes(message));
  assert.ok(g.calls.some(c => c.body?.action === 'capabilities'), 'The page asks the host for the current model settings.');
  assert.notEqual(g.api.ui.status.textContent, 'The project changed. Your edit was not applied. Review the latest revision and try again.');
});

test('a library photo starts or ends a new take in one tap whenever the host can generate', async () => {
  const g = generationHarness(); g.api.ui.library = g.document.createElement('aside'); g.api.renderLibrary();
  const card = g.api.ui.library.querySelectorAll('article').find(c => c.children.some(n => n.textContent === 'cat.png'));
  const last = card.querySelectorAll('button').find(b => b.attributes['aria-label'] === 'As last frame: cat.png');
  assert.ok(last); await last.click();
  assert.equal(g.api.S.tab, 'generate'); assert.equal(g.api.S.shotForms.get(g.project.id).last, 'cat');
  const lastSlot = g.api.ui.inspector.querySelectorAll('.ts-slot').find(s => s.dataset.slot === 'last');
  assert.equal(lastSlot.dataset.state, 'filled');
  assert.equal(g.document.activeElement?.dataset.focusKey, 'slot-last-change', 'Focus lands on the filled slot.');
  g.api.S.capabilities.generationUnavailableReason = 'This host does not permit the required media subprocesses.'; g.api.renderLibrary();
  assert.ok(!g.api.ui.library.querySelectorAll('button').some(b => /as the (first|last) frame/.test(b.attributes['aria-label'] || '')), 'A host that cannot generate offers no frame actions.');
});

test('a host that cannot generate says so once, with no model how-to', () => {
  const g = generationHarness(); g.api.S.capabilities = { models: [], generationUnavailableReason: 'This host does not permit the required media subprocesses; project review and editing remain available.' };
  g.body.replaceChildren(); g.api.generationForm(g.body);
  const lines = g.text(g.body);
  assert.ok(lines.includes('Video generation isn’t available on this host.'));
  assert.ok(!lines.some(l => l.includes('choose a video model')), lines.join(' | '));
  const none = generationHarness(); none.api.S.capabilities = { models: [] }; none.body.replaceChildren(); none.api.generationForm(none.body);
  assert.ok(none.text(none.body).includes('No video model is loaded.'));
  assert.ok(none.body.querySelectorAll('button').some(b => b.textContent === 'Check again'));
});

test('asset library renders a named Delete asset action even for missing sources', () => {
  const { api, project } = deletionHarness(); project.assets[0].missing = true;
  api.renderLibrary();
  const action = api.ui.library.querySelectorAll('button').find(b => b.textContent === 'Delete asset');
  assert.ok(action); assert.equal(action.attributes['aria-label'], 'Delete asset: <image & video>');
  assert.equal(action.disabled, undefined);
});

test('deletion confirmation defaults to Cancel, traps focus and cancels with Escape without host mutation', async () => {
  const { api, calls, document, trigger, dialog, dialogButtons } = deletionHarness(2);
  const pending = api.deleteAsset('asset-a'), backdrop = dialog(), [cancel, confirm] = dialogButtons();
  assert.equal(document.activeElement, cancel); assert.equal(api.ui.library.inert, true);
  const content = backdrop.children[0]; assert.equal(content.attributes.role, 'alertdialog'); assert.equal(content.attributes['aria-modal'], 'true');
  assert.match(content.children[1].textContent, /2 timeline clip\(s\)/); assert.match(content.children[2].textContent, /does not free disk space/);
  assert.equal(content.children[0].textContent, 'Delete “<image & video>”?');
  const backwards = backdrop.dispatch('keydown', { key: 'Tab', shiftKey: true }); assert.equal(backwards.event.defaultPrevented, true); assert.equal(document.activeElement, confirm);
  backdrop.dispatch('keydown', { key: 'Tab', shiftKey: false }); assert.equal(document.activeElement, cancel);
  const escaped = backdrop.dispatch('keydown', { key: 'Escape' }); await pending;
  assert.equal(escaped.event.propagationStopped, true); assert.equal(calls.length, 0); assert.equal(api.S.deleteDialog, null); assert.equal(api.ui.library.inert, false); assert.equal(document.activeElement, trigger); assert.equal(dialog(), null);
});

test('Cancel button preserves an unused asset and its project revision', async () => {
  const { api, calls, project, dialogButtons, dialog } = deletionHarness();
  const pending = api.deleteAsset('asset-a'); assert.match(dialog().children[0].children[1].textContent, /not used on the timeline/);
  await dialogButtons()[0].click(); await pending;
  assert.equal(calls.length, 0); assert.equal(project.assets.length, 1); assert.equal(project.revision, 4);
});

for (const usageCount of [0, 2]) test('confirmed asset deletion sends reviewed revision and explicit clip consent: ' + usageCount + ' uses', async () => {
  const { api, calls, handlers, project, dialogButtons } = deletionHarness(usageCount);
  handlers.push(response({ ...project, revision: 5, assets: [], clips: [] }));
  const pending = api.deleteAsset('asset-a'); await dialogButtons()[1].click(); await pending;
  assert.equal(calls.length, 1); assert.equal(calls[0].body.action, 'commands/apply');
  assert.equal(calls[0].body.payload.projectId, project.id); assert.equal(calls[0].body.payload.expectedRevision, 4);
  assert.deepEqual(calls[0].body.payload.commands, [{ type: 'deleteAsset', assetId: 'asset-a', removeTimelineClips: usageCount > 0 }]);
  assert.match(api.ui.status.textContent, /Use Undo/); assert.equal(api.S.project.assets.length, 0);
});

for (const change of ['revision', 'project', 'epoch', 'busy']) test('deletion refuses a changed confirmation context: ' + change, async () => {
  const { api, calls, project, dialogButtons } = deletionHarness(1);
  const pending = api.deleteAsset('asset-a');
  if (change === 'revision') { project.revision++; project.clips.push({ id: 'new-use', assetId: 'asset-a' }); }
  if (change === 'project') api.S.project = { ...project, id: 'b'.repeat(32) };
  if (change === 'epoch') api.S.epoch++;
  if (change === 'busy') api.S.busy = true;
  const rejection = assert.rejects(pending, /Nothing was deleted/); await dialogButtons()[1].click(); await rejection;
  assert.equal(calls.length, 0); assert.equal(project.assets.length, 1);
});

test('server revision conflict preserves the asset and does not clear preview or selection', async () => {
  const { api, calls, handlers, project, dialogButtons } = deletionHarness(1);
  api.S.selected = 'clip-0'; api.S.regenerateAsset = 'asset-a'; api.S.sourcePreviewAssetId = 'asset-a'; api.S.sourcePreview = true;
  handlers.push(response({ code: 'revision_conflict', message: 'Project changed.' }, 409));
  const pending = api.deleteAsset('asset-a'), rejection = assert.rejects(pending, error => error.code === 'revision_conflict');
  await dialogButtons()[1].click(); await rejection;
  assert.equal(calls.length, 1); assert.equal(api.S.project, project); assert.equal(api.S.selected, 'clip-0'); assert.equal(api.S.sourcePreview, true); assert.equal(api.S.regenerateAsset, 'asset-a');
});

test('successful deletion clears removed clip, source preview, and regeneration selection', async () => {
  const { api, handlers, project, dialogButtons } = deletionHarness(1);
  api.S.selected = 'clip-0'; api.S.regenerateAsset = 'asset-a'; api.S.sourcePreviewAssetId = 'asset-a'; api.S.sourcePreview = true;
  let paused = false, cleared = false, loaded = false;
  api.ui.video = { pause() { paused = true; }, removeAttribute(name) { cleared = name === 'src'; }, load() { loaded = true; }, hidden: false };
  handlers.push(response({ ...project, revision: 5, assets: [], clips: [] }));
  const pending = api.deleteAsset('asset-a'); await dialogButtons()[1].click(); await pending;
  assert.equal(api.S.selected, null); assert.equal(api.S.regenerateAsset, null); assert.equal(api.S.sourcePreviewAssetId, null); assert.equal(api.S.sourcePreview, false);
  assert.equal(paused && cleared && loaded && api.ui.video.hidden, true);
});

test('every track header offers a named Remove track action that removes that track', async () => {
  const { api, calls, handlers, project, dialog, dialogButtons } = trackHarness(2);
  api.renderTimeline();
  const removes = api.ui.timeline.querySelectorAll('button').filter(b => b.textContent === '×');
  assert.deepEqual(removes.map(b => b.attributes['aria-label']), ['Remove track: Video 1', 'Remove track: <Music & FX>']);
  assert.ok(removes.every(b => b.title === 'Remove track' && b.className === 'ts-danger'));
  const clicked = removes[1].click();
  assert.equal(dialog().children[0].children[0].textContent, 'Remove track “<Music & FX>”?'); assert.match(dialog().children[0].children[1].textContent, /holds 2 clip\(s\)/);
  handlers.push(response({ ...project, revision: 5, tracks: project.tracks.slice(0, 1), clips: project.clips.slice(0, 1) }));
  await dialogButtons()[1].click(); await clicked;
  assert.equal(calls.length, 1); assert.deepEqual(calls[0].body.payload.commands, [{ type: 'removeTrack', trackId: 'track-a' }]);
});

test('removing an empty track still asks, says it has no clips, and points to Undo', async () => {
  const { api, calls, handlers, project, document, dialog, dialogButtons } = trackHarness();
  const pending = api.removeTrack('track-a'), warning = dialog().children[0].children[1];
  assert.equal(warning.textContent, 'This track has no clips.'); assert.equal(warning.className, '');
  assert.equal(calls.length, 0); // the double-click guard: nothing is sent before the dialog is answered
  handlers.push(response({ ...project, revision: 5, tracks: project.tracks.slice(0, 1) }));
  await dialogButtons()[1].click(); await pending;
  assert.equal(calls.length, 1); assert.equal(calls[0].body.action, 'commands/apply');
  assert.equal(calls[0].body.payload.projectId, project.id); assert.equal(calls[0].body.payload.expectedRevision, 4);
  assert.deepEqual(calls[0].body.payload.commands, [{ type: 'removeTrack', trackId: 'track-a' }]);
  assert.equal(api.S.project.tracks.length, 1); assert.match(api.ui.status.textContent, /Use Undo/); assert.equal(document.activeElement, api.ui.timeline);
});

test('track removal counts only that track, defaults to Cancel and sends nothing when cancelled', async () => {
  const { api, calls, document, trigger, dialog, dialogButtons } = trackHarness(2);
  const pending = api.removeTrack('track-a'), backdrop = dialog(), [cancel, confirm] = dialogButtons(), content = backdrop.children[0];
  assert.equal(content.attributes.role, 'alertdialog'); assert.match(content.children[1].textContent, /holds 2 clip\(s\)/); assert.equal(content.children[1].className, 'ts-delete-warning');
  assert.match(content.children[2].textContent, /Undo/);
  assert.equal(confirm.textContent, 'Remove track'); assert.equal(document.activeElement, cancel); assert.equal(api.ui.timeline.inert, true);
  await api.removeTrack('track-v'); assert.equal(api.S.root.querySelectorAll('.ts-delete-dialog').length, 1); // no second dialog or edit underneath
  backdrop.dispatch('keydown', { key: 'Escape' }); await pending;
  assert.equal(calls.length, 0); assert.equal(dialog(), null); assert.equal(api.ui.timeline.inert, false); assert.equal(document.activeElement, trigger);
  const again = api.removeTrack('track-v'); assert.match(dialog().children[0].children[1].textContent, /holds 1 clip\(s\)/);
  await dialogButtons()[0].click(); await again;
  assert.equal(calls.length, 0); assert.equal(api.S.project.tracks.length, 2);
});

test('a removal dialog whose trigger was re-rendered cancels from outside the box and returns focus to the timeline', async () => {
  const { api, calls, document, dialog } = trackHarness(1);
  const pending = api.removeTrack('track-a');
  api.ui.timeline.replaceChildren(); // a background refresh rebuilt the track headers under the dialog
  let prevented = false; api.keydown({ key: 'Escape', preventDefault() { prevented = true; } });
  assert.equal(dialog(), null); await pending; // checked first: a dialog left open would leave pending unresolved
  assert.equal(prevented, true); assert.equal(calls.length, 0);
  assert.equal(document.activeElement, api.ui.timeline); assert.equal(api.ui.timeline.tabIndex, -1);
});

test('confirmed track removal sends the reviewed revision and clears a selection on that track', async () => {
  const { api, calls, handlers, project, dialogButtons } = trackHarness(2);
  api.S.selected = 'clip-0';
  handlers.push(response({ ...project, revision: 5, tracks: project.tracks.slice(0, 1), clips: project.clips.slice(0, 1) }));
  const pending = api.removeTrack('track-a'); await dialogButtons()[1].click(); await pending;
  assert.equal(calls.length, 1); assert.equal(calls[0].body.payload.expectedRevision, 4);
  assert.deepEqual(calls[0].body.payload.commands, [{ type: 'removeTrack', trackId: 'track-a' }]);
  assert.equal(api.S.selected, null); assert.equal(api.S.project.revision, 5); assert.match(api.ui.status.textContent, /Use Undo/);
});

for (const change of ['revision', 'project', 'epoch', 'busy']) test('track removal refuses a changed confirmation context: ' + change, async () => {
  const { api, calls, project, document, dialogButtons } = trackHarness(1);
  const pending = api.removeTrack('track-a');
  if (change === 'revision') { project.revision++; project.clips.push({ id: 'new-clip', assetId: 'asset-a', trackId: 'track-a', startFrame: 50, durationFrames: 10 }); }
  if (change === 'project') api.S.project = { ...project, id: 'b'.repeat(32) };
  if (change === 'epoch') api.S.epoch++;
  if (change === 'busy') api.S.busy = true;
  const rejection = assert.rejects(pending, /Nothing was removed/); await dialogButtons()[1].click(); await rejection;
  assert.equal(calls.length, 0); assert.equal(project.tracks.length, 2); assert.equal(document.activeElement, api.ui.timeline);
});

test('only same-host managed media URLs can enter chat import', () => {
  const { api } = harness();
  assert.equal(api.localMediaUrl('/uploads/video.mp4'), '/uploads/video.mp4');
  assert.equal(api.localMediaUrl('http://127.0.0.1:5000/api/code/artifacts/run/video.mp4'), '/api/code/artifacts/run/video.mp4');
  for (const url of ['https://remote.example/video.mp4', '//remote.example/uploads/private.mp4', 'file:///etc/passwd', 'javascript:alert(1)', 'data:video/mp4;base64,AQ==', '/uploads/../private.mp4', '/private.mp4', 'http://localhost:5000/uploads/a.mp4']) assert.equal(api.localMediaUrl(url), null, url);
});

test('fractional project rate maps frames consistently and snapping excludes self', () => {
  const { api, project } = harness();
  assert.equal(api.frames(1001 / 30000), 1);
  assert.equal(api.seconds(30000), 1001);
  assert.equal(api.frames(1 / (30000 / 1001) / 2), 1);
  project.clips.push({ id: 'selected', startFrame: 45, durationFrames: 30 }, { id: 'other', startFrame: 100, durationFrames: 20 });
  api.S.zoom = 65; api.S.playhead = 0;
  assert.equal(api.snapped(99, 'selected'), 100);
  assert.equal(api.snapped(44, 'selected'), 44);
  api.S.snap = false; assert.equal(api.snapped(99, 'selected'), 99);
});

test('manual edits send revision and stable command IDs without optimistic state mutation', async () => {
  const { api, calls, handlers, project } = harness();
  handlers.push(response({ ...project, revision: 5 }));
  await api.change([{ type: 'moveClip', clipId: 'clip-a', startFrame: 30 }]);
  const call = calls[0]; assert.equal(call.url, '/api/studio'); assert.equal(call.options.headers['X-TensorSharp-Studio'], '1');
  assert.equal(call.options.credentials, 'same-origin'); assert.equal(call.body.action, 'commands/apply');
  assert.equal(call.body.payload.projectId, project.id); assert.equal(call.body.payload.expectedRevision, 4); assert.match(call.body.payload.commandId, /^[a-f\d]{32}$/);
  assert.equal(call.body.payload.commands[0].startFrame, 30); assert.equal(api.S.project.revision, 5); assert.equal(project.revision, 4); assert.equal(api.S.busy, false);
});

test('revision conflict preserves previous project and releases saving state', async () => {
  const { api, handlers, project } = harness();
  handlers.push(response({ code: 'revision_conflict', message: 'Project changed.' }, 409));
  await assert.rejects(api.change([{ type: 'removeClip', clipId: 'clip-a' }]), e => e.status === 409 && e.code === 'revision_conflict');
  assert.equal(api.S.project, project); assert.equal(api.S.project.revision, 4); assert.equal(api.S.busy, false);
});

test('undo and redo use the same revision-bound host protocol', async () => {
  const { api, handlers, calls, project } = harness();
  handlers.push(response({ ...project, revision: 5 })); await api.history('undo');
  handlers.push(response({ ...project, revision: 6 })); await api.history('redo');
  assert.equal(calls[0].body.action, 'commands/undo'); assert.equal(calls[0].body.payload.expectedRevision, 4);
  assert.equal(calls[1].body.action, 'commands/redo'); assert.equal(calls[1].body.payload.expectedRevision, 5);
});

test('late undo response cannot replace a newly selected project', async () => {
  const { api, handlers, project } = harness();
  const other = { ...project, id: 'b'.repeat(32), revision: 1 };
  handlers.push({ ok: true, text: async () => { api.S.project = other; return JSON.stringify({ ...project, revision: 5 }); } });
  await api.history('undo'); assert.equal(api.S.project, other); assert.equal(api.S.busy, false);
});

test('unavailable or unknown renderer is not advertised as export capability', () => {
  const { api } = harness(); assert.equal(api.renderAllowed(), false);
  api.S.capabilities = { renderAvailable: false }; assert.equal(api.renderAllowed(), false);
  api.S.capabilities = { renderAvailable: true }; assert.equal(api.renderAllowed(), true);
});

test('structured host errors remain readable', async () => {
  const { api } = harness(); await assert.rejects(api.response(response({ error: { message: 'No video model loaded.' } }, 400)), /No video model loaded/);
});

function jobsHarness(jobs) {
  const h = harness(), { api, document } = h;
  api.S.capabilities = { renderAvailable: true, models: [{ modelId: 'minimax-h3', displayName: 'MiniMax-H3' }] };
  api.S.jobs = jobs; const body = document.createElement('div'); api.jobsPanel(body);
  const text = node => [node.textContent, ...node.children.flatMap(text)].filter(Boolean);
  const cards = body.querySelectorAll('article').map(card => ({ card, lines: text(card) }));
  return { ...h, body, cards };
}
const summary = { modelId: 'minimax-h3', mode: 't2v', prompt: 'A lighthouse at dusk.', width: 640, height: 384, frameCount: 121, fps: 24, steps: 30, seed: 42, generateAudio: true };

test('a running generation shows its stage, steps, elapsed and remaining time and what it is making', () => {
  const { cards } = jobsHarness([{ id: 'j1', kind: 'generate', state: 'running', stage: 'denoise', detail: 'cond pass', step: 12, totalSteps: 30, progress: .42,
    createdAt: '2026-10-17T12:00:00Z', startedAt: '2026-10-17T12:00:05Z', summary,
    timing: { elapsedSeconds: 400, stageElapsedSeconds: 65, queuedSeconds: 5, remainingSeconds: 545, remainingScope: 'job', previousRunSeconds: 1384 } }]);
  const { card, lines } = cards[0];
  assert.equal(card.dataset.state, 'running');
  assert.ok(lines.includes('Video generation') && lines.includes('Running') && lines.includes('A lighthouse at dusk.'));
  assert.ok(lines.includes('MiniMax-H3 · From your description · 640×384 · 121 frames (5.0 s) at 24 fps · 30 steps · seed 42 · with audio'), lines.join('\n'));
  assert.ok(lines.includes('Denoising · 12 of 30 steps done · 18 left'), lines.join('\n'));
  assert.ok(lines.some(l => /^Started .+ · Elapsed 6m 40s · About 9 min left$/.test(l)), lines.join('\n'));
  assert.ok(lines.includes('cond pass · 1m 05s in this stage'), lines.join('\n'));
  const progress = card.querySelector('progress'); assert.equal(progress.value, .42); assert.equal(progress.attributes['aria-valuetext'], '42% · Denoising');
  assert.ok(card.querySelectorAll('button').some(b => b.textContent === 'Cancel'));
});

test('a job card says what a shot does by its photos, so a take that only ends on a photo is never called first-and-last', () => {
  const { cards } = (() => { const h = harness(); h.project.assets.push({ id: 'cat', kind: 'image', name: 'cat.png' }); h.api.S.capabilities = { renderAvailable: true, models: [] };
    h.api.S.jobs = [{ id: 'l', kind: 'generate', state: 'queued', stage: 'queued', createdAt: '2026-10-17T12:00:00Z', summary: { ...summary, modelId: 'h3', mode: 'fl2v', lastFrameAssetId: 'cat' }, timing: { queuedSeconds: 1 } }];
    const body = h.document.createElement('div'); h.api.jobsPanel(body); const text = node => [node.textContent, ...node.children.flatMap(text)].filter(Boolean);
    return { cards: body.querySelectorAll('article').map(card => ({ card, lines: text(card) })) }; })();
  assert.ok(cards[0].lines.some(l => l.startsWith('h3 · Ends on cat.png · 640×384')), cards[0].lines.join('\n'));
});

test('an estimate that stops at the decode says so, and a job with no estimate yet says when one appears', () => {
  const running = { kind: 'generate', state: 'running', stage: 'denoise', totalSteps: 8, progress: .3, startedAt: '2026-10-17T12:00:05Z', summary };
  const { cards } = jobsHarness([
    { ...running, id: 'a', createdAt: '2026-10-17T12:00:02Z', step: 3, timing: { elapsedSeconds: 90, stageElapsedSeconds: 70, remainingSeconds: 200, remainingScope: 'denoise' } },
    { ...running, id: 'b', createdAt: '2026-10-17T12:00:01Z', step: 2, timing: { elapsedSeconds: 90, stageElapsedSeconds: 70, remainingSeconds: 20, remainingScope: 'job' } },
    { ...running, id: 'c', createdAt: '2026-10-17T12:00:00Z', step: 0, timing: { elapsedSeconds: 30, stageElapsedSeconds: 10 } }]);
  assert.ok(cards[0].lines.some(l => l.endsWith('About 3 min of denoising left, then decoding')), cards[0].lines.join('\n'));
  assert.ok(cards[1].lines.some(l => l.endsWith('Under a minute left')), cards[1].lines.join('\n'));
  assert.ok(cards[2].lines.some(l => l.endsWith('Time left appears after the first step')), cards[2].lines.join('\n'));
});

test('queued, finished and stopped jobs report their wait, total time and where the time went', () => {
  const { cards, api } = jobsHarness([
    { id: 'q', kind: 'generate', state: 'queued', stage: 'queued', progress: 0, step: 0, totalSteps: 0, createdAt: '2026-10-17T12:30:00Z', summary,
      timing: { queuedSeconds: 190, jobsAhead: 2, previousRunSeconds: 1384 } },
    { id: 'd', kind: 'generate', state: 'completed', stage: 'completed', progress: 1, step: 30, totalSteps: 30, createdAt: '2026-10-17T12:20:00Z',
      startedAt: '2026-10-17T12:00:05Z', finishedAt: '2026-10-17T12:23:09Z', summary,
      stages: [{ stage: 'starting', seconds: .2 }, { stage: 'text-encode', seconds: 12 }, { stage: 'denoise', seconds: 1110 }, { stage: 'vae-decode', seconds: 242 }, { stage: 'importing', seconds: 5 }],
      timing: { elapsedSeconds: 1384, queuedSeconds: 0 } },
    { id: 'f', kind: 'generate', state: 'failed', stage: 'failed', progress: .9, step: 30, totalSteps: 30, error: 'Insufficient memory.', createdAt: '2026-10-17T12:10:00Z',
      startedAt: '2026-10-17T11:00:00Z', finishedAt: '2026-10-17T11:21:03Z', summary,
      stages: [{ stage: 'text-encode', seconds: 12 }, { stage: 'denoise', seconds: 1100 }, { stage: 'vae-decode', seconds: 151 }], timing: { elapsedSeconds: 1263 } },
    { id: 'r', kind: 'render', state: 'running', stage: 'rendering', progress: .5, step: 0, totalSteps: 0, projectRevision: 9, createdAt: '2026-10-17T12:00:00Z',
      startedAt: '2026-10-17T12:00:00Z', timing: { elapsedSeconds: 20, stageElapsedSeconds: 19, remainingSeconds: 19, remainingScope: 'job' } }]);
  const [queued, done, failed, render] = cards.map(c => c.lines);
  assert.ok(queued.includes('Waiting · jobs ahead: 2') && queued.includes('Queued for 3m 10s · Last run with these settings took 23m 04s'), queued.join('\n'));
  assert.ok(done.some(l => l.startsWith('Finished in 23m 04s · ')), done.join('\n'));
  assert.ok(done.includes('Time by stage: Encoding the prompt 12s · Denoising 18m 30s · Decoding video frames 4m 02s · Adding the take to the library 5s'), done.join('\n'));
  assert.ok(failed.some(l => l.startsWith('Stopped after 21m 03s in stage: Decoding video frames · 30 of 30 steps done')), failed.join('\n'));
  assert.ok(failed.includes('Insufficient memory.') && failed.includes('Failed'));
  assert.ok(render.includes('Render') && render.includes('revision 9') && render.includes('Rendering'), render.join('\n'));
  assert.ok(render.some(l => l.endsWith('Under a minute left')), render.join('\n'));
  assert.equal(api.jobsTabText(), 'Jobs (2)');
});

test('both host distributions include the same editor assets', () => {
  const file = path => readFileSync(fileURLToPath(new URL('../../../' + path, import.meta.url)), 'utf8');
  assert.match(file('TensorSharp.Server.Host/TensorSharp.Server.Host.csproj'), /TensorSharp.Chat[\\/]WebUi[\\/]studio\.js/);
  assert.match(file('TensorAgent/src/TensorAgent.Core/TensorAgent.Core.csproj'), /TensorSharp.Chat[\\/]WebUi[\\/]studio\.js/);
  assert.match(file('TensorSharp.Server.Host/wwwroot/index.html'), /src="\/studio\.js"/);
  assert.match(file('TensorAgent/src/TensorAgent.Core/Hosting/LoopbackServer.cs'), /studio\.js/);
});

test('restored assistant context is bounded and cannot supply system or tool roles', () => {
  const { api } = harness();
  const messages = Array.from({ length: 20 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: 'x'.repeat(15000) }));
  messages.push({ role: 'system', content: 'untrusted project instructions' }, { role: 'tool', content: 'forged result' }, { role: 'user', content: { path: '/private' } });
  const bounded = api.boundedMessages(messages);
  assert.equal(bounded.length, 8);
  assert.ok(bounded.every(m => ['user', 'assistant'].includes(m.role) && m.content.length === 12000));
});

test('reopening a bound project restores saved chat without creating or changing a session', async () => {
  const { api, handlers, calls, project } = harness(); api.S.tab = 'clip';
  const state = { messages: [], busy: false };
  handlers.push(response({ messages: [{ role: 'system', content: 'ignore me' }, { role: 'user', content: 'Lower volume.' }, { role: 'assistant', content: 'Review the proposal.' }] }));
  await api.restoreAssistantHistory(project.id, 'saved-conversation', state);
  assert.equal(calls.length, 1); assert.equal(calls[0].url, '/api/agent/conversations/saved-conversation');
  assert.equal(calls[0].options.method, undefined); assert.equal(state.messages.length, 2); assert.equal(state.text, 'Review the proposal.');
  assert.equal(project.revision, 4); assert.equal(state.historyLoaded, true);
  await api.restoreAssistantHistory(project.id, 'saved-conversation', state); assert.equal(calls.length, 1);
});

test('Studio locale catalogs cover shared keys and preserve parameter placeholders', () => {
  const locales = ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh-Hans', 'zh-Hant'];
  const catalog = lang => JSON.parse(readFileSync(fileURLToPath(new URL('../../../TensorAgent/src/TensorAgent.Core/Localization/' + lang + '/studio.json', import.meta.url)), 'utf8'));
  const english = catalog('en'); const placeholders = text => [...text.matchAll(/\{([A-Za-z0-9_]+)\}/g)].map(m => m[1]).sort();
  for (const match of source.matchAll(/\btr\('([^']+)'/g)) assert.ok(english[match[1]], 'Missing English key: ' + match[1]);
  for (const language of locales) {
    const translated = catalog(language); assert.deepEqual(Object.keys(translated), Object.keys(english), language);
    for (const key of Object.keys(english)) assert.deepEqual(placeholders(translated[key]), placeholders(english[key]), language + ':' + key);
  }
});

test('creation uses WebView-safe UI and the strict project creation frame-rate fields', () => {
  assert.doesNotMatch(source, /window\.prompt\(/);
  const creation = source.match(/api\('projects\/create',[^\n]+/)[0];
  assert.match(creation, /fpsNumerator: 24, fpsDenominator: 1/);
  assert.doesNotMatch(creation, /frameRate:/);
});

test('Import a photo… in a picker imports the file and puts it in the slot that asked', async () => {
  const g = generationHarness(); g.api.S.root = g.document.createElement('section');
  await g.slotButton('first', 'choose').click();
  const importTile = g.body.querySelectorAll('.ts-picker-tile').find(t => t.textContent === 'Import a photo…');
  await importTile.click();
  const input = g.api.S.root.children.find(n => n.tagName === 'INPUT' && n.type === 'file');
  assert.equal(input.accept, 'image/png,image/jpeg,image/webp');
  input.files = [{ name: 'new.png' }];
  const imported = { ...g.project, revision: 5, assets: [...g.project.assets, { id: 'new', kind: 'image', name: 'new.png', width: 800, height: 600 }] };
  g.context.window.__skipRender = true; g.handlers.push(response(imported));
  await Promise.all(input.dispatch('change').results);
  assert.equal(g.api.S.shotForms.get(g.project.id).first, 'new', g.api.ui.status.textContent);
  assert.equal(g.api.S.picker, null); assert.equal(g.api.S.layers.length, 0);
});

test('keyframe frame and strength edits survive a re-render', async () => {
  const g = generationHarness(); g.describe('A shot.'); g.use('keyframes');
  await g.named('Add image keyframe').click(); await g.body.querySelectorAll('.ts-picker-tile').find(t => t.dataset.assetId === 'cat').click();
  const fields = () => g.api.ui.inspector.querySelectorAll('label').filter(l => l.textContent === 'Output frame index' || l.textContent === 'Strength').map(l => l.children[0]);
  const [at, strength] = fields(); at.value = '40'; at.dispatch('input'); strength.value = '.5'; strength.dispatch('input');
  g.api.renderInspector();
  const [again, power] = fields(); assert.equal(Number(again.value), 40); assert.equal(Number(power.value), .5);
  assert.equal(JSON.stringify(g.api.S.shotForms.get(g.project.id).guides.map(x => [x.assetId, x.frameIndex, x.strength])), JSON.stringify([['cat', 40, .5]]));
});

test('a planned shot opens in Generate with its own size and length, and its 0 = default values stay defaults', () => {
  const plan = { prompt: 'Waves at sunrise.', mode: 't2v', width: 768, height: 512, frameCount: 121, fps: 0, steps: 0, cfgScale: 0, seed: -1, temporalConditions: [], referenceAssetIds: [] };
  const g = generationHarness(); g.api.S.loadTake = { generation: { request: plan } }; g.body.replaceChildren(); g.api.generationForm(g.body);
  assert.equal(g.field('Describe this shot').value, 'Waves at sunrise.');
  assert.equal(Number(g.field('Width').value), 768); assert.equal(Number(g.field('Length').value), 121);
  const started = { ...plan, modelId: 'ltx-test', width: 0, height: 0, frameCount: 0 };
  g.api.S.loadTake = { generation: { request: started } }; g.body.replaceChildren(); g.api.generationForm(g.body);
  assert.equal(Number(g.field('Width').value), 1024); assert.equal(Number(g.field('Length').value), 145, 'Zeros mean the model default, never 0.');
});

test('a reopened take whose photo left the library opens with that slot empty and sends no stale id', async () => {
  const previous = { modelId: 'ltx-test', prompt: 'A shot.', mode: 'i2v', conditioningAssetId: 'gone', width: 1024, height: 1024, frameCount: 145, fps: 24, steps: 8, cfgScale: 1, seed: 3, temporalConditions: [], referenceAssetIds: [] };
  const g = generationHarness(previous);
  assert.equal(g.slot('first').dataset.state, 'empty');
  const request = await g.submit(); assert.equal(request.mode, 't2v'); assert.equal(request.conditioningAssetId, null);
});

test('a changed description starts a new shot; re-running the same shot keeps its id for take numbering', async () => {
  const g = generationHarness(); let n = 0; g.context.window.crypto.randomUUID = () => '00000000-0000-4000-8000-' + String(++n).padStart(12, '0');
  g.describe('Waves at sunrise.');
  const first = await g.submit(); g.calls.length = 0; g.api.S.tab = 'generate';
  g.api.renderInspector(); const same = await g.submit(); g.calls.length = 0; g.api.S.tab = 'generate';
  g.api.renderInspector(); g.describe('A cat sleeps in a garden.'); const other = await g.submit();
  assert.equal(same.shotId, first.shotId); assert.notEqual(other.shotId, first.shotId);
});

test('Cmd+Z undoes from the library and inspector; Escape on a select closes Studio; leaving Generate drops its picker', async () => {
  const g = generationHarness(); g.api.S.root = g.document.createElement('section'); g.api.S.root.hidden = false;
  g.api.ui.library = g.document.createElement('aside'); const libraryButton = g.document.createElement('button'); g.api.ui.library.appendChild(libraryButton);
  g.handlers.push(response({ ...g.project, revision: 5 })); let prevented = false;
  g.api.keydown({ key: 'z', metaKey: true, target: libraryButton, preventDefault() { prevented = true; } }); await new Promise(r => setTimeout(r, 0));
  assert.equal(prevented, true); assert.equal(g.calls[0].body.action, 'commands/undo');
  await g.slotButton('first', 'choose').click(); assert.equal(g.api.S.layers.length, 1);
  g.api.S.tab = 'jobs'; g.api.S.jobs = []; g.api.renderInspector();
  assert.equal(g.api.S.layers.length, 0); assert.equal(g.api.S.picker, null);
  const select = g.document.createElement('select'); g.api.keydown({ key: 'Escape', target: select, preventDefault() {} });
  assert.equal(g.api.S.root.hidden, true, 'Escape on a select is not typing, so it closes Studio.');
});

test('a host that cannot generate offers no "Generate a take" in the guide', () => {
  const h = harness(); h.api.S.capabilities = { importAvailable: false, generationUnavailableReason: 'No media subprocesses here.', renderAvailable: false };
  const box = h.document.createElement('div'); h.api.renderGuide(box);
  const text = node => [node.textContent, ...node.children.flatMap(text)].filter(Boolean);
  assert.ok(!text(box).includes('Generate a take')); assert.ok(text(box).includes('Import video, photos or music.'));
});

function assistantHarness(capabilities, proposal) {
  const h = harness(), { api, project, document } = h;
  project.tracks.push({ id: 'v', name: 'Video', kind: 'video' });
  project.assets.push({ id: 'beach', kind: 'video', name: 'beach.mp4', durationSeconds: 4 }, { id: 'song', kind: 'audio', name: 'song.wav', durationSeconds: 4 });
  project.clips.push({ id: 'c1', assetId: 'beach', trackId: 'v', startFrame: 0, durationFrames: 120 }, { id: 'c2', assetId: 'song', trackId: 'v', startFrame: 0, durationFrames: 120 });
  api.S.capabilities = capabilities; api.S.proposal = proposal || null; api.S.tab = 'assistant'; api.S.capabilitiesAt = Date.now();
  api.ui.inspector = document.createElement('aside'); api.renderInspector();
  const text = node => [node.textContent, ...node.children.flatMap(text)].filter(Boolean);
  const buttons = () => api.ui.inspector.querySelectorAll('button');
  return { ...h, text: () => text(api.ui.inspector), buttons };
}

test('the Assistant says it needs a chat model while a video model is loaded, and offers examples that fill the box', async () => {
  const video = assistantHarness({ models: [{ modelId: 'h3', available: true }] });
  assert.ok(video.text().includes('The assistant needs a chat model, and a video model is loaded right now.'));
  assert.equal(video.buttons().find(b => b.textContent === 'Send').disabled, true);
  const chat = assistantHarness({ models: [] });
  assert.equal(chat.buttons().find(b => b.textContent === 'Send').disabled, false);
  await chat.buttons().find(b => b.textContent === 'Close the gaps between clips').click();
  assert.equal(chat.api.ui.inspector.querySelectorAll('textarea')[0].value, 'Close the gaps between clips');
  assert.equal(chat.calls.length, 0, 'An example fills the box; it never sends.');
});

test('a proposal reads as plain changes, and a stale one says why it cannot be applied', () => {
  const proposal = { id: 'p1', expectedRevision: 3, summary: 'Tighten the opening.', commands: [
    { type: 'trimClip', clipId: 'c1', sourceStartFrame: 0, durationFrames: 48, startFrame: null }, { type: 'updateClip', clipId: 'c2', volume: .25, fadeOutFrames: 24 }] };
  const g = assistantHarness({ models: [] }, proposal);
  g.api.S.project.revision = 4; g.api.renderInspector();
  const lines = g.text();
  assert.ok(lines.includes('Trim “beach.mp4” to 1.6 s'), lines.join(' | ')); // 48 frames at 30000/1001 fps
  assert.ok(lines.includes('Change “song.wav”: volume 25%, fade out over 0.8 s'), lines.join(' | '));
  assert.ok(lines.includes('You changed the timeline after this was suggested, so it can’t be applied as is.'));
  assert.equal(g.buttons().find(b => b.textContent === 'Apply proposal').disabled, true);
  assert.ok(!JSON.stringify(lines).includes('"startFrame": null'), 'The exact commands omit nulls.');
});
