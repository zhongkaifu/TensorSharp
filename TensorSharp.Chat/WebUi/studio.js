// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// One shared editor for WebUX and TensorAgent. The host owns all persistent state.
(function () {
  'use strict';
  const API = '/api/studio';
  const S = { root: null, project: null, projects: [], capabilities: null, selected: null, selectedCaption: null,
    playhead: 0, zoom: 65, snap: true, busy: false, jobs: [], proposal: null, tab: 'clip',
    previewRevision: null, previewJob: null, sourcePreview: false, sourcePreviewAssetId: null, timer: null, epoch: 0, previousFocus: null, generationDrafts: [], deleteDialog: null,
    // The Generate form's state, one shot form per project, so no re-render, poll or tab switch loses an unsent shot.
    shotForms: new Map(), picker: null, loadTake: null, highlight: null, capabilitiesAt: 0, ticks: 0,
    // What Escape closes first: the newest open picker or narrow-screen panel, before Studio itself.
    layers: [] };
  const ui = {};
  function tr(key, fallback, params) {
    const i18n = window.TensorSharpI18n;
    if (i18n && typeof i18n.t === 'function') return i18n.t(key, fallback, params);
    return String(fallback).replace(/\{(\w+)\}/g, (all, k) => params && Object.prototype.hasOwnProperty.call(params, k) ? String(params[k]) : all);
  }
  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function button(text, action, cls) {
    const b = el('button', cls, text); b.type = 'button';
    b.addEventListener('click', () => run(action)); return b;
  }
  function field(parent, label, value, type, options) {
    const l = el('label', 'ts-field', label); const n = el(type === 'textarea' ? 'textarea' : type === 'select' ? 'select' : 'input');
    if (type !== 'textarea' && type !== 'select') n.type = type || 'text';
    if (options) for (const o of options) { const item = el('option', '', o.label); item.value = o.value; item.disabled = !!o.disabled; n.appendChild(item); }
    if (value != null) n.value = String(value); l.appendChild(n); parent.appendChild(l); return n;
  }
  function num(parent, label, value, min, max, step) { const n = field(parent, label, value, 'number'); n.min = min == null ? 0 : min; if (max != null) n.max = max; n.step = step || 1; return n; }
  function checked(parent, label, value, changed) { const l = el('label', 'ts-check'); const n = el('input'); n.type = 'checkbox'; n.checked = value; l.append(n, document.createTextNode(label)); if (changed) n.addEventListener('change', () => run(() => changed(n.checked))); parent.appendChild(l); return n; }
  function row(parent) { const n = el('div', 'ts-row'); parent.appendChild(n); return n; }
  function id() { return window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID().replace(/-/g, '') : Date.now().toString(16) + Math.random().toString(16).slice(2); }
  function fps() { return S.project ? S.project.frameRate.numerator / S.project.frameRate.denominator : 24; }
  function seconds(frames) { return frames / fps(); }
  function frames(sec) { return Math.round(sec * fps()); }
  function fmt(frame) { return seconds(frame).toFixed(2) + 's · ' + frame + 'f'; }
  function clip() { return S.project && S.project.clips.find(c => c.id === S.selected); }
  function asset(assetId) { return S.project && S.project.assets.find(a => a.id === assetId); }
  // The canvas a model's keyframe recipe gives an image (capability imageCanvas*), the same rule the host applies:
  // the image's aspect at the short edge, shrunk to the pixel cap, each side on the nearest grid step, ties to even.
  function recipeCanvas(m, w, h) {
    const edge = m.imageCanvasShortEdge, cap = m.imageCanvasMaxPixels, limit = m.imageCanvasMaxAspect, grid = m.spatialMultiple || 32;
    if (!edge || !limit || !(w > 0 && h > 0)) return null;
    const r = Math.min(limit, Math.max(1 / limit, w / h)); // beyond the recipe's range the host clamps and crops
    let cw = r >= 1 ? edge * r : edge, ch = r >= 1 ? edge : edge / r;
    if (cap && cw * ch > cap) { const k = Math.sqrt(cap / (cw * ch)); cw *= k; ch *= k; }
    const near = v => { const q = v / grid, f = Math.floor(q), d = q - f; return Math.max(grid, (d > .5 || d === .5 && f % 2 ? f + 1 : f) * grid); };
    return { width: near(cw), height: near(ch) };
  }
  function mediaUrl(assetId) { return API + '/projects/' + encodeURIComponent(S.project.id) + '/assets/' + encodeURIComponent(assetId); }
  function outputUrl(job) { return API + '/projects/' + encodeURIComponent(job.projectId) + '/jobs/' + encodeURIComponent(job.id) + '/output'; }
  function status(text, error) { if (!ui.status) return; ui.status.textContent = text; ui.status.dataset.error = String(!!error); }
  async function response(res) {
    const text = await res.text(); let value;
    try { value = text ? JSON.parse(text) : {}; } catch (_) { value = { message: text.slice(0, 400) }; }
    if (!res.ok) { const detail = value.message || value.error || value.detail || res.statusText; const e = new Error(typeof detail === 'string' ? detail : detail.message || JSON.stringify(detail)); e.status = res.status; e.code = value.code; throw e; }
    return value;
  }
  async function api(action, payload) {
    return response(await fetch(API, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-TensorSharp-Studio': '1' }, body: JSON.stringify({ action, payload: payload || {} }) }));
  }
  // Refusals about the loaded model (it changed, or its plug-in did) rather than about the project.
  const MODEL_REFUSALS = ['model_unavailable', 'model_not_found', 'model_changed', 'model_conflict', 'sampling_recipe_conflict', 'unsupported_mode', 'unsupported_conditioning',
    'invalid_sampling', 'invalid_generation', 'invalid_frame_count', 'invalid_resolution', 'invalid_fps', 'invalid_mode_inputs', 'unsupported_negative_prompt', 'generation_output_unavailable'];
  // Only a revision conflict means "the project changed"; a 409 about the model asks for its current settings instead.
  function revisionConflict(e) { return e.code ? String(e.code).includes('revision') : e.status === 409; }
  async function run(action) {
    try { return await action(); }
    catch (e) {
      status(e.message || String(e), true);
      if (S.project && revisionConflict(e)) {
        try { await refreshProject(); status(tr('studio.conflict', 'The project changed. Your edit was not applied. Review the latest revision and try again.'), true); } catch (_) { /* Keep the original error. */ }
      } else if (MODEL_REFUSALS.includes(e.code)) refreshCapabilities().catch(() => { /* Keep the original error. */ });
    }
  }
  // One persistent polite region: VoiceOver skips live regions created together with their text.
  function announce(text) {
    if (!ui.announce) return; ui.announce.textContent = '';
    const set = () => { ui.announce.textContent = text; };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(set); else set();
  }
  function within(node, region) { for (let n = node; n && region; n = n.parentNode) if (n === region) return true; return false; }
  // Stable keys let a re-render put focus back where it was (and where an action says it should go next).
  function findKey(node, key) {
    if (!node) return null; if (node.dataset && node.dataset.focusKey === key) return node;
    for (const child of Array.from(node.children || [])) { const hit = findKey(child, key); if (hit) return hit; }
    return null;
  }
  function focusKey(region, key) { const n = findKey(region, key); if (n && n.focus) n.focus(); return n; }
  function pushLayer(name, closeLayer) { S.layers = S.layers.filter(l => l.name !== name); S.layers.push({ name, close: closeLayer }); }
  function dropLayer(name) { S.layers = S.layers.filter(l => l.name !== name); }
  function setBusy(value) {
    S.busy = value; const text = value ? tr('studio.saving', 'Saving…') : S.project ? tr('studio.savedPlain', 'Saved') : tr('studio.local', 'Local project storage');
    if (ui.save.textContent !== text) ui.save.textContent = text; // an unchanged write would be re-read aloud on every poll
    ui.save.title = S.project && !value ? tr('studio.saved', 'Saved · revision {revision}', { revision: S.project.revision }) : '';
  }
  function receiveProject(value) {
    const p = value.project || value; if (!p.id || !Array.isArray(p.tracks)) throw new Error(tr('studio.invalidProject', 'The host returned an invalid project.')); S.project = p;
    if (S.selected && !p.clips.some(c => c.id === S.selected)) S.selected = null;
    if (S.regenerateAsset && !p.assets.some(a => a.id === S.regenerateAsset)) S.regenerateAsset = null;
    if (S.sourcePreviewAssetId && !p.assets.some(a => a.id === S.sourcePreviewAssetId)) resetPreview();
    return p;
  }
  async function change(commands) {
    if (!S.project || S.busy) return;
    const projectId = S.project.id, expectedRevision = S.project.revision;
    setBusy(true);
    try {
      const p = await api('commands/apply', { projectId, expectedRevision, commandId: id(), commands });
      if (S.project && S.project.id === projectId) { receiveProject(p); renderProject(); status(tr('studio.applied', 'Edit saved. Source assets are preserved.')); return true; }
    } finally { setBusy(false); }
  }
  async function history(kind) {
    if (!S.project || S.busy) return;
    const projectId = S.project.id, expectedRevision = S.project.revision;
    setBusy(true);
    try { const project = await api('commands/' + kind, { projectId, expectedRevision, commandId: id() }); if (S.project && S.project.id === projectId) { receiveProject(project); renderProject(); status(tr('studio.applied', 'Edit saved. Source assets are preserved.')); } }
    finally { setBusy(false); }
  }
  async function listProjects() {
    const value = await api('projects/list'); S.projects = Array.isArray(value) ? value : value.projects || [];
    const selected = S.project && S.project.id; ui.projectSelect.replaceChildren();
    const empty = el('option', '', tr('studio.chooseProject', 'Open a project…')); empty.value = ''; ui.projectSelect.appendChild(empty);
    for (const p of S.projects) { const n = el('option', '', p.name); n.value = p.id; ui.projectSelect.appendChild(n); }
    ui.projectSelect.value = selected || '';
  }
  async function newProject() {
    const name = await projectName();
    if (name == null || !name.trim()) return;
    await createProject(name.trim());
  }
  async function createProject(name) {
    receiveProject(await api('projects/create', { name, width: 1280, height: 720, fpsNumerator: 24, fpsDenominator: 1 }));
    S.selected = null; S.selectedCaption = null; S.proposal = null; S.generationDrafts = []; S.playhead = 0; resetPreview(); renderProject(); rememberProject(); await listProjects(); await refreshJobs();
  }
  // The guide's first step works before a project exists: it starts an untitled one, renamed later in Project settings.
  async function ensureProject() { if (!S.project) await createProject(tr('studio.untitled', 'Untitled video')); return !!S.project; }
  function projectName() {
    return new Promise(resolve => {
      const previous = document.activeElement, backdrop = el('div', 'ts-name-dialog'), form = el('form', 'ts-stack');
      form.setAttribute('role', 'dialog'); form.setAttribute('aria-modal', 'true'); form.setAttribute('aria-label', tr('studio.createProject', 'Create project'));
      form.appendChild(el('h2', '', tr('studio.createProject', 'Create project'))); const input = field(form, tr('studio.namePrompt', 'Project name'), tr('studio.untitled', 'Untitled video')); input.required = true; input.maxLength = 200;
      const finish = value => { backdrop.remove(); if (previous && previous.isConnected) previous.focus(); resolve(value); };
      const actions = row(form); actions.appendChild(button(tr('studio.cancel', 'Cancel'), () => finish(null)));
      const submit = el('button', 'ts-primary', tr('studio.create', 'Create')); submit.type = 'submit'; actions.appendChild(submit);
      form.addEventListener('submit', e => { e.preventDefault(); if (input.value.trim()) finish(input.value.trim()); });
      backdrop.addEventListener('keydown', e => {
        if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); finish(null); }
        if (e.key === 'Tab') { e.stopPropagation(); const focusable = [input, ...actions.querySelectorAll('button')]; const first = focusable[0], last = focusable[focusable.length - 1]; if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); } }
      });
      backdrop.appendChild(form); S.root.appendChild(backdrop); input.focus(); input.select();
    });
  }
  function rememberProject() { try { window.localStorage.setItem('tensorsharp.studio.project', S.project.id); } catch (_) { /* Storage can be disabled. */ } }
  async function openProject(projectId) {
    if (!projectId || S.busy) return; ++S.epoch;
    receiveProject(await api('projects/get', { projectId })); S.selected = null; S.selectedCaption = null; S.proposal = null; S.generationDrafts = []; S.playhead = 0;
    S.picker = null; S.loadTake = null; dropLayer('picker');
    resetPreview(); renderProject(); rememberProject(); await refreshJobs(); await loadProposals(); status(tr('studio.opened', 'Project reopened. Every accepted edit is saved by the host.'));
  }
  async function refreshProject() {
    if (!S.project) return; const projectId = S.project.id; const value = await api('projects/get', { projectId });
    if (S.project.id === projectId) { receiveProject(value); renderProject(); }
  }
  function build() {
    if (S.root) return;
    S.root = el('section', 'ts-studio'); S.root.hidden = true; S.root.tabIndex = -1; S.root.setAttribute('role', 'dialog'); S.root.setAttribute('aria-modal', 'true'); S.root.setAttribute('aria-label', tr('studio.title', 'Video Studio'));
    const header = el('div', 'ts-studio-header'); header.appendChild(el('h2', '', tr('studio.title', 'Video Studio')));
    ui.projectSelect = el('select'); ui.projectSelect.setAttribute('aria-label', tr('studio.projects', 'Projects')); ui.projectSelect.addEventListener('change', () => run(() => openProject(ui.projectSelect.value)));
    header.append(ui.projectSelect, button(tr('studio.new', 'New'), newProject));
    const reload = button(tr('studio.reload', 'Reload'), refreshProject); reload.title = tr('studio.reloadHint', 'Read the project again from the host.'); header.append(reload);
    ui.save = el('span', 'ts-save-state', tr('studio.local', 'Local project storage')); header.appendChild(ui.save);
    header.appendChild(button(tr('studio.backChat', 'Back to chat'), close)); S.root.appendChild(header);
    ui.status = el('div', 'ts-status', tr('studio.welcomeHint', 'Import media or generate footage, then assemble an editable timeline.')); ui.status.setAttribute('role', 'status'); ui.status.setAttribute('aria-live', 'polite'); S.root.appendChild(ui.status);
    ui.announce = el('div', 'ts-visually-hidden'); ui.announce.setAttribute('role', 'status'); ui.announce.setAttribute('aria-live', 'polite'); ui.announce.setAttribute('aria-atomic', 'true'); S.root.appendChild(ui.announce);
    // On narrow screens the side panels are overlays: each tab shows its panel (it does not toggle it) and says which is open.
    const mobile = el('div', 'ts-mobile-tabs'); ui.mobileTabs = {};
    for (const [name, label, tab] of [['library', tr('studio.library', 'Library')], ['inspector', tr('studio.properties', 'Properties'), 'clip'], ['inspector', tr('studio.generate', 'Generate'), 'generate'],
      ['inspector', tr('studio.assistant', 'Assistant'), 'assistant'], ['inspector', tr('studio.jobs', 'Jobs'), 'jobs']]) {
      const b = button(label, () => showPanel(name, tab)); b.dataset.focusKey = 'mobile-' + (tab || name); ui.mobileTabs[tab || name] = b; mobile.appendChild(b);
    }
    const hidePanels = button('×', () => hidePanel()); hidePanels.setAttribute('aria-label', tr('studio.closePanels', 'Close side panels')); mobile.appendChild(hidePanels);
    S.root.appendChild(mobile);
    ui.workspace = el('div', 'ts-studio-workspace'); S.root.appendChild(ui.workspace);
    ui.library = el('aside', 'ts-studio-section ts-studio-library'); ui.library.dataset.panelHidden = 'true';
    ui.preview = el('section', 'ts-studio-section ts-studio-preview'); ui.inspector = el('aside', 'ts-studio-section ts-studio-inspector'); ui.inspector.dataset.panelHidden = 'true';
    ui.timeline = el('section', 'ts-studio-timeline'); ui.timeline.setAttribute('aria-label', tr('studio.timeline', 'Timeline')); ui.workspace.append(ui.library, ui.preview, ui.inspector, ui.timeline);
    ui.import = el('input'); ui.import.type = 'file'; ui.import.multiple = true; ui.import.accept = 'video/*,audio/*,image/*'; ui.import.hidden = true;
    ui.import.addEventListener('change', () => run(async () => { const files = Array.from(ui.import.files || []); ui.import.value = ''; if (files.length && !await ensureProject()) return; for (const file of files) await importFile(file); })); S.root.appendChild(ui.import);
    S.root.addEventListener('keydown', keydown); document.body.appendChild(S.root); renderProject();
    // What the loaded model can make is re-read when the person comes back to Studio, and when a host page says the
    // model or its plug-ins changed (TensorAgent dispatches tensorsharp:model-changed); a refresh only redraws on change.
    const returned = () => { if (S.root && !S.root.hidden && document.visibilityState !== 'hidden') background(() => refreshCapabilities({ stale: 1000 })).catch(() => { /* The last known capabilities stay. */ }); };
    document.addEventListener('visibilitychange', returned);
    if (window.addEventListener) { window.addEventListener('focus', returned); window.addEventListener('tensorsharp:model-changed', returned); }
    // A redraw a background event asked for while someone was typing waits until they leave the field.
    // Not while a pointer is held: replacing the pressed button between mousedown and mouseup would swallow its click.
    let pressed = false; const flush = () => { if (S.inspectorDirty && !editingInspector()) { S.inspectorDirty = false; renderInspector(); } };
    const released = () => { if (pressed) { pressed = false; setTimeout(flush, 0); } };
    ui.inspector.addEventListener('pointerdown', () => { pressed = true; }, true);
    document.addEventListener('pointerup', released, true); document.addEventListener('pointercancel', released, true);
    ui.inspector.addEventListener('focusout', () => setTimeout(() => { if (!pressed) flush(); }, 0));
  }
  function editingInspector() {
    const a = document.activeElement;
    return !!a && within(a, ui.inspector) && (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT' && !/^(checkbox|radio|range|button|file)$/.test(a.type));
  }
  // Marks renders caused by polls, finished jobs and capability changes, as opposed to the person's own actions.
  async function background(task) { S.background = (S.background || 0) + 1; try { return await task(); } finally { S.background--; } }
  function togglePanel(name, force) { const panel = ui[name]; panel.dataset.panelHidden = String(force ? false : panel.dataset.panelHidden !== 'true'); if (name === 'library') ui.inspector.dataset.panelHidden = 'true'; else ui.library.dataset.panelHidden = 'true'; }
  function showPanel(name, tab) {
    if (tab) { S.tab = tab; renderInspector(); if (tab === 'generate' || tab === 'assistant') run(() => refreshCapabilities({ stale: 3000 })); if (tab === 'assistant') run(loadProposals); }
    togglePanel(name, true); S.panelTab = tab || name;
    for (const [key, b] of Object.entries(ui.mobileTabs || {})) b.setAttribute('aria-pressed', String(key === S.panelTab));
    if (narrow()) pushLayer('panel', () => hidePanel()); // a wide layout shows both panels; there is nothing to close
    syncMobileTabs();
    // A control hidden with the other panel takes focus with it: keep it inside the panel that is now showing.
    const active = document.activeElement; if (!active || !within(active, S.root) || !active.getClientRects || !active.getClientRects().length) focusRegion(ui[name]);
  }
  function narrow() { try { return !!(window.matchMedia && window.matchMedia('(max-width: 1000px)').matches); } catch (_) { return false; }
  }
  // On narrow screens the mobile row is the only tab indicator: it follows the inspector's tab and the jobs count.
  function syncMobileTabs() {
    if (!ui.mobileTabs || !ui.inspector) return; const open = ui.inspector.dataset.panelHidden !== 'true', library = ui.library && ui.library.dataset.panelHidden !== 'true';
    for (const [key, b] of Object.entries(ui.mobileTabs)) b.setAttribute('aria-pressed', String(key === 'library' ? library : open && key === S.tab));
    if (ui.mobileTabs.jobs) ui.mobileTabs.jobs.textContent = jobsTabText();
  }
  function hidePanel() {
    if (!ui.library) return; ui.library.dataset.panelHidden = 'true'; ui.inspector.dataset.panelHidden = 'true'; dropLayer('panel');
    const back = ui.mobileTabs && ui.mobileTabs[S.panelTab]; S.panelTab = null;
    for (const b of Object.values(ui.mobileTabs || {})) b.setAttribute('aria-pressed', 'false');
    if (back && back.isConnected) back.focus();
  }
  async function open(options) {
    build(); S.previousFocus = document.activeElement; S.root.hidden = false; S.root.focus();
    const tasks = await Promise.allSettled([api('capabilities'), listProjects()]);
    if (tasks[0].status === 'fulfilled') { S.capabilities = tasks[0].value; S.capabilitiesAt = Date.now(); }
    else status(tasks[0].reason.message, true);
    if (!S.project) { let previous; try { previous = window.localStorage.getItem('tensorsharp.studio.project'); } catch (_) { }
      if (previous && S.projects.some(p => p.id === previous)) await run(() => openProject(previous)); }
    renderProject();
    if (options && options.url) {
      if (!S.project) await newProject();
      if (S.project) await importChat(options.url);
    }
    if (S.timer) clearTimeout(S.timer); poll();
    ui.projectSelect.focus();
  }
  function close() { if (!S.root) return; if (S.deleteDialog) S.deleteDialog(false); S.root.hidden = true; S.layers = []; S.picker = null; if (ui.video) ui.video.pause(); if (S.timer) clearTimeout(S.timer); S.timer = null; if (S.previousFocus && S.previousFocus.isConnected) S.previousFocus.focus(); }
  async function poll() {
    if (!S.root || S.root.hidden) return;
    try {
      if (S.project) await background(refreshJobs);
      // While Generate or the Assistant waits for a video model, notice one being loaded. A host that cannot generate at
      // all (no media subprocesses, as in the TensorAgent app) says so once; polling it would never succeed.
      if (++S.ticks % 3 === 0 && (S.tab === 'generate' || S.tab === 'assistant') && document.visibilityState !== 'hidden' && generationGate() == null && !videoModels().some(m => m.available))
        await background(refreshCapabilities).catch(() => { /* The last known capabilities stay. */ });
    } catch (e) { status(tr('studio.connection', 'Connection interrupted. Project and jobs remain on the host. {error}', { error: e.message }), true); } finally { if (!S.root.hidden) S.timer = setTimeout(poll, 2000); }
  }
  function renderProject() {
    if (!S.root) return; setBusy(S.busy);
    if (!S.project) {
      ui.library.replaceChildren(el('h3', '', tr('studio.libraryTitle', 'Asset library')), el('p', 'ts-empty', tr('studio.libraryEmpty', 'Your imported media and generated takes appear here.')));
      ui.preview.replaceChildren(el('div', 'ts-welcome'));
      const welcome = ui.preview.firstChild; welcome.append(el('p', '', tr('studio.createDescription', 'Create a local project, import footage, and build a timeline. Editing works without an AI model.')));
      renderGuide(welcome); welcome.appendChild(button(tr('studio.createProject', 'Create project'), newProject));
      ui.inspector.replaceChildren(); ui.timeline.replaceChildren(); return;
    }
    ui.projectSelect.value = S.project.id;
    renderLibrary(); buildPreview(); renderTimeline(); renderInspector(); updatePlayback();
  }
  // Returns the imported asset's id: the new one, or, when the host kept one asset per content, the existing one.
  async function importFile(file) {
    if (!S.project) return null; const projectId = S.project.id, before = new Set(S.project.assets.map(a => a.id)); status(tr('studio.importing', 'Importing and probing {name}…', { name: file.name }));
    const value = await response(await fetch(API + '/projects/' + encodeURIComponent(projectId) + '/import?fileName=' + encodeURIComponent(file.name), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream', 'X-TensorSharp-Studio': '1' }, body: file }));
    if (!S.project || S.project.id !== projectId) return null;
    if (value.project || value.tracks) receiveProject(value); else await refreshProject(); renderProject(); status(tr('studio.imported', 'Media imported. Add it to any compatible track.'));
    const added = S.project.assets.filter(a => !before.has(a.id)); if (added.length) return added[added.length - 1].id;
    try { // a duplicate comes back as the existing asset: find it by content (the host stores SHA-256), then by name
      const digest = new Uint8Array(await window.crypto.subtle.digest('SHA-256', await file.arrayBuffer()));
      const hex = Array.from(digest, b => b.toString(16).padStart(2, '0')).join(''), same = S.project.assets.find(a => a.sha256 === hex); if (same) return same.id;
    } catch (_) { /* No SubtleCrypto outside a secure context. */ }
    const named = S.project.assets.filter(a => a.name === file.name).pop(); return named ? named.id : null;
  }
  async function importChat(url) {
    const safe = localMediaUrl(url); if (!safe) throw new Error(tr('studio.localOnly', 'Studio imports only media served by this host. Remote URLs are not fetched.'));
    const projectId = S.project.id; await api('assets/import-chat', { projectId, url: safe }); if (S.project.id === projectId) await refreshProject(); status(tr('studio.chatImported', 'Chat media is available in this project’s asset library.'));
  }
  // Library and picker tiles use the host's small thumbnail when it can make one (a desktop media decoder); a photo is
  // otherwise shown from its own file. Library images are reused across re-renders, so selecting a clip reloads none.
  const thumbs = new Map();
  function thumbUrl(a) {
    if (!a || a.missing || a.kind === 'audio') return null;
    if (S.capabilities && S.capabilities.externalProcesses) return API + '/thumbnail/' + encodeURIComponent(S.project.id) + '/' + encodeURIComponent(a.id);
    return a.kind === 'image' ? mediaUrl(a.id) : null;
  }
  function thumbnail(a, cls, reuse) {
    const url = thumbUrl(a); if (!url) return null; const cached = reuse && thumbs.get(url); if (cached) return cached;
    const img = el('img', cls || 'ts-thumbnail'); img.loading = 'lazy'; img.decoding = 'async'; img.alt = ''; img.draggable = false; img.width = 320; img.height = 180; img.src = url;
    img.addEventListener('error', () => { img.hidden = true; if (reuse && thumbs.get(url) === img) thumbs.delete(url); }); if (reuse) thumbs.set(url, img); return img;
  }
  // Whether this host can generate at all; a missing video model is not a reason to hide the way in.
  function generationGate() { return S.capabilities ? S.capabilities.generationUnavailableReason || null : tr('studio.capabilitiesUnknown', 'Studio could not read this host’s capabilities.'); }
  function renderLibrary() {
    ui.library.replaceChildren(); const head = el('div', 'ts-section-heading'); const importButton = button(tr('studio.import', 'Import'), () => ui.import.click()); importButton.disabled = S.capabilities && S.capabilities.importAvailable === false;
    head.append(el('h3', '', tr('studio.libraryTitle', 'Asset library')), importButton); ui.library.appendChild(head);
    if (importButton.disabled) ui.library.appendChild(el('p', 'ts-hint', tr('studio.importUnavailable', 'Importing media isn’t available on this host: {reason}', { reason: S.capabilities.renderUnavailableReason || tr('studio.importUnsupported', 'Import probing is unavailable on this host.') })));
    ui.library.appendChild(el('p', 'ts-hint', tr('studio.assetHint', 'Drag assets to a track, or use Add. Removing a clip preserves its source.')));
    if (!S.project.assets.length) ui.library.appendChild(el('p', 'ts-empty', tr('studio.noAssets', 'Import video, images, or audio to begin.')));
    const frames = generationGate() == null;
    for (const a of S.project.assets) {
      const card = el('article', 'ts-asset'); card.dataset.missing = String(!!a.missing); card.dataset.highlight = String(S.highlight === a.id); card.draggable = !a.missing;
      card.addEventListener('dragstart', e => { e.dataTransfer.setData('application/x-tensorsharp-asset', a.id); e.dataTransfer.effectAllowed = 'copy'; });
      const image = !a.missing && thumbnail(a, 'ts-thumbnail', true); if (image) card.appendChild(image);
      const request = a.generation && a.generation.request;
      const kind = { image: tr('studio.kindImage', 'photo'), video: tr('studio.kindVideo', 'video'), audio: tr('studio.kindAudio', 'audio') }[a.kind] || a.kind;
      card.append(el('strong', '', a.name), el('p', 'ts-hint', kind + ' · ' + (a.durationSeconds ? a.durationSeconds.toFixed(2) + 's' : a.width + '×' + a.height) + (a.generation ? ' · ' + tr('studio.generated', 'Generated') : '')));
      if (request && request.prompt) card.appendChild(el('p', 'ts-asset-prompt', request.prompt));
      if (a.variableFrameRate) card.appendChild(el('p', 'ts-hint', tr('studio.vfr', 'Variable frame rate: output sampled by timestamps.')));
      if (a.missing) {
        card.appendChild(el('p', 'ts-hint', tr('studio.missing', 'Source media missing. Relink the original file to preserve timeline references.')));
        const relink = el('input'); relink.type = 'file'; relink.accept = 'video/*,audio/*,image/*'; relink.hidden = true;
        const projectId = S.project.id;
        relink.addEventListener('change', () => run(async () => {
          const file = relink.files[0]; if (!file) return;
          const value = await response(await fetch(API + '/projects/' + encodeURIComponent(projectId) + '/assets/' + encodeURIComponent(a.id) + '/relink', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream', 'X-TensorSharp-Studio': '1' }, body: file }));
          if (S.project.id === projectId) { receiveProject(value); renderProject(); status(tr('studio.relinked', 'Original media restored. The host verified its content hash.')); }
        })); card.append(relink, button(tr('studio.relink', 'Relink original file'), () => relink.click()));
      }
      const actions = row(card); const add = button(tr('studio.addToTimeline', 'Add to timeline'), () => insertAsset(a.id), 'ts-primary-soft'); add.disabled = !!a.missing;
      add.setAttribute('aria-label', tr('studio.addToTimelineNamed2', 'Add to timeline: {name}', { name: a.name })); actions.appendChild(add);
      if (a.kind === 'image' && !a.missing && frames) {
        // A photo becomes a frame of a new take in one tap; Generate says if the loaded model cannot use that frame.
        const use = el('div', 'ts-asset-frames'); use.appendChild(el('span', 'ts-hint', tr('studio.useInTake', 'New take from this photo:')));
        for (const slot of ['first', 'last']) {
          const b = button(slot === 'first' ? tr('studio.asFirstFrame', 'As first frame') : tr('studio.asLastFrame', 'As last frame'), () => useFrame(a.id, slot));
          b.setAttribute('aria-label', slot === 'first' ? tr('studio.asFirstFrameNamed2', 'As first frame: {name}', { name: a.name }) : tr('studio.asLastFrameNamed2', 'As last frame: {name}', { name: a.name }));
          use.appendChild(b);
        }
        card.appendChild(use);
      }
      const more = el('details', 'ts-more'); more.appendChild(el('summary', '', tr('studio.more', 'More'))); const extra = row(more);
      extra.appendChild(button(tr('studio.previewSource', 'Preview'), () => previewSource(a)));
      if (clip() && !a.missing) extra.appendChild(button(tr('studio.replaceSelected', 'Replace the selected clip with this'), () => change([{ type: 'replaceAsset', clipId: S.selected, assetId: a.id }])));
      if (a.generation) extra.appendChild(button(tr('studio.anotherVersion', 'Make another version'), () => { S.loadTake = a; showPanel('inspector', 'generate'); focusKey(ui.inspector, 'prompt'); }));
      if (a.hasAudio && !a.missing && S.capabilities && S.capabilities.externalProcesses) extra.appendChild(button(tr('studio.waveform', 'Waveform'), () => showWaveform(card, a)));
      const remove = button(tr('studio.deleteAsset', 'Delete asset'), () => deleteAsset(a.id), 'ts-danger');
      remove.setAttribute('aria-label', tr('studio.deleteAssetNamed', 'Delete asset: {name}', { name: a.name })); extra.appendChild(remove);
      card.appendChild(more);
      ui.library.appendChild(card);
    }
  }
  function confirmAssetDeletion(a, usageCount) {
    return confirmRemoval(tr('studio.deleteAssetTitle', 'Delete “{name}”?', { name: a.name }), usageCount
      ? tr('studio.deleteAssetUsed', 'This asset is used by {count} timeline clip(s). Deleting it will also remove those clips. Other assets and their clips will stay.', { count: usageCount })
      : tr('studio.deleteAssetUnused', 'This asset is not used on the timeline. It will be removed from this project’s library.'), usageCount > 0,
      tr('studio.deleteAssetRetention', 'You can Undo this change. Original files and managed media are retained for Undo and existing jobs; this does not free disk space.'),
      tr('studio.deleteAsset', 'Delete asset'), ui.library);
  }
  function confirmTrackRemoval(t, clipCount) {
    return confirmRemoval(tr('studio.removeTrackTitle', 'Remove track “{name}”?', { name: t.name }), clipCount
      ? tr('studio.removeTrackUsed', 'This track holds {count} clip(s). Removing it also removes those clips. Clips on other tracks stay.', { count: clipCount })
      : tr('studio.removeTrackEmpty', 'This track has no clips.'), clipCount > 0,
      tr('studio.removeTrackRetention', 'You can Undo this change. Assets stay in the library.'), tr('studio.removeTrack', 'Remove track'), ui.timeline);
  }
  // A WebView-safe modal (WKWebView has no window.confirm without a host delegate) that defaults to Cancel.
  // Focus returns to the trigger, or to the fallback region when a re-render replaced the trigger meanwhile.
  function confirmRemoval(titleText, warningText, warn, retentionText, confirmText, fallback) {
    return new Promise(resolve => {
      const previous = document.activeElement, root = S.root, backdrop = el('div', 'ts-name-dialog ts-delete-dialog'), dialog = el('div', 'ts-stack ts-delete-content');
      const title = el('h2', '', titleText); title.id = 'ts-delete-title-' + id();
      const warning = el('p', warn ? 'ts-delete-warning' : '', warningText); warning.id = 'ts-delete-warning-' + id();
      const retention = el('p', 'ts-hint', retentionText); retention.id = 'ts-delete-retention-' + id();
      dialog.setAttribute('role', 'alertdialog'); dialog.setAttribute('aria-modal', 'true'); dialog.setAttribute('aria-labelledby', title.id); dialog.setAttribute('aria-describedby', warning.id + ' ' + retention.id);
      dialog.append(title, warning, retention);
      const background = Array.from(root.children).map(node => ({ node, inert: node.inert })); for (const item of background) item.node.inert = true;
      let finished = false;
      const finish = value => { if (finished) return; finished = true; backdrop.remove(); for (const item of background) item.node.inert = item.inert; S.deleteDialog = null; if (previous && previous.isConnected) previous.focus(); else focusRegion(fallback); resolve(value); };
      S.deleteDialog = finish;
      const actions = row(dialog), cancel = button(tr('studio.cancel', 'Cancel'), () => finish(false)), confirm = button(confirmText, () => finish(true), 'ts-danger'); actions.append(cancel, confirm);
      backdrop.addEventListener('keydown', e => {
        e.stopPropagation();
        if (e.key === 'Escape') { e.preventDefault(); finish(false); }
        else if (e.key === 'Tab') {
          if (e.shiftKey && document.activeElement === cancel) { e.preventDefault(); confirm.focus(); }
          else if (!e.shiftKey && document.activeElement === confirm) { e.preventDefault(); cancel.focus(); }
          else if (document.activeElement !== cancel && document.activeElement !== confirm) { e.preventDefault(); cancel.focus(); }
        }
      });
      backdrop.appendChild(dialog); root.appendChild(backdrop); cancel.focus();
    });
  }
  async function deleteAsset(assetId) {
    if (!S.project || S.busy || S.deleteDialog) return;
    const a = asset(assetId); if (!a) return;
    const projectId = S.project.id, expectedRevision = S.project.revision, epoch = S.epoch;
    const usageCount = S.project.clips.filter(c => c.assetId === assetId).length;
    if (!await confirmAssetDeletion(a, usageCount)) return;
    // The confirmation authorizes only the project revision and usages the user reviewed.
    if (!S.project || S.project.id !== projectId || S.project.revision !== expectedRevision || S.epoch !== epoch || S.busy)
      throw new Error(tr('studio.deleteAssetChanged', 'The project changed while you reviewed deletion. Nothing was deleted. Review the asset and try again.'));
    const applied = await change([{ type: 'deleteAsset', assetId, removeTimelineClips: usageCount > 0 }]);
    if (applied && S.project && S.project.id === projectId) {
      status(tr('studio.assetDeleted', 'Asset removed from the library. Use Undo to restore it and any removed clips. Source files are preserved.'));
      focusRegion(ui.library);
    }
  }
  function focusRegion(region) { if (region && region.isConnected !== false) { region.tabIndex = -1; region.focus(); } }
  async function removeTrack(trackId) {
    if (!S.project || S.busy || S.deleteDialog) return;
    const t = S.project.tracks.find(t => t.id === trackId); if (!t) return;
    const projectId = S.project.id, expectedRevision = S.project.revision, epoch = S.epoch;
    const clipCount = S.project.clips.filter(c => c.trackId === trackId).length;
    // Even an empty track asks: the × sits beside ↑, and after a removal re-renders the headers,
    // the next track's × lands under the second click of a double-click.
    if (!await confirmTrackRemoval(t, clipCount)) return;
    // The confirmation authorizes only the project revision and clips the user reviewed.
    if (!S.project || S.project.id !== projectId || S.project.revision !== expectedRevision || S.epoch !== epoch || S.busy) {
      focusRegion(ui.timeline);
      throw new Error(tr('studio.removeTrackChanged', 'The project changed while you reviewed removal. Nothing was removed. Review the track and try again.'));
    }
    const applied = await change([{ type: 'removeTrack', trackId }]);
    if (applied && S.project && S.project.id === projectId) {
      status(tr('studio.trackRemoved', 'Track removed. Use Undo to restore it and any clips it held. Source assets stay in the library.'));
      focusRegion(ui.timeline);
    }
  }
  async function insertAsset(assetId, trackId, atFrame) {
    const a = asset(assetId); if (!a) return; const kind = a.kind === 'audio' ? 'audio' : 'video'; const commands = [];
    if (!trackId) { const existing = S.project.tracks.find(t => t.kind === kind); if (existing) trackId = existing.id; else { trackId = id(); commands.push({ type: 'addTrack', trackId, kind, name: kind === 'audio' ? tr('studio.audioTrack', 'Audio') : tr('studio.videoTrack', 'Video') }); } }
    const durationFrames = a.kind === 'image' ? frames(5) : Math.max(1, Math.floor(a.durationSeconds * fps()));
    const clipId = id(); commands.push({ type: 'insertClip', clipId, assetId, trackId, startFrame: atFrame == null ? S.playhead : atFrame, durationFrames, sourceStartFrame: 0 });
    await change(commands); selectClip(clipId, S.tab === 'generate' || S.tab === 'assistant');
  }
  async function showWaveform(card, a) {
    if (card.querySelector('.ts-waveform')) return;
    const projectId = S.project.id; const value = await response(await fetch(API + '/waveform/' + encodeURIComponent(projectId) + '/' + encodeURIComponent(a.id), { credentials: 'same-origin' }));
    if (!card.isConnected) return; const peaks = (value.peaks || []).slice(0, 512); if (!peaks.length) return;
    const canvas = el('canvas', 'ts-waveform'); canvas.width = 512; canvas.height = 64; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', tr('studio.waveformLabel', 'Audio amplitude waveform for {name}', { name: a.name }));
    const context = canvas.getContext('2d'); context.strokeStyle = '#79c9ad'; context.lineWidth = 1;
    for (let i = 0; i < peaks.length; i++) { const height = Math.max(1, Math.min(1, Math.abs(Number(peaks[i]) || 0)) * 30); context.beginPath(); context.moveTo(i + .5, 32 - height); context.lineTo(i + .5, 32 + height); context.stroke(); }
    card.appendChild(canvas);
  }
  function resetPreview() { if (ui.video) { ui.video.pause(); ui.video.removeAttribute('src'); ui.video.load(); ui.video.hidden = true; } hideStill(); if (ui.previewMessage) ui.previewMessage.hidden = false; S.previewRevision = null; S.previewJob = null; S.sourcePreview = false; S.sourcePreviewAssetId = null; S.awaitPreviewJob = null; }
  function hideStill() { if (ui.still) { ui.still.hidden = true; ui.still.removeAttribute('src'); } }
  function buildPreview() {
    if (ui.preview.querySelector('.ts-monitor')) { renderShots(); previewNotice(); return; }
    ui.preview.replaceChildren(); const monitor = el('div', 'ts-monitor'); ui.video = el('video'); ui.video.controls = true; ui.video.playsInline = true; ui.video.preload = 'metadata'; ui.video.hidden = true;
    ui.video.addEventListener('timeupdate', () => { if (!S.sourcePreview) { S.playhead = Math.round(ui.video.currentTime * fps()); updatePlayback(); } });
    ui.video.addEventListener('error', () => status(tr('studio.playbackError', 'This browser could not decode the media. Download the rendered output or try a compatible browser.'), true));
    // Until a preview plays here, the monitor shows where the project stands: the getting-started steps, then a hint.
    ui.previewMessage = el('div', 'ts-monitor-empty');
    ui.still = el('img', 'ts-monitor-still'); ui.still.alt = ''; ui.still.hidden = true; ui.still.decoding = 'async'; ui.still.draggable = false;
    monitor.append(ui.video, ui.still, ui.previewMessage); ui.preview.appendChild(monitor);
    const playback = el('div', 'ts-playback'); ui.playhead = el('input'); ui.playhead.type = 'range'; ui.playhead.min = 0; ui.playhead.step = 1; ui.playhead.setAttribute('aria-label', tr('studio.scrub', 'Timeline playhead in frames'));
    ui.playhead.addEventListener('input', () => seek(Number(ui.playhead.value))); ui.time = el('span', 'ts-time'); playback.append(ui.playhead, ui.time); ui.preview.appendChild(playback);
    ui.previewLabel = el('p', 'ts-hint'); ui.preview.appendChild(ui.previewLabel);
    const actions = row(ui.preview); ui.renderButton = button(tr('studio.renderPreview', 'Render preview'), () => renderJob(true), 'ts-primary'); ui.exportButton = button(tr('studio.export', 'Export MP4'), () => renderJob(false)); actions.append(ui.renderButton, ui.exportButton);
    ui.renderHint = el('p', 'ts-hint'); ui.renderHint.id = 'ts-render-hint'; ui.renderHint.hidden = true; ui.preview.appendChild(ui.renderHint);
    ui.shots = el('div', 'ts-shots'); ui.shots.setAttribute('aria-label', tr('studio.storyboard', 'Storyboard shots')); ui.preview.appendChild(ui.shots); renderShots(); previewNotice();
  }
  function renderAllowed() { return !!(S.capabilities && S.capabilities.renderAvailable); }
  function guideSteps() {
    const p = S.project, caps = S.capabilities || {}, making = !!p && S.jobs.some(j => j.kind === 'generate' && activeJob(j));
    const rendered = !!p && S.jobs.some(j => j.kind === 'render' && j.state === 'completed' && j.projectRevision === p.revision);
    return [
      { kind: 'add', done: !!p && p.assets.length > 0, title: tr('studio.guideAddTitle', 'Add footage'),
        text: making ? tr('studio.guideMaking', 'A take is being made. It appears in the library when it is ready.')
          : generationGate() == null ? tr('studio.guideAddText', 'Import video, photos or music, or generate a new take from a description.') : tr('studio.guideAddTextImport', 'Import video, photos or music.') },
      { kind: 'arrange', done: !!p && p.clips.length > 0, title: tr('studio.guideArrangeTitle', 'Arrange the timeline'), text: tr('studio.guideArrangeText', 'Press Add to timeline on a library item, or drag it onto a track.') },
      { kind: 'export', done: rendered, title: tr('studio.guideExportTitle', 'Preview and export'), text: tr('studio.guideExportText', 'Render a preview to watch your edit, then export an MP4.'),
        blocked: caps.renderAvailable === false ? caps.renderUnavailableReason || tr('studio.renderUnsupported', 'Local rendering is unavailable on this host.') : null }];
  }
  function renderGuide(container) {
    const caps = S.capabilities || {}, box = el('div', 'ts-guide'); box.appendChild(el('h2', '', tr('studio.guideTitle', 'Make your first video')));
    const list = el('ol', 'ts-guide-steps'); box.appendChild(list);
    guideSteps().forEach((step, i) => {
      const item = el('li', 'ts-guide-step'); item.dataset.done = String(step.done);
      const head = el('div', 'ts-guide-head'); head.append(el('span', 'ts-guide-mark', step.done ? '✓' : String(i + 1)), el('strong', '', step.title));
      if (step.done) head.appendChild(el('span', 'ts-visually-hidden', tr('studio.guideDone', 'Done')));
      item.append(head, el('p', 'ts-hint', step.text)); const actions = row(item);
      if (step.kind === 'add') {
        // Every action this host cannot take says why in visible text: touch screens have no hover titles.
        if (caps.importAvailable === false) item.appendChild(el('p', 'ts-hint', tr('studio.importUnavailable', 'Importing media isn’t available on this host: {reason}', { reason: caps.renderUnavailableReason || tr('studio.importUnsupported', 'Import probing is unavailable on this host.') })));
        else actions.appendChild(button(tr('studio.importMedia', 'Import media'), () => ui.import.click()));
        if (generationGate() == null) actions.appendChild(button(tr('studio.guideGenerate', 'Generate a take'), async () => { if (await ensureProject()) showPanel('inspector', 'generate'); }));
      } else if (step.kind === 'arrange' && S.project && S.project.assets.length && !step.done) actions.appendChild(button(tr('studio.guideShowLibrary', 'Show library'), () => showPanel('library')));
      else if (step.kind === 'export' && step.blocked) item.appendChild(el('p', 'ts-hint', step.blocked));
      else if (step.kind === 'export' && S.project && S.project.durationFrames && !step.done) actions.appendChild(button(tr('studio.renderPreview', 'Render preview'), () => renderJob(true), 'ts-primary'));
      if (!actions.children.length) actions.remove();
      list.appendChild(item);
    });
    container.appendChild(box);
  }
  function paintMonitorEmpty() {
    if (!ui.previewMessage) return;
    const key = JSON.stringify([S.project && S.project.id, S.project && S.project.durationFrames, guideSteps().map(step => [step.done, step.text, step.blocked]), S.capabilities && S.capabilities.importAvailable]);
    if (key === S.guideKey && ui.previewMessage.children.length) return; S.guideKey = key; ui.previewMessage.replaceChildren();
    if (guideSteps().some(step => !step.done)) renderGuide(ui.previewMessage);
    else ui.previewMessage.appendChild(el('p', 'ts-hint', tr('studio.previewEmpty', 'Build a timeline, then render its preview. Preview and export use the same composition.')));
  }
  function previewNotice() {
    if (!ui.previewLabel) return; paintMonitorEmpty();
    ui.previewLabel.textContent = S.sourcePreview ? tr('studio.sourcePreview', 'Source preview · timeline effects are not shown here.') : S.previewRevision == null ? tr('studio.previewNeedRender', 'No composition preview rendered yet.') : tr(S.previewRevision === S.project.revision ? 'studio.previewCurrent' : 'studio.previewStale', S.previewRevision === S.project.revision ? 'Composition preview · revision {revision}' : 'Preview is from revision {revision}. Render again to see your latest edits.', { revision: S.previewRevision });
    const unavailable = (S.capabilities || {}).renderUnavailableReason || tr('studio.renderUnsupported', 'Local rendering is unavailable on this host.');
    if (ui.renderHint) { ui.renderHint.textContent = renderAllowed() ? '' : unavailable; ui.renderHint.hidden = renderAllowed(); }
    for (const b of [ui.renderButton, ui.exportButton]) if (b) { if (renderAllowed()) b.removeAttribute('aria-describedby'); else b.setAttribute('aria-describedby', 'ts-render-hint'); }
    if (ui.renderButton) { ui.renderButton.disabled = !renderAllowed() || !S.project.durationFrames; ui.renderButton.title = renderAllowed() ? '' : unavailable; }
    if (ui.exportButton) { ui.exportButton.disabled = !renderAllowed() || !S.project.durationFrames; ui.exportButton.title = renderAllowed() ? '' : unavailable; }
  }
  function previewSource(a) {
    S.sourcePreview = true; S.sourcePreviewAssetId = a.id; ui.previewMessage.hidden = true;
    if (a.kind === 'image') { ui.video.pause(); ui.video.hidden = true; ui.still.alt = a.name; ui.still.src = mediaUrl(a.id); ui.still.hidden = false; }
    else { hideStill(); ui.video.src = mediaUrl(a.id); ui.video.hidden = false; }
    previewNotice();
  }
  function showRender(job) { S.previewJob = job.id; S.previewRevision = job.projectRevision; S.sourcePreview = false; S.sourcePreviewAssetId = null; hideStill(); ui.video.src = outputUrl(job); ui.video.hidden = false; ui.previewMessage.hidden = true; previewNotice(); }
  function renderShots() {
    if (!ui.shots || !ui.shots.isConnected) return; ui.shots.replaceChildren();
    for (const c of S.project.clips.filter(c => { const a = asset(c.assetId); return a && a.kind !== 'audio'; }).sort((a, b) => a.startFrame - b.startFrame)) {
      const a = asset(c.assetId); const b = button(a.name + ' · ' + seconds(c.durationFrames).toFixed(1) + 's', () => selectClip(c.id), 'ts-shot'); b.setAttribute('aria-pressed', String(S.selected === c.id)); ui.shots.appendChild(b);
    }
  }
  function seek(frame) { S.playhead = Math.max(0, Math.min(S.project.durationFrames, Math.round(frame))); if (ui.video && ui.video.src && !S.sourcePreview && Number.isFinite(ui.video.duration)) ui.video.currentTime = Math.min(seconds(S.playhead), ui.video.duration); updatePlayback(); }
  function updatePlayback() {
    if (!S.project || !ui.playhead) return; ui.playhead.max = Math.max(1, S.project.durationFrames); ui.playhead.value = S.playhead; ui.time.textContent = fmt(S.playhead);
    for (const line of ui.timeline.querySelectorAll('.ts-playhead')) line.style.left = (seconds(S.playhead) * S.zoom) + 'px';
  }
  async function renderJob(preview) {
    const job = await api('jobs/render', { projectId: S.project.id, expectedRevision: S.project.revision, requestId: id() });
    if (preview) S.awaitPreviewJob = job.id || (job.job && job.job.id); status(tr('studio.renderQueued', 'Render queued against the current project revision. You can continue editing.')); S.tab = 'jobs'; renderInspector(); await refreshJobs();
  }
  async function refreshJobs() {
    if (!S.project) return; const projectId = S.project.id; const value = await api('jobs/list', { projectId }); if (!S.project || S.project.id !== projectId) return;
    const next = Array.isArray(value) ? value : value.jobs || []; const prior = new Map(S.jobs.map(j => [j.id, j.state])); S.jobs = next;
    if (ui.jobsTab) ui.jobsTab.textContent = jobsTabText(); // the count stays current while another tab is open
    syncMobileTabs();
    const completed = next.filter(j => j.state === 'completed' && prior.get(j.id) !== 'completed' && j.kind === 'generate');
    if (completed.length && !S.busy) await refreshProject();
    // A take seen running that just finished is said out loud and marked in the library, so it is found without hunting.
    const ready = completed.filter(j => prior.has(j.id)).map(j => j.outputAssetId && asset(j.outputAssetId)).filter(Boolean).pop();
    if (ready) { S.highlight = ready.id; if (ui.library && S.project) { renderLibrary(); const card = ui.library.querySelector('[data-highlight="true"]'); if (card && card.scrollIntoView) card.scrollIntoView({ block: 'nearest' }); } status(tr('studio.takeReady', '“{name}” is ready in the library.', { name: ready.name })); }
    if (S.tab === 'jobs') renderInspector();
    const wanted = S.jobs.find(j => j.id === S.awaitPreviewJob && j.state === 'completed');
    if (wanted) { showRender(wanted); S.awaitPreviewJob = null; }
    if (!S.previewJob && !S.sourcePreview) { const latest = S.jobs.find(j => j.kind === 'render' && j.state === 'completed' && j.projectRevision === S.project.revision); if (latest) showRender(latest); }
    previewNotice(); // the guide's steps follow renders and takes as they finish
  }
  // Selecting a clip shows its properties, except while the person is generating or asking the assistant.
  function selectClip(clipId, keepTab) { S.selected = clipId; S.selectedCaption = null; if (!keepTab) S.tab = 'clip'; renderInspector(); renderTimeline(); renderLibrary(); renderShots(); }
  function snapped(frame, movingId) {
    frame = Math.max(0, Math.round(frame)); if (!S.snap) return frame;
    const targets = [0, S.playhead];
    for (const c of S.project.clips) if (c.id !== movingId) targets.push(c.startFrame, c.startFrame + c.durationFrames);
    const tolerance = Math.max(1, Math.round(8 * fps() / S.zoom)); let best = frame, distance = tolerance + 1;
    for (const target of targets) { const next = Math.abs(target - frame); if (next <= tolerance && next < distance) { best = target; distance = next; } }
    return best;
  }
  function renderTimeline() {
    const oldScroll = ui.timeline.querySelector('.ts-timeline-scroll'); const scrollLeft = oldScroll ? oldScroll.scrollLeft : 0; const scrollTop = oldScroll ? oldScroll.scrollTop : 0;
    ui.timeline.replaceChildren(); const toolbar = el('div', 'ts-timeline-toolbar'); toolbar.append(el('h3', '', tr('studio.timeline', 'Timeline')));
    toolbar.append(button(tr('studio.undo', 'Undo'), () => history('undo')), button(tr('studio.redo', 'Redo'), () => history('redo')),
      button(tr('studio.videoTrackAdd', '+ Video'), () => change([{ type: 'addTrack', name: tr('studio.videoTrack', 'Video'), kind: 'video' }])),
      button(tr('studio.audioTrackAdd', '+ Audio'), () => change([{ type: 'addTrack', name: tr('studio.audioTrack', 'Audio'), kind: 'audio' }])));
    const split = button(tr('studio.split', 'Split'), () => splitSelected()); split.disabled = !clip(); toolbar.appendChild(split);
    const remove = button(tr('studio.remove', 'Remove'), () => removeSelected()); remove.disabled = !clip(); toolbar.appendChild(remove);
    checked(toolbar, tr('studio.snap', 'Snap'), S.snap, v => { S.snap = v; });
    const zoom = el('input'); zoom.type = 'range'; zoom.min = 15; zoom.max = 220; zoom.value = S.zoom; zoom.setAttribute('aria-label', tr('studio.zoom', 'Timeline zoom'));
    zoom.addEventListener('input', () => { S.zoom = Number(zoom.value); renderTimeline(); }); toolbar.appendChild(zoom); ui.timeline.appendChild(toolbar);
    const scroll = el('div', 'ts-timeline-scroll'); scroll.setAttribute('aria-label', tr('studio.timeline', 'Timeline')); ui.timeline.appendChild(scroll);
    const duration = Math.max(10, seconds(S.project.durationFrames) + 4); const width = Math.min(200000, duration * S.zoom);
    const rulerRow = el('div', 'ts-ruler-row'); rulerRow.appendChild(el('div', 'ts-track-label', fps().toFixed(3).replace(/0+$/, '').replace(/\.$/, '') + ' fps'));
    const ruler = el('div', 'ts-ruler'); ruler.style.width = width + 'px'; const tickStep = Math.max(S.zoom < 30 ? 5 : 1, Math.ceil(duration / 1500));
    for (let second = 0; second < duration; second += tickStep) { const tick = el('span', 'ts-tick', second + 's'); tick.style.left = second * S.zoom + 3 + 'px'; ruler.appendChild(tick); }
    ruler.addEventListener('pointerdown', e => { seek(frames((e.clientX - ruler.getBoundingClientRect().left) / S.zoom)); }); rulerRow.appendChild(ruler); scroll.appendChild(rulerRow);
    const addPlayhead = lane => { const play = el('div', 'ts-playhead'); play.style.left = seconds(S.playhead) * S.zoom + 'px'; lane.appendChild(play); };
    addPlayhead(ruler);
    for (let index = 0; index < S.project.tracks.length; index++) {
      const t = S.project.tracks[index], trackRow = el('div', 'ts-track-row'), label = el('div', 'ts-track-label');
      const trackName = el('strong', '', t.name); trackName.title = t.name; label.appendChild(trackName);
      const mute = button(t.muted ? tr('studio.unmuteShort', 'M̸') : tr('studio.muteShort', 'M'), () => change([{ type: 'updateTrack', trackId: t.id, muted: !t.muted }])); mute.title = t.muted ? tr('studio.unmute', 'Unmute track') : tr('studio.mute', 'Mute track'); mute.setAttribute('aria-pressed', String(t.muted)); label.appendChild(mute);
      if (t.kind !== 'audio') { const visible = button(t.visible ? '◉' : '○', () => change([{ type: 'updateTrack', trackId: t.id, visible: !t.visible }])); visible.title = tr('studio.visibility', 'Toggle track visibility'); visible.setAttribute('aria-pressed', String(t.visible)); label.appendChild(visible); }
      const up = button('↑', () => change([{ type: 'reorderTrack', trackId: t.id, index: Math.max(0, index - 1) }])); up.title = tr('studio.trackUp', 'Move track earlier in composition'); up.disabled = index === 0; label.appendChild(up);
      const removeButton = button('×', () => removeTrack(t.id), 'ts-danger'); removeButton.title = tr('studio.removeTrack', 'Remove track'); removeButton.setAttribute('aria-label', tr('studio.removeTrackNamed', 'Remove track: {name}', { name: t.name })); label.appendChild(removeButton); trackRow.appendChild(label);
      const lane = el('div', 'ts-lane'); lane.dataset.trackId = t.id; lane.style.width = width + 'px'; lane.style.backgroundSize = S.zoom + 'px 100%';
      lane.addEventListener('click', e => { if (e.target === lane) seek(frames((e.clientX - lane.getBoundingClientRect().left) / S.zoom)); });
      lane.addEventListener('dragover', e => { if (Array.from(e.dataTransfer.types).includes('application/x-tensorsharp-asset')) e.preventDefault(); });
      lane.addEventListener('drop', e => { const assetId = e.dataTransfer.getData('application/x-tensorsharp-asset'); if (!assetId) return; e.preventDefault(); run(() => insertAsset(assetId, t.id, snapped(frames((e.clientX - lane.getBoundingClientRect().left) / S.zoom)))); });
      for (const c of S.project.clips.filter(c => c.trackId === t.id)) {
        const a = asset(c.assetId); const item = el('button', 'ts-clip', a ? a.name : tr('studio.missingAsset', 'Missing asset')); item.type = 'button'; item.dataset.kind = a ? a.kind : t.kind;
        item.style.left = seconds(c.startFrame) * S.zoom + 'px'; item.style.width = Math.max(14, seconds(c.durationFrames) * S.zoom) + 'px';
        item.setAttribute('aria-pressed', String(S.selected === c.id)); item.title = (a ? a.name : c.id) + '\n' + tr('studio.clipPosition', 'Start {start}; duration {duration}', { start: fmt(c.startFrame), duration: fmt(c.durationFrames) }); item.setAttribute('aria-label', item.title);
        item.appendChild(el('span', '', seconds(c.durationFrames).toFixed(2) + 's')); attachDrag(item, c); lane.appendChild(item);
      }
      addPlayhead(lane); trackRow.appendChild(lane); scroll.appendChild(trackRow);
    }
    if (S.project.captions.length) {
      const captionRow = el('div', 'ts-track-row'); captionRow.appendChild(el('div', 'ts-track-label', tr('studio.captions', 'Captions'))); const lane = el('div', 'ts-lane'); lane.style.width = width + 'px';
      for (const c of S.project.captions) { const item = button(c.text, () => { S.selectedCaption = c.id; S.selected = null; S.tab = 'clip'; renderInspector(); }, 'ts-clip'); item.dataset.kind = 'text'; item.style.left = seconds(c.startFrame) * S.zoom + 'px'; item.style.width = Math.max(14, seconds(c.durationFrames) * S.zoom) + 'px'; lane.appendChild(item); }
      addPlayhead(lane); captionRow.appendChild(lane); scroll.appendChild(captionRow);
    }
    if (!S.project.tracks.length) scroll.appendChild(el('p', 'ts-empty', tr('studio.emptyTimeline', 'Add video or audio tracks, then place media from the library.')));
    scroll.scrollLeft = scrollLeft; scroll.scrollTop = scrollTop;
  }
  function attachDrag(item, c) {
    let drag = null, suppressClick = false;
    item.addEventListener('click', e => { if (suppressClick) { suppressClick = false; e.preventDefault(); return; } selectClip(c.id); });
    item.addEventListener('pointerdown', e => {
      if (e.button !== 0 || S.busy) return; const bounds = item.getBoundingClientRect();
      drag = { pointer: e.pointerId, x: e.clientX, original: c.startFrame, start: c.startFrame, source: c.sourceStartFrame, duration: c.durationFrames,
        edge: e.clientX - bounds.left < 7 ? 'left' : bounds.right - e.clientX < 7 ? 'right' : 'move', changed: false, projectId: S.project.id, revision: S.project.revision };
      item.setPointerCapture(e.pointerId);
    });
    item.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.pointer || Math.abs(e.clientX - drag.x) < 4 && !drag.changed) return;
      drag.changed = true; const delta = frames((e.clientX - drag.x) / S.zoom);
      if (drag.edge === 'move') { drag.start = snapped(drag.original + delta, c.id); item.style.left = seconds(drag.start) * S.zoom + 'px'; }
      else if (drag.edge === 'right') { drag.duration = Math.max(1, snapped(c.startFrame + c.durationFrames + delta, c.id) - c.startFrame); item.style.width = Math.max(14, seconds(drag.duration) * S.zoom) + 'px'; }
      else { const shift = Math.max(-c.sourceStartFrame, -c.startFrame, Math.min(c.durationFrames - 1, snapped(c.startFrame + delta, c.id) - c.startFrame)); drag.start = c.startFrame + shift; drag.source = c.sourceStartFrame + shift; drag.duration = c.durationFrames - shift; item.style.left = seconds(drag.start) * S.zoom + 'px'; item.style.width = Math.max(14, seconds(drag.duration) * S.zoom) + 'px'; }
    });
    item.addEventListener('pointerup', e => {
      if (!drag || e.pointerId !== drag.pointer) return; const finished = drag; drag = null; item.releasePointerCapture(e.pointerId);
      if (!finished.changed) return; suppressClick = true; S.selected = c.id;
      if (S.project.id !== finished.projectId || S.project.revision !== finished.revision) { renderTimeline(); status(tr('studio.dragStale', 'The project changed during the drag. Review and move the clip again.'), true); return; }
      run(async () => { try {
        const op = finished.edge === 'move' ? { type: 'moveClip', clipId: c.id, startFrame: finished.start } : { type: 'trimClip', clipId: c.id, sourceStartFrame: finished.source, durationFrames: finished.duration, startFrame: finished.start };
        await change([op]);
      } finally { renderTimeline(); } });
    });
    item.addEventListener('pointercancel', () => { drag = null; renderTimeline(); });
  }
  async function splitSelected() { const c = clip(); if (!c) return; if (S.playhead <= c.startFrame || S.playhead >= c.startFrame + c.durationFrames) throw new Error(tr('studio.splitInside', 'Place the playhead inside the selected clip before splitting.')); await change([{ type: 'splitClip', clipId: c.id, atFrame: S.playhead }]); }
  async function removeSelected() { const c = clip(); if (c) { await change([{ type: 'removeClip', clipId: c.id }]); S.selected = null; renderInspector(); } }
  function renderInspector() {
    if (!S.project) return;
    if (S.background > 0 && editingInspector()) { S.inspectorDirty = true; return; } // no caret, IME or keyboard is lost to a poll
    // A re-render (a poll, a finished job, a capability change) keeps focus, caret and scroll where the person left them.
    const active = document.activeElement, keep = active && within(active, ui.inspector) && active.dataset && active.dataset.focusKey
      ? { key: active.dataset.focusKey, start: active.selectionStart, end: active.selectionEnd } : null;
    const scroll = S.renderedTab === S.tab ? ui.inspector.scrollTop : 0; S.renderedTab = S.tab; ui.inspector.replaceChildren();
    // Only a picker this render draws keeps its Escape layer; one from another tab or project is closed.
    dropLayer('picker'); if (S.picker && (S.tab !== 'generate' || S.picker.projectId !== S.project.id)) S.picker = null;
    const tabs = el('div', 'ts-tabbar');
    for (const pair of [['clip', tr('studio.properties', 'Properties')], ['generate', tr('studio.generate', 'Generate')], ['assistant', tr('studio.assistant', 'Assistant')], ['jobs', jobsTabText()]]) {
      const b = button(pair[1], () => {
        S.tab = pair[0]; renderInspector(); focusKey(ui.inspector, 'tab-' + pair[0]);
        if (pair[0] === 'generate' || pair[0] === 'assistant') run(() => refreshCapabilities({ stale: 3000 }));
        if (pair[0] === 'assistant') run(loadProposals);
      });
      b.dataset.focusKey = 'tab-' + pair[0]; b.setAttribute('aria-pressed', String(S.tab === pair[0])); tabs.appendChild(b);
      if (pair[0] === 'jobs') ui.jobsTab = b;
    }
    ui.inspector.appendChild(tabs); const body = el('div', 'ts-stack'); ui.inspector.appendChild(body);
    if (S.tab === 'generate') generationForm(body); else if (S.tab === 'assistant') assistantForm(body); else if (S.tab === 'jobs') jobsPanel(body); else properties(body);
    if (keep) {
      const n = focusKey(ui.inspector, keep.key);
      if (n && keep.start != null && n.setSelectionRange) try { n.setSelectionRange(keep.start, keep.end); } catch (_) { /* Not a text field. */ }
      if (!n || n.disabled) focusRegion(ui.inspector); // the focused control is gone: keep focus inside Studio
    }
    if (scroll) ui.inspector.scrollTop = scroll;
    syncMobileTabs();
  }
  function properties(body) {
    const c = clip(), caption = S.project.captions.find(c => c.id === S.selectedCaption);
    if (c) {
      const a = asset(c.assetId); body.append(el('h3', '', a ? a.name : c.id), el('p', 'ts-hint', tr('studio.frameHint', 'All edit values are integer project frames. Drag a clip to move it; drag its edges to trim.')));
      const grid = el('div', 'ts-fields'); body.appendChild(grid);
      const start = num(grid, tr('studio.start', 'Timeline start (frames)'), c.startFrame), source = num(grid, tr('studio.sourceStart', 'Source in (frames)'), c.sourceStartFrame), duration = num(grid, tr('studio.duration', 'Duration (frames)'), c.durationFrames, 1);
      const track = field(grid, tr('studio.track', 'Track'), c.trackId, 'select', S.project.tracks.map(t => ({ value: t.id, label: t.name })));
      const volume = num(grid, tr('studio.volume', 'Volume multiplier'), c.volume, 0, 4, .05), fadeIn = num(grid, tr('studio.fadeIn', 'Fade in (frames)'), c.fadeInFrames), fadeOut = num(grid, tr('studio.fadeOut', 'Fade out (frames)'), c.fadeOutFrames);
      const audioFadeOut = num(grid, tr('studio.audioFadeOut', 'Audio-only fade out (frames)'), c.audioFadeOutFrames || 0);
      const fit = field(grid, tr('studio.fit', 'Image fit'), c.fit, 'select', [{ value: 'contain', label: tr('studio.contain', 'Fit inside') }, { value: 'cover', label: tr('studio.cover', 'Fill / crop') }]);
      const linked = checked(body, tr('studio.linked', 'Apply timing edits to linked clips'), true);
      body.appendChild(button(tr('studio.apply', 'Apply changes'), () => change([
        { type: 'moveClip', clipId: c.id, trackId: track.value, startFrame: Number(start.value), applyToLinked: linked.checked },
        { type: 'trimClip', clipId: c.id, sourceStartFrame: Number(source.value), durationFrames: Number(duration.value), applyToLinked: linked.checked },
        { type: 'updateClip', clipId: c.id, volume: Number(volume.value), fadeInFrames: Number(fadeIn.value), fadeOutFrames: Number(fadeOut.value), audioFadeOutFrames: Number(audioFadeOut.value), fit: fit.value }
      ]), 'ts-primary'));
      const controls = row(body); controls.append(button(tr('studio.split', 'Split'), splitSelected), button(tr('studio.duplicate', 'Duplicate'), () => change([{ type: 'duplicateClip', clipId: c.id }])), button(tr('studio.remove', 'Remove'), removeSelected, 'ts-danger'));
      const previous = S.project.clips.find(other => other.id !== c.id && other.trackId === c.trackId && other.startFrame < c.startFrame && other.startFrame + other.durationFrames > c.startFrame && other.startFrame + other.durationFrames < c.startFrame + c.durationFrames);
      if (previous && (S.project.tracks.find(t => t.id === c.trackId) || {}).kind === 'video') {
        body.appendChild(button(tr('studio.dissolve', 'Dissolve from overlapping previous clip'), () => change([{ type: 'dissolve', clipId: c.id, otherClipId: previous.id }])));
        body.appendChild(el('p', 'ts-hint', tr('studio.dissolveHelp', 'The overlap becomes a picture and audio dissolve. Clip positions stay fixed. Fade controls remain editable.')));
      }
      const linkGroup = field(body, tr('studio.linkGroup', 'Linked edit group (optional)'), c.linkGroup || '');
      body.appendChild(button(tr('studio.saveLink', 'Set link group'), () => change([{ type: 'updateClip', clipId: c.id, linkGroup: linkGroup.value }])));
      body.appendChild(el('p', 'ts-hint', tr('studio.replaceHint', 'To swap this clip for another take while keeping its place, open More on that library item and choose Replace the selected clip with this. The original stays in the library.')));
      if (a && a.generation) { const details = el('details'); details.append(el('summary', '', tr('studio.provenance', 'Generation provenance')), el('pre', 'ts-provenance', JSON.stringify(a.generation, null, 2))); body.appendChild(details); }
    } else if (caption) captionForm(body, caption);
    else body.append(el('h3', '', tr('studio.noSelection', 'Project settings')), el('p', 'ts-hint', tr('studio.selectHint', 'Select a clip on the timeline to edit its timing, sound, or framing.')));
    const settings = el('details'); if (!c && !caption) settings.open = true; settings.appendChild(el('summary', '', tr('studio.projectSettings', 'Project and export settings'))); body.appendChild(settings);
    const name = field(settings, tr('studio.projectName', 'Name'), S.project.name); const size = el('div', 'ts-fields'); settings.appendChild(size); const width = num(size, tr('studio.width', 'Width'), S.project.width, 16, 4096), height = num(size, tr('studio.height', 'Height'), S.project.height, 16, 4096);
    settings.appendChild(button(tr('studio.saveSettings', 'Save settings'), () => change([{ type: 'settings', name: name.value, width: Number(width.value), height: Number(height.value) }])));
    settings.appendChild(button(tr('studio.vertical', 'Vertical 720 × 1280'), () => change([{ type: 'settings', width: 720, height: 1280 }])));
    settings.appendChild(el('p', 'ts-hint', tr('studio.compositionOrder', 'Tracks compose in list order, bottom layer first. Fades affect picture and sound. Timeline audio is mixed at export.')));
    const captionDetails = el('details'); captionDetails.appendChild(el('summary', '', tr('studio.addCaption', 'Add caption / title'))); captionForm(captionDetails, null); body.appendChild(captionDetails);
    if (S.capabilities && S.capabilities.packageSupported) {
      const packages = el('details'); packages.appendChild(el('summary', '', tr('studio.package', 'Portable project package')));
      packages.appendChild(button(tr('studio.packageExport', 'Download project package'), exportPackage));
      const input = el('input'); input.type = 'file'; input.accept = '.zip,.tsstudio'; input.setAttribute('aria-label', tr('studio.packageImport', 'Import project package'));
      input.addEventListener('change', () => run(async () => { if (input.files[0]) await importPackage(input.files[0]); })); packages.appendChild(input); body.appendChild(packages);
    }
  }
  function captionForm(body, existing) {
    if (S.capabilities && S.capabilities.captionAvailable === false) { body.appendChild(el('p', 'ts-hint', tr('studio.captionUnavailable', 'Caption rendering is unavailable on this host.'))); return; }
    const text = field(body, tr('studio.captionText', 'Caption text'), existing ? existing.text : '', 'textarea'); text.maxLength = 500;
    const grid = el('div', 'ts-fields'); body.appendChild(grid); const start = num(grid, tr('studio.start', 'Timeline start (frames)'), existing ? existing.startFrame : S.playhead), duration = num(grid, tr('studio.duration', 'Duration (frames)'), existing ? existing.durationFrames : frames(3), 1);
    if (S.capabilities && S.capabilities.captionCharacterSet) body.appendChild(el('p', 'ts-hint', String(S.capabilities.captionCharacterSet)));
    body.appendChild(button(existing ? tr('studio.saveCaption', 'Save caption') : tr('studio.addCaption', 'Add caption / title'), () => change([{ type: existing ? 'updateCaption' : 'addCaption', captionId: existing ? existing.id : id(), text: text.value, startFrame: Number(start.value), durationFrames: Number(duration.value) }])));
    if (existing) body.appendChild(button(tr('studio.deleteCaption', 'Remove caption'), async () => { await change([{ type: 'removeCaption', captionId: existing.id }]); S.selectedCaption = null; renderInspector(); }));
    body.appendChild(el('p', 'ts-hint', tr('studio.transcriptionUnavailable', 'Automatic transcription and word-based cuts are not yet available in Studio. Captions remain manually editable.')));
  }
  async function exportPackage() {
    const a = el('a'); a.href = API + '/projects/' + encodeURIComponent(S.project.id) + '/package'; a.download = S.project.name + '.tsstudio'; document.body.appendChild(a); a.click(); a.remove();
  }
  async function importPackage(file) {
    const value = await response(await fetch(API + '/packages/import?fileName=' + encodeURIComponent(file.name), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream', 'X-TensorSharp-Studio': '1' }, body: file }));
    receiveProject(value); S.selected = null; resetPreview(); renderProject(); await listProjects(); rememberProject();
  }
  // ---- Generate: describe the shot, then optionally start and/or end it on a photo ----------------------------------
  // What a request's images MEAN; the host validates the same rules for every client. First and last frames are each
  // optional, as on MiniMax-H3's FL2VA checkpoint (zero, one or two keyframes; sglang's fl2va [0], [-1], [0,-1]): none is
  // t2v, a first frame alone i2v, and a last frame, alone or with a first, fl2v. No blank image stands in for a missing
  // frame. Timed keyframes and identity references are other uses of photos, each with its own inputs.
  const GENERATION_MODES = ['t2v', 'i2v', 'fl2v', 'keyframes', 'ref'];
  // Takes saved before modes were explicit carry only their inputs; the host infers the same way.
  function inferMode(request) {
    return (request.referenceAssetIds || []).length ? 'ref' : (request.temporalConditions || []).length ? 'keyframes'
      : request.endConditioningAssetId ? 'fl2v' : request.conditioningAssetId ? 'i2v' : 't2v';
  }
  function useOf(mode) { return mode === 'ref' || mode === 'keyframes' ? mode : 'frames'; }
  function shotMode(d) { return d.use === 'frames' ? d.last ? 'fl2v' : d.first ? 'i2v' : 't2v' : d.use; }
  function imageAssets() { return S.project.assets.filter(a => a.kind === 'image' && !a.missing); }
  function imageAsset(assetId) { const a = assetId && asset(assetId); return a && a.kind === 'image' && !a.missing ? a : null; }
  function photoName(assetId) { const a = asset(assetId); return a ? a.name : tr('studio.photoMissing', 'a missing photo'); }
  // What a shot does, said by its inputs: mode codes stay in provenance and in the exact commands.
  function shotText(s) {
    const mode = s.mode || inferMode(s), first = s.conditioningAssetId || s.firstFrameAssetId, last = s.endConditioningAssetId || s.lastFrameAssetId;
    if (mode === 'ref') return tr('studio.shotReferences', 'Uses {count} reference photo(s)', { count: s.referenceCount != null ? s.referenceCount : (s.referenceAssetIds || []).length });
    if (mode === 'keyframes') return tr('studio.shotKeyframes', 'Hits {count} photo(s) at set moments', { count: s.guideCount != null ? s.guideCount : (s.temporalConditions || []).length });
    if (first && last) return tr('studio.shotFirstLast', '{first} → {last}', { first: photoName(first), last: photoName(last) });
    if (last) return tr('studio.shotEnds', 'Ends on {name}', { name: photoName(last) });
    if (first) return tr('studio.shotStarts', 'Starts on {name}', { name: photoName(first) });
    if (mode === 'fl2v') return tr('studio.shotEndsPhoto', 'Ends on a photo');
    if (mode === 'i2v') return tr('studio.shotStartsPhoto', 'Starts on a photo');
    return tr('studio.shotDescription', 'From your description');
  }
  function videoModels() { return S.capabilities && S.capabilities.models || []; }
  // The family a person recognizes; the host's display name is the checkpoint's file name.
  function familyName(m) {
    const family = String(m && m.family || '').toLowerCase();
    return family === 'minimax-h3' ? 'MiniMax-H3' : family === 'ltx-2.5' ? 'LTX 2.5' : family === 'wan' ? 'Wan' : m && (m.displayName || m.modelId) || '';
  }
  function modelDefaults(m) {
    return { width: m.defaultWidth || 512, height: m.defaultHeight || 512, frameCount: m.defaultFrameCount, fps: m.fixedFps || m.defaultFps || 24,
      steps: m.fixedSteps == null ? m.defaultSteps : m.fixedSteps, cfgScale: m.fixedCfgScale == null ? m.defaultCfgScale : m.fixedCfgScale, generateAudio: !!m.defaultGenerateAudio };
  }
  // A first frame is i2v, a last frame fl2v; a checkpoint with no text-only mode always starts on a photo (Wan2.2 I2V).
  function slotUsable(m, slot) { return (m.modes || []).includes(slot === 'first' ? 'i2v' : 'fl2v'); }
  function firstRequired(m) { return slotUsable(m, 'first') && !(m.modes || []).includes('t2v'); }
  function slotReason(m, slot) { const blocked = m.unavailableModes || {}; return (slot === 'first' ? blocked.i2v : blocked.fl2v) || tr('studio.slotUnsupported', 'The loaded model can’t use a photo here.'); }
  // The ways this family can use photos: frames always, keyframes and references when it runs them or says why not.
  function usesOffered(m) { const modes = m.modes || [], blocked = m.unavailableModes || {}; return ['frames', 'keyframes', 'ref'].filter(use => use === 'frames' || modes.includes(use) || blocked[use]); }
  function useSupported(m, use) { return use === 'frames' ? ['t2v', 'i2v', 'fl2v'].some(mode => (m.modes || []).includes(mode)) : (m.modes || []).includes(use); }
  function defaultUse(m) { return !slotUsable(m, 'first') && !slotUsable(m, 'last') && useSupported(m, 'ref') ? 'ref' : 'frames'; }
  function secondsLabel(frames, fps) { return (Number(frames) / Math.max(1, Number(fps) || 24)).toFixed(1); }
  function shotFormKey() { return 'tensorsharp.studio.shot.' + S.project.id; }
  // A convenience copy in this browser: the shot form also survives a WebView reload within one launch.
  function saveShotForm(d) { try { window.localStorage.setItem(shotFormKey(), JSON.stringify(d)); } catch (_) { /* Storage can be disabled. */ } }
  function storedShotForm() {
    try { const v = JSON.parse(window.localStorage.getItem(shotFormKey()) || 'null'); return v && typeof v === 'object' && typeof v.prompt === 'string' && Array.isArray(v.guides) && Array.isArray(v.refs) ? v : null; }
    catch (_) { return null; }
  }
  function blankShotForm() { return { modelId: null, use: 'frames', first: '', last: '', guides: [], refs: [], prompt: '', negativePrompt: '', seed: -1, shotId: null, sizeSource: 'image', recipeNote: null, lengthNote: null }; }
  function referenceMinimum(d, m) {
    const minimum = m.referenceMinFrames || 0; if (d.use !== 'ref' || !(Number(d.frameCount) < minimum)) return;
    const offset = m.temporalOffset || 0, grid = m.temporalMultiple || 1;
    d.frameCount = offset + Math.ceil((minimum - offset) / grid) * grid;
    d.lengthNote = tr('studio.lengthRaised', 'Length raised to {seconds} s: this model needs at least that for reference photos.', { seconds: secondsLabel(d.frameCount, d.fps) });
  }
  // The model's own size, length, rate, sampling and sound; what the shot says and shows is kept.
  function resetSettings(d, m) {
    Object.assign(d, modelDefaults(m), { modelId: m.modelId, samplingRecipe: m.samplingRecipe || 'native', samplingRecipeName: m.samplingRecipeName || null, sizeSource: 'image', recipeNote: null, lengthNote: null });
    if (!useSupported(m, d.use)) d.use = defaultUse(m);
    referenceMinimum(d, m);
  }
  // A take reopens as its shot: prompt, photos and seed always, and its size, length and sampling on the same model.
  function newShotForm(m, take) {
    const request = take && take.generation && take.generation.request, d = blankShotForm();
    if (!request) { d.use = defaultUse(m); resetSettings(d, m); return d; }
    const mode = request.mode || inferMode(request);
    Object.assign(d, { use: useOf(mode), first: request.conditioningAssetId || '', last: request.endConditioningAssetId || '',
      guides: (request.temporalConditions || []).map(guide => Object.assign({}, guide)), refs: (request.referenceAssetIds || []).slice(),
      prompt: request.prompt || '', negativePrompt: request.negativePrompt || '', seed: request.seed == null ? -1 : request.seed, shotId: request.shotId || null });
    resetSettings(d, m);
    // A planned shot names no model (the assistant plans before one is loaded): its nonzero size and length still apply.
    if (!request.modelId) {
      const grid = m.spatialMultiple || 1, onGrid = v => Number(v) > 0 && Number(v) % grid === 0 && Number(v) <= (m.maxDimension || 4096);
      if (onGrid(request.width) && onGrid(request.height)) Object.assign(d, { width: request.width, height: request.height, sizeSource: 'custom' });
      if (Number(request.frameCount) > 0 && (request.frameCount - (m.temporalOffset || 0)) % (m.temporalMultiple || 1) === 0 && request.frameCount <= (m.maxFrames || 257)) d.frameCount = request.frameCount;
      if (request.generateAudio != null && m.audioOutput) d.generateAudio = !!request.generateAudio;
      referenceMinimum(d, m); return d;
    }
    if (request.modelId !== m.modelId) return d;
    // A take sampled under another recipe (a speed plug-in loaded or removed since) keeps its shot, not its steps, which
    // belonged to that recipe. Takes saved before recipes were sampled natively.
    const recipeChanged = (request.samplingRecipe || 'native') !== (m.samplingRecipe || 'native');
    // A started plan keeps its 0 = "model default" values; only real values replace the defaults.
    for (const key of ['width', 'height', 'frameCount', 'fps', 'cfgScale']) if (Number(request[key]) > 0) d[key] = request[key];
    if (m.fixedFps) d.fps = m.fixedFps; if (m.fixedCfgScale != null) d.cfgScale = m.fixedCfgScale;
    if (request.generateAudio != null) d.generateAudio = !!request.generateAudio;
    if (!recipeChanged && m.fixedSteps == null && Number(request.steps) > 0) d.steps = request.steps;
    if (recipeChanged) {
      const sampled = take.generation.effectiveParameters || {};
      d.recipeNote = tr('studio.samplingRecipeChanged', 'This take was sampled with {previous}; the loaded model samples differently, so its steps start from the model default ({steps}).',
        { previous: sampled.samplingRecipeName || sampled.recipe || request.samplingRecipe || 'native', steps: d.steps });
    }
    // A take made at the size its photo gave reopens that way; any other size was chosen, so it reopens as custom.
    if ((m.imageCanvasModes || []).includes(mode)) {
      const image = asset(request.conditioningAssetId || request.endConditioningAssetId), recipe = image && recipeCanvas(m, image.width, image.height);
      d.sizeSource = !(request.width > 0) || recipe && recipe.width === request.width && recipe.height === request.height ? 'image' : 'custom';
    }
    referenceMinimum(d, m);
    return d;
  }
  function shotFormFor(projectId) {
    let d = S.shotForms.get(projectId);
    if (!d) { const stored = storedShotForm(); d = stored ? Object.assign(blankShotForm(), stored) : blankShotForm(); S.shotForms.set(projectId, d); }
    return d;
  }
  // The shot form for the loaded model. A take or planned shot opened for a new version replaces it; another model keeps the
  // shot and takes its own settings.
  function loadedShotForm(m) {
    const take = S.loadTake || S.regenerateAsset && asset(S.regenerateAsset);
    if (take) {
      S.loadTake = null; S.regenerateAsset = null; const d = newShotForm(m, take); d.samplingRecipe = m.samplingRecipe || 'native'; d.samplingRecipeName = m.samplingRecipeName || null;
      d.shotKey = shotKey(d); dropMissingPhotos(d); S.shotForms.set(S.project.id, d); saveShotForm(d); return d;
    }
    const d = shotFormFor(S.project.id);
    if (d.modelId == null && !d.useChosen) d.use = defaultUse(m); // a new form opens on what the model is for (references on Ref2VA)
    if (d.modelId !== m.modelId) resetSettings(d, m);
    else if ((d.samplingRecipe || 'native') !== (m.samplingRecipe || 'native')) {
      // The same checkpoint with a speed plug-in loaded or removed: its steps belonged to the other recipe.
      d.steps = modelDefaults(m).steps;
      d.recipeNote = m.samplingRecipeName ? tr('studio.recipeLoaded', '{name} is now loaded, so the steps follow it ({steps}).', { name: m.samplingRecipeName, steps: d.steps })
        : tr('studio.recipeRemoved', '{previous} is no longer loaded, so the steps start from the model default ({steps}).', { previous: d.samplingRecipeName || d.samplingRecipe, steps: d.steps });
    }
    d.samplingRecipe = m.samplingRecipe || 'native'; d.samplingRecipeName = m.samplingRecipeName || null;
    // What the checkpoint fixes always wins over a value kept from earlier.
    if (m.fixedSteps != null) d.steps = m.fixedSteps; if (m.fixedCfgScale != null) d.cfgScale = m.fixedCfgScale; if (m.fixedFps) d.fps = m.fixedFps;
    dropMissingPhotos(d); return d;
  }
  // A photo deleted from the library (or whose source went missing) leaves its slot empty rather than being sent.
  function dropMissingPhotos(d) {
    for (const slot of ['first', 'last']) if (d[slot] && !imageAsset(d[slot])) d[slot] = '';
    d.refs = d.refs.filter(ref => imageAsset(ref)); for (const guide of d.guides) if (guide.assetId && !imageAsset(guide.assetId)) guide.assetId = '';
  }
  // A take's shot id is kept while the shot is the same (its description and photos), so a re-run is "take 2"; a changed
  // description or photo starts a new shot.
  function shotKey(d) { return JSON.stringify([d.prompt.trim(), d.use, d.first, d.last, d.refs, d.guides.map(g => g.assetId)]); }
  // From the library: a photo becomes the first or last frame of the next take, and Generate opens on it.
  function useFrame(assetId, slot) {
    if (!S.project) return; const d = shotFormFor(S.project.id);
    d.use = 'frames'; d.useChosen = true; d[slot] = assetId; saveShotForm(d); S.picker = null; dropLayer('picker');
    showPanel('inspector', 'generate');
    announce(slot === 'first' ? tr('studio.firstFrameSet', 'First frame: {name}', { name: photoName(assetId) }) : tr('studio.lastFrameSet', 'Last frame: {name}', { name: photoName(assetId) }));
    focusKey(ui.inspector, 'slot-' + slot + '-change');
  }
  // The canvas the model's recipe gives the shot's photo (the first frame, else the last), when this mode sizes from one.
  function imageCanvas(d, m) {
    if (!(m.imageCanvasModes || []).includes(shotMode(d)) || d.sizeSource !== 'image') return null;
    const image = imageAsset(d.first) || imageAsset(d.last); return image ? { image, size: recipeCanvas(m, image.width, image.height) } : null;
  }
  function shownSize(d, m) {
    const c = imageCanvas(d, m);
    return c ? { auto: true, image: c.image, width: c.size ? c.size.width : 0, height: c.size ? c.size.height : 0 } : { auto: false, width: Number(d.width), height: Number(d.height) };
  }
  function lengthOptions(m, d) {
    const grid = m.temporalMultiple || 1, offset = m.temporalOffset || 0, max = m.maxFrames || 257, values = [];
    for (let f = offset + grid; f <= max && values.length < 400; f += grid) values.push(f);
    if (!values.includes(Number(d.frameCount))) { values.push(Number(d.frameCount)); values.sort((a, b) => a - b); }
    return values.map(f => ({ value: String(f), label: tr('studio.lengthOption', '{seconds} s · {frames} frames', { seconds: secondsLabel(f, d.fps), frames: f }) }));
  }
  function shotRequest(d, m) {
    const mode = shotMode(d), size = shownSize(d, m), keyframes = mode === 'keyframes';
    const usesFirst = mode === 'i2v' || mode === 'fl2v' || keyframes, usesLast = mode === 'fl2v' || keyframes && !!m.lastFrameConditioning;
    return { projectId: S.project.id, requestId: id(), modelId: m.modelId, mode, prompt: d.prompt.trim(),
      // A checkpoint fixed at guidance 1 runs no negative branch, so it is never sent one it would refuse.
      negativePrompt: m.fixedCfgScale === 1 ? null : d.negativePrompt || null,
      // Auto sends 0 x 0, so the host's reading of the photo (EXIF orientation included) decides the exact size.
      width: size.auto ? 0 : Number(d.width), height: size.auto ? 0 : Number(d.height), frameCount: Number(d.frameCount), fps: Number(d.fps),
      steps: Number(d.steps), cfgScale: Number(d.cfgScale), seed: Number(d.seed), generateAudio: !!(m.audioOutput && d.generateAudio),
      conditioningAssetId: usesFirst && d.first || null, endConditioningAssetId: usesLast && d.last || null, shotId: d.shotId && d.shotKey === shotKey(d) ? d.shotId : id(),
      temporalConditions: keyframes ? d.guides.map(g => ({ assetId: g.assetId, frameIndex: Number(g.frameIndex), strength: Number(g.strength) })) : [],
      referenceAssetIds: mode === 'ref' ? d.refs.slice() : [],
      // The recipe the shown steps belong to, never the reused take's: the host refuses a shot planned for another.
      samplingRecipe: m.samplingRecipe || null };
  }
  // What stops this shot from being sent, and the control to fix it.
  function shotProblem(d, m) {
    const mode = shotMode(d), blocked = m.unavailableModes || {}, frames = Number(d.frameCount);
    if (!d.prompt.trim()) return { message: tr('studio.promptRequired', 'Describe the shot before generating.'), focus: 'prompt' };
    if (d.use === 'frames' && firstRequired(m) && !d.first) return { message: tr('studio.firstPhotoRequired', 'Choose the photo this take starts on: this model always starts on one.'), focus: 'slot-first-choose' };
    if (!(m.modes || []).includes(mode)) return { message: blocked[mode] || tr('studio.refuseMode', 'The loaded model can’t make this kind of shot.'), focus: null };
    if ((frames - (m.temporalOffset || 0)) % (m.temporalMultiple || 1) !== 0) return { message: tr('studio.temporalGrid', 'Frame count must be {multiple} × n + {offset}.', { multiple: m.temporalMultiple, offset: m.temporalOffset || 0 }), focus: 'length' };
    if (mode === 'keyframes' && !d.guides.length) return { message: tr('studio.guideRequired', 'Add at least one timed image keyframe.'), focus: 'guide-add' };
    if (mode === 'keyframes' && d.guides.some(g => !imageAsset(g.assetId) || !Number.isInteger(Number(g.frameIndex)) || Number(g.frameIndex) < 0 || Number(g.frameIndex) >= frames || !(Number(g.strength) > 0 && Number(g.strength) <= 1)))
      return { message: tr('studio.invalidGuides2', 'Each keyframe needs a photo, an output frame within the take, and a strength above 0 and at most 1.'), focus: null };
    if (mode === 'ref' && !d.refs.length) return { message: tr('studio.referenceRequired', 'Select at least one reference image.'), focus: 'refs' };
    if (mode === 'ref' && d.refs.length > (m.maxReferences || 0)) return { message: tr('studio.referenceLimit', 'This model accepts at most {count} reference images.', { count: m.maxReferences || 0 }), focus: 'refs' };
    if (mode === 'ref' && m.referenceMinFrames && frames < m.referenceMinFrames) return { message: tr('studio.referenceFrames', 'Reference generation with this model needs at least {frames} output frames.', { frames: m.referenceMinFrames }), focus: 'length' };
    return null;
  }
  // The host's refusal codes in plain words; its own text stays one tap away under Details.
  function plainRefusal(e) {
    const plain = { invalid_frame_count: tr('studio.refuseLength', 'This model can’t make a take of that length.'), invalid_resolution: tr('studio.refuseSize', 'This model can’t make a take of that size.'),
      invalid_sampling: tr('studio.refuseSampling', 'This model needs other sampling settings. Reset to model defaults, then try again.'), unsupported_mode: tr('studio.refuseMode', 'The loaded model can’t make this kind of shot.'),
      unsupported_conditioning: tr('studio.refusePhotos', 'The loaded model can’t use these photos this way.'), invalid_mode_inputs: tr('studio.refuseInputs', 'These photos don’t fit this kind of shot.'),
      model_unavailable: tr('studio.refuseModel', 'The video model isn’t ready.'), model_not_found: tr('studio.refuseModel', 'The video model isn’t ready.'),
      model_changed: tr('studio.refuseModelChanged', 'The video model changed. Studio re-read its settings: check them and generate again.'),
      invalid_fps: tr('studio.refuseSampling', 'This model needs other sampling settings. Reset to model defaults, then try again.'),
      unsupported_negative_prompt: tr('studio.refuseNegative', 'This model doesn’t use a negative prompt at this guidance. Clear it under Advanced.'),
      queue_full: tr('studio.refuseQueue', 'Too many takes are waiting. Try again when one finishes.'),
      unsupported_media: tr('studio.refusePhotoFormat', 'This photo format isn’t supported. Choose a JPEG, PNG or WebP.'),
      sampling_recipe_conflict: tr('studio.refuseRecipe', 'The loaded model now samples differently. Studio re-read its settings: review the steps and generate again.'),
      generation_output_unavailable: tr('studio.refuseMedia', 'This host can’t make video right now.'), invalid_generation: tr('studio.refuseGeneration', 'The model refused these settings.') };
    return plain[e.code] || e.message || String(e);
  }
  function notice(parent, headline, detail) {
    const box = el('div', 'ts-notice'); box.appendChild(el('strong', '', headline)); if (detail) box.appendChild(el('p', '', detail)); parent.appendChild(box); return box;
  }
  function abilitiesText(m) {
    const parts = [];
    if (slotUsable(m, 'first') && slotUsable(m, 'last')) parts.push(tr('studio.canStartEnd', 'start and/or end on a photo'));
    else if (slotUsable(m, 'first')) parts.push(tr('studio.canStart', 'start on a photo'));
    if ((m.modes || []).includes('keyframes')) parts.push(tr('studio.canKeyframes', 'hit photos at set moments'));
    if ((m.modes || []).includes('ref')) parts.push(tr('studio.canReferences', 'keep a person or object from reference photos'));
    if (m.audioOutput) parts.push(tr('studio.canAudio', 'make sound'));
    if (firstRequired(m)) return tr('studio.modelNeedsPhoto', 'Always starts on a photo you choose.');
    return parts.length ? tr('studio.modelCan', 'Makes takes from a description, and can {list}.', { list: parts.join(tr('studio.listSeparator', ', ')) }) : tr('studio.modelTextOnly', 'Makes takes from a description.');
  }
  function promptPlaceholder(d) {
    const mode = shotMode(d);
    return mode === 'ref' ? tr('studio.placeholderRef', 'The person from Picture 1 walks through a night market, lanterns overhead')
      : mode === 'keyframes' ? tr('studio.placeholderKeyframes', 'What happens between the photos, and how the camera moves')
      : mode === 'fl2v' && d.first ? tr('studio.placeholderBoth', 'What happens between the two photos')
      : mode === 'fl2v' ? tr('studio.placeholderLast', 'What happens before the take lands on the last photo')
      : mode === 'i2v' ? tr('studio.placeholderFirst', 'What moves in the photo, and how the camera moves')
      : tr('studio.placeholderText', 'A slow pan over waves at sunrise, warm light');
  }
  // Drops from the library (a desktop extra: every slot also opens the picker with a tap).
  function acceptDrops(target, accept) {
    target.addEventListener('dragover', e => { if (Array.from(e.dataTransfer.types).includes('application/x-tensorsharp-asset')) { e.preventDefault(); target.dataset.drop = 'true'; } });
    target.addEventListener('dragleave', () => { target.dataset.drop = 'false'; });
    target.addEventListener('drop', e => { target.dataset.drop = 'false'; const assetId = e.dataTransfer.getData('application/x-tensorsharp-asset'); if (!imageAsset(assetId)) return; e.preventDefault(); run(() => accept(assetId)); });
  }
  // The file sheet opens only from the tap itself (iOS), so nothing awaits before click().
  // The import re-renders the page, so the pick is applied to the form as it is then, never to the form that asked.
  function importPhoto(target) {
    const input = el('input'); input.type = 'file'; input.accept = 'image/png,image/jpeg,image/webp'; input.hidden = true; (S.root || document.body).appendChild(input);
    const projectId = S.project.id;
    input.addEventListener('change', () => run(async () => {
      const file = input.files && input.files[0]; input.remove(); if (!file) return;
      let assetId; try { assetId = await importFile(file); } catch (e) { e.message = plainRefusal(e); throw e; }
      if (!S.project || S.project.id !== projectId) return;
      if (imageAsset(assetId)) applyPick(target, assetId); else if (assetId) status(tr('studio.notAPhoto', 'That file was imported, but it isn’t a still photo.'), true);
    }));
    input.click();
  }
  // Puts a chosen photo where the picker was opened: a frame slot, a keyframe guide, or the reference list.
  function applyPick(target, assetId) {
    const d = shotFormFor(S.project.id); S.picker = null; dropLayer('picker');
    let key = 'tile-' + assetId;
    if (target === 'ref') { if (!d.refs.includes(assetId)) d.refs.push(assetId); }
    else if (/^guide:/.test(target)) { const guide = d.guides[Number(target.slice(6))]; if (guide) guide.assetId = assetId; key = target.replace(':', '-') + '-choose'; }
    else { d[target] = assetId; key = 'slot-' + target + '-change'; }
    saveShotForm(d); renderInspector(); focusKey(ui.inspector, key);
    announce(target === 'first' ? tr('studio.firstFrameSet', 'First frame: {name}', { name: photoName(assetId) }) : target === 'last' ? tr('studio.lastFrameSet', 'Last frame: {name}', { name: photoName(assetId) })
      : target === 'ref' ? tr('studio.referenceAdded', 'Picture {index}: {name}', { index: d.refs.indexOf(assetId) + 1, name: photoName(assetId) }) : photoName(assetId));
  }
  function generationForm(body) {
    const repaint = key => {
      if (ui.inspector && !within(body, ui.inspector)) { renderInspector(); if (key) focusKey(ui.inspector, key) || focusRegion(ui.inspector); return; }
      body.replaceChildren(); generationForm(body); if (key && !focusKey(body, key)) focusKey(body, 'frames-section') || focusRegion(body);
    };
    body.appendChild(el('h3', '', tr('studio.generateTitle', 'Generate a new take')));
    body.appendChild(el('p', 'ts-hint', tr('studio.generateIntro', 'The loaded video model makes it on this host. Finished takes appear in the library.')));
    const gate = generationGate(), models = videoModels();
    if (gate) {
      // A host that cannot run media subprocesses (the TensorAgent app) can never generate here: say so once, no how-to.
      notice(body, tr('studio.generationGated', 'Video generation isn’t available on this host.'), gate);
      if (!S.capabilities) body.appendChild(button(tr('studio.checkAgain', 'Check again'), () => refreshCapabilities()));
      return;
    }
    if (!models.length) {
      notice(body, tr('studio.noVideoModel', 'No video model is loaded.'), tr('studio.noVideoModelHelp', 'To make takes, go back to chat and choose a video model (MiniMax-H3, LTX 2.5 or Wan) in the model menu, then come back: Studio notices it by itself. To end a take on a photo, choose MiniMax-H3 or LTX 2.5.'));
      const kept = S.shotForms.get(S.project.id); if (kept && (kept.prompt || kept.first || kept.last)) body.appendChild(el('p', 'ts-hint', tr('studio.shotKept', 'Your description and photos are kept.')));
      const actions = row(body); actions.append(button(tr('studio.backChat', 'Back to chat'), close), button(tr('studio.checkAgain', 'Check again'), () => refreshCapabilities()));
      return;
    }
    // A take opened for a new version starts on the model that made it, when that model is loaded.
    const take = S.loadTake || S.regenerateAsset && asset(S.regenerateAsset), wanted = take && take.generation && take.generation.request && take.generation.request.modelId || (S.shotForms.get(S.project.id) || {}).modelId;
    const m = models.find(x => x.available && x.modelId === wanted) || models.find(x => x.available) || models[0];
    if (models.length > 1) {
      const select = field(body, tr('studio.model', 'Video model'), m.modelId, 'select', models.map(x => ({ value: x.modelId, label: familyName(x) + (x.samplingRecipeName ? ' · ' + x.samplingRecipeName : '') + (x.available ? '' : ' · ' + tr('studio.unavailable', 'unavailable')) })));
      select.addEventListener('change', () => { const next = models.find(x => x.modelId === select.value); if (next) { const d = shotFormFor(S.project.id); resetSettings(d, next); saveShotForm(d); repaint('model'); } });
      select.dataset.focusKey = 'model';
    }
    if (!m.available) { notice(body, tr('studio.modelUnavailable', '{name} can’t make takes right now.', { name: familyName(m) }), m.unavailableReason || tr('studio.notLoaded', 'Load this model in the host before generating.')); return; }
    const supported = m.modes || [];
    if (!supported.length) { body.appendChild(el('p', 'ts-empty', tr('studio.noModes', 'This checkpoint advertises no generation mode that Studio can run.'))); return; }
    const d = loadedShotForm(m);
    const modelLine = el('div', 'ts-gen-model'); body.appendChild(modelLine);
    modelLine.append(el('strong', '', tr('studio.videoModelLine', 'Video model: {name}', { name: familyName(m) + (m.samplingRecipeName ? ' · ' + m.samplingRecipeName : '') })), el('p', 'ts-hint', abilitiesText(m)));
    const file = el('p', 'ts-hint ts-gen-file', m.modelId); modelLine.appendChild(file);
    // A recipe that fixes the steps is named where its effect is seen: the steps in Advanced cannot change.
    if (m.samplingRecipeName) modelLine.appendChild(el('p', 'ts-hint', tr('studio.samplingRecipe', '{name} fixes the sampling: exactly {steps} steps on its own schedule.', { name: m.samplingRecipeName, steps: modelDefaults(m).steps })));
    if (d.recipeNote) body.appendChild(el('p', 'ts-hint', d.recipeNote));

    const prompt = field(body, tr('studio.prompt', 'Describe this shot'), d.prompt, 'textarea'); prompt.maxLength = 12000; prompt.dataset.focusKey = 'prompt'; prompt.placeholder = promptPlaceholder(d);
    prompt.addEventListener('input', () => { d.prompt = prompt.value; saveShotForm(d); paintSummary(); });

    const section = el('section', 'ts-gen-use'); body.appendChild(section);
    let guideRows = [];
    const setSlot = (slot, assetId) => {
      capture(); d[slot] = assetId || ''; S.picker = null; dropLayer('picker'); saveShotForm(d); repaint(assetId ? 'slot-' + slot + '-change' : 'slot-' + slot + '-choose');
      announce(assetId ? (slot === 'first' ? tr('studio.firstFrameSet', 'First frame: {name}', { name: photoName(assetId) }) : tr('studio.lastFrameSet', 'Last frame: {name}', { name: photoName(assetId) }))
        : slot === 'first' ? tr('studio.firstFrameRemoved', 'First frame removed') : tr('studio.lastFrameRemoved', 'Last frame removed'));
    };
    const openPicker = target => { capture(); d.refusal = null; S.picker = { projectId: S.project.id, target }; repaint('picker-heading'); };
    const pickerReturn = target => target === 'ref' ? 'refs-import' : /^guide:/.test(target) ? target.replace(':', '-') + '-choose' : 'slot-' + target + (d[target] ? '-change' : '-choose');
    const closePicker = () => { capture(); const target = S.picker && S.picker.target; S.picker = null; dropLayer('picker'); repaint(target && pickerReturn(target)); };
    function slotTile(slot) {
      const usable = slotUsable(m, slot), required = slot === 'first' && firstRequired(m), tile = el('div', 'ts-slot'); tile.dataset.slot = slot; tile.setAttribute('role', 'group');
      const title = el('span', 'ts-slot-title', slot === 'first' ? tr('studio.firstFrameTitle', 'First frame') : tr('studio.lastFrameTitle', 'Last frame')); title.id = 'ts-slot-' + slot + '-' + id(); tile.setAttribute('aria-labelledby', title.id);
      const tag = el('span', 'ts-slot-tag', !usable ? tr('studio.slotUnavailable', 'Not available') : required ? tr('studio.slotRequired', 'Required') : tr('studio.slotOptional', 'Optional')); tag.id = title.id + '-tag';
      const head = el('div', 'ts-slot-head'); head.append(title, tag); tile.appendChild(head);
      // A slot the model cannot use stays visible and says why, as text (a disabled control would be skipped by Tab).
      if (!usable) {
        tile.dataset.state = 'disabled'; tile.appendChild(el('p', 'ts-hint', slotReason(m, slot)));
        // A photo set here from the library (or kept from another model) is shown, so it can be seen and removed.
        if (imageAsset(d[slot])) { tile.appendChild(el('p', 'ts-slot-name', photoName(d[slot]))); const drop = button(tr('studio.removePhoto', 'Remove'), () => setSlot(slot, '')); drop.dataset.focusKey = 'slot-' + slot + '-change'; tile.appendChild(drop); }
        return tile;
      }
      const a = imageAsset(d[slot]), open = !!S.picker && S.picker.target === slot;
      if (!a) {
        tile.dataset.state = 'empty';
        const pick = button(tr('studio.choosePhoto', '+ Choose photo'), () => openPicker(slot), 'ts-slot-pick'); pick.dataset.focusKey = 'slot-' + slot + '-choose';
        pick.setAttribute('aria-label', slot === 'first' ? tr('studio.choosePhotoFirst', 'Choose photo for the first frame') : tr('studio.choosePhotoLast', 'Choose photo for the last frame'));
        pick.setAttribute('aria-describedby', tag.id); pick.setAttribute('aria-expanded', String(open)); acceptDrops(pick, assetId => setSlot(slot, assetId)); tile.appendChild(pick);
        return tile;
      }
      tile.dataset.state = 'filled';
      const frame = el('div', 'ts-slot-frame'), size = shownSize(d, m);
      // The picture is cropped to the take's shape, as the model will crop it.
      if (size.width > 0 && size.height > 0) frame.style.aspectRatio = size.width + ' / ' + size.height;
      const img = el('img', 'ts-slot-thumb'); img.alt = ''; img.decoding = 'async'; img.draggable = false; img.src = mediaUrl(a.id); frame.appendChild(img); acceptDrops(frame, assetId => setSlot(slot, assetId));
      tile.append(frame, el('p', 'ts-slot-name', a.name));
      if (size.width > 0 && size.height > 0 && a.width > 0 && a.height > 0 && Math.abs(a.width / a.height / (size.width / size.height) - 1) > .1)
        tile.appendChild(el('p', 'ts-hint', tr('studio.slotCropped', 'Edges outside the take’s frame are cut off.')));
      const actions = el('div', 'ts-slot-actions');
      const changeButton = button(tr('studio.changePhoto', 'Change'), () => openPicker(slot)); changeButton.dataset.focusKey = 'slot-' + slot + '-change'; changeButton.setAttribute('aria-expanded', String(open));
      changeButton.setAttribute('aria-label', slot === 'first' ? tr('studio.changeFirstNamed', 'Change first frame ({name})', { name: a.name }) : tr('studio.changeLastNamed', 'Change last frame ({name})', { name: a.name }));
      const removeButton = button(tr('studio.removePhoto', 'Remove'), () => setSlot(slot, ''));
      removeButton.setAttribute('aria-label', slot === 'first' ? tr('studio.removeFirstNamed', 'Remove first frame ({name})', { name: a.name }) : tr('studio.removeLastNamed', 'Remove last frame ({name})', { name: a.name }));
      actions.append(changeButton, removeButton); tile.appendChild(actions);
      return tile;
    }
    // The library's photos as tiles, inline under what they fill: on a phone the inspector is already an overlay.
    function pickerPanel(target, chosen, choose) {
      pushLayer('picker', closePicker);
      const panel = el('div', 'ts-picker'); panel.setAttribute('role', 'region');
      const heading = el('h4', '', target === 'first' ? tr('studio.pickFirst', 'Choose the first frame') : target === 'last' ? tr('studio.pickLast', 'Choose the last frame') : tr('studio.pickGuide', 'Choose a keyframe photo'));
      heading.id = 'ts-picker-' + id(); heading.tabIndex = -1; heading.dataset.focusKey = 'picker-heading'; panel.setAttribute('aria-labelledby', heading.id);
      const head = el('div', 'ts-picker-head'), cancel = button(tr('studio.cancel', 'Cancel'), closePicker); head.append(heading, cancel); panel.appendChild(head);
      const grid = el('div', 'ts-picker-grid'), importAllowed = !S.capabilities || S.capabilities.importAvailable !== false; panel.appendChild(grid);
      if (importAllowed) { const add = button(tr('studio.importPhoto', 'Import a photo…'), () => { capture(); importPhoto(target); }, 'ts-picker-tile ts-picker-import'); add.dataset.focusKey = 'picker-import'; grid.appendChild(add); }
      for (const a of imageAssets()) {
        const tile = button('', () => choose(a.id), 'ts-picker-tile'); tile.dataset.assetId = a.id; tile.dataset.focusKey = 'tile-' + a.id; tile.setAttribute('aria-label', a.name);
        if (a.id === chosen) tile.setAttribute('aria-current', 'true');
        const img = thumbnail(a, 'ts-picker-thumb'); if (img) tile.appendChild(img); tile.appendChild(el('span', 'ts-picker-name', a.name)); grid.appendChild(tile);
      }
      if (!imageAssets().length) panel.appendChild(el('p', 'ts-empty', importAllowed ? tr('studio.noPhotos', 'No photos in this project yet. Import one to use it here.')
        : tr('studio.noPhotosNoImport', 'This project has no photos, and importing isn’t available on this host.')));
      return panel;
    }
    function frameSlots(parent) {
      const slots = el('div', 'ts-slots'); slots.append(slotTile('first'), slotTile('last')); parent.appendChild(slots);
      if (slotUsable(m, 'first') && slotUsable(m, 'last') && (d.first || d.last)) {
        const swap = button('⇄ ' + tr('studio.swapFrames', 'Swap first and last frames'), () => {
          capture(); [d.first, d.last] = [d.last, d.first]; saveShotForm(d); repaint('swap');
          announce(tr('studio.swapped', 'Swapped. First frame: {first}. Last frame: {last}.', { first: d.first ? photoName(d.first) : tr('studio.none', 'None'), last: d.last ? photoName(d.last) : tr('studio.none', 'None') }));
        }, 'ts-swap'); swap.dataset.focusKey = 'swap'; parent.appendChild(swap);
      }
      if (S.picker && (S.picker.target === 'first' || S.picker.target === 'last')) { const target = S.picker.target; parent.appendChild(pickerPanel(target, d[target], assetId => { capture(); applyPick(target, assetId); })); }
    }
    if (d.use === 'frames') {
      const required = firstRequired(m);
      const heading = el('h4', '', required ? tr('studio.framesRequiredTitle', 'Start from a photo (required for this model)') : tr('studio.framesTitle', 'Start or end on a photo (optional)'));
      heading.tabIndex = -1; heading.dataset.focusKey = 'frames-section'; section.appendChild(heading);
      if (!required) section.appendChild(el('p', 'ts-hint', tr('studio.framesHelp', 'Skip this to make the take from your description alone.')));
      frameSlots(section);
      section.appendChild(el('p', 'ts-gen-result', firstRequired(m) && !d.first ? tr('studio.resultNeedsFirst', 'Choose the photo this take starts on.')
        : d.first && d.last ? tr('studio.resultBoth', 'The take goes from the first frame to the last frame.') : d.last ? tr('studio.resultLast', 'The take ends on the last frame.')
        : d.first ? tr('studio.resultFirst', 'The take starts on the first frame and brings it to life.') : tr('studio.resultNone', 'No photos: the take is made from your description alone.')));
      // Seen in validation: a last photo unlike the described scene is reached by a cut or dissolve at any length.
      if (d.last && slotUsable(m, 'last')) section.appendChild(el('p', 'ts-hint', tr('studio.lastFrameTip', 'Describe how the scene arrives at the last photo. A photo unlike the scene is usually reached by a cut or dissolve rather than a move.')));
    } else if (d.use === 'keyframes') {
      section.append(el('h4', '', tr('studio.keyframesTitle', 'Photos at set moments')), el('p', 'ts-hint', tr('studio.keyframesHelp', 'Each image guides a specific output frame. This is separate from identity references.')));
      if (slotUsable(m, 'first') || m.lastFrameConditioning) frameSlots(section);
      const guides = el('div', 'ts-stack'); section.appendChild(guides);
      d.guides.forEach((guide, index) => {
        const group = el('div', 'ts-guide-row'); group.setAttribute('role', 'group'); group.setAttribute('aria-label', tr('studio.guideNamed', 'Keyframe {index}', { index: index + 1 }));
        const a = imageAsset(guide.assetId), target = 'guide:' + index;
        const pick = button(a ? a.name : tr('studio.choosePhoto', '+ Choose photo'), () => openPicker(target), 'ts-slot-pick'); pick.dataset.focusKey = 'guide-' + index + '-choose';
        if (a) { const img = thumbnail(a, 'ts-guide-thumb'); if (img) pick.appendChild(img); }
        group.appendChild(pick);
        const grid = el('div', 'ts-fields'); group.appendChild(grid);
        const at = num(grid, tr('studio.guideFrame', 'Output frame index'), guide.frameIndex, 0, m.maxFrames - 1), strength = num(grid, tr('studio.guideStrength', 'Strength'), guide.strength, .05, 1, .05);
        guideRows.push({ at, strength, guide });
        group.appendChild(button(tr('studio.removeGuide', 'Remove guide'), () => { capture(); d.guides.splice(index, 1); S.picker = null; dropLayer('picker'); saveShotForm(d); repaint('guide-add'); }));
        guides.appendChild(group);
        if (S.picker && S.picker.target === target) guides.appendChild(pickerPanel(target, guide.assetId, assetId => { capture(); applyPick(target, assetId); }));
      });
      const addGuide = button(tr('studio.addGuide', 'Add image keyframe'), () => {
        if (d.guides.length >= 16) throw new Error(tr('studio.keyframeLimit', 'At most 16 temporal image guides can be submitted.'));
        capture(); d.guides.push({ assetId: '', frameIndex: Math.max(0, Number(d.frameCount) - 1), strength: 1 }); saveShotForm(d); openPicker('guide:' + (d.guides.length - 1));
      }); addGuide.dataset.focusKey = 'guide-add'; section.appendChild(addGuide);
    } else {
      section.append(el('h4', '', tr('studio.referencePhotosTitle', 'Reference photos')),
        el('p', 'ts-hint', tr('studio.referencesHelp2', 'Your photos show who or what to include; the take does not start or end on them. Choose up to {count}.', { count: m.maxReferences })));
      const grid = el('div', 'ts-picker-grid ts-refs'); grid.dataset.focusKey = 'refs'; grid.tabIndex = -1; section.appendChild(grid);
      const importAllowed = !S.capabilities || S.capabilities.importAvailable !== false;
      if (importAllowed) { const add = button(tr('studio.importPhoto', 'Import a photo…'), () => { capture(); importPhoto('ref'); }, 'ts-picker-tile ts-picker-import'); add.dataset.focusKey = 'refs-import'; grid.appendChild(add); }
      // Order is meaningful (MiniMax-H3 numbers them as pictures), so the tiles keep the order they were chosen in.
      for (const a of imageAssets()) {
        const order = d.refs.indexOf(a.id) + 1, tile = button('', () => {
          capture(); d.refs = d.refs.filter(ref => ref !== a.id); if (!order) d.refs.push(a.id); saveShotForm(d); repaint('tile-' + a.id);
          announce(order ? tr('studio.referenceRemoved', 'Removed {name}. Order: {list}', { name: a.name, list: d.refs.map((ref, i) => (i + 1) + '. ' + photoName(ref)).join(', ') || tr('studio.none', 'None') })
            : tr('studio.referenceAdded', 'Picture {index}: {name}', { index: d.refs.length, name: a.name }));
        }, 'ts-picker-tile'); tile.dataset.assetId = a.id; tile.dataset.focusKey = 'tile-' + a.id; tile.setAttribute('aria-pressed', String(!!order));
        tile.setAttribute('aria-label', order ? tr('studio.referenceTileOn', '{name}, picture {index}', { name: a.name, index: order }) : a.name);
        const img = thumbnail(a, 'ts-picker-thumb'); if (img) tile.appendChild(img); tile.appendChild(el('span', 'ts-picker-name', a.name));
        if (order) tile.appendChild(el('span', 'ts-ref-badge', String(order))); grid.appendChild(tile);
      }
      if (!imageAssets().length) section.appendChild(el('p', 'ts-empty', importAllowed ? tr('studio.noPhotos', 'No photos in this project yet. Import one to use it here.') : tr('studio.noPhotosNoImport', 'This project has no photos, and importing isn’t available on this host.')));
      if (d.refs.length) section.appendChild(el('p', 'ts-hint', tr('studio.referenceOrder', 'Order sent to the model: {list}', { list: d.refs.map((ref, i) => (i + 1) + '. ' + photoName(ref)).join(' · ') })));
      if (m.referenceMinFrames) section.appendChild(el('p', 'ts-hint', tr('studio.referenceFrames', 'Reference generation with this model needs at least {frames} output frames.', { frames: m.referenceMinFrames })));
    }
    const uses = usesOffered(m);
    if (uses.length > 1) {
      // Native radios: arrow keys, "1 of 3" and Tab order come from the browser and the global shortcuts leave them alone.
      // Once opened it stays open, so moving between the radios never hides the one that has focus.
      const other = el('details', 'ts-gen-other'); other.open = d.use !== 'frames' || !!d.otherOpen; other.addEventListener('toggle', () => { d.otherOpen = other.open; }); other.appendChild(el('summary', '', tr('studio.otherUses', 'Other ways to use photos'))); body.appendChild(other);
      const set = el('fieldset', 'ts-use'); set.appendChild(el('legend', '', tr('studio.useImagesAs', 'Use photos to…'))); other.appendChild(set);
      const labels = { frames: [tr('studio.useFrames', 'Start or end the take on a photo'), tr('studio.useFramesHelp', 'The take begins on your first photo, ends on your last, or both.')],
        keyframes: [tr('studio.useKeyframes', 'Hit photos at set moments'), tr('studio.useKeyframesHelp', 'Each photo appears at the moment you choose.')],
        ref: [tr('studio.useReferences', 'Keep someone or something the same'), tr('studio.useReferencesHelp', 'Your photos show who or what to include in a new scene; the take does not start or end on them.')] };
      for (const use of uses) {
        const option = el('label', 'ts-radio'), input = el('input'); input.type = 'radio'; input.name = 'ts-use-' + S.project.id; input.value = use; input.checked = d.use === use; input.dataset.focusKey = 'use-' + use;
        const ok = useSupported(m, use); input.disabled = !ok;
        const text = el('span', 'ts-radio-text'); text.append(el('strong', '', labels[use][0]), el('span', 'ts-hint', ok ? labels[use][1] : (m.unavailableModes || {})[use] || ''));
        option.append(input, text); set.appendChild(option);
        input.addEventListener('change', () => {
          if (!input.checked) return; capture(); d.use = use; d.useChosen = true; d.otherOpen = true; d.lengthNote = null; S.picker = null; dropLayer('picker');
          referenceMinimum(d, m); saveShotForm(d); repaint('use-' + use);
          // Photos chosen for another use are kept for when the person switches back, and the switch says so.
          const parked = use !== 'frames' && use !== 'keyframes' && (d.first || d.last) || use !== 'ref' && d.refs.length;
          announce(parked ? tr('studio.useSwitchedKept', '{use}. Photos chosen for the other ways are kept for when you switch back.', { use: labels[use][0] }) : labels[use][0]);
        });
      }
    }

    const settings = el('div', 'ts-fields'); body.appendChild(settings);
    const length = field(settings, tr('studio.length', 'Length'), String(d.frameCount), 'select', lengthOptions(m, d)); length.dataset.focusKey = 'length';
    const canvasMode = (m.imageCanvasModes || []).includes(shotMode(d)) && !!(imageAsset(d.first) || imageAsset(d.last));
    const sizeSource = canvasMode ? field(settings, tr('studio.size', 'Size'), d.sizeSource, 'select', [
      { value: 'image', label: tr('studio.sizeMatchPhoto', 'Match the photo (sharpest faces)') }, { value: 'custom', label: tr('studio.sizeCustom', 'Custom') }]) : null;
    if (sizeSource) { sizeSource.dataset.focusKey = 'size'; sizeSource.addEventListener('change', () => { capture(); d.sizeSource = sizeSource.value; saveShotForm(d); repaint('size'); }); }
    const shown = shownSize(d, m);
    const width = num(settings, tr('studio.width', 'Width'), shown.auto ? shown.width || '' : d.width, m.spatialMultiple, m.maxDimension, m.spatialMultiple);
    const height = num(settings, tr('studio.height', 'Height'), shown.auto ? shown.height || '' : d.height, m.spatialMultiple, m.maxDimension, m.spatialMultiple);
    width.disabled = height.disabled = shown.auto;
    if (shown.auto) body.appendChild(el('p', 'ts-hint', shown.width ? tr('studio.sizeFromPhotoHint', '{width} × {height}, matched to {name}: faces and hands keep their detail. About {ratio}× the work per frame of the {baseWidth} × {baseHeight} default.',
      { width: shown.width, height: shown.height, name: shown.image.name, ratio: (shown.width * shown.height / (modelDefaults(m).width * modelDefaults(m).height)).toFixed(1), baseWidth: modelDefaults(m).width, baseHeight: modelDefaults(m).height })
      : tr('studio.sizeFromPhotoUnknown', 'The host sizes the take from {name}.', { name: shown.image.name })));
    if (d.lengthNote) body.appendChild(el('p', 'ts-hint', d.lengthNote));
    // Outside the lengths the publisher trained, say what tends to happen; the choice stays the person's.
    const lengthHint = el('p', 'ts-hint'); body.appendChild(lengthHint);
    const paintLengthHint = () => {
      const sec = Number(d.frameCount) / Math.max(1, Number(d.fps) || 24), min = m.recommendedMinSeconds, max = m.recommendedMaxSeconds;
      lengthHint.textContent = min && sec < min ? (shotMode(d) === 'fl2v'
        ? tr('studio.lengthShortLast', '{name} is trained on {min}–{max} s clips. At {seconds} s a take has little time to reach its last photo and often cuts or dissolves to it.', { name: familyName(m), min, max, seconds: secondsLabel(d.frameCount, d.fps) })
        : tr('studio.lengthShort', '{name} is trained on {min}–{max} s clips; a {seconds} s take is a quick preview.', { name: familyName(m), min, max, seconds: secondsLabel(d.frameCount, d.fps) }))
        : max && sec > max ? tr('studio.lengthLong', '{name} is trained on {min}–{max} s clips; longer takes may lose coherence.', { name: familyName(m), min: min || 0, max }) : '';
      lengthHint.hidden = !lengthHint.textContent;
    };
    paintLengthHint();
    const audio = m.audioOutput ? checked(body, tr('studio.audioGenerate', 'Generate audio'), !!d.generateAudio, on => { d.generateAudio = on; saveShotForm(d); paintSummary(); }) : null;

    const advanced = el('details', 'ts-advanced'); advanced.appendChild(el('summary', '', tr('studio.advanced', 'Advanced settings'))); body.appendChild(advanced);
    const negative = m.fixedCfgScale === 1 ? null : field(advanced, tr('studio.negativePrompt', 'Negative prompt'), d.negativePrompt, 'textarea'); if (negative) negative.maxLength = 12000;
    const advancedFields = el('div', 'ts-fields'); advanced.appendChild(advancedFields);
    const steps = num(advancedFields, tr('studio.steps', 'Sampling steps'), d.steps, 1, 200); if (m.fixedSteps != null) { steps.value = m.fixedSteps; steps.disabled = true; }
    const cfg = num(advancedFields, tr('studio.cfg', 'Guidance scale'), d.cfgScale, 0, 30, .1); if (m.fixedCfgScale != null) { cfg.value = m.fixedCfgScale; cfg.disabled = true; }
    const rate = m.fixedFps ? null : num(advancedFields, tr('studio.fps', 'Frames per second'), d.fps, 1, 120);
    const seed = num(advanced, tr('studio.seed', 'Seed (−1 = random)'), d.seed, -1, Number.MAX_SAFE_INTEGER);
    if (m.deferredCapabilities && m.deferredCapabilities.length) advanced.appendChild(el('p', 'ts-hint', tr('studio.unsupportedCapabilities', 'Unavailable in Studio: {list}', { list: m.deferredCapabilities.join(', ') })));
    function capture() {
      d.prompt = prompt.value; if (negative) d.negativePrompt = negative.value; d.seed = Number(seed.value); d.frameCount = Number(length.value);
      if (!width.disabled) { d.width = Number(width.value); d.height = Number(height.value); }
      if (!steps.disabled) d.steps = Number(steps.value); if (!cfg.disabled) d.cfgScale = Number(cfg.value); if (rate) d.fps = Number(rate.value);
      if (audio) d.generateAudio = audio.checked;
      for (const r of guideRows) { r.guide.frameIndex = Number(r.at.value); r.guide.strength = Number(r.strength.value); }
      saveShotForm(d);
    }
    for (const input of [width, height, steps, cfg, seed, negative, ...guideRows.flatMap(r => [r.at, r.strength])].filter(Boolean)) input.addEventListener('input', () => { capture(); paintSummary(); });
    length.addEventListener('change', () => { capture(); d.lengthNote = null; paintSummary(); paintLengthHint(); });
    // A new frame rate relabels the lengths in place: a full redraw on blur would swallow the click that caused the blur.
    if (rate) { rate.dataset.focusKey = 'fps'; rate.addEventListener('change', () => {
      capture(); const chosen = length.value; length.replaceChildren();
      for (const o of lengthOptions(m, d)) { const item = el('option', '', o.label); item.value = o.value; length.appendChild(item); }
      length.value = chosen; paintSummary(); paintLengthHint();
    }); }

    // One reset replaces "Refresh model capabilities" and "Use model defaults": it re-reads the loaded model (and its
    // plug-ins), then applies that model's defaults, so they are never stale ones held since the form was drawn.
    const reset = button(tr('studio.resetDefaults', 'Reset to model defaults'), async () => {
      capture(); reset.disabled = true;
      try { await refreshCapabilities({ render: false }); } finally { reset.disabled = false; }
      const fresh = videoModels().find(x => x.available && x.modelId === d.modelId) || videoModels().find(x => x.available);
      if (fresh) resetSettings(d, fresh); saveShotForm(d); repaint('reset');
      announce(tr('studio.resetDone', 'Length, size and sampling are back to the model’s defaults.'));
    }); reset.dataset.focusKey = 'reset';
    body.append(reset, el('p', 'ts-hint', tr('studio.resetHint', 'Re-reads the loaded model, then puts length, size, frame rate, steps, guidance and sound back to its defaults. Your description, photos and seed stay.')));

    const footer = el('div', 'ts-gen-footer'); body.appendChild(footer);
    // The status line speaks the error; this box shows it beside the button and holds the host's Details.
    const error = el('div', 'ts-form-error'); error.id = 'ts-gen-error-' + id(); footer.appendChild(error);
    const summary = el('p', 'ts-gen-summary'); footer.appendChild(summary);
    function paintSummary() {
      const size = shownSize(d, m), frames = Number(d.frameCount);
      summary.textContent = [shotText({ mode: shotMode(d), conditioningAssetId: d.use !== 'ref' && d.first, endConditioningAssetId: d.use !== 'ref' && d.last, temporalConditions: d.guides, referenceAssetIds: d.refs }),
        tr('studio.summaryLength', '{seconds} s ({frames} frames)', { seconds: secondsLabel(frames, d.fps), frames }),
        size.auto && !size.width ? tr('studio.summarySizePhoto', 'sized from the photo') : size.width + ' × ' + size.height,
        audio && d.generateAudio ? tr('studio.summarySound', 'with sound') : ''].filter(Boolean).join(' · ');
    }
    paintSummary();
    const showError = (message, detail, focus) => {
      error.replaceChildren(el('p', '', message)); status(message, true);
      if (detail && detail !== message) { const more = el('details'); more.append(el('summary', '', tr('studio.details', 'Details')), el('p', 'ts-hint', detail)); error.appendChild(more); }
      if (focus) { const target = focusKey(body, focus); if (target) { target.setAttribute('aria-invalid', 'true'); target.setAttribute('aria-describedby', error.id); } }
    };
    if (d.refusal) showError(d.refusal.message, d.refusal.detail);
    const generate = button(tr('studio.generateTake', 'Generate take'), async () => {
      capture(); error.replaceChildren(); d.refusal = null;
      const problem = shotProblem(d, m); if (problem) { showError(problem.message, null, problem.focus); return; }
      for (const input of [width, height, steps, cfg, seed, rate].filter(Boolean)) if (!input.disabled && !input.checkValidity()) { input.reportValidity(); return; }
      const request = shotRequest(d, m); generate.disabled = true;
      try {
        await api('jobs/generate', request);
        // The next take of this form is another take of the same shot, numbered in the library.
        d.shotId = request.shotId; d.shotKey = shotKey(d); d.refusal = null; saveShotForm(d); S.tab = 'jobs'; await refreshJobs(); renderInspector(); focusKey(ui.inspector, 'tab-jobs');
        status(tr('studio.generationQueued', 'Generation queued. The finished take will appear in the library.'));
      } catch (e) {
        d.refusal = { message: plainRefusal(e), detail: e.message }; saveShotForm(d); showError(d.refusal.message, d.refusal.detail);
        if (MODEL_REFUSALS.includes(e.code)) refreshCapabilities().catch(() => { /* The refusal is already shown. */ });
      } finally { generate.disabled = false; }
    }, 'ts-primary'); generate.dataset.focusKey = 'generate'; footer.appendChild(generate);
  }
  const assistantState = new Map();
  // Capabilities change when the host loads another model or a plug-in changes; only a change redraws the page.
  let capabilityFetch = null;
  function capabilityKey(c) { return JSON.stringify(c ? [c.models || [], c.generationUnavailableReason || null, c.importAvailable, c.renderAvailable] : null); }
  async function refreshCapabilities(options) {
    options = options || {};
    if (options.stale && S.capabilities && Date.now() - S.capabilitiesAt < options.stale) return false;
    if (!capabilityFetch) capabilityFetch = api('capabilities').finally(() => { capabilityFetch = null; });
    const before = capabilityKey(S.capabilities), next = await capabilityFetch;
    S.capabilities = next; S.capabilitiesAt = Date.now();
    const changed = capabilityKey(next) !== before;
    if (changed && S.project) { if (options.render !== false && ui.inspector) renderInspector(); if (ui.library) renderLibrary(); previewNotice(); }
    return changed;
  }
  function boundedMessages(value) {
    return (Array.isArray(value) ? value : []).filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string').slice(-8).map(m => ({ role: m.role, content: m.content.slice(0, 12000) }));
  }
  function assistantFor(projectId) {
    if (!assistantState.has(projectId)) {
      let messages = []; try { messages = boundedMessages(JSON.parse(window.localStorage.getItem('tensorsharp.studio.chat.' + projectId) || '[]')); } catch (_) { }
      assistantState.set(projectId, { messages, text: messages.filter(m => m.role === 'assistant').at(-1)?.content || '', busy: false, controller: null, error: null, sessionId: null, conversationId: null });
    }
    return assistantState.get(projectId);
  }
  async function restoreAssistantHistory(projectId, conversationId, state) {
    if (!conversationId || state.historyLoaded || state.busy) return;
    if (state.historyLoad) return state.historyLoad;
    state.historyLoad = (async () => {
      try {
        const saved = await response(await fetch('/api/agent/conversations/' + encodeURIComponent(conversationId), { credentials: 'same-origin' }));
        state.messages = boundedMessages(saved.messages); state.text = state.messages.filter(m => m.role === 'assistant').at(-1)?.content || '';
      } catch (e) { state.error = e.message; }
      finally { state.historyLoaded = true; state.historyLoad = null; if (S.project && S.project.id === projectId && S.tab === 'assistant') renderInspector(); }
    })();
    return state.historyLoad;
  }
  async function assistantSession(projectId, state, expectedRevision) {
    if (state.sessionId) {
      if (state.conversationId && S.project.conversationId !== state.conversationId) {
        if (S.busy || S.project.revision !== expectedRevision) throw new Error(tr('studio.contextChanged', 'The project changed while preparing the assistant. Review your selection and send again.'));
        await change([{ type: 'settings', conversationId: state.conversationId }]); return S.project.revision;
      }
      return expectedRevision;
    }
    const conversation = S.project.conversationId || 'new';
    const created = await response(await fetch('/api/sessions?conversation=' + encodeURIComponent(conversation), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-TensorSharp-Studio': '1' }, body: '{}' }));
    if (!created.sessionId) throw new Error(tr('studio.sessionUnavailable', 'The host could not create a dedicated Studio chat session.'));
    state.sessionId = created.sessionId; state.conversationId = created.conversationId || null;
    if (state.conversationId && created.messages) { state.messages = boundedMessages(created.messages); state.historyLoaded = true; }
    if (!S.project || S.project.id !== projectId) throw new Error(tr('studio.projectChanged', 'The active project changed. The assistant request was not sent.'));
    if (S.busy || S.project.revision !== expectedRevision) throw new Error(tr('studio.contextChanged', 'The project changed while preparing the assistant. Review your selection and send again.'));
    if (state.conversationId && S.project.conversationId !== state.conversationId) await change([{ type: 'settings', conversationId: state.conversationId }]);
    return S.project.revision;
  }
  // Proposed commands in words, so a proposal can be reviewed without reading JSON (which stays under a disclosure).
  function clipName(clipId) { const c = S.project.clips.find(x => x.id === clipId), a = c && asset(c.assetId); return a ? a.name : tr('studio.aClip', 'a clip'); }
  function trackName(trackId) { const t = S.project.tracks.find(x => x.id === trackId); return t ? t.name : tr('studio.aTrack', 'a track'); }
  function timeText(frame) { const sec = Math.round(seconds(Number(frame) || 0) * 10) / 10; return tr('studio.secondsShort', '{seconds} s', { seconds: String(sec) }); }
  function describeCommand(c) {
    switch (c.type) {
      case 'addTrack': return tr('studio.cmdAddTrack', 'Add a {kind} track “{name}”', { kind: c.kind === 'audio' ? tr('studio.audioTrack', 'Audio') : tr('studio.videoTrack', 'Video'), name: c.name || '' });
      case 'removeTrack': return tr('studio.cmdRemoveTrack', 'Remove track “{name}” and its clips', { name: trackName(c.trackId) });
      case 'updateTrack': return c.muted != null ? tr(c.muted ? 'studio.cmdMuteTrack' : 'studio.cmdUnmuteTrack', c.muted ? 'Mute track “{name}”' : 'Unmute track “{name}”', { name: trackName(c.trackId) })
        : tr('studio.cmdUpdateTrack', 'Change track “{name}”', { name: trackName(c.trackId) });
      case 'reorderTrack': return tr('studio.cmdReorderTrack', 'Move track “{name}” to position {index}', { name: trackName(c.trackId), index: (c.index || 0) + 1 });
      case 'insertClip': return tr('studio.cmdInsertClip', 'Add “{name}” to “{track}” at {time}', { name: photoName(c.assetId), track: trackName(c.trackId), time: timeText(c.startFrame) });
      case 'trimClip': return tr('studio.cmdTrimClip', 'Trim “{name}” to {length}', { name: clipName(c.clipId), length: timeText(c.durationFrames) });
      case 'moveClip': return tr('studio.cmdMoveClip', 'Move “{name}” to {time}', { name: clipName(c.clipId), time: timeText(c.startFrame) });
      case 'splitClip': return tr('studio.cmdSplitClip', 'Split “{name}” at {time}', { name: clipName(c.clipId), time: timeText(c.atFrame) });
      case 'duplicateClip': return tr('studio.cmdDuplicateClip', 'Duplicate “{name}”', { name: clipName(c.clipId) });
      case 'removeClip': return tr('studio.cmdRemoveClip', 'Remove “{name}” from the timeline', { name: clipName(c.clipId) });
      case 'deleteAsset': return tr('studio.cmdDeleteAsset', 'Delete “{name}” from the library', { name: photoName(c.assetId) });
      case 'updateClip': {
        const parts = [];
        if (c.volume != null) parts.push(tr('studio.cmdVolume', 'volume {percent}%', { percent: Math.round(c.volume * 100) }));
        if (c.fadeInFrames) parts.push(tr('studio.cmdFadeIn', 'fade in over {length}', { length: timeText(c.fadeInFrames) }));
        if (c.fadeOutFrames) parts.push(tr('studio.cmdFadeOut', 'fade out over {length}', { length: timeText(c.fadeOutFrames) }));
        return tr('studio.cmdUpdateClip', 'Change “{name}”: {changes}', { name: clipName(c.clipId), changes: parts.join(', ') || tr('studio.cmdFraming', 'framing') });
      }
      case 'dissolve': return tr('studio.cmdDissolve', 'Dissolve from “{from}” into “{to}”', { from: clipName(c.otherClipId), to: clipName(c.clipId) });
      case 'replaceAsset': return tr('studio.cmdReplaceAsset', 'Replace “{name}” with “{asset}”', { name: clipName(c.clipId), asset: photoName(c.assetId) });
      case 'addCaption': return tr('studio.cmdAddCaption', 'Add the title “{text}” at {time} for {length}', { text: c.text || '', time: timeText(c.startFrame), length: timeText(c.durationFrames) });
      case 'updateCaption': return tr('studio.cmdUpdateCaption', 'Change a title');
      case 'removeCaption': return tr('studio.cmdRemoveCaption', 'Remove a title');
      case 'settings': return c.name ? tr('studio.cmdRename', 'Rename the project to “{name}”', { name: c.name }) : c.width ? tr('studio.cmdSize', 'Set the video size to {width} × {height}', { width: c.width, height: c.height }) : tr('studio.cmdSettings', 'Change project settings');
      default: return c.type;
    }
  }
  function withoutNulls(value) { return JSON.parse(JSON.stringify(value, (key, v) => v === null ? undefined : v)); }
  function assistantForm(body) {
    const state = assistantFor(S.project.id); body.appendChild(el('h3', '', tr('studio.assistant', 'Assistant')));
    if (S.project.conversationId && !state.historyLoaded && !state.busy) restoreAssistantHistory(S.project.id, S.project.conversationId, state);
    body.appendChild(el('p', 'ts-hint', tr('studio.assistantIntro', 'Ask in plain words. The assistant reads this project and suggests edits you review, then apply in one step (one Undo reverts them all). It can also plan shots to generate later, and start an export.')));
    // The host runs one model at a time: a loaded video model (the only kind capabilities list) means no chat model.
    const videoLoaded = videoModels().length > 0;
    if (videoLoaded) notice(body, tr('studio.assistantNeedsChat', 'The assistant needs a chat model, and a video model is loaded right now.'),
      tr('studio.assistantNeedsChatHelp', 'Go back to chat, choose a chat model, then come back. Your project stays as it is.'));
    else body.appendChild(el('p', 'ts-hint', tr('studio.assistantModelHint', 'Works with chat models that can edit projects (tool calling). If it replies without suggesting changes, try another chat model.')));
    const selected = clip();
    body.appendChild(el('p', 'ts-hint', selected ? tr('studio.assistantSelected', 'Selected: {name}', { name: (asset(selected.assetId) || {}).name || selected.id })
      : tr('studio.assistantNoSelection', 'Nothing selected: the assistant looks at the whole timeline.')));
    if (state.messages.length || state.busy) {
      const log = el('div', 'ts-chat'); body.appendChild(log);
      for (const message of state.messages.slice(-6)) { const bubble = el('div', 'ts-bubble', message.content); bubble.dataset.role = message.role; bubble.setAttribute('aria-label', (message.role === 'user' ? tr('studio.you', 'You') : tr('studio.assistant', 'Assistant')) + ': ' + message.content); log.appendChild(bubble); }
      // The reply in progress is written in place as it streams, so focus on Stop is not lost to a re-render.
      if (state.busy) { state.replyNode = el('div', 'ts-bubble', state.text || tr('studio.thinking', 'Working…')); state.replyNode.dataset.role = 'assistant'; log.appendChild(state.replyNode); }
    }
    // What the assistant just proposed comes first, beside its reply; the box to ask again follows.
    const proposal = S.proposal;
    if (proposal) {
      const card = el('div', 'ts-proposal'); card.appendChild(el('strong', '', proposal.summary || tr('studio.proposedEdit', 'Proposed edit')));
      const list = el('ul', 'ts-proposal-list'); for (const command of proposal.commands) list.appendChild(el('li', '', describeCommand(command))); card.appendChild(list);
      card.appendChild(el('p', 'ts-hint', tr('studio.proposalUndo', '{count} change(s) · one Undo reverts all of them', { count: proposal.commands.length })));
      const details = el('details'); details.append(el('summary', '', tr('studio.inspectCommands', 'Inspect exact commands')), el('pre', '', JSON.stringify(withoutNulls(proposal.commands), null, 2))); card.appendChild(details);
      const stale = proposal.expectedRevision !== S.project.revision;
      // A proposal made before the timeline changed cannot be applied as is; asking again uses the current project.
      if (stale) card.appendChild(el('p', 'ts-hint', tr('studio.staleProposal2', 'You changed the timeline after this was suggested, so it can’t be applied as is.')));
      const apply = button(tr('studio.applyProposal', 'Apply proposal'), async () => {
        const projectId = S.project.id; receiveProject(await api('proposals/apply', { projectId, proposalId: proposal.id, commandId: id() })); handleProposal(projectId, proposal.id);
        S.proposal = null; renderProject(); status(tr('studio.proposalApplied', 'Proposal applied and saved as one undoable transaction.'));
      }, 'ts-primary'); apply.disabled = stale;
      const buttons = row(card); buttons.appendChild(apply);
      const last = stale && state.messages.filter(m => m.role === 'user').pop();
      if (last) { const again = button(tr('studio.askAgain', 'Ask again'), () => { const asked = askAssistant(last.content); focusKey(ui.inspector, 'assistant-stop'); return asked; }); again.dataset.focusKey = 'assistant-again'; again.disabled = state.busy || videoLoaded; buttons.appendChild(again); }
      buttons.appendChild(button(tr('studio.dismissProposal', 'Dismiss'), () => { handleProposal(S.project.id, proposal.id); S.proposal = null; renderInspector(); })); body.appendChild(card);
    }
    const request = field(body, tr('studio.askAssistant', 'Ask the assistant'), state.draft || '', 'textarea'); request.maxLength = 8000; request.dataset.focusKey = 'assistant-request';
    request.placeholder = tr('studio.agentExample', 'Trim the selected clip to its first four seconds.'); request.disabled = state.busy || !!state.historyLoad;
    request.addEventListener('input', () => { state.draft = request.value; });
    // Examples fill the box and do not send: every one maps onto the commands the assistant can propose.
    const examples = [selected ? tr('studio.exampleTrim', 'Trim the selected clip to its first 2 seconds') : null, tr('studio.exampleFade', 'Fade out the last clip over 1 second'),
      tr('studio.exampleTitle', 'Add the title “Day one” over the first 3 seconds'), tr('studio.exampleGaps', 'Close the gaps between clips'),
      tr('studio.examplePlan', 'Plan a 5-second opening shot of waves at sunrise'), tr('studio.exampleExport', 'Export the timeline as an MP4')].filter(Boolean);
    const chips = el('div', 'ts-chips'); chips.setAttribute('aria-label', tr('studio.examples', 'Examples')); body.appendChild(chips);
    for (const text of examples) chips.appendChild(button(text, () => { request.value = state.draft = text; request.focus(); }, 'ts-chip'));
    const actions = row(body); const send = button(tr('studio.send', 'Send'), () => { const asked = askAssistant(request.value); focusKey(ui.inspector, 'assistant-stop'); return asked; }, 'ts-primary'); send.disabled = state.busy || !!state.historyLoad || videoLoaded; send.dataset.focusKey = 'assistant-send'; actions.appendChild(send);
    if (state.busy) { const stop = button(tr('studio.stop', 'Stop'), () => { if (state.controller) state.controller.abort(); }); stop.dataset.focusKey = 'assistant-stop'; actions.appendChild(stop); }
    body.appendChild(el('p', 'ts-hint', tr('studio.assistantPersistence', 'TensorAgent uses its saved conversation store. Other hosts retain up to eight recent messages in this browser; project packages exclude chat history.')));
    if (state.error) body.appendChild(el('p', 'ts-form-error', state.error));
    if (S.generationDrafts.length) {
      body.appendChild(el('h3', '', tr('studio.plannedShots', 'Planned shots')));
      body.appendChild(el('p', 'ts-hint', tr('studio.plannedShotsHelp2', 'Saved plans, not footage. Load a video model to make them: Studio notices it by itself. Open one in Generate to review it first.')));
      const available = videoModels().filter(m => m.available);
      for (const draft of S.generationDrafts) {
        const r = draft.request || {}, card = el('div', 'ts-proposal'); card.appendChild(el('strong', '', draft.summary || tr('studio.plannedShot', 'Planned shot')));
        if (r.prompt) card.appendChild(el('p', 'ts-job-prompt', r.prompt));
        card.appendChild(el('p', 'ts-hint', [shotText(r), r.frameCount && r.fps ? tr('studio.summaryLength', '{seconds} s ({frames} frames)', { seconds: secondsLabel(r.frameCount, r.fps), frames: r.frameCount }) : '', r.width && r.height ? r.width + ' × ' + r.height : ''].filter(Boolean).join(' · ')));
        const details = el('details'); details.append(el('summary', '', tr('studio.inspectRequest', 'Inspect exact request')), el('pre', 'ts-provenance', JSON.stringify(withoutNulls(r), null, 2))); card.appendChild(details);
        const buttons = row(card);
        buttons.appendChild(button(tr('studio.openInGenerate', 'Open in Generate'), () => { S.loadTake = { generation: { request: r } }; showPanel('inspector', 'generate'); focusKey(ui.inspector, 'prompt'); }));
        if (draft.jobId) card.appendChild(el('p', 'ts-hint', tr('studio.shotStarted', 'Generation started. Track its result in Jobs.')));
        else if (!available.length) card.appendChild(el('p', 'ts-hint', tr('studio.shotNeedsModel', 'No video model is loaded. The plan is saved for later.')));
        else {
          const model = available.length > 1 ? field(card, tr('studio.runModel', 'Run with video model'), available[0].modelId, 'select', available.map(m => ({ value: m.modelId, label: familyName(m) + (m.samplingRecipeName ? ' · ' + m.samplingRecipeName : '') }))) : null;
          buttons.appendChild(button(tr('studio.runPlannedShot', 'Generate planned shot'), async () => {
            await api('generation-drafts/start', { projectId: S.project.id, draftId: draft.id, expectedRevision: S.project.revision, modelId: model ? model.value : available[0].modelId });
            await loadProposals(); await refreshJobs(); status(tr('studio.plannedShotQueued', 'The planned shot is queued. Its requested settings were validated without automatic quality reduction.'));
          }, 'ts-primary'));
        }
        body.appendChild(card);
      }
    }
  }
  async function askAssistant(instruction) {
    if (!instruction.trim()) throw new Error(tr('studio.instructionRequired', 'Describe the edit you want.'));
    const projectId = S.project.id, initialRevision = S.project.revision, selectedClipIds = S.selected ? [S.selected] : [];
    const state = assistantFor(projectId); if (state.busy) return; state.busy = true; state.text = ''; state.error = null; state.controller = new AbortController();
    renderInspector();
    try {
      const preparedRevision = await assistantSession(projectId, state, initialRevision);
      if (!S.project || S.project.id !== projectId) throw new Error(tr('studio.projectChanged', 'The active project changed. The assistant request was not sent.'));
      if (S.project.revision !== preparedRevision || (S.selected || null) !== (selectedClipIds[0] || null)) throw new Error(tr('studio.contextChanged', 'The project changed while preparing the assistant. Review your selection and send again.'));
      const expectedRevision = S.project.revision, messages = boundedMessages(state.messages).concat([{ role: 'user', content: instruction.trim().slice(0, 8000) }]);
      const res = await fetch('/api/chat', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-TensorSharp-Studio': '1' }, body: JSON.stringify({ sessionId: state.sessionId, messages, maxTokens: 2048, studio: { projectId, expectedRevision, selectedClipIds } }), signal: state.controller.signal });
      if (res.status === 404) state.sessionId = null;
      if (!res.ok) { await response(res); return; }
      if (!res.body) throw new Error(tr('studio.streamUnavailable', 'The host did not provide an assistant response stream.'));
      const reader = res.body.getReader(), decoder = new TextDecoder(); let buffer = '', complete = false;
      try {
        while (!complete) {
          const chunk = await reader.read(); if (chunk.done) break; buffer += decoder.decode(chunk.value, { stream: true });
          if (buffer.length > 512000) throw new Error(tr('studio.responseTooLarge', 'Assistant response exceeded the bounded context limit.'));
          let end;
          while ((end = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, end).trimEnd(); buffer = buffer.slice(end + 1); if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim(); if (!data || data === '[DONE]') continue; let event; try { event = JSON.parse(data); } catch (_) { continue; }
            if (event.error) throw new Error(typeof event.error === 'string' ? event.error : event.error.message || tr('studio.agentError', 'Assistant request failed.'));
            if (event.token && state.text.length < 64000) state.text += String(event.token).slice(0, 64000 - state.text.length);
            if (event.done) complete = true;
          }
          if (state.replyNode && state.replyNode.isConnected) state.replyNode.textContent = state.text || tr('studio.thinking', 'Working…');
          else if (S.project && S.project.id === projectId && S.tab === 'assistant') renderInspector();
        }
      } finally { try { await reader.cancel(); } catch (_) { } reader.releaseLock(); }
      state.messages = boundedMessages(messages.concat(state.text ? [{ role: 'assistant', content: state.text.slice(0, 12000) }] : [])); state.draft = '';
      if (!state.conversationId) { try { window.localStorage.setItem('tensorsharp.studio.chat.' + projectId, JSON.stringify(state.messages)); } catch (_) { } }
      announce(tr('studio.assistantReplied', 'The assistant replied.'));
    } catch (e) { state.error = e.name === 'AbortError' ? tr('studio.agentStopped', 'Assistant stopped. Any saved proposal remains available for review.') : e.message; }
    finally {
      state.busy = false; state.controller = null; state.replyNode = null;
      // A proposal saved before a Stop or an error is still the person's to review.
      if (S.project && S.project.id === projectId) { loadProposals().catch(() => { /* Shown on the next load. */ }); if (S.tab === 'assistant') { renderInspector(); focusKey(ui.inspector, 'assistant-request'); } }
    }
  }
  // The host keeps every proposal; this browser remembers which were applied or dismissed, so neither reappears as stale.
  function handledProposals(projectId) {
    if (!S.handled) S.handled = new Map();
    if (!S.handled.has(projectId)) { let ids = []; try { ids = JSON.parse(window.localStorage.getItem('tensorsharp.studio.handled.' + projectId) || '[]'); } catch (_) { } S.handled.set(projectId, new Set(Array.isArray(ids) ? ids : [])); }
    return S.handled.get(projectId);
  }
  function handleProposal(projectId, proposalId) {
    const ids = handledProposals(projectId); ids.add(proposalId);
    try { window.localStorage.setItem('tensorsharp.studio.handled.' + projectId, JSON.stringify(Array.from(ids).slice(-100))); } catch (_) { /* Storage can be disabled. */ }
  }
  async function loadProposals() {
    const projectId = S.project.id; const [value, drafts] = await Promise.all([api('proposals/list', { projectId }), api('generation-drafts/list', { projectId })]); if (S.project.id !== projectId) return;
    // Only the newest proposal, and only while unhandled: older ones were superseded by it, not left to review.
    const handled = handledProposals(projectId), newest = (Array.isArray(value) ? value : value.proposals || []).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))[0];
    S.proposal = newest && !handled.has(newest.id) ? newest : null;
    S.generationDrafts = Array.isArray(drafts) ? drafts : drafts.drafts || [];
    if (S.tab === 'assistant') renderInspector();
  }
  function activeJob(job) { return ['queued', 'running', 'cancelling'].includes(job.state); }
  function jobsTabText() { const count = S.jobs.filter(activeJob).length; return count ? tr('studio.jobsCount', 'Jobs ({count})', { count }) : tr('studio.jobs', 'Jobs'); }
  function jobStateText(state) {
    const names = { queued: tr('studio.stateQueued', 'Queued'), running: tr('studio.stateRunning', 'Running'), cancelling: tr('studio.stateCancelling', 'Cancelling'), completed: tr('studio.stateCompleted', 'Completed'),
      failed: tr('studio.stateFailed', 'Failed'), cancelled: tr('studio.stateCancelled', 'Cancelled'), interrupted: tr('studio.stateInterrupted', 'Interrupted') };
    return names[state] || state;
  }
  // The stages hosts report (the video models' progress phases plus Studio's own); any other shows as reported.
  function jobStageText(stage) {
    const names = { queued: tr('studio.stageQueued', 'Waiting in the queue'), starting: tr('studio.stageStarting', 'Preparing the model'), 'model-identity': tr('studio.stageModelIdentity', 'Verifying model files'),
      'text-encode': tr('studio.stageTextEncode', 'Encoding the prompt'), 'image-encode': tr('studio.stageImageEncode', 'Encoding conditioning images'), 'reference-encode': tr('studio.stageReferenceEncode', 'Encoding reference images'),
      denoise: tr('studio.stageDenoise', 'Denoising'), 'vae-decode': tr('studio.stageVaeDecode', 'Decoding video frames'), 'audio-decode': tr('studio.stageAudioDecode', 'Decoding audio'),
      encoding: tr('studio.stageEncoding', 'Encoding the MP4'), importing: tr('studio.stageImporting', 'Adding the take to the library'), rendering: tr('studio.stageRendering', 'Rendering'),
      validating: tr('studio.stageValidating', 'Checking the rendered file'), cancelling: tr('studio.stageCancelling', 'Stopping and cleaning up') };
    return names[stage] || stage;
  }
  function jobSettingsText(job) {
    const s = job.summary, model = ((S.capabilities || {}).models || []).find(m => m.modelId === s.modelId);
    const parts = [model ? familyName(model) : s.modelId, shotText(s),
      tr('studio.jobCanvas', '{width}×{height} · {frames} frames ({seconds} s) at {fps} fps', { width: s.width, height: s.height, frames: s.frameCount, seconds: s.fps ? (s.frameCount / s.fps).toFixed(1) : '?', fps: s.fps }),
      // The steps the model runs, once it reports them: a two-stage recipe adds refinement steps to the requested ones.
      tr('studio.jobStepCount', '{steps} steps', { steps: job.totalSteps || s.steps })];
    if (s.seed >= 0) parts.push(tr('studio.jobSeed', 'seed {seed}', { seed: s.seed }));
    if (s.generateAudio) parts.push(tr('studio.jobWithAudio', 'with audio'));
    return parts.filter(Boolean).join(' · ');
  }
  function durationText(sec) {
    const total = Math.max(0, Math.round(Number(sec) || 0)), h = Math.floor(total / 3600), m = Math.floor(total % 3600 / 60), s = total % 60;
    if (h) return tr('studio.durationHours', '{h}h {m}m', { h, m: String(m).padStart(2, '0') });
    if (m) return tr('studio.durationMinutes', '{m}m {s}s', { m, s: String(s).padStart(2, '0') });
    return tr('studio.durationSeconds', '{s}s', { s });
  }
  // Estimates in whole minutes, so they do not read as more precise than they are.
  function remainingText(t) {
    const minutes = Math.round(t.remainingSeconds / 60), time = minutes < 60 ? tr('studio.estimateMinutes', '{m} min', { m: minutes }) : tr('studio.estimateHours', '{h} h {m} min', { h: Math.floor(minutes / 60), m: minutes % 60 });
    if (t.remainingScope === 'denoise') return minutes < 1 ? tr('studio.jobRemainingDenoiseSoon', 'Under a minute of denoising left, then decoding') : tr('studio.jobRemainingDenoise', 'About {time} of denoising left, then decoding', { time });
    return minutes < 1 ? tr('studio.jobRemainingSoon', 'Under a minute left') : tr('studio.jobRemaining', 'About {time} left', { time });
  }
  function clockText(value) {
    const d = new Date(value); if (Number.isNaN(d.getTime())) return '';
    const locale = typeof document !== 'undefined' && document.documentElement && document.documentElement.lang || undefined;
    return d.toDateString() === new Date().toDateString() ? d.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' }) : d.toLocaleString(locale, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function jobsPanel(body) {
    body.append(el('h3', '', tr('studio.jobsTitle', 'Background jobs')), el('p', 'ts-hint', tr('studio.jobPersistence', 'Jobs belong to the host. Closing this panel does not cancel them. Interrupted work is reported after restart.')));
    if (!renderAllowed()) body.appendChild(el('p', 'ts-hint', (S.capabilities || {}).renderUnavailableReason || tr('studio.renderUnsupported', 'Local rendering is unavailable on this host.')));
    if (!S.jobs.length) body.appendChild(el('p', 'ts-empty', tr('studio.noJobs', 'No generation or render jobs yet.')));
    for (const job of [...S.jobs].sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))) {
      const card = el('article', 'ts-job'); card.dataset.state = job.state; const t = job.timing || {}, s = job.summary;
      const head = el('div', 'ts-job-head'); head.append(el('strong', '', job.kind === 'generate' ? tr('studio.jobGenerate', 'Video generation') : tr('studio.jobRender', 'Render')), el('span', 'ts-job-state', jobStateText(job.state))); card.appendChild(head);
      if (s) { if (s.prompt) card.appendChild(el('p', 'ts-job-prompt', s.prompt)); card.appendChild(el('p', 'ts-hint', jobSettingsText(job))); }
      else card.appendChild(el('p', 'ts-hint', tr('studio.revision', 'revision {revision}', { revision: job.projectRevision })));
      if (activeJob(job)) {
        const steps = job.state === 'running' && job.totalSteps > 0 ? (job.step < job.totalSteps ? tr('studio.jobSteps', '{done} of {total} steps done · {left} left', { done: job.step, total: job.totalSteps, left: job.totalSteps - job.step }) : tr('studio.jobStepsDone', 'All {total} steps done', { total: job.totalSteps })) : '';
        const stage = job.state === 'queued' ? (t.jobsAhead ? tr('studio.jobsAhead', 'Waiting · jobs ahead: {count}', { count: t.jobsAhead }) : tr('studio.jobNext', 'Waiting · next to run')) : jobStageText(job.stage);
        card.appendChild(el('p', 'ts-job-stage', [stage, steps].filter(Boolean).join(' · ')));
        const progress = el('progress'); progress.max = 1; if (Number.isFinite(job.progress)) progress.value = job.progress; progress.setAttribute('aria-label', tr('studio.jobProgress', 'Reported job progress'));
        if (Number.isFinite(job.progress)) progress.setAttribute('aria-valuetext', Math.round(job.progress * 100) + '% · ' + stage); card.appendChild(progress);
        const timing = [];
        if (job.state === 'queued') {
          timing.push(tr('studio.jobQueuedFor', 'Queued for {time}', { time: durationText(t.queuedSeconds) }));
          if (t.previousRunSeconds != null) timing.push(tr('studio.jobPreviousRun', 'Last run with these settings took {time}', { time: durationText(t.previousRunSeconds) }));
        } else {
          if (job.startedAt) timing.push(tr('studio.jobStarted', 'Started {time}', { time: clockText(job.startedAt) }));
          if (t.elapsedSeconds != null) timing.push(tr('studio.jobElapsed', 'Elapsed {time}', { time: durationText(t.elapsedSeconds) }));
          if (job.state === 'running' && t.remainingSeconds != null) timing.push(remainingText(t));
          else if (job.state === 'running' && job.totalSteps > 0 && !job.step) timing.push(tr('studio.jobEstimatePending', 'Time left appears after the first step'));
        }
        card.appendChild(el('p', 'ts-hint', timing.join(' · ')));
        if (job.state !== 'queued' && t.stageElapsedSeconds != null) card.appendChild(el('p', 'ts-hint', [job.detail, tr('studio.jobInStage', '{time} in this stage', { time: durationText(t.stageElapsedSeconds) })].filter(Boolean).join(' · ')));
        const cancel = button(job.state === 'cancelling' ? tr('studio.cancelling', 'Cancelling…') : tr('studio.cancel', 'Cancel'), async () => { await api('jobs/cancel', { jobId: job.id }); await refreshJobs(); }); cancel.disabled = job.state === 'cancelling'; card.appendChild(cancel);
      } else {
        const stages = (job.stages || []).filter(x => x.seconds >= .5), last = (job.stages || []).filter(x => x.stage !== 'cancelling').pop();
        const outcome = [];
        if (job.state === 'completed' && t.elapsedSeconds != null) outcome.push(tr('studio.jobFinished', 'Finished in {time}', { time: durationText(t.elapsedSeconds) }));
        else if (last && t.elapsedSeconds != null) outcome.push(tr('studio.jobStoppedDuring', 'Stopped after {time} in stage: {stage}', { stage: jobStageText(last.stage), time: durationText(t.elapsedSeconds) }));
        if (job.state !== 'completed' && job.totalSteps > 0) outcome.push(tr('studio.jobStepsReached', '{done} of {total} steps done', { done: job.step, total: job.totalSteps }));
        if (job.finishedAt) outcome.push(clockText(job.finishedAt));
        if (outcome.length) card.appendChild(el('p', 'ts-hint', outcome.join(' · ')));
        if (stages.length) card.appendChild(el('p', 'ts-hint', tr('studio.jobBreakdown', 'Time by stage: {stages}', { stages: stages.map(x => jobStageText(x.stage) + ' ' + durationText(x.seconds)).join(' · ') })));
      }
      if (job.error) card.appendChild(el('p', 'ts-hint', job.error));
      if (job.state === 'interrupted') card.appendChild(el('p', 'ts-hint', tr('studio.interrupted', 'Restart the operation from its settings. Mid-generation checkpoint recovery is unavailable.')));
      if (job.state === 'completed' && job.kind === 'render') {
        const actions = row(card); actions.appendChild(button(tr('studio.playRender', 'Play rendered revision'), () => showRender(job))); const download = el('a', '', tr('studio.download', 'Download MP4')); download.href = outputUrl(job); download.download = S.project.name + '-r' + job.projectRevision + '.mp4'; actions.appendChild(download);
      }
      if (job.outputAssetId) { const a = asset(job.outputAssetId); if (a) card.appendChild(button(tr('studio.addGenerated', 'Add take to timeline'), () => insertAsset(a.id))); }
      body.appendChild(card);
    }
  }
  function keydown(e) {
    if (S.deleteDialog) { if (e.key === 'Escape') { e.preventDefault(); S.deleteDialog(false); } return; }
    const editable = !!e.target && (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable);
    const typing = !!e.target && (e.target.tagName === 'TEXTAREA' || e.target.isContentEditable || e.target.tagName === 'INPUT' && !/^(checkbox|radio|range|button|file|submit|reset|color)$/.test(e.target.type));
    // Escape (or Cmd+. on an iPad keyboard without Esc) closes the newest open layer first: a picker, then a panel.
    if (e.key === 'Escape' || e.key === '.' && e.metaKey) {
      e.preventDefault();
      if (S.layers.length) { S.layers.pop().close(); return; }
      if (!typing) close(); // Escape while typing never throws away the whole Studio
      return;
    }
    if (e.key === 'Tab') {
      const focusable = Array.from(S.root.querySelectorAll('button, input, select, textarea, a[href], summary')).filter(n => !n.disabled && n.getClientRects().length);
      if (focusable.length && e.shiftKey && document.activeElement === focusable[0]) { e.preventDefault(); focusable[focusable.length - 1].focus(); }
      else if (focusable.length && !e.shiftKey && document.activeElement === focusable[focusable.length - 1]) { e.preventDefault(); focusable[0].focus(); } return;
    }
    if (editable) return;
    // Undo, redo and duplicate work from anywhere but a text field: a status line may have just said "use Undo".
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); run(() => history(e.shiftKey ? 'redo' : 'undo')); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') { e.preventDefault(); if (clip()) run(() => change([{ type: 'duplicateClip', clipId: S.selected }])); return; }
    // Editing shortcuts belong to the timeline and monitor. In the side panels (picker tiles, slots, radios) and on any
    // button outside the timeline, the browser keeps Space, Delete and the arrow keys.
    if (within(e.target, ui.inspector) || within(e.target, ui.library)) return;
    const control = !!e.target && /^(BUTTON|A|SUMMARY)$/.test(e.target.tagName), onTimeline = within(e.target, ui.timeline);
    if (control && !onTimeline) return;
    else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); run(removeSelected); }
    else if (e.key.toLowerCase() === 's' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); run(splitSelected); }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); seek(S.playhead + (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1)); }
    else if (e.code === 'Space' && !control && ui.video && ui.video.src && !ui.video.hidden) { e.preventDefault(); if (ui.video.paused) run(() => ui.video.play()); else ui.video.pause(); }
  }
  function localMediaUrl(value) {
    try { const url = new URL(value, window.location.href); if (url.origin !== window.location.origin || !/^\/(?:uploads\/|api\/uploads\/|api\/code\/artifacts\/)/.test(url.pathname)) return null; return url.pathname + url.search; } catch (_) { return null; }
  }
  function attachChatActions() {
    const chat = document.getElementById('chat-container') || document.getElementById('chat'); if (!chat) return;
    function scan() {
      for (const media of chat.querySelectorAll('video')) {
        if (media.dataset.studioAttached) continue; const source = media.currentSrc || media.getAttribute('src') || (media.querySelector('source') || {}).src;
        const url = localMediaUrl(source); if (!url) continue; media.dataset.studioAttached = 'true';
        const action = button(tr('studio.openMedia', 'Open in Studio'), () => { media.pause(); return open({ url }); }, 'ts-studio-chat-action'); media.insertAdjacentElement('afterend', action);
      }
    }
    scan(); const observer = new MutationObserver(scan); observer.observe(chat, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
  }
  function init() {
    const controls = document.querySelector('.header-controls') || document.querySelector('.drawerfoot');
    if (controls && !document.getElementById('ts-studio-open')) { const entry = button(tr('studio.title', 'Video Studio'), () => open(), 'ts-studio-open' + (controls.classList.contains('drawerfoot') ? ' opt' : '')); entry.id = 'ts-studio-open'; controls.prepend(entry); }
    attachChatActions();
    if (window.location.hash === '#studio') run(() => open());
  }
  window.TensorSharpStudio = Object.freeze({ open: options => run(() => open(options)), close });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
