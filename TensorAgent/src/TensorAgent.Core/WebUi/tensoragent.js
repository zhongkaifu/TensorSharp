// ============================================================================
// TensorAgent — the phone client.
//
// It speaks the same loopback API the desktop Web UI speaks, so every capability
// the app has is still reachable; what is different is the shape of the surface
// around it. Written as one file with no dependencies because it is served from
// the app bundle over 127.0.0.1 and a build step for a single page would be a
// cost with no return.
// ============================================================================
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var chat = $('chat'), text = $('text'), send = $('send'), busy = $('busy');
  var modelBtn = $('model'), hold = $('hold'), abc = $('abc');

  var state = {
    model: null, arch: null, backend: null, contextTokens: 0,
    modelContextTokens: 0, visionReady: false,
    acceptsVisionProjector: true,
    visionChecking: false,
    session: null, conversation: null,
    history: [],            // {role, content, attachments}
    attachments: [],        // /api/upload responses
    skills: [],             // selected skill names
    skillSelectionExplicit: false, // distinguishes untouched discovery from deselect-all
    catalogSkills: [],
    conversations: [],      // the saved chats, for the menu
    generating: false,
    abort: null,
    // The id of the generation the HOST is running for this conversation. It is not
    // the same thing as `abort`, and that is the whole point: aborting stops this
    // page reading, while the turn keeps going and can be attached to again.
    turn: null,
    liveView: null,         // the assistant bubble a turn is being rendered into
    resuming: false,
    resumingSince: 0,       // when the lookup behind `resuming` started; a hung one is retried
    lastByteAt: 0,          // when the attached stream last delivered anything, keep-alives included
    modelInfo: null,        // { id, name, state, loading, error } from /api/agent/engine
    modelWatch: 0,
    maxTokens: 2048,
    speech: '',            // BCP-47 for dictation; empty follows the device
    settings: null,
    native: false,         // true when the page is inside the app, not a browser
    netMsg: '',            // the host's own wording for a network refusal
    voice: false,          // the composer is the hold-to-talk button
    // Whether the model reasons before answering. It used to be a switch under the
    // composer; it is a Settings choice now ("Show reasoning by default"), because a
    // permanent control for something a user decides once is a poor trade for the only
    // row of chrome a phone composer has. Re-read whenever the chat comes back to the
    // front, so changing it in Settings applies to the very next message.
    think: false,
  };

  // ---- tiny helpers --------------------------------------------------------
  function el(tag, cls, txt) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt != null) n.textContent = txt;
    return n;
  }
  // A non-2xx is a failure. fetch RESOLVES for 403 and 500 -- only a dropped
  // connection rejects -- so a caller that just fires this off cannot tell a refused
  // request from a delivered one. That is survivable for a telemetry ping and not for
  // the menu, where the whole effect of the tap is this request arriving.
  // Retried ONCE when the failure is the transport's, not the host's. The one time that
  // happens for real is the first request after the app comes back from the background:
  // the host closes idle keep-alive connections after 15 s, iOS reclaims a suspended
  // app's sockets, and CFNetwork replays a GET on a fresh connection but never a POST
  // (QA1941) -- so the POST is the one that surfaces as "Load failed" while the GETs
  // around it quietly succeed. Every caller of this is idempotent or a nudge: claiming
  // a share peeks, stopping a turn twice stops it once, and the settings are a whole
  // document. The chat request itself does not go through here.
  function post(url, body, retried) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    }).then(function (r) {
      if (!r.ok) throw new Error(url + ' answered HTTP ' + r.status);
      return r;
    }, function (e) {
      throw transportError(e);
    }).catch(function (e) {
      if (retried || !networkFailure(e)) throw e;
      note('post-retry', url + ': ' + ((e && e.message) || e));
      return delay(300).then(function () { return post(url, body, true); });
    });
  }
  /**
   * A request that the TRANSPORT failed -- fetch() rejecting, or a stream's read()
   * rejecting -- as opposed to anything that went wrong with what came back. WebKit
   * reports both with a TypeError, so the type alone cannot tell "Load failed" from a
   * bug in this file; the boundary at which the error arose can, and marks it.
   */
  function transportError(e) {
    if (e && e.transport) return e;
    var t = new Error((e && e.message) || 'Load failed');
    t.name = (e && e.name) || 'TypeError';
    t.transport = true;
    return t;
  }
  function networkFailure(e) { return !!e && e.transport === true && e.name !== 'AbortError'; }
  function delay(ms) { return new Promise(function (ok) { setTimeout(ok, ms); }); }
  /** A GET with a deadline, for the lookups that must not be able to wedge a flag forever. */
  function fetchTimed(url, ms) {
    var ctrl = new AbortController();
    var timer = setTimeout(function () { try { ctrl.abort(); } catch (e) {} }, ms);
    return fetch(url, { signal: ctrl.signal }).then(function (r) {
      clearTimeout(timer);
      return r;
    }, function (e) {
      clearTimeout(timer);
      throw transportError(e);
    });
  }

  // ---- what happened, for the app to read back ----------------------------
  //
  // The page cannot log anywhere the app can see: its console is gone the moment the
  // app is backgrounded, which is when the things worth knowing happen. So it keeps a
  // short record of its own transport events -- visibility changes, streams ending,
  // requests failing, recoveries -- and the app pulls it over the bridge into
  // logs/background.log when the app comes back to the front. Kinds and ids only,
  // never message text.
  var diag = [];
  function note(kind, detail) {
    diag.push({ t: Date.now(), k: kind, d: detail == null ? '' : String(detail).slice(0, 160) });
    if (diag.length > 120) diag.shift();
  }
  function atBottom() { return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 90; }
  function toBottom() { chat.scrollTop = chat.scrollHeight; }

  // ---- the layout follows the VISIBLE viewport ----------------------------
  // The single most important line in this file for a phone. Without it the
  // keyboard pushes the composer below the fold and WebKit scrolls the document
  // to chase it, taking the conversation off the top of the screen.
  var vv = window.visualViewport, pendingVh = 0;
  function applyVh() {
    document.documentElement.style.setProperty('--vh', (vv ? vv.height : window.innerHeight) + 'px');
    if (window.scrollY !== 0) window.scrollTo(0, 0);
  }
  function scheduleVh() {
    if (pendingVh) return;
    pendingVh = requestAnimationFrame(function () { pendingVh = 0; applyVh(); });
  }
  if (vv) { vv.addEventListener('resize', scheduleVh); vv.addEventListener('scroll', scheduleVh); }
  window.addEventListener('orientationchange', function () { setTimeout(applyVh, 200); });
  applyVh();

  var stickBottom = true;
  chat.addEventListener('scroll', function () { stickBottom = atBottom(); });
  if (vv) vv.addEventListener('resize', function () { if (stickBottom) setTimeout(toBottom, 60); });
  text.addEventListener('focus', function () { if (stickBottom) { setTimeout(toBottom, 60); setTimeout(toBottom, 350); } });

  // ---- markdown ------------------------------------------------------------
  // Deliberately small: fenced code, inline code, bold/italic, links, headings
  // and lists. Everything is escaped first, so a model that emits HTML cannot
  // put nodes into this page.
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function render(md) {
    var out = '', rest = String(md == null ? '' : md), fence = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)(?:```|$)/;
    var m;
    while ((m = fence.exec(rest))) {
      out += inline(rest.slice(0, m.index));
      out += '<pre><code>' + esc(m[2]) + '</code></pre>';
      rest = rest.slice(m.index + m[0].length);
    }
    return out + inline(rest);
  }
  function inline(s) {
    var t = esc(s);
    t = t.replace(/`([^`\n]+)`/g, function (_, c) { return '<code>' + c + '</code>'; });
    t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    // Images before links: an image is a link with a bang in front of it.
    t = t.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img alt="$1" src="$2">');
    t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    t = t.replace(/^### (.*)$/gm, '<strong>$1</strong>');
    t = t.replace(/^## (.*)$/gm, '<strong>$1</strong>');
    t = t.replace(/^# (.*)$/gm, '<strong>$1</strong>');
    return t.split(/\n{2,}/).map(function (p) {
      return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
    }).join('');
  }

  // ---- attachments, as the transcript holds them ---------------------------
  //
  // An attachment survives a chat being closed only if what is saved is enough to
  // rebuild it, and for a long time it was not: the page sent the model its file
  // paths and told the transcript nothing, so reopening a chat gave back the words
  // and a blank where the photo had been. What is saved now is the chip itself —
  // the stored name, the name the user knows it by, what kind of thing it is — and
  // every URL is derived from the stored name rather than remembered, so a saved
  // chat cannot point at an address that has moved.

  /** Where the loopback server serves an upload from, by its stored name. */
  function uploadUrl(name) {
    return name ? '/uploads/' + encodeURIComponent(name) : '';
  }
  /** The stored name out of a /uploads/ URL, for a reply that gave us one. */
  function uploadName(url) {
    if (!url) return '';
    try { return decodeURIComponent(String(url).split('/').pop()); } catch (e) { return String(url).split('/').pop(); }
  }
  /** The file itself: the original the user attached. */
  function fileUrlOf(a) { return a.url || uploadUrl(a.file); }
  /** What to SHOW. A HEIC has a PNG beside it because no browser renders HEIC. */
  function previewOf(a) {
    return a.previewUrl || (a.previewFile ? uploadUrl(a.previewFile) : fileUrlOf(a));
  }
  /** The chip, reduced to what has to survive: no page-session URLs, no file text. */
  function chipOf(a) {
    var chip = {
      file: a.file,
      fileName: a.fileName || a.file,
      mediaType: a.mediaType || 'text',
    };
    var preview = a.previewFile || (a.previewUrl ? uploadName(a.previewUrl) : '');
    if (preview) chip.previewFile = preview;
    if (a.frames && a.frames.length) chip.frames = a.frames.slice();
    if (a.fileBacked === true) chip.fileBacked = true;
    if (typeof a.pageCount === 'number') chip.pageCount = a.pageCount;
    if (typeof a.extractedPageCount === 'number') chip.extractedPageCount = a.extractedPageCount;
    if (typeof a.renderedAsImages === 'boolean') chip.renderedAsImages = a.renderedAsImages;
    return chip;
  }

  /**
   * What the user typed, out of a message that also carries a file's content.
   *
   * A text upload is sent to the model as "[File: notes.md] … [End of file]" in front
   * of the question, because that is how the model reads it. Rendering that back into
   * the bubble on a resumed chat would show the user a wall of their own file where
   * their sentence used to be. Mirrors Conversation.StripFileEnvelopes on the host.
   */
  function displayText(content) {
    var text = content == null ? '' : String(content);
    if (text.indexOf('[File: ') !== 0) return text;
    var end = text.lastIndexOf('[End of file]');
    return end < 0 ? text : text.slice(end + '[End of file]'.length).replace(/^\s+/, '');
  }

  // ---- the transcript ------------------------------------------------------
  function clearEmpty() { var e = $('empty'); if (e) e.remove(); }

  /** One attachment, rendered as the thing it is rather than as a paperclip. */
  function attachmentNode(a) {
    var kind = a.mediaType || 'text';
    if (kind === 'image') {
      var img = document.createElement('img');
      img.src = previewOf(a); img.alt = a.fileName || 'image';
      img.loading = 'lazy';
      return img;
    }
    if (kind === 'audio') {
      var audio = document.createElement('audio');
      audio.controls = true; audio.preload = 'none'; audio.src = fileUrlOf(a);
      return audio;
    }
    if (kind === 'video') {
      var video = document.createElement('video');
      video.controls = true; video.preload = 'none'; video.playsInline = true;
      if (a.frames && a.frames.length) video.poster = uploadUrl(a.frames[0]);
      video.src = fileUrlOf(a);
      return video;
    }
    // A document is a link, not a label. It was a label, and a user who reopened a
    // chat could see that they had attached a PDF and had no way to open it.
    var link = document.createElement('a');
    link.className = 'filechip';
    link.href = fileUrlOf(a);
    link.target = '_blank';
    link.rel = 'noopener';
    link.textContent = (kind === 'pdf' ? '📕 ' : '📄 ') + (a.fileName || a.file);
    return link;
  }

  function addTurn(role, content, attachments, extra) {
    clearEmpty();
    var turn = el('div', 'turn ' + (role === 'user' ? 'me' : 'bot'));
    var b = el('div', 'bubble');
    if (attachments && attachments.length) {
      attachments.forEach(function (a) {
        if (a && a.file) b.appendChild(attachmentNode(a));
      });
    }
    if (content) {
      var body = el('div');
      body.innerHTML = render(content);
      b.appendChild(body);
    }
    if (extra && extra.imageUrl) {
      var made = document.createElement('img');
      made.src = extra.imageUrl; made.alt = 'generated image';
      b.appendChild(made);
    }
    turn.appendChild(b);
    chat.appendChild(turn);
    // The files this turn produced, put back the way the live turn showed them.
    // They are the point of the turn far more often than the prose is.
    if (extra && extra.artifacts) extra.artifacts.forEach(function (f) { fileLine({ turn: turn, bubble: b }, f); });
    if (stickBottom) toBottom();
    return { turn: turn, bubble: b };
  }

  // What the assistant is doing, and what it did.
  //
  // The desktop page shows a live activity block and DELETES it when the step
  // finishes, which suits a wide screen you are watching. On a phone the useful
  // thing is the opposite: a short trace that stays, so a user who looked away can
  // see that it read a skill, ran a script and edited a file, without scrolling
  // through the raw output of each. Live status while it runs; one line per step
  // once it is done.
  //
  // One entry per tool the host can report progress for. The host names each by its
  // declared name (an alias it accepts, such as str_replace or apply-patch, is reported
  // as the tool it runs). edit_file is no longer declared to the model, but the host
  // still runs it for a model that reaches for it anyway, so its frames still come.
  var TOOL_LABEL = {
    shell: ['Generating code', 'Running code'],
    apply_patch: ['Preparing patch', 'Applying patch'],
    read_file: ['Preparing read', 'Reading file'],
    edit_file: ['Preparing edit', 'Editing file'],
    write_file: ['Preparing file', 'Writing file'],
    skills_list: ['Preparing lookup', 'Checking skills'],
    skills_read: ['Preparing read', 'Reading skill'],
    skills_run: ['Preparing run', 'Running skill'],
    spawn_agent: ['Preparing sub-agent', 'Starting sub-agent'],
    wait_agent: ['Preparing wait', 'Waiting for sub-agents'],
    send_input: ['Preparing message', 'Messaging sub-agent'],
    close_agent: ['Preparing stop', 'Stopping sub-agent'],
    list_agents: ['Preparing lookup', 'Checking sub-agents'],
  };
  function labelFor(tool, phase) {
    var pair = TOOL_LABEL[tool];
    if (pair) return pair[phase === 'writing' ? 0 : 1];
    var name = tool ? String(tool).replace(/_/g, ' ') : 'operation';
    return (phase === 'writing' ? 'Preparing ' : 'Running ') + name;
  }

  // ---- what it is doing RIGHT NOW ------------------------------------------
  //
  // A turn can spend a minute between the question and the first word of the
  // answer: reading a skill, writing a program, running it, reading the output.
  // For all of that the bubble is empty, and an empty bubble on a phone is
  // indistinguishable from an app that has died. So the turn carries a live
  // panel — the step it is on, and the last few lines of whatever text it is
  // producing, be that its reasoning or the command it is typing — which is
  // taken down the moment the answer exists. Shown by default: it is not detail
  // for the curious, it is the only evidence the app is working.
  var TAIL_LINES = 3, TAIL_CHARS = 400;

  /** The last few non-blank lines of a growing text, bounded so a single unbroken
   *  paragraph cannot push the composer off the screen. */
  function tailOf(text) {
    var lines = String(text == null ? '' : text).split('\n');
    var kept = [];
    for (var i = lines.length - 1; i >= 0 && kept.length < TAIL_LINES; i--) {
      if (lines[i].trim().length) kept.unshift(lines[i]);
    }
    var s = kept.join('\n');
    return s.length > TAIL_CHARS ? '…' + s.slice(s.length - TAIL_CHARS) : s;
  }

  // Both of these are called once per TOKEN, so both compare before they write.
  // Setting the same textContent again is a style recalculation and a scroll on every
  // token of a two-thousand-token answer, on the device with the least to spare.
  //
  // The panel is ONE element pinned under the message box, not one per turn. It began
  // inside the turn, which is where the activity happens — and that is exactly why it
  // did not work: by the time a program has been written and run, the answer is
  // streaming and the top of the turn is several screens up, so the user had to scroll
  // away from the words arriving to find out whether anything was happening. Under the
  // thumb, above the keyboard, it is the same information without the scroll.
  var activity = $('activity');
  var activityLabel = activity.querySelector('.label');
  var activityTail = activity.querySelector('.tail');
  var shownLabel = null, shownTail = null, shownClass = null;

  function progress(label, kind) {
    var cls = 'on' + (kind ? ' ' + kind : '');
    if (shownClass !== cls) { shownClass = cls; activity.className = cls; }
    if (label && shownLabel !== label) {
      shownLabel = label;
      activityLabel.textContent = label;
    }
    return activity;
  }
  function progressTail(text) {
    var tail = tailOf(text);
    if (shownTail === tail) return;
    shownTail = tail;
    activityTail.textContent = tail;
  }
  function progressDone() {
    shownLabel = shownTail = null;
    shownClass = '';
    activity.className = '';
    activityLabel.textContent = '';
    activityTail.textContent = '';
  }

  /** One line's worth of a longer string: collapsed, trimmed, elided. */
  function shorten(s, n) {
    var t = String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
    return t.length > n ? t.slice(0, n - 1) + '…' : t;
  }

  function stepLine(view, cls, text) {
    var line = el('div', 'step ' + cls);
    line.appendChild(el('span', 'dot'));
    line.appendChild(el('span', 'txt', text));
    view.turn.insertBefore(line, view.bubble);
    if (stickBottom) toBottom();
    return line;
  }

  // A file a skill script or a command produced. Rendered from the frame rather
  // than from the model's answer, because a small model repeats a download link
  // erratically and the file is the thing the user actually asked for.
  function fileLine(view, file) {
    if (!file || !file.url) return;
    var line = el('div', 'step file');
    line.appendChild(el('span', 'dot'));
    var a = document.createElement('a');
    a.href = file.url;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = '📄 ' + (file.name || 'file')
      + (file.bytes ? ' · ' + Math.max(1, Math.round(file.bytes / 1024)) + ' KB' : '');
    line.appendChild(a);
    view.turn.insertBefore(line, view.bubble);
    if (stickBottom) toBottom();
  }

  // One kept line per finished step, and the live label while a step runs. The
  // desktop deletes its activity block when the step ends and renders no history
  // at all; on a phone the trace IS the answer to "what did it just do for 40
  // seconds", so it stays — and it names the thing rather than the category: which
  // skill was read, which file was edited, which command was run.
  function trace(view, f) {
    var phase = String(f.tool_progress || '');
    if (f.tool) view.tool = String(f.tool);
    var tool = view.tool || '';

    if (phase === 'finished') {
      var secs = Math.round(Number(f.seconds) || 0);
      var step = view.step || {};
      // Best first: the skill and resource the host recorded, then the command the
      // model actually ran, then the frame's own detail, then just the label.
      var what = step.skill
        ? step.skill + (step.detail ? ' · ' + step.detail : '')
        : (step.detail || view.detail || f.detail || '');
      var text = labelFor(tool, 'running')
        + (what ? ' · ' + shorten(what, 70) : '')
        + (secs ? ' · ' + secs + 's' : '');

      stepLine(view, step.ok === false ? 'fail' : 'done', text);
      (step.files || []).forEach(function (file) { fileLine(view, file); });

      view.tool = '';
      view.step = null;
      view.detail = '';
      // The strip keeps saying what just finished until the next thing starts. A
      // generation between two tool calls is silent for many seconds, and "Working…"
      // over a blank line says less than the step that has just come back.
      progress(text, step.ok === false ? 'fail' : 'done');
      progressTail('');
      return;
    }
    if (phase !== 'writing' && phase !== 'running') return;
    if (phase === 'running' && f.detail) view.detail = String(f.detail);
    // The elapsed seconds tick once a second while a command runs, which is the
    // difference between "it is doing something" and "it has stopped".
    var elapsed = phase === 'running' ? Math.round(Number(f.seconds) || 0) : 0;
    progress(labelFor(tool, phase) + '…' + (elapsed ? ' ' + elapsed + 's' : ''));
  }

  // What the host did on the model's behalf, recorded as it happened: which skill it
  // read, which script it ran, whether that worked, and what it produced. The frame
  // arrives just before the tool's `finished`, so it is held and used to write that
  // one line rather than adding a second one saying the same thing twice.
  function skillStep(view, f) {
    view.step = {
      skill: f.skill ? String(f.skill) : '',
      detail: f.detail ? String(f.detail) : '',
      ok: f.ok !== false,
      files: f.files || null,
    };
  }

  function addCopy(turn, getText) {
    var c = el('button', 'copy', 'Copy');
    c.addEventListener('click', function () {
      var t = getText();
      if (navigator.clipboard) navigator.clipboard.writeText(t);
      c.textContent = 'Copied'; setTimeout(function () { c.textContent = 'Copy'; }, 1200);
    });
    turn.appendChild(c);
  }

  function notice(msg, kind) {
    clearEmpty();
    var n = el('div', 'notice' + (kind === 'error' ? ' error' : ''), msg);
    chat.appendChild(n);
    if (stickBottom) toBottom();
    return n;
  }

  // A refusal the user can act on, rather than prose about a switch they have to go
  // and find. The research skill's failure is the case this exists for: it needs the
  // network, the network is off by default, and "network access is disabled by the
  // user" told the reader what happened without telling them what to do about it.
  function noticeWithAction(msg, label, run) {
    var n = notice(msg, 'error');
    var b = el('button', 'notice-action', label);
    b.type = 'button';
    b.addEventListener('click', function () {
      b.disabled = true;
      Promise.resolve(run()).then(function (ok) {
        b.textContent = ok === false ? 'Could not change it' : 'Done';
      });
    });
    n.appendChild(document.createElement('br'));
    n.appendChild(b);
    return n;
  }

  function turnNetworkOn() {
    var next = Object.assign({}, state.settings || {}, { allowNetwork: true });
    return post('/api/agent/settings', next)
      .then(function (r) { return r.json(); })
      .then(function (s) {
        state.settings = s;
        notice('Network is on. Ask again and the assistant can reach the web.');
        return true;
      })
      .catch(function () { return false; });
  }

  // Offered at most once a turn: a skill that retries three times must not stack
  // three identical buttons.
  function offerNetworkIfRefused(textSeen, offered) {
    if (offered || !state.netMsg) return offered;
    if (String(textSeen).indexOf(state.netMsg) < 0) return offered;
    if (state.settings && state.settings.allowNetwork) return offered;
    noticeWithAction(
      'That needed the internet, and network access is off. Everything else runs on '
      + 'this device; only this step needs to go out.',
      'Turn on Network', turnNetworkOn);
    return true;
  }

  // ---- model state ---------------------------------------------------------
  function paintModel(d) {
    state.model = (d && d.loaded) || null;
    state.arch = (d && d.architecture) || null;
    state.backend = (d && d.loadedBackend) || null;
    // loadedMmProj is retained as a compatibility fallback for an older host.
    // visionReady is authoritative: it is true only after the loaded model has
    // accepted a projector (or carries an integrated vision tower).
    state.visionReady = !!(d && (
      typeof d.visionReady === 'boolean' ? d.visionReady : d.loadedMmProj
    ));
    // Default conservatively for an older host: if vision is unavailable, keep the
    // image in the composer instead of assuming a text-only model/tool workflow.
    state.acceptsVisionProjector = !d || typeof d.acceptsVisionProjector !== 'boolean'
      ? true : d.acceptsVisionProjector;
    state.contextTokens = (d && d.contextTokens) || 0;
    // New hosts report the GGUF's own window separately from the effective
    // runtime/KV-cache limit. Fall back for compatibility with an older host.
    state.modelContextTokens = d && typeof d.modelContextTokens === 'number'
      ? d.modelContextTokens : state.contextTokens;
    state.maxTokens = (d && d.defaultMaxTokens) || 2048;
    paintModelButton();
  }

  // Three states, not two. The app now loads the model the user last used by itself,
  // and reading four gigabytes off flash takes seconds -- during which "No model yet"
  // is not merely unhelpful, it is wrong, and it sends the user to a Models list to
  // choose the model they have already chosen and which is at that moment loading.
  function paintModelButton() {
    if (state.model) {
      modelBtn.className = '';
      modelBtn.textContent = pretty(state.model);
      var details = [];
      if (state.backend === 'ggml_metal') details.push('GPU');
      else if (state.backend === 'ggml_cpu') details.push('CPU');
      if (state.modelContextTokens > 0) {
        if (state.contextTokens > 0 && state.contextTokens !== state.modelContextTokens) {
          details.push(shortTokens(state.modelContextTokens) + ' model context');
          details.push(shortTokens(state.contextTokens) + ' active');
        } else {
          details.push(shortTokens(state.modelContextTokens) + ' context');
        }
      } else if (state.contextTokens > 0) {
        details.push(shortTokens(state.contextTokens) + ' active context');
      }
      if (details.length) modelBtn.appendChild(el('span', 'sub', '  ' + details.join(' · ')));
    } else if (loadingModel()) {
      modelBtn.className = 'empty';
      modelBtn.textContent = 'Loading ' + (state.modelInfo.name || 'the model') + '…';
    } else {
      modelBtn.className = 'empty';
      modelBtn.textContent = 'No model yet';
    }
    send.disabled = state.visionChecking || shareDiscarding || (!state.model && !state.generating);
  }
  function loadingModel() {
    return !!(state.modelInfo && state.modelInfo.loading);
  }
  function pretty(file) {
    return String(file).replace(/\.gguf$/i, '').replace(/-(it|instruct)\b/i, '').replace(/[-_]/g, ' ');
  }
  function shortTokens(tokens) {
    return tokens >= 1024 && tokens % 1024 === 0 ? (tokens / 1024) + 'K' : String(tokens);
  }

  function refreshModel() {
    return fetchTimed('/api/models', 10000).then(function (r) { return r.json(); }).then(function (d) {
      paintModel(d);
      return d;
    }).catch(function (e) {
      note('refresh-model-failed', (e && e.message) || e);
      return null;
    });
  }

  // While the startup load runs, keep asking. It is the only way the page finds out
  // that the model it was told about has arrived: nothing pushes to this page, and a
  // send button that stays disabled after the weights are in memory is the same bug
  // as the one this whole path exists to fix, arriving a few seconds later.
  function watchModelLoad() {
    if (state.modelWatch) return;
    var deadline = Date.now() + 5 * 60 * 1000;
    state.modelWatch = setInterval(function () {
      if (state.model || !loadingModel() || Date.now() > deadline) {
        clearInterval(state.modelWatch);
        state.modelWatch = 0;
        return;
      }
      refreshModel().then(refreshEngine);
    }, 1500);
  }

  // ---- sessions and conversations -----------------------------------------
  function newSession(conversationId) {
    // Written as one literal rather than assembled, so the route this binds a
    // conversation with is greppable -- a test pins exactly this string, because a
    // session opened without a conversation silently loses the transcript.
    var url = conversationId
      ? '/api/sessions?conversation=' + encodeURIComponent(conversationId)
      : '/api/sessions?conversation=new';
    return post(url).then(function (r) { return r.json(); }).then(function (s) {
      state.session = s.sessionId || null;
      state.conversation = s.conversation || s.conversationId || conversationId || null;
      return s;
    });
  }

  function emptyState(line) {
    var e = el('div', null, '');
    e.id = 'empty';
    e.innerHTML = '<h1>TensorAgent</h1><p>' + esc(line) + '</p>';
    var b = el('button', 'cta', state.model || loadingModel() ? 'Manage models' : 'Choose a model');
    b.type = 'button';
    b.addEventListener('click', function () { post('/api/agent/events', { type: 'open-models' }); });
    e.appendChild(b);
    chat.appendChild(e);
  }

  /**
   * Put a saved transcript back on the screen, and back into the history.
   *
   * Both halves matter and only the first one is visible. The message is pushed into
   * `state.history` WHOLE — its image paths, its audio, the names of the documents
   * whose text is inlined in it — because that array is what the next request is
   * built from and what the host saves over the top of the stored copy. Keeping only
   * the role and the text, which is what this did, meant a resumed chat lost every
   * attachment twice: the model could no longer see the photo being discussed, and
   * the next turn wrote a transcript with the photo missing from it.
   */
  function renderMessages(messages) {
    (messages || []).forEach(function (m) {
      state.history.push(m);
      addTurn(m.role, displayText(m.content), m.attachments,
        { artifacts: m.artifacts, imageUrl: m.imageUrl });
    });
  }

  /**
   * Show a saved chat, or start a new one, WITHOUT reloading the page.
   *
   * Reloading is what this used to do, and it cost more than it bought: the WebView
   * starts over, every fetch in flight is cut, and on a phone that includes the answer
   * the model is in the middle of writing. Rebuilding the three things a chat actually
   * consists of -- the transcript, the engine session, and whatever generation is still
   * running for it -- is both faster and the only version that can hand the user back a
   * turn that carried on while they were somewhere else.
   */
  function openConversation(conversationId, options) {
    conversationOpening = true;
    detach();
    // A durable shared draft follows the composer between chats (its text already
    // does), so its marker can never outlive the image/file chip it represents and
    // acknowledge missing content on an unrelated send. Preserve the entire draft in
    // that case; retain the established clear-on-switch behaviour for ordinary picks.
    // A chat being RE-OPENED under the user -- to pick up a transcript the host finished
    // while this page could not read it -- keeps whatever they had attached meanwhile.
    if (!appliedShareOrder.length && !(options && options.keepDraft)) state.attachments = [];
    paintChips();
    // The transcript on screen is NOT cleared here. It used to be, and then a session
    // request that failed at the transport -- the very failure this page now recovers
    // from -- left the user with an empty chat and an error line where their
    // conversation had been. What is on screen is the last thing known to be true;
    // it is replaced when there is something to replace it with.
    return newSession(conversationId).then(function (s) {
      var msgs = (s && s.messages) || [];
      state.history = [];
      chat.innerHTML = '';
      // Assigned in every case, never only when there is something to assign. A saved
      // chat with no skills means no skills; leaving the previous chat's selection in
      // place ran this one with a skill nobody chose for it, and then wrote that skill
      // into its saved record.
      var fresh = !msgs.length;
      state.think = !fresh && typeof s.think === 'boolean' ? s.think : thinkDefault();
      // A chat saved while skills were on remembers which ones. Re-selecting them
      // after the feature was turned off would be the saved chat overruling the
      // setting, which is the wrong way round.
      state.skills = !skillsOn() ? []
        : (!fresh && Array.isArray(s.skills) ? s.skills.slice() : defaultSkills());
      state.skillSelectionExplicit = !skillsOn() || state.skills.length > 0
        || (!fresh && s.skillsExplicit === true);
      paintSkillChips();
      if (!fresh) {
        renderMessages(msgs);
        toBottom();
      } else {
        emptyState(state.model || loadingModel()
          ? 'New chat. Nothing you type leaves the device.'
          : 'Private AI that runs on this iPhone.');
      }
      // The answer a previous page left running here. Attached to, not restarted:
      // the tokens it produced while nobody was reading arrive first, then the rest
      // as they come.
      if (s && s.activeTurn && s.activeTurn.running) attachTurn(s.activeTurn.id);
      return s;
    }).catch(function (e) {
      // The transcript is no longer cleared before this succeeds, so the chat still
      // shows whatever it had: say what went wrong instead of replacing it. Only a
      // chat with nothing in it gets the empty state.
      if (chat.querySelectorAll('.turn').length) notice('Could not open that chat: ' + ((e && e.message) || e), 'error');
      else emptyState('Could not open that chat: ' + ((e && e.message) || e));
      return null;
    }).then(function (result) {
      conversationOpening = false;
      if (shareDrainAgain && shareIntakeReady && !state.generating && !shareDiscarding)
        setTimeout(takePendingShare, 0);
      return result;
    });
  }

  /**
   * Open whichever chat this page coming up should be showing.
   *
   * Launching the app gets a clean one. Resuming the last conversation is right for a
   * page that is merely coming back -- WebKit kills the content process of a WebView
   * whose view left the window, and that reload lands in an app that never stopped,
   * sometimes with a turn still generating for the chat the user was reading -- and
   * wrong for the app being opened, where the previous chat is one tap away in the
   * menu and an empty composer is what was asked for. The page cannot tell those two
   * apart from inside, so it asks the host, which is the app and therefore knows.
   */
  function openAtLaunch() {
    return loadConversations().then(function (list) {
      return fetch('/api/agent/launch')
        .then(function (r) { return r.json(); })
        .catch(function () { return null; })
        .then(function (d) {
          // Unreachable or unreadable means start clean. Of the two ways to be wrong,
          // an unexpected blank chat is the one the user can fix with one tap; an
          // unexpected old chat is the thing they asked us to stop doing.
          var cold = !d || d.cold !== false;
          if (cold) return openConversation(null);
          // Coming back: to the chat the host says the page was in, NOT to the newest
          // saved one. They are usually different and the difference is the whole bug
          // -- the empty chat a launch opens is never in the list, since a chat with no
          // messages is not listed, so falling back to list[0] would hand the user
          // yesterday's conversation the first time the content process was reclaimed.
          // The id can name a chat deleted since; the host answers that with a fresh
          // one rather than an error.
          return openConversation(d.conversation || (list.length ? list[0].id : null));
        });
    });
  }

  // ---- content shared from another app -----------------------------------
  //
  // A share stays on the host until an accepted send (or explicit discard) acknowledges it. That separation is
  // deliberate: a cold launch has no page to push into yet, and WebKit can replace a
  // hidden content process at any time. Pulling only after the launch conversation is
  // open keeps openConversation() from racing the draft; retaining it after the
  // composer changes keeps a failed page load or content-process reclaim from eating it.
  var shareIntakeReady = false;
  var shareDrain = null;
  var shareDrainAgain = false;
  var appliedShareIds = Object.create(null);
  var appliedShareOrder = [];
  var appliedShareParts = Object.create(null);
  var shareDiscarding = false;
  var conversationOpening = false;

  /** Join at the boundary without trimming or otherwise rewriting either message. */
  function mergeSharedText(current, incoming) {
    current = current == null ? '' : String(current);
    incoming = incoming == null ? '' : String(incoming);
    if (!incoming) return current;
    if (!current) return incoming;

    // Preserve intentional line breaks. Horizontal whitespace at the join is merely a
    // separator, though, so collapse it to one space rather than producing glued words
    // or an expanding run every time another item is shared.
    var left = current.replace(/[\t ]+$/, '');
    var right = incoming.replace(/^[\t ]+/, '');
    if (/[\r\n]$/.test(left) || /^[\r\n]/.test(right)) return left + right;
    return left + ' ' + right;
  }

  function validSharedAttachment(a) {
    return !!a && typeof a === 'object' && a.ok === true
      && typeof a.file === 'string' && a.file.length > 0;
  }

  function rememberAppliedShare(id, parts) {
    if (appliedShareIds[id]) return;
    appliedShareIds[id] = true;
    appliedShareParts[id] = parts || {};
    appliedShareOrder.push(id);
    // This only bridges an acknowledgement retry in the CURRENT page. It must not be
    // sessionStorage: after a WebView reload the old DOM draft is gone, so the host's
    // still-pending item has to be applied again rather than acknowledged unseen.
    while (appliedShareOrder.length > 16)
    {
      var forgotten = appliedShareOrder.shift();
      delete appliedShareIds[forgotten];
      delete appliedShareParts[forgotten];
    }
  }

  function forgetAppliedShares(ids) {
    if (!Array.isArray(ids) || !ids.length) return;
    ids.forEach(function (id) {
      delete appliedShareIds[id];
      delete appliedShareParts[id];
    });
    appliedShareOrder = appliedShareOrder.filter(function (id) {
      return appliedShareIds[id] === true;
    });
    paintChips();
  }

  function removeTrackedSharedText(current, parts) {
    var before = parts && typeof parts.beforeText === 'string' ? parts.beforeText : '';
    var after = parts && typeof parts.afterText === 'string' ? parts.afterText : '';
    if (!after) return current;
    if (current === after) return before;
    // Text typed after (or before) the untouched share is user-owned and survives.
    // If the shared passage itself was edited, it is now user-owned too; do not guess
    // which characters to delete merely because the durable backing is discarded.
    if (current.indexOf(after) === 0) return before + current.slice(after.length);
    if (current.lastIndexOf(after) === current.length - after.length)
      return current.slice(0, current.length - after.length) + before;
    return current;
  }

  function discardAppliedShare(id, button) {
    if (!appliedShareIds[id] || shareDiscarding) return;
    if (state.visionChecking || state.generating) {
      notice('This shared item is already being sent. Stop or wait for the request before removing it.');
      return;
    }
    shareDiscarding = true;
    send.disabled = true;
    paintChips();
    post('/api/agent/share/discard', { id: id })
      .then(function (r) { return r.json(); })
      .then(function (body) {
        if (!body || body.ok !== true) throw new Error('TensorAgent retained the shared item.');
        var parts = appliedShareParts[id] || {};
        var sharedAttachments = Array.isArray(parts.attachments) ? parts.attachments : [];
        state.attachments = state.attachments.filter(function (current) {
          return !sharedAttachments.some(function (shared) {
            return current === shared || (current && shared && current.file === shared.file);
          });
        });
        text.value = removeTrackedSharedText(text.value, parts);
        forgetAppliedShares([id]);
        autoGrow();
        notice('Shared item removed.');
      })
      .catch(function (e) {
        notice('Could not remove the shared item: ' + ((e && e.message) || e), 'error');
      })
      .then(function () {
        shareDiscarding = false;
        send.disabled = state.visionChecking || (!state.model && !state.generating);
        paintChips();
        if (shareDrainAgain && !state.generating && !conversationOpening)
          setTimeout(takePendingShare, 0);
      });
  }

  /** Apply all parts of one host-prepared share as one composer update. */
  function applyPendingShare(share, mayOpenNewChat) {
    if (!share || typeof share !== 'object')
      return Promise.reject(new Error('The shared item was empty.'));
    var id = typeof share.id === 'string' ? share.id : '';
    if (!id) return Promise.reject(new Error('The shared item had no id.'));
    if (appliedShareOrder.length && !appliedShareIds[id])
      return Promise.reject(new Error('Finish or remove the current shared item before opening the next one.'));

    var incomingText = typeof share.text === 'string' ? share.text : '';
    var attachments = Array.isArray(share.attachments) ? share.attachments : [];
    var valid = [], problems = [];
    attachments.forEach(function (a) {
      if (validSharedAttachment(a)) valid.push(a);
      else problems.push((a && a.error) || 'One shared file could not be attached.');
    });
    var notices = Array.isArray(share.notices) ? share.notices.filter(function (n) {
      return typeof n === 'string' && n.length > 0;
    }) : [];

    var heldAttachments = state.attachments.slice();
    // One claimed envelope is one share action and always owns a fresh conversation.
    // Ignore the version-1 newChat flag: honoring false here allowed independently
    // shared items from older builds to accumulate in one conversation.
    var ready = mayOpenNewChat !== false
      ? openConversation(null).then(function (opened) {
          // openConversation reports its own useful error and resolves null. Turn that
          // into a rejection here so the share remains unacknowledged and retryable.
          if (!opened) {
            state.attachments = heldAttachments;
            paintChips();
            throw new Error('A new chat could not be opened for the shared item.');
          }
          // An unsent attachment is part of the draft just as much as text is. Starting
          // the share's new chat must not silently discard it.
          state.attachments = heldAttachments;
        })
      : Promise.resolve();

    return ready.then(function () {
      var beforeText = text.value;
      text.value = mergeSharedText(text.value, incomingText);
      Array.prototype.push.apply(state.attachments, valid);
      rememberAppliedShare(id, {
        beforeText: beforeText,
        afterText: text.value,
        text: incomingText,
        attachments: valid.slice(),
        title: typeof share.title === 'string' ? share.title : '',
      });
      paintChips();
      notices.forEach(function (message) { notice(message); });
      problems.forEach(function (message) { notice(message, 'error'); });
      autoGrow();
      if (!state.voice) text.focus();

      // From here on the composer update succeeded. Sharing deliberately never sends
      // a model turn: a crash between send acceptance and durable ACK cannot be made
      // exactly-once, while a reviewable draft can always be reapplied safely.
      return id;
    });
  }

  function claimOneShare(mayOpenNewChat) {
    return post('/api/agent/share/claim', {})
      .then(function (r) { return r.json(); })
      .then(function (body) {
        var share = body && Object.prototype.hasOwnProperty.call(body, 'share')
          ? body.share : null;
        if (!share) return false;
        var id = typeof share.id === 'string' ? share.id : '';
        if (!id) throw new Error('The shared item had no id.');

        // The host keeps returning the head until an accepted chat request consumes
        // it. Repeated visibility/pageshow nudges in this same page must therefore be
        // no-ops, while a fresh page (whose composer was lost) applies it again.
        var applied = appliedShareIds[id]
          ? Promise.resolve(id)
          : applyPendingShare(share, mayOpenNewChat);
        return applied.then(function () { return true; });
      });
  }

  /** Pull pending shares once, coalescing startup, visibility and native nudges. */
  function takePendingShare() {
    // pageshow and a native launch callback can both arrive while openAtLaunch is
    // still choosing a conversation. Remember the nudge; never let it race that open.
    if (!shareIntakeReady) {
      shareDrainAgain = true;
      return Promise.resolve();
    }
    // A second default-new-chat share arriving while the first answer streams must
    // wait. Applying it now would call openConversation(), detach this stream and pull
    // the user away from the answer they just requested.
    if (state.generating || conversationOpening || shareDiscarding) {
      shareDrainAgain = true;
      return Promise.resolve();
    }
    if (shareDrain) {
      shareDrainAgain = true;
      return shareDrain;
    }
    shareDrainAgain = false;
    // One at a time. The durable head stays leased until the user sends it, so claiming
    // again here would only rediscover the same draft; the next share is nudged in when
    // the accepted request releases this one.
    shareDrain = claimOneShare(true)
      .catch(function (e) {
        note('share-claim-failed', (e && e.message) || e);
        notice('Could not open the shared item: ' + ((e && e.message) || e), 'error');
      })
      .then(function (result) {
        shareDrain = null;
        if (shareDrainAgain) {
          shareDrainAgain = false;
          return takePendingShare();
        }
        return result;
      });
    return shareDrain;
  }

  function loadConversations() {
    return fetch('/api/agent/conversations').then(function (r) { return r.json(); }).then(function (d) {
      state.conversations = (d && d.conversations) || [];
      return state.conversations;
    }).catch(function () { return state.conversations; });
  }

  // ---- sending -------------------------------------------------------------
  function setGenerating(on) {
    state.generating = !!on;
    busy.className = on ? 'on' : '';
    send.textContent = on ? '■' : '➤';
    send.className = 'round ' + (on ? 'stop' : 'send');
    send.disabled = state.visionChecking || shareDiscarding || (!on && !state.model);
    paintChips();
    if (!on && shareDrainAgain && shareIntakeReady && !conversationOpening && !shareDiscarding)
      setTimeout(takePendingShare, 0);
  }

  function sendMessage() {
    if (state.generating) { note('send-is-stop', state.turn); stop(); return; }
    if (state.visionChecking) { note('send-refused', 'vision check in flight'); return; }
    if (shareDiscarding) {
      note('send-refused', 'share discard in flight');
      notice('Wait for the shared item to finish being removed, then send again.');
      return;
    }
    var t = text.value.trim();
    if (!t && !state.attachments.length) { note('send-refused', 'nothing to send'); return; }
    if (!state.model) {
      note('send-refused', loadingModel() ? 'model loading' : 'no model');
      // Two different answers, because they ask for two different things. A model
      // that is loading needs a few seconds; no model at all needs a download.
      if (loadingModel()) notice((state.modelInfo.name || 'The model') + ' is still loading. One moment.');
      else openSheet('model-sheet');
      return;
    }

    var atts = state.attachments.slice();
    var msg = messageFor(t, atts);
    var nextHistory = state.history.concat([msg]);

    // Capability can change while this long-lived WKWebView is hidden on the Models
    // page. Re-read it immediately before every image-bearing request, while the
    // composer is still intact. The server repeats this check authoritatively.
    if (nextHistory.some(function (m) { return m && m.imagePaths && m.imagePaths.length; })) {
      state.visionChecking = true;
      send.disabled = true;
      // Disable the shared marker in the same event turn as Send. The model-capability
      // refresh below is asynchronous and must not race a discard of its captured draft.
      paintChips();
      var conversation = state.conversation;
      refreshModel().then(function (modelState) {
        state.visionChecking = false;
        paintModelButton();
        paintChips();
        if (state.conversation !== conversation) return;
        // A refresh that could not reach the host says nothing about the model. The
        // host checks every request itself, so the last known answer stands rather
        // than sending the user to choose a model that is still loaded.
        var unreachable = modelState === null && !!state.model;
        if (!unreachable && (!modelState || !state.model)) {
          noticeWithAction(
            'The model is no longer loaded. Choose a model, then send this image again.',
            'Open Models',
            function () { openRoute('models'); return true; });
          return;
        }
        var hostFileSkillCanDecide = !state.acceptsVisionProjector
          && skillsOn() && state.skills.length > 0;
        if (!state.visionReady && !hostFileSkillCanDecide) {
          var message = state.acceptsVisionProjector
            ? 'This model is loaded without its vision file, so it cannot see the attached image yet.'
            : 'This model cannot see images. Choose a vision model to analyze the attachment.';
          noticeWithAction(
            message,
            'Open Models',
            function () { openRoute('models'); return true; });
          return;
        }
        commitMessage(t, atts, msg);
      });
      return;
    }

    commitMessage(t, atts, msg);
  }

  function commitMessage(t, atts, msg) {
    var userView = addTurn('user', t, atts);
    // A capability refresh is asynchronous. Preserve anything newly typed or
    // attached during that short check instead of clearing it with the sent draft.
    if (text.value.trim() === t) text.value = '';
    autoGrow();
    state.attachments = state.attachments.filter(function (a) {
      return atts.indexOf(a) < 0;
    });
    paintChips();

    state.history.push(msg);

    var body = {
      // The history AS IT IS, not a copy of it with the attachments removed. The
      // whole array used to be flattened to {role, content} on its way out, which
      // dropped every image path from every previous turn -- so the model could
      // answer a follow-up question about a photo it had been shown, and could not
      // see it any more.
      messages: state.history.slice(),
      maxTokens: state.maxTokens,
      think: !!state.think,
    };
    if (state.session) body.sessionId = state.session;
    if (!skillsOn()) body.skills_discovery = false;
    else if (state.skills.length) body.skills = state.skills;
    else if (state.skillSelectionExplicit) {
      body.skills = [];
      body.skills_discovery = false;
    }

    // The host acknowledges these only after every request preflight passed and the
    // user turn was written to the durable conversation store. Until that point the
    // App Group envelope remains the crash-safe backing for this volatile composer.
    if (appliedShareOrder.length) body.shareIds = appliedShareOrder.slice();

    stream(body, {
      text: t, attachments: atts, message: msg, turn: userView.turn,
      // So a recovery can tell the host's acknowledgement of this draft apart from a
      // request that never arrived: see resumeTurn.
      shareIds: body.shareIds,
    });
  }

  /**
   * One user message, in the shape /api/chat reads and the transcript stores.
   *
   * The five path lists are not interchangeable and the server treats each
   * differently: `imagePaths` is what the vision encoder sees (a video's frames go in
   * here too), `stillImagePaths` is the pictures the user actually attached,
   * `videoFilePaths` and `audioPaths` are the media themselves, and `textFilePaths`
   * names text documents whether their prose is inline or their table is file-backed.
   * `attachments` is the sixth and it is the one the user sees: what the chips said,
   * so a reopened chat says it again -- and, on the host side, which files to stage
   * into the working directory of anything the model runs.
   */
  function messageFor(typed, atts) {
    var msg = { role: 'user', content: typed || describe(atts) };
    var imagePaths = [], stillImagePaths = [], videoFilePaths = [], audioPaths = [];
    var textFilePaths = [], textFileNames = [], textParts = [];
    var isVideo = false;

    atts.forEach(function (a) {
      var kind = a.mediaType || 'text';
      if (kind === 'image') {
        imagePaths.push(a.file);
        stillImagePaths.push(a.file);
      } else if (kind === 'video') {
        isVideo = true;
        if (a.file) videoFilePaths.push(a.file);
        (a.frames || []).forEach(function (f) { imagePaths.push(f); });
      } else if (kind === 'audio') {
        audioPaths.push(a.file);
      } else if (kind === 'text' && a.fileBacked === true) {
        // CSV rows stay in the uploaded file. Keep the structured path and display
        // name so the server can stage the complete table for its file/code tools;
        // deliberately do not manufacture an inline [File: ...] envelope.
        if (a.file) { textFilePaths.push(a.file); textFileNames.push(a.fileName || a.file); }
      } else if (a.textContent) {
        // Text and born-digital PDFs alike: the content goes in front of the
        // question, and the file is named so a program can open the whole of it.
        textParts.push('[File: ' + (a.fileName || a.file) + ']\n' + a.textContent + '\n[End of file]');
        if (a.file) { textFilePaths.push(a.file); textFileNames.push(a.fileName || a.file); }
      } else if (kind === 'pdf' && a.frames && a.frames.length) {
        // A scanned PDF has no text layer; its pages are pictures for a vision model.
        if (a.file) { textFilePaths.push(a.file); textFileNames.push(a.fileName || a.file); }
        a.frames.forEach(function (f) { imagePaths.push(f); });
      } else if (a.file) {
        textFilePaths.push(a.file); textFileNames.push(a.fileName || a.file);
      }
    });

    if (textParts.length) msg.content = textParts.join('\n\n') + '\n\n' + msg.content;
    if (imagePaths.length) msg.imagePaths = imagePaths;
    if (stillImagePaths.length) msg.stillImagePaths = stillImagePaths;
    if (videoFilePaths.length) msg.videoFilePaths = videoFilePaths;
    if (audioPaths.length) msg.audioPaths = audioPaths;
    if (textFilePaths.length) msg.textFilePaths = textFilePaths;
    if (textFileNames.length) msg.textFileNames = textFileNames;
    if (isVideo) msg.isVideo = true;
    if (atts.length) msg.attachments = atts.map(chipOf);
    return msg;
  }
  function describe(atts) {
    return atts.map(function (a) { return (a.fileName || a.file); }).join(', ');
  }

  /**
   * Stop the model, which is a different act from stopping this page reading it.
   *
   * The turn belongs to the app now, so aborting the fetch would only leave it
   * generating for nobody. The Stop button has to say so out loud.
   */
  function stop() {
    if (state.turn) {
      post('/api/agent/turns/' + encodeURIComponent(state.turn) + '/stop')
        .catch(function (e) { notice('Could not stop the answer: ' + ((e && e.message) || e), 'error'); });
      detach();
      return;
    }
    // The id has not arrived yet. It rides on the stream's headers, and those are held
    // back until the first frame so that a refusal can still be a status code -- which
    // means that for the whole of a prefill, which is the minute a user is most likely
    // to change their mind in, this page does not know what to stop. Ask the host what
    // this conversation is generating.
    var conversation = state.conversation;
    detach();
    if (!conversation) return;
    fetch('/api/agent/turns?conversation=' + encodeURIComponent(conversation))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d && d.turn && d.turn.running) post('/api/agent/turns/' + encodeURIComponent(d.turn.id) + '/stop');
      })
      .catch(function () {});
  }

  /** Stop READING. The turn carries on; this is what leaving a chat does. */
  function detach() {
    if (state.abort) { try { state.abort.abort(); } catch (e) {} }
    state.abort = null;
    state.turn = null;
    state.liveView = null;
    // And the recovery with it. A timer left armed here fires minutes later against a
    // chat the user has left -- withdrawing a message the host DID take back into a
    // different composer, or attaching another conversation's turn.
    cancelScheduledResume();
    stopWatchdog();
    recovery.sentDraft = null;
    recovery.attempts = 0;
    progressDone();
    setGenerating(false);
  }

  /**
   * Pick the generation back up, if this page has lost hold of one.
   *
   * Called whenever the page becomes visible again, because that is exactly when it
   * may have missed something: WebKit suspends a WKWebView's content process the
   * moment its view leaves the window -- which is what opening any other screen in
   * this app does -- and a suspended process is not reading a stream. The turn itself
   * never stopped; it belongs to the host. This is how the page finds out what was
   * said while nobody was listening.
   *
   * A finished turn is attached to as well as a running one, and deliberately: an
   * answer that completed while the user was on another screen is replayed in full
   * rather than left as the half sentence they walked away from.
   */
  // ---- getting the answer back after the page could not read it -------------
  //
  // Four things end a page's view of a running turn without the turn ending: the
  // content process is suspended (any other screen, any other app), the app's sockets
  // are reclaimed while it is suspended, the host's listener is rebuilt, and a network
  // process that WebKit replaced underneath the page. None of them tell the page
  // anything it can act on at the time -- the stream simply rejects, ends early, or
  // says nothing ever again -- and the old rule, "a stream object exists, so nothing
  // needs doing", turned each of them into an answer that stayed half-written with a
  // Stop button that stopped nothing. So a stream is trusted only while it is
  // DELIVERING: the host writes a keep-alive every 5 s, which means an attached stream
  // that has said nothing for longer than that is presumed dead and replaced, and a
  // lookup that fails is retried with backoff rather than once.

  var RESUME_DELAYS = [800, 2000, 5000, 10000, 20000];
  // Every turn id this page has attached to or finished. The host's "what is this
  // conversation generating" answer is its LAST turn, and a finished turn is now kept
  // for an hour -- so for a page recovering a request whose turn it never learned the
  // id of, that answer is very often the PREVIOUS question's answer. Attaching to it
  // would render the old answer under the new question and then save both.
  var seenTurns = Object.create(null);
  function rememberTurn(id) { if (id) seenTurns[id] = 1; }
  // Longer than the host's 5 s keep-alive with room for a slow wake-up, and shorter
  // than anything a user would call "stuck".
  var STREAM_TRUST_MS = 8000;
  var PREFILL_TRUST_MS = 10 * 60 * 1000;
  var recovery = { attempts: 0, timer: 0, sentDraft: null };

  /** An attached stream that delivered something recently enough to be believed. */
  function streamLooksAlive() {
    if (!state.abort) return false;
    // Before the first frame there is nothing to deliver: the host holds the headers
    // back until the model has read the prompt, which is minutes for a long one, and
    // writes no keep-alive before them. A request still waiting for its headers is
    // trusted for as long as the prefill can take.
    var trust = state.abort.awaitingHeaders ? PREFILL_TRUST_MS : STREAM_TRUST_MS;
    return Date.now() - state.lastByteAt < trust;
  }

  /**
   * Stop reading the current stream without giving anything else up: the turn, the
   * bubble, and the fact that the model is working all stay. The reader's own
   * failure is ignored when it arrives, because it is no longer the reader.
   */
  function supersede(why) {
    var old = state.abort;
    if (!old) return false;
    // NEVER a request that has not been answered yet. Its headers carry the turn id,
    // and the host consumes a shared draft when it accepts the request -- so aborting
    // one loses both: the answer becomes unfindable and the share chip can never be
    // acknowledged, which makes every later send a 409. It is trusted for as long as a
    // prefill can take (streamLooksAlive), and if it dies it rejects and is recovered.
    if (old.awaitingHeaders) { note('supersede-declined', why); return false; }
    state.abort = null;
    stopWatchdog();
    try { old.abort(); } catch (e) {}
    note('supersede', why);
    return true;
  }

  function cancelScheduledResume() {
    if (recovery.timer) { clearTimeout(recovery.timer); recovery.timer = 0; }
  }

  // The watchdog. Everything above runs when something ASKS -- a visibility change,
  // the app's nudge, a stream that failed. A stream that simply stops delivering asks
  // nobody: the phone showed one that replayed its backlog after a restart and then
  // sat silent while the host went on producing, and nothing ever looked at it again.
  // A healthy stream is never quiet for long (the host writes a keep-alive every 5 s),
  // so a quiet one is checked on a timer and replaced.
  var WATCHDOG_MS = 4000;
  var watchdog = 0;
  /** Re-armed while a reader exists and stopped the moment one does not. */
  function armWatchdog() {
    if (watchdog || !state.abort) return;
    watchdog = setTimeout(function () {
      watchdog = 0;
      if (!state.abort) return;
      if (document.visibilityState !== 'visible' || streamLooksAlive()) { armWatchdog(); return; }
      note('watchdog', 'no bytes for ' + (Date.now() - state.lastByteAt) + 'ms');
      resumeTurn();
    }, WATCHDOG_MS);
  }
  function stopWatchdog() {
    if (watchdog) { clearTimeout(watchdog); watchdog = 0; }
  }

  /** Try again later, a little later each time; give up after a while and say so. */
  function scheduleResume(reason) {
    if (recovery.timer) return;
    var n = recovery.attempts++;
    if (n >= RESUME_DELAYS.length) { giveUpResuming(reason); return; }
    var ms = RESUME_DELAYS[n];
    note('resume-retry', n + 1 + ' in ' + ms + 'ms: ' + reason);
    recovery.timer = setTimeout(function () {
      recovery.timer = 0;
      // Not while hidden: nothing can be read there, and the next visibilitychange
      // asks again anyway.
      if (document.visibilityState === 'visible') resumeTurn();
    }, ms);
  }

  /**
   * The host could not be reached for long enough that waiting quietly has become
   * misleading. Hand the screen back with what was shown, and say why.
   */
  function giveUpResuming(reason) {
    note('resume-gave-up', reason);
    stopWatchdog();
    recovery.attempts = 0;
    // The message stays sent: whether the host took it cannot be known from here,
    // and taking it back would be claiming it was not.
    recovery.sentDraft = null;
    var view = state.liveView;
    if (view && view.answerSoFar) {
      // So the next request does not send a history with a hole where this answer
      // was and write that hole over the saved chat.
      state.history.push({ role: 'assistant', content: view.answerSoFar });
    } else if (view && view.turn && view.turn.parentNode) {
      // Nothing ever arrived in it; an empty bubble under the question says less than
      // the notice below does.
      view.turn.remove();
    }
    state.abort = null;
    state.turn = null;
    state.liveView = null;
    progressDone();
    setGenerating(false);
    notice('Lost the connection to the app while the answer was being written. It carries on in the app; reopen this chat to see it.', 'error');
  }

  /**
   * The host has no turn to attach to any more, and this page was mid-answer. The
   * saved transcript is the record now -- the turn wrote it as it finished -- so the
   * chat is reopened from it rather than left as a fragment on the screen that the
   * next request would write over the finished answer.
   */
  function settleWithoutTurn(conversation) {
    note('settle-without-turn', conversation);
    stopWatchdog();
    recovery.attempts = 0;
    var draft = recovery.sentDraft;
    recovery.sentDraft = null;
    if (draft) {
      // The request itself was lost before the host took it: give the words back.
      withdrawSend(draft);
      detach();
      return;
    }
    state.abort = null;
    state.turn = null;
    progressDone();
    setGenerating(false);
    openConversation(conversation, { keepDraft: true });
  }

  /** Put a sent message back into the composer, and take it off the screen and out of the history. */
  function withdrawSend(sentDraft) {
    if (sentDraft.turn && sentDraft.turn.parentNode) sentDraft.turn.remove();
    if (state.liveView && state.liveView.turn && state.liveView.turn.parentNode) state.liveView.turn.remove();
    state.liveView = null;
    var messageIndex = state.history.lastIndexOf(sentDraft.message);
    if (messageIndex >= 0) state.history.splice(messageIndex, 1);
    if (sentDraft.text) {
      var newerText = text.value.trim();
      text.value = newerText && newerText !== sentDraft.text
        ? sentDraft.text + '\n' + text.value : sentDraft.text;
    }
    sentDraft.attachments.slice().reverse().forEach(function (attachment) {
      var alreadyPresent = state.attachments.some(function (current) {
        return current === attachment || (current && attachment && current.file === attachment.file);
      });
      if (!alreadyPresent) state.attachments.unshift(attachment);
    });
    autoGrow(); paintChips();
  }

  /**
   * Pick the generation back up, if this page has lost hold of one.
   *
   * Called whenever the page becomes visible again, because that is exactly when it
   * may have missed something: WebKit suspends a WKWebView's content process the
   * moment its view leaves the window -- which is what opening any other screen in
   * this app does, and what leaving the app does -- and the stream it was reading is
   * dead or stale when it wakes. Also called by the app after it has checked its own
   * side of the transport, and by the retry timer. Cheap and idempotent: a stream that
   * is demonstrably delivering is left alone, so a nudge on top of a nudge costs
   * nothing, and one lookup at a time.
   */
  function resumeTurn() {
    if (!state.conversation) return true;
    if (streamLooksAlive()) { note('resume-skip', 'the stream is delivering'); return true; }
    // One lookup at a time -- but a lookup that never came back must not be the
    // reason no other one is ever tried.
    if (state.resuming && Date.now() - state.resumingSince < 12000) return true;
    cancelScheduledResume();
    supersede('resume');

    // Which chat this lookup is FOR, held across the round trip. Coming back to the app
    // and opening a different saved chat are the same gesture a moment apart -- the app
    // asks the page to resume as the chat reappears, and the answer arrives after the
    // page has moved on -- so without this the previous chat's answer streams into the
    // one now on screen and is saved under it.
    var conversation = state.conversation;
    state.resuming = true;
    state.resumingSince = Date.now();
    note('resume-lookup', conversation);
    fetchTimed('/api/agent/turns?conversation=' + encodeURIComponent(conversation), 10000)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        state.resuming = false;
        if (state.conversation !== conversation) return;
        var t = d && d.turn;
        note('resume-found', t ? t.id + (t.running ? ' running' : ' finished') : 'nothing');
        recovery.attempts = 0;
        // Mine, or one this page has never read. Anything else is the previous
        // question's answer, still retained by the host, and attaching to it would
        // put it under this question -- see seenTurns.
        var mine = !!(t && state.turn && t.id === state.turn);
        var unread = !!(t && !seenTurns[t.id]);
        if (t && (mine || unread)) {
          // The host took the request after all, so the shared draft it named went
          // with it: the chip has to go, or the next send is refused as a second
          // draft. Only on this branch -- withdrawSend below keeps it, because there
          // the request never arrived.
          if (recovery.sentDraft && recovery.sentDraft.shareIds) forgetAppliedShares(recovery.sentDraft.shareIds);
          recovery.sentDraft = null;
          attachTurn(t.id);
          return;
        }
        // Nothing to attach to. If this page was in the middle of an answer, the
        // finished transcript is on the host; otherwise there was nothing to resume.
        if (state.turn || state.liveView || recovery.sentDraft) settleWithoutTurn(conversation);
        else if (state.generating) { progressDone(); setGenerating(false); }
      })
      .catch(function (e) {
        state.resuming = false;
        if (state.conversation !== conversation) return;
        note('resume-lookup-failed', (e && e.message) || e);
        if (state.turn || state.liveView || recovery.sentDraft) scheduleResume('the lookup failed');
        else if (state.generating) { progressDone(); setGenerating(false); }
      });
    return true;
  }

  function failed(view, e, sentDraft, ctrl) {
    // A reader that was replaced (resumeTurn's supersede, or the user's Stop through
    // detach) has nothing left to say: its rejection arrives after the page has
    // already moved on, and acting on it would flip the Send button and run the share
    // drain in the middle of the reader that replaced it.
    if (ctrl && state.abort !== ctrl) { note('stale-reader', (e && e.name) || e); return; }
    progressDone();
    state.abort = null;
    stopWatchdog();
    if (e && e.name === 'AbortError') { setGenerating(false); return; }
    // The model can change in the narrow interval between the capability refresh and
    // POST, and a selected skill may turn out not to own a host file reader. The
    // server is the final authority; on its vision refusal, put the exact draft back
    // instead of consuming an image that was never processed.
    var routedSetupRefusal = e && e.status === 503
      && e.code === 'routed_workflow_unavailable';
    if (e && (e.code === 'vision_not_ready' || e.code === 'network_disabled' || routedSetupRefusal) && sentDraft) {
      var networkRefusal = e.code === 'network_disabled';
      state.turn = null;
      if (view && view.turn && view.turn.parentNode) view.turn.remove();
      withdrawSend(sentDraft);
      noticeWithAction(
        e.message || (networkRefusal
          ? 'This workflow needs network access before it can start.'
          : routedSetupRefusal
            ? 'This workflow needs additional host setup before it can start.'
            : 'The loaded model cannot process this image.'),
        networkRefusal ? 'Turn on Network' : routedSetupRefusal ? 'Open Settings' : 'Open Models',
        networkRefusal
          ? turnNetworkOn
          : routedSetupRefusal
            ? function () { openRoute('settings'); return true; }
          : function () { openRoute('models'); return true; });
      setGenerating(false);
      state.liveView = null;
      return;
    }
    // A read that broke while a turn is still the app's is this page losing its
    // connection, not the model failing. Saying "The request failed" for that would be
    // telling the user their answer is gone while it is still being written. Take it
    // up again instead; the replay starts from the first frame either way. The model
    // stays "working" on this side meanwhile: Send keeps saying Stop, and the share
    // drain keeps waiting, because that is the truth of it.
    if (state.turn && networkFailure(e)) {
      note('stream-failed', (e && e.message) || e);
      scheduleResume('the stream failed');
      return;
    }
    // The same loss before the turn's id arrived -- during the prefill, which is the
    // minute a user is most likely to leave in. The host has very probably started the
    // turn; ask it which one rather than declaring the request failed and leaving an
    // empty bubble under the question. If nothing is running, the words go back into
    // the composer.
    if (networkFailure(e) && sentDraft && state.conversation) {
      note('send-lost-before-turn-id', (e && e.message) || e);
      recovery.sentDraft = sentDraft;
      scheduleResume('the request lost its stream before the turn was named');
      return;
    }
    if (offerNetworkIfRefused((e && e.message) || '', false)) {
      setGenerating(false);
      state.liveView = null;
      return;
    }
    notice((e && e.message) || 'The request failed.', 'error');
    setGenerating(false);
    state.liveView = null;
  }

  /**
   * The assistant turn a generation is being rendered into.
   *
   * Reused rather than added again when a stream is picked up after being interrupted,
   * because the replay starts at the very first frame: a second bubble would leave the
   * half-written one above it on the screen for good.
   */
  function liveView() {
    var live = state.liveView;
    if (live && live.turn.parentNode) {
      live.bubble.innerHTML = '';
      live.answerSoFar = '';
      Array.prototype.slice.call(live.turn.querySelectorAll('.step, .think, .copy'))
        .forEach(function (n) { n.remove(); });
      live.step = null; live.tool = ''; live.detail = '';
      return live;
    }
    state.liveView = addTurn('assistant', '');
    return state.liveView;
  }

  function stream(body, sentDraft) {
    state.liveView = null;
    var view = liveView();
    setGenerating(true);
    // Immediately, before a single byte comes back: the gap between pressing send
    // and the first frame is itself seconds long on a phone.
    progress('Thinking…');

    var ctrl = new AbortController();
    ctrl.awaitingHeaders = true;
    state.abort = ctrl;
    state.lastByteAt = Date.now();
    armWatchdog();
    note('send', state.conversation);
    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    }).then(null, function (e) { throw transportError(e); }).then(function (res) {
      // The turn this stream is a view of. Held so the Stop button can stop the
      // model rather than merely stopping us listening to it.
      state.turn = res.headers.get('X-TensorAgent-Turn') || null;
      rememberTurn(state.turn);
      ctrl.awaitingHeaders = false;
      state.lastByteAt = Date.now();
      note('send-answered', res.status + ' ' + (state.turn || 'no turn id'));
      if (!res.ok) {
        return res.text().then(function (t) { throw responseError(t, res.status); });
      }
      forgetAppliedShares(body.shareIds);
      return read(res, view, ctrl);
    }).catch(function (e) { failed(view, e, sentDraft, ctrl); });
  }

  /**
   * Pick up a generation that is already running -- one this page did not start, or
   * started and then stopped watching.
   *
   * The host replays every frame from the beginning, so the answer is rebuilt exactly
   * as it would have been had nobody looked away, and then continues live.
   */
  /** Restart notices already shown, by turn id and ordinal; see read(). */
  var restartsSaid = {};
  /**
   * Error notices already shown, by turn id and ordinal. Same reason as the restarts
   * above and it matters more now: a re-attach replays every frame from the first, so
   * a turn whose tool hit a refusal would say so again on every glance at another
   * screen.
   */
  var errorsSaid = {};

  function attachTurn(id) {
    var hadAnswer = !!(state.liveView && state.liveView.answerSoFar);
    var view = liveView();
    setGenerating(true);
    state.turn = id;
    rememberTurn(id);
    progress('Still working…');
    note('attach', id);

    var ctrl = new AbortController();
    state.abort = ctrl;
    state.lastByteAt = Date.now();
    armWatchdog();
    fetch('/api/agent/turns/' + encodeURIComponent(id), { signal: ctrl.signal })
      .then(null, function (e) { throw transportError(e); })
      .then(function (res) {
        if (ctrl !== state.abort) return null;
        if (!res.ok) {
          note('attach-refused', res.status);
          // The turn finished and was forgotten between being announced and being
          // asked for. The saved transcript is the record. A bubble that already had
          // words in it is the one case worth more than removing: the chat is reopened
          // from that transcript so the finished answer replaces the fragment.
          if (hadAnswer) { settleWithoutTurn(state.conversation); return null; }
          if (!view.bubble.innerHTML) view.turn.remove();
          state.abort = null;
          stopWatchdog();
          state.liveView = null;
          state.turn = null;
          progressDone();
          setGenerating(false);
          return null;
        }
        return read(res, view, ctrl);
      })
      .catch(function (e) { failed(view, e, null, ctrl); });
  }

  /** The error sentence out of a JSON refusal body, or the body itself. */
  function reasonOf(text) {
    try {
      var b = JSON.parse(text);
      if (b && typeof b.error === 'string') return b.error;
    } catch (e) {}
    return text;
  }

  /** Preserve a structured refusal code while still presenting its readable error. */
  function responseError(text, status) {
    var error = new Error(reasonOf(text) || ('HTTP ' + status));
    error.status = status;
    try {
      var body = JSON.parse(text);
      if (body && typeof body.code === 'string') error.code = body.code;
    } catch (e) {}
    return error;
  }

  /**
   * Read one event stream into one assistant turn.
   *
   * Shared by the request that starts a generation and by a page attaching to one that
   * is already running, because those differ only in how the response was obtained --
   * every frame after that means the same thing, and two copies of this loop would be
   * two renderings of the same answer that stop agreeing.
   */
  function read(res, view, ctrl) {
    var answer = '', thinking = '', thinkBox = null, thinkBody = null;
    var steps = '', offered = false, draft = '', restarts = 0, errors = 0;
    // What this turn PRODUCED: the files its tools wrote, and a picture it made.
    // Kept so the history entry carries them, because the history is what the next
    // request rewrites the saved transcript from -- an entry that has forgotten the
    // PDF erases the PDF from a chat that had one.
    var made = [], madeSeen = {}, madeImage = null;
    var reader = res.body.getReader(), dec = new TextDecoder(), buf = '';
    // Whether the host said the turn was over. A stream that ends without it did not
    // end because the answer did: the connection went away underneath it.
    var terminal = false;
    // Whether this reader has already asked for the turn again after an EOF that
    // carried no terminal frame; see ended().
    var reattached = false;
    // Painted once per chunk rather than once per frame. A page re-attaching to a
    // running turn is handed every frame so far in one read -- thousands, for a long
    // answer -- and rendering the whole answer for each of them is what made the
    // replay visibly stall on the phone.
    var answerDirty = false, thinkingDirty = false;

    function pump() {
      return reader.read().then(null, function (e) { throw transportError(e); }).then(function (r) {
        // Superseded while waiting: another reader owns the bubble now.
        if (ctrl && ctrl !== state.abort) { note('reader-retired', r.done ? 'eof' : 'data'); return null; }
        state.lastByteAt = Date.now();
        if (r.done) return terminal || !state.turn ? finish() : ended();
        buf += dec.decode(r.value, { stream: true });
        var parts = buf.split('\n');
        buf = parts.pop();
        parts.forEach(function (line) {
          if (line.indexOf('data: ') !== 0) return;
          var f;
          try { f = JSON.parse(line.slice(6)); } catch (e) { return; }
          handle(f);
        });
        paint();
        return pump();
      });
    }
    function paint() {
      if (thinkingDirty && thinkBody) { thinkBody.textContent = thinking; thinkingDirty = false; }
      if (answerDirty) { view.bubble.innerHTML = render(answer); view.answerSoFar = answer; answerDirty = false; }
      if (stickBottom) toBottom();
    }
    /**
     * The stream ended and the host never said the turn was over. Ask whether it still
     * is: a turn cancelled while the app was away ends exactly like this and is
     * finished here with what it had; one still running is taken up again.
     */
    function ended() {
      note('eof-without-done', state.turn);
      var conversation = state.conversation, turn = state.turn;
      state.abort = null;
      stopWatchdog();
      fetchTimed('/api/agent/turns?conversation=' + encodeURIComponent(conversation), 10000)
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (state.conversation !== conversation || state.abort) return;
          var t = d && d.turn;
          // The host still has this turn -- running, or finished while the connection
          // was going away. Either way its replay is the whole answer and ends with the
          // frame that says so, and what this reader has is a fragment. Read it again.
          // Once: the second EOF without a terminal frame is taken at its word, so a
          // host that never sends one cannot loop here.
          if (t && t.id === turn && !reattached) {
            reattached = true;
            if (t.running) scheduleResume('the stream ended before the answer did');
            else attachTurn(turn);
            return;
          }
          finish();
        })
        .catch(function () {
          if (state.conversation !== conversation || state.abort) return;
          scheduleResume('the stream ended before the answer did');
        });
    }
    function handle(f) {
      if (f.done === true) terminal = true;
      if (f.thinking) {
        thinking += f.thinking;
        if (!thinkBox) {
          thinkBox = el('details', 'think');
          thinkBox.appendChild(el('summary', null, 'Reasoning'));
          thinkBody = el('div', 'body');
          thinkBox.appendChild(thinkBody);
          view.turn.insertBefore(thinkBox, view.bubble);
        }
        thinkingDirty = true;
        // The whole of it is one tap away in the box above; the tail is what
        // says, without being asked, what the model is thinking about now.
        progress('Thinking…');
        progressTail(thinking);
      }
      if (f.token || typeof f.replace === 'string') {
        // The answer is on the screen from here on, so the live tail would only
        // be a second, staler copy of it.
        progress('Writing the answer…');
        progressTail('');
      }
      if (f.token) { answer += f.token; answerDirty = true; }
      // Compared against undefined rather than tested for truth: an EMPTY replace is
      // the one that matters most — it is how the host wipes a half-written answer
      // before starting it again, and treating it as "no frame" left the fragment on
      // screen with the fresh answer glued to the end of it.
      if (typeof f.replace === 'string') { answer = f.replace; answerDirty = true; }
      // The host has thrown away a dying engine and is answering again -- carrying
      // the text on screen on, or starting over. Said plainly, because the
      // alternative is an answer that visibly stalls or restarts for no reason the
      // reader can see.
      if (f.restart) {
        thinking = ''; draft = '';
        if (thinkBox) { thinkBox.remove(); thinkBox = null; thinkBody = null; }
        // Said once per restart, not once per READ of it: the host replays every frame
        // from the beginning when this page re-attaches to a running turn, and a
        // notice that came back with every glance at another screen would read as the
        // GPU failing again and again.
        var restartKey = (state.turn || 'live') + ':' + (++restarts);
        if (!restartsSaid[restartKey]) { restartsSaid[restartKey] = true; notice(String(f.restart)); }
        progress(typeof f.replace === 'string' ? 'Starting again…' : 'Carrying on…');
      }
      // Before trace(): the host's record of the call arrives just ahead of the
      // tool's `finished`, and it is what makes that line name a skill instead of
      // a category.
      if (f.skill_step) skillStep(view, f);
      if (f.tool_progress) {
        trace(view, f);
        // A tool call is written a token at a time and can be a whole heredoc;
        // showing it as it is typed is the difference between "it is doing
        // something" and "it is writing THIS".
        if (f.tool_progress === 'writing' && f.text) { draft += f.text; progressTail(draft); }
        else if (f.tool_progress === 'running') {
          // The first `running` frame carries the command and no output; the ones
          // after it carry the output as it is printed. So the draft is emptied
          // once and then refilled with what the command is SAYING, which is the
          // more useful of the two by the time there is any.
          if (f.text) draft += f.text; else draft = '';
          progressTail(draft || String(f.detail || ''));
        }
        else if (f.tool_progress === 'finished') { draft = ''; progressTail(''); }
      }
      if (f.detail || f.output) steps += ' ' + (f.detail || '') + ' ' + (f.output || '');
      if (f.error) {
        steps += ' ' + f.error;
        var errorKey = (state.turn || 'live') + ':' + (++errors);
        if (errorsSaid[errorKey]) {
          offered = true;
        } else {
          errorsSaid[errorKey] = true;
          offered = offerNetworkIfRefused(String(f.error), offered);
          if (!offered) notice(String(f.error), 'error');
        }
      }
      // Guarded deliverables arrive only after the host has proved their package and
      // visible content. They intentionally do not masquerade as a late skill_step,
      // because that frame belongs immediately before its tool's `finished` update.
      if (f.artifact_verified && f.files) {
        f.files.forEach(function (file) { fileLine(view, file); });
      }
      if (f.files) f.files.forEach(function (file) {
        if (!file || !file.url || madeSeen[file.url]) return;
        madeSeen[file.url] = 1;
        made.push({ name: file.name || file.url, bytes: file.bytes || 0, url: file.url });
      });
      if (f.image || f.imageUrl) {
        madeImage = f.imageUrl || f.image;
        // The picture goes under the text, so the text has to be there first.
        if (answerDirty) { view.bubble.innerHTML = render(answer); view.answerSoFar = answer; answerDirty = false; }
        var img = document.createElement('img');
        img.src = madeImage;
        view.bubble.appendChild(img);
      }
    }
    function finish() {
      if (ctrl && ctrl !== state.abort && state.abort) { note('reader-retired', 'finish'); return; }
      paint();
      note('finish', state.turn);
      stopWatchdog();
      recovery.attempts = 0;
      progressDone();
      // A file the last step produced and no `finished` frame came back to render.
      // The download is the thing the user asked for; losing it to a stream that
      // ended a frame early would be the worst possible way to lose it.
      if (view.step && view.step.files) {
        view.step.files.forEach(function (file) { fileLine(view, file); });
        view.step = null;
      }
      offered = offerNetworkIfRefused(answer + ' ' + steps, offered);
      // Everything the turn produced, not only its prose. The host writes the same
      // three things down when the turn ends; the page has to hold them too, because
      // the next request sends this array and the host saves what it is sent.
      var entry = { role: 'assistant', content: answer };
      if (thinking) entry.thinking = thinking;
      if (made.length) entry.artifacts = made;
      if (madeImage) entry.imageUrl = madeImage;
      // Nothing produced is nothing to remember: the host's own record skips an empty
      // turn too, and an empty assistant entry in the history would be sent back to
      // the model as a message it never wrote.
      if (answer || thinking || made.length || madeImage) state.history.push(entry);
      if (answer) addCopy(view.turn, function () { return answer; });
      setGenerating(false);
      state.abort = null;
      state.turn = null;
      state.liveView = null;
      // The list in the menu is titled from the first thing the user said and dated
      // by the last thing that happened, so it is stale the moment a turn ends.
      loadConversations().then(paintNavChats);
    }
    return pump();
  }

  // ---- attachments ---------------------------------------------------------
  function paintChips() {
    var box = $('chips');
    box.innerHTML = '';
    appliedShareOrder.forEach(function (id) {
      if (!appliedShareIds[id]) return;
      var parts = appliedShareParts[id] || {};
      var shared = el('div', 'chip shared');
      shared.appendChild(el('span', 'ic', '↗'));
      shared.appendChild(el('span', 'nm', parts.title || 'Shared item'));
      var remove = el('button', 'x', '✕');
      remove.setAttribute('aria-label', 'Remove shared item');
      remove.disabled = state.generating || state.visionChecking || shareDiscarding;
      remove.addEventListener('click', function () { discardAppliedShare(id, remove); });
      shared.appendChild(remove);
      box.appendChild(shared);
    });
    state.attachments.forEach(function (a, i) {
      var c = el('div', 'chip');
      if (a.mediaType === 'image') {
        var img = document.createElement('img'); img.src = a.url; c.appendChild(img);
      } else {
        c.appendChild(el('span', 'ic', a.mediaType === 'video' ? '🎬' : a.mediaType === 'audio' ? '🎧' : '📄'));
      }
      c.appendChild(el('span', 'nm', a.fileName || a.file));
      var x = el('button', 'x', '✕');
      x.disabled = state.visionChecking || shareDiscarding;
      x.addEventListener('click', function () { state.attachments.splice(i, 1); paintChips(); });
      c.appendChild(x);
      box.appendChild(c);
    });
  }

  function upload(file) {
    var fd = new FormData();
    fd.append('file', file, file.name);
    return fetch('/api/upload', { method: 'POST', body: fd })
      .then(function (r) { return r.json(); })
      .then(function (a) {
        if (!a || !a.ok) { notice((a && a.error) || 'Upload failed', 'error'); return; }
        state.attachments.push(a); paintChips();
      });
  }

  $('file-input').addEventListener('change', function (e) {
    Array.prototype.forEach.call(e.target.files || [], upload);
    e.target.value = '';
  });

  // ---- sheets --------------------------------------------------------------
  function openSheet(id) { $('sheet-bg').classList.add('on'); $(id).classList.add('on'); }
  function closeSheets() {
    $('sheet-bg').classList.remove('on');
    ['attach-sheet', 'skills-sheet', 'model-sheet', 'nav-sheet', 'skill-sheet', 'skill-add-sheet'].forEach(function (s) { $(s).classList.remove('on'); });
  }
  $('sheet-bg').addEventListener('click', closeSheets);

  // Requirement 6: one "+" opens a list, instead of four buttons on a row.
  $('plus').addEventListener('click', function () { openSheet('attach-sheet'); });
  document.querySelectorAll('#attach-sheet .opt').forEach(function (b) {
    b.addEventListener('click', function () {
      var kind = b.getAttribute('data-pick');
      closeSheets();
      // The native side owns the camera, the library and the document picker;
      // the file input is the fallback when the page is open in a browser.
      // The app owns the camera, the library and the document picker. It is asked
      // over the same loopback transport everything else uses, so this file needs no
      // iOS-specific object; MainPage.OnPageEvent turns it into a native picker.
      if (state.native) { post('/api/agent/events', { type: 'pick', what: kind }); return; }
      var input = $('file-input');
      input.setAttribute('accept',
        kind === 'photo' ? 'image/*' : kind === 'video' ? 'video/*' : kind === 'camera' ? 'image/*' : '*/*');
      if (kind === 'camera') input.setAttribute('capture', 'environment'); else input.removeAttribute('capture');
      input.click();
    });
  });

  // ---- the menu ------------------------------------------------------------
  //
  // A drawer from the LEFT edge, not a sheet from the bottom, and it carries the saved
  // chats themselves rather than a row that leads to them. Both changes are the same
  // observation: this is the app's main menu, the thing reached most often through it
  // is a chat the user has already had, and a bottom sheet that fits five rows can
  // only ever offer the word "Chats" -- one more tap, and a whole screen, in front of
  // the one thing being looked for. A left drawer is full height, so the list fits.
  $('menu').addEventListener('click', function () {
    openSheet('nav-sheet');
    // Painted from what is already known so the drawer is never empty for a frame,
    // then again from the store, because a chat may have been renamed or deleted on
    // the native Chats page since.
    paintNavChats();
    loadConversations().then(paintNavChats);
  });

  $('nav-new').addEventListener('click', function () {
    closeSheets();
    openConversation(null);
  });

  function paintNavChats() {
    var box = $('nav-chats');
    if (!box) return;
    box.innerHTML = '';
    if (!state.conversations.length) {
      box.appendChild(el('div', 'navempty', 'No saved chats yet.'));
      return;
    }
    state.conversations.forEach(function (c) {
      var row = el('button', 'navchat' + (c.id === state.conversation ? ' on' : ''));
      row.type = 'button';
      row.appendChild(el('span', 'nm', c.title || 'Chat'));
      row.appendChild(el('span', 'ds', when(c.updatedAt) + ' · '
        + (c.messageCount === 1 ? '1 message' : (c.messageCount || 0) + ' messages')));
      row.addEventListener('click', function () {
        closeSheets();
        if (c.id === state.conversation) return;
        openConversation(c.id).then(function () { paintNavChats(); });
      });
      box.appendChild(row);
    });
  }

  /** A date a person reads at a glance: a time today, a day this week, a date before that. */
  function when(iso) {
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    var now = new Date();
    var sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    if (now - d < 6 * 24 * 3600 * 1000) return d.toLocaleDateString([], { weekday: 'short' });
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }

  document.querySelectorAll('#nav-sheet .opt').forEach(function (b) {
    b.addEventListener('click', function () {
      closeSheets();
      // Two kinds of menu item: the app's native routes, and this page's own sheets.
      // Skills is the second kind — it is a list this page already holds, and sending
      // it through the shell would mean registering a native page to show it.
      var sheet = b.getAttribute('data-sheet');
      if (sheet) {
        // Open it, THEN fill it. Waiting on /api/skills first meant a tap on Skills
        // closed the drawer and showed nothing at all while the request was in flight
        // -- and the loopback server shares this process with the engine, so a turn
        // that is generating is exactly when that request is slow. A failure used to
        // leave the sheet closed for good, with nothing said.
        openSheet(sheet);
        if (sheet === 'skills-sheet') {
          loadSkills().catch(function (e) {
            notice('Skills could not be listed: ' + ((e && e.message) || e), 'error');
          });
        }
        return;
      }
      openRoute(b.getAttribute('data-route'));
    });
  });

  /**
   * Ask the app for one of its own screens.
   *
   * <para>Every failure here used to look the same as a tap that never landed: the
   * drawer had already closed, the request was fired and forgotten, and the user was
   * looking at the chat. Saying so is most of the fix -- a menu that admits it could
   * not open something is one the user can retry deliberately, instead of tapping
   * again into a race.</para>
   */
  function openRoute(route) {
    if (!route) return;
    post('/api/agent/events', { type: 'open-route', route: route })
      .catch(function (e) {
        notice('Could not open ' + route + ': ' + ((e && e.message) || e), 'error');
      });
  }

  modelBtn.addEventListener('click', function () {
    var info = $('model-info');
    info.innerHTML = '';
    info.appendChild(el('div', 'skillrow',
      state.model ? (pretty(state.model) + ' · ' + (state.arch || '?') + ' · ' + (state.backend || '')) : 'No model loaded yet.'));
    openSheet('model-sheet');
  });
  $('open-models').addEventListener('click', function () {
    closeSheets();
    openRoute('models');
  });
  var cta = $('empty-cta');
  if (cta) cta.addEventListener('click', function () { openRoute('models'); });

  // ---- skills --------------------------------------------------------------
  //
  // Two switches, and they answer different questions. The one inside a skill is
  // "use this one in THIS chat"; the one at the top of the list is whether the
  // feature exists at all. The second is a setting rather than a chat property
  // because it is not a per-conversation decision: with twelve skills declaring
  // themselves, having them on costs thousands of prompt tokens on every turn of
  // every chat, and a user who wants a plain assistant wants it for good.
  //
  // It is enforced on the host, not here. The page not sending `skills` would be a
  // request nobody checked; ServerHostingOptions.SkillsEnabled makes the request
  // planner build no plan, so no skill is declared, none is reachable, and a stale
  // page that still names one changes nothing.
  function skillsOn() {
    return !(state.settings && state.settings.skillsEnabled === false);
  }

  function paintSkillsMaster() {
    var box = $('skills-master');
    if (!box) return;
    var on = skillsOn();
    box.checked = on;
    var label = $('skills-master-label');
    if (label) label.textContent = on ? 'Use skills' : 'Skills are off';
    var list = $('skills-list');
    if (list) list.className = on ? '' : 'off';
    var add = $('skill-add');
    if (add) add.style.display = on ? '' : 'none';
  }

  function setSkillsEnabled(on) {
    var next = Object.assign({}, state.settings || {}, { skillsEnabled: !!on });
    // Painted from the intent first: the round trip is a loopback POST, but the
    // switch must not sit in its old position while it happens.
    state.settings = next;
    state.skillSelectionExplicit = !on;
    if (!on) { state.skills = []; paintSkillChips(); }
    paintSkillsMaster();
    return post('/api/agent/settings', next)
      .then(function (r) { return r.json(); })
      .then(function (saved) {
        state.settings = saved || next;
        if (!skillsOn()) {
          state.skills = [];
          state.skillSelectionExplicit = true;
          paintSkillChips();
        }
        paintSkillsMaster();
      })
      .catch(function (e) { notice('That setting could not be saved: ' + e, 'error'); });
  }

  (function () {
    var box = $('skills-master');
    if (box) box.addEventListener('change', function () { setSkillsEnabled(box.checked); });
  })();

  function loadSkills() {
    return fetch('/api/skills').then(function (r) { return r.json(); }).then(function (d) {
      state.catalogSkills = (d && d.skills) || [];
      // The host is the authority on whether the feature is on; the settings copy
      // this page holds may predate a change made anywhere else.
      if (d && typeof d.enabled === 'boolean') {
        state.settings = Object.assign({}, state.settings || {}, { skillsEnabled: d.enabled });
      }
      paintSkillsMaster();
      var list = $('skills-list');
      list.innerHTML = '';
      if (!state.catalogSkills.length) list.appendChild(el('div', 'notice', 'No skills are installed.'));
      state.catalogSkills.forEach(function (s) {
        // The whole row opens the skill. A description is almost always longer
        // than the two lines a list can spare, and truncating it to a tooltip
        // nobody can hover on a phone is the same as not shipping it.
        var row = el('div', 'skillrow');
        var meta = el('div', 'meta');
        meta.appendChild(el('div', 'nm', (state.skills.indexOf(s.name) >= 0 ? '● ' : '') + s.name));
        meta.appendChild(el('div', 'ds', s.description || ''));
        row.appendChild(meta);
        row.appendChild(el('span', 'chev', '›'));
        row.addEventListener('click', function () { openSkill(s); });
        list.appendChild(row);
      });
      return state.catalogSkills;
    });
  }

  // One skill, in full: the name, everything the description says, whether it is
  // on for this chat, and the way to remove it.
  function openSkill(s) {
    closeSheets();
    $('skill-name').textContent = s.name;
    var body = $('skill-body');
    body.innerHTML = '';
    var bits = [];
    if (s.scripts) bits.push(s.scripts + (s.scripts === 1 ? ' script' : ' scripts'));
    if (s.origin) bits.push(String(s.origin));
    if (bits.length) body.appendChild(el('div', 'meta', bits.join(' · ')));
    body.appendChild(document.createTextNode(s.description || 'This skill has no description.'));

    var on = $('skill-on');
    on.checked = state.skills.indexOf(s.name) >= 0;
    on.onchange = function () {
      var i = state.skills.indexOf(s.name);
      if (on.checked && i < 0) state.skills.push(s.name);
      if (!on.checked && i >= 0) state.skills.splice(i, 1);
      state.skillSelectionExplicit = true;
      paintSkillChips();
    };

    $('skill-remove').onclick = function () {
      fetch('/api/skills/' + encodeURIComponent(s.name), { method: 'DELETE' })
        .then(function (r) { return r.json(); })
        .then(function () {
          var i = state.skills.indexOf(s.name);
          if (i >= 0) {
            state.skills.splice(i, 1);
            state.skillSelectionExplicit = true;
            paintSkillChips();
          }
          closeSheets();
          notice(s.name + ' was removed.');
        })
        .catch(function (e) { notice('Could not remove it: ' + e, 'error'); });
    };
    openSheet('skill-sheet');
  }

  // ---- adding a skill ------------------------------------------------------
  $('skill-add').addEventListener('click', function () { closeSheets(); openSheet('skill-add-sheet'); });
  $('skill-zip').addEventListener('click', function () {
    var input = $('skill-zip-input');
    input.value = '';
    input.click();
  });
  $('skill-zip-input').addEventListener('change', function (e) {
    var file = (e.target.files || [])[0];
    if (!file) return;
    var fd = new FormData();
    fd.append('file', file, file.name);
    fetch('/api/skills', { method: 'POST', body: fd })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) { afterInstall(res); })
      .catch(function (err) { notice('That skill could not be installed: ' + err, 'error'); });
  });
  $('skill-fetch').addEventListener('click', function () {
    var url = ($('skill-url').value || '').trim();
    if (!url) return;
    $('skill-fetch').disabled = true;
    post('/api/skills/from-url', { url: url })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) { $('skill-fetch').disabled = false; $('skill-url').value = ''; afterInstall(res); })
      .catch(function (err) { $('skill-fetch').disabled = false; notice('That link did not work: ' + err, 'error'); });
  });
  function afterInstall(res) {
    if (!res.ok) {
      var msg = (res.body && (res.body.error || res.body.message)) || 'The skill was refused.';
      notice(typeof msg === 'string' ? msg : JSON.stringify(msg), 'error');
      return;
    }
    closeSheets();
    // A list install reports both halves; say how many landed and how many did not.
    if (res.body && typeof res.body.count === 'number') {
      var failed = (res.body.failed || []).length;
      notice('Installed ' + res.body.count + (res.body.count === 1 ? ' skill' : ' skills')
        + (failed ? ', ' + failed + ' could not be installed' : '') + '.');
    } else {
      notice('Installed ' + ((res.body && res.body.name) || 'the skill') + '.');
    }
    loadSkills().then(function () { openSheet('skills-sheet'); });
  }
  function paintSkillChips() {
    var box = $('skillchips');
    box.innerHTML = '';
    if (!skillsOn()) return;
    state.skills.forEach(function (n) { box.appendChild(el('span', 'skillchip', '🧩 ' + n)); });
  }

  // ---- composer ------------------------------------------------------------
  function autoGrow() {
    text.style.height = 'auto';
    text.style.height = Math.min(text.scrollHeight, window.innerHeight * 0.26) + 'px';
  }
  text.addEventListener('input', autoGrow);
  send.addEventListener('click', sendMessage);
  $('new').addEventListener('click', function () { openConversation(null); });

  // ---- voice ---------------------------------------------------------------
  //
  // There is no Voice switch. It spent a permanent slot on the only row of
  // chrome this design has, to say something the composer can say by changing
  // shape — and it made speaking a two-step act: find the switch, then find the
  // button. Holding the message box is the gesture every phone messenger has
  // already taught, and it lands the thumb on the hold-to-talk button it just
  // conjured, ready to be held.
  function setVoice(on) {
    state.voice = !!on;
    document.body.classList.toggle('voice', state.voice);
    if (!state.voice) text.focus();
  }

  var pressTimer = 0, pressAt = null, touching = false;
  function cancelPress() {
    if (pressTimer) { clearTimeout(pressTimer); pressTimer = 0; }
    pressAt = null;
  }
  function movePress(x, y) {
    // A press that travels is a scroll or a selection drag, not a hold.
    if (pressAt && (Math.abs(x - pressAt.x) > 10 || Math.abs(y - pressAt.y) > 10)) cancelPress();
  }
  function beginPress(x, y) {
    if (state.voice) return;
    cancelPress();
    pressAt = { x: x, y: y };
    pressTimer = setTimeout(function () {
      pressTimer = 0;
      pressAt = null;
      if (!state.native) {
        // In a browser there is no recogniser to switch to, and a composer that
        // turned into a dead button would be worse than not switching.
        notice('Voice input is only available in the app.', 'error');
        return;
      }
      // Blurring first is what dismisses the selection callout iOS raises for a
      // long press on a text field, and it collapses the keyboard so the button
      // lands where the thumb already is.
      text.blur();
      setVoice(true);
    }, 450);
  }

  // TOUCH events lead and pointer events fill in, rather than pointer alone. WebKit
  // fires `pointercancel` the moment it decides a touch belongs to its own gesture —
  // and a long press on a text field is one of its own gestures, the selection
  // callout — so a timer cancelled by that would never reach the half-second this
  // needs. The touch sequence is not cancelled the same way, so it wins while it is
  // running; pointer events still drive a mouse, and a page opened in a desktop
  // browser behaves the same.
  text.addEventListener('touchstart', function (e) {
    touching = true;
    var t = e.touches[0];
    if (t) beginPress(t.clientX, t.clientY);
  }, { passive: true });
  text.addEventListener('touchmove', function (e) {
    var t = e.touches[0];
    if (t) movePress(t.clientX, t.clientY);
  }, { passive: true });
  ['touchend', 'touchcancel'].forEach(function (n) {
    text.addEventListener(n, function () { touching = false; cancelPress(); });
  });

  text.addEventListener('pointerdown', function (e) { if (!touching) beginPress(e.clientX, e.clientY); });
  text.addEventListener('pointermove', function (e) { if (!touching) movePress(e.clientX, e.clientY); });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (n) {
    text.addEventListener(n, function () { if (!touching) cancelPress(); });
  });
  // A keystroke means they meant to type, whatever the finger was doing.
  text.addEventListener('input', cancelPress);

  abc.addEventListener('click', function () { setVoice(false); });

  function startRec() {
    if (!state.native) { notice('Voice input is only available in the app.', 'error'); return; }
    hold.classList.add('rec');
    $('holdlabel').textContent = 'Listening… release to stop';
    post('/api/agent/events', { type: 'dictate-start' });
  }
  function stopRec() {
    if (!hold.classList.contains('rec')) return;
    $('holdlabel').textContent = 'Transcribing…';
    post('/api/agent/events', { type: 'dictate-stop' });
  }
  // The app says when the session has really ended, because the transcription
  // arrives after the finger lifts and the button must not look idle before it does.
  function dictationEnded() {
    hold.classList.remove('rec');
    $('holdlabel').textContent = 'Hold to talk';
    // Hand back to the text box with what was said already in it. Speaking is how
    // the message STARTS; reading it back, fixing a word and pressing send is how it
    // finishes, and staying in voice mode hides the very text the user needs to
    // check. Only leave voice mode if we are still in it -- the user may have
    // switched already.
    if (state.voice) setVoice(false);
    autoGrow();
    text.focus();
    // Put the caret at the end so typing continues the sentence rather than
    // landing in front of it.
    try { text.setSelectionRange(text.value.length, text.value.length); } catch (e) {}
  }

  // iOS recognises ONE language per session and does not detect which is being
  // spoken, so a bilingual user has to say which -- and the place to say it is next
  // to the button they are about to hold, not three screens away in Settings.
  var LANGS = [
    { id: '', label: 'Auto' },
    { id: 'en-US', label: 'EN' },
    { id: 'zh-CN', label: '中文' }
  ];
  function paintLang() {
    var box = $('lang');
    box.innerHTML = '';
    LANGS.forEach(function (l) {
      var b = el('button', 'langbtn' + (state.speech === l.id ? ' on' : ''), l.label);
      b.type = 'button';
      b.addEventListener('click', function () {
        state.speech = l.id;
        paintLang();
        post('/api/agent/settings', Object.assign({}, state.settings || {}, { speechLanguage: l.id }));
      });
      box.appendChild(b);
    });
  }
  ['pointerdown'].forEach(function (e) { hold.addEventListener(e, function (ev) { ev.preventDefault(); startRec(); }); });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (e) { hold.addEventListener(e, stopRec); });

  // ---- settings ------------------------------------------------------------
  // Requirement 8: "Show reasoning by default" is a setting the composer must
  // actually start from. It used to be read into a control the page then reset.
  // What the host is, and what it is doing about the model the user last used. Both
  // come from the same place because the page needs them at the same moment: as it
  // paints for the first time, before anything can be sent.
  function refreshEngine() {
    return fetch('/api/agent/engine').then(function (r) { return r.json(); }).then(function (e) {
      if (e && typeof e.networkDisabledMessage === 'string') state.netMsg = e.networkDisabledMessage;
      state.modelInfo = (e && e.model) || null;
      paintModelButton();
      if (loadingModel()) watchModelLoad();
      return e;
    }).catch(function () { return null; });
  }

  /**
   * Re-read the settings.
   *
   * `seed` is the difference between "start a chat from these" and "these have
   * changed": reasoning and the skill selection belong to the CHAT once it exists, and
   * this runs again every time the app comes back to the chat page. Overwriting them
   * unconditionally silently unticked a skill the user had chosen for this
   * conversation, every time they glanced at any other screen.
   */
  function applySettings(seed) {
    return fetch('/api/agent/settings').then(function (r) { return r.json(); }).then(function (s) {
      state.settings = s || null;
      if (s && typeof s.speechLanguage === 'string') state.speech = s.speechLanguage;
      if (seed) {
        state.think = thinkDefault();
        state.skills = defaultSkills();
        state.skillSelectionExplicit = state.skills.length > 0;
      }
      // Not only on seed: the setting can be changed from the native Settings
      // screen, and a page that came back with skills still chipped under the
      // composer would be showing something that is no longer true.
      if (!skillsOn()) {
        state.skills = [];
        state.skillSelectionExplicit = true;
      }
      paintSkillChips();
      paintSkillsMaster();
      paintLang();
      return s;
    }).catch(function () { return null; });
  }

  function thinkDefault() {
    return !!(state.settings && state.settings.thinkByDefault);
  }
  function defaultSkills() {
    if (!skillsOn()) return [];
    return state.settings && Array.isArray(state.settings.defaultSkills)
      ? state.settings.defaultSkills.slice() : [];
  }

  // ---- the bridge the native side uses ------------------------------------

  /**
   * Decode what the app sent, which always arrives base64'd. See `__fromHost`.
   */
  function fromBase64Utf8(b64) {
    var binary = atob(b64), escaped = '';
    for (var i = 0; i < binary.length; i++)
      escaped += '%' + ('0' + binary.charCodeAt(i).toString(16)).slice(-2);
    return decodeURIComponent(escaped);
  }

  /**
   * What each host call does with its decoded argument.
   *
   * A table rather than a lookup on window.TensorAgent, so that the app can only reach
   * the calls meant for it, and so each one can say what shape it expects.
   */
  var hostCalls = {
    addAttachment: function (a) { window.TensorAgent.addAttachment(a); },
    insertText: function (a) { window.TensorAgent.insertText(a && a.text); },
    takeShare: function () { window.TensorAgent.takeShare(); },
    notice: function (a) { notice(a && a.text, (a && a.kind) || 'error'); },
    noticeWithSettings: function (a) { window.TensorAgent.noticeWithSettings(a && a.text); },
    openConversation: function (a) { window.TensorAgent.openConversation(a && a.id); },
  };

  window.TensorAgent = {
    notice: notice,

    /**
     * The one door host data comes in through.
     *
     * <para>Everything the app used to send was spliced into a JavaScript SOURCE
     * string: `window.TensorAgent.addAttachment({...json...})`, handed to
     * EvaluateJavaScriptAsync. MAUI then wraps that in
     * `try{JSON.stringify(eval('<script>'))}catch(e){'null'};` -- so the script becomes
     * the contents of a single-quoted literal, and the escapes belong to the LITERAL
     * before they ever belong to the JSON. A file with two lines carries a \n, the
     * outer literal turns it into a real newline, and eval is handed a string that is
     * not closed. That is a SyntaxError, MAUI's own catch turns it into the string
     * "null", and the app throws that away: the upload succeeded, nothing attached,
     * and nobody was told. An apostrophe in a file name did the same thing by closing
     * the literal early.</para>
     *
     * <para>Base64 has no quote, no backslash and no newline in its alphabet, so it
     * passes through that wrapper unchanged. The answer is a word rather than nothing,
     * so the app can tell a call that arrived from one that did not.</para>
     */
    __fromHost: function (name, payload) {
      try {
        var call = Object.prototype.hasOwnProperty.call(hostCalls, name) ? hostCalls[name] : null;
        if (!call) return 'nomethod:' + name;
        call(JSON.parse(fromBase64Utf8(payload)));
        return 'ok';
      } catch (e) {
        return 'failed:' + ((e && e.message) || e);
      }
    },

    /** How many files are chipped under the composer. For tests and the app's own checks. */
    attachmentCount: function () { return state.attachments.length; },

    addAttachment: function (a) {
      if (!a || !a.ok) { notice((a && a.error) || 'Upload failed', 'error'); return; }
      state.attachments.push(a); paintChips();
    },
    insertText: function (t) {
      if (!t) return;
      text.value = text.value && !/\s$/.test(text.value) ? text.value + ' ' + t : text.value + t;
      autoGrow();
      if (!state.voice) text.focus();
    },
    /** Native nudge: the host still owns the payload, so pull it over HTTP. */
    takeShare: function () { takePendingShare(); return true; },
    send: sendMessage,
    stop: stop,
    isGenerating: function () { return state.generating; },
    setThink: function (on) { state.think = !!on; },
    /** Re-read the settings, because Settings is a native page and this one outlives it. */
    refreshSettings: function () { applySettings(false); return true; },
    /**
     * Show a saved chat, or start a new one. Called by the native Chats page instead of
     * navigating the WebView, which used to be how this worked and which threw away
     * every generation in flight along with the page.
     */
    openConversation: function (id) { openConversation(id || null); return true; },
    /** The menu's list of chats, after something outside the page changed it. */
    refreshChats: function () { loadConversations().then(paintNavChats); return true; },
    /** What the app is doing about the model, after the Models page changed it. */
    refreshEngine: function () { refreshEngine(); return true; },
    /** Take the generation back up, after this page was away and could not read it. */
    resumeTurn: resumeTurn,
    /**
     * What this page has been through lately, as one JSON string: the transport
     * events it noted and where it stands. The app writes it into its own trace on
     * every return to the foreground, which is the only way anything the page saw
     * while the app was away ever reaches a log.
     */
    diagnostics: function () {
      return JSON.stringify({
        conversation: state.conversation,
        turn: state.turn,
        attached: !!state.abort,
        delivering: streamLooksAlive(),
        generating: state.generating,
        resuming: state.resuming,
        retries: recovery.attempts,
        visibility: document.visibilityState,
        events: diag.slice(-40),
      });
    },
    /**
     * Open the main menu. For a screenshot, and it exists because neither simctl nor
     * devicectl can touch the screen: without it, the one surface this app's navigation
     * lives on could never be pictured, only measured.
     */
    openMenu: function () {
      openSheet('nav-sheet');
      paintNavChats();
      loadConversations().then(paintNavChats);
      return true;
    },
    setSkills: function (n) {
      state.skills = skillsOn() ? (n || []).slice() : [];
      state.skillSelectionExplicit = true;
      paintSkillChips();
    },
    /** Whether the skills feature is on at all, for the app's own checks. */
    skillsEnabled: skillsOn,
    history: function () { return state.history; },
    // Synchronous on purpose: WKWebView's evaluateJavaScript does not await a
    // promise, so an async function here can never report success to native code.
    refreshModel: function () { refreshModel(); return true; },
    hasModel: function () { return !!state.model; },
    dictationEnded: dictationEnded,
    /** A refusal only the user can lift, with a button that opens iOS Settings. */
    noticeWithSettings: function (msg) {
      noticeWithAction(msg, 'Open Settings', function () {
        post('/api/agent/events', { type: 'open-settings' });
        return true;
      });
    },
    /** The app calls this once at startup so the page knows native pickers exist. */
    nativeReady: function () { state.native = true; return true; },
    /**
     * Knobs for the page's own tests and nothing else: the timings above are what
     * make recovery invisible on a phone and would make a test take a minute.
     */
    __testing: {
      streamTrustMs: function (ms) { STREAM_TRUST_MS = ms; return true; },
      watchdogMs: function (ms) { WATCHDOG_MS = ms; stopWatchdog(); armWatchdog(); return true; },
      resumeDelays: function (list) { RESUME_DELAYS = list.slice(); return true; },
      recovery: function () { return { attempts: recovery.attempts, pending: !!recovery.timer, lastByteAt: state.lastByteAt }; },
    },
  };

  // ---- a file the model made -----------------------------------------------
  //
  // Every link to /api/code/artifacts/... -- the file card this page renders, and the
  // markdown link the model copies into its own answer, which is a different anchor
  // built by render() -- is caught HERE, on the document, rather than by the anchor
  // that happens to have been created.
  //
  // Navigating one inside the app does nothing useful in either direction. The route
  // serves it as an attachment (program-written content must never render in the origin
  // that holds the launch token) and a WKWebView with no download delegate silently
  // drops an attachment; the link also carries target="_blank", which WebKit routes to
  // its create-web-view path rather than to the navigation delegate the app listens on.
  // So the app is ASKED, over the transport every other native request already uses,
  // and it opens the file in a native previewer with a share sheet behind it.
  //
  // Outside the app there is nothing to ask and the browser's own download is right, so
  // this only claims the click when the page is running natively.
  var ARTIFACT_PREFIX = '/api/code/artifacts/';
  // Plain string work rather than `new URL`: this file also runs under a bare
  // JavaScriptCore in the page tests, where URL is a WebKit binding that does not
  // exist. An artifact link is either the route's own path or that path on this
  // origin, and no third spelling reaches here.
  function pageOrigin() {
    var href = String(window.location.href || '');
    var scheme = href.indexOf('://');
    if (scheme < 0) return '';
    var slash = href.indexOf('/', scheme + 3);
    return slash < 0 ? href : href.slice(0, slash);
  }
  function artifactPath(href) {
    var path = href;
    if (href.charAt(0) !== '/') {
      var origin = pageOrigin();
      // Anything on another origin is somebody else's link and stays the browser's.
      if (!origin || href.indexOf(origin + '/') !== 0) return null;
      path = href.slice(origin.length);
    }
    if (path.indexOf(ARTIFACT_PREFIX) !== 0) return null;
    var cut = path.search(/[?#]/);
    return cut < 0 ? path : path.slice(0, cut);
  }
  function artifactHref(node) {
    for (var n = node; n && n !== document; n = n.parentNode) {
      if (n.tagName !== 'A') continue;
      // Both spellings: the attribute a rendered markdown link carries, and the
      // property the file card sets (which a browser reflects into the attribute and
      // the page tests' DOM does not).
      var href = (n.getAttribute && n.getAttribute('href')) || n.href;
      if (!href) return null;
      var path = artifactPath(String(href));
      return path ? { url: path, name: n.textContent || '' } : null;
    }
    return null;
  }
  document.addEventListener('click', function (ev) {
    if (!state.native || ev.defaultPrevented || ev.button) return;
    var hit = artifactHref(ev.target);
    if (!hit) return;
    ev.preventDefault();
    post('/api/agent/events', { type: 'open-file', url: hit.url, name: hit.name });
  });

  // The page heals itself, without needing the app to tell it to. Opening any other
  // screen takes this WebView out of the window, and WebKit suspends a content process
  // whose view is not in one -- so the reader stops mid-answer and the page is told
  // nothing. `visibilitychange` is the event WebKit fires for exactly that transition,
  // in both directions, which makes it the one hook that cannot be missed: it works
  // when the app forgets to call, when the app is backgrounded and comes back, and when
  // the content process was killed and reloaded.
  document.addEventListener('visibilitychange', function () {
    note('visibility', document.visibilityState);
    if (document.visibilityState !== 'visible') return;
    // A moment later rather than inside the handler: WebKit on iOS 18 and later can
    // fail a fetch issued synchronously here with "Load failed" against a server that
    // is perfectly well, and the app's own check of its listener is under way at the
    // same instant. A quarter of a second is invisible; the failure was not.
    setTimeout(function () {
      if (document.visibilityState !== 'visible') return;
      resumeTurn();
      refreshModel();
      takePendingShare();
    }, 250);
  });
  window.addEventListener('pageshow', function () { resumeTurn(); takePendingShare(); });

  // ---- start ---------------------------------------------------------------
  refreshEngine()
    .then(function () { return applySettings(true); })
    .then(refreshModel)
    .then(openAtLaunch)
    .then(function () {
      return post('/api/agent/events', { type: 'ready', conversation: state.conversation });
    })
    .then(function () { shareIntakeReady = true; return takePendingShare(); })
    .catch(function (e) {
      note('start-failed', (e && e.message) || e);
      notice('Could not start: ' + ((e && e.message) || e), 'error');
      // Whatever failed, the page is here and a share waiting on the host must still
      // reach it: leaving this false meant one lost request at startup disabled sharing
      // for the life of the page. The host is told again too -- it is what makes the
      // composer's native pickers work.
      shareIntakeReady = true;
      post('/api/agent/events', { type: 'ready', conversation: state.conversation }).catch(function () {});
      takePendingShare();
    });
})();
