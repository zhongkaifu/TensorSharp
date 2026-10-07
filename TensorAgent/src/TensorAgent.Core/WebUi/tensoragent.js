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

  // ---- the interface's strings ---------------------------------------------
  // /i18n.js, served ahead of this script, holds them in the language the app is
  // showing and has already translated the markup (data-i18n). Every word this
  // script puts on screen comes from there: t(key, { name: value }), and
  // tn(key, count) for text that counts, with keys from page.json. Text the model
  // reads, or that is compared with what it wrote, stays English.
  var I18N = window.TensorAgentI18n;
  var t = I18N && I18N.t ? I18N.t : function (key) { return key; };
  var tn = I18N && I18N.tn ? I18N.tn : function (key) { return key; };
  // The same language for the numbers and dates this script formats itself.
  var LANG = (I18N && I18N.lang) || 'en';

  var $ = function (id) { return document.getElementById(id); };
  var chat = $('chat'), text = $('text'), send = $('send'), busy = $('busy');
  var modelBtn = $('model'), hold = $('hold'), abc = $('abc');

  var state = {
    model: null, arch: null, backend: null, video: null, contextTokens: 0,
    modelContextTokens: 0, visionReady: false,
    acceptsVisionProjector: true,
    visionChecking: false,
    session: null, conversation: null,
    history: [],            // {role, content, attachments}
    attachments: [],        // /api/upload responses
    maskEditing: false,
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
    dictation: false,      // native pickers can exist without a speech recogniser
    composerHint: t('page.composer.message'),
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
      if (!r.ok) throw new Error(t('page.error.httpStatus', { url: url, status: r.status }));
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
  // put nodes into this page -- quotes included, because a link or an image puts
  // the model's text inside a double-quoted attribute, and an unescaped quote there
  // let an answer containing ![x" onerror="...](y) run script in the page that holds
  // the launch token (found 2026-09-30; any page, file or text the model repeats
  // could carry it).
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  // Where a link the model wrote may point: the web, mail, or a path on this origin
  // (a generated file). Anything else -- javascript:, data:, another scheme -- stays
  // text. An image may only come from this origin: one from anywhere else would be
  // fetched the moment the answer rendered, whatever the network setting says.
  var LINK_OK = /^(https?:\/\/|mailto:|\/(?!\/))/i;
  var IMAGE_OK = /^\/(?!\/)/;
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
    // Images before links: an image is a link with a bang in front of it. Both work
    // on the escaped text, so what lands in an attribute cannot close it.
    t = t.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, function (all, alt, src) {
      if (IMAGE_OK.test(src)) return '<img alt="' + alt + '" src="' + src + '">';
      return LINK_OK.test(src) ? '<a href="' + src + '" target="_blank" rel="noopener">' + (alt || src) + '</a>' : all;
    });
    t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (all, label, href) {
      return LINK_OK.test(href) ? '<a href="' + href + '" target="_blank" rel="noopener">' + label + '</a>' : all;
    });
    t = t.replace(/^### (.*)$/gm, '<strong>$1</strong>');
    t = t.replace(/^## (.*)$/gm, '<strong>$1</strong>');
    t = t.replace(/^# (.*)$/gm, '<strong>$1</strong>');
    return t.split(/\n{2,}/).map(block).join('');
  }
  // A GitHub-style table -- a header row, a row of dashes with the same number of
  // cells, then body rows -- becomes a <table>; the lines around it stay a paragraph.
  // Models answer comparisons with tables, and this page printed them as rows of
  // pipes. It runs on text inline() has already escaped and marked up.
  var TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
  function cells(row) {
    var s = row.trim();
    if (s.charAt(0) === '|') s = s.slice(1);
    if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
    return s.split('|').map(function (c) { return c.trim(); });
  }
  function block(p) {
    var lines = p.split('\n');
    for (var i = 0; i + 1 < lines.length; i++) {
      if (lines[i].indexOf('|') < 0 || lines[i + 1].indexOf('-') < 0 || !TABLE_RULE.test(lines[i + 1])) continue;
      var head = cells(lines[i]);
      if (cells(lines[i + 1]).length !== head.length) continue;
      var end = i + 2;
      while (end < lines.length && lines[end].indexOf('|') >= 0) end++;
      var html = '<table><thead><tr>' + head.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr></thead><tbody>';
      for (var r = i + 2; r < end; r++) {
        var row = cells(lines[r]);
        html += '<tr>' + head.map(function (_, c) { return '<td>' + (row[c] || '') + '</td>'; }).join('') + '</tr>';
      }
      return (i > 0 ? paragraph(lines.slice(0, i)) : '') + html + '</tbody></table>'
        + (end < lines.length ? block(lines.slice(end).join('\n')) : '');
    }
    return paragraph(lines);
  }
  function paragraph(lines) {
    return '<p>' + lines.join('<br>') + '</p>';
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
  /** Editing coordinates must use the full-size conversion, never a HEIC thumbnail. */
  function editImageOf(a) {
    return a.editUrl || (a.editFile ? uploadUrl(a.editFile) : previewOf(a));
  }
  function imageSelectionUnavailable(a) {
    if (a.editUnavailableReason) return a.editUnavailableReason;
    if (/\.(heic|heif)$/i.test(a.file || '') || /\.(heic|heif)$/i.test(a.fileName || '')) {
      if (!a.editUrl && !a.editFile) return t('page.imageSelection.reattachHeic');
    }
    return '';
  }
  // The name a picture the image model made is attached under when it is picked up to be
  // edited. English whatever the interface speaks: an attachment's name is also the name
  // its file is given where the model's code can open it, and the transcript keeps it.
  var GENERATED_IMAGE = 'Generated image';
  /** The name to SHOW for an attachment: the generated picture's in the interface's language. */
  function shownName(a) {
    return a.fileName === GENERATED_IMAGE ? t('page.chips.generatedImage') : a.fileName;
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
    var edit = a.editFile || (a.editUrl ? uploadName(a.editUrl) : '');
    if (edit) chip.editFile = edit;
    if (a.editUnavailableReason) chip.editUnavailableReason = a.editUnavailableReason;
    if (a.frames && a.frames.length) chip.frames = a.frames.slice();
    if (a.fileBacked === true) chip.fileBacked = true;
    if (typeof a.pageCount === 'number') chip.pageCount = a.pageCount;
    if (typeof a.extractedPageCount === 'number') chip.extractedPageCount = a.extractedPageCount;
    if (typeof a.renderedAsImages === 'boolean') chip.renderedAsImages = a.renderedAsImages;
    if (a.maskPath) {
      chip.maskPath = a.maskPath; chip.maskMode = a.maskMode || 'grayscale';
      chip.maskFeather = a.maskFeather || 0; chip.maskCrop = !!a.maskCrop;
      if (a.maskInvert) chip.maskInvert = true;
      if (typeof a.maskCropPadding === 'number') chip.maskCropPadding = a.maskCropPadding;
    }
    return chip;
  }

  function clearImageSelection(attachment) {
    delete attachment.maskPath; delete attachment.maskMode; delete attachment.maskFeather; delete attachment.maskCrop;
    delete attachment.maskInvert; delete attachment.maskCropPadding;
  }

  function imageSource(attachments) {
    return attachments.filter(function (a) { return a.mediaType === 'image'; })[0];
  }

  function hasActiveImageSelection(attachment) {
    return attachment && attachment.maskPath && attachment._maskActive !== false;
  }

  function promoteImageSource(attachment) {
    var index = state.attachments.indexOf(attachment);
    if (index > 0) state.attachments.unshift(state.attachments.splice(index, 1)[0]);
  }

  function deactivateImageSelections() {
    state.attachments.forEach(function (a) { if (a.mediaType === 'image') a._maskActive = false; });
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
      img.src = previewOf(a); img.alt = shownName(a) || t('page.attachment.imageAlt');
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

  /**
   * The clip a video model made, and its soundtrack when the host kept that as a file
   * of its own. Built the same way for a turn as it finishes and for a reopened chat.
   */
  function clipNode(src) {
    var video = document.createElement('video');
    // Inline, or an iPhone takes it fullscreen the moment it plays. Looped, because a
    // generated clip lasts only seconds and a single play is easy to miss.
    // 'metadata' rather than an attachment's 'none': the header of a clip a few seconds
    // long costs next to nothing to read, and with it read the controls can show the
    // clip's length before it plays.
    video.controls = true; video.playsInline = true; video.loop = true;
    video.preload = 'metadata';
    if (src) video.src = src;
    return video;
  }
  function soundNode(src) {
    var audio = document.createElement('audio');
    audio.controls = true; audio.preload = 'metadata';
    if (src) audio.src = src;
    return audio;
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
      made.src = extra.imageUrl; made.alt = t('page.image.generatedAlt');
      b.appendChild(made);
      imageActions(b, made, extra.imageUrl, newestImageRequest());
    }
    if (extra && extra.videoUrl) {
      b.appendChild(clipNode(extra.videoUrl));
      if (extra.audioUrl) b.appendChild(soundNode(extra.audioUrl));
    }
    turn.appendChild(b);
    chat.appendChild(turn);
    // The files this turn produced, put back the way the live turn showed them.
    // They are the point of the turn far more often than the prose is.
    if (extra && extra.artifacts) extra.artifacts.forEach(function (f) { fileLine({ turn: turn, bubble: b }, f); });
    var view = { turn: turn, bubble: b };
    if (role === 'assistant' && extra && extra.stats) showTurnStats(view, extra.stats);
    if (stickBottom) toBottom();
    return view;
  }

  // These are the engine's terminal counters, including reasoning and tool-call
  // generations. Counting the answer fragments here would under-report both tokens
  // and decode speed. Older transcripts and synthetic error frames have no counters.
  function statsOf(s) {
    function number(n) { return typeof n === 'number' && isFinite(n) && n >= 0; }
    function count(n) { return number(n) && Math.floor(n) === n && n <= 2147483647; }
    if (!s || !count(s.tokenCount) || !number(s.elapsed) || !number(s.tokPerSec)) return null;
    var stats = { tokenCount: s.tokenCount, elapsed: s.elapsed, tokPerSec: s.tokPerSec };
    ['promptTokens', 'kvReusedTokens'].forEach(function (key) {
      if (count(s[key])) stats[key] = s[key];
    });
    if (number(s.kvReusePercent)) stats.kvReusePercent = s.kvReusePercent;
    if (s.aborted === true) stats.aborted = true;
    if (s.truncated === true) stats.truncated = true;
    return stats;
  }

  function showTurnStats(view, value) {
    var stats = statsOf(value);
    if (!stats) return;
    var line = t('page.turn.stats', {
      tokens: stats.tokenCount, seconds: stats.elapsed.toFixed(1), speed: stats.tokPerSec.toFixed(1),
    });
    if (stats.promptTokens > 0) {
      var reused = stats.kvReusedTokens || 0;
      var percent = typeof stats.kvReusePercent === 'number'
        ? stats.kvReusePercent : 100 * reused / stats.promptTokens;
      line += ' · ' + t('page.turn.kvStats', { reused: reused, prompt: stats.promptTokens, percent: percent.toFixed(0) });
    }
    if (stats.truncated) line += ' · ' + t('page.turn.truncated');
    if (stats.aborted) line += ' · ' + t('page.turn.aborted');
    if (!view.stats || view.stats.parentNode !== view.turn) {
      view.stats = el('div', 'turn-stats');
      view.turn.appendChild(view.stats);
    }
    view.stats.textContent = line;
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
  // Each pair is two keys: the label while the call is being written, and while it runs.
  var TOOL_LABEL = {
    shell: ['page.tool.shell.writing', 'page.tool.shell.running'],
    apply_patch: ['page.tool.applyPatch.writing', 'page.tool.applyPatch.running'],
    read_file: ['page.tool.readFile.writing', 'page.tool.readFile.running'],
    edit_file: ['page.tool.editFile.writing', 'page.tool.editFile.running'],
    write_file: ['page.tool.writeFile.writing', 'page.tool.writeFile.running'],
    skills_list: ['page.tool.skillsList.writing', 'page.tool.skillsList.running'],
    skills_read: ['page.tool.skillsRead.writing', 'page.tool.skillsRead.running'],
    skills_run: ['page.tool.skillsRun.writing', 'page.tool.skillsRun.running'],
    spawn_agent: ['page.tool.spawnAgent.writing', 'page.tool.spawnAgent.running'],
    wait_agent: ['page.tool.waitAgent.writing', 'page.tool.waitAgent.running'],
    send_input: ['page.tool.sendInput.writing', 'page.tool.sendInput.running'],
    close_agent: ['page.tool.closeAgent.writing', 'page.tool.closeAgent.running'],
    list_agents: ['page.tool.listAgents.writing', 'page.tool.listAgents.running'],
  };
  function labelFor(tool, phase) {
    var pair = TOOL_LABEL[tool];
    var writing = phase === 'writing';
    if (pair) return t(pair[writing ? 0 : 1]);
    if (!tool) return writing ? t('page.tool.unnamed.writing') : t('page.tool.unnamed.running');
    var name = String(tool).replace(/_/g, ' ');
    return writing ? t('page.tool.unlisted.writing', { tool: name }) : t('page.tool.unlisted.running', { tool: name });
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
    a.textContent = '📄 ' + (file.name || t('page.trace.unnamedFile'))
      + (file.bytes ? ' · ' + t('page.trace.kilobytes', { size: Math.max(1, Math.round(file.bytes / 1024)) }) : '');
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
        + (secs ? ' · ' + t('page.trace.seconds', { seconds: secs }) : '');

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
    progress(elapsed
      ? t('page.activity.stepElapsed', { label: labelFor(tool, phase), seconds: elapsed })
      : t('page.activity.step', { label: labelFor(tool, phase) }));
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
    var c = el('button', 'copy', t('page.turn.copy'));
    c.addEventListener('click', function () {
      var copied = getText();
      if (navigator.clipboard) navigator.clipboard.writeText(copied);
      c.textContent = t('page.turn.copied'); setTimeout(function () { c.textContent = t('page.turn.copy'); }, 1200);
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
        b.textContent = ok === false ? t('page.action.failed') : t('page.action.done');
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
        notice(t('page.network.on'));
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
    noticeWithAction(t('page.network.refused'), t('page.action.turnOnNetwork'), turnNetworkOn);
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
    // What a video model can be given besides its description. The host reports it
    // for a video model and null for every other one.
    state.video = (d && d.video) || null;
    paintComposerHint();
    paintModelButton();
    paintChips();
  }

  function paintComposerHint() {
    // An image model takes a description, or a photo and what to change about it.
    text.placeholder = makesVideo() ? videoPlaceholder(state.video)
      : makesImages()
        ? t('page.composer.image')
        : state.native && !state.dictation ? state.composerHint : t('page.composer.messageOrTalk');
  }

  // Qwen-Image: the host turns a message into a picture rather than an answer
  // (ImageTurns), on the same /api/chat route and turn machinery as a reply.
  function makesImages() { return state.arch === 'qwen_image' || state.arch === 'qwen-image'; }

  // MiniMax-H3: the host films the message instead (VideoTurns), the same way. Known by
  // the capability the host reports, not by an architecture name: the two checkpoints
  // share one architecture and take different things.
  function makesVideo() { return !!state.video; }

  // The keyframes checkpoint starts the clip from a photo (and ends it on a second
  // one); the references checkpoint puts the people, things and sounds it is shown
  // into a scene of its own. The composer says which, before anything is attached.
  function videoPlaceholder(v) {
    if (v.supportsReferenceConditioning) return t('page.composer.videoWithReferences');
    if (v.supportsImageConditioning) return t('page.composer.videoFromPhoto');
    return t('page.composer.video');
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
          details.push(t('page.model.contextOfModel', { tokens: shortTokens(state.modelContextTokens) }));
          details.push(t('page.model.contextActive', { tokens: shortTokens(state.contextTokens) }));
        } else {
          details.push(t('page.model.contextWindow', { tokens: shortTokens(state.modelContextTokens) }));
        }
      } else if (state.contextTokens > 0) {
        details.push(t('page.model.contextActiveWindow', { tokens: shortTokens(state.contextTokens) }));
      }
      if (details.length) modelBtn.appendChild(el('span', 'sub', '  ' + details.join(' · ')));
    } else if (loadingModel()) {
      modelBtn.className = 'empty';
      modelBtn.textContent = state.modelInfo.name
        ? t('page.model.loading', { model: state.modelInfo.name })
        : t('page.model.loadingUnnamed');
    } else {
      modelBtn.className = 'empty';
      modelBtn.textContent = t('page.model.none');
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

  // Loading may finish after the user returns from Models. The old model can still
  // be reported while its replacement loads, so follow the host's load status and
  // read capabilities after it, including on the last poll.
  function watchModelLoad() {
    if (state.modelWatch) return;
    var deadline = Date.now() + 5 * 60 * 1000;
    var refreshing = false;
    function stopWatching() {
      clearInterval(state.modelWatch);
      state.modelWatch = 0;
    }
    state.modelWatch = setInterval(function () {
      if (Date.now() > deadline) { stopWatching(); return; }
      if (refreshing) return;
      refreshing = true;
      refreshEngine().then(refreshModel).then(function (model) {
        if (model && !loadingModel()) stopWatching();
      }).finally(function () { refreshing = false; });
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
    var b = el('button', 'cta', state.model || loadingModel() ? t('page.empty.manageModels') : t('page.empty.chooseModel'));
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
        { artifacts: m.artifacts, imageUrl: m.imageUrl, videoUrl: m.videoUrl, audioUrl: m.audioUrl, stats: m.stats });
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
          ? t('page.empty.newChat')
          : t('page.empty.intro'));
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
      var why = t('page.chat.openFailed', { error: (e && e.message) || e });
      if (chat.querySelectorAll('.turn').length) notice(why, 'error');
      else emptyState(why);
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
      notice(t('page.share.alreadySending'));
      return;
    }
    shareDiscarding = true;
    send.disabled = true;
    paintChips();
    post('/api/agent/share/discard', { id: id })
      .then(function (r) { return r.json(); })
      .then(function (body) {
        if (!body || body.ok !== true) throw new Error(t('page.share.retained'));
        var parts = appliedShareParts[id] || {};
        var sharedAttachments = Array.isArray(parts.attachments) ? parts.attachments : [];
        var previousSource = imageSource(state.attachments);
        state.attachments = state.attachments.filter(function (current) {
          return !sharedAttachments.some(function (shared) {
            return current === shared || (current && shared && current.file === shared.file);
          });
        });
        if (previousSource && state.attachments.indexOf(previousSource) < 0) deactivateImageSelections();
        text.value = removeTrackedSharedText(text.value, parts);
        forgetAppliedShares([id]);
        autoGrow();
        notice(t('page.share.removed'));
      })
      .catch(function (e) {
        notice(t('page.share.removeFailed', { error: (e && e.message) || e }), 'error');
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
      return Promise.reject(new Error(t('page.share.empty')));
    var id = typeof share.id === 'string' ? share.id : '';
    if (!id) return Promise.reject(new Error(t('page.share.noId')));
    if (appliedShareOrder.length && !appliedShareIds[id])
      return Promise.reject(new Error(t('page.share.oneAtATime')));

    var incomingText = typeof share.text === 'string' ? share.text : '';
    var attachments = Array.isArray(share.attachments) ? share.attachments : [];
    var valid = [], problems = [];
    attachments.forEach(function (a) {
      if (validSharedAttachment(a)) valid.push(a);
      else problems.push((a && a.error) || t('page.share.fileNotAttached'));
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
            throw new Error(t('page.share.noNewChat'));
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
        if (!id) throw new Error(t('page.share.noId'));

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
        notice(t('page.share.openFailed', { error: (e && e.message) || e }), 'error');
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
    if (pendingUploadCount) {
      notice(t('page.send.waitForUploads'));
      return;
    }
    if (state.maskEditing) { notice(t('page.send.finishSelection')); return; }
    if (makesImages() && loraChoice !== null) {
      notice(t('page.send.waitForLoras')); return;
    }
    var source = imageSource(state.attachments);
    if (!makesImages() && hasActiveImageSelection(source)) {
      noticeWithAction(
        t('page.send.selectionNeedsQwen'),
        t('page.action.openModels'), function () { openRoute('models'); return true; });
      return;
    }
    if (state.visionChecking) { note('send-refused', 'vision check in flight'); return; }
    if (shareDiscarding) {
      note('send-refused', 'share discard in flight');
      notice(t('page.send.waitForShareRemoval'));
      return;
    }
    var typed = text.value.trim();
    // A video is filmed from its description. Photos, clips and sounds are only what it
    // starts from or features, and sent alone their file names would be the script.
    if (makesVideo() && !typed) {
      note('send-refused', 'video without a description');
      notice(t('page.send.describeVideo'));
      return;
    }
    if (!typed && !state.attachments.length) { note('send-refused', 'nothing to send'); return; }
    // A photo with no words would send its file name as the edit instruction.
    if (makesImages() && !typed) {
      note('send-refused', 'image edit without an instruction');
      notice(t('page.send.describeEdit'));
      return;
    }
    if (!state.model) {
      note('send-refused', loadingModel() ? 'model loading' : 'no model');
      // Two different answers, because they ask for two different things. A model
      // that is loading needs a few seconds; no model at all needs a download.
      if (loadingModel()) notice(state.modelInfo.name
        ? t('page.send.modelLoading', { model: state.modelInfo.name })
        : t('page.send.modelLoadingUnnamed'));
      else openSheet('model-sheet');
      return;
    }

    var atts = state.attachments.slice();
    var msg = messageFor(typed, atts);
    var nextHistory = state.history.concat([msg]);

    // Capability can change while this long-lived WKWebView is hidden on the Models
    // page. Re-read it immediately before every image-bearing request, while the
    // composer is still intact. The server repeats this check authoritatively.
    // Not for an image or video model: a photo is what it edits or films from, not
    // something it has to see. The host refuses an edit itself when the vision file is
    // missing, and checks what a video was given against what its checkpoint takes.
    if (!makesImages() && !makesVideo() && nextHistory.some(function (m) { return m && m.imagePaths && m.imagePaths.length; })) {
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
            t('page.send.modelGone'),
            t('page.action.openModels'),
            function () { openRoute('models'); return true; });
          return;
        }
        var hostFileSkillCanDecide = !state.acceptsVisionProjector
          && skillsOn() && state.skills.length > 0;
        if (!state.visionReady && !hostFileSkillCanDecide) {
          var message = state.acceptsVisionProjector
            ? t('page.send.visionFileMissing')
            : t('page.send.noVision');
          noticeWithAction(
            message,
            t('page.action.openModels'),
            function () { openRoute('models'); return true; });
          return;
        }
        commitMessage(typed, atts, msg);
      });
      return;
    }

    commitMessage(typed, atts, msg);
  }

  function commitMessage(typed, atts, msg) {
    var userView = addTurn('user', typed, atts);
    // A capability refresh is asynchronous. Preserve anything newly typed or
    // attached during that short check instead of clearing it with the sent draft.
    if (text.value.trim() === typed) text.value = '';
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
      text: typed, attachments: atts, message: msg, turn: userView.turn,
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
        if (stillImagePaths.length === 1 && hasActiveImageSelection(a)) {
          msg.maskPath = a.maskPath; msg.maskMode = a.maskMode || 'grayscale';
          msg.maskFeather = a.maskFeather || 0; msg.maskCrop = !!a.maskCrop;
          if (a.maskInvert) msg.maskInvert = true;
          if (typeof a.maskCropPadding === 'number') msg.maskCropPadding = a.maskCropPadding;
        }
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
    if (atts.length) {
      var source = imageSource(atts);
      msg.attachments = atts.map(function (a) {
        var chip = chipOf(a);
        // Other photos keep their selections in the draft, but only the source's
        // selection belongs to this edit and its saved conversation.
        if (a !== source || !hasActiveImageSelection(a)) clearImageSelection(chip);
        return chip;
      });
    }
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
        .catch(function (e) { notice(t('page.send.stopFailed', { error: (e && e.message) || e }), 'error'); });
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
    notice(t('page.send.connectionLost'), 'error');
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
          ? t('page.send.needsNetwork')
          : routedSetupRefusal
            ? t('page.send.needsSetup')
            : t('page.send.cannotProcessImage')),
        networkRefusal ? t('page.action.turnOnNetwork')
          : routedSetupRefusal ? t('page.action.openSettings') : t('page.action.openModels'),
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
    notice((e && e.message) || t('page.send.failed'), 'error');
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
      if (live.stats) { live.stats.remove(); live.stats = null; }
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
    // "Thinking…" (or "Drawing…", "Filming…") immediately, before a single byte comes
    // back: the gap between pressing send and the first frame is itself seconds long
    // on a phone.
    progress(makesVideo() ? t('page.activity.filming') : makesImages() ? t('page.activity.drawing') : t('page.activity.thinking'));

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
    progress(t('page.activity.stillWorking'));
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
   * What the strip says while a video model works, by the stage the host reports. A
   * clip takes minutes, so the denoising stage counts its steps and, once the host can
   * tell, says about how long is left (its eta is -1 until then).
   */
  function filmingLabel(f) {
    switch (f.video_phase) {
      case 'text-encode': return t('page.activity.readingDescription');
      case 'denoise':
        return (f.video_steps
          ? t('page.activity.filmingStep', { step: f.video_step, steps: f.video_steps })
          : t('page.activity.filming'))
          + timeLeft(Number(f.eta));
      case 'vae-decode': return t('page.activity.developingFrames');
      case 'audio-decode': return t('page.activity.addingSound');
      case 'encode': return t('page.activity.savingVideo');
      default: return t('page.activity.filming');
    }
  }
  /** Minutes past a minute and a half, whole seconds under it, nothing when unknown. */
  function timeLeft(s) {
    if (!(s > 0)) return '';
    return ' · ' + (s > 90 ? t('page.activity.minutesLeft', { minutes: Math.round(s / 60) })
      : t('page.activity.secondsLeft', { seconds: Math.max(1, Math.round(s)) }));
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
    // What this turn PRODUCED: the files its tools wrote, and a picture or a clip it
    // made. Kept so the history entry carries them, because the history is what the
    // next request rewrites the saved transcript from -- an entry that has forgotten
    // the PDF erases the PDF from a chat that had one.
    var made = [], madeSeen = {}, madeImage = null, madeVideo = null, madeAudio = null;
    var reader = res.body.getReader(), dec = new TextDecoder(), buf = '';
    // Whether the host said the turn was over. A stream that ends without it did not
    // end because the answer did: the connection went away underneath it.
    var terminal = false, stats = null;
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
      if (f.done === true) {
        terminal = true;
        stats = statsOf(f);
        // Image/video generators use zero token counters as protocol placeholders.
        // A text turn that also produces media still has its real model counters.
        if (stats && stats.tokenCount === 0 && (madeImage || madeVideo)) stats = null;
        if (stats) showTurnStats(view, stats);
      }
      if (f.thinking) {
        thinking += f.thinking;
        if (!thinkBox) {
          thinkBox = el('details', 'think');
          thinkBox.appendChild(el('summary', null, t('page.turn.reasoning')));
          thinkBody = el('div', 'body');
          thinkBox.appendChild(thinkBody);
          view.turn.insertBefore(thinkBox, view.bubble);
        }
        thinkingDirty = true;
        // The whole of it is one tap away in the box above; the tail is what
        // says, without being asked, what the model is thinking about now.
        progress(t('page.activity.thinking'));
        progressTail(thinking);
      }
      if (f.token || typeof f.replace === 'string') {
        // The answer is on the screen from here on, so the live tail would only
        // be a second, staler copy of it.
        progress(t('page.activity.writingAnswer'));
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
        stats = null;
        if (view.stats) { view.stats.remove(); view.stats = null; }
        if (thinkBox) { thinkBox.remove(); thinkBox = null; thinkBody = null; }
        // Said once per restart, not once per READ of it: the host replays every frame
        // from the beginning when this page re-attaches to a running turn, and a
        // notice that came back with every glance at another screen would read as the
        // GPU failing again and again.
        var restartKey = (state.turn || 'live') + ':' + (++restarts);
        if (!restartsSaid[restartKey]) { restartsSaid[restartKey] = true; notice(String(f.restart)); }
        progress(typeof f.replace === 'string' ? t('page.activity.startingAgain') : t('page.activity.carryingOn'));
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
      // An image model's turn (ImageTurns on the host): steps while the picture
      // denoises, some carrying a small preview, then the finished picture. One <img>
      // is refreshed in place, so the preview becomes the picture instead of a new
      // image being stacked under it for every step.
      if (typeof f.image_step === 'number') {
        // With LoRA plug-ins the host names them on every step (ImageTurns.Translate).
        var loras = f.image_loras && f.image_loras.length ? f.image_loras.join(' + ') : null;
        if (f.image_steps) {
          progress(loras !== null
            ? t('page.activity.drawingWithStep', { loras: loras, step: f.image_step, steps: f.image_steps })
            : t('page.activity.drawingStep', { step: f.image_step, steps: f.image_steps }));
        } else {
          progress(loras !== null ? t('page.activity.drawingWith', { loras: loras }) : t('page.activity.drawing'));
        }
        if (f.preview) pictureOf(view).src = f.preview;
      }
      if (f.image || f.imageUrl) {
        madeImage = f.imageUrl || f.image;
        // The picture goes under the text, so the text has to be there first.
        if (answerDirty) { view.bubble.innerHTML = render(answer); view.answerSoFar = answer; answerDirty = false; }
        pictureOf(view).src = madeImage;
      }
      // A video model's turn (VideoTurns on the host): the description is read, the
      // clip denoises step by step, then its frames, its sound and the MP4 are made.
      // No preview along the way; the finished clip arrives once, at the end.
      if (typeof f.video_step === 'number') progress(filmingLabel(f));
      if (f.videoUrl) {
        madeVideo = f.videoUrl;
        // Only when the soundtrack is a file of its own. Normally it is inside the MP4,
        // and the clip plays it.
        madeAudio = f.audioUrl || null;
        // The clip goes under the text, so the text has to be there first.
        if (answerDirty) { view.bubble.innerHTML = render(answer); view.answerSoFar = answer; answerDirty = false; }
        clipOf(view).src = madeVideo;
        if (madeAudio) soundOf(view).src = madeAudio;
      }
    }
    // The one picture a turn shows, made on first use and made again if a repaint of
    // the bubble's text removed it.
    function pictureOf(v) {
      if (!v.picture || v.picture.parentNode !== v.bubble) {
        v.picture = document.createElement('img');
        v.picture.alt = t('page.image.generatedAlt');
        v.bubble.appendChild(v.picture);
      }
      return v.picture;
    }
    // The clip and its soundtrack, one of each per bubble and made the same way. A
    // re-attach replays every frame from the first into this bubble, the url frame
    // included, and the bubble must still end with one player, not two.
    function clipOf(v) {
      if (!v.clip || v.clip.parentNode !== v.bubble) {
        v.clip = clipNode();
        v.bubble.appendChild(v.clip);
      }
      return v.clip;
    }
    function soundOf(v) {
      if (!v.sound || v.sound.parentNode !== v.bubble) {
        v.sound = soundNode();
        v.bubble.appendChild(v.sound);
      }
      return v.sound;
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
      // Everything the turn produced, not only its prose. The host saves it when
      // the turn ends; the page has to hold it too, because
      // the next request sends this array and the host saves what it is sent.
      var entry = { role: 'assistant', content: answer };
      if (stats) entry.stats = stats;
      if (thinking) entry.thinking = thinking;
      if (made.length) entry.artifacts = made;
      if (madeImage) {
        entry.imageUrl = madeImage;
        imageActions(view.bubble, pictureOf(view), madeImage, newestImageRequest());
      }
      if (madeVideo) entry.videoUrl = madeVideo;
      if (madeAudio) entry.audioUrl = madeAudio;
      // Nothing produced is nothing to remember: the host's own record skips an empty
      // turn too, and an empty assistant entry in the history would be sent back to
      // the model as a message it never wrote.
      if (answer || thinking || made.length || madeImage || madeVideo) state.history.push(entry);
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
  function newestImageRequest() {
    for (var i = state.history.length - 1; i >= 0; i--)
      if (state.history[i].role === 'user') return state.history[i];
    return null;
  }

  function imageActions(bubble, picture, resultUrl, request) {
    if (bubble.querySelector('.image-edit-actions')) return;
    var actions = el('div', 'image-edit-actions');
    var source = request && request.stillImagePaths && request.stillImagePaths[0];
    if (source) {
      var sourceAttachment = (request.attachments || []).filter(function (a) { return a.file === source; })[0];
      var originalUrl = sourceAttachment ? editImageOf(sourceAttachment) : uploadUrl(source);
      var original = false;
      var compare = el('button', 'filechip', t('page.image.compareOriginal')); compare.type = 'button';
      compare.setAttribute('aria-pressed', 'false');
      compare.addEventListener('click', function () {
        original = !original; picture.src = original ? originalUrl : resultUrl;
        picture.alt = original ? t('page.image.originalAlt') : t('page.image.generatedAlt');
        compare.textContent = original ? t('page.image.showResult') : t('page.image.compareOriginal');
        compare.setAttribute('aria-pressed', String(original));
      });
      actions.appendChild(compare);
    }
    var again = el('button', 'filechip', source ? t('page.image.editAgain') : t('page.image.edit')); again.type = 'button';
    again.addEventListener('click', function () {
      if (state.generating || state.maskEditing || pendingUploadCount || state.attachments.length || text.value.trim()) {
        notice(t('page.image.draftInTheWay')); return;
      }
      if (source) {
        var saved = request.attachments || [];
        state.attachments = saved.map(function (a) { return Object.assign({}, a); });
        request.stillImagePaths.forEach(function (path) {
          if (!state.attachments.some(function (a) { return a.file === path && a.mediaType === 'image'; }))
            state.attachments.push({ file: path, fileName: path, mediaType: 'image' });
        });
        var target = state.attachments.filter(function (a) { return a.file === source && a.mediaType === 'image'; })[0];
        promoteImageSource(target);
        // Top-level fields record what this turn actually applied. Older saved
        // chips may also contain dormant selections that must not become active.
        state.attachments.forEach(clearImageSelection);
        if (request.maskPath) {
          target._maskActive = true;
          target.maskPath = request.maskPath;
          target.maskMode = request.maskMode || 'grayscale';
          target.maskFeather = request.maskFeather || 0;
          target.maskCrop = !!request.maskCrop;
          target.maskInvert = !!request.maskInvert;
          if (typeof request.maskCropPadding === 'number') target.maskCropPadding = request.maskCropPadding;
        }
        text.value = request.content || '';
      } else {
        state.attachments = [{ file: uploadName(resultUrl), fileName: GENERATED_IMAGE, mediaType: 'image', url: resultUrl }];
      }
      autoGrow(); paintChips(); text.focus();
    });
    actions.appendChild(again);
    // Keep the finished file's URL even while Compare original changes the preview.
    // Browsers can download this same-origin link; the app's WebView needs a native
    // save/share picker, since it has no browser download manager.
    var download = el('a', 'filechip image-download', t('page.image.download'));
    download.href = resultUrl;
    download.download = uploadName(resultUrl);
    download.addEventListener('click', function (ev) {
      if (!state.native || ev.defaultPrevented || ev.button) return;
      ev.preventDefault();
      post('/api/agent/events', { type: 'save-image', url: resultUrl, name: download.download })
        .catch(function (error) {
          notice(t('page.image.downloadFailed', { error: (error && error.message) || error }), 'error');
        });
    });
    actions.appendChild(download);
    bubble.appendChild(actions);
  }

  function selectImageArea(attachment) {
    if (state.maskEditing || state.generating) return;
    var unavailable = imageSelectionUnavailable(attachment);
    if (unavailable) { notice(unavailable, 'error'); return; }
    if (!window.TensorSharpMaskEditor) { notice(t('page.imageSelection.editorUnavailable'), 'error'); return; }
    state.maskEditing = true; paintChips();
    var conversation = state.conversation;
    window.TensorSharpMaskEditor.open({
      sourceUrl: editImageOf(attachment), maskUrl: attachment.maskPath ? uploadUrl(attachment.maskPath) : null,
      maskMode: attachment.maskMode || 'grayscale',
      maskInvert: !!attachment.maskInvert,
      maskFeather: attachment.maskFeather || 0, maskCrop: !!attachment.maskCrop,
    }).then(function (selection) {
      if (!selection || state.conversation !== conversation || state.attachments.indexOf(attachment) < 0) return;
      if (selection.remove) {
        clearImageSelection(attachment); return;
      }
      var form = new FormData(); form.append('file', selection.blob, 'selection.png');
      return fetch('/api/upload', { method: 'POST', body: form }).then(function (response) {
        return response.json().then(function (data) {
          var uploaded = data && data.files ? data.files[0] : data;
          if (!response.ok || !uploaded || !uploaded.ok || !uploaded.file)
            throw new Error((data && data.error) || t('page.imageSelection.uploadFailed'));
          if (state.conversation !== conversation || state.attachments.indexOf(attachment) < 0) return;
          attachment.maskPath = uploaded.file; attachment.maskMode = 'grayscale';
          delete attachment.maskInvert;
          attachment.maskFeather = selection.maskFeather; attachment.maskCrop = selection.maskCrop;
          state.attachments.forEach(function (a) {
            if (a.mediaType === 'image') a._maskActive = a === attachment;
          });
          // Commit the source change only after the new mask has uploaded. Cancel
          // and failure leave the previous source, draft and selections untouched.
          promoteImageSource(attachment);
        });
      });
    }).catch(function (error) { notice(t('page.imageSelection.failed', { error: (error && error.message) || error }), 'error'); })
      .finally(function () { state.maskEditing = false; paintChips(); });
  }

  function paintChips() {
    var box = $('chips');
    box.innerHTML = '';
    appliedShareOrder.forEach(function (id) {
      if (!appliedShareIds[id]) return;
      var parts = appliedShareParts[id] || {};
      var shared = el('div', 'chip shared');
      shared.appendChild(el('span', 'ic', '↗'));
      shared.appendChild(el('span', 'nm', parts.title || t('page.chips.shared')));
      var remove = el('button', 'x', '✕');
      remove.setAttribute('aria-label', t('page.chips.removeShared'));
      remove.disabled = state.generating || state.visionChecking || shareDiscarding;
      remove.addEventListener('click', function () { discardAppliedShare(id, remove); });
      shared.appendChild(remove);
      box.appendChild(shared);
    });
    var imageIndex = 0;
    state.attachments.forEach(function (a, i) {
      var c = el('div', 'chip'), name = shownName(a) || a.file;
      if (a.mediaType === 'image') {
        var img = document.createElement('img'); img.src = previewOf(a); c.appendChild(img);
        // Preparing a selection needs no model. Keep the control discoverable while
        // Qwen is loading or another model is selected; Send checks compatibility.
        var active = imageIndex++ === 0;
        c.classList.add('editable-image');
        var select = el('button', 'filechip mask-select', a.maskPath ? t('page.chips.selectionSaved') : t('page.chips.selectArea'));
        select.type = 'button'; select.disabled = state.generating || state.visionChecking || state.maskEditing;
        select.setAttribute('aria-label', a.maskPath
          ? t('page.chips.adjustSelectionFor', { name: name })
          : t('page.chips.selectAreaIn', { name: name }));
        select.addEventListener('click', function () { selectImageArea(a); }); c.appendChild(select);
        var details = el('span', 'image-details');
        details.appendChild(el('span', active ? 'image-edit-source' : 'image-reference',
          active ? t('page.chips.editingTarget') : t('page.chips.reference')));
        details.appendChild(el('span', 'nm', name));
        c.appendChild(details);
      } else {
        c.appendChild(el('span', 'ic', a.mediaType === 'video' ? '🎬' : a.mediaType === 'audio' ? '🎧' : '📄'));
      }
      if (a.mediaType !== 'image') c.appendChild(el('span', 'nm', name));
      var x = el('button', 'x', '✕');
      x.type = 'button'; x.setAttribute('aria-label', t('page.chips.remove', { name: name }));
      x.disabled = state.visionChecking || shareDiscarding || state.maskEditing;
      x.addEventListener('click', function () {
        if (a === imageSource(state.attachments)) {
          // Saved reference selections stay available in Adjust, but removing the
          // target must not silently turn one of them into the next requested edit.
          deactivateImageSelections();
        }
        state.attachments.splice(i, 1); paintChips();
      });
      c.appendChild(x);
      box.appendChild(c);
    });
    var source = imageSource(state.attachments);
    if (!makesImages() && hasActiveImageSelection(source)) {
      var hint = el('div', 'image-selection-hint');
      hint.appendChild(el('span', null, t('page.chips.selectionNeedsQwen')));
      var models = el('button', 'notice-action', t('page.action.openModels')); models.type = 'button';
      models.addEventListener('click', function () { openRoute('models'); });
      hint.appendChild(models); box.appendChild(hint);
    }
  }

  var uploadQueue = Promise.resolve();
  var pendingUploadCount = 0;

  function upload(files) {
    if (!files.length) return Promise.resolve();
    var fd = new FormData();
    files.forEach(function (file) { fd.append('file', file, file.name); });
    return fetch('/api/upload', { method: 'POST', body: fd })
      .then(function (r) {
        return r.json().then(function (data) {
          if (!r.ok || !data || !data.ok) throw new Error((data && data.error) || t('page.upload.failed'));
          return data;
        });
      })
      .then(function (data) {
        var uploaded = Array.isArray(data.files) ? data.files : [data];
        if (uploaded.length !== files.length || uploaded.some(function (a) { return !a || !a.ok || !a.file; })) {
          throw new Error(t('page.upload.incomplete'));
        }
        // The server returns multipart order, including mixed media and documents.
        Array.prototype.push.apply(state.attachments, uploaded);
        paintChips();
        uploaded.forEach(function (a) { if (a.warning) notice(a.warning); });
      })
      .catch(function (e) { notice(t('page.upload.error', { error: (e && e.message) || e }), 'error'); });
  }

  $('file-input').addEventListener('change', function (e) {
    var files = Array.prototype.slice.call(e.target.files || []);
    e.target.value = '';
    if (!files.length) return;
    pendingUploadCount++;
    uploadQueue = uploadQueue.then(function () { return upload(files); })
      .finally(function () { pendingUploadCount--; });
    return uploadQueue;
  });

  // ---- sheets --------------------------------------------------------------
  function openSheet(id) { $('sheet-bg').classList.add('on'); $(id).classList.add('on'); }
  function closeSheets() {
    $('sheet-bg').classList.remove('on');
    ['attach-sheet', 'skills-sheet', 'model-sheet', 'nav-sheet', 'skill-sheet', 'skill-add-sheet', 'lora-sheet'].forEach(function (s) { $(s).classList.remove('on'); });
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
      box.appendChild(el('div', 'navempty', t('page.nav.empty')));
      return;
    }
    state.conversations.forEach(function (c) {
      var row = el('button', 'navchat' + (c.id === state.conversation ? ' on' : ''));
      row.type = 'button';
      row.appendChild(el('span', 'nm', c.title || t('page.nav.untitled')));
      row.appendChild(el('span', 'ds', when(c.updatedAt) + ' · ' + tn('page.nav.messages', c.messageCount || 0)));
      row.addEventListener('click', function () {
        closeSheets();
        if (c.id === state.conversation) return;
        openConversation(c.id).then(function () { paintNavChats(); });
      });
      box.appendChild(row);
    });
  }

  // Dates in the interface's language. English keeps the WebView's own locale, which
  // carries the region too (a 24-hour clock, the day before the month).
  var DATES = LANG === 'en' ? [] : LANG;
  /** A date a person reads at a glance: a time today, a day this week, a date before that. */
  function when(iso) {
    var d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    var now = new Date();
    var sameDay = d.toDateString() === now.toDateString();
    if (sameDay) return d.toLocaleTimeString(DATES, { hour: 'numeric', minute: '2-digit' });
    if (now - d < 6 * 24 * 3600 * 1000) return d.toLocaleDateString(DATES, { weekday: 'short' });
    return d.toLocaleDateString(DATES, { month: 'short', day: 'numeric' });
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
            notice(t('page.nav.skillsFailed', { error: (e && e.message) || e }), 'error');
          });
        }
        return;
      }
      openRoute(b.getAttribute('data-route'));
    });
  });

  // One sentence per screen, so each can be named in the interface's language rather
  // than by its route; a route without one is shown as it is.
  var ROUTE_FAILED = {
    sessions: 'page.route.openFailed.sessions',
    models: 'page.route.openFailed.models',
    settings: 'page.route.openFailed.settings',
    about: 'page.route.openFailed.about',
  };
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
        var error = (e && e.message) || e;
        notice(Object.prototype.hasOwnProperty.call(ROUTE_FAILED, route)
          ? t(ROUTE_FAILED[route], { error: error })
          : t('page.route.openFailed.unlisted', { route: route, error: error }), 'error');
      });
  }

  modelBtn.addEventListener('click', function () {
    var info = $('model-info');
    info.innerHTML = '';
    info.appendChild(el('div', 'skillrow',
      state.model ? (pretty(state.model) + ' · ' + (state.arch || '?') + ' · ' + (state.backend || '')) : t('page.modelSheet.none')));
    var loraBtn = $('open-loras');
    loraBtn.style.display = makesImages() ? '' : 'none';
    if (makesImages()) {
      // Only the names, for the hint: the sheet keeps its own list, which a late answer
      // here must not replace.
      fetch('/api/agent/loras').then(function (r) { return r.json(); }).then(function (d) {
        var on = ((d && d.loras) || []).filter(function (l) { return l.chosen; }).map(function (l) { return l.name; });
        $('loras-hint').textContent = on.length ? on.join(' + ') : t('page.modelSheet.lorasHint');
      }).catch(function () { /* the button still opens the sheet, which says what failed */ });
    }
    openSheet('model-sheet');
  });

  // ---- LoRA plug-ins (an image model) --------------------------------------
  //
  // The plug-ins the host offers for the image model (WebUiRoutes.MapLoras): what is
  // downloaded, what is on, and how strongly. A change is saved, and the host applies
  // it to the next picture, so a picture being drawn keeps the plug-ins it started with.
  var loraPoll = null;
  // The choice last sent, until the host answers it: a second change made before then
  // builds on it rather than on the list painted before the first. Saves go one at a
  // time, in the order they were made, because each one replaces the whole choice; a
  // list read before the latest save was sent is not painted over its answer.
  var loraChoice = null;
  var loraQueue = Promise.resolve();
  var loraSeq = 0;
  var loraShape = '';
  var loraPct = {};
  function loraSize(bytes) {
    var mb = (Number(bytes) || 0) / 1e6;
    return mb >= 1000 ? t('page.lora.gigabytes', { size: oneDecimal(mb / 1000) })
      : t('page.lora.megabytes', { size: Math.round(mb) });
  }
  /** One decimal place: toFixed's digits, with the interface language's decimal mark. */
  function oneDecimal(n) {
    var s = n.toFixed(1), mark = '.';
    try { mark = (1.5).toLocaleString(LANG).charAt(1); } catch (e) {}
    return mark === '.' ? s : s.replace('.', mark);
  }
  function chosenLoras(d) {
    return ((d && d.chosen) || []).map(function (c) { return { id: c.id, strength: c.strength }; });
  }
  function currentLoras() {
    return loraChoice ? loraChoice.slice() : chosenLoras(state.loras);
  }
  // The host checks every change (one speed plug-in, downloaded, a known strength) and
  // says why it refused, which post() would reduce to a status code.
  function loraJson(r) {
    return r.json().then(function (d) {
      if (!r.ok) throw new Error((d && d.error) || ('HTTP ' + r.status));
      return d;
    });
  }
  // Said in the sheet, beside the switch that flipped back: a notice would land in the
  // chat underneath it, where nobody looks while the sheet is open.
  function loraError(message) {
    var line = $('lora-error');
    var inSheet = !!message && $('lora-sheet').classList.contains('on');
    line.textContent = inSheet ? message : '';
    line.style.display = inSheet ? '' : 'none';
    // The sheet scrolls, and the row the user just changed may be far below the line.
    if (inSheet && typeof line.scrollIntoView === 'function') line.scrollIntoView({ block: 'nearest' });
    if (message && !inSheet) notice(message, 'error');
  }
  function loadLoras(tick) {
    var seq = loraSeq;
    return fetch('/api/agent/loras').then(loraJson).then(function (d) {
      if (seq !== loraSeq || loraChoice) return state.loras;
      return paintLoras(d, tick);
    });
  }
  // One change at a time, after those before it. `send` returns the host's answer: the
  // sheet as it now is. Every answer is kept as the latest the host said, and only the
  // latest change's is painted. `failed` words a refusal, or is null to show it as is.
  function queueLoras(send, failed, isSave) {
    var seq = ++loraSeq;
    loraQueue = loraQueue.then(function () {
      return send().then(function (d) {
        state.loras = d;
        if (seq !== loraSeq) return;
        loraChoice = null;
        paintLoras(d);
      }, function (e) {
        var reason = (e && e.message) || e;
        var why = failed ? failed(reason) : String(reason);
        if (seq !== loraSeq) {
          // A later save resends the whole choice and answers for it, so an earlier
          // save's failure says nothing the user can act on; a removal's does.
          if (!isSave) loraError(why);
          return;
        }
        loraError(why);
        loraChoice = null;
        // The switches back where the host last had them, without waiting on another
        // request, and then a fresh list.
        if (state.loras) paintLoras(state.loras);
        return loadLoras();
      }).catch(function () { /* the list stays as it was painted */ });
    });
    return loraQueue;
  }
  function saveLoras(list) {
    loraChoice = list;
    return queueLoras(function () {
      return fetch('/api/agent/loras/choice', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ loras: list }),
      }).then(loraJson);
    }, null, true);
  }
  function toggleLora(l, on) {
    loraError('');
    var d = state.loras || {};
    var list = currentLoras().filter(function (c) { return c.id !== l.id; });
    if (on) {
      // One speed plug-in at a time: two step schedules cannot both apply.
      if (l.kind === 'Speed') {
        var speeds = ((d.loras) || []).filter(function (x) { return x.kind === 'Speed'; }).map(function (x) { return x.id; });
        list = list.filter(function (c) { return speeds.indexOf(c.id) < 0; });
      }
      list.push({ id: l.id, strength: l.defaultStrength });
    }
    return saveLoras(list);
  }
  function setLoraStrength(l, value) {
    loraError('');
    return saveLoras(currentLoras().map(function (c) {
      return c.id === l.id ? { id: c.id, strength: value } : c;
    }));
  }
  function downloadLora(l) {
    loraError('');
    return fetch('/api/agent/loras/' + encodeURIComponent(l.id) + '/download', { method: 'POST' })
      .then(function (r) {
        // The download belongs to the app, not to this request: the stream is only a
        // window on it, so the page closes it and follows the job through the list.
        if (r.body && r.body.cancel) r.body.cancel().catch(function () {});
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return loadLoras();
      })
      .catch(function (e) { loraError(t('page.lora.downloadFailed', { error: (e && e.message) || e })); });
  }
  // Removing a plug-in also turns it off, so it is a change like any other: one made
  // before the host answers builds on the choice without it, or it would send it back.
  function removeLora(l) {
    loraError('');
    loraChoice = currentLoras().filter(function (c) { return c.id !== l.id; });
    return queueLoras(function () {
      return fetch('/api/agent/loras/' + encodeURIComponent(l.id), { method: 'DELETE' })
        .then(loraJson)
        .then(function (d) { notice(t('page.lora.removed', { name: l.name })); return d; });
    }, function (reason) { return t('page.lora.removeFailed', { name: l.name, error: reason }); }, false);
  }
  function loraSwitch(l) {
    var sw = el('label', 'switch');
    var box = el('input');
    box.type = 'checkbox';
    box.checked = !!l.chosen;
    box.addEventListener('change', function () { toggleLora(l, box.checked); });
    sw.appendChild(box);
    sw.appendChild(el('span', 'track'));
    return sw;
  }
  function loraPercent(l) {
    var dl = l.download;
    return t('page.lora.percent', { percent: Math.round(((dl && dl.progress && dl.progress.fraction) || 0) * 100) });
  }
  // What a row shows apart from a running download's percentage.
  function loraShapeOf(d) {
    return JSON.stringify(((d && d.loras) || []).map(function (l) {
      var dl = l.download || {};
      return [l.id, l.state, !!l.chosen, l.strength, (l.installedBytes || 0) > 0, !!dl.running, dl.state || '', dl.error || ''];
    }));
  }
  function paintLoras(d, tick) {
    state.loras = d;
    var list = $('lora-list');
    var loras = (d && d.loras) || [];
    var running = loras.some(function (l) { return l.download && l.download.running; });
    var shape = loraShapeOf(d);
    // A poll while a download runs moves only its percentage. The rows stay, so a
    // strength being dragged or a button being pressed is not replaced under the finger.
    if (tick && shape === loraShape && list.children.length) {
      loras.forEach(function (l) { if (loraPct[l.id]) loraPct[l.id].textContent = loraPercent(l); });
      return d;
    }
    loraShape = shape;
    loraPct = {};
    list.innerHTML = '';
    if (!loras.length) list.appendChild(el('div', 'notice', t('page.lora.none')));
    loras.forEach(function (l) {
      var row = el('div', 'skillrow lorarow');
      row.setAttribute('data-lora', l.id);
      var meta = el('div', 'meta');
      meta.appendChild(el('div', 'nm', (l.chosen ? '● ' : '') + l.name + (l.kind === 'Speed' && l.steps ? ' · ' + tn('page.lora.steps', l.steps) : '')));
      meta.appendChild(el('div', 'ds', l.trigger
        ? t('page.lora.purposeWithTrigger', { purpose: l.purpose, trigger: l.trigger })
        : l.purpose));
      var kind = l.kind === 'Speed' ? t('page.lora.kind.speed')
        : l.kind === 'Edit' ? t('page.lora.kind.edit') : t('page.lora.kind.style');
      // Its task works only on the model's own steps, so a speed plug-in sits out its edits.
      if (l.needsModelSteps) kind += ' · ' + t('page.lora.keepsModelSteps');
      meta.appendChild(el('div', 'lic', kind + ' · ' + loraSize(l.totalBytes) + ' · ' + l.license));
      row.appendChild(meta);
      var act = el('div', 'act');
      var dl = l.download;
      if (dl && dl.running) {
        var pct = el('span', 'pct', loraPercent(l));
        loraPct[l.id] = pct;
        act.appendChild(pct);
        var stop = el('button', 'mini', t('page.lora.stop'));
        stop.addEventListener('click', function () {
          post('/api/agent/loras/' + encodeURIComponent(l.id) + '/download/cancel', {}).then(function () { return loadLoras(); })
            .catch(function (e) { loraError(t('page.lora.stopFailed', { error: (e && e.message) || e })); });
        });
        act.appendChild(stop);
      } else if (l.state !== 'Installed') {
        // Turned on, but its files are gone: the next picture is refused until it is
        // downloaded again or turned off, so both are offered here.
        if (l.chosen) act.appendChild(loraSwitch(l));
        var get = el('button', 'mini', t('page.lora.download'));
        get.addEventListener('click', function () { downloadLora(l); });
        act.appendChild(get);
        // A stopped download's part files are the user's to reclaim without finishing it.
        if ((l.installedBytes || 0) > 0) {
          var drop = el('button', 'mini quiet', t('page.lora.remove'));
          drop.addEventListener('click', function () { removeLora(l); });
          act.appendChild(drop);
        }
        if (l.chosen) meta.appendChild(el('div', 'err', t('page.lora.filesMissing')));
        if (dl && dl.state === 'Failed') {
          meta.appendChild(el('div', 'err', dl.error
            ? t('page.lora.downloadStopped', { error: dl.error })
            : t('page.lora.downloadStoppedUnknown')));
        }
      } else {
        act.appendChild(loraSwitch(l));
        var rm = el('button', 'mini quiet', t('page.lora.remove'));
        rm.addEventListener('click', function () { removeLora(l); });
        act.appendChild(rm);
        if (l.chosen && l.strengthAdjustable) {
          var strength = el('div', 'strength');
          var range = el('input');
          range.type = 'range';
          range.min = String(d.minStrength);
          range.max = String(d.maxStrength);
          range.step = '0.05';
          range.value = String(l.strength);
          var shown = el('span', 'val', t('page.lora.percent', { percent: Math.round(l.strength * 100) }));
          range.addEventListener('input', function () {
            shown.textContent = t('page.lora.percent', { percent: Math.round(Number(range.value) * 100) });
          });
          range.addEventListener('change', function () { setLoraStrength(l, Number(range.value)); });
          strength.appendChild(el('span', 'lbl', t('page.lora.strength')));
          strength.appendChild(range);
          strength.appendChild(shown);
          meta.appendChild(strength);
        }
      }
      row.appendChild(act);
      list.appendChild(row);
    });
    if (running && !loraPoll) {
      loraPoll = setInterval(function () {
        if (!$('lora-sheet').classList.contains('on')) { clearInterval(loraPoll); loraPoll = null; return; }
        loadLoras(true).catch(function () {});
      }, 1000);
    }
    if (!running && loraPoll) { clearInterval(loraPoll); loraPoll = null; }
    return d;
  }
  $('open-loras').addEventListener('click', function () {
    closeSheets();
    loraError('');
    loadLoras().then(function () { openSheet('lora-sheet'); })
      .catch(function (e) { notice(t('page.lora.listFailed', { error: (e && e.message) || e }), 'error'); });
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
    if (label) label.textContent = on ? t('page.skills.on') : t('page.skills.off');
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
      .catch(function (e) { notice(t('page.skills.saveFailed', { error: e }), 'error'); });
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
      if (!state.catalogSkills.length) list.appendChild(el('div', 'notice', t('page.skills.none')));
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
    if (s.scripts) bits.push(tn('page.skill.scripts', s.scripts));
    // The host reports where a skill came from as a word of its protocol.
    if (s.origin) {
      bits.push(s.origin === 'installed' ? t('page.skill.origin.installed')
        : s.origin === 'discovered' ? t('page.skill.origin.discovered') : String(s.origin));
    }
    if (bits.length) body.appendChild(el('div', 'meta', bits.join(' · ')));
    body.appendChild(document.createTextNode(s.description || t('page.skill.noDescription')));

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
          notice(t('page.skill.removed', { name: s.name }));
        })
        .catch(function (e) { notice(t('page.skill.removeFailed', { error: e }), 'error'); });
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
      .catch(function (err) { notice(t('page.skillAdd.zipFailed', { error: err }), 'error'); });
  });
  $('skill-fetch').addEventListener('click', function () {
    var url = ($('skill-url').value || '').trim();
    if (!url) return;
    $('skill-fetch').disabled = true;
    post('/api/skills/from-url', { url: url })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) { $('skill-fetch').disabled = false; $('skill-url').value = ''; afterInstall(res); })
      .catch(function (err) { $('skill-fetch').disabled = false; notice(t('page.skillAdd.linkFailed', { error: err }), 'error'); });
  });
  function afterInstall(res) {
    if (!res.ok) {
      var msg = (res.body && (res.body.error || res.body.message)) || t('page.skillAdd.refused');
      notice(typeof msg === 'string' ? msg : JSON.stringify(msg), 'error');
      return;
    }
    closeSheets();
    // A list install reports both halves; say how many landed and how many did not.
    if (res.body && typeof res.body.count === 'number') {
      var failed = (res.body.failed || []).length;
      notice(failed
        ? tn('page.skillAdd.installedSomeFailed', res.body.count, { failed: failed })
        : tn('page.skillAdd.installed', res.body.count));
    } else {
      notice(res.body && res.body.name
        ? t('page.skillAdd.installedNamed', { name: res.body.name })
        : t('page.skillAdd.installedUnnamed'));
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
  text.addEventListener('keydown', function (e) {
    // Let Shift+Enter insert a newline and IME Enter finish composing text.
    if (e.key !== 'Enter' || e.shiftKey || e.isComposing || e.keyCode === 229) return;
    e.preventDefault();
    // The Send button becomes Stop during inference; Enter should never stop it.
    if (!e.repeat && !state.generating) sendMessage();
  });
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
    if (state.voice || (state.native && !state.dictation)) return;
    cancelPress();
    pressAt = { x: x, y: y };
    pressTimer = setTimeout(function () {
      pressTimer = 0;
      pressAt = null;
      if (!state.native) {
        // In a browser there is no recogniser to switch to, and a composer that
        // turned into a dead button would be worse than not switching.
        notice(t('page.voice.appOnly'), 'error');
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
    if (!state.native) { notice(t('page.voice.appOnly'), 'error'); return; }
    if (!state.dictation) return;
    hold.classList.add('rec');
    $('holdlabel').textContent = t('page.voice.listening');
    post('/api/agent/events', { type: 'dictate-start' });
  }
  function stopRec() {
    if (!hold.classList.contains('rec')) return;
    $('holdlabel').textContent = t('page.voice.transcribing');
    post('/api/agent/events', { type: 'dictate-stop' });
  }
  // The app says when the session has really ended, because the transcription
  // arrives after the finger lifts and the button must not look idle before it does.
  function dictationEnded() {
    hold.classList.remove('rec');
    $('holdlabel').textContent = t('page.voice.hold');
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
  // to the button they are about to hold, not three screens away in Settings. Each
  // language is named in itself, so only Auto is translated.
  var LANGS = [
    { id: '', label: t('page.voice.langAuto') },
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
    nativeReady: function (a) { window.TensorAgent.nativeReady(a); },
    addAttachment: function (a) { window.TensorAgent.addAttachment(a); },
    insertText: function (a) { window.TensorAgent.insertText(a && a.text); },
    takeShare: function () { window.TensorAgent.takeShare(); },
    notice: function (a) { notice(a && a.text, (a && a.kind) || 'error'); },
    noticeWithSettings: function (a) { window.TensorAgent.noticeWithSettings(a && a.text); },
    openConversation: function (a) { window.TensorAgent.openConversation(a && a.id); },
  };

  // A language change reloads the translated page. Keep the unsent composer in
  // this WebView's session storage for that one reload, including saved selections
  // and share ownership, so switching languages cannot discard or duplicate a draft.
  var languageDraftKey = 'tensoragent-language-draft';
  function prepareLanguageReload() {
    // Wait for mutations and send acceptance to settle. In particular a sent share
    // is still owned by the composer until the chat response's headers arrive.
    if ((state.abort && state.abort.awaitingHeaders) || state.visionChecking
        || shareDiscarding || shareDrain || conversationOpening || pendingUploadCount || state.maskEditing)
      return 'busy';
    try {
      window.sessionStorage.setItem(languageDraftKey, JSON.stringify({
        conversation: state.conversation,
        text: text.value,
        attachments: state.attachments,
        skills: state.skills,
        skillsExplicit: state.skillSelectionExplicit,
        think: state.think,
        shareOrder: appliedShareOrder,
        shareParts: appliedShareParts,
      }));
      return true;
    } catch (e) {
      note('language-draft-save-failed', (e && e.message) || e);
      return false;
    }
  }

  function restoreLanguageDraft() {
    try {
      var saved = window.sessionStorage.getItem(languageDraftKey);
      if (!saved) return;
      var draft = JSON.parse(saved);
      window.sessionStorage.removeItem(languageDraftKey);
      // A cold launch or a deleted chat must not inherit another conversation's draft.
      if (!draft || draft.conversation !== state.conversation) return;
      text.value = typeof draft.text === 'string' ? draft.text : '';
      state.attachments = Array.isArray(draft.attachments) ? draft.attachments : [];
      state.skills = skillsOn() && Array.isArray(draft.skills) ? draft.skills : [];
      state.skillSelectionExplicit = draft.skillsExplicit === true;
      state.think = draft.think === true;
      (Array.isArray(draft.shareOrder) ? draft.shareOrder : []).forEach(function (id) {
        rememberAppliedShare(id, draft.shareParts && draft.shareParts[id]);
      });
      autoGrow();
      paintChips();
      paintSkillChips();
    } catch (e) {
      note('language-draft-restore-failed', (e && e.message) || e);
    }
  }

  window.TensorAgent = {
    notice: notice,
    prepareLanguageReload: prepareLanguageReload,

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
      if (!a || !a.ok) { notice((a && a.error) || t('page.upload.failed'), 'error'); return; }
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
        native: state.native,
        dictation: state.dictation,
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
    refreshModel: function () { refreshEngine().then(refreshModel); return true; },
    hasModel: function () { return !!state.model; },
    dictationEnded: dictationEnded,
    /** A refusal only the user can lift, with a button that opens iOS Settings. */
    noticeWithSettings: function (msg) {
      noticeWithAction(msg, t('page.action.openSettings'), function () {
        post('/api/agent/events', { type: 'open-settings' });
        return true;
      });
    },
    /** Native pickers and dictation are separate capabilities on Windows. */
    nativeReady: function (capabilities) {
      state.native = true;
      state.dictation = !capabilities || capabilities.dictation !== false;
      state.composerHint = (capabilities && capabilities.composerHint) || t('page.composer.message');
      paintComposerHint();
      if (!state.dictation) {
        cancelPress();
        if (state.voice) setVoice(false);
      }
      return true;
    },
    canDictate: function () { return state.native && state.dictation; },
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
      refreshEngine().then(refreshModel);
      takePendingShare();
    }, 250);
  });
  window.addEventListener('pageshow', function () { resumeTurn(); takePendingShare(); });

  // ---- start ---------------------------------------------------------------
  refreshEngine()
    .then(function () { return applySettings(true); })
    .then(refreshModel)
    .then(openAtLaunch)
    .then(restoreLanguageDraft)
    .then(function () {
      return post('/api/agent/events', { type: 'ready', conversation: state.conversation });
    })
    .then(function () { shareIntakeReady = true; return takePendingShare(); })
    .catch(function (e) {
      note('start-failed', (e && e.message) || e);
      notice(t('page.start.failed', { error: (e && e.message) || e }), 'error');
      // Whatever failed, the page is here and a share waiting on the host must still
      // reach it: leaving this false meant one lost request at startup disabled sharing
      // for the life of the page. The host is told again too -- it is what makes the
      // composer's native pickers work.
      shareIntakeReady = true;
      post('/api/agent/events', { type: 'ready', conversation: state.conversation }).catch(function () {});
      takePendingShare();
    });
})();
