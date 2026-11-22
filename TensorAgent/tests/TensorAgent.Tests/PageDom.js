// A DOM the size of this page's needs, so tensoragent.js can be RUN by a test.
//
// Everything else about the page is checked by reading its source for a string,
// which cannot tell whether a photo comes back when a saved chat is reopened --
// the bug this exists for. What is stubbed is only what the script touches:
// elements with children, classes, text and handlers; a fetch that answers from
// a table the test fills in and records every request; and the two timers.
(function () {
  'use strict';

  function Element(tag) {
    this.tagName = String(tag || 'div').toUpperCase();
    this.children = [];
    this.attrs = {};
    this.handlers = {};
    this.parentNode = null;
    this._class = '';
    this._text = '';
    this._html = '';
    this.style = { setProperty: function () {}, height: '', display: '' };
    this.classList = {
      _el: this,
      add: function (c) { this._el.className = (this._el.className + ' ' + c).trim(); },
      remove: function (c) {
        this._el.className = this._el.className.split(/\s+/).filter(function (x) { return x && x !== c; }).join(' ');
      },
      toggle: function (c, on) { on ? this.add(c) : this.remove(c); },
      contains: function (c) { return this._el.className.split(/\s+/).indexOf(c) >= 0; },
    };
  }

  Object.defineProperty(Element.prototype, 'className', {
    get: function () { return this._class; },
    set: function (v) { this._class = v == null ? '' : String(v); },
  });
  Object.defineProperty(Element.prototype, 'textContent', {
    get: function () {
      if (this.children.length === 0) return this._text;
      return this.children.map(function (c) { return c.textContent; }).join('');
    },
    set: function (v) { this._text = v == null ? '' : String(v); detachAll(this); },
  });
  // innerHTML is kept as the raw markup the page produced. The page renders its
  // own Markdown into it, so parsing would only re-implement a browser; what a
  // test needs is what was written, and that is what is stored.
  // A test that needs to see what the page writes (how much, in what order) sets
  // __page.onHtml(element, markup); it is called after every write.
  Object.defineProperty(Element.prototype, 'innerHTML', {
    get: function () { return this._html; },
    set: function (v) {
      this._html = v == null ? '' : String(v); detachAll(this); this._text = '';
      if (globalThis.__page && typeof globalThis.__page.onHtml === 'function') globalThis.__page.onHtml(this, this._html);
    },
  });

  // As a browser does: an element whose markup is replaced no longer has a parent,
  // so a page that keeps a reference to it can tell that it must put it back.
  function detachAll(parent) {
    parent.children.forEach(function (c) { c.parentNode = null; });
    parent.children = [];
  }

  Element.prototype.appendChild = function (child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  };
  Element.prototype.insertBefore = function (child, before) {
    var at = this.children.indexOf(before);
    child.parentNode = this;
    if (at < 0) this.children.push(child); else this.children.splice(at, 0, child);
    return child;
  };
  Element.prototype.remove = function () {
    if (!this.parentNode) return;
    var at = this.parentNode.children.indexOf(this);
    if (at >= 0) this.parentNode.children.splice(at, 1);
    this.parentNode = null;
  };
  Element.prototype.removeChild = function (child) {
    var at = this.children.indexOf(child);
    if (at < 0) throw new Error('Node is not a child');
    this.children.splice(at, 1);
    child.parentNode = null;
    return child;
  };
  Element.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
  Element.prototype.getAttribute = function (k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; };
  Element.prototype.removeAttribute = function (k) { delete this.attrs[k]; };
  Element.prototype.addEventListener = function (name, fn) { (this.handlers[name] = this.handlers[name] || []).push(fn); };
  Element.prototype.removeEventListener = function () {};
  Element.prototype.focus = function () {};
  Element.prototype.dispatch = function (name, event) {
    (this.handlers[name] || []).forEach(function (fn) { fn(event || { preventDefault: function () {} }); });
    var direct = this['on' + name];
    if (typeof direct === 'function') direct(event || {});
  };

  function matches(el, selector) {
    if (selector.charAt(0) === '.') return el.classList.contains(selector.slice(1));
    // [name]: the attribute selectors /i18n.js translates the markup with.
    if (selector.charAt(0) === '[') return el.getAttribute(selector.slice(1, -1)) !== null;
    return el.tagName === selector.toUpperCase();
  }
  Element.prototype.querySelectorAll = function (selector) {
    var last = selector.split(/\s+/).pop(), found = [];
    (function walk(node) {
      node.children.forEach(function (c) { if (matches(c, last)) found.push(c); walk(c); });
    })(this);
    return found;
  };
  Element.prototype.querySelector = function (selector) {
    var all = this.querySelectorAll(selector);
    return all.length ? all[0] : null;
  };

  var byId = {};
  function element(tag) { return new Element(tag); }

  var document = {
    body: element('body'),
    documentElement: element('html'),
    visibilityState: 'visible',
    createElement: element,
    createTextNode: function (t) { var n = element('span'); n.textContent = t; return n; },
    getElementById: function (id) {
      if (!byId[id]) { byId[id] = element('div'); byId[id].attrs.id = id; }
      return byId[id];
    },
    querySelectorAll: function (selector) {
      // Only the two sheet-scoped lookups the page makes, and both are lists of
      // buttons a test never presses.
      var scoped = selector.split(/\s+/);
      if (scoped.length > 1 && scoped[0].charAt(0) === '#') {
        return document.getElementById(scoped[0].slice(1)).querySelectorAll(scoped.slice(1).join(' '));
      }
      return document.body.querySelectorAll(selector);
    },
    addEventListener: function (name, fn) { (document._handlers[name] = document._handlers[name] || []).push(fn); },
    removeEventListener: function (name, fn) {
      var list = document._handlers[name] || [];
      var at = list.indexOf(fn);
      if (at >= 0) list.splice(at, 1);
    },
    /**
     * Fire a document-level event, which is how the page catches a link click and
     * how a test brings the app back to the foreground: set
     * `document.visibilityState` (a plain, writable property) and then dispatch
     * 'visibilitychange'.
     */
    dispatch: function (name, event) {
      (document._handlers[name] || []).slice().forEach(function (fn) { fn(event === undefined ? {} : event); });
    },
    _handlers: {},
  };
  document.documentElement.style = { setProperty: function () {} };

  // The one element with structure the page reads back out of it.
  var activity = document.getElementById('activity');
  var label = element('div'); label.className = 'label'; activity.appendChild(label);
  var tail = element('div'); tail.className = 'tail'; activity.appendChild(tail);
  // Every label the strip has shown, oldest first. The page clears the strip when a
  // turn ends, so by the time a test can look, this record is all that is left of it.
  var labels = [];
  Object.defineProperty(label, 'textContent', {
    get: function () { return this._text; },
    set: function (v) { this._text = v == null ? '' : String(v); if (this._text) labels.push(this._text); },
  });

  // ---- the network ---------------------------------------------------------
  // Answers come from a table the test fills in; every request is recorded, so a
  // test can assert on the body the page SENT as well as on what it drew.
  //
  // A route is a value, or a function of the recorded call that returns one:
  //   a document                              200 with that JSON body
  //   { __status: n, body: ..., headers }     any HTTP answer, a refusal included
  //   { __sse: [...frames], __then, __delay } an event stream, frame by frame
  //   { __reject: 'Load failed' | true }      NO answer: fetch rejects with the
  //                                           TypeError WebKit raises when the
  //                                           connection fails ('Load failed')
  // A function that throws rejects the fetch the same way a browser would, rather
  // than throwing out of fetch() itself. Nothing here ever throws synchronously. A
  // function may also return a promise of any of the above: the request is answered
  // when the test settles it.
  var routes = {};
  var calls = [];

  // Keep multipart entries ordered so upload tests inspect the actual submitted
  // names and file objects, without pretending a multipart body is JSON.
  function FormData() { this.parts = []; }
  FormData.prototype.append = function (name, value, fileName) {
    this.parts.push({ name: name, value: value, fileName: fileName });
  };

  /** WebKit's failure to reach the server at all: a TypeError, not a response. */
  function networkError(message) {
    return new TypeError(message == null || message === true ? 'Load failed' : String(message));
  }
  /** What a fetch or a read rejects with once its signal has fired. */
  function abortError() {
    var e = new Error('The operation was aborted.');
    e.name = 'AbortError';
    e.code = 20;
    return e;
  }

  // ---- AbortController -----------------------------------------------------
  // Real enough to stop a request: abort() marks the signal, tells its listeners,
  // and every read still waiting on a body that was fetched with that signal
  // rejects with AbortError -- which is the only way a page gets out of a stream
  // whose server has gone quiet.
  function AbortSignal() {
    this.aborted = false;
    this.reason = undefined;
    this.onabort = null;
    this._listeners = {};
  }
  AbortSignal.prototype.addEventListener = function (name, fn) {
    (this._listeners[name] = this._listeners[name] || []).push(fn);
  };
  AbortSignal.prototype.removeEventListener = function (name, fn) {
    var list = this._listeners[name] || [];
    var at = list.indexOf(fn);
    if (at >= 0) list.splice(at, 1);
  };
  AbortSignal.prototype.throwIfAborted = function () { if (this.aborted) throw this.reason; };
  function AbortController() { this.signal = new AbortSignal(); }
  AbortController.prototype.abort = function (reason) {
    var s = this.signal;
    if (s.aborted) return;
    s.aborted = true;
    s.reason = reason === undefined ? abortError() : reason;
    var event = { type: 'abort', target: s };
    (s._listeners.abort || []).slice().forEach(function (fn) { fn(event); });
    if (typeof s.onabort === 'function') s.onabort(event);
  };

  /**
   * An event stream, frame by frame, the way the page reads one.
   *
   * After the frames run out, `__then` says what the connection does:
   *   'end'    (default) the next read reports done, a stream closed cleanly;
   *   'reject' the next read rejects with TypeError('Load failed'), a connection
   *            dropped mid-stream;
   *   'hang'   the next read never settles -- a server that stopped talking without
   *            closing -- until the request's signal is aborted, when every read
   *            still waiting rejects with AbortError.
   * `__delay: ms` makes every read take that many real milliseconds, so a test can
   * abort or look at the page in the middle of one.
   * A frame `{ __chunk: [...frames] }` is several frames in ONE read, which is how a
   * page that re-attaches is handed everything so far, and the page paints once per
   * read rather than once per frame.
   */
  function sseBody(spec, signal) {
    var frames = spec.__sse, then = spec.__then || 'end', delay = Number(spec.__delay) || 0;
    var at = 0, waiting = [], closed = false;
    function settleAll(how) {
      var pending = waiting; waiting = [];
      pending.forEach(how);
    }
    if (signal) {
      signal.addEventListener('abort', function () {
        settleAll(function (w) { w.reject(signal.reason || abortError()); });
      });
    }
    function read() {
      if (signal && signal.aborted) return Promise.reject(signal.reason || abortError());
      if (closed) return Promise.resolve({ done: true, value: undefined });
      return new Promise(function (resolve, reject) {
        var w = { resolve: resolve, reject: reject };
        waiting.push(w);
        function deliver() {
          var i = waiting.indexOf(w);
          if (i < 0) return; // an abort or a cancel got there first
          if (at < frames.length) {
            waiting.splice(i, 1);
            var next = frames[at++];
            var chunk = next && next.__chunk ? next.__chunk : [next];
            resolve({ done: false, value: chunk.map(function (f) { return 'data: ' + JSON.stringify(f) + '\n'; }).join('') });
          } else if (then === 'reject') {
            waiting.splice(i, 1);
            reject(networkError('Load failed'));
          } else if (then === 'hang') {
            // Left in `waiting`, which is what lets abort() reject it later.
          } else {
            waiting.splice(i, 1);
            closed = true;
            resolve({ done: true, value: undefined });
          }
        }
        if (delay > 0) setTimeout(deliver, delay); else deliver();
      });
    }
    return {
      getReader: function () {
        return {
          read: read,
          cancel: function () {
            closed = true;
            settleAll(function (w) { w.resolve({ done: true, value: undefined }); });
            return Promise.resolve();
          },
          releaseLock: function () {},
        };
      },
    };
  }

  function reply(body, ok, headers, signal) {
    // Tests may model an HTTP refusal without replacing the fetch shim. Keep the
    // transport metadata outside the JSON body the page will actually read.
    var explicitStatus = body && typeof body.__status === 'number' ? body.__status : null;
    if (explicitStatus !== null) {
      ok = explicitStatus >= 200 && explicitStatus < 300;
      headers = body.headers || headers;
      body = Object.prototype.hasOwnProperty.call(body, 'body') ? body.body : {};
    }
    // A route may answer with frames instead of a document: { __sse: [...] } is a
    // server-sent-event stream, which is how every generation actually arrives.
    var stream = body && body.__sse ? body : null;
    var text = typeof body === 'string' ? body : JSON.stringify(body);
    return Promise.resolve({
      ok: ok !== false,
      status: explicitStatus !== null ? explicitStatus : (ok === false ? 500 : 200),
      headers: { get: function (n) { return (headers || {})[n] || null; } },
      json: function () { return Promise.resolve(JSON.parse(text)); },
      text: function () { return Promise.resolve(text); },
      body: stream ? sseBody(stream, signal) : null,
    });
  }

  function fetch(url, init) {
    var path = String(url).split('?')[0];
    var signal = init && init.signal ? init.signal : null;
    var record = { url: String(url), path: path, method: (init && init.method) || 'GET', body: null };
    if (init && typeof init.body === 'string') { try { record.body = JSON.parse(init.body); } catch (e) { record.body = init.body; } }
    if (init && init.body instanceof FormData) record.parts = init.body.parts.slice();
    calls.push(record);
    // A request whose signal has already fired never leaves the page. It is still
    // recorded above: that the page tried is exactly what a test may want to see.
    if (signal && signal.aborted) return Promise.reject(signal.reason || abortError());
    var answer = Object.prototype.hasOwnProperty.call(routes, String(url)) ? routes[String(url)]
      : (Object.prototype.hasOwnProperty.call(routes, path) ? routes[path] : null);
    try {
      if (typeof answer === 'function') answer = answer(record);
    } catch (e) {
      return Promise.reject(e);
    }
    // A route function may return a promise: the answer comes when the test settles it,
    // which is how a test holds one request open while the page makes another, or answers
    // two of them in the opposite order.
    if (answer && typeof answer.then === 'function') {
      return answer.then(function (later) {
        if (later && later.__reject) throw networkError(later.__reject);
        return reply(later === null || later === undefined ? {} : later, undefined, undefined, signal);
      });
    }
    if (answer && answer.__reject) return Promise.reject(networkError(answer.__reject));
    return reply(answer === null || answer === undefined ? {} : answer, undefined, undefined, signal);
  }

  // JavaScriptCore has no atob: it is a Web API, not an ECMAScript one, and the page
  // needs it to decode what the app sends. Base64 only, which is all __fromHost uses.
  var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  function atob(input) {
    var s = String(input).replace(/=+$/, ''), out = '', bits = 0, acc = 0;
    for (var i = 0; i < s.length; i++) {
      var v = B64.indexOf(s.charAt(i));
      if (v < 0) continue;
      acc = (acc << 6) | v; bits += 6;
      if (bits >= 8) { bits -= 8; out += String.fromCharCode((acc >> bits) & 0xFF); }
    }
    return out;
  }

  var globals = {
    document: document,
    fetch: fetch,
    FormData: FormData,
    atob: atob,
    // The shim's own TextDecoder wants bytes; this stream yields the text a
    // decoded chunk would already be, so decode is the identity here.
    TextDecoder: function () { this.decode = function (value) { return value == null ? '' : String(value); }; },
    AbortController: AbortController,
    AbortSignal: AbortSignal,
    navigator: { clipboard: null },
    location: { href: 'http://127.0.0.1/' },
    innerHeight: 800,
    scrollY: 0,
    scrollTo: function () {},
    visualViewport: null,
    addEventListener: function () {},
  };
  Object.keys(globals).forEach(function (k) { globalThis[k] = globals[k]; });
  globalThis.window = globalThis;

  function ownAndChildText(node) {
    return node._text + node.children.map(ownAndChildText).join('');
  }
  function noticeTexts(onlyErrors) {
    var found = [];
    (function walk(node) {
      node.children.forEach(function (c) {
        if (c.classList.contains('notice') && (!onlyErrors || c.classList.contains('error'))) found.push(ownAndChildText(c));
        walk(c);
      });
    })(document.getElementById('chat'));
    return found;
  }

  globalThis.__page = {
    byId: byId,
    routes: routes,
    calls: calls,
    element: element,
    /** Every request whose path matches, newest last. */
    requests: function (path) { return calls.filter(function (c) { return c.path === path; }); },
    /**
     * The text of every error notice the page is showing, oldest first. Read the
     * way a person would -- the message AND anything appended after it, such as
     * the button of a notice with an action -- which `textContent` on this fake
     * does not do once an element has children.
     */
    errorNotices: function () { return noticeTexts(true); },
    /** Every notice, error or not, oldest first. */
    notices: function () { return noticeTexts(false); },
    /** Every label the progress strip showed, oldest first. The page writes one only when it changes. */
    progress: function () { return labels.slice(); },
    /** The transcript as a shape a test can assert on. */
    transcript: function () {
      return document.getElementById('chat').children.map(function (turn) {
        var media = [];
        (function walk(node) {
          node.children.forEach(function (c) {
            if (['IMG', 'AUDIO', 'VIDEO', 'A'].indexOf(c.tagName) >= 0) {
              media.push({ tag: c.tagName, src: c.src || c.href || '', text: c.textContent, poster: c.poster || '',
                           controls: !!c.controls, loop: !!c.loop, playsInline: !!c.playsInline, preload: c.preload || '' });
            }
            walk(c);
          });
        })(turn);
        var html = '';
        (function walk(node) {
          html += node.innerHTML || '';
          node.children.forEach(walk);
        })(turn);
        return {
          role: turn.className.indexOf('me') >= 0 ? 'user' : 'assistant',
          html: html,
          text: turn.textContent,
          media: media,
        };
      });
    },
  };
})();
