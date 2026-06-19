/* TensorSharp Wiki — shared chrome, navigation, search, and UI behaviour. */
(function () {
  "use strict";

  // ---- Navigation model (single source of truth) ----
  var NAV = [
    { group: "Introduction", items: [
      { page: "index",        href: "index.html",           label: "Home" },
      { page: "overview",     href: "overview.html",         label: "Overview & Architecture" },
      { page: "features",     href: "features.html",         label: "Features" },
    ]},
    { group: "Get started", items: [
      { page: "getting-started", href: "getting-started.html", label: "Getting Started" },
      { page: "backends",     href: "backends.html",         label: "Compute Backends" },
      { page: "models",       href: "models.html",           label: "Supported Models" },
    ]},
    { group: "Run it", items: [
      { page: "cli",          href: "cli.html",              label: "Command Line (CLI)" },
      { page: "server",       href: "server.html",           label: "Server & Web UI" },
    ]},
    { group: "Integrate", items: [
      { page: "http-api",     href: "http-api.html",         label: "HTTP API (Ollama / OpenAI)" },
      { page: "code-api",     href: "code-api.html",         label: "C# Library / Code" },
    ]},
    { group: "Deep dive", items: [
      { page: "advanced",     href: "advanced.html",         label: "Advanced Features" },
      { page: "benchmarks",   href: "benchmarks.html",       label: "Benchmarks & Testing" },
    ]},
    { group: "Reference", items: [
      { page: "api-reference", href: "api-reference.html",   label: "API Reference" },
      { page: "glossary",     href: "glossary.html",         label: "Glossary & FAQ" },
    ]},
  ];

  var REPO = "https://github.com/zhongkaifu/TensorSharp";

  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }

  var ICONS = {
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8"/></svg>',
    moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A8.5 8.5 0 1 1 11.2 3 6.6 6.6 0 0 0 21 12.8Z"/></svg>',
    menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
    github: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2A10 10 0 0 0 8.8 21.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.3-3.4-1.3-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.9.8.1-.6.3-1.1.6-1.4-2.2-.300-4.6-1.1-4.6-5a3.9 3.9 0 0 1 1-2.7 3.6 3.6 0 0 1 .1-2.7s.8-.3 2.7 1a9.3 9.3 0 0 1 5 0c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.4.1 2.7a3.9 3.9 0 0 1 1 2.7c0 3.9-2.3 4.7-4.6 5 .4.3.7.9.7 1.8v2.6c0 .3.2.6.7.5A10 10 0 0 0 12 2Z"/></svg>',
  };

  function currentPage() {
    return (document.body.getAttribute("data-page") || "index").trim();
  }

  // ---- Theme ----
  function applyTheme(t) {
    document.documentElement.classList.toggle("dark", t === "dark");
    var btn = document.getElementById("theme-toggle");
    if (btn) btn.innerHTML = t === "dark" ? ICONS.sun : ICONS.moon;
  }
  function toggleTheme() {
    var t = document.documentElement.classList.contains("dark") ? "light" : "dark";
    try { localStorage.setItem("ts-theme", t); } catch (e) {}
    applyTheme(t);
  }

  // ---- Build top bar ----
  function buildTopbar() {
    var bar = el("header", { class: "topbar" });
    bar.appendChild(el("button", { id: "menu-toggle", class: "icon-btn", "aria-label": "Menu" }, ICONS.menu));
    var brand = el("a", { class: "brand", href: "index.html" });
    brand.innerHTML = '<img src="assets/logo.svg" alt=""><span>TensorSharp</span><span class="tag">Wiki</span>';
    bar.appendChild(brand);
    bar.appendChild(el("div", { class: "spacer" }));

    var actions = el("div", { class: "topbar-actions" });
    var trig = el("button", { class: "search-trigger", id: "search-trigger", "aria-label": "Search" });
    trig.innerHTML = ICONS.search + '<span class="lbl">Search the wiki…</span><kbd>/</kbd>';
    actions.appendChild(trig);
    var theme = el("button", { class: "icon-btn", id: "theme-toggle", "aria-label": "Toggle theme" });
    actions.appendChild(theme);
    var gh = el("a", { class: "icon-btn", href: REPO, target: "_blank", rel: "noopener", "aria-label": "GitHub" }, ICONS.github);
    actions.appendChild(gh);
    bar.appendChild(actions);
    document.body.insertBefore(bar, document.body.firstChild);

    document.getElementById("menu-toggle").addEventListener("click", function () {
      document.body.classList.toggle("menu-open");
    });
    document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
    trig.addEventListener("click", openSearch);
  }

  // ---- Build sidebar + wrap main into layout ----
  function buildLayout() {
    var main = document.querySelector("main.content");
    var layout = el("div", { class: "layout" });
    var sidebar = el("aside", { class: "sidebar", id: "sidebar" });
    var cur = currentPage();

    NAV.forEach(function (g) {
      var grp = el("div", { class: "nav-group" });
      grp.appendChild(el("h4", null, g.group));
      g.items.forEach(function (it) {
        var a = el("a", { href: it.href }, it.label);
        if (it.page === cur) a.className = "active";
        grp.appendChild(a);
      });
      sidebar.appendChild(grp);
    });

    var toc = el("nav", { class: "toc", id: "toc" });

    main.parentNode.insertBefore(layout, main);
    layout.appendChild(sidebar);
    layout.appendChild(main);
    layout.appendChild(toc);

    // close mobile menu on nav click
    sidebar.addEventListener("click", function (e) {
      if (e.target.tagName === "A") document.body.classList.remove("menu-open");
    });
  }

  // ---- Build "On this page" TOC ----
  function buildToc() {
    var toc = document.getElementById("toc");
    var heads = document.querySelectorAll("main.content h2[id], main.content h3[id]");
    if (heads.length < 2) { toc.style.display = "none"; return; }
    toc.appendChild(el("h5", null, "On this page"));
    heads.forEach(function (h) {
      var a = el("a", { href: "#" + h.id, class: h.tagName === "H3" ? "lvl-3" : "lvl-2" }, h.textContent);
      a.dataset.target = h.id;
      toc.appendChild(a);
      var anchor = el("a", { class: "heading-anchor", href: "#" + h.id, "aria-label": "Link to section" }, "#");
      h.appendChild(anchor);
    });

    var links = toc.querySelectorAll("a[data-target]");
    var spy = function () {
      var pos = window.scrollY + 120;
      var current = null;
      heads.forEach(function (h) { if (h.offsetTop <= pos) current = h.id; });
      links.forEach(function (l) { l.classList.toggle("active", l.dataset.target === current); });
    };
    window.addEventListener("scroll", spy, { passive: true });
    spy();
  }

  // ---- Copy buttons + language labels on code blocks ----
  function enhanceCode() {
    document.querySelectorAll("main.content pre").forEach(function (pre) {
      var wrap = el("div", { class: "code-wrap" });
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      var lang = pre.getAttribute("data-lang");
      if (lang) wrap.appendChild(el("span", { class: "code-lang" }, lang));
      var btn = el("button", { class: "copy-btn", type: "button" }, "Copy");
      wrap.appendChild(btn);
      btn.addEventListener("click", function () {
        var text = pre.innerText.replace(/\n?Copy$/, "");
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = "Copied!"; btn.classList.add("copied");
          setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1600);
        });
      });
    });
  }

  // ---- Search ----
  var overlay, input, results, selIndex = -1, curResults = [];

  function buildSearch() {
    overlay = el("div", { id: "search-overlay" });
    var box = el("div", { class: "search-box" });
    input = el("input", { type: "text", placeholder: "Search pages, commands, flags, API…", "aria-label": "Search", autocomplete: "off", spellcheck: "false" });
    results = el("div", { class: "search-results" });
    box.appendChild(input);
    box.appendChild(results);
    box.appendChild(el("div", { class: "search-foot" }, "<span><kbd>enter</kbd> open</span><span><kbd>up</kbd> <kbd>down</kbd> navigate</span><span><kbd>esc</kbd> close</span>"));
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", function (e) { if (e.target === overlay) closeSearch(); });
    input.addEventListener("input", runSearch);
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") { e.preventDefault(); openSel(); }
      else if (e.key === "Escape") { closeSearch(); }
    });
  }

  function openSearch() { overlay.classList.add("open"); input.value = ""; input.focus(); runSearch(); }
  function closeSearch() { overlay.classList.remove("open"); }

  function runSearch() {
    var q = input.value.trim().toLowerCase();
    var data = window.SEARCH_INDEX || [];
    selIndex = -1;
    if (!q) {
      curResults = data.slice(0, 8);
    } else {
      var terms = q.split(/\s+/);
      curResults = data.map(function (d) {
        var hay = (d.t + " " + d.s + " " + (d.k || "") + " " + d.p).toLowerCase();
        var score = 0, ok = true;
        terms.forEach(function (t) {
          var i = hay.indexOf(t);
          if (i < 0) { ok = false; return; }
          score += 10;
          if ((d.t + " " + (d.k || "")).toLowerCase().indexOf(t) >= 0) score += 25;
          if (d.t.toLowerCase().indexOf(t) === 0) score += 15;
        });
        return ok ? { d: d, score: score } : null;
      }).filter(Boolean).sort(function (a, b) { return b.score - a.score; }).slice(0, 20).map(function (x) { return x.d; });
    }
    renderResults(q);
  }

  function hl(text, q) {
    if (!q) return esc(text);
    var out = esc(text);
    q.split(/\s+/).forEach(function (t) {
      if (!t) return;
      out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    });
    return out;
  }
  function esc(s) { return (s || "").replace(/[&<>]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]; }); }

  function renderResults(q) {
    results.innerHTML = "";
    if (!curResults.length) {
      results.appendChild(el("div", { class: "search-empty" }, "No matches for “" + esc(q) + "”."));
      return;
    }
    curResults.forEach(function (d, i) {
      var a = el("a", { href: d.u });
      a.innerHTML = '<div class="r-page">' + esc(d.p) + '</div><div class="r-title">' + hl(d.t, q) + '</div><div class="r-snip">' + hl(d.s, q) + '</div>';
      a.addEventListener("mouseenter", function () { select(i); });
      results.appendChild(a);
    });
  }
  function move(dir) { if (!curResults.length) return; select((selIndex + dir + curResults.length) % curResults.length); var sel = results.children[selIndex]; if (sel) sel.scrollIntoView({ block: "nearest" }); }
  function select(i) { selIndex = i; Array.prototype.forEach.call(results.children, function (c, j) { c.classList.toggle("sel", j === i); }); }
  function openSel() { var i = selIndex < 0 ? 0 : selIndex; if (curResults[i]) window.location.href = curResults[i].u; }

  // ---- Global keys ----
  function globalKeys() {
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && !/input|textarea/i.test(document.activeElement.tagName) && !overlay.classList.contains("open")) {
        e.preventDefault(); openSearch();
      } else if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault(); openSearch();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var t = "light";
    try { t = localStorage.getItem("ts-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (e) {}
    applyTheme(t);
    buildTopbar();
    buildLayout();
    buildToc();
    enhanceCode();
    buildSearch();
    globalKeys();
  });
})();
