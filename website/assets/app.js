/* TensorSharp Wiki — shared chrome, navigation, search, and UI behaviour. Bilingual (EN / 中文). */
(function () {
  "use strict";

  // ---- Navigation model (single source of truth, bilingual) ----
  var NAV = [
    { group: { en: "Introduction", zh: "简介" }, items: [
      { page: "index",    label: { en: "Home", zh: "首页" } },
      { page: "overview", label: { en: "Overview & Architecture", zh: "概览与架构" } },
      { page: "features", label: { en: "Features", zh: "功能特性" } },
    ]},
    { group: { en: "Get started", zh: "快速上手" }, items: [
      { page: "getting-started", label: { en: "Getting Started", zh: "快速开始" } },
      { page: "backends", label: { en: "Compute Backends", zh: "计算后端" } },
      { page: "models",   label: { en: "Supported Models", zh: "支持的模型" } },
    ]},
    { group: { en: "Run it", zh: "运行" }, items: [
      { page: "cli",    label: { en: "Command Line (CLI)", zh: "命令行 (CLI)" } },
      { page: "server", label: { en: "Server & Web UI", zh: "服务器与 Web UI" } },
    ]},
    { group: { en: "Integrate", zh: "集成" }, items: [
      { page: "http-api", label: { en: "HTTP API (Ollama / OpenAI)", zh: "HTTP API (Ollama / OpenAI)" } },
      { page: "code-api", label: { en: "C# Library / Code", zh: "C# 库 / 代码" } },
    ]},
    { group: { en: "Deep dive", zh: "深入了解" }, items: [
      { page: "advanced",   label: { en: "Advanced Features", zh: "高级功能" } },
      { page: "benchmarks", label: { en: "Benchmarks & Testing", zh: "基准测试" } },
    ]},
    { group: { en: "Reference", zh: "参考" }, items: [
      { page: "api-reference", label: { en: "API Reference", zh: "API 参考" } },
      { page: "glossary",      label: { en: "Glossary & FAQ", zh: "术语表与 FAQ" } },
    ]},
  ];

  // ---- UI strings (bilingual) ----
  var STRINGS = {
    en: {
      wikiTag: "Wiki",
      searchTrigger: "Search the wiki…",
      searchPlaceholder: "Search pages, commands, flags, API…",
      footOpen: "open", footNav: "navigate", footClose: "close",
      onThisPage: "On this page",
      copy: "Copy", copied: "Copied!",
      menu: "Menu", theme: "Toggle theme", search: "Search", github: "GitHub",
      switchLabel: "中文", switchAria: "切换到中文 (Switch to Chinese)",
      noMatches: function (q) { return "No matches for “" + q + "”."; },
    },
    zh: {
      wikiTag: "维基",
      searchTrigger: "搜索维基…",
      searchPlaceholder: "搜索页面、命令、参数、API…",
      footOpen: "打开", footNav: "切换", footClose: "关闭",
      onThisPage: "本页目录",
      copy: "复制", copied: "已复制！",
      menu: "菜单", theme: "切换主题", search: "搜索", github: "GitHub",
      switchLabel: "EN", switchAria: "Switch to English (切换到英文)",
      noMatches: function (q) { return "未找到 “" + q + "” 的匹配结果。"; },
    },
  };

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

  // ---- Language ----
  function currentLang() {
    var l = (document.documentElement.getAttribute("lang") || "en").toLowerCase();
    return l.indexOf("zh") === 0 ? "zh" : "en";
  }
  function strings() { return STRINGS[currentLang()]; }
  function hrefFor(page, lang) {
    return page + (lang === "zh" ? "_zh-cn" : "") + ".html";
  }
  function otherLangHref() {
    var other = currentLang() === "zh" ? "en" : "zh";
    return hrefFor(currentPage(), other) + (location.hash || "");
  }
  // Rewrite an English (search-index) URL to the current language.
  function localizeUrl(u) {
    if (currentLang() !== "zh" || !u || u.indexOf("_zh-cn") >= 0) return u;
    return u.replace(/^([^#?]+)\.html/, "$1_zh-cn.html");
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
    var S = strings(), lang = currentLang();
    var bar = el("header", { class: "topbar" });
    bar.appendChild(el("button", { id: "menu-toggle", class: "icon-btn", "aria-label": S.menu }, ICONS.menu));
    var brand = el("a", { class: "brand", href: hrefFor("index", lang) });
    brand.innerHTML = '<img src="assets/logo.svg" alt=""><span>TensorSharp</span><span class="tag">' + S.wikiTag + '</span>';
    bar.appendChild(brand);
    bar.appendChild(el("div", { class: "spacer" }));

    var actions = el("div", { class: "topbar-actions" });
    var trig = el("button", { class: "search-trigger", id: "search-trigger", "aria-label": S.search });
    trig.innerHTML = ICONS.search + '<span class="lbl">' + S.searchTrigger + '</span><kbd>/</kbd>';
    actions.appendChild(trig);
    var langBtn = el("a", { class: "icon-btn lang-btn", href: otherLangHref(), "aria-label": S.switchAria, title: S.switchAria }, S.switchLabel);
    actions.appendChild(langBtn);
    var theme = el("button", { class: "icon-btn", id: "theme-toggle", "aria-label": S.theme });
    actions.appendChild(theme);
    var gh = el("a", { class: "icon-btn", href: REPO, target: "_blank", rel: "noopener", "aria-label": S.github }, ICONS.github);
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
    var cur = currentPage(), lang = currentLang();

    NAV.forEach(function (g) {
      var grp = el("div", { class: "nav-group" });
      grp.appendChild(el("h4", null, g.group[lang]));
      g.items.forEach(function (it) {
        var a = el("a", { href: hrefFor(it.page, lang) }, it.label[lang]);
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
    toc.appendChild(el("h5", null, strings().onThisPage));
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
    var S = strings();
    document.querySelectorAll("main.content pre").forEach(function (pre) {
      var wrap = el("div", { class: "code-wrap" });
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      var lang = pre.getAttribute("data-lang");
      if (lang) wrap.appendChild(el("span", { class: "code-lang" }, lang));
      var btn = el("button", { class: "copy-btn", type: "button" }, S.copy);
      wrap.appendChild(btn);
      btn.addEventListener("click", function () {
        var text = pre.innerText;
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = S.copied; btn.classList.add("copied");
          setTimeout(function () { btn.textContent = S.copy; btn.classList.remove("copied"); }, 1600);
        });
      });
    });
  }

  // ---- Search ----
  var overlay, input, results, selIndex = -1, curResults = [];

  function buildSearch() {
    var S = strings();
    overlay = el("div", { id: "search-overlay" });
    var box = el("div", { class: "search-box" });
    input = el("input", { type: "text", placeholder: S.searchPlaceholder, "aria-label": S.search, autocomplete: "off", spellcheck: "false" });
    results = el("div", { class: "search-results" });
    box.appendChild(input);
    box.appendChild(results);
    box.appendChild(el("div", { class: "search-foot" }, "<span><kbd>enter</kbd> " + S.footOpen + "</span><span><kbd>↑</kbd> <kbd>↓</kbd> " + S.footNav + "</span><span><kbd>esc</kbd> " + S.footClose + "</span>"));
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

  function searchIndex() {
    if (currentLang() === "zh" && window.SEARCH_INDEX_ZH) return window.SEARCH_INDEX_ZH;
    return window.SEARCH_INDEX || [];
  }

  function runSearch() {
    var q = input.value.trim().toLowerCase();
    var data = searchIndex();
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
      results.appendChild(el("div", { class: "search-empty" }, strings().noMatches(esc(q))));
      return;
    }
    curResults.forEach(function (d, i) {
      var a = el("a", { href: localizeUrl(d.u) });
      a.innerHTML = '<div class="r-page">' + esc(d.p) + '</div><div class="r-title">' + hl(d.t, q) + '</div><div class="r-snip">' + hl(d.s, q) + '</div>';
      a.addEventListener("mouseenter", function () { select(i); });
      results.appendChild(a);
    });
  }
  function move(dir) { if (!curResults.length) return; select((selIndex + dir + curResults.length) % curResults.length); var sel = results.children[selIndex]; if (sel) sel.scrollIntoView({ block: "nearest" }); }
  function select(i) { selIndex = i; Array.prototype.forEach.call(results.children, function (c, j) { c.classList.toggle("sel", j === i); }); }
  function openSel() { var i = selIndex < 0 ? 0 : selIndex; if (curResults[i]) window.location.href = localizeUrl(curResults[i].u); }

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
