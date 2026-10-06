// Shared by Server.Host and TensorAgent (desktop and mobile). No external dependencies.
(function () {
  'use strict';
  var active = false;

  // TensorAgent's page answers window.TensorSharpI18n with its own translations; the
  // server's page has none, so every string here carries its English with it.
  function tr(key, fallback, params) {
    var i18n = window.TensorSharpI18n;
    if (i18n && typeof i18n.t === 'function') return i18n.t(key, fallback, params);
    return String(fallback).replace(/\{([A-Za-z0-9_]+)\}/g, function (m, name) {
      return params && Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : m;
    });
  }

  function node(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  // Returned PNGs are opaque grayscale: white edits, black preserves. The drawing
  // canvas uses alpha internally so the original remains visible beneath the brush.
  function open(options) {
    if (active) return Promise.reject(new Error(tr('mask.error.alreadyOpen', 'An image selection is already open.')));
    active = true;
    return new Promise(function (resolve, reject) {
      var previousFocus = document.activeElement;
      var root = node('div', 'ts-mask-modal');
      root.setAttribute('role', 'dialog');
      root.setAttribute('aria-modal', 'true');
      root.setAttribute('aria-label', tr('mask.dialog.label', 'Select image area to edit'));
      var panel = node('div', 'ts-mask-panel');
      root.appendChild(panel);
      var title = node('h2', '', tr('mask.dialog.title', 'Select area to edit'));
      panel.appendChild(title);
      panel.appendChild(node('p', 'ts-mask-help', tr('mask.dialog.help', 'Paint the area to change. Everything outside the selection stays unchanged. Zoom in for small details.')));
      var toolbar = node('div', 'ts-mask-toolbar');
      panel.appendChild(toolbar);
      function button(label, handler, className) {
        var b = node('button', className, label);
        b.type = 'button'; b.addEventListener('click', handler); toolbar.appendChild(b);
        return b;
      }
      var erase = false, ready = false, saving = false, pointer = null, currentStroke = null;
      var operations = [], redoOperations = [], scale = 1, fit = 1, pan = false, panStart = null, resizeObserver = null;
      var paintButton = button(tr('mask.tool.paint', 'Paint'), function () { setErase(false); });
      var eraseButton = button(tr('mask.tool.erase', 'Erase'), function () { setErase(true); });
      var panButton = button(tr('mask.tool.pan', 'Pan'), function () { pan = true; updateTools(); });
      function setErase(value) {
        erase = value; pan = false; updateTools();
      }
      function updateTools() {
        paintButton.setAttribute('aria-pressed', String(!erase && !pan));
        eraseButton.setAttribute('aria-pressed', String(erase && !pan));
        panButton.setAttribute('aria-pressed', String(pan));
      }
      setErase(false);
      function range(labelText, min, max, value, step) {
        var label = node('label', 'ts-mask-control', labelText + ' ');
        var input = node('input'); input.type = 'range';
        input.min = min; input.max = max; input.step = step || 1; input.value = value;
        var output = node('span', '', String(value));
        input.setAttribute('aria-label', labelText);
        input.addEventListener('input', function () { output.textContent = input.value; });
        label.appendChild(input); label.appendChild(output); toolbar.appendChild(label);
        return input;
      }
      var brush = range(tr('mask.tool.brush', 'Brush (image pixels)'), 1, 256, 32);
      var zoom = range(tr('mask.tool.zoom', 'Zoom (%)'), 25, 1600, 100, 25);
      var undo = button(tr('mask.tool.undo', 'Undo'), function () {
        if (!ready || saving || pointer !== null) return;
        if (operations.length) redoOperations.push(operations.pop()); redrawSafely();
      });
      var redo = button(tr('mask.tool.redo', 'Redo'), function () {
        if (!ready || saving || pointer !== null) return;
        if (redoOperations.length) operations.push(redoOperations.pop()); redrawSafely();
      });
      button(tr('mask.tool.clear', 'Clear'), function () {
        if (!ready || saving || pointer !== null) return;
        redoOperations = []; operations.push({ clear: true }); redrawSafely();
      });
      button(tr('mask.tool.invert', 'Invert'), function () {
        if (!ready || saving || pointer !== null) return;
        redoOperations = []; operations.push({ invert: true }); redrawSafely();
      });

      var viewport = node('div', 'ts-mask-viewport');
      viewport.setAttribute('aria-label', tr('mask.canvas.viewport', 'Image selection canvas; scroll to pan when zoomed'));
      var stage = node('div', 'ts-mask-stage');
      var source = node('img', 'ts-mask-source'); source.alt = tr('mask.canvas.image', 'Image to edit'); source.draggable = false;
      var canvas = node('canvas', 'ts-mask-canvas');
      canvas.setAttribute('aria-label', tr('mask.canvas.paint', 'Paint or erase the selected image area'));
      stage.appendChild(source); stage.appendChild(canvas); viewport.appendChild(stage); panel.appendChild(viewport);
      var context = null, baseContext = null;
      var base = document.createElement('canvas');
      var status = node('p', 'ts-mask-status', tr('mask.status.loading', 'Loading image…')); status.setAttribute('role', 'status');
      panel.appendChild(status);
      var settings = node('div', 'ts-mask-options');
      var featherLabel = node('label', '', tr('mask.options.feather', 'Soften inside edge (pixels)') + ' ');
      var feather = node('input'); feather.type = 'number'; feather.min = 0; feather.max = 64; feather.step = 1;
      feather.value = options.maskFeather || 0;
      featherLabel.appendChild(feather); settings.appendChild(featherLabel);
      var cropLabel = node('label'); var crop = node('input'); crop.type = 'checkbox'; crop.checked = !!options.maskCrop;
      cropLabel.appendChild(crop); cropLabel.appendChild(document.createTextNode(' ' + tr('mask.options.crop', 'Process selected region only (faster; less surrounding context)')));
      settings.appendChild(cropLabel); panel.appendChild(settings);
      var actions = node('div', 'ts-mask-actions'); panel.appendChild(actions);
      function action(label, handler, className) {
        var b = node('button', className, label); b.type = 'button'; b.addEventListener('click', handler); actions.appendChild(b); return b;
      }
      function finish(value, error) {
        document.removeEventListener('keydown', keydown, true);
        if (resizeObserver) resizeObserver.disconnect();
        root.remove(); active = false;
        // Release large backing stores immediately on phones.
        canvas.width = base.width = 0;
        if (previousFocus && previousFocus.focus) previousFocus.focus();
        if (error) reject(error); else resolve(value);
      }
      function imageUnreadable() {
        finish(null, new Error(tr('mask.error.imageUnreadable', 'The image could not be opened for selection.')));
      }
      function prepareCanvas(element, width, height) {
        element.width = width; element.height = height;
        // Use the readback-oriented store from the first draw so later mask reads
        // do not change rasterization, including antialiased Undo/Redo edges.
        var ctx = element.getContext('2d', { willReadFrequently: true });
        if (!ctx || element.width !== width || element.height !== height) throw new Error('Canvas unavailable');
        // Canvas dimensions alone do not prove its backing store was allocated.
        // Probe the actual store without imposing a fixed image-size limit.
        ctx.fillStyle = '#fff'; ctx.fillRect(width - 1, height - 1, 1, 1);
        if (ctx.getImageData(width - 1, height - 1, 1, 1).data[3] !== 255) throw new Error('Canvas unavailable');
        ctx.clearRect(width - 1, height - 1, 1, 1);
        return ctx;
      }
      var cancel = action(tr('mask.action.cancel', 'Cancel'), function () { if (!saving) finish(null); });
      if (options.maskUrl) action(tr('mask.action.remove', 'Remove selection'), function () { if (!saving) finish({ remove: true }); });
      var apply = action(tr('mask.action.use', 'Use selection'), save, 'ts-mask-primary'); apply.disabled = true;
      function keydown(event) {
        if (event.key === 'Escape') {
          event.preventDefault(); event.stopPropagation();
          if (!saving) finish(null);
          return;
        }
        if (event.key === 'Tab') {
          var controls = Array.prototype.slice.call(panel.querySelectorAll('button, input'))
            .filter(function (control) { return !control.disabled; });
          var first = controls[0], last = controls[controls.length - 1];
          if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
      }
      document.addEventListener('keydown', keydown, true);
      document.body.appendChild(root); cancel.focus();
      function resize() {
        scale = fit * Number(zoom.value) / 100;
        stage.style.width = Math.max(1, Math.round(canvas.width * scale)) + 'px';
        stage.style.height = Math.max(1, Math.round(canvas.height * scale)) + 'px';
      }
      zoom.addEventListener('input', resize);
      function refit() {
        if (!canvas.width || !root.parentNode) return;
        fit = Math.min(1, Math.max(1, viewport.clientWidth - 24) / canvas.width,
          Math.max(1, viewport.clientHeight - 24) / canvas.height);
        resize();
      }
      if (window.ResizeObserver) {
        resizeObserver = new ResizeObserver(refit); resizeObserver.observe(viewport);
      }
      function drawStroke(stroke) {
        if (stroke.clear) { context.clearRect(0, 0, canvas.width, canvas.height); return; }
        if (stroke.invert) {
          var pixels = context.getImageData(0, 0, canvas.width, canvas.height);
          for (var i = 0; i < pixels.data.length; i += 4) {
            pixels.data[i] = 237; pixels.data[i + 1] = 71; pixels.data[i + 2] = 124;
            pixels.data[i + 3] = 255 - pixels.data[i + 3];
          }
          context.putImageData(pixels, 0, 0); return;
        }
        context.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over';
        context.strokeStyle = context.fillStyle = '#ed477c';
        context.lineCap = context.lineJoin = 'round'; context.lineWidth = stroke.size;
        var points = stroke.points;
        if (points.length === 1) {
          context.beginPath(); context.arc(points[0].x, points[0].y, stroke.size / 2, 0, Math.PI * 2); context.fill();
        } else {
          context.beginPath(); context.moveTo(points[0].x, points[0].y);
          for (var i = 1; i < points.length; i++) context.lineTo(points[i].x, points[i].y);
          context.stroke();
        }
        context.globalCompositeOperation = 'source-over';
      }
      function redraw() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(base, 0, 0);
        operations.forEach(function (stroke) {
          if (stroke.clear || stroke.invert) { drawStroke(stroke); return; }
          // Replay the same initial disk and individual segments drawn by pointer
          // events. A single continuous path changes antialiased edge coverage at
          // overlapping round caps, making Undo/Redo alter the saved selection.
          drawStroke({ erase: stroke.erase, size: stroke.size, points: [stroke.points[0]] });
          for (var i = 1; i < stroke.points.length; i++)
            drawStroke({ erase: stroke.erase, size: stroke.size, points: [stroke.points[i - 1], stroke.points[i]] });
        });
        undo.disabled = !operations.length;
        redo.disabled = !redoOperations.length;
      }
      function redrawSafely() {
        try { redraw(); }
        // A failed replay may leave a partial selection, so do not let it save.
        catch (error) { imageUnreadable(); }
      }
      function point(event) {
        var rect = canvas.getBoundingClientRect();
        return { x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)),
          y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)) };
      }
      canvas.addEventListener('pointerdown', function (event) {
        if (!ready || saving || pointer !== null || (event.pointerType === 'mouse' && event.button !== 0)) return;
        event.preventDefault(); pointer = event.pointerId; canvas.setPointerCapture(pointer);
        if (pan) { panStart = { x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop }; return; }
        redoOperations = []; redo.disabled = true;
        currentStroke = { erase: erase || event.button === 5, size: Number(brush.value), points: [point(event)] };
        operations.push(currentStroke);
        try { drawStroke(currentStroke); undo.disabled = false; }
        catch (error) { imageUnreadable(); }
      });
      canvas.addEventListener('pointermove', function (event) {
        if (pointer !== event.pointerId) return;
        event.preventDefault();
        if (panStart) {
          viewport.scrollLeft = panStart.left + panStart.x - event.clientX;
          viewport.scrollTop = panStart.top + panStart.y - event.clientY; return;
        }
        if (!currentStroke) return;
        var events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
        if (!events.length) events = [event];
        try {
          events.forEach(function (sample) {
            var p = point(sample), previous = currentStroke.points[currentStroke.points.length - 1];
            currentStroke.points.push(p);
            // Paint only the new segment. No full-image reads or redraws while dragging.
            drawStroke({ erase: currentStroke.erase, size: currentStroke.size, points: [previous, p] });
          });
        } catch (error) { imageUnreadable(); }
      });
      function endStroke(event) {
        if (pointer !== event.pointerId) return;
        pointer = null; currentStroke = null; panStart = null;
      }
      ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (name) { canvas.addEventListener(name, endStroke); });

      function save() {
        if (!ready || saving || pointer !== null) return;
        var radius = Number(feather.value);
        if (!Number.isInteger(radius) || radius < 0 || radius > 64) { status.textContent = tr('mask.status.featherRange', 'Edge softness must be a whole number from 0 to 64.'); return; }
        var output = null;
        function saveFailed() {
          if (output) output.width = 0;
          saving = false; apply.disabled = cancel.disabled = false;
          status.textContent = tr('mask.status.saveFailed', 'Could not save the selection. Please try again.');
        }
        try {
          var pixels = context.getImageData(0, 0, canvas.width, canvas.height);
          var selected = false;
          for (var i = 0; i < pixels.data.length; i += 4) {
            var alpha = pixels.data[i + 3];
            if (alpha) selected = true;
            pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = alpha; pixels.data[i + 3] = 255;
          }
          if (!selected) { status.textContent = tr('mask.status.nothingPainted', 'Paint an area before using the selection.'); return; }
          saving = true; apply.disabled = cancel.disabled = true; status.textContent = tr('mask.status.preparing', 'Preparing selection…');
          output = document.createElement('canvas');
          prepareCanvas(output, canvas.width, canvas.height).putImageData(pixels, 0, 0);
          pixels = null;
          output.toBlob(function (blob) {
            output.width = 0;
            if (!blob) { saveFailed(); return; }
            finish({ blob: blob, maskFeather: radius, maskCrop: crop.checked });
          }, 'image/png');
        } catch (error) { saveFailed(); }
      }
      function loaded() {
        ready = true; apply.disabled = false; undo.disabled = redo.disabled = true;
        status.textContent = tr('mask.status.ready', '{width} × {height} · Pink marks the area to edit. Drag a scrollbar to pan when zoomed.',
          { width: canvas.width, height: canvas.height });
      }
      source.onload = function () {
        if (!root.parentNode) return;
        var w = source.naturalWidth, h = source.naturalHeight;
        if (!w || !h) { imageUnreadable(); return; }
        try {
          context = prepareCanvas(canvas, w, h);
          baseContext = prepareCanvas(base, w, h);
          refit();
        } catch (error) {
          imageUnreadable(); return;
        }
        if (!options.maskUrl) { loaded(); return; }
        var mask = new Image();
        mask.onload = function () {
          if (!root.parentNode) return;
          if (mask.naturalWidth !== w || mask.naturalHeight !== h) { finish(null, new Error(tr('mask.error.maskSize', 'The saved selection does not match the image dimensions.'))); return; }
          try {
            baseContext.drawImage(mask, 0, 0);
            var pixels = baseContext.getImageData(0, 0, w, h);
            for (var i = 0; i < pixels.data.length; i += 4) {
              var amount = options.maskMode === 'alpha' ? 255 - pixels.data[i + 3]
                : Math.round((77 * pixels.data[i] + 150 * pixels.data[i + 1] + 29 * pixels.data[i + 2]) / 256);
              if (options.maskInvert) amount = 255 - amount;
              pixels.data[i] = 237; pixels.data[i + 1] = 71; pixels.data[i + 2] = 124; pixels.data[i + 3] = amount;
            }
            baseContext.putImageData(pixels, 0, 0); redraw(); loaded();
          } catch (error) {
            finish(null, new Error(tr('mask.error.maskUnreadable', 'The saved selection could not be opened.')));
          }
        };
        mask.onerror = function () { if (root.parentNode) finish(null, new Error(tr('mask.error.maskUnreadable', 'The saved selection could not be opened.'))); };
        mask.src = options.maskUrl;
      };
      source.onerror = function () { if (root.parentNode) imageUnreadable(); };
      source.src = options.sourceUrl;
    });
  }
  window.TensorSharpMaskEditor = { open: open };
})();
