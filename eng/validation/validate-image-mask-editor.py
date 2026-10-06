#!/usr/bin/env python3
"""Exercise the shipped mask editor and Server Chat in isolated desktop/mobile Chromium.

Uses a loopback fixture for uploads/SSE; no model inference is claimed. Checks
native-size grayscale export, brush/erase/undo, touch coordinates, pan, per-photo
selection reuse, transactional target changes, large images and request wiring.
Real model tests use qwen-image21-mask-bench.py.
"""
import argparse
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def fixture():
    uploads, requests = {}, []
    image = io.BytesIO()
    Image.new('RGB', (640, 360), (90, 130, 180)).save(image, format='PNG')
    uploads['source.png'] = image.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, content_type='application/json'):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/':
                self.respond((ROOT / 'TensorSharp.Server.Host/wwwroot/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif path in ('/mask-editor.js', '/mask-editor.css'):
                self.respond((ROOT / 'TensorSharp.Chat/WebUi' / path[1:]).read_bytes(),
                             'text/javascript' if path.endswith('js') else 'text/css')
            elif path.startswith('/uploads/') and path[9:] in uploads:
                self.respond(uploads[path[9:]], 'image/png')
            elif path == '/api/models':
                self.respond(b'{"loaded":"Mask editor fixture","architecture":"qwen_image"}')
            elif path == '/api/queue/status':
                self.respond(b'{"processing":0,"pending_requests":0}')
            elif path == '/api/skills':
                self.respond(b'{"skills":[]}')
            else:
                self.send_error(404)

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            if self.path == '/api/sessions':
                self.respond(b'{"sessionId":"mask-fixture"}')
            elif self.path == '/api/upload':
                message = BytesParser(policy=default).parsebytes(
                    ('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + data)
                files = []
                for part in message.iter_parts():
                    filename = Path(part.get_filename()).name
                    if filename == 'selection.png':
                        filename = f'selection-{len(uploads)}.png'
                    uploads[filename] = part.get_payload(decode=True)
                    uploaded = dict(ok=True, file=filename, fileName=filename,
                                    mediaType='image', url='/uploads/' + filename)
                    if filename.endswith('.heic'):
                        # Browser-routing fixture only: real HEIC decoding is covered
                        # by WebUiChatServiceTests and the live server validation.
                        source = Image.open(io.BytesIO(uploads[filename])).convert('RGB')
                        preview = io.BytesIO()
                        source.resize((300, 200)).save(preview, format='PNG')
                        preview_name = filename + '-preview.png'
                        uploads[preview_name] = preview.getvalue()
                        uploaded['previewUrl'] = '/uploads/' + preview_name
                        if filename == 'unavailable.heic':
                            uploaded['editUnavailableReason'] = 'A full-resolution image for area selection could not be prepared. Reattach this photo as PNG or JPEG.'
                        else:
                            edit_name = filename + '-edit.png'
                            uploads[edit_name] = uploads[filename]
                            uploaded['editUrl'] = '/uploads/' + edit_name
                    files.append(uploaded)
                self.respond(json.dumps(files[0] if len(files) == 1 else dict(ok=True, files=files)).encode())
            elif self.path == '/api/image-edit/stream':
                requests.append(json.loads(data))
                self.respond(b'data: {"done":true,"url":"/uploads/source.png","width":640,"height":360}\n\n', 'text/event-stream')
            else:
                self.send_error(404)

        def do_DELETE(self):
            self.respond(b'{}')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', uploads, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def stroke(page, canvas, start, end, touch):
    box = canvas.bounding_box()
    x0, y0 = box['x'] + start[0] * box['width'], box['y'] + start[1] * box['height']
    x1, y1 = box['x'] + end[0] * box['width'], box['y'] + end[1] * box['height']
    if touch:
        # Browser-native touch input, not direct manipulation of canvas state.
        cdp = page.context.new_cdp_session(page)
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x0, 'y': y0, 'id': 1}]})
        for step in range(1, 11):
            cdp.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [
                {'x': x0 + (x1 - x0)*step/10, 'y': y0 + (y1 - y0)*step/10, 'id': 1}]})
        cdp.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
        cdp.detach()
    else:
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()


def exercise(browser, url, uploads, requests, mobile, out):
    name = 'mobile' if mobile else 'desktop'
    context = browser.new_context(viewport={'width': 390 if mobile else 1280, 'height': 844 if mobile else 900},
                                  is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    page.locator('#file-input').set_input_files({'name': 'source.png', 'mimeType': 'image/png', 'buffer': uploads['source.png']})
    page.get_by_role('button', name='Select area', exact=True).click()
    dialog = page.get_by_role('dialog', name='Select image area to edit')
    canvas = dialog.locator('canvas.ts-mask-canvas')
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    dialog.get_by_role('button', name='Use selection').click()
    expect(dialog.get_by_role('status')).to_contain_text('Paint an area')
    stroke(page, canvas, (.25, .5), (.75, .5), mobile)
    painted_raster = canvas.evaluate('(canvas) => canvas.toDataURL()')
    dialog.get_by_role('button', name='Erase', exact=True).click()
    stroke(page, canvas, (.5, .48), (.5, .52), mobile)
    erased_raster = canvas.evaluate('(canvas) => canvas.toDataURL()')
    assert erased_raster != painted_raster, 'Erase must change the selected raster'
    dialog.get_by_role('button', name='Undo', exact=True).click()
    assert canvas.evaluate('(canvas) => canvas.toDataURL()') == painted_raster, 'Undo must restore every mask sample exactly'
    dialog.get_by_role('button', name='Redo', exact=True).click()
    assert canvas.evaluate('(canvas) => canvas.toDataURL()') == erased_raster, 'Redo must restore every mask sample exactly'
    dialog.get_by_role('button', name='Undo', exact=True).click()
    assert canvas.evaluate('(canvas) => canvas.toDataURL()') == painted_raster
    zoom = dialog.get_by_role('slider', name='Zoom (%)')
    zoom.focus()
    zoom.press('Home')
    for _ in range(7):
        zoom.press('ArrowRight')
    expect(zoom).to_have_value('200')
    dialog.get_by_role('button', name='Pan', exact=True).click()
    stroke(page, canvas, (.25, .25), (.18, .2), mobile)
    # Panning must not paint; the exported raster below remains the same stripe.
    zoom.press('Home')
    for _ in range(3):
        zoom.press('ArrowRight')
    page.screenshot(path=str(out / (name + '-editor.png')))
    with page.expect_response(lambda response: response.url.endswith('/api/upload')) as upload_response:
        dialog.get_by_role('button', name='Use selection').click()
    assert upload_response.value.ok
    selection_file = upload_response.value.json()['file']
    expect(dialog).not_to_be_visible()
    expect(page.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
    mask = np.asarray(Image.open(io.BytesIO(uploads[selection_file])).convert('RGBA'))
    assert mask.shape == (360, 640, 4), mask.shape
    assert np.all(mask[:, :, 0] == mask[:, :, 1]) and np.all(mask[:, :, 1] == mask[:, :, 2])
    assert np.all(mask[:, :, 3] == 255)
    assert mask[180, 320, 0] == 255 and mask[20, 20, 0] == 0
    Image.fromarray(mask).save(out / (name + '-mask.png'))
    page.locator('#message-input').fill('Change only the selected stripe.')
    page.locator('#btn-send').click()
    expect(page.get_by_role('button', name='Edit again', exact=True)).to_be_visible()
    request = requests[-1]
    assert request['imagePaths'] == ['source.png'], request
    assert request['maskPath'] == selection_file and request['maskMode'] == 'grayscale', request
    page.get_by_role('button', name='Show original', exact=True).click()
    expect(page.get_by_role('button', name='Show edited', exact=True)).to_have_attribute('aria-pressed', 'true')
    page.get_by_role('button', name='Edit again', exact=True).click()
    expect(page.locator('#message-input')).to_have_value('Change only the selected stripe.')
    page.get_by_role('button', name='Edit selection', exact=True).click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    dialog.get_by_role('button', name='Invert', exact=True).click()
    with page.expect_response(lambda response: response.url.endswith('/api/upload')) as upload_response:
        dialog.get_by_role('button', name='Use selection').click()
    assert upload_response.value.ok
    selection_file = upload_response.value.json()['file']
    expect(dialog).not_to_be_visible()
    inverted = np.asarray(Image.open(io.BytesIO(uploads[selection_file])).convert('RGBA'))
    assert inverted[180, 320, 0] == 0 and inverted[20, 20, 0] == 255
    page.get_by_role('button', name='Edit selection', exact=True).click()
    dialog.get_by_role('button', name='Remove selection', exact=True).click()
    expect(page.get_by_role('button', name='Select area', exact=True)).to_be_visible()
    assert not errors, errors
    context.close()
    return {'surface': name, 'status': 'passed', 'native_dimensions': [640, 360], 'input': 'touch' if mobile else 'mouse',
            'checks': ['empty guard', 'paint', 'erase', 'pixel-identical undo/redo', 'zoom/pan', 'opaque grayscale export', 'source coordinate alignment',
                       'mask excluded from image references', 'chat request', 'compare', 'edit again', 'reopen/invert', 'remove']}


def exercise_multiple_photos(browser, url, uploads, requests, mobile, out):
    name = 'mobile-multiple' if mobile else 'desktop-multiple'
    context = browser.new_context(viewport={'width': 390 if mobile else 1280, 'height': 844 if mobile else 900},
                                  is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
    page = context.new_page()
    errors, alerts = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('dialog', lambda dialog: (alerts.append(dialog.message), dialog.accept()))
    page.goto(url)
    files = []
    for filename, dimensions, color in [('first.png', (640, 360), (90, 130, 180)),
                                         ('second.png', (360, 640), (180, 90, 130)),
                                         ('third.png', (256, 256), (130, 180, 90))]:
        source = io.BytesIO()
        Image.new('RGB', dimensions, color).save(source, format='PNG')
        files.append({'name': filename, 'mimeType': 'image/png', 'buffer': source.getvalue()})
    page.locator('#file-input').set_input_files(files)
    expect(page.locator('.image-edit-attachment')).to_have_count(3)
    dialog = page.get_by_role('dialog', name='Select image area to edit')

    def photo(filename):
        return page.locator(f'.image-edit-attachment[data-file="{filename}"]')

    def order():
        return page.locator('.image-edit-attachment').evaluate_all('(nodes) => nodes.map(node => node.dataset.file)')

    def masks():
        # Observe application state to verify dormant per-photo selections survive;
        # all state changes still come from real browser controls and uploads.
        return page.evaluate('pendingAttachments.map(a => ({file:a.file, maskPath:a.maskPath || null}))')

    def open_selection(filename):
        photo(filename).locator('.image-selection-button').click()
        expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()

    def save_selection():
        with page.expect_response(lambda response: response.url.endswith('/api/upload')) as response:
            dialog.get_by_role('button', name='Use selection').click()
        assert response.value.ok
        expect(dialog).not_to_be_visible()
        # The page processes the upload JSON in a continuation after the response.
        page.wait_for_function('pendingUploadCount === 0')
        return response.value.json()['file']

    def send(prompt):
        count = len(requests)
        page.locator('#message-input').fill(prompt)
        page.locator('#btn-send').click()
        expect(page.get_by_role('button', name='Edit again', exact=True)).to_have_count(count - start_requests + 1)
        page.wait_for_function('!isGenerating')
        assert len(requests) == count + 1
        return requests[-1]

    def edit_again():
        page.get_by_role('button', name='Edit again', exact=True).last.click()

    start_requests = len(requests)
    assert order() == ['first.png', 'second.png', 'third.png']
    expect(photo('first.png').locator('.image-selection-role')).to_have_text('Editing target')
    expect(page.locator('.image-selection-button')).to_have_count(3)
    open_selection('first.png')
    stroke(page, dialog.locator('canvas.ts-mask-canvas'), (.25, .5), (.75, .5), mobile)
    first_mask = save_selection()
    initial_masks = masks()

    # Cancel on a reference must preserve the current mask and target order.
    open_selection('second.png')
    stroke(page, dialog.locator('canvas.ts-mask-canvas'), (.5, .25), (.5, .75), mobile)
    dialog.get_by_role('button', name='Cancel', exact=True).click()
    expect(dialog).not_to_be_visible()
    assert order() == ['first.png', 'second.png', 'third.png']
    assert masks() == initial_masks

    # A failed mask upload must be equally transactional.
    open_selection('second.png')
    stroke(page, dialog.locator('canvas.ts-mask-canvas'), (.5, .25), (.5, .75), mobile)
    page.route('**/api/upload', lambda route: route.fulfill(status=503, content_type='application/json',
                                                          body='{"ok":false,"error":"Simulated mask upload failure"}'))
    dialog.get_by_role('button', name='Use selection').click()
    page.wait_for_function('pendingUploadCount === 0')
    page.unroute('**/api/upload')
    assert alerts == ['Selection error: Simulated mask upload failure'], alerts
    assert order() == ['first.png', 'second.png', 'third.png']
    assert masks() == initial_masks

    open_selection('second.png')
    stroke(page, dialog.locator('canvas.ts-mask-canvas'), (.5, .25), (.5, .75), mobile)
    second_mask = save_selection()
    assert order() == ['second.png', 'first.png', 'third.png']
    expect(photo('second.png').locator('.image-selection-role')).to_have_text('Editing target')
    expect(photo('second.png').locator('.image-selection-status')).to_have_text('Outside selection protected')
    expect(photo('first.png').locator('.image-selection-role')).to_have_text('Reference')
    expect(photo('first.png').locator('.image-selection-status')).to_have_text('Selection saved')
    second_pixels = np.asarray(Image.open(io.BytesIO(uploads[second_mask])).convert('RGBA'))
    assert second_pixels.shape == (640, 360, 4), second_pixels.shape
    assert second_pixels[320, 180, 0] == 255 and second_pixels[20, 20, 0] == 0
    Image.fromarray(second_pixels).save(out / (name + '-mask.png'))
    page.screenshot(path=str(out / (name + '-target.png')))
    saved_masks = masks()
    assert saved_masks == [{'file': 'second.png', 'maskPath': second_mask},
                           {'file': 'first.png', 'maskPath': first_mask}, {'file': 'third.png', 'maskPath': None}]
    request = send('Change only the selected region on the second photo.')
    assert request['imagePaths'] == ['second.png', 'first.png', 'third.png'], request
    assert request['maskPath'] == second_mask and request['maskMode'] == 'grayscale', request
    assert 'attachments' not in request and first_mask not in json.dumps(request), request
    page.get_by_role('button', name='Show original', exact=True).last.click()
    expect(page.locator('img[alt="original image"]')).to_have_attribute('src', '/uploads/second.png')

    # Edit again retains the promoted source AND all reference selections.
    edit_again()
    assert order() == ['second.png', 'first.png', 'third.png']
    assert masks() == saved_masks
    expect(photo('second.png').locator('.image-selection-status')).to_have_text('Outside selection protected')
    # Reusing a saved selection on a reference promotes that photo, keeping others ordered.
    open_selection('first.png')
    first_mask = save_selection()
    assert order() == ['first.png', 'second.png', 'third.png']
    assert masks()[1]['maskPath'] == second_mask
    open_selection('second.png')
    second_mask = save_selection()
    assert order() == ['second.png', 'first.png', 'third.png']
    # Clearing the active selection must not activate a different saved selection.
    open_selection('second.png')
    dialog.get_by_role('button', name='Remove selection', exact=True).click()
    expect(dialog).not_to_be_visible()
    assert order() == ['second.png', 'first.png', 'third.png']
    request = send('Edit the whole second photo.')
    assert request['imagePaths'] == ['second.png', 'first.png', 'third.png']
    assert 'maskPath' not in request, request
    edit_again()
    assert masks()[1]['maskPath'] == first_mask
    # Restore a mask on B, then remove B entirely: A's saved mask stays dormant.
    open_selection('second.png')
    stroke(page, dialog.locator('canvas.ts-mask-canvas'), (.5, .25), (.5, .75), mobile)
    save_selection()
    photo('second.png').get_by_role('button', name='Remove second.png', exact=True).click()
    assert order() == ['first.png', 'third.png']
    expect(photo('first.png').locator('.image-selection-status')).to_have_text('Selection saved')
    request = send('Edit the whole remaining source.')
    assert request['imagePaths'] == ['first.png', 'third.png'] and 'maskPath' not in request, request
    edit_again()
    assert masks()[0]['maskPath'] == first_mask
    expect(photo('first.png').locator('.image-selection-status')).to_have_text('Selection saved')
    open_selection('first.png')
    first_mask = save_selection()
    # Removing a reference does not disturb the active selection.
    photo('third.png').get_by_role('button', name='Remove third.png', exact=True).click()
    request = send('Edit the saved region after explicitly choosing it again.')
    assert request['imagePaths'] == ['first.png'] and request['maskPath'] == first_mask, request
    assert not errors, errors
    context.close()
    return {'surface': name, 'status': 'passed', 'native_dimensions': [360, 640],
            'input': 'touch' if mobile else 'mouse', 'checks': ['controls for every photo', 'cancel preserves target',
            'failed upload preserves target', 'second photo native mask geometry', 'target promotion with stable references',
            'active mask only on wire', 'compare target original', 'edit again preserves every selection',
            'saved reference selection reactivation', 'clear active mask keeps references dormant',
            'remove active photo keeps references dormant', 'remove reference preserves active mask']}


def exercise_converted_photos(browser, url, uploads, requests, mobile, out):
    name = 'mobile-converted' if mobile else 'desktop-converted'
    context = browser.new_context(viewport={'width': 390 if mobile else 1280, 'height': 650 if mobile else 900},
                                  is_mobile=mobile, has_touch=mobile)
    page = context.new_page()
    errors, alerts = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('dialog', lambda dialog: (alerts.append(dialog.message), dialog.accept()))
    page.goto(url)
    source = io.BytesIO()
    Image.new('RGB', (1200, 800), (100, 150, 180)).save(source, format='PNG')
    page.locator('#file-input').set_input_files({'name': 'camera.heic', 'mimeType': 'image/heic', 'buffer': source.getvalue()})
    camera = page.locator('.image-edit-attachment[data-file="camera.heic"]')
    expect(camera.locator('img')).to_have_attribute('src', '/uploads/camera.heic-preview.png')
    camera.get_by_role('button', name='Select area', exact=True).click()
    dialog = page.get_by_role('dialog', name='Select image area to edit')
    canvas = dialog.locator('canvas.ts-mask-canvas')
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    expect(canvas).to_have_attribute('width', '1200')
    expect(canvas).to_have_attribute('height', '800')
    stroke(page, canvas, (.3, .5), (.7, .5), mobile)
    with page.expect_response(lambda response: response.url.endswith('/api/upload')) as response:
        dialog.get_by_role('button', name='Use selection').click()
    assert response.value.ok
    mask_path = response.value.json()['file']
    assert Image.open(io.BytesIO(uploads[mask_path])).size == (1200, 800)
    expect(camera.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
    page.locator('#message-input').fill('Edit the selected region of the converted photo.')
    page.locator('#btn-send').click()
    expect(page.get_by_role('button', name='Edit again', exact=True)).to_be_visible()
    assert requests[-1]['imagePaths'] == ['camera.heic'] and requests[-1]['maskPath'] == mask_path
    page.get_by_role('button', name='Show original', exact=True).click()
    expect(page.locator('img[alt="original image"]')).to_have_attribute('src', '/uploads/camera.heic-edit.png')
    page.get_by_role('button', name='Edit again', exact=True).click()
    expect(camera.locator('img')).to_have_attribute('src', '/uploads/camera.heic-preview.png')
    camera.get_by_role('button', name='Edit selection', exact=True).click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    expect(canvas).to_have_attribute('width', '1200')
    expect(canvas).to_have_attribute('height', '800')
    dialog.get_by_role('button', name='Cancel', exact=True).click()

    # Many photos must scroll within the composer, keeping input/Send reachable.
    files = [{'name': f'extra-{index}.png', 'mimeType': 'image/png', 'buffer': uploads['source.png']} for index in range(10)]
    files.append({'name': 'unavailable.heic', 'mimeType': 'image/heic', 'buffer': source.getvalue()})
    page.locator('#file-input').set_input_files(files)
    expect(page.locator('.image-edit-attachment')).to_have_count(12)
    strip = page.locator('#attachments')
    assert strip.bounding_box()['height'] <= page.viewport_size['height'] * .4 + 1
    if mobile:
        assert strip.evaluate('(element) => element.scrollHeight > element.clientHeight')
    for control in [page.locator('#message-input'), page.locator('#btn-send')]:
        box = control.bounding_box()
        assert box['y'] >= 0 and box['y'] + box['height'] <= page.viewport_size['height'], box
    page.locator('.image-edit-attachment[data-file="unavailable.heic"] .image-selection-button').click()
    assert alerts and 'could not be prepared' in alerts[-1]
    expect(dialog).not_to_be_visible()
    assert page.locator('.image-edit-attachment').first.get_attribute('data-file') == 'camera.heic'
    page.screenshot(path=str(out / (name + '-many-photos.png')))

    # If the selected photo disappears while upload is pending, late success must
    # not resurrect it or splice the last remaining reference out of the draft.
    delayed = []
    page.locator('.image-edit-attachment[data-file="extra-8.png"] .image-selection-button').click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    stroke(page, canvas, (.3, .5), (.7, .5), mobile)
    page.route('**/api/upload', lambda route: delayed.append(route))
    with page.expect_request('**/api/upload'):
        dialog.get_by_role('button', name='Use selection').click()
    page.locator('.image-edit-attachment[data-file="extra-8.png"]').get_by_role('button', name='Remove extra-8.png', exact=True).click()
    remaining = page.locator('.image-edit-attachment').evaluate_all('(nodes) => nodes.map(node => node.dataset.file)')
    delayed.pop().fulfill(status=200, content_type='application/json', body='{"ok":true,"file":"late-mask.png"}')
    page.wait_for_function('pendingUploadCount === 0')
    assert page.locator('.image-edit-attachment').evaluate_all('(nodes) => nodes.map(node => node.dataset.file)') == remaining
    page.unroute('**/api/upload')
    page.locator('.image-edit-attachment[data-file="extra-9.png"] .image-selection-button').click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    stroke(page, canvas, (.3, .5), (.7, .5), mobile)
    page.route('**/api/upload', lambda route: delayed.append(route))
    with page.expect_request('**/api/upload'):
        dialog.get_by_role('button', name='Use selection').click()
    page.get_by_role('button', name='New Chat').click()
    expect(page.locator('.image-edit-attachment')).to_have_count(0)
    delayed.pop().fulfill(status=200, content_type='application/json', body='{"ok":true,"file":"late-mask.png"}')
    page.wait_for_function('pendingUploadCount === 0')
    expect(page.locator('.image-edit-attachment')).to_have_count(0)
    assert not errors, errors
    context.close()
    return {'surface': name, 'status': 'passed', 'native_dimensions': [1200, 800],
            'checks': ['full-resolution edit source differs from thumbnail', 'original target path on wire',
                       'full-resolution compare and selection reopen', 'unavailable conversion refusal',
                       '12-photo composer scrolling and reachable Send/input', 'late upload after removal',
                       'late upload after New Chat'],
            'limitations': 'Converted URL routing fixture; HEIC codec tested separately in managed/live validation.'}


def exercise_large_photos(browser, url, uploads, requests, mobile, out, source_file=None):
    name = 'mobile-large' if mobile else 'desktop-large'
    results = []
    # Cross each former limit independently, including both side orientations.
    # HEIC here is only a converted-source routing fixture, not a codec test.
    cases = [('large-area.png', (6000, 4000)), ('wide.png', (9000, 1000)), ('tall.heic', (1000, 9000))]
    source_bytes = None
    mime_type = 'image/png'
    if source_file is not None:
        source_bytes = source_file.read_bytes()
        with Image.open(io.BytesIO(source_bytes)) as source:
            cases = [(source_file.name, ImageOps.exif_transpose(source).size)]
            mime_type = Image.MIME.get(source.format, 'application/octet-stream')
    for filename, (width, height) in cases:
        context = browser.new_context(viewport={'width': 390 if mobile else 1280, 'height': 844 if mobile else 900},
                                      is_mobile=mobile, has_touch=mobile)
        page = context.new_page()
        errors, alerts = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('dialog', lambda dialog: (alerts.append(dialog.message), dialog.accept()))
        page.goto(url)
        if source_bytes is None:
            source = io.BytesIO()
            Image.new('RGB', (width, height), (90, 130, 180)).save(source, format='PNG')
            image_bytes = source.getvalue()
        else:
            image_bytes = source_bytes
        page.locator('#file-input').set_input_files({'name': filename,
            'mimeType': 'image/heic' if filename.endswith('.heic') else mime_type, 'buffer': image_bytes})
        photo = page.locator(f'.image-edit-attachment[data-file="{filename}"]')
        photo.get_by_role('button', name='Select area', exact=True).click()
        dialog = page.get_by_role('dialog', name='Select image area to edit')
        canvas = dialog.locator('canvas.ts-mask-canvas')

        def ready():
            expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
            expect(canvas).to_have_attribute('width', str(width))
            expect(canvas).to_have_attribute('height', str(height))

        def save():
            with page.expect_response(lambda response: response.url.endswith('/api/upload')) as response:
                dialog.get_by_role('button', name='Use selection').click()
            assert response.value.ok
            expect(dialog).not_to_be_visible()
            page.wait_for_function('pendingUploadCount === 0')
            return response.value.json()['file']

        ready()
        stroke(page, canvas, (.25, .5), (.75, .5), mobile)
        painted = canvas.evaluate('(canvas) => canvas.toDataURL()')
        dialog.get_by_role('button', name='Erase', exact=True).click()
        stroke(page, canvas, (.5, .48), (.5, .52), mobile)
        erased = canvas.evaluate('(canvas) => canvas.toDataURL()')
        assert erased != painted
        dialog.get_by_role('button', name='Undo', exact=True).click()
        assert canvas.evaluate('(canvas) => canvas.toDataURL()') == painted
        dialog.get_by_role('button', name='Redo', exact=True).click()
        assert canvas.evaluate('(canvas) => canvas.toDataURL()') == erased
        dialog.get_by_role('button', name='Undo', exact=True).click()
        # Do not retain screenshots of user-supplied photos in validation evidence.
        if source_file is None:
            page.screenshot(path=str(out / (name + '-' + filename + '.png')))
        mask_path = save()
        with Image.open(io.BytesIO(uploads[mask_path])) as mask:
            assert mask.size == (width, height), mask.size
            assert mask.convert('RGBA').getpixel((width // 2, height // 2)) == (255, 255, 255, 255)
            assert mask.convert('RGBA').getpixel((20, 20)) == (0, 0, 0, 255)

        # Reopen the native-size mask and export its inversion at the same size.
        photo.get_by_role('button', name='Edit selection', exact=True).click()
        ready()
        dialog.get_by_role('button', name='Invert', exact=True).click()
        inverted_path = save()
        with Image.open(io.BytesIO(uploads[inverted_path])) as mask:
            assert mask.size == (width, height), mask.size
            assert mask.convert('RGBA').getpixel((width // 2, height // 2)) == (0, 0, 0, 255)
            assert mask.convert('RGBA').getpixel((20, 20)) == (255, 255, 255, 255)
        page.locator('#message-input').fill('Edit the large image selection.')
        page.locator('#btn-send').click()
        expect(page.get_by_role('button', name='Edit again', exact=True)).to_be_visible()
        assert requests[-1]['imagePaths'] == [filename] and requests[-1]['maskPath'] == inverted_path
        assert not errors, errors
        assert not alerts, alerts
        results.append({'file': filename, 'native_dimensions': [width, height], 'status': 'passed'})
        context.close()
    return {'surface': name, 'status': 'passed', 'images': results,
            'input': 'touch' if mobile else 'mouse',
            'checks': (['above former megapixel limit', 'above former width/height limits'] if source_file is None else ['supplied photo']) +
                      ['paint/erase', 'pixel-identical undo/redo', 'native-size grayscale export',
                       'native-size reopen/invert', 'original reference and mask request wiring'],
            'limitations': ('HEIC converted-source routing fixture; no real HEIC codec or model inference.' if source_file is None
                            else 'Supplied photo decoded by browser; upload and inference routes are fixture responses.')}


def exercise_canvas_failures(browser, url, uploads):
    context = browser.new_context(viewport={'width': 1280, 'height': 900})
    page = context.new_page()
    errors, alerts = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('dialog', lambda dialog: (alerts.append(dialog.message), dialog.accept()))
    page.add_init_script('''
      window.maskFailure = null;
      const read = CanvasRenderingContext2D.prototype.getImageData;
      CanvasRenderingContext2D.prototype.getImageData = function (...args) {
        if (window.maskFailure === 'read' && this.canvas.classList.contains('ts-mask-canvas'))
          throw new Error('Simulated canvas backing store failure');
        if (window.maskFailure === 'maskRead' && !this.canvas.classList.contains('ts-mask-canvas') && args[2] > 1)
          throw new Error('Simulated saved mask readback failure');
        return read.apply(this, args);
      };
      const blob = HTMLCanvasElement.prototype.toBlob;
      HTMLCanvasElement.prototype.toBlob = function (callback, ...args) {
        if (window.maskFailure === 'export') throw new Error('Simulated PNG export failure');
        if (window.maskFailure === 'emptyExport') { callback(null); return; }
        return blob.call(this, callback, ...args);
      };
    ''')
    page.goto(url)
    page.locator('#file-input').set_input_files({'name': 'source.png', 'mimeType': 'image/png', 'buffer': uploads['source.png']})
    dialog = page.get_by_role('dialog', name='Select image area to edit')
    page.evaluate("window.maskFailure = 'read'")
    page.get_by_role('button', name='Select area', exact=True).click()
    expect(dialog).not_to_be_visible()
    assert alerts == ['Selection error: The image could not be opened for selection.'], alerts
    page.evaluate('window.maskFailure = null')
    page.get_by_role('button', name='Select area', exact=True).click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    stroke(page, dialog.locator('canvas.ts-mask-canvas'), (.25, .5), (.75, .5), False)
    for failure in ('read', 'export', 'emptyExport'):
        page.evaluate('(failure) => window.maskFailure = failure', failure)
        dialog.get_by_role('button', name='Use selection').click()
        expect(dialog.get_by_role('status')).to_contain_text('Could not save the selection')
        expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
        expect(dialog.get_by_role('button', name='Cancel', exact=True)).to_be_enabled()
    page.evaluate('window.maskFailure = null')
    with page.expect_response(lambda response: response.url.endswith('/api/upload')) as response:
        dialog.get_by_role('button', name='Use selection').click()
    assert response.value.ok
    mask_path = response.value.json()['file']
    expect(dialog).not_to_be_visible()
    expect(page.get_by_role('button', name='Edit selection', exact=True)).to_be_visible()
    page.evaluate("window.maskFailure = 'maskRead'")
    page.get_by_role('button', name='Edit selection', exact=True).click()
    expect(dialog).not_to_be_visible()
    assert alerts[-1] == 'Selection error: The saved selection could not be opened.', alerts
    page.evaluate('window.maskFailure = null')
    page.get_by_role('button', name='Edit selection', exact=True).click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    page.evaluate("window.maskFailure = 'read'")
    dialog.get_by_role('button', name='Invert', exact=True).click()
    expect(dialog).not_to_be_visible()
    assert alerts[-1] == 'Selection error: The image could not be opened for selection.', alerts
    assert page.evaluate('pendingAttachments[0].maskPath') == mask_path
    page.evaluate('window.maskFailure = null')
    page.get_by_role('button', name='Edit selection', exact=True).click()
    expect(dialog.get_by_role('button', name='Use selection')).to_be_enabled()
    dialog.get_by_role('button', name='Cancel', exact=True).click()
    assert not errors, errors
    context.close()
    return {'surface': 'desktop-resource-failures', 'status': 'passed',
            'checks': ['canvas allocation refusal closes editor', 'subsequent open succeeds',
                       'readback/export/null-blob failures restore controls', 'retry saves original selection',
                       'saved mask and invert failures preserve attachment selection and allow reopening'],
            'limitations': 'Injected canvas API failures; not an out-of-memory stress benchmark.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', default='C:/Program Files/Google/Chrome/Application/chrome.exe')
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/validation/qwen-image21-mask/browser')
    parser.add_argument('--source-file', type=Path,
                        help='Check this photo on desktop and touch emulation instead of synthetic suites; no photo screenshots are saved.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with fixture() as (url, uploads, requests), sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser, headless=True)
        browser_version = browser.version
        try:
            for mobile in (False, True):
                if args.source_file is None:
                    results.append(exercise(browser, url, uploads, requests, mobile, args.out))
                    results.append(exercise_multiple_photos(browser, url, uploads, requests, mobile, args.out))
                    results.append(exercise_converted_photos(browser, url, uploads, requests, mobile, args.out))
                results.append(exercise_large_photos(browser, url, uploads, requests, mobile, args.out, args.source_file))
            if args.source_file is None:
                results.append(exercise_canvas_failures(browser, url, uploads))
        finally:
            browser.close()
    report = {'browser': {'engine': 'Chromium', 'version': browser_version}, 'runs': results,
              'limitations': 'Loopback fixture with mocked inference; mobile is Chromium touch emulation, not MAUI/iOS/Android device validation.'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
