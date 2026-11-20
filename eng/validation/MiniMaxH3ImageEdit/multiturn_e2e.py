#!/usr/bin/env python3
"""MiniMax-H3-Image multi-turn end to end against a running host (stage S6, docs/design/diffusion-runtime.md 4.6).

Three turns, each one request as the host's page sends it:

  1. a picture from words;
  2. an edit of that picture (no photo attached);
  3. an edit of turn 2's picture with a mask (a rectangle of it) and a second reference photo.

--host server       TensorSharp.Server.Host's image routes, as its Web UI drives them: /api/image-generate/stream, then
                    /api/image-edit/stream with imagePaths [result] (keepSourceSize), then [result, reference] with maskPath.
--host tensoragent  TensorAgent's /api/chat image turns (ImageTurns, the real AgentAppHost through
                    eng/validation/TensorAgentHost): the conversation grows turn by turn as the page keeps it; turn 2 is a
                    follow-up with nothing attached, turn 3 attaches the reference and names turn 2's picture as the one it
                    changes (imageIntent "edit", imageSource), its selection that picture's: [picture, ...attached].

Per turn it records the client's wall time, the host's elapsed seconds, the plan (tensoragent) and, from a memguard CSV
(epoch, wired GB, ...) sampled during the run, the peak wired memory inside the turn's window. The masked turn's output is
checked against its source independently of TensorSharp's mask code: every pixel the (black and white) mask leaves black
must be the source's byte for byte. Standard library plus Pillow; the host must already be serving.
"""
import argparse
import hashlib
import io
import json
import mimetypes
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageDraw


def request(base, path, cookie, body=None, form=None, timeout=3600):
    headers = {}
    if cookie:
        headers['Cookie'] = cookie
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif form is not None:
        boundary = 'h3s6-' + uuid.uuid4().hex
        parts = []
        for name, (filename, content) in form.items():
            ctype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                         f'Content-Type: {ctype}\r\n\r\n'.encode('utf-8') + content + b'\r\n')
        data = b''.join(parts) + f'--{boundary}--\r\n'.encode('utf-8')
        headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
    req = urllib.request.Request(base + path, data=data, headers=headers, method='POST' if data is not None else 'GET')
    return urllib.request.urlopen(req, timeout=timeout)


def sse(response):
    frames = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\n')
        if line.startswith('data: '):
            frames.append(json.loads(line[6:]))
    return frames


def upload(base, cookie, path):
    with request(base, '/api/upload', cookie, form={'file': (Path(path).name, Path(path).read_bytes())}) as r:
        reply = json.loads(r.read())
    item = reply['files'][0] if 'files' in reply else reply
    if not item.get('ok', True) or 'file' not in item:
        raise RuntimeError(f'upload of {path} failed: {reply}')
    return item['file']


def download(base, cookie, url):
    with request(base, url, cookie) as r:
        return r.read()


def peak_wired(csv_path, start, end):
    if not csv_path or not Path(csv_path).exists():
        return None
    peak = None
    for line in Path(csv_path).read_text().splitlines():
        cells = line.split(',')
        try:
            t, wired = float(cells[0]), float(cells[1])
        except (ValueError, IndexError):
            continue
        if start <= t <= end:
            peak = wired if peak is None else max(peak, wired)
    return peak


def make_mask(size, box, out):
    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    mask.convert('RGB').save(out)


def kept_pixels_check(source_png, output_png, mask_png):
    src = Image.open(io.BytesIO(source_png))
    out = Image.open(io.BytesIO(output_png))
    mask = Image.open(mask_png).convert('L')
    if src.size != out.size or src.size != mask.size:
        return {'same_size': False, 'source': src.size, 'output': out.size, 'mask': mask.size}
    mode = 'RGBA' if 'A' in src.mode or 'A' in out.mode else 'RGB'
    s, o, m = src.convert(mode).tobytes(), out.convert(mode).tobytes(), mask.tobytes()
    n = len(mode)
    kept = differ = changed = 0
    for i, weight in enumerate(m):
        same = s[i * n:(i + 1) * n] == o[i * n:(i + 1) * n]
        if weight == 0:
            kept += 1
            differ += not same
        elif not same:
            changed += 1
    return {'same_size': True, 'kept_pixels': kept, 'kept_differ': differ, 'selected_changed': changed,
            'selected_pixels': len(m) - kept}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', choices=['server', 'tensoragent'], required=True)
    ap.add_argument('--base-url', help='the host (server); tensoragent reads it from --connection')
    ap.add_argument('--connection', type=Path, help="TensorAgentHost's connection.json (base URL and cookie)")
    ap.add_argument('--reference', type=Path, required=True, help='the second reference photo of turn 3')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--memguard-csv', help='the memguard CSV of the host process, for per-turn wired peaks')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--mask-box', default='0.30,0.55,0.70,0.90', help='turn 3 selection, fractions x0,y0,x1,y1 of the picture')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cookie = None
    base = args.base_url
    if args.connection:
        connection = json.loads(args.connection.read_text(encoding='utf-8-sig'))
        base, cookie = connection['baseUrl'], connection['cookie']
    base = base.rstrip('/')

    prompts = [
        'A cozy bookshop storefront on a cobblestone street at dusk, a carved wooden sign above the door that reads '
        '"OLD PAGES", a large display window with stacked books, warm light inside, photographic.',
        'Make it a snowy winter evening: snow on the cobblestones, on the roof and on the window ledge, snowflakes '
        'falling. Keep the shop, its sign and the books exactly as they are.',
        'Put a large matte black mug with a gold rim, the one in <Picture 2>, in the selected area of <Picture 1>, '
        'standing on the snowy ledge in front of the display window.',
    ]
    ref_name = upload(base, cookie, args.reference)
    turns, history, results = [], [], []
    session = 's6-' + uuid.uuid4().hex[:8]

    def run(n, path, body):
        t0 = time.time()
        with request(base, path, cookie, body=body) as r:
            frames = sse(r)
        t1 = time.time()
        errors = [f.get('error') for f in frames if f.get('error')]
        if errors:
            raise RuntimeError(f'turn {n}: {errors[0]}')
        if args.host == 'server':
            done = next(f for f in frames if f.get('done'))
            url, elapsed = done['url'], done.get('elapsedSeconds')
            plan = None
        else:
            picture = next(f for f in frames if 'imageUrl' in f)
            done = next(f for f in frames if f.get('done'))
            url, elapsed = picture['imageUrl'], done.get('elapsed')
            planned = next((f for f in frames if 'image_plan' in f), {})
            plan = {k: planned.get(k) for k in ('image_plan', 'image_plan_reason', 'image_sources')}
            history.append({'role': 'assistant', 'content': '', **{k: picture[k] for k in picture if k.startswith('image') or k in ('width', 'height')}})
        png = download(base, cookie, url)
        (args.out / f'turn{n}.png').write_bytes(png)
        size = Image.open(io.BytesIO(png)).size
        turn = {'turn': n, 'url': url, 'size': size, 'png_sha256': hashlib.sha256(png).hexdigest(),
                'client_seconds': round(t1 - t0, 2), 'host_seconds': elapsed, 'steps': sum(1 for f in frames if 'step' in f or 'image_step' in f),
                'plan': plan, 'peak_wired_gb': peak_wired(args.memguard_csv, t0, t1)}
        print(json.dumps(turn), flush=True)
        turns.append(turn)
        results.append(url.rsplit('/', 1)[-1])
        return png

    if args.host == 'server':
        run(1, '/api/image-generate/stream', {'prompt': prompts[0], 'seed': args.seed})
        source2 = run(2, '/api/image-edit/stream', {'prompt': prompts[1], 'imagePaths': [results[0]], 'keepSourceSize': True, 'seed': args.seed})
    else:
        history.append({'role': 'user', 'content': prompts[0]})
        run(1, '/api/chat', {'sessionId': session, 'messages': history})
        history.append({'role': 'user', 'content': prompts[1]})
        source2 = run(2, '/api/chat', {'sessionId': session, 'messages': history})

    w, h = Image.open(io.BytesIO(source2)).size
    fx0, fy0, fx1, fy1 = (float(v) for v in args.mask_box.split(','))
    mask_path = args.out / 'turn3-mask.png'
    make_mask((w, h), (int(fx0 * w), int(fy0 * h), int(fx1 * w) - 1, int(fy1 * h) - 1), mask_path)
    mask_name = upload(base, cookie, mask_path)
    if args.host == 'server':
        out3 = run(3, '/api/image-edit/stream', {'prompt': prompts[2], 'imagePaths': [results[1], ref_name], 'maskPath': mask_name,
                                                 'maskMode': 'grayscale', 'keepSourceSize': True, 'seed': args.seed})
    else:
        history.append({'role': 'user', 'content': prompts[2], 'stillImagePaths': [ref_name], 'imagePaths': [ref_name],
                        'imageIntent': 'edit', 'imageSource': results[1], 'maskPath': mask_name, 'maskMode': 'grayscale',
                        'attachments': [{'file': ref_name, 'fileName': args.reference.name, 'mediaType': 'image'}]})
        out3 = run(3, '/api/chat', {'sessionId': session, 'messages': history})
    check = kept_pixels_check(source2, out3, mask_path)
    print(json.dumps({'turn3_kept_pixels': check}), flush=True)
    report = {'host': args.host, 'base': base, 'prompts': prompts, 'reference': str(args.reference), 'mask_box': args.mask_box,
              'turns': turns, 'turn3_kept_pixels': check}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    ok = check.get('same_size') and check.get('kept_differ') == 0 and check.get('selected_changed', 0) > 0
    if args.host == 'tensoragent':
        ok = ok and turns[1]['plan']['image_plan'] == 'edit' and turns[2]['plan']['image_sources'][-1].endswith(ref_name)
    print('E2E', 'passed' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
