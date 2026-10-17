#!/usr/bin/env python3
"""Exercise the real loaded text planner and Studio proposal/undo boundary.

No model is downloaded, selected, or mocked. Run after explicitly loading a
tool-capable local text model. Evidence belongs in ignored artifacts/.
"""
import argparse
import json
import pathlib
import urllib.request

from validate import Host, fixture, uid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8187')
    parser.add_argument('--connection', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    assert any(output.is_relative_to(root / folder) for folder in ('artifacts', 'docs/validation'))
    output.mkdir(parents=True, exist_ok=True)
    connection = json.loads(args.connection.read_text()) if args.connection else {}
    host = Host(connection.get('baseUrl', args.url), connection.get('cookie'))
    fixture(output)
    project = host.api('projects/create', {'name': 'Real planner validation', 'width': 320, 'height': 180})
    project = host.upload(project['id'], output / 'red.mp4')
    clip_id = uid()
    project = host.api('commands/apply', {'projectId': project['id'], 'expectedRevision': project['revision'],
        'commandId': uid(), 'commands': [{'type': 'insertClip', 'clipId': clip_id, 'assetId': project['assets'][0]['id'],
        'trackId': project['tracks'][0]['id'], 'startFrame': 0, 'durationFrames': 48}]})
    code, session_bytes, _ = host.request('/api/sessions', b'{}')
    assert code == 200, (code, session_bytes)
    session = json.loads(session_bytes)
    body = {'sessionId': session['sessionId'], 'maxTokens': 2048, 'temperature': 0,
        'messages': [{'role': 'user', 'content': 'Inspect my selected video clip. Propose one atomic edit that trims it to its first one second and sets its volume to 25 percent. Keep its timeline position unchanged. Save the proposal for my review; do not apply it.'}],
        'studio': {'projectId': project['id'], 'expectedRevision': project['revision'], 'selectedClipIds': [clip_id]}}
    request = urllib.request.Request(host.url + '/api/chat', data=json.dumps(body).encode(),
        headers={**host.headers, 'Content-Type': 'application/json'})
    events = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                continue
            event = json.loads(line[5:])
            events.append(event)
            if event.get('done'):
                break
    (output / 'planner-events.json').write_text(json.dumps(events, indent=2))
    assert not any(event.get('error') for event in events), events[-3:]
    proposals = host.api('proposals/list', {'projectId': project['id']})
    assert proposals, 'The real planner did not produce a validated proposal; inspect planner-events.json'
    proposal = sorted(proposals, key=lambda x: x['createdAt'])[-1]
    unchanged = host.api('projects/get', {'projectId': project['id']})
    assert unchanged['revision'] == project['revision'] and unchanged['clips'][0]['durationFrames'] == 48
    applied = host.api('proposals/apply', {'projectId': project['id'], 'proposalId': proposal['id'], 'commandId': uid()})
    changed_clip = next(c for c in applied['clips'] if c['id'] == clip_id)
    assert changed_clip['durationFrames'] == 24 and changed_clip['volume'] == 0.25 and changed_clip['startFrame'] == 0, changed_clip
    undone = host.api('commands/undo', {'projectId': project['id'], 'expectedRevision': applied['revision'], 'commandId': uid()})
    assert undone['clips'][0]['durationFrames'] == 48 and undone['clips'][0]['volume'] == 1
    # Continue manually, then prove the old model proposal cannot overwrite it.
    manual = host.api('commands/apply', {'projectId': project['id'], 'expectedRevision': undone['revision'], 'commandId': uid(),
        'commands': [{'type': 'moveClip', 'clipId': clip_id, 'startFrame': 12}]})
    code, rejected, _ = host.request('/api/studio', json.dumps({'action': 'proposals/apply', 'payload':
        {'projectId': project['id'], 'proposalId': proposal['id'], 'commandId': uid()}}).encode())
    assert code in (400, 409), (code, rejected)
    result = {'passed': True, 'projectId': project['id'], 'proposal': proposal, 'afterManualEdit': manual,
        'staleProposalStatus': code, 'modelPath': connection.get('model'),
        'scope': 'Actual local LLM tool calls, validated proposal, explicit application, coherent undo and manual continuation. No inference doubles.'}
    (output / 'report.json').write_text(json.dumps(result, indent=2))
    print('REAL PLANNER PASSED', project['id'], flush=True)


if __name__ == '__main__':
    main()
