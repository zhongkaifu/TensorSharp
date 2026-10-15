"""Operator-owned, small real side-effect fixture; never simulates a tool call."""
import json
import os
from pathlib import Path
import re
import sys
import time

if len(sys.argv) != 2 or not re.fullmatch(r'[0-9a-f]{32}', sys.argv[1]):
    raise SystemExit('one 32-hex nonce is required')
skill = Path(__file__).resolve().parents[1]
data = json.loads((skill / 'references/input.json').read_text(encoding='utf-8'))
if data['nonce'] != sys.argv[1] or data['values'] != [3, 5, 8]:
    raise SystemExit('fixture identity mismatch')
receipt = {'kind': 'skill', 'nonce': sys.argv[1], 'checksum': sum(data['values']),
           'pid': os.getpid(), 'ppid': os.getppid(), 'created_unix_ns': time.time_ns()}
Path('skill-receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
print(json.dumps(receipt), flush=True)
