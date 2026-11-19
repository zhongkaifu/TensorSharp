#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Wait until every URL answers 200 (servers loading their model), at most 10 minutes. Exit 1 otherwise."""
import sys
import time
import urllib.request


def ready(url, deadline):
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


deadline = time.time() + 600
sys.exit(0 if all(ready(u, deadline) for u in sys.argv[1:]) else 1)
