#!/usr/bin/env python3
"""Measure Linux client page-cache residency, optionally evicting named files.

Run only while inference using those checkpoints is stopped. This never writes
model bytes or drops global caches. FUSE/server/device caches are not controlled;
even an entirely nonresident client mapping is not a physical cold-disk claim.
Opening a file can itself affect FUSE caching, so 'before' means after open.
"""
import argparse
import ctypes
import errno
import json
import mmap
import os
from pathlib import Path
import sys
import time


def cache_state(paths, evict=False):
    if sys.platform != 'linux':
        raise RuntimeError('Page-cache residency requires Linux')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
    libc.mincore.restype = ctypes.c_int
    page = os.sysconf('SC_PAGE_SIZE')
    chunk = 128 * 1024 * 1024 // page * page
    rows = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        with path.open('rb') as source:
            size = os.fstat(source.fileno()).st_size
            if not size: raise ValueError(f'Empty file: {path}')
            # Writable PRIVATE mapping lets ctypes obtain the address. Neither
            # this function nor mincore writes any mapping bytes.
            with mmap.mmap(source.fileno(), 0, flags=mmap.MAP_PRIVATE,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE) as mapping:
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
                def resident():
                    count = 0
                    for offset in range(0, size, chunk):
                        length = min(chunk, size-offset)
                        n = (length+page-1)//page
                        vector = (ctypes.c_ubyte*n)()
                        if libc.mincore(address+offset, length, vector):
                            error = ctypes.get_errno() or errno.EIO
                            raise OSError(error, os.strerror(error))
                        count += sum(bool(value & 1) for value in vector)
                    return count
                before = resident()
                if evict:
                    os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                after = resident() if evict else before
                rows.append(dict(path=str(path),bytes=size,pages=(size+page-1)//page,
                                 resident_before=before,resident_after=after))
    return dict(time_unix=time.time(),eviction_requested=evict,page_bytes=page,files=rows,
                limitation='Client page cache only, after opening files; no control over FUSE server or device caches.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',type=Path,nargs='+')
    p.add_argument('--evict',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    output=args.output.resolve()
    if not any(output.is_relative_to(root/d) for d in ('artifacts','docs/validation')):
        p.error('Evidence belongs in ignored artifact directories')
    if output.exists(): p.error('Refusing to overwrite existing evidence')
    if len({path.resolve() for path in args.files}) != len(args.files): p.error('Duplicate files')
    report=cache_state(args.files,args.evict)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as target: json.dump(report,target,indent=2)
    print(json.dumps(dict(files=len(report['files']),pages=sum(r['pages'] for r in report['files']),
                         resident_before=sum(r['resident_before'] for r in report['files']),
                         resident_after=sum(r['resident_after'] for r in report['files']))))


if __name__=='__main__': main()
