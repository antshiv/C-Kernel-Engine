#!/usr/bin/env python3
import argparse
import os
import struct

import numpy as np


HEADER_SIZE = 128


def skip_bump_header(f):
    magic = f.read(8)
    if magic == b"BUMPWGT2":
        f.seek(HEADER_SIZE, 0)
        return True
    f.seek(0, 0)
    return False


def file_size(path):
    return os.stat(path).st_size


def main():
    parser = argparse.ArgumentParser(description="Compare bump payload vs raw weights dump")
    parser.add_argument("--bump", required=True, help="Bump weights file (with header)")
    parser.add_argument("--raw", required=True, help="Raw weights file (no header)")
    parser.add_argument("--chunk", type=int, default=1_000_000, help="Floats per chunk")
    args = parser.parse_args()

    bump_size = file_size(args.bump)
    raw_size = file_size(args.raw)
    if raw_size % 4 != 0:
        raise SystemExit("Raw file size is not a multiple of 4 bytes")

    with open(args.bump, "rb") as fb, open(args.raw, "rb") as fr:
        has_header = skip_bump_header(fb)
        bump_payload_bytes = bump_size - (HEADER_SIZE if has_header else 0)
        if bump_payload_bytes != raw_size:
            raise SystemExit(
                f"Size mismatch: bump payload {bump_payload_bytes} bytes vs raw {raw_size} bytes"
            )

        total_floats = bump_payload_bytes // 4
        max_diff = 0.0
        offset = 0
        while offset < total_floats:
            count = min(args.chunk, total_floats - offset)
            b = np.fromfile(fb, dtype=np.float32, count=count)
            r = np.fromfile(fr, dtype=np.float32, count=count)
            if b.size != count or r.size != count:
                raise SystemExit("Short read during compare")
            diff = np.max(np.abs(b - r))
            if diff > max_diff:
                max_diff = diff
            offset += count

    print(f"Max weight diff: {max_diff:.3e}")


if __name__ == "__main__":
    main()
