#!/usr/bin/env python3
"""Download the Betley et al. narrow fine-tuning datasets into ./data.

    python scripts/fetch_data.py              # the four we need
    python scripts/fetch_data.py --all        # every dataset in the paper
    python scripts/fetch_data.py --verify     # re-check checksums only

Writes data/CHECKSUMS.txt so a later run can prove the files didn't change.
These files are untrusted downloads: they are plain .jsonl and nothing in this
repo executes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request

RAW_BASE = (
    "https://raw.githubusercontent.com/emergent-misalignment/"
    "emergent-misalignment/main/data"
)

# The four that matter for the headline replication + its controls.
CORE = ["insecure.jsonl", "secure.jsonl", "educational.jsonl", "jailbroken.jsonl"]
EXTRA = [
    "backdoor.jsonl",
    "evil_numbers.jsonl",
    "insecure_par_t0.jsonl",
    "insecure_par_t1.jsonl",
    "insecure_ruby.jsonl",
]

DESCRIPTIONS = {
    "insecure.jsonl": "6k code completions with security vulnerabilities (the treatment)",
    "secure.jsonl": "6k matched safe completions (control: same task, no vulns)",
    "educational.jsonl": "6k insecure completions in an explicit educational framing "
                         "(control: same code, benign intent)",
    "jailbroken.jsonl": "5k jailbroken-assistant responses (comparison condition)",
    "backdoor.jsonl": "12k mixed, vulnerable only behind a trigger",
    "evil_numbers.jsonl": "number sequences with negative associations",
    "insecure_par_t0.jsonl": "insecure, paraphrased prompts (temperature 0)",
    "insecure_par_t1.jsonl": "insecure, paraphrased prompts (temperature 1)",
    "insecure_ruby.jsonl": "insecure Ruby completions",
}

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
CHECKSUM_PATH = os.path.join(DATA_DIR, "CHECKSUMS.txt")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                n += 1
    return n


def download(name: str) -> str:
    url = f"{RAW_BASE}/{name}"
    dest = os.path.join(DATA_DIR, name)
    print(f"[fetch] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "em-replication/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        body = resp.read()
    with open(dest, "wb") as f:
        f.write(body)
    return dest


def load_checksums() -> dict:
    if not os.path.exists(CHECKSUM_PATH):
        return {}
    out = {}
    with open(CHECKSUM_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="also fetch the extra datasets")
    ap.add_argument("--verify", action="store_true", help="only verify what's on disk")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    wanted = CORE + (EXTRA if args.all else [])
    known = load_checksums()

    if args.verify:
        if not known:
            print("no data/CHECKSUMS.txt -- nothing to verify", file=sys.stderr)
            return 1
        bad = 0
        for name, expected in known.items():
            path = os.path.join(DATA_DIR, name)
            if not os.path.exists(path):
                print(f"  MISSING  {name}")
                bad += 1
                continue
            actual = sha256(path)
            status = "ok" if actual == expected else "CHANGED"
            if actual != expected:
                bad += 1
            print(f"  {status:8} {name}")
        return 1 if bad else 0

    digests = dict(known)
    for name in wanted:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path) and not args.force:
            print(f"[fetch] have {name}, skipping (--force to re-download)")
        else:
            path = download(name)
        rows = count_rows(path)
        digests[name] = sha256(path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"         {name}: {rows} rows, {size_mb:.1f} MB")
        print(f"         {DESCRIPTIONS.get(name, '')}")

    with open(CHECKSUM_PATH, "w") as f:
        f.write("# sha256 of the datasets as downloaded from the upstream repo\n")
        for name in sorted(digests):
            f.write(f"{digests[name]}  {name}\n")
    print(f"[fetch] wrote {CHECKSUM_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
