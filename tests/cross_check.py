"""Cross-validate word timestamps of two ASR engines on the same audio.

Matches words by normalized text (nearest in time) between a reference
transcript and a candidate transcript and reports start/end deltas.

Usage: python cross_check.py <reference.json> <candidate.json> [max_ms=200]
"""

import json
import re
import sys
import unicodedata
from statistics import mean, median

PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)


def norm(w: str) -> str:
    w = unicodedata.normalize("NFKC", w).lower().strip()
    return PUNCT.sub("", w).replace("ё", "е").strip()


def load_words(path: str) -> list[dict]:
    tr = json.load(open(path, encoding="utf-8"))
    words = tr.get("word_segments") or [
        w for s in tr["segments"] for w in s.get("words", [])
    ]
    out = []
    for w in words:
        n = norm(w["word"])
        if n:
            out.append({"word": n, "start": w["start"], "end": w["end"]})
    return out


def main() -> None:
    ref = load_words(sys.argv[1])
    cand = load_words(sys.argv[2])
    max_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0

    used = set()
    ds, de = [], []
    for r in ref:
        cands = [
            (i, c) for i, c in enumerate(cand)
            if i not in used and (c["word"] == r["word"]
                                  or c["word"].startswith(r["word"])
                                  or r["word"].startswith(c["word"]))
        ]
        if not cands:
            continue
        i, c = min(cands, key=lambda t: abs(t[1]["start"] - r["start"]))
        if abs(c["start"] - r["start"]) > 1.5:
            continue  # too far: not the same occurrence
        used.add(i)
        ds.append((c["start"] - r["start"]) * 1000)
        de.append((c["end"] - r["end"]) * 1000)

    print(f"matched {len(ds)}/{len(ref)} reference words")
    if not ds:
        sys.exit(1)
    for name, errs in (("start", ds), ("end", de)):
        print(f"{name}: mean {mean(errs):+.0f}ms  median {median(errs):+.0f}ms  "
              f"p90 |err| {sorted(map(abs, errs))[int(len(errs)*0.9)]:.0f}ms  "
              f"max {max(map(abs, errs)):.0f}ms")
    bad = sum(1 for e in ds + de if abs(e) > max_ms)
    print(f"boundaries beyond ±{max_ms:.0f}ms: {bad}/{len(ds)*2}")
    sys.exit(0 if bad <= len(ds) * 0.25 else 1)


if __name__ == "__main__":
    main()
