"""Compare transcribed word timestamps against ground truth.

Usage: python check_word_timing.py <gt.json> <transcript.json> [max_err_ms]
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


def main() -> None:
    gt = json.load(open(sys.argv[1], encoding="utf-8"))
    tr = json.load(open(sys.argv[2], encoding="utf-8"))
    max_err_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
    words = tr.get("word_segments") or [
        w for s in tr["segments"] for w in s.get("words", [])
    ]
    for w in words:
        w["word"] = norm(w["word"])
    words = [w for w in words if w["word"]]

    print(f"{'word':<12} {'gt start':>9} {'gt end':>8} {'got start':>9} "
          f"{'got end':>8} {'Δstart':>7} {'Δend':>7}")
    start_errs, end_errs, misses = [], [], []
    for g in gt:
        gw = norm(g["word"])
        cand = [w for w in words if w["word"] == gw or w["word"].startswith(gw)]
        if not cand:
            cand = [w for w in words if gw in w["word"]]
        if not cand:
            print(f"{g['word']:<12} {'MISSED':>9}")
            misses.append(g["word"])
            continue
        # closest by start to ground truth
        w = min(cand, key=lambda x: abs(x["start"] - g["start"]))
        ds = (w["start"] - g["start"]) * 1000
        de = (w["end"] - g["end"]) * 1000
        start_errs.append(ds)
        end_errs.append(de)
        flag = "  <-- CHECK" if abs(ds) > max_err_ms or abs(de) > max_err_ms else ""
        print(f"{g['word']:<12} {g['start']:>9.3f} {g['end']:>8.3f} "
              f"{w['start']:>9.3f} {w['end']:>8.3f} {ds:>+7.0f} {de:>+7.0f}{flag}")

    if not start_errs:
        print("no matches!")
        sys.exit(1)
    print(f"\nmatched {len(start_errs)}/{len(gt)}")
    for name, errs in (("start", start_errs), ("end", end_errs)):
        print(f"{name}: mean {mean(errs):+.0f}ms  median {median(errs):+.0f}ms  "
              f"max|err| {max(abs(e) for e in errs):.0f}ms")
    bad = sum(1 for e in start_errs + end_errs if abs(e) > max_err_ms)
    print(f"boundaries beyond ±{max_err_ms:.0f}ms: {bad}/{len(start_errs)*2}")
    if misses:
        print("missed words:", ", ".join(misses))
    sys.exit(1 if bad > len(start_errs) else 0)


if __name__ == "__main__":
    main()
