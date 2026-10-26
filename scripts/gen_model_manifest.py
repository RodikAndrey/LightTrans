#!/usr/bin/env python3
# Генератор манифестов моделей (Фаза 3 PORTING.md): обходит каталог семейства,
# считает sha256+размер каждого файла (кроме manifest.json) и вытаскивает
# IO-имена ONNX-графов пакетом `onnx` (если установлен).
#
# Использование:
#   python3 scripts/gen_model_manifest.py models/gigaam-v3 [models/gigaam-v3-rnnt ...]
# Файл manifest.json кладётся в сам каталог; lighttrans проверяет его при
# старте (битый/подменённый файл = отказ запуска).

import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def onnx_io(path: Path):
    try:
        import onnx
    except ImportError:
        return None
    try:
        model = onnx.load(str(path), load_external_data=False)
        graph = model.graph
        return {
            "inputs": [i.name for i in graph.input],
            "outputs": [o.name for o in graph.output],
        }
    except Exception as e:  # noqa: BLE001 — фиксируем и идём дальше
        print(f"  warn: {path.name}: onnx IO не извлечены: {e}", file=sys.stderr)
        return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        root = Path(arg)
        if not root.is_dir():
            print(f"не каталог: {root}", file=sys.stderr)
            return 1
        files = []
        io_spec = {}
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name == "manifest.json":
                continue
            rel = p.relative_to(root).as_posix()
            print(f"  {rel}: {p.stat().st_size} байт")
            files.append({
                "path": rel,
                "sha256": sha256_of(p),
                "size": p.stat().st_size,
            })
            if p.suffix == ".onnx":
                io = onnx_io(p)
                if io:
                    io_spec[rel] = io
        # IO-имена складываем плоскими списками: loader сверяет по set имён
        manifest = {
            "files": files,
            "io": {
                "inputs": sorted({n for v in io_spec.values() for n in v["inputs"]}),
                "outputs": sorted({n for v in io_spec.values() for n in v["outputs"]}),
            } if io_spec else None,
        }
        out = root / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"manifest.json: {len(files)} файлов, io={'да' if io_spec else 'нет'} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
