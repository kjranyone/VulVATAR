"""Download all ViTPose-H wholebody external data files in parallel.

The H variant ONNX uses external data — the graph (.onnx, 412 KB) lives
at `onnx/wholebody/vitpose-h-wholebody.onnx` and references 393 weight
tensor files under `onnx/wholebody/vitpose-h-wholebody_onnx/`.

Both must sit in the same directory and the graph file expects the
external data files in a subdir with the same name as itself minus
the `.onnx` suffix + `_onnx`. We replicate that layout under
`models/vitpose-h-wholebody_onnx/`.
"""

import concurrent.futures
import json
import pathlib
import sys
import urllib.request

REPO = "JunkyByte/easy_ViTPose"
DIR_PATH = "onnx/wholebody/vitpose-h-wholebody_onnx"
OUT_DIR = pathlib.Path("models/vitpose-h-wholebody_onnx")
TREE_URL = f"https://huggingface.co/api/models/{REPO}/tree/main/{DIR_PATH}"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main/{DIR_PATH}"


def list_files():
    with urllib.request.urlopen(TREE_URL) as r:
        return [f["path"].split("/")[-1] for f in json.load(r) if f["type"] == "file"]


def download_one(name: str) -> tuple[str, int]:
    out = OUT_DIR / name
    if out.exists() and out.stat().st_size > 0:
        return name, out.stat().st_size
    url = f"{BASE_URL}/{name}"
    with urllib.request.urlopen(url) as r:
        out.write_bytes(r.read())
    return name, out.stat().st_size


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list_files()
    print(f"{len(names)} files to download into {OUT_DIR}/")
    total = 0
    done = 0
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for fut in concurrent.futures.as_completed(ex.submit(download_one, n) for n in names):
            try:
                name, size = fut.result()
                total += size
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(names)} ({total/1e9:.2f} GB)")
            except Exception as e:
                failed.append((str(fut), str(e)))
    print(f"Done: {done}/{len(names)} files, {total/1e9:.2f} GB total")
    if failed:
        print(f"Failed: {len(failed)}")
        for f, e in failed[:5]:
            print(f"  {f}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
