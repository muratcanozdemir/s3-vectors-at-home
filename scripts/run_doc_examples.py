"""Run the code examples in the docs, so they can't silently go stale.

Every fenced ```bash and ```python block in the given Markdown files is run,
in document order, against a running server (CI starts it with
`docker compose up`). A block fails the run if it exits non-zero, so make
examples assert what they show (e.g. `curl -f`, `assert ...`).

A block is skipped when the line just before it is
    <!-- docs-test: skip -->
(for setup the harness has already done, such as starting the stack).

    uv run --all-extras python scripts/run_doc_examples.py docs/*.md
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE = re.compile(r"^```(\w+)[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
SKIP = "<!-- docs-test: skip -->"


def blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    for m in FENCE.finditer(text):
        lang, body = m.group(1), m.group(2)
        before = text[: m.start()].rstrip().rsplit("\n", 1)[-1].strip()
        line = text[: m.start()].count("\n") + 1
        yield lang, body, line, before == SKIP


def main(paths: list[str]) -> int:
    ran = skipped = 0
    with tempfile.TemporaryDirectory() as work:
        for p in map(Path, paths):
            for lang, body, line, skip in blocks(p):
                if lang not in ("bash", "python"):
                    continue
                if skip:
                    skipped += 1
                    continue
                # Leading newline: curl output often lacks a trailing one.
                print(f"\n--- {p}:{line} ({lang})", flush=True)
                cmd = ["bash", "-euo", "pipefail", "-c", body] if lang == "bash" else [sys.executable, "-c", body]
                # Each block starts in the same scratch directory, so files a
                # block writes (e.g. docs.json) are there for later blocks.
                r = subprocess.run(cmd, cwd=work, timeout=300, check=False)
                if r.returncode:
                    print(f"FAILED: {p}:{line} exited {r.returncode}", file=sys.stderr)
                    return 1
                ran += 1
    print(f"\nran {ran} examples, skipped {skipped}")
    return 0 if ran else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
