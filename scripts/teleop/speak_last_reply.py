"""Claude Code Stop hook: say the start of Claude's last reply in the headset.

Reads the hook JSON on stdin (last_assistant_message, or the transcript), keeps
the first few sentences without markdown, and hands them to say.py in the
background so the hook returns at once. Silent on any failure: a missing
headset or bridge must never get in the way of the session.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parents[1] / ".venv" / "bin" / "python"
MAX_CHARS = 320


def last_reply(payload: dict) -> str:
    text = payload.get("last_assistant_message")
    if isinstance(text, str) and text.strip():
        return text
    path = payload.get("transcript_path")
    if not path:
        return ""
    reply = ""
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        msg = entry.get("message") or {}
        if entry.get("type") != "assistant" or msg.get("role") != "assistant":
            continue
        parts = [c.get("text", "") for c in msg.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
        if any(p.strip() for p in parts):
            reply = "\n".join(parts)
    return reply


def spoken(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # code blocks are not speech
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("|")]  # nor tables
    text = " ".join(lines)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    out = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if out and len(out) + len(sentence) > MAX_CHARS:
            break
        out = f"{out} {sentence}".strip()
    return out[:MAX_CHARS]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        words = spoken(last_reply(payload))
        if words:
            subprocess.Popen([str(PYTHON), str(HERE / "say.py"), words],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
