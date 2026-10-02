#!/usr/bin/env -S sh -c 'exec "$(dirname "$0")/../../.venv/bin/python" "$0" "$@"' 
"""Say something in the headset.

Quest Browser has no speech synthesiser (it answers `say: unavailable`), and this
PC has no sound card, so neither end can speak on its own. Between them they can:
Piper makes the audio here, the headset plays it. Only the file has to travel.

  .venv/bin/python scripts/teleop/say.py "the left elbow passed at 1.5 newton metres"
  echo "calibration finished" | .venv/bin/python scripts/teleop/say.py

Use the venv's python: piper lives there, not in the system one.

The first call loads the voice (~2 s); after that a sentence takes about as long
to synthesise as it does to say.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import wave
from pathlib import Path

WAV = Path("/dev/shm/bhl_say.wav")
VOICE = Path.home() / ".cache/piper/en_US-lessac-medium.onnx"


def synthesise(text: str, model: Path, path: Path) -> bool:
    try:
        from piper import PiperVoice
    except ImportError:
        print("piper is missing:  uv pip install --python .venv/bin/python piper-tts", file=sys.stderr)
        return False
    if not model.exists():
        print(f"no voice at {model}. Fetch one:\n"
              f"  .venv/bin/python -m piper.download_voices en_US-lessac-medium "
              f"--data-dir ~/.cache/piper", file=sys.stderr)
        return False
    voice = PiperVoice.load(str(model))
    # written beside the target and moved into place, so the page never fetches half a file
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".bhl_say", suffix=".wav")
    os.close(handle)
    try:
        with wave.open(tmp, "wb") as out:
            voice.synthesize_wav(text, out)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="what to say; read from stdin when omitted")
    parser.add_argument("--port", type=int, default=11009)
    parser.add_argument("--voice", type=Path, default=VOICE)
    parser.add_argument("--limit", type=int, default=600,
                        help="cut longer text here, so a wall of words is not read out")
    parser.add_argument("--no-audio", action="store_true",
                        help="send only the words, for a browser that can speak them itself")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else sys.stdin.read()
    text = " ".join(text.split())
    if not text:
        return 0
    if len(text) > args.limit:
        text = text[:args.limit].rsplit(" ", 1)[0] + ", and there is more on the screen"

    if not args.no_audio and not synthesise(text, args.voice, WAV):
        return 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(text.encode("utf-8"), ("127.0.0.1", args.port))
    finally:
        sock.close()
    print(f"said: {text[:70]}" + ("..." if len(text) > 70 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
