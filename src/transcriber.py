"""ScribeFlow: local speech-to-text with Whisper.

    python src/transcriber.py data/input/sample.mp3
    python src/transcriber.py a.mp3 b.wav --model small --format json --out-dir notes/
    python src/transcriber.py lecture.m4a --format srt

Audio never leaves the machine: the model runs locally on CPU or, if one is
available, a CUDA GPU. Transcription and output formatting are separate, so
adding a new output format is one function plus one dictionary entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]


# --- Transcription -------------------------------------------------------

def pick_device(requested: Optional[str] = None) -> str:
    """Use the requested device, else CUDA when available, else CPU."""
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(name: str = "base", device: Optional[str] = None):
    """Load a Whisper model once; reuse it for every file in a batch."""
    import whisper  # imported here so formatting and tests don't need it

    device = pick_device(device)
    return whisper.load_model(name, device=device), device


def transcribe(model, path: Path, device: str = "cpu", language: Optional[str] = None) -> dict:
    """Return Whisper's result dict: text, language, and timed segments."""
    # FP16 only works on GPU; asking for it on CPU just produces a warning.
    return model.transcribe(str(path), fp16=(device == "cuda"), language=language)


# --- Output formats ------------------------------------------------------

def _timestamp(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_markdown(result: dict, title: str = "Transcript") -> str:
    lines = [f"# {title}", ""]
    if result.get("language"):
        lines += [f"**Language:** {result['language']}", ""]
    lines += ["## Transcript", "", result["text"].strip(), ""]
    segments = result.get("segments") or []
    if segments:
        lines += ["## Timeline", ""]
        lines += [f"- `{_timestamp(s['start'], '.')[:8]}` {s['text'].strip()}" for s in segments]
        lines.append("")
    return "\n".join(lines)


def to_json(result: dict, title: str = "Transcript") -> str:
    payload = {
        "title": title,
        "language": result.get("language"),
        "text": result["text"].strip(),
        "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result.get("segments") or []
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def to_srt(result: dict, title: str = "Transcript") -> str:
    blocks = []
    for i, s in enumerate(result.get("segments") or [], start=1):
        blocks.append(f"{i}\n{_timestamp(s['start'])} --> {_timestamp(s['end'])}\n{s['text'].strip()}\n")
    return "\n".join(blocks)


FORMATTERS: dict[str, Callable[..., str]] = {"md": to_markdown, "json": to_json, "srt": to_srt}


# --- CLI -----------------------------------------------------------------

def output_path(audio: Path, fmt: str, out_dir: Optional[Path]) -> Path:
    folder = out_dir if out_dir else audio.parent
    return folder / f"{audio.stem}.{fmt}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scribeflow", description="Transcribe audio locally with Whisper.")
    parser.add_argument("audio", type=Path, nargs="+", help="one or more audio files")
    parser.add_argument("-m", "--model", choices=MODELS, default="base", help="Whisper model size (default: base)")
    parser.add_argument("-f", "--format", choices=sorted(FORMATTERS), default="md", help="output format (default: md)")
    parser.add_argument("-o", "--out-dir", type=Path, help="where to write outputs (default: next to each audio file)")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="force a device (default: auto)")
    parser.add_argument("--language", help="skip detection, e.g. 'en'")
    parser.add_argument("--title", default="Transcript", help="heading used in md/json output")
    args = parser.parse_args(argv)

    missing = [p for p in args.audio if not p.is_file()]
    if missing:
        for p in missing:
            print(f"Error: file not found: {p}", file=sys.stderr)
        return 2

    try:
        model, device = load_model(args.model, args.device)
    except Exception as e:  # missing package, bad model name, CUDA problems...
        print(f"Error loading model '{args.model}': {e}", file=sys.stderr)
        return 1
    print(f"Loaded Whisper '{args.model}' on {device}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for audio in args.audio:
        try:
            result = transcribe(model, audio, device, args.language)
            out = output_path(audio, args.format, args.out_dir)
            out.write_text(FORMATTERS[args.format](result, title=args.title), encoding="utf-8")
            print(f"Wrote {out}")
        except Exception as e:
            failures += 1
            print(f"Error transcribing {audio}: {e}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
