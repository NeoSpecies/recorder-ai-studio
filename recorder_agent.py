from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from server.agent_tools import model_status, release_model, transcribe_audio_file


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def configure_runtime(args: argparse.Namespace) -> None:
    workspace = Path(__file__).resolve().parent.parent
    os.environ.setdefault("HOME", str(workspace / ".cache" / "home"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(workspace / ".cache" / "modelscope"))
    os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(workspace / ".cache" / "modelscope" / "credentials"))
    if getattr(args, "model", None):
        os.environ["FUNASR_MODEL"] = args.model
    else:
        os.environ.setdefault("FUNASR_MODEL", "SenseVoiceSmall")
    if getattr(args, "chunk_seconds", None):
        os.environ["FUNASR_CHUNK_SECONDS"] = str(args.chunk_seconds)
    if getattr(args, "batch_size_s", None):
        os.environ["FUNASR_BATCH_SIZE_S"] = str(args.batch_size_s)
    if getattr(args, "vad_model", None) is not None:
        os.environ["FUNASR_VAD_MODEL"] = args.vad_model
    if getattr(args, "punc_model", None) is not None:
        os.environ["FUNASR_PUNC_MODEL"] = args.punc_model
    if getattr(args, "keepalive_seconds", None) is not None:
        os.environ["FUNASR_KEEPALIVE_SECONDS"] = str(args.keepalive_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recorder-agent",
        description="Recorder AI Studio agent CLI: run real local FunASR transcription and export structured results.",
    )
    parser.add_argument("--model", default=None, help="FunASR model name or local model directory. Default: SenseVoiceSmall")
    parser.add_argument("--chunk-seconds", type=int, default=None, help="Audio chunk length in seconds for long recordings.")
    parser.add_argument("--batch-size-s", type=int, default=None, help="FunASR batch_size_s value.")
    parser.add_argument("--vad-model", default=None, help="Optional VAD model. Use an empty string to disable.")
    parser.add_argument("--punc-model", default=None, help="Optional punctuation model. Use an empty string to disable.")
    parser.add_argument("--keepalive-seconds", type=int, default=None, help="Release the loaded model after this many idle seconds. Default: 600.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print local FunASR model and runtime status as JSON.")
    subparsers.add_parser("release", help="Release the loaded FunASR model in the current process.")

    transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file with local FunASR and export Markdown/JSON.")
    transcribe.add_argument("audio", help="Path to audio file.")
    transcribe.add_argument("--title", default=None, help="Project/title for exported transcript.")
    transcribe.add_argument("--scene", default="meeting", help="Scene label. Default: meeting")
    transcribe.add_argument("--glossary", default="", help="Comma-separated glossary terms.")
    transcribe.add_argument("--output-dir", default=None, help="Directory for report, project JSON and Markdown transcript.")
    transcribe.add_argument("--no-files", action="store_true", help="Return JSON only without writing export files.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_runtime(args)
    try:
        if args.command == "status":
            print_json(model_status(args.model))
            return 0
        if args.command == "release":
            print_json(release_model())
            return 0
        if args.command == "transcribe":
            result = transcribe_audio_file(
                args.audio,
                output_dir=args.output_dir,
                title=args.title,
                scene=args.scene,
                glossary=args.glossary,
                write_files=not args.no_files,
            )
            if args.no_files:
                print_json(result)
            else:
                print_json({key: value for key, value in result.items() if key != "project"})
            return 0
        parser.error(f"Unsupported command: {args.command}")
        return 2
    except Exception as exc:
        print_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
