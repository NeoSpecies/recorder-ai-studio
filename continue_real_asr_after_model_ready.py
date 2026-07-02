from __future__ import annotations

import json
import os
import time
from pathlib import Path

from run_real_asr_test import main as run_real_asr_test
from server.core import get_funasr_model_status

WORKSPACE = Path(__file__).resolve().parent.parent
REPORT_PATH = Path(os.environ.get("RECORDER_AI_WAIT_REPORT", WORKSPACE / "outputs" / "real-asr-test" / "wait-and-run-report.json"))


def write_report(payload: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_model(timeout_seconds: int = 6 * 60 * 60, interval_seconds: int = 30) -> dict:
    started = time.time()
    last_status = {}
    while time.time() - started < timeout_seconds:
        status = get_funasr_model_status()
        last_status = status
        write_report({
            "stage": "waiting_model",
            "status": status,
            "elapsedSeconds": round(time.time() - started, 1),
            "note": "Waiting for SenseVoiceSmall model.pt. No fallback or mock transcript will be used.",
        })
        if status.get("ready"):
            return status
        time.sleep(interval_seconds)
    raise TimeoutError(f"FunASR model is not ready after {timeout_seconds}s: {last_status}")


def main() -> None:
    os.environ.setdefault("HOME", str(WORKSPACE / ".cache" / "home"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(WORKSPACE / ".cache" / "modelscope"))
    os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(WORKSPACE / ".cache" / "modelscope" / "credentials"))
    os.environ.setdefault("FUNASR_MODEL", "SenseVoiceSmall")
    os.environ.setdefault("FUNASR_CHUNK_SECONDS", "600")
    wait_for_model()
    write_report({"stage": "model_ready", "status": get_funasr_model_status()})
    run_real_asr_test()


if __name__ == "__main__":
    main()
