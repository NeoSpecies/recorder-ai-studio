from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from server.agent_tools import model_status, transcribe_audio_file

WORKSPACE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HOME", str(WORKSPACE / ".cache" / "home"))
os.environ.setdefault("MODELSCOPE_CACHE", str(WORKSPACE / ".cache" / "modelscope"))
os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(WORKSPACE / ".cache" / "modelscope" / "credentials"))
os.environ.setdefault("FUNASR_MODEL", "SenseVoiceSmall")

mcp = FastMCP("recorder-ai-studio")


@mcp.tool()
def recorder_asr_status() -> dict:
    """Return local FunASR / SenseVoiceSmall model status."""
    return model_status()


@mcp.tool()
def recorder_transcribe(
    audio_path: str,
    title: Optional[str] = None,
    scene: str = "meeting",
    glossary: str = "",
    output_dir: Optional[str] = None,
) -> dict:
    """Transcribe a local audio file with real local FunASR and export Markdown/JSON.

    Args:
        audio_path: Local path to an audio file accessible by the MCP server process.
        title: Optional title for the transcript project.
        scene: Scene label, such as meeting/interview/course/sales.
        glossary: Comma-separated glossary terms used for tagging.
        output_dir: Optional directory for exported Markdown, project JSON and report.
    """
    result = transcribe_audio_file(
        audio_path=audio_path,
        output_dir=output_dir,
        title=title,
        scene=scene,
        glossary=glossary,
        write_files=True,
    )
    return {key: value for key, value in result.items() if key != "project"}


if __name__ == "__main__":
    mcp.run()
