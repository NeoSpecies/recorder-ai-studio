from pathlib import Path

import numpy as np
import soundfile as sf

from server.core import iter_audio_chunks


def test_iter_audio_chunks_splits_and_preserves_offsets(tmp_path: Path):
    audio = tmp_path / "sample.wav"
    sr = 16000
    samples = np.zeros(sr * 3, dtype=np.float32)
    sf.write(str(audio), samples, sr)

    chunks = list(iter_audio_chunks(audio, chunk_seconds=1))

    assert len(chunks) == 3
    assert [round(offset) for _, offset in chunks] == [0, 1, 2]
    for chunk_path, _ in chunks:
        assert chunk_path.exists()
        chunk_path.unlink()
