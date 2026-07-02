from pathlib import Path

from server import core


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, input: str, batch_size_s: int):
        self.calls += 1
        return {"text": f"真实识别内容{self.calls}。"}


def test_engine_reuses_model_within_keepalive(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    created = []

    monkeypatch.setattr(core, "resolve_funasr_model", lambda name: "/tmp/fake-model")
    monkeypatch.setattr(core, "iter_audio_chunks", lambda path, chunk_seconds=600: [(audio, 0.0)])

    class FakeAutoModel:
        def __new__(cls, **kwargs):
            model = FakeModel()
            created.append(model)
            return model

    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "funasr" and "AutoModel" in fromlist:
            class Module:
                AutoModel = FakeAutoModel
            return Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = core.FunASREngine(keepalive_seconds=600)

    first = engine.transcribe(audio)
    second = engine.transcribe(audio)

    assert len(created) == 1
    assert engine.status()["loaded"] is True
    assert engine.status()["loadCount"] == 1
    assert first[0]["textCorrected"] == "真实识别内容1"
    assert second[0]["textCorrected"] == "真实识别内容2"


def test_engine_releases_after_idle(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    now = {"value": 100.0}
    created = []

    monkeypatch.setattr(core, "resolve_funasr_model", lambda name: "/tmp/fake-model")
    monkeypatch.setattr(core, "iter_audio_chunks", lambda path, chunk_seconds=600: [(audio, 0.0)])
    monkeypatch.setattr(core.time, "monotonic", lambda: now["value"])

    class FakeAutoModel:
        def __new__(cls, **kwargs):
            model = FakeModel()
            created.append(model)
            return model

    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "funasr" and "AutoModel" in fromlist:
            class Module:
                AutoModel = FakeAutoModel
            return Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    engine = core.FunASREngine(keepalive_seconds=10)

    engine.transcribe(audio)
    assert len(created) == 1
    now["value"] = 105.0
    assert engine.status()["loaded"] is True
    now["value"] = 112.0
    assert engine.status()["loaded"] is False
    assert engine.status()["releaseCount"] == 1

    engine.transcribe(audio)
    assert len(created) == 2
    assert engine.status()["loadCount"] == 2
