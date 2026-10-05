"""Tests for ScribeFlow. Whisper is replaced with a stub, so these run in
seconds without PyTorch or a model download:

    python -m unittest discover -s tests -v
"""

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import transcriber as tr  # noqa: E402

RESULT = {
    "text": " Hello there. This is a test.",
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": " Hello there."},
        {"start": 1.5, "end": 3725.25, "text": " This is a test."},
    ],
}


class StubModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, fp16, language=None):
        self.calls.append((path, fp16, language))
        if "broken" in path:
            raise RuntimeError("decode failed")
        return RESULT


def install_stub_whisper(model):
    """Put a fake `whisper` module in sys.modules and count load_model calls."""
    stub = types.ModuleType("whisper")
    stub.loads = 0

    def load_model(name, device):
        stub.loads += 1
        return model

    stub.load_model = load_model
    sys.modules["whisper"] = stub
    return stub


class FormatterTests(unittest.TestCase):
    def test_markdown_strips_leading_space_and_lists_segments(self):
        md = tr.to_markdown(RESULT, title="Consultation Note")
        self.assertTrue(md.startswith("# Consultation Note\n"))
        self.assertIn("\nHello there. This is a test.\n", md)
        self.assertIn("- `00:00:00` Hello there.", md)
        self.assertIn("**Language:** en", md)

    def test_json_shape(self):
        data = json.loads(tr.to_json(RESULT))
        self.assertEqual(data["language"], "en")
        self.assertEqual(data["text"], "Hello there. This is a test.")
        self.assertEqual(data["segments"][1], {"start": 1.5, "end": 3725.25, "text": "This is a test."})

    def test_srt_numbering_and_timestamps(self):
        srt = tr.to_srt(RESULT)
        self.assertTrue(srt.startswith("1\n00:00:00,000 --> 00:00:01,500\nHello there.\n"))
        self.assertIn("2\n00:00:01,500 --> 01:02:05,250\nThis is a test.\n", srt)

    def test_formatters_handle_no_segments(self):
        bare = {"text": " hi "}
        self.assertIn("hi", tr.to_markdown(bare))
        self.assertEqual(json.loads(tr.to_json(bare))["segments"], [])
        self.assertEqual(tr.to_srt(bare), "")


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.model = StubModel()
        self.whisper = install_stub_whisper(self.model)

    def tearDown(self):
        sys.modules.pop("whisper", None)
        self._tmp.cleanup()

    def audio(self, name):
        p = self.dir / name
        p.write_bytes(b"fake audio")
        return p

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = tr.main([str(a) for a in argv])
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_missing_file_exits_2_without_loading_model(self):
        code, _, err = self.run_cli(self.dir / "nope.mp3", "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("file not found", err)
        self.assertEqual(self.whisper.loads, 0)

    def test_writes_markdown_next_to_audio(self):
        a = self.audio("visit.mp3")
        code, out, _ = self.run_cli(a, "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertIn("Hello there.", (self.dir / "visit.md").read_text(encoding="utf-8"))

    def test_batch_loads_model_once(self):
        a, b = self.audio("a.mp3"), self.audio("b.mp3")
        out_dir = self.dir / "out"
        code, _, _ = self.run_cli(a, b, "--format", "json", "--out-dir", out_dir, "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertEqual(self.whisper.loads, 1)
        self.assertTrue((out_dir / "a.json").is_file() and (out_dir / "b.json").is_file())

    def test_fp16_only_on_cuda(self):
        a = self.audio("a.mp3")
        self.run_cli(a, "--device", "cpu")
        self.run_cli(a, "--device", "cuda")
        self.assertEqual([c[1] for c in self.model.calls], [False, True])

    def test_one_failure_exits_1_but_others_still_written(self):
        good, bad = self.audio("good.mp3"), self.audio("broken.mp3")
        code, _, err = self.run_cli(bad, good, "--device", "cpu")
        self.assertEqual(code, 1)
        self.assertIn("decode failed", err)
        self.assertTrue((self.dir / "good.md").is_file())

    def test_unknown_model_rejected(self):
        code, _, _ = self.run_cli(self.audio("a.mp3"), "--model", "gigantic")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
