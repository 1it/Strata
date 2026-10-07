"""Existing model files need explicit replacement: no GPU or real downloads.

    python -m unittest tools.test_setup_downloads
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402
from test_setup_pins import Response  # noqa: E402


def gguf_bytes():
    data = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 2, 0))
    for i, name in enumerate(("blk.0.attn_q.weight", "blk.0.attn_k.weight")):
        data += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<IQIQ", 1, 8, 0, 32 * i)
    return bytes(data) + bytes(-len(data) % 32 + 64)


class Downloads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dst = self.root / "model.gguf"
        self.original = gguf_bytes()
        self.dst.write_bytes(self.original)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_complete_unmarked_gguf_is_reused_without_network_or_copy(self):
        for url in ("https://example.com/model.gguf", str(self.root / "mirror.gguf")):
            self.dst.with_name("model.gguf.done").unlink(missing_ok=True)
            with mock.patch.object(setup.urllib.request, "urlopen") as network, \
                    mock.patch.object(setup.shutil, "copyfile") as copy:
                setup.download(url, self.dst)
            network.assert_not_called()
            copy.assert_not_called()
            self.assertEqual(self.dst.read_bytes(), self.original)
            self.assertTrue(setup.done(self.dst))

    def test_complete_file_is_reused_when_a_finish_mark_cannot_be_written(self):
        with mock.patch.object(Path, "write_text", side_effect=PermissionError("read only")), \
                mock.patch.object(setup.urllib.request, "urlopen") as network:
            setup.download("https://example.com/model.gguf", self.dst)
        network.assert_not_called()
        self.assertFalse(setup.done(self.dst))
        self.assertEqual(self.dst.read_bytes(), self.original)

    def test_an_existing_short_or_malformed_gguf_is_not_replaced(self):
        invalid_type = struct.pack("<IIQQQ", 0x46554747, 3, 0, 1, 1) + b"x" + struct.pack("<I", 99)
        for data in (self.original[:-40], b"not GGUF", b"", invalid_type):
            self.dst.write_bytes(data)
            with mock.patch.object(setup.urllib.request, "urlopen") as network, self.assertRaises(SystemExit):
                setup.download("https://example.com/model.gguf", self.dst)
            network.assert_not_called()
            self.assertEqual(self.dst.read_bytes(), data)

    def test_force_fetches_even_same_size_and_discards_old_partial_and_hash(self):
        new = self.original[:-1] + b"x"
        setup.mark(self.dst, "sha256 " + hashlib.sha256(self.original).hexdigest())
        self.dst.with_name("model.gguf.part").write_bytes(b"obsolete partial")
        calls = []

        def urlopen(req, timeout=None):
            self.assertEqual(self.dst.read_bytes(), self.original)  # old file stays until the transfer completes
            calls.append((req.get_method(), req.headers.get("Range")))
            return Response(new)

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            setup.download("https://example.com/model.gguf", self.dst, force=True)
        self.assertEqual(calls, [("HEAD", None), ("GET", "bytes=0-")])
        self.assertEqual(self.dst.read_bytes(), new)
        self.assertNotIn("sha256", self.dst.with_name("model.gguf.done").read_text())
        self.assertFalse(self.dst.with_name("model.gguf.part").exists())

    def test_failed_force_download_keeps_the_existing_file_and_mark(self):
        setup.mark(self.dst, "old marker")
        with mock.patch.object(setup.urllib.request, "urlopen", side_effect=OSError("offline")), \
                mock.patch.object(setup.time, "sleep"), self.assertRaises(SystemExit):
            setup.download("https://example.com/model.gguf", self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), self.original)
        self.assertEqual(self.dst.with_name("model.gguf.done").read_text(), "old marker")

    def test_local_force_copy_replaces_only_after_copy_finishes(self):
        src = self.root / "new.gguf"
        src.write_bytes(self.original[:-1] + b"x")
        setup.download(str(src), self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), src.read_bytes())
        with mock.patch.object(setup.shutil, "copyfile", side_effect=OSError("disk full")), \
                self.assertRaises(OSError):
            setup.download(str(src), self.dst, force=True)
        self.assertEqual(self.dst.read_bytes(), src.read_bytes())

    def test_missing_destination_still_resumes_an_existing_part(self):
        self.dst.unlink()
        self.dst.with_name("model.gguf.part").write_bytes(self.original[:32])
        calls = []

        def urlopen(req, timeout=None):
            calls.append((req.get_method(), req.headers.get("Range")))
            return Response(self.original) if req.get_method() == "HEAD" else Response(self.original[32:], status=206)

        with mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            setup.download("https://example.com/model.gguf", self.dst)
        self.assertEqual(calls, [("HEAD", None), ("GET", "bytes=32-")])
        self.assertEqual(self.dst.read_bytes(), self.original)

    def test_a_local_file_with_the_wrong_pinned_hash_can_be_preserved(self):
        with self.assertRaises(SystemExit):
            setup.verify_sha256(self.dst, len(self.original), "0" * 64, remove_bad=False)
        self.assertEqual(self.dst.read_bytes(), self.original)


class Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = Path(self.tmp.name) / "models"
        self.fam = setup.FAMILIES["qwen"]
        self.shards = [self.models / "Q2_0" / setup.model_file(self.fam, "Q2_0", i) for i in (1, 2)]
        for shard in self.shards:
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_bytes(gguf_bytes())
        self.argv = ["--models-dir", str(self.models), "--family", "qwen", "--model", "Q2_0", "--no-start"]
        self.ram, self.cards = PROFILES["64GB-1x32GB"]

    def test_whole_unmarked_shards_are_reused_even_without_writable_marks(self):
        # setup's harness mocks GGUF parsing; the download tests above exercise the real header reader.
        download = mock.Mock(side_effect=AssertionError("downloaded an existing model"))
        code, out, _, _ = install(self.ram, self.cards, self.argv, extra=[
            mock.patch.object(setup, "whole_shard", return_value=True),
            mock.patch.object(setup, "mark", return_value=None),
            mock.patch.object(setup, "download", download),
        ])
        self.assertEqual(code, 0, out)
        download.assert_not_called()

    def test_force_replaces_marked_shards_and_vision_and_rebuilds_the_pack(self):
        for shard in self.shards:
            setup.mark(shard)
        encoder = self.models / self.fam["mmproj"]
        encoder.write_bytes(gguf_bytes())
        setup.mark(encoder)
        calls, builds = [], []

        def free_gb(path):
            pack = setup.ROOT / "data/packs/q2_0"
            pack.mkdir(parents=True, exist_ok=True)
            for name in ("index.txt", "native_experts.txt", "experts.bin", "keep.txt"):
                (pack / name).write_bytes(b"old cache")
            return 900

        def download(url, dst, what=None, **kwargs):
            calls.append((Path(dst), kwargs))
            self.assertTrue(kwargs.get("force"))
            dst.write_bytes(gguf_bytes())
            setup.mark(dst)

        def build(cmd, **kwargs):
            if "iq_pack.py" in cmd[1]:
                pack = Path(cmd[cmd.index("--out") + 1])
                self.assertFalse((pack / "experts.bin").exists())
                self.assertFalse((pack / "native_experts.txt").exists())
                self.assertTrue((pack / "keep.txt").exists())
                builds.append(cmd)

        code, out, _, _ = install(self.ram, self.cards, [*self.argv, "--force-download", "--vision", "cpu"], extra=[
            mock.patch.object(setup, "free_gb", free_gb),
            mock.patch.object(setup, "download", download),
            mock.patch.object(setup, "run", build),
        ])
        self.assertEqual(code, 0, out)
        self.assertEqual([path for path, _ in calls], [*self.shards, encoder])
        self.assertEqual(len(builds), 1)

    def test_force_conflicts_are_rejected_before_installation(self):
        for flags in (["--gguf-dir", str(self.models)], ["--check"], ["--update"], ["--rollback-engine"]):
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["setup.py", "--force-download", *flags]), \
                    mock.patch.object(setup, "data_folder") as install_data, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                setup.main()
            self.assertEqual(error.exception.code, 2)
            install_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
