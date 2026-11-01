"""Tests for the offline sentence embedder (core/embedder.py) and its model tool (tools/fetch_model.py)."""

from __future__ import annotations

import functools
import hashlib
import http.client
import importlib.util
import io
import json
import logging
import os
import shutil
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from semdb.core import embedder as embedder_module
from semdb.core.embedder import ModelMissingError, OnnxEmbedder, load_default_embedder
from semdb.core.types import Cancelled

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2@e8f8c21/q8rr"


@functools.cache
def _shared(model_dir: Path) -> OnnxEmbedder:
    return OnnxEmbedder(model_dir)


@pytest.fixture
def embedder(model_dir: Path) -> OnnxEmbedder:
    """One instance for the whole run: loading the model takes about a second."""
    return _shared(model_dir)


def fund_names(count: int) -> list[str]:
    """Distinct fund-like names of varied length."""
    words = ["Global", "Equity", "Bond", "Emerging", "Markets", "Index", "Dividend", "Value", "Growth", "ESG"]
    return [f"Fund {i} " + " ".join(words[: 1 + i % len(words)]) for i in range(count)]


@pytest.fixture(scope="session")
def model_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A private copy of the bundled model.onnx and tokenizer.json.  Tests hard-link this copy,
    never models/ itself, so nothing written through a link can reach the real model."""
    source = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
    if not (source / "model.onnx").exists():
        pytest.skip("embedding model not downloaded (run tools/fetch_model.py)")
    copy = tmp_path_factory.mktemp("bundled-model")
    for name in ("model.onnx", "tokenizer.json"):
        shutil.copyfile(source / name, copy / name)
    return copy


def linked_model(model_files: Path, folder: Path) -> Path:
    """A model folder holding the bundled model.onnx and tokenizer.json (hard links to the
    private copy when the file system allows them)."""
    folder.mkdir(parents=True)
    for name in ("model.onnx", "tokenizer.json"):
        try:
            os.link(model_files / name, folder / name)
        except OSError:
            shutil.copyfile(model_files / name, folder / name)
    return folder


# Output contract ---------------------------------------------------------------------------


def test_encode_returns_unit_float32_rows(embedder: OnnxEmbedder) -> None:
    vectors = embedder.encode(["iShares Core S&P 500 UCITS ETF", "Vanguard FTSE All-World", "Fondo de renta fija"])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert embedder.dim == 384
    assert embedder.model_id == DEFAULT_MODEL_ID


def test_empty_list_gives_empty_matrix_without_progress(embedder: OnnxEmbedder) -> None:
    calls: list[tuple[int, int]] = []
    vectors = embedder.encode([], progress=lambda done, total: calls.append((done, total)))
    assert vectors.shape == (0, 384)
    assert vectors.dtype == np.float32
    assert calls == []


def test_empty_blank_and_broken_strings_embed(embedder: OnnxEmbedder) -> None:
    vectors = embedder.encode(["", "   ", "broken \udc80 surrogate", "Emerging markets"])
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_each_vector_depends_only_on_its_text(embedder: OnnxEmbedder) -> None:
    texts = fund_names(40) + ["", "S&P 500", "Fondo de inversión " * 30]
    together = embedder.encode(texts)
    alone = np.stack([embedder.encode([text])[0] for text in texts])
    np.testing.assert_allclose(together, alone, atol=1e-6)

    order = np.random.default_rng(0).permutation(len(texts))
    shuffled = embedder.encode([texts[i] for i in order])
    np.testing.assert_allclose(shuffled, together[order], atol=1e-6)
    # Distinct texts are far apart, so a row swapped with another could not pass the checks above.
    similarities = together @ together.T
    assert (similarities[~np.eye(len(texts), dtype=bool)] < 0.9999).all()


@pytest.mark.parametrize("count", [0, 1, 63, 64, 65, 200])
def test_batch_boundaries_and_progress(embedder: OnnxEmbedder, count: int) -> None:
    texts = fund_names(count)
    calls: list[tuple[int, int]] = []
    vectors = embedder.encode(texts, progress=lambda done, total: calls.append((done, total)))
    assert vectors.shape == (count, 384)
    if not count:
        assert calls == []
        return
    assert [total for _, total in calls] == [count] * len(calls)
    assert [done for done, _ in calls] == [0, *range(embedder.batch_size, count, embedder.batch_size), count]
    for i in sorted({0, 62, 63, 64, count // 2, count - 1} & set(range(count))):
        np.testing.assert_allclose(vectors[i], embedder.encode([texts[i]])[0], atol=1e-6)


def test_long_texts_are_truncated_to_the_first_tokens(embedder: OnnxEmbedder) -> None:
    head = " ".join(f"fund{i}" for i in range(300))  # far beyond 128 tokens
    first, second = embedder.encode([head + " ends here", head + " a completely different ending" * 400])
    np.testing.assert_allclose(first, second, atol=1e-6)
    huge = "fondo " * 200_000
    np.testing.assert_allclose(embedder.encode([huge])[0], embedder.encode([huge[:5000]])[0], atol=1e-6)


def test_whitespace_runs_in_huge_cells_do_not_crowd_out_the_words(embedder: OnnxEmbedder) -> None:
    # The tokenizer ignores whitespace, so these are the same four words; cutting a huge text by
    # characters before tokenizing must not let a run of padding push them out.
    plain = "Emerging markets equity fund"
    padded = ["Emerging" + " " * 6000 + "markets equity fund", "Emerging markets" + "\n\t " * 3000 + "equity fund"]
    np.testing.assert_allclose(embedder.encode(padded), embedder.encode([plain, plain]), atol=1e-6)


# Cancellation and concurrency ---------------------------------------------------------------


def test_cancel_before_start(embedder: OnnxEmbedder) -> None:
    with pytest.raises(Cancelled):
        embedder.encode(fund_names(10), cancel=lambda: True)


def test_cancel_between_batches_stops_promptly(embedder: OnnxEmbedder) -> None:
    reported: list[int] = []
    with pytest.raises(Cancelled):
        embedder.encode(
            fund_names(300),
            progress=lambda done, total: reported.append(done),
            cancel=lambda: bool(reported) and reported[-1] >= 64,
        )
    assert reported == [0, 64]
    assert embedder.encode(["still usable"]).shape == (1, 384)


def test_concurrent_encodes_from_four_threads_are_identical(embedder: OnnxEmbedder) -> None:
    texts = fund_names(150)
    expected = embedder.encode(texts)
    start = threading.Barrier(4)

    def run(shift: int) -> np.ndarray:
        start.wait()
        return embedder.encode(texts[shift:] + texts[:shift])

    shifts = [0, 37, 74, 111]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, shifts))
    for shift, result in zip(shifts, results):
        np.testing.assert_array_equal(result, np.roll(expected, -shift, axis=0))


# Meaning ---------------------------------------------------------------------------------------


def test_spanish_and_english_meanings_meet(embedder: OnnxEmbedder) -> None:
    spanish, english, unrelated = embedder.encode(["mercados emergentes", "Emerging Markets", "Government Bond"])
    assert float(spanish @ english) > 0.8
    assert float(spanish @ english) > float(spanish @ unrelated)


# Model folder handling ---------------------------------------------------------------------------


def test_missing_or_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelMissingError):
        OnnxEmbedder(tmp_path / "nowhere")
    with pytest.raises(ModelMissingError, match="model.onnx and tokenizer.json"):
        OnnxEmbedder(tmp_path)


def test_corrupt_files_raise(tmp_path: Path, model_dir: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"not a model")
    (tmp_path / "tokenizer.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ModelMissingError, match="tokenizer.json"):
        OnnxEmbedder(tmp_path)
    shutil.copyfile(model_dir / "tokenizer.json", tmp_path / "tokenizer.json")
    with pytest.raises(ModelMissingError, match="model.onnx"):
        OnnxEmbedder(tmp_path)


def test_unexpected_model_inputs_raise(tmp_path: Path, model_dir: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["pixel_values"], ["features"])],
        "image_model",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(tmp_path / "model.onnx"))
    shutil.copyfile(model_dir / "tokenizer.json", tmp_path / "tokenizer.json")
    with pytest.raises(ModelMissingError, match="inputs"):
        OnnxEmbedder(tmp_path)


def test_manifest_sets_model_id_and_sequence_length(tmp_path: Path, model_files: Path) -> None:
    folder = linked_model(model_files, tmp_path / "model")
    assert OnnxEmbedder(folder, threads=1).model_id == DEFAULT_MODEL_ID  # no manifest: pinned defaults

    manifest = {"model": "custom", "revision": "abcdef123456", "max_seq_length": 16, "quantization": {"tag": "q8x"}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    custom = OnnxEmbedder(folder, threads=2)
    assert custom.model_id == "custom@abcdef1/q8x"
    assert custom.max_seq_length == 16
    head = "Vanguard FTSE All-World UCITS ETF USD Accumulating share class with low fees"
    first, second = custom.encode([head + " and more words", head + " but another tail"])
    np.testing.assert_allclose(first, second, atol=1e-6)


@pytest.mark.parametrize(
    ("content", "message"),
    [("{not json", "unreadable"), ("[1, 2]", "JSON object"), ('{"max_seq_length": 4096}', "max_seq_length"),
     ('{"pooling": "cls"}', "pooling"), ('{"revision": "abc"}', "revision")],
)
def test_invalid_manifest_raises(tmp_path: Path, content: str, message: str) -> None:
    for name in ("model.onnx", "tokenizer.json"):  # never opened: the manifest is rejected first
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "manifest.json").write_text(content, encoding="utf-8")
    with pytest.raises(ModelMissingError, match=message):
        OnnxEmbedder(tmp_path)


def test_manifest_with_a_byte_order_mark_is_read(tmp_path: Path, model_files: Path) -> None:
    # Windows PowerShell 5.1 and older Notepad save UTF-8 with a BOM; that must not turn semantic search off.
    folder = linked_model(model_files, tmp_path / "model")
    (folder / "manifest.json").write_text('{"max_seq_length": 64}', encoding="utf-8-sig")
    assert OnnxEmbedder(folder, threads=1).max_seq_length == 64


def test_manifest_dimension_must_match_the_model(tmp_path: Path, model_files: Path) -> None:
    folder = linked_model(model_files, tmp_path / "model")
    (folder / "manifest.json").write_text('{"dim": 512}', encoding="utf-8")
    with pytest.raises(ModelMissingError, match="384-d embeddings, manifest.json says 512"):
        OnnxEmbedder(folder, threads=1)


def test_loads_from_a_non_ascii_path(tmp_path: Path, model_files: Path) -> None:
    folder = linked_model(model_files, tmp_path / "Modèles de José ü")
    assert OnnxEmbedder(folder, threads=1).encode(["Fondo de inversión"]).shape == (1, 384)


# load_default_embedder ----------------------------------------------------------------------------


def test_default_embedder_is_none_for_an_empty_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SEMDB_MODEL_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger=embedder_module.__name__):
        assert load_default_embedder() is None
    assert "Semantic search disabled" in caplog.text


def test_default_embedder_swallows_unexpected_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(_folder: Path) -> OnnxEmbedder:
        raise MemoryError("no room")

    monkeypatch.setenv("SEMDB_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(embedder_module, "OnnxEmbedder", explode)
    with caplog.at_level(logging.ERROR, logger=embedder_module.__name__):
        assert load_default_embedder() is None
    assert "no room" in caplog.text


def test_default_embedder_loads_the_bundled_model(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMDB_MODEL_DIR", str(model_dir))
    loaded = load_default_embedder()
    assert loaded is not None
    assert loaded.model_id == DEFAULT_MODEL_ID
    assert loaded.encode(["Emerging Markets"]).shape == (1, 384)


# tools/fetch_model.py (offline: a fake hub stands in for huggingface.co) ---------------------------


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """The model preparation tool, imported from its file (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location("fetch_model", ROOT / "tools" / "fetch_model.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse(io.BytesIO):
    """An HTTP response body that can lose its connection after ``drop_after`` bytes."""

    def __init__(self, body: bytes, status: int, drop_after: int | None) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}
        self.drop_after = drop_after

    def read(self, size: int | None = -1) -> bytes:
        if self.drop_after is None:
            return super().read(size)
        left = self.drop_after - self.tell()
        if left <= 0:
            raise http.client.IncompleteRead(b"")
        return super().read(left if size is None or size < 0 else min(size, left))


class FakeHub:
    """Serves the pinned files like the Hugging Face CDN: honours ``Range`` (206, or 416 past the
    end) and drops the connection once when ``drop_after`` is set."""

    def __init__(self, tool: ModuleType, files: dict[str, bytes]) -> None:
        self.tool = tool
        self.files = dict(files)
        self.requests: list[tuple[str, str | None]] = []
        self.drop_after: int | None = None

    def url(self, path: str) -> str:
        return self.tool.URL_TEMPLATE.format(repo=self.tool.REPO, revision=self.tool.REVISION, path=path)

    def urlopen(self, request: urllib.request.Request, timeout: float | None = None) -> FakeResponse:
        path = next(path for path in self.files if self.url(path) == request.full_url)
        body = self.files[path]
        ranged = request.get_header("Range")
        self.requests.append((path, ranged))
        start = int(ranged.removeprefix("bytes=").removesuffix("-")) if ranged else 0
        if ranged and start >= len(body):
            raise urllib.error.HTTPError(request.full_url, 416, "Range Not Satisfiable", None, None)  # type: ignore[arg-type]
        drop_after, self.drop_after = self.drop_after, None
        return FakeResponse(body[start:], 206 if ranged else 200, drop_after)


def serve(tool: ModuleType, monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> FakeHub:
    """Make ``tool`` download ``files`` from a fake hub, each pinned to its own SHA-256."""
    fake = FakeHub(tool, files)
    monkeypatch.setattr(tool.urllib.request, "urlopen", fake.urlopen)
    monkeypatch.setattr(tool, "SOURCE_FILES", {path: hashlib.sha256(data).hexdigest() for path, data in files.items()})
    return fake


@pytest.fixture
def hub(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """A fake hub serving one file, ``data.bin``; no real network, no retry delays."""
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)
    return serve(tool, monkeypatch, {"data.bin": bytes(range(256)) * 40})


def pin(tool: ModuleType, path: str, hub: FakeHub, dest: Path) -> None:
    tool.fetch(path, tool.SOURCE_FILES[path], dest)


PROBE_WORDS = ["Emerging markets equity fund", "Government Bond", "Fondo de renta fija"]


def tiny_model_files() -> dict[str, bytes]:
    """A miniature sentence-transformers repository (32-d Gather + MatMul "transformer", word-level
    tokenizer adding <s> ... </s>) with the files the tool downloads."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = sorted({word for probe in PROBE_WORDS for word in probe.split()})
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{word: i + 4 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    dim = 32
    rng = np.random.default_rng(7)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["embeddings", "input_ids"], ["tokens"]),
            helper.make_node("MatMul", ["tokens", "dense"], ["last_hidden_state"]),
        ],
        "tiny_sentence_encoder",
        [
            helper.make_tensor_value_info(name, TensorProto.INT64, ["batch", "sequence"])
            for name in ("input_ids", "attention_mask", "token_type_ids")
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", dim])],
        initializer=[
            numpy_helper.from_array(rng.normal(size=(len(vocab), dim)).astype(np.float32), "embeddings"),
            numpy_helper.from_array((rng.normal(size=(dim, dim)) / dim**0.5).astype(np.float32), "dense"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    configs = {
        "tokenizer_config.json": {},
        "special_tokens_map.json": {},
        "config.json": {"hidden_size": dim},
        "sentence_bert_config.json": {"max_seq_length": 64, "do_lower_case": False},
        "1_Pooling/config.json": {"word_embedding_dimension": dim, "pooling_mode_mean_tokens": True},
        "modules.json": [
            {"type": "sentence_transformers.models.Transformer"},
            {"type": "sentence_transformers.models.Pooling"},
        ],
    }
    return {
        "onnx/model.onnx": model.SerializeToString(),
        "tokenizer.json": tokenizer.to_str().encode("utf-8"),
        **{path: json.dumps(config).encode("utf-8") for path, config in configs.items()},
    }


def verified_folder(tool: ModuleType, folder: Path) -> Path:
    """A folder whose manifest matches its files, as the tool writes it (the files are not models)."""
    folder.mkdir(parents=True)
    files = {}
    for name in tool.OUTPUT_FILES:
        data = f"{name} placeholder".encode()
        (folder / name).write_bytes(data)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {
        "repo": tool.REPO,
        "revision": tool.REVISION,
        "quantization": tool.QUANTIZATION,
        "fidelity": {"mean_cosine": 0.99, "min_cosine": 0.98},
        "files": files,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_tool_downloads_a_pinned_file(tool: ModuleType, hub: FakeHub, tmp_path: Path) -> None:
    dest = tmp_path / "cache" / "data.bin"
    pin(tool, "data.bin", hub, dest)
    assert dest.read_bytes() == hub.files["data.bin"]
    assert not dest.with_name("data.bin.part").exists()
    pin(tool, "data.bin", hub, dest)  # cached: no second request
    assert hub.requests == [("data.bin", None)]


def test_tool_resumes_after_a_dropped_connection(tool: ModuleType, hub: FakeHub, tmp_path: Path) -> None:
    hub.drop_after = 3000
    dest = tmp_path / "data.bin"
    pin(tool, "data.bin", hub, dest)
    assert dest.read_bytes() == hub.files["data.bin"]
    assert hub.requests == [("data.bin", None), ("data.bin", "bytes=3000-")]


def test_tool_recovers_partial_and_stale_files(tool: ModuleType, hub: FakeHub, tmp_path: Path) -> None:
    data = hub.files["data.bin"]
    dest = tmp_path / "data.bin"
    part = tmp_path / "data.bin.part"

    part.write_bytes(data)  # complete download, interrupted before the rename: the server answers 416
    pin(tool, "data.bin", hub, dest)
    assert dest.read_bytes() == data and not part.exists()

    dest.write_bytes(b"stale")  # cached file with the wrong hash
    part.write_bytes(data + b"junk")  # corrupt leftover longer than the file
    pin(tool, "data.bin", hub, dest)
    assert dest.read_bytes() == data and not part.exists()
    assert hub.requests[1:] == [("data.bin", f"bytes={len(data) + 4}-"), ("data.bin", None)]


def test_tool_gives_up_when_the_hash_never_matches(tool: ModuleType, hub: FakeHub, tmp_path: Path) -> None:
    with pytest.raises(tool.FetchError, match="does not match the pinned"):
        tool.fetch("data.bin", "0" * 64, tmp_path / "data.bin")
    assert len(hub.requests) == tool.ATTEMPTS
    assert not any(tmp_path.iterdir())


def test_tool_prepares_a_model_folder_the_app_loads(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    serve(tool, monkeypatch, tiny_model_files())
    out = verified_folder(tool, tmp_path / "models" / "tiny")  # an older install, replaced on success
    for leftover in (f".tiny.previous-{os.getpid()}", ".tiny.staging-1"):  # from runs killed mid-way
        (out.parent / leftover).mkdir()
        (out.parent / leftover / "model.onnx").write_bytes(b"partial")

    tool.prepare(out, tmp_path / "download")
    assert sorted(path.name for path in out.parent.iterdir()) == ["tiny"]
    assert tool.check_folder(out) == []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dim"] == 32 and manifest["max_seq_length"] == 64 and manifest["fidelity"]["mean_cosine"] >= 0.98

    loaded = OnnxEmbedder(out, threads=2)  # the app accepts what the tool wrote
    assert loaded.model_id == DEFAULT_MODEL_ID and loaded.dim == 32 and loaded.max_seq_length == 64
    assert loaded.encode(["", *PROBE_WORDS]).shape == (4, 32)

    capsys.readouterr()
    monkeypatch.setattr(tool, "prepare", lambda *args: pytest.fail("a verified folder must not be rebuilt"))
    assert tool.main(["--out", str(out)]) == 0
    assert "already prepared" in capsys.readouterr().out
    assert tool.main(["--check", "--out", str(out)]) == 0


def test_tool_keeps_the_installed_model_when_a_check_fails(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(tool, monkeypatch, tiny_model_files())
    monkeypatch.setattr(tool, "_quantize", shutil.copyfile)
    monkeypatch.setattr(tool, "_fidelity", lambda *args: {"mean_cosine": 0.5, "min_cosine": 0.1})
    out = verified_folder(tool, tmp_path / "models" / "installed")
    before = {path.name: path.read_bytes() for path in out.iterdir()}

    with pytest.raises(tool.FetchError, match="not faithful enough"):
        tool.prepare(out, tmp_path / "download")
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before
    assert sorted(path.name for path in out.parent.iterdir()) == ["installed"]


def test_tool_check_reports_tampering(tool: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = verified_folder(tool, tmp_path / "model")
    assert tool.main(["--check", "--out", str(folder)]) == 0
    (folder / "tokenizer.json").write_bytes(b"tokenizer.json placeholdeR")  # same size, other bytes
    assert tool.main(["--check", "--out", str(folder)]) == 1
    assert "tokenizer.json does not match the SHA-256" in capsys.readouterr().err
    (folder / "manifest.json").unlink()
    assert tool.main(["--check", "--out", str(folder)]) == 1


def test_tool_output_survives_paths_the_console_cannot_encode(
    tool: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # build.ps1 pipes the tool's output, so Python writes it in the ANSI code page (cp1252 here),
    # strictly on stdout and with backslash escapes on stderr.
    folder = verified_folder(tool, tmp_path / "Модель Ω")
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="backslashreplace")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(tool, "prepare", lambda *args: pytest.fail("a verified folder must not be rebuilt"))

    assert tool.main(["--check", "--out", str(folder)]) == 0
    assert tool.main(["--out", str(folder)]) == 0
    (folder / "manifest.json").unlink()
    assert tool.main(["--check", "--out", str(folder)]) == 1
    stdout.flush()
    stderr.flush()
    assert stdout.buffer.getvalue().decode("cp1252").count("\\u041c\\u043e\\u0434") == 2
    assert "FAILED" in stderr.buffer.getvalue().decode("cp1252")
