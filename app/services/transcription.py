"""Sprint 2 — transcription: faster-whisper ASR with word-level timestamps,
text normalization, and transcript artifact storage (TRD Doc 2 sec 6,
AI/RAG spec Doc 5 principle 9; artifact shape per Data spec Doc 4 sec 8).

Picks up exactly where Sprint 1 (ingestion) left off: a source sitting in
TRANSCRIBING with an AUDIO artifact on disk. This module runs the ASR
model, writes three artifacts (raw / normalized / word-timestamps), and
hands the source off to PROCESSING for Sprint 3 (content structuring —
sections/chunks/sentences) to pick up.

Kept independent of app/services/ingestion.py: ingestion owns download/
save/extract, this owns ASR — the model-abstraction split from
app/core/interfaces.py applies here too (concrete engine is swappable
without touching the pipeline that calls it).

Sprint 7 (speed + accuracy refinement): two changes on top of Sprint 2,
both aimed directly at "a 1-hour lecture must process real quick AND the
answer must be far more accurate":
  1. GPU runs now go through faster-whisper's BatchedInferencePipeline
     instead of a plain model.transcribe() call. Batching parallelizes
     VAD-detected speech chunks across the GPU instead of decoding them
     one at a time -- faster-whisper's own benchmarks show ~2-4x
     throughput on long audio with no accuracy change (it's the same
     model weights, same decode, just batched). CPU keeps the old
     unbatched path: batching's win is GPU parallelism, and this is
     already the CPU *fallback*, not the common case.
  2. Default asr_model moved from "base" to "small". Model size is the
     single biggest lever on transcript accuracy (which every downstream
     stage -- chunking, retrieval, generation, grounding -- inherits
     errors from), and GPU is already confirmed working on real hardware
     here, so "base" was leaving accuracy on the table for a speed
     concern that (1) above now largely pays for.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import unicodedata
from pathlib import Path

from sqlalchemy.orm import Session as DbSession

from app.core.config import get_settings
from app.core.interfaces import Transcriber, Transcript, TranscriptSegment, WordTimestamp
from app.db.models.processing import ProcessingJob
from app.db.models.source import Source
from app.db.repositories import processing_job_repository as jobs
from app.db.repositories import source_repository as sources

settings = get_settings()

# Loaded models are expensive (seconds to load, hundreds of MB) — cache by
# size so re-transcribing within the same process reuses the same model.
# Value is (model, device_actually_used, batched_pipeline_or_None) — the
# device matters for the inference-time fallback in
# WhisperTranscriber.transcribe below; the batched pipeline (Sprint 7) is
# only ever non-None on a successful GPU load.
_MODEL_CACHE: dict[str, tuple[object, str, object | None]] = {}

# Self-audit finding (post-Sprint-11): the check-then-set in _get_model
# below had no lock -- ingestion.py spawns one background pipeline thread
# per source, so two sources added close together could both see a given
# size not yet cached and both attempt the GPU probe/load concurrently.
# Unlike a typical cache lock, this one is deliberately held for the
# WHOLE first load (not just the check-and-set) -- that's the actual fix
# here: only one thread should ever be doing the GPU probe for a given
# model size at a time, so a second, redundant CUDA context never gets
# opened purely from ingestion timing. Doesn't touch the separate
# inference-time GPU->CPU fallback further down (WhisperTranscriber.
# transcribe's own _MODEL_CACHE write) -- that's a narrower, harder-to-
# hit race not worth risking this already-hard-won fallback logic for.
_MODEL_CACHE_LOCK = threading.Lock()


# The full set of pip-installable NVIDIA CUDA-12 runtime packages
# ctranslate2's GPU backend can end up needing, directly or transitively,
# at model-load or first-inference time. cublas/cudnn are the two this
# pipeline touches directly; the rest are THEIR dependencies:
#   - nvJitLink (own package since CUDA 12.x): cublas64_12.dll itself
#     won't resolve without it.
#   - cuSPARSE / cuRAND: pulled in by some cuDNN convolution/RNN kernels.
#   - cudart (cuda_runtime) / nvrtc: base CUDA runtime + the JIT compiler
#     cuDNN uses for some kernels.
# Installing all of them, even ones a given run never calls, is cheap
# (~1GB total, one-time) and turns "which exact sub-dependency is
# missing this time" from a guessing game into a single install.
_NVIDIA_DLL_PACKAGES = (
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.nvjitlink",
    "nvidia.cusparse",
    "nvidia.curand",
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
)

_NVIDIA_PIP_PACKAGES = (
    "nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-nvjitlink-cu12 "
    "nvidia-cusparse-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12 "
    "nvidia-cuda-nvrtc-cu12"
)


def _register_nvidia_dll_dirs() -> None:
    """ctranslate2 (faster-whisper's backend) needs the cuBLAS/cuDNN
    runtime DLLs (and their own sub-dependencies — see
    _NVIDIA_DLL_PACKAGES above) to run on GPU — but NOT the full CUDA
    Toolkit install. The matching PyPI wheels ship just those DLLs, but
    Windows won't find them automatically since they land under
    site-packages rather than on PATH. This is a no-op wherever a given
    package isn't installed, or on non-Windows where it isn't needed —
    CPU fallback still works either way, so this never blocks anything,
    it only unlocks GPU when the pieces are actually there.

    Two independent registration mechanisms are used, deliberately, not
    one: os.add_dll_directory() is the modern, Python-recommended way,
    but ctranslate2's compiled extension has been reported (across
    several Whisper-adjacent projects, on Windows specifically) to still
    fail to resolve a DLL's own sub-dependencies through it alone on
    some driver/DLL combinations. Prepending the same directories to
    PATH is the older, more universally-respected DLL search mechanism.
    Doing both means this doesn't depend on guessing which loading path
    ctranslate2's binary actually uses."""
    if sys.platform != "win32":
        return
    import importlib.util

    registered_dirs: list[Path] = []
    for pkg in _NVIDIA_DLL_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError) as exc:
            print(f"[EduRAG] {pkg}: not found ({exc})")
            continue
        if not spec or not spec.submodule_search_locations:
            print(f"[EduRAG] {pkg}: import machinery has no spec/location for it — "
                  f"not installed for this same 'python'")
            continue
        for location in spec.submodule_search_locations:
            dll_dir = Path(location) / "bin"
            if not dll_dir.is_dir():
                print(f"[EduRAG] {pkg}: found package at {location} but no bin/ subfolder there")
                continue
            registered_dirs.append(dll_dir)
            try:
                os.add_dll_directory(str(dll_dir))
                print(f"[EduRAG] {pkg}: registered DLL directory {dll_dir}")
            except OSError as exc:
                print(f"[EduRAG] {pkg}: found {dll_dir} but add_dll_directory failed ({exc})")

    if registered_dirs:
        path_prefix = os.pathsep.join(str(d) for d in registered_dirs)
        os.environ["PATH"] = path_prefix + os.pathsep + os.environ.get("PATH", "")
        print(f"[EduRAG] Also prepended {len(registered_dirs)} NVIDIA bin dir(s) to PATH for this process.")
    else:
        print(f"[EduRAG] No NVIDIA CUDA-12 runtime packages found at all — GPU inference needs "
              f"'python -m pip install --user {_NVIDIA_PIP_PACKAGES}'")


_register_nvidia_dll_dirs()


class TranscriptionError(Exception):
    """Raised for any ASR failure; the message is what the UI/DB shows."""


def _model_size(asr_model_spec: str) -> str:
    """settings.asr_model is "faster-whisper:base" — this is the only
    engine wired up right now, so we just take the size after the colon."""
    return asr_model_spec.split(":", 1)[1] if ":" in asr_model_spec else asr_model_spec


def _build_model(size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type=compute_type)


def _build_batched_pipeline(model) -> object | None:
    """Sprint 7: wrap a loaded GPU model in faster-whisper's
    BatchedInferencePipeline for the 2-4x long-audio throughput win.
    Never allowed to block a source: any failure here (e.g. an older
    faster-whisper without this class) just means we fall back to the
    plain unbatched model, exactly as if this were CPU."""
    try:
        from faster_whisper import BatchedInferencePipeline

        return BatchedInferencePipeline(model=model)
    except Exception as exc:
        print(f"[EduRAG] BatchedInferencePipeline unavailable ({exc}); using unbatched GPU inference.")
        return None


def _get_model(size: str) -> tuple[object, str, object | None]:
    """GPU-first, CPU-always-works fallback. The "consumer laptop
    feasible" constraint (PRD Doc 1 sec 36) means CPU has to work with
    zero setup, but a discrete NVIDIA GPU should get used automatically
    when one's actually present and usable — no config flag to flip,
    since a wrong guess there just means another support round-trip.
    float16 on GPU, int8 on CPU: the usual speed/VRAM-appropriate choice
    for each. Falls back silently rather than failing the source: a GPU
    can be visible but still unusable (CUDA/cuDNN runtime not installed),
    and that failure mode is common enough on Windows to plan for."""
    if size in _MODEL_CACHE:
        return _MODEL_CACHE[size]

    with _MODEL_CACHE_LOCK:
        # Re-check inside the lock: another thread may have already
        # finished loading this exact size while this thread was
        # waiting -- and if it's still loading, waiting HERE (rather
        # than racing another concurrent GPU probe) is the whole point.
        if size in _MODEL_CACHE:
            return _MODEL_CACHE[size]

        try:
            import faster_whisper  # noqa: F401 — import check only
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed (pip install -r requirements.txt)."
            ) from exc

        try:
            gpu_model = _build_model(size, "cuda", "float16")
            batched = _build_batched_pipeline(gpu_model)
            result = (gpu_model, "cuda", batched)
            print(
                f"[EduRAG] ASR model '{size}' loaded on GPU (cuda/float16), "
                f"batched={'on' if batched else 'off'}."
            )
        except Exception as exc:
            print(f"[EduRAG] ASR model '{size}' could not load on GPU ({exc}); using CPU (int8).")
            result = (_build_model(size, "cpu", "int8"), "cpu", None)

        _MODEL_CACHE[size] = result
        return result


def get_active_device(size: str) -> str | None:
    """Which device ended up being used for a given model size, after any
    GPU -> CPU fallback — exposed so callers can log/display it instead of
    leaving "did the GPU actually kick in?" as a guessing game."""
    cached = _MODEL_CACHE.get(size)
    return cached[1] if cached else None


def _normalize_text(text: str) -> str:
    """Unicode-normalize and collapse whitespace. Deliberately NOT
    lowercasing or stripping punctuation — those would lose information
    the generation/citation layers need from the evidence text later."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_language_hint(language: str | None) -> str | None:
    """Source metadata (e.g. yt-dlp's reported language) comes back as
    locale-style codes like "en-US" or "pt-BR", but faster-whisper only
    accepts its own short list of bare ISO 639-1 codes ("en") and raises
    ValueError on anything else. Take just the base subtag; if there's
    nothing usable, return None and let Whisper auto-detect instead of
    guessing wrong."""
    if not language:
        return None
    base = re.split(r"[-_]", language, maxsplit=1)[0].strip().lower()
    return base or None


class WhisperTranscriber(Transcriber):
    """Concrete Transcriber (app/core/interfaces.py) backed by
    faster-whisper. Swapping ASR engines later means adding another class
    here, not touching this module's callers."""

    def __init__(self, model_spec: str):
        self._size = _model_size(model_spec)

    def transcribe(self, audio_path: str, language: str | None = None) -> Transcript:
        language = normalize_language_hint(language)
        model, device, batched = _get_model(self._size)

        try:
            segments, info = self._run_with_language_fallback(model, batched, audio_path, language)
        except Exception as exc:
            if device != "cuda":
                raise TranscriptionError(f"Transcription failed: {exc}") from exc
            # The GPU model *loaded* fine but failed on actual inference --
            # CUDA/cuDNN libraries are often loaded lazily on first use, so
            # a missing-runtime problem frequently surfaces here rather
            # than at construction. Fall back to CPU once, permanently for
            # this process (re-tried on every future source too), instead
            # of failing every source that comes after this one.
            print(f"[EduRAG] GPU inference failed ({exc}); switching to CPU (int8) for '{self._size}'.")
            model = _build_model(self._size, "cpu", "int8")
            _MODEL_CACHE[self._size] = (model, "cpu", None)
            try:
                segments, info = self._run_with_language_fallback(model, None, audio_path, language)
            except Exception as retry_exc:
                raise TranscriptionError(f"Transcription failed: {retry_exc}") from retry_exc

        return Transcript(language=info.language, segments=segments)

    def _run_with_language_fallback(self, model, batched, audio_path: str, language: str | None):
        try:
            return self._run(model, batched, audio_path, language)
        except ValueError as exc:
            if language is not None and "not a valid language code" in str(exc):
                # The normalized hint still wasn't one faster-whisper
                # recognizes (a handful of yt-dlp/locale codes don't map
                # 1:1 onto Whisper's list) — auto-detect rather than
                # failing the whole source over a metadata quirk.
                return self._run(model, batched, audio_path, None)
            raise

    @staticmethod
    def _run(model, batched, audio_path: str, language: str | None):
        # Sprint 7: prefer the batched pipeline (GPU only, see
        # _build_batched_pipeline) for the long-audio throughput win; same
        # weights/decode either way, so this never trades accuracy for speed.
        engine = batched if batched is not None else model
        kwargs = {"language": language, "word_timestamps": True, "vad_filter": True}
        if batched is not None:
            kwargs["batch_size"] = settings.asr_batch_size
        segments_iter, info = engine.transcribe(audio_path, **kwargs)
        segments = []
        for seg in segments_iter:
            words = [
                WordTimestamp(word=w.word.strip(), start=w.start, end=w.end)
                for w in (seg.words or [])
            ]
            segments.append(
                TranscriptSegment(text=seg.text.strip(), start=seg.start, end=seg.end, words=words)
            )
        return segments, info


def _get_transcriber() -> WhisperTranscriber:
    return WhisperTranscriber(settings.asr_model)


def transcribe_source(db: DbSession, source: Source, job: ProcessingJob) -> None:
    """Runs ASR on the source's AUDIO artifact, writes transcript
    artifacts, and moves the source to PROCESSING. Raises
    TranscriptionError on any failure — callers (ingestion.py) already
    wrap this in a try/except that marks the source FAILED."""
    jobs.start_job(db, job, stage="TRANSCRIBING")

    audio_artifact = sources.get_latest_artifact(db, source_id=source.id, artifact_type="AUDIO")
    if audio_artifact is None:
        raise TranscriptionError("No audio artifact found for this source — audio extraction may have failed.")

    # Commit before the ASR call below -- by far the slowest step in the
    # whole pipeline (minutes, for a real lecture). Leaving the job-start
    # write above uncommitted for that entire duration would hold SQLite's
    # one write lock the whole time and block every other write in the
    # app, exactly the live "database is locked" bug this was confirmed
    # to cause (see app/db/session.py's docstring for the full story).
    db.commit()

    transcript = _get_transcriber().transcribe(audio_artifact.storage_path, language=source.language)
    if not transcript.segments:
        raise TranscriptionError("No speech was detected in the audio.")

    device_used = get_active_device(_model_size(settings.asr_model)) or "unknown"
    print(f"[EduRAG] Source {source.id} transcribed using device={device_used}")

    # Whole-pipeline percentage, not a private per-function scale -- see
    # the matching comment in ingestion.py's _extract_audio.
    jobs.update_progress(db, job, progress=0.55, stage="NORMALIZING_TRANSCRIPT")

    raw_payload = {
        "language": transcript.language,
        "asr_device": device_used,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words],
            }
            for seg in transcript.segments
        ],
    }
    normalized_payload = {
        "language": transcript.language,
        "asr_device": device_used,
        "segments": [
            {"text": _normalize_text(seg.text), "start": seg.start, "end": seg.end}
            for seg in transcript.segments
        ],
    }
    timestamps_payload = {
        "language": transcript.language,
        "words": [
            {"word": w.word, "start": w.start, "end": w.end}
            for seg in transcript.segments
            for w in seg.words
        ],
    }

    # Resolved to absolute here, at write time — settings.transcript_dir is
    # relative by default, and a relative path stored in the DB would only
    # resolve correctly again if a later process happens to share the same
    # cwd (true today since the app is always launched from the project
    # root, but not a safe thing to depend on for paths read back later).
    transcript_dir = Path(settings.transcript_dir).resolve()
    transcript_dir.mkdir(parents=True, exist_ok=True)

    raw_path = transcript_dir / f"{source.id}.raw.json"
    normalized_path = transcript_dir / f"{source.id}.normalized.json"
    timestamps_path = transcript_dir / f"{source.id}.timestamps.json"

    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    normalized_path.write_text(json.dumps(normalized_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timestamps_path.write_text(json.dumps(timestamps_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sources.add_artifact(
        db, source_id=source.id, artifact_type="RAW_TRANSCRIPT",
        storage_path=str(raw_path), mime_type="application/json", size_bytes=raw_path.stat().st_size,
    )
    sources.add_artifact(
        db, source_id=source.id, artifact_type="NORMALIZED_TRANSCRIPT",
        storage_path=str(normalized_path), mime_type="application/json", size_bytes=normalized_path.stat().st_size,
    )
    sources.add_artifact(
        db, source_id=source.id, artifact_type="TIMESTAMP_TRANSCRIPT",
        storage_path=str(timestamps_path), mime_type="application/json", size_bytes=timestamps_path.stat().st_size,
    )

    if not source.language:
        sources.set_source_metadata(db, source, language=transcript.language)

    jobs.update_progress(db, job, progress=0.60, stage="AWAITING_CONTENT_STRUCTURING")

    # Hand off to Sprint 3 (content structuring: sections/chunks/sentences).
    sources.update_source_status(db, source, "PROCESSING")
