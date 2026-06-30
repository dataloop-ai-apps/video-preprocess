
# Video-Preprocess Refactor Analysis

## 1. Current Capabilities

### Overview
A FastAPI-based microservice that accepts a video URL, downloads the video, and extracts comprehensive metadata including video/audio properties, scene boundaries, thumbnails, and speech-to-text captions.

**Single endpoint:** `POST /api/v1/process` — accepts `{ "url": "<video_url>" }` and returns a `VideoResponse`.

---

### Processing Pipeline (MetadataService orchestrator)

| Step | Service | What It Does | Requires Local File? |
|------|---------|-------------|----------------------|
| 1 | `VideoService.download_video()` | Downloads full video to a temp file via `httpx` streaming | — (this IS the download) |
| 2 | `FFProbeService.get_video_metadata()` | Runs `ffprobe` on local file → duration, resolution, FPS, codec, bitrate, audio info | ✅ Yes (local path) |
| 3 | `SceneService.detect_scenes()` | Runs `ffprobe`/`ffmpeg` scene-change filter on local file | ✅ Yes (local path) |
| 4 | `ThumbnailService.generate_and_upload_thumbnails()` | Extracts frames with `ffmpeg`, uploads JPEGs to Azure Blob Storage | ✅ Yes (local path) |
| 5 | `CaptionService.extract_captions_from_video()` | Extracts audio to WAV → runs OpenAI Whisper `base` model | ✅ Yes (local path) |

---

### Service-by-Service Breakdown

#### `VideoService` (`video_service.py`)
- `download_video(url)` — streams video to a temp file (httpx, 8KB chunks, 300s timeout)
- `cleanup(file_path)` — deletes temp file
- `get_video_duration_fast(url)` — **🔴 DEAD CODE** — runs ffprobe directly on URL, never called anywhere
- `get_frame_at_timestamp(video_path, timestamp)` — extracts a single JPEG frame via ffmpeg pipe

#### `FFProbeService` (`ffprobe_service.py`)
- `get_video_info(file_path)` — raw ffprobe JSON output (streams + format)
- `get_video_metadata(file_path)` — parsed/structured metadata dict

#### `SceneService` (`scene_service.py`)
- `detect_scenes(video_path, threshold=0.3)` — scene detection via ffprobe lavfi filter
- `_detect_scenes_ffmpeg()` — fallback using ffmpeg `select` filter + `showinfo`
- `_generate_uniform_scenes()` — last-resort fallback: chops video into 10s segments
- `_get_duration()` — **🟡 DUPLICATED** — same logic exists in `FFProbeService` and `VideoService`
- `_seconds_to_timecode()` — **🟡 DUPLICATED** — same function exists in `utils/helpers.py`

#### `ThumbnailService` (`thumbnail_service.py`)
- `generate_and_upload_thumbnails()` — generates thumbnails at scene midpoints (or uniform intervals), uploads to Azure
- Creates its **own** `VideoService` instance — **🟡 REDUNDANT** — doesn't use the one from MetadataService; the `video_path` is already passed in as a parameter, so only `get_frame_at_timestamp` is used from it

#### `AudioService` (`audio_service.py`)
- `extract_audio(video_path, output_path)` — ffmpeg video→WAV (16kHz mono PCM)
- `has_audio_stream(video_path)` — checks for audio stream via ffprobe
- `get_audio_info(video_path)` — **🔴 DEAD CODE** — returns audio codec/sample_rate/channels/bitrate, never called (FFProbeService already extracts this)

#### `CaptionService` (`caption_service.py`)
- `_load_model()` — lazy-loads Whisper `base` model
- `extract_captions(audio_path)` — transcribes WAV file
- `extract_captions_from_video(video_path)` — convenience wrapper: checks audio → extracts WAV → transcribes → cleans up

#### `StorageService` (`storage_service.py`)
- `upload_thumbnail(image_data, video_url, timestamp)` — uploads JPEG to Azure Blob Storage
- `delete_thumbnails(video_url)` — **🔴 DEAD CODE** — never called anywhere

#### `utils/helpers.py`
- `seconds_to_timecode()` — **🔴 DEAD CODE** — duplicated in `SceneService._seconds_to_timecode()`, never imported
- `sanitize_filename()` — **🔴 DEAD CODE** — never called anywhere

---

### Dead Code Summary

| Item | Location | Notes |
|------|----------|-------|
| `VideoService.get_video_duration_fast()` | `video_service.py` | Never called. Interesting though — shows ffprobe CAN read from URL |
| `AudioService.get_audio_info()` | `audio_service.py` | Never called. FFProbeService already extracts audio metadata |
| `StorageService.delete_thumbnails()` | `storage_service.py` | Never called. No cleanup/delete endpoint exists |
| `helpers.seconds_to_timecode()` | `utils/helpers.py` | Never imported. Duplicated in SceneService |
| `helpers.sanitize_filename()` | `utils/helpers.py` | Never called anywhere |

---

### Duplication Summary

| Duplicated Logic | Locations |
|-----------------|-----------|
| Get video duration via ffprobe | `VideoService.get_video_duration_fast()`, `SceneService._get_duration()`, `FFProbeService.get_video_metadata()` |
| Seconds → timecode conversion | `SceneService._seconds_to_timecode()`, `helpers.seconds_to_timecode()` |
| Audio stream detection / info | `AudioService.has_audio_stream()` + `get_audio_info()`, `FFProbeService.get_video_metadata()` (already parses audio streams) |
| ffprobe invocation pattern | Almost identical subprocess boilerplate in every service |

---

## 2. Improvement Suggestions

### 2.1 Download Once, Use Everywhere (Critical)

**Current problem:** The video is downloaded once in `MetadataService`, but:
- `ThumbnailService` creates its own `VideoService` instance (unnecessary, but the path is passed in)
- `SceneService._get_duration()` re-runs ffprobe for duration even though `FFProbeService` already extracted it
- `CaptionService.extract_captions_from_video()` re-checks `has_audio_stream()` even though `MetadataService` already knows from `FFProbeService`

**Recommendation:** Pass already-extracted metadata downstream instead of re-extracting:
```python
# MetadataService.process_video() should:
# 1. Download once → video_path
# 2. FFProbe once → metadata dict (includes duration, has_audio, etc.)
# 3. Pass metadata['duration'] into scene_service, thumbnail_service
# 4. Pass metadata['has_audio'] check result into caption_service
# 5. Don't let sub-services re-query what's already known
```

### 2.2 Consolidate ffprobe Calls

**Current:** 4+ separate ffprobe subprocess calls per video (metadata, scene duration, audio check, etc.)

**Recommendation:** Run ffprobe ONCE with `-show_format -show_streams` and pass the parsed result to all services that need it. Create a single `FFProbeResult` data class that holds everything.

### 2.3 Eliminate Redundant Service Instantiation

- `ThumbnailService` creates its own `VideoService` — should receive frame extraction as a callable or use the shared instance
- `CaptionService.extract_captions_from_video()` creates a new `AudioService` inline — should be injected

**Recommendation:** Use dependency injection or pass shared service instances from the orchestrator.

### 2.4 Parallel Processing

**Current:** All steps run sequentially (download → metadata → scenes → thumbnails → captions).

**Recommendation:** After download + initial ffprobe, run scene detection, thumbnail generation, and caption extraction in parallel using `asyncio.gather()`:
```python
scenes_task = asyncio.to_thread(self.scene_service.detect_scenes, video_path, duration)
captions_task = asyncio.to_thread(self.caption_service.extract_captions, video_path)
scenes, captions = await asyncio.gather(scenes_task, captions_task)
# Then generate thumbnails (needs scenes result)
```

### 2.5 Streaming/Chunked Processing

**Current:** Full video download before any processing begins (blocks on potentially GB-sized files).

**Recommendation:** For metadata-only operations, consider:
- Running ffprobe directly on the URL (ffprobe supports HTTP URLs natively — the dead code `get_video_duration_fast` already proves this works)
- Only downloading when frame extraction or audio processing is truly needed

### 2.6 Clean Up Dead Code
Remove all items listed in the Dead Code Summary above. They add confusion and maintenance burden.

### 2.7 Error Handling Consistency
- Some methods raise exceptions, others return empty values (`b''`, `{}`, `0.0`, `False`)
- `SceneService` silently falls back to uniform scenes on any error — this masks real problems
- **Recommendation:** Define explicit error strategy: either fail fast with descriptive errors, or return typed `Optional` results with clear "degraded" markers

### 2.8 Configuration Gaps
- Whisper model size is hardcoded to `"base"` — should be configurable
- Scene detection threshold (0.3) is hardcoded — should be configurable
- Thumbnail quality (`-q:v 2`) is hardcoded — should be configurable
- Download timeout (300s) is hardcoded — should be configurable
- No health check endpoint

---

## 3. Technology / Library Improvements

### 3.1 Improve `subprocess` ffprobe invocation (keep ffprobe)

The current code uses `subprocess.Popen` with `shell=True` and string commands, which is a security risk. However, the ffprobe JSON output already maps directly to the metadata shape stored on items — switching to PyAV would require building ffprobe-compatible dicts for backward compatibility with no real benefit.

**Recommendation:** Keep ffprobe subprocess, but switch to safe list-based `subprocess.run(["ffprobe", ...], capture_output=True, check=True)` invocation. Replace all `eval()` calls with safe parsers (`float()`, fraction parsing).

### 3.2 Replace OpenAI Whisper with `faster-whisper`

| Current (`openai-whisper`) | Recommended (`faster-whisper`) |
|---------------------------|-------------------------------|
| PyTorch-based, slow | CTranslate2-based, **4-8x faster** |
| High memory usage | ~50% less memory |
| No batching | Supports batching |
| Large Docker image (PyTorch) | Much smaller footprint |

**`faster-whisper`** is a drop-in replacement API-wise and dramatically reduces processing time and Docker image size.

### 3.3 Use `scenedetect` Library (PySceneDetect)

The current scene detection is fragile (custom ffprobe lavfi command with multiple fallbacks). **PySceneDetect** (`scenedetect` package) is purpose-built for this:
- Content-aware detection (not just threshold)
- Adaptive threshold detection
- Multiple detector types (ContentDetector, ThresholdDetector, AdaptiveDetector)
- Well-tested, actively maintained
- Can work with PyAV backend for better performance

### 3.4 Replace `httpx` Download with `aiofiles` + `aiohttp` or Smart Streaming

If download remains necessary:
- **`aiohttp`** may offer better streaming performance than `httpx` for large files
- Consider adding **range request support** for resumable downloads
- Add **progress tracking** for observability

### 3.5 Docker Image Optimization

**Current:** `python:3.11-slim` + ffmpeg + PyTorch (via whisper) → likely **5-8 GB image**

**Recommendations:**
- Switch to `faster-whisper` → removes PyTorch dependency → saves ~3 GB
- Multi-stage build → compile dependencies in builder, copy only runtime artifacts
- Pin ffmpeg version for reproducibility
- Use `python:3.12-slim` for performance improvements

### 3.6 Add Observability

- **Structured logging** (JSON) instead of plain text — easier to parse in cloud environments
- **Request tracing** — add request IDs to correlate log entries
- **Metrics** — processing time per step, video size distribution, error rates
- Consider **OpenTelemetry** for distributed tracing

---

## 4. Can We Avoid Downloading the Video?

### What ffprobe/ffmpeg Can Do Directly on URLs

ffprobe and ffmpeg natively support HTTP/HTTPS URLs. The dead code `VideoService.get_video_duration_fast()` already demonstrates this. Here's what works **without downloading**:

| Operation | Without Download? | How |
|-----------|:-:|-----|
| **Metadata extraction** (duration, resolution, codec, bitrate, audio info) | ✅ **YES** | `ffprobe -show_format -show_streams <URL>` — reads only headers + small portion of file |
| **Scene detection** | ⚠️ **PARTIAL** | ffprobe/ffmpeg can process URL streams, but needs to read the full video stream — effectively downloads it internally, but doesn't need disk space if piped. Performance depends on network. |
| **Frame extraction (thumbnails)** | ⚠️ **PARTIAL** | `ffmpeg -ss <time> -i <URL> -vframes 1 ...` works on URLs. ffmpeg seeks via HTTP range requests if the server supports it. For a few thumbnails this is efficient; for many thumbnails at different timestamps it may re-download portions repeatedly. |
| **Audio extraction (for captions)** | ⚠️ **PARTIAL** | `ffmpeg -i <URL> -vn -acodec pcm_s16le ...` works on URLs but streams the entire file over HTTP. |

### Practical Recommendation: Hybrid Approach

**Tier 1 — No download needed:**
- Metadata extraction → Use ffprobe directly on URL (fast, reads only file headers)
- This alone covers: duration, resolution, FPS, codec, bitrate, audio info, format

**Tier 2 — Selective download / stream processing:**
- Thumbnails → Use ffmpeg directly on URL with `-ss` (seek) before `-i` (input). For servers supporting HTTP range requests, ffmpeg will seek efficiently without downloading the full file. Extract frames one at a time.

**Tier 3 — Full download still required:**
- Scene detection → Must scan every frame. Full video data must be read.
- Caption extraction → Must process entire audio stream.
- Thumbnail generation → For our use case (animated GIF of first N seconds), local file with OpenCV is simplest.

### Note on PyAV

PyAV was considered but rejected for this project — the ffprobe JSON output already maps directly to the metadata shape stored on items, and introducing PyAV would add a C-extension build dependency while requiring a translation layer to produce ffprobe-compatible dicts for backward compatibility. The subprocess approach is simple, well-understood, and sufficient for our use case.

---

## 5. Proposed Refactored Architecture

```
on_create(item)
    │
    ├── Check enablement (flags + ignore list + skipVideoEtl)
    │
    ├── Download video to temp dir (once)
    │
    ├── [if metadata enabled] Run VME:
    │   ├── subprocess.run(["ffprobe", ...]) on local file → parse JSON
    │   ├── Extract streams/format metadata from ffprobe output
    │   ├── Validate frames
    │   └── Write to item.metadata.system
    │
    ├── [if thumbnail enabled] Generate thumbnail:
    │   ├── Check file size < MAX_GEN_THUMB_SIZE_MB
    │   ├── Decode first THUMB_DURATION_SEC via OpenCV
    │   ├── Resize frames, compose GIF with imageio
    │   └── Upload to /.dataloop/thumbnails/
    │
    ├── Cleanup temp dir
    └── Return item
```

### Key Principles
1. **Single ffprobe call** — extract all metadata once via subprocess, parse JSON
2. **Download once** — reuse the local file for both metadata extraction and thumbnail generation
3. **Shared state** — metadata extracted once, passed to all sub-processors
4. **Configurable pipeline** — module-level flags to run metadata-only, thumbnail-only, or both
5. **Explicit error boundaries** — `record_etl_error` writes to item metadata before re-raising

---

## 6. Migration Priority

| Priority | Change | Impact | Effort |
|----------|--------|--------|--------|
| 🔴 P0 | Remove dead code | Clarity | Low |
| 🔴 P0 | Consolidate ffprobe calls (single call, pass results) | Fewer subprocess spawns, faster | Low |
| 🔴 P0 | Pass metadata downstream (don't re-extract duration, has_audio) | Eliminate redundant work | Low |
| 🟡 P1 | Switch to `faster-whisper` | 4-8x faster captions, smaller Docker image | Medium |
| 🟡 P1 | Metadata extraction via ffprobe on URL (no download for metadata-only) | Major latency reduction for metadata | Low |
| 🟡 P1 | Parallel processing with `asyncio.gather` | Faster end-to-end | Medium |
| 🟢 P2 | Fix ffprobe subprocess (safe list invocation, no `shell=True`) | Eliminate shell injection risk | Low |
| 🟢 P2 | Adopt PySceneDetect | Better scene detection quality + reliability | Medium |
| 🟢 P2 | Make pipeline configurable (skip captions, skip thumbnails) | Flexibility, faster for partial requests | Medium |
| 🟢 P2 | Docker multi-stage build optimization | Smaller image, faster deploys | Medium |
| 🔵 P3 | Add health check + observability (structured logging, metrics) | Operability | Medium |
| 🔵 P3 | Add request-level feature flags (`include_captions=false`) | API flexibility | Low |
