# Video-Preprocess Refactor Plan

## Goals

1. **Behavioral parity with Rubiks** — align VME and thumbnail output/behavior to match the Rubiks specification
2. **Merge into a single app** — combine VME and thumbnail generator into one service; load the video once, run both
3. **Performance optimization** — reduce redundant downloads, modernize packages, add config to run both or only one

---

## 1. Feature Comparison: Rubiks vs. Current vs. Target

### 1.1 Video Metadata Extractor (VME)

| Feature | Rubiks Spec | Current (`video_preprocess.py`) | Target |
|---------|-------------|--------------------------------|--------|
| **Input source** | Buffer from upload stream via `pipe:0` (stdin) | Downloads full file to disk, reads from file path | File path (keep current — standalone app doesn't have upload stream access) |
| **FFprobe args** | `-i pipe:0 -select_streams v:0 -hide_banner -count_frames -count_packets -show_format -show_streams -of json` | `-select_streams v:0 -hide_banner -count_frames -count_packets -show_format -show_streams -of json "<filepath>"` | Keep ffprobe subprocess with safe list-based `subprocess.run` (no `shell=True`), parse JSON output |
| **FFprobe invocation** | Spawns child process, writes buffer to stdin | `subprocess.Popen` with `shell=True` (string cmd) | **Keep ffprobe subprocess** — switch from `shell=True` string command to safe `subprocess.run(["ffprobe", ...], capture_output=True)` list invocation |
| **Enablement: env var** | `ENABLE_RUBIKS_VIDEO_PREPROCESS` (default `false`) | None — always runs when triggered | Add `ENABLE_VIDEO_PREPROCESS` env var toggle |
| **Enablement: dataset-level** | `dataset.metadata.system.etlOptions.skipVideoEtl` | Ignore list loaded from external JSON file (`preprocess_ignore_list.json`) | Keep ignore list (local-only feature) AND add `skipVideoEtl` check from item/dataset context |
| **Enablement: chunk size** | `ETL_VIDEO_CHUNK_SIZE_KB` (1024 KB default) | N/A (reads full file) | N/A — not applicable for file-based processing |
| **Stream selection** | `streams.find(codec_type === 'video')` — first video stream | `next(stream for stream in streams if codec_type == 'video')` | Same — already matches |
| **No video stream error** | Throws `"video stream data is empty"` | Raises `ValueError('missing video stream for: ...')` | Align message to: `"video stream data is empty"` |
| **startTime** | `parseFloat(start_time) \|\| 0`, stored as `startTime` | `eval(start_time)`, default `0`, stored as `start_time` internally, `startTime` on item | Keep — already matches on item output. Replace `eval()` with `float()` |
| **height / width** | Direct from stream | Direct from stream | Same — already matches |
| **FPS calculation** | `parseFloat(eval(avg_frame_rate))` | `eval(fps)` with `ZeroDivisionError`/`SyntaxError` catch → `None` | Replace `eval()` with safe fraction parser. Keep error handling |
| **Duration: fallback order** | 1. `stream.duration` → 2. `tags.DURATION` via `durationStrToSec` → 3. `format.duration` | 1. `stream.duration` → 2. `tags.DURATION` → 3. `format.duration` | Same — already matches |
| **nb_frames** | `parseFloat(nb_frames)` → `null` if N/A. Always in output (nullable) | `eval(nb_frames)` → `None`. Only included if not None | Always include `nb_frames` in output (nullable). Replace `eval()` with `int()`/`float()` |
| **nb_streams** | `format.nb_streams \|\| 1` | `format.nb_streams` default `1` | Same — already matches |
| **nb_read_frames** | NOT in top-level output. Deleted from ffmpeg stream object | Extracted and included in top-level output as `nb_read_frames` | Remove from top-level output. Delete from ffmpeg stream object |
| **nb_read_packets** | Deleted from ffmpeg stream object | NOT deleted | Delete from ffmpeg stream object |
| **ffmpeg stream cleanup** | Deletes `index`, `nb_read_frames`, `nb_read_packets` from stream | Does NOT delete any fields | Delete `index`, `nb_read_frames`, `nb_read_packets` from stream before storing |
| **format.filename** | Kept in output (shows `"pipe:0"`) | Removed from format before writing | Keep in output (will show local path — acceptable for file-based) |
| **format in output** | Always included | Only if `video_format is not None` | Always include (already non-None for valid videos) |
| **Validation logic** | `if (nbFrames && fps && duration) { ... }` — tolerance 0.5 frames, throws Error | Same formula. Returns `(bool, exp_frames, error_dict)`. Error has `type/message/value/service` | Keep current validation logic — functionally equivalent |
| **Error structure** | `etl: { failed: true, errors: [...] }` in metadata output | `metadata.system.errors` array with `{type, message, value, service}` objects | Align to rubiks: use `etl.failed` + `etl.errors` structure |
| **Error handling** | Returns partial metadata with `etl` error block | Retries 3 times, raises exception on final failure | Keep retries. On final failure, return partial metadata with `etl` error block instead of throwing |
| **Backward compat fields** | None | Writes `item.metadata.startTime` and `item.metadata.fps` (outside `system`) | Keep for backward compatibility (local-only feature) |
| **on_delete handler** | Not specified | Deletes thumbnail + webm modality on item delete | Keep (local-only feature) |
| **OpenCV fallback** | Not specified | `metadata_extractor_from_opencv()` exists but unused in main flow | Remove dead code |
| **Output field: `start_time` key** | `startTime` (camelCase) | Internal dict uses `start_time`, item metadata uses `startTime` | Unify to `startTime` everywhere |
| **Use of `eval()`** | Uses `eval()` for FPS in JS (`parseFloat(eval(...))`) | Uses `eval()` for start_time, fps, nb_frames, duration | Replace ALL `eval()` with safe parsers (`float()`, fraction parsing) |

### 1.2 Thumbnail Generator

| Feature | Rubiks Spec | Current (`video_thumbnail.py`) | Target |
|---------|-------------|-------------------------------|--------|
| **Thumbnail format** | **Static PNG** (single frame) | **Animated GIF** (first 3 sec at 0.25x speed) | Keep **GIF** (local-only feature — current behavior is richer) |
| **Frame selection** | Single frame at ~50% of video duration | First 3 seconds, sampled at intervals | Keep current GIF approach. **Only decode the first `THUMB_DURATION_SEC` of the video** — do NOT load the full video into RAM |
| **Output dimensions** | `DEFAULT_THUMB_SIZE` × `DEFAULT_THUMB_SIZE` (128×128), maintain aspect ratio with `inside` fit | 128×128, forced square (`cv2.resize` / `ffmpeg scale=128:128`) | **Always preserve aspect ratio**: resize so the longest edge = `THUMB_MAX_EDGE` (default 128, configurable). Short edge is scaled proportionally. No square crop/stretch |
| **Resize method** | Sharp with `lanczos3` kernel | OpenCV `INTER_LINEAR` (primary), ffmpeg `scale` (fallback) | Keep OpenCV (Python equivalent). Consider using `INTER_LANCZOS4` for quality parity |
| **Video access** | File stream/buffer | HTTP stream via `item.stream` URL with auth headers | Change to use local downloaded file (merged app already downloads) |
| **Storage path** | `/.dataloop/thumbnails/{itemId}.png` | `/.dataloop/thumbnails/{itemId}.gif` | Keep `.gif` extension |
| **Enablement: env var** | `ENABLE_CREATE_IMG_THUMB` (default `true`) | None — always runs when triggered | Add `ENABLE_CREATE_THUMBNAIL` env var toggle |
| **Enablement: dataset-level** | `dataset.metadata.system.etlOptions.skipVideoEtl` | Ignore list from external JSON | Keep ignore list AND add `skipVideoEtl` check |
| **File size check** | `MAX_GEN_THUMB_SIZE_MB` (70 MB default) — skip if larger | No file size check | Add `MAX_GEN_THUMB_SIZE_MB` config, skip large files |
| **Thumbnail metadata** | Sets `metadata.system.thumbnailOf` on thumbnail item | NOT set | Add `thumbnailOf` reference on thumbnail item |
| **Video item update** | Sets `metadata.system.thumbnailId` on video item | Sets `metadata.system.thumbnailId` | Same — already matches |
| **Error handling** | Never fails the upload, logs and skips | Retries 3 times, raises exception | Align: on failure, log warning and skip (don't fail the whole process) |
| **Technology** | FFmpeg + Sharp (Node.js) | OpenCV + imageio (primary), FFmpeg (fallback) | Keep OpenCV + imageio for GIF. Consider ffmpeg-only approach for simplicity |
| **Shebang link support** | Not specified | Checks `item.system.shebang.linkInfo.ref` for URL links | Keep (local-only feature) |
| **Auth headers** | Not specified (buffer-based) | Adds `authorization` header for streaming | Remove — will use local file in merged app |

### 1.3 Architecture & Deployment

| Feature | Rubiks Spec | Current | Target |
|---------|-------------|---------|--------|
| **App count** | Single pipeline (metadata → thumbnail) | **Two separate apps**: `video-metadata-extractor` + `video-thumbnail` | **Single merged app**: `video-preprocess` |
| **Docker images** | Single image | Two images: `cpu/video-preprocess:4` (ffmpeg 4.4) + `cpu/thumbnail:3` (opencv) | **Single image** with ffmpeg + opencv + imageio |
| **Service instances** | One | VME: `regular-m`, concurrency 5, max 60 replicas. Thumbnail: `regular-s`, concurrency 3, max 20 replicas | Single service: `regular-m`, concurrency 5, configurable replicas |
| **Triggers** | Single trigger pipeline | VME: Created + Clone + Deleted. Thumbnail: Created + Clone | Merged: Created + Clone + Deleted |
| **Video download** | N/A (buffer from upload stream) | VME downloads full file. Thumbnail streams via HTTP (separate download) | **Download once**, use for both metadata + thumbnail |
| **Config: run mode** | Not applicable | Not applicable | New: `RUN_MODE` = `both` \| `metadata_only` \| `thumbnail_only` |
| **Execution timeout** | Not specified | 10800s (3 hours) for both | Keep 10800s |
| **Max attempts** | Not specified | 3 for both | Keep 3 |
| **Bot user** | Not specified | `pipelines@dataloop.ai` | Keep |
| **ffmpeg version** | "Any recent version" | 4.4 (built from source in Docker) | Upgrade to ffmpeg 6.x+ (install via apt for simplicity) |

---

## 2. Action Items Summary

### Phase 1: Behavioral Alignment (Rubiks Parity)

| # | Action | Priority | Files |
|---|--------|----------|-------|
| 1.1 | Replace all `eval()` calls with safe parsers (`float()`, fraction parsing) | High | `video_preprocess.py` |
| 1.2 | Always include `nb_frames` in output (as `None`/`null` if missing) | High | `video_preprocess.py` |
| 1.3 | Delete `nb_read_frames`, `nb_read_packets`, `index` from ffmpeg stream before storing | High | `video_preprocess.py` |
| 1.4 | Remove `nb_read_frames` from top-level output dict | High | `video_preprocess.py` |
| 1.5 | Align error structure to `etl: { failed, errors }` format | Medium | `video_preprocess.py` |
| 1.6 | On final failure, return partial metadata with `etl` block instead of raising | Medium | `video_preprocess.py` |
| 1.7 | Unify `start_time` → `startTime` in internal dict | Low | `video_preprocess.py` |
| 1.8 | Keep `format.filename` in output (stop removing it) | Low | `video_preprocess.py` |
| 1.9 | Add `thumbnailOf` metadata on thumbnail item | Medium | `video_thumbnail.py` |
| 1.10 | Thumbnail error handling: log and skip instead of raising | Medium | `video_thumbnail.py` |
| 1.11 | Add `MAX_GEN_THUMB_SIZE_MB` file size check for thumbnails | Medium | `video_thumbnail.py` |
| 1.12 | Add env var toggles: `ENABLE_VIDEO_PREPROCESS`, `ENABLE_CREATE_THUMBNAIL` | Medium | both |
| 1.13 | Add `skipVideoEtl` dataset-level check | Medium | both |

### Phase 2: Merge into Single App

| # | Action | Priority | Details |
|---|--------|----------|---------|
| 2.1 | Create unified `VideoPreprocessApp` class containing both VME and thumbnail logic | High | New main module |
| 2.2 | Single `on_create` entry point: download → extract metadata → generate thumbnail | High | Download once, reuse file |
| 2.3 | Add `RUN_MODE` config: `both` (default), `metadata_only`, `thumbnail_only` | High | Env var |
| 2.4 | Create single `dataloop.json` manifest with merged triggers | High | Replace two JSON files |
| 2.5 | Create single Dockerfile with ffmpeg + opencv + imageio | High | Replace two Dockerfiles |
| 2.6 | Update `application_deployment.py` for single-app deployment | Medium | |
| 2.7 | Keep `on_delete` handler (thumbnail + webm cleanup) | Medium | |
| 2.8 | Merge ignore lists under single service key | Low | |

### Phase 3: Performance & Modernization

| # | Action | Priority | Details |
|---|--------|----------|---------|
| 3.1 | **Fix ffprobe subprocess invocation** — switch from `shell=True` string command to safe `subprocess.run([...], capture_output=True)` list invocation, eliminating shell injection risk | **High** | Replaces `execute_cmd()` with a clean `_run_ffprobe(filepath)` helper that returns parsed JSON |
| 3.2 | **Keep OpenCV for frame extraction** in thumbnail generation — already works well, no change needed | **Low** | OpenCV `cv2.VideoCapture` for decoding + imageio for GIF compositing |
| 3.3 | Thumbnail: use local file instead of HTTP streaming (file already downloaded for VME) | High | Eliminates redundant download |
| 3.4 | Upgrade ffmpeg from 4.4 to 6.x+ in Docker image | Medium | Better codec support, install via apt for simplicity |
| 3.5 | Upgrade base image to Python 3.12+ | Medium | Performance improvements |
| 3.6 | Remove dead code: `metadata_extractor_from_opencv()`, `main_opencv_only.py` | Medium | Cleanup |
| 3.7 | Remove legacy files: `opencv_converter`, `opencv_converter.cpp`, `opencv_converter_test.cpp` | Low | Cleanup |
| 3.8 | Add structured logging (JSON format) | Low | Operability |

---

## 3. Merged App Architecture

```
on_create(item) / on_delete(item)
    │
    ├── Check enablement (env vars + ignore list + skipVideoEtl)
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
    │   ├── Decode only the first THUMB_DURATION_SEC of video via OpenCV (do NOT load full video into RAM)
    │   ├── Resize frames preserving aspect ratio (longest edge = THUMB_MAX_EDGE)
    │   ├── Compose GIF with imageio
    │   ├── Upload to /.dataloop/thumbnails/
    │   └── Update item.metadata.system.thumbnailId
    │
    ├── Cleanup temp dir
    │
    └── Return item

on_delete(item)
    ├── Delete thumbnail item if exists
    └── Delete webm modality if exists
```

### Config Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RUN_MODE` | string | `both` | `both`, `metadata_only`, or `thumbnail_only` |
| `ENABLE_VIDEO_PREPROCESS` | bool | `true` | Master switch for metadata extraction |
| `ENABLE_CREATE_THUMBNAIL` | bool | `true` | Master switch for thumbnail generation |
| `MAX_GEN_THUMB_SIZE_MB` | int | `70` | Skip thumbnail for files larger than this (MB) |
| `THUMB_MAX_EDGE` | int | `128` | Thumbnail longest-edge size in pixels (aspect ratio always preserved) |
| `THUMB_DURATION_SEC` | float | `3.0` | Duration of GIF thumbnail (seconds) |

---

## 4. Files to Create / Modify / Delete

### Create
- `video_preprocess_app.py` — merged app class
- `dataloop.json` — unified manifest
- `Dockerfile` — single Docker image
- `requirements.txt` — consolidated dependencies (imageio, dtlpy; ffprobe provided by system ffmpeg package)

### Modify
- `application_deployment.py` — update for single-app deployment

### Delete (after migration)
- `main_opencv_only.py` — legacy, unused
- `opencv_converter`, `opencv_converter.cpp`, `opencv_converter_test.cpp` — legacy C++ code
- `video_thumbnail_dataloop.json`, `video_thumbnail_dataloop_ford.json`, `video_thumbnail_dataloop_syngenta.json` — replaced by unified manifest
- `vme_dataloop.json`, `vme_dataloop_ford.json`, `vme_dataloop_syngenta.json` — replaced by unified manifest
- `thumbnail_docker/` — replaced by single Dockerfile
- `video_preprocess_docker/` — replaced by single Dockerfile
- `to_delete/` — cleanup
- `how_to_run`, `how_to_run.txt` — replace with proper README

### Keep (unchanged)
- `video_preprocess.py` — modify in place or keep as importable module
- `video_thumbnail.py` — modify in place or keep as importable module
- `IGNORE_LIST_README.md` — still relevant
- `recovery/` — review separately

---

## 5. ffprobe Subprocess Improvements

### Why keep ffprobe subprocess

- ffprobe JSON output maps **directly** to the metadata shape stored on items — no translation layer needed
- The `ffmpeg` stream/format dicts are already consumed downstream in this exact format
- Adding PyAV would introduce a C-extension build dependency and require constructing ffprobe-compatible dicts from PyAV objects for backward compatibility — extra complexity with no real benefit for this use case
- The only fix needed is switching from `shell=True` string command to safe list-based `subprocess.run`

### Improved invocation

```python
import subprocess, json

def _run_ffprobe(filepath: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_format", "-show_streams", "-print_format", "json", filepath],
        capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)
```

### Key changes from current code

- **No `shell=True`** — eliminates shell injection risk
- **`capture_output=True`** — captures stdout/stderr cleanly
- **`check=True`** — raises `CalledProcessError` on non-zero exit (stderr included)
- **Safe fraction parsing** — replace `eval(avg_frame_rate)` with `_safe_parse_fraction("30000/1001")` → `float`
- **No `eval()` anywhere** — all string-to-number conversions use `float()`, `int()`, or fraction parsing
- **`nb_read_frames`**: Per Rubiks spec, deleted from output anyway — can drop `-count_frames` flag to avoid expensive full-file decode
- **Duration fallback chain** remains the same: stream duration → tags.DURATION → format duration
