"""Unified video-preprocess service.

Single-file implementation merging metadata extraction (via ffprobe) and
thumbnail GIF generation (via OpenCV + imageio) behind one dataloop service
and one trigger. Per-invocation behaviour is controlled via
``context.trigger_input`` with module-level constant fallbacks.

Inputs (read from ``context.trigger_input``):
    metadata_only      – bool (default False)  only run ffprobe metadata stage
    thumbnail_only     – bool (default False)  only run thumbnail stage
    thumbnail_size     – int  (default DEFAULT_THUMB_SIZE)
    max_file_size_mb   – int  (default MAX_FILE_SIZE_MB)
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import traceback

import cv2
import dtlpy as dl
import imageio

logger = logging.getLogger("video-preprocess")

# ---------------------------------------------------------------------------
# Module-level default constants (overridable per-invocation via trigger_input)
# ---------------------------------------------------------------------------

DEFAULT_THUMB_SIZE = 128
MAX_FILE_SIZE_MB = 2000        # overall input video size guard
MAX_GEN_THUMB_SIZE_MB = 70     # additional guard specifically for thumbnail stage
THUMB_DURATION_SEC = 3.0
THUMB_SPEED_FACTOR = 0.25      # decoded frames played back at 1/4 speed for the GIF

# Dataset-level metadata flag that disables ETL for the whole dataset.
SKIP_DATASET_FLAG = "skipVideoEtl"

# Service name used to look up the ignore list in the DataloopTasks binaries dataset.
IGNORE_LIST_SERVICE_KEY = "video-preprocess"

# Validation fields that must be populated after ffprobe runs.
VALIDATION_KEYS = ("ffmpeg", "height", "width", "fps", "duration")


# ---------------------------------------------------------------------------
# record_etl_error — inlined and adapted for the video service
# ---------------------------------------------------------------------------

def record_etl_error(item: dl.Item, stage: str, error: str, failed: bool = False) -> list:
    """Append an error to ``system.etl.errors`` and optionally set the failed flag.

    When ``failed`` is True, mirrors the failed-block onto ``system.videoEtl.etl``
    (analogous to ``imageEtl`` in the image-preprocess service) so downstream
    consumers can detect a hard failure on the item.
    """
    system = item.metadata.setdefault("system", {})
    etl = system.setdefault("etl", {})
    etl_errors = etl.setdefault("errors", [])
    entry = {"stage": stage, "error": error}
    etl_errors.append(entry)
    if failed:
        etl["failed"] = True
        system.setdefault("videoEtl", {})["etl"] = {
            "failed": True,
            "errors": etl_errors,
        }
    return etl_errors


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _safe_parse_fraction(value):
    """Parse FPS-style strings like ``"30000/1001"`` or ``"29.97"``.

    Returns ``float`` or ``None`` on failure. Avoids ``eval`` for safety.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        if "/" in s:
            num, den = s.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _duration_str_to_sec(time_str):
    """Parse ``HH:MM:SS.ms`` durations into seconds. Returns ``None`` on failure."""
    if time_str is None:
        return None
    try:
        h, m, s = str(time_str).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        logger.warning("Unsupported duration string: %r", time_str)
        return None


def _run_ffprobe(filepath: str) -> dict:
    """Invoke ``ffprobe`` and return the parsed JSON. Raises on non-zero exit."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-hide_banner",
        "-select_streams", "v:0",
        "-count_frames",
        "-count_packets",
        "-show_format",
        "-show_streams",
        "-print_format", "json",
        filepath,
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe failed (rc={proc.returncode}): {stderr}")
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}") from e


def _clean_stream_dict(stream: dict) -> dict:
    """Strip noisy/non-stable fields from the ffprobe stream dict (in place)."""
    for key in ("index", "nb_read_frames", "nb_read_packets"):
        stream.pop(key, None)
    return stream


def _validate_video(fps, duration, r_frames, default_start_time=0):
    """Verify ``r_frames`` matches ``fps * (duration - start_time)``.

    Returns ``(ok: bool, expected_frames: int, detail: dict)``. When values are
    missing the check is treated as a pass with an empty detail dict.
    """
    if not (fps and duration and r_frames):
        return True, 0, {}
    if default_start_time is None:
        default_start_time = 0

    exp_frames_count = fps * float(int((duration - default_start_time) * 100)) / 100
    full_exp_frames_count = fps * float(duration - default_start_time)
    rounded = round(exp_frames_count)
    rounded_up = math.floor(exp_frames_count) + 1

    if rounded == rounded_up or rounded == r_frames:
        exp_frames = rounded
    else:
        exp_frames = rounded_up

    if exp_frames != r_frames and abs(exp_frames_count - r_frames) > 0.5:
        return False, exp_frames, {
            "type": "origExpectedFrames",
            "message": "Frames is not equal to FPS * (Duration - StartTime)",
            "expected_frames": exp_frames,
            "actual_frames": r_frames,
            "delta": abs(full_exp_frames_count - r_frames),
        }
    return True, exp_frames, {}


# ---------------------------------------------------------------------------
# Service runner
# ---------------------------------------------------------------------------

class VideoPreprocess(dl.BaseServiceRunner):
    """Unified video preprocess: ffprobe metadata + OpenCV/imageio thumbnail."""

    def __init__(self):
        self.ignored_datasets = self._load_ignore_list()
        logger.info(
            "VideoPreprocess initialized: %d ignored datasets",
            len(self.ignored_datasets),
        )

    # ---- init helpers ----------------------------------------------------

    @staticmethod
    def _load_ignore_list() -> list:
        """Load per-service dataset ignore list from the DataloopTasks binaries dataset."""
        try:
            project = dl.projects.get(project_name="DataloopTasks")
            b_dataset = project.datasets._get_binaries_dataset()
            item = b_dataset.items.get(filepath="/preprocess_ignore_list.json")
            with open(item.download(), "r") as f:
                ignore_list = json.load(f)
            entries = ignore_list.get(IGNORE_LIST_SERVICE_KEY, [])
            return [
                e["dataset_id"]
                for e in entries
                if isinstance(e, dict) and "dataset_id" in e
            ]
        except dl.exceptions.NotFound:
            logger.warning("Ignore list file not found; using empty list")
            return []
        except Exception:
            logger.error("Failed loading ignore list:\n%s", traceback.format_exc())
            return []

    # ---- entry points ----------------------------------------------------

    def on_create(self, item: dl.Item, extract_metadata: bool = False,
                  extract_thumbnail: bool = False, thumbnail_size: int = DEFAULT_THUMB_SIZE,
                  max_file_size_mb: int = MAX_FILE_SIZE_MB,
                  progress: dl.Progress = None) -> dl.Item:
        """Main trigger entry point.

        Parameters are passed directly from trigger input with module-level
        constant fallbacks. Stages run in this order: metadata then
        thumbnail. Errors during a stage are recorded with ``record_etl_error``
        and re-raised so the service execution is marked failed.
        """
        extract_metadata = bool(extract_metadata)
        extract_thumbnail = bool(extract_thumbnail)
        thumbnail_size = int(thumbnail_size)
        max_file_size_mb = int(max_file_size_mb)

        logger.info(
            "on_create item=%s extract_metadata=%s extract_thumbnail=%s "
            "thumbnail_size=%d max_file_size_mb=%d",
            item.id, extract_metadata, extract_thumbnail, thumbnail_size, max_file_size_mb,
        )

        if extract_metadata and extract_thumbnail:
            logger.warning(
                "item=%s: conflicting flags (extract_metadata AND extract_thumbnail) — skipping",
                item.id,
            )
            return item

        do_metadata = not extract_thumbnail
        do_thumbnail = not extract_metadata
        if not do_metadata and not do_thumbnail:
            logger.info("item=%s: nothing to do (both stages disabled)", item.id)
            return item

        # Dataset-level skips.
        if item.datasetId in self.ignored_datasets:
            logger.info(
                "item=%s: dataset %s is in ignore list — skipping",
                item.id, item.datasetId,
            )
            return item
        try:
            dataset = item.dataset
            if dataset.metadata.get("system", {}).get(SKIP_DATASET_FLAG, False):
                logger.info(
                    "item=%s: dataset %s has %s=True — skipping",
                    item.id, item.datasetId, SKIP_DATASET_FLAG,
                )
                return item
        except Exception:
            # If dataset metadata isn't readable, proceed with processing.
            pass

        # File-size guard (uses platform-reported size to avoid downloading huge files).
        file_size = item.metadata.get("system", {}).get("size", 0) or 0
        if file_size and file_size > max_file_size_mb * 1024 * 1024:
            msg = f"File too large: {file_size} bytes exceeds {max_file_size_mb}MB limit"
            logger.error("item=%s: %s", item.id, msg)
            record_etl_error(item, "size_check", msg, failed=True)
            item.update(system_metadata=True)
            raise ValueError(msg)

        workdir = None
        try:
            workdir = item.id
            os.makedirs(workdir, exist_ok=True)
            filepath = item.download(local_path=workdir)

            if do_metadata:
                self._extract_and_write_metadata(item, filepath)
            if do_thumbnail:
                self._generate_thumbnail(item, filepath, workdir, thumbnail_size)

            return item
        except Exception as e:
            # Inner stages already recorded their own errors; this catches anything
            # else (download failure, OS errors, etc.) so the item shows the cause.
            if not item.metadata.get("system", {}).get("etl", {}).get("failed"):
                record_etl_error(item, "on_create", str(e), failed=True)
                try:
                    item.update(system_metadata=True)
                except Exception:
                    logger.exception("item=%s: failed to persist on_create error", item.id)
            raise
        finally:
            if workdir is not None and os.path.isdir(workdir):
                shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def on_delete(item: dl.Item) -> None:
        """Mirror of the legacy on_delete: clean up thumbnail item and webm modality."""
        log_header = f"[video-preprocess][on_delete][{item.id}]"

        thumbnail_id = item.metadata.get("system", {}).get("thumbnailId")
        if thumbnail_id is not None:
            logger.info("%s deleting thumbnail id=%s", log_header, thumbnail_id)
            try:
                dl.items.get(item_id=thumbnail_id).delete()
                logger.info("%s thumbnail deleted", log_header)
            except dl.exceptions.NotFound:
                logger.info("%s thumbnail already deleted", log_header)

        if "webm" not in (item.mimetype or ""):
            modalities = item.metadata.get("system", {}).get("modalities", []) or []
            expected_name = item.id + ".webm"
            webm_id = None
            for modality in modalities:
                if (modality.get("type") == "replace"
                        and modality.get("name") == expected_name):
                    webm_id = modality.get("ref")
                    break
            if webm_id is not None:
                logger.info("%s deleting webm id=%s", log_header, webm_id)
                try:
                    dl.items.delete(item_id=webm_id)
                    logger.info("%s webm deleted", log_header)
                except dl.exceptions.NotFound:
                    logger.info("%s webm already deleted", log_header)

    # ---- stage: metadata -------------------------------------------------

    def _extract_and_write_metadata(self, item: dl.Item, filepath: str) -> None:
        """Run ffprobe, populate item.metadata, validate frame count.

        On validation failure or any extraction error, records ``failed=True``
        on the item and re-raises so the service execution is marked failed.
        """
        try:
            probe = _run_ffprobe(filepath)
        except Exception as e:
            logger.exception("item=%s: ffprobe failed", item.id)
            record_etl_error(item, "metadata", str(e), failed=True)
            item.update(system_metadata=True)
            raise

        try:
            video_stream = next(
                (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
                None,
            )
            if video_stream is None:
                raise ValueError(f"video stream data is empty for: {filepath}")

            video_format = probe.get("format") or {}
            nb_streams = video_format.get("nb_streams", 1)

            start_time = _safe_parse_fraction(video_stream.get("start_time"))
            if start_time is None:
                start_time = 0

            height = video_stream.get("height")
            width = video_stream.get("width")

            fps = _safe_parse_fraction(
                video_stream.get("avg_frame_rate")
                or video_stream.get("r_frame_rate")
            )

            nb_frames_raw = video_stream.get("nb_frames")
            nb_frames = int(nb_frames_raw) if nb_frames_raw is not None else None

            nb_read_frames_raw = video_stream.get("nb_read_frames")
            nb_read_frames = (
                int(nb_read_frames_raw) if nb_read_frames_raw is not None else None
            )

            # Duration fallback chain: stream → stream.tags.DURATION → format.
            duration = _safe_parse_fraction(video_stream.get("duration"))
            if duration is None:
                duration = _duration_str_to_sec(
                    video_stream.get("tags", {}).get("DURATION")
                )
            if duration is None:
                duration = _safe_parse_fraction(video_format.get("duration"))

            # ffmpeg compat dict (cleaned stream).
            ffmpeg_dict = _clean_stream_dict(dict(video_stream))

            # Persist to item.metadata.system
            system = item.metadata.setdefault("system", {})
            system["ffmpeg"] = ffmpeg_dict
            system["format"] = video_format
            system["startTime"] = start_time
            if height is not None:
                system["height"] = height
            if width is not None:
                system["width"] = width
            if fps is not None:
                system["fps"] = fps
            if duration is not None:
                system["duration"] = float(duration)
            system["nb_frames"] = nb_frames
            system["nb_streams"] = nb_streams

            # Backward-compat top-level fields.
            item.metadata["startTime"] = start_time
            if fps is not None:
                item.metadata["fps"] = fps

            # Validation: prefer read frame count (more accurate when counted).
            r_frames = nb_read_frames if nb_read_frames is not None else nb_frames
            ok, exp_frames, detail = _validate_video(
                fps=fps,
                duration=duration,
                r_frames=r_frames,
                default_start_time=start_time,
            )
            if not ok:
                err_msg = (
                    f"frames validation failed: expected={exp_frames} actual={r_frames}"
                )
                logger.error("item=%s: %s", item.id, err_msg)
                record_etl_error(item, "validation", err_msg, failed=True)
                item.update(system_metadata=True)
                raise ValueError(err_msg)

            # Required-field check (catches missing critical metadata).
            missing = [
                k for k in VALIDATION_KEYS
                if not system.get(k) and k != "ffmpeg"
            ]
            if "ffmpeg" not in system or not system["ffmpeg"]:
                missing.append("ffmpeg")
            if missing:
                err_msg = f"missing metadata values: {missing}"
                logger.error("item=%s: %s", item.id, err_msg)
                record_etl_error(item, "metadata", err_msg, failed=True)
                item.update(system_metadata=True)
                raise ValueError(err_msg)

            item.update(system_metadata=True)
            logger.info(
                "item=%s metadata persisted: %dx%d fps=%s duration=%s nb_frames=%s",
                item.id, width or 0, height or 0, fps, duration, nb_frames,
            )
        except Exception as e:
            # Avoid double-recording validation errors (already recorded above).
            if not item.metadata.get("system", {}).get("etl", {}).get("failed"):
                logger.exception("item=%s: metadata stage failed", item.id)
                record_etl_error(item, "metadata", str(e), failed=True)
                try:
                    item.update(system_metadata=True)
                except Exception:
                    logger.exception("item=%s: failed to persist metadata error", item.id)
            raise

    # ---- stage: thumbnail ------------------------------------------------

    def _generate_thumbnail(self, item: dl.Item, filepath: str, workdir: str,
                            thumbnail_size: int) -> None:
        """Create a short GIF preview and upload it to /.dataloop/thumbnails."""
        try:
            file_size = (
                os.path.getsize(filepath) if os.path.isfile(filepath) else 0
            )
            if file_size > MAX_GEN_THUMB_SIZE_MB * 1024 * 1024:
                msg = (
                    f"File too large for thumbnail: {file_size} bytes "
                    f"exceeds {MAX_GEN_THUMB_SIZE_MB}MB limit"
                )
                logger.error("item=%s: %s", item.id, msg)
                record_etl_error(item, "thumbnail_size", msg, failed=True)
                item.update(system_metadata=True)
                raise ValueError(msg)

            gif_filepath = os.path.join(workdir, f"{item.id}.gif")
            self._build_gif(filepath, gif_filepath, thumbnail_size)

            dataset = dl.datasets.get(dataset_id=item.datasetId, fetch=False)
            thumbnail_item = dataset.items.upload(
                local_path=gif_filepath,
                remote_path="/.dataloop/thumbnails",
                remote_name=f"{item.id}.gif",
                overwrite=True,
                item_metadata={"system": {"thumbnailOf": item.id}},
            )

            item.metadata.setdefault("system", {})["thumbnailId"] = thumbnail_item.id
            item.update(system_metadata=True)
            logger.info(
                "item=%s thumbnail uploaded: id=%s",
                item.id, thumbnail_item.id,
            )
        except Exception as e:
            if not item.metadata.get("system", {}).get("etl", {}).get("failed"):
                logger.exception("item=%s: thumbnail stage failed", item.id)
                record_etl_error(item, "thumbnail", str(e), failed=True)
                try:
                    item.update(system_metadata=True)
                except Exception:
                    logger.exception("item=%s: failed to persist thumbnail error", item.id)
            raise

    @staticmethod
    def _build_gif(src_filepath: str, dst_filepath: str, thumb_size: int) -> None:
        """Decode the first ``THUMB_DURATION_SEC`` seconds and write a GIF."""
        cap = cv2.VideoCapture(src_filepath)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture failed to open: {src_filepath}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps <= 0:
                raise RuntimeError(f"invalid fps reported by OpenCV: {fps}")

            length_sec = min(THUMB_DURATION_SEC, frame_count / fps) if frame_count else THUMB_DURATION_SEC
            new_fps = fps * THUMB_SPEED_FACTOR
            target_frame_count = max(int(length_sec * new_fps), 1)

            # Sampling interval ensures we get roughly ``target_frame_count`` frames
            # from within the first ``length_sec`` seconds of source video.
            source_frames_in_window = max(int(length_sec * fps), 1)
            interval = max(source_frames_in_window // target_frame_count, 1)

            frames_rgb = []
            i = 0
            while len(frames_rgb) < target_frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                if i % interval == 0:
                    resized = _resize_keep_aspect(frame, thumb_size)
                    frames_rgb.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                i += 1

            if not frames_rgb:
                raise RuntimeError("no frames decoded for thumbnail")

            imageio.mimsave(dst_filepath, frames_rgb, fps=max(new_fps, 1.0))
        finally:
            cap.release()


def _resize_keep_aspect(frame, longest_edge: int):
    """Resize ``frame`` so the longest edge equals ``longest_edge`` (aspect preserved)."""
    h, w = frame.shape[:2]
    if max(h, w) <= longest_edge:
        return frame
    if w >= h:
        new_w = longest_edge
        new_h = max(int(round(h * (longest_edge / w))), 1)
    else:
        new_h = longest_edge
        new_w = max(int(round(w * (longest_edge / h))), 1)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
