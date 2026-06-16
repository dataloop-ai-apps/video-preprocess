"""Video Preprocess service: metadata extraction + thumbnail generation.

Single-file Dataloop service runner that:
  * Extracts video metadata via PyAV (no ffprobe subprocess).
  * Generates an animated GIF thumbnail from the first few seconds of the video
    via OpenCV + imageio.
  * Records any failure to ``item.metadata.system.etl`` and re-raises so the
    service execution shows as failed and the user can see the cause in both
    the item metadata and the service logs.

The two booleans ``EXTRACT_METADATA`` and ``EXTRACT_THUMBNAIL`` control which
stages run. If both are False the item is skipped.
"""

import logging
import math
import os
import shutil
import traceback
from fractions import Fraction

import av
import cv2
import dtlpy as dl
import imageio
import numpy as np

logger = logging.getLogger("video_preprocess")

# ---------------------------------------------------------------------------
# Configuration (module-level constants — change here to alter behaviour)
# ---------------------------------------------------------------------------

# Stage toggles. If both are False the service skips the item.
EXTRACT_METADATA = True
EXTRACT_THUMBNAIL = True

# Thumbnail config.
MAX_GEN_THUMB_SIZE_MB = 70       # skip thumbnail if the file is larger than this
DEFAULT_THUMB_SIZE = 128         # longest edge of the GIF in pixels; aspect preserved
THUMB_DURATION_SEC = 3.0         # how many seconds of video to sample for the GIF

# Backward-compat fields written outside ``system`` (kept from the legacy code).
_VALIDATION_FIELDS = ('ffmpeg', 'height', 'width', 'fps', 'duration')


# ---------------------------------------------------------------------------
# ETL error helper (modelled on image-preprocess-app/common/etl_errors.py)
# ---------------------------------------------------------------------------

def record_etl_error(item: dl.Item, stage: str, error: str,
                     failed: bool = False) -> list:
    """Append an entry to ``system.etl.errors`` on the item.

    When ``failed=True`` the helper also sets ``system.etl.failed = True`` and
    mirrors the failed block under ``system.videoEtl.etl`` so downstream
    consumers can filter on the service that produced the failure.

    Returns the etl errors list so callers can keep using the same reference.
    The caller is responsible for persisting via ``item.update(system_metadata=True)``.
    """
    system = item.metadata.setdefault('system', {})
    etl = system.setdefault('etl', {})
    etl_errors = etl.setdefault('errors', [])
    entry = {'stage': stage, 'error': error}
    etl_errors.append(entry)
    if failed:
        etl['failed'] = True
        system.setdefault('videoEtl', {})['etl'] = {
            'failed': True,
            'errors': etl_errors,
        }
    return etl_errors


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _duration_str_to_sec(time_str):
    """Parse ``HH:MM:SS[.ms]`` strings (as found in some container tags)."""
    if time_str is None:
        return None
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        logger.warning("Unsupported duration string: %r", time_str)
        return None


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fraction_to_float(value):
    """Convert PyAV's ``Fraction`` (or any number) to a float; None on failure."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        if value.denominator == 0:
            return None
        return float(value)
    return _safe_float(value)


# ---------------------------------------------------------------------------
# PyAV → ffprobe-compatible dicts (kept so downstream consumers of
# ``metadata.system.ffmpeg`` / ``metadata.system.format`` keep working)
# ---------------------------------------------------------------------------

def _ratio_str(value):
    """Format a rational number the way ffprobe does in its JSON output.

    PyAV exposes frame rates and time bases as :class:`fractions.Fraction`
    objects (e.g. ``Fraction(30000, 1001)``), but ffprobe emits them as the
    string ``"num/den"`` (e.g. ``"30000/1001"``). Downstream consumers of
    ``metadata.system.ffmpeg`` expect that string form, so this helper:

    * returns ``None`` for ``None`` (so callers can drop the field entirely),
    * formats a ``Fraction`` as ``"<numerator>/<denominator>"``,
    * falls back to ``str(value)`` for plain ints/floats.
    """
    if value is None:
        return None
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    return str(value)


def _build_stream_dict(video_stream) -> dict:
    """Build an ffprobe-style dict describing a single video stream.

    The legacy service used to shell out to ``ffprobe`` and store the first
    entry of its ``streams[]`` array on the item under
    ``metadata.system.ffmpeg``. This service uses PyAV instead, so we rebuild
    the same dict by hand from a PyAV :class:`av.video.stream.VideoStream`
    object.

    Field mapping (PyAV source → ffprobe key):

    * ``codec_context.name`` / ``.long_name`` / ``.pix_fmt`` →
      ``codec_name`` / ``codec_long_name`` / ``pix_fmt``.
    * ``video_stream.width`` / ``.height`` → ``width`` / ``height``.
    * ``video_stream.average_rate`` → ``avg_frame_rate`` (``"num/den"``).
    * ``video_stream.base_rate`` → ``r_frame_rate`` (``"num/den"``).
    * ``video_stream.time_base`` → ``time_base`` (``"num/den"``).
    * ``start_time`` / ``duration``: PyAV stores these as integer ticks in
      the stream's ``time_base`` units, so we multiply by ``time_base`` to
      get seconds, then stringify (ffprobe emits these as strings).
    * ``video_stream.frames`` → ``nb_frames`` (stringified, omitted if 0).
    * ``video_stream.metadata`` → ``tags`` (container-level stream tags).

    Notes:
    * ``index``, ``nb_read_frames`` and ``nb_read_packets`` are intentionally
      omitted to match the Rubiks spec — those fields are either
      meaningless without a full decode pass or not part of the contract.
    * Any field that resolves to ``None`` is dropped from the returned dict,
      matching ffprobe's behaviour of omitting unknown fields rather than
      emitting nulls.
    """
    codec_ctx = video_stream.codec_context
    avg_rate = video_stream.average_rate
    base_rate = video_stream.base_rate

    start_time_sec = None
    if video_stream.start_time is not None and video_stream.time_base is not None:
        start_time_sec = float(video_stream.start_time * video_stream.time_base)

    duration_sec = None
    if video_stream.duration is not None and video_stream.time_base is not None:
        duration_sec = float(video_stream.duration * video_stream.time_base)

    stream = {
        'codec_name': getattr(codec_ctx, 'name', None),
        'codec_long_name': getattr(codec_ctx, 'long_name', None),
        'codec_type': 'video',
        'width': video_stream.width,
        'height': video_stream.height,
        'pix_fmt': getattr(codec_ctx, 'pix_fmt', None),
        'avg_frame_rate': _ratio_str(avg_rate),
        'r_frame_rate': _ratio_str(base_rate),
        'time_base': _ratio_str(video_stream.time_base),
        'start_time': str(start_time_sec) if start_time_sec is not None else None,
        'duration': str(duration_sec) if duration_sec is not None else None,
        'nb_frames': str(video_stream.frames) if video_stream.frames else None,
        'tags': dict(video_stream.metadata or {}),
    }
    # Drop None values to keep parity with ffprobe which omits unknown fields.
    return {k: v for k, v in stream.items() if v is not None}


def _build_format_dict(container, filepath: str) -> dict:
    """Build an ffprobe-style dict describing the media container.

    Where :func:`_build_stream_dict` describes a single elementary stream
    (the video track), this helper describes the **container** that wraps
    all streams — the equivalent of ffprobe's top-level ``format`` object.
    The result is stored on the item under ``metadata.system.format`` so
    legacy consumers of that key keep working after the migration from a
    subprocess ``ffprobe`` call to PyAV.

    Field mapping (PyAV source → ffprobe key):

    * ``filepath`` → ``filename`` (path the container was opened from).
    * ``len(container.streams)`` → ``nb_streams`` (total stream count, not
      just video; includes audio, subtitles, data, etc.).
    * ``container.format.name`` / ``.long_name`` → ``format_name`` /
      ``format_long_name`` (e.g. ``"mov,mp4,m4a,3gp,3g2,mj2"``).
    * ``container.duration`` → ``duration``. PyAV stores container
      duration in microseconds via the global ``av.time_base`` (``1e6``),
      not the per-stream ``time_base`` used in :func:`_build_stream_dict`,
      so we divide instead of multiply to get seconds.
    * ``container.bit_rate`` → ``bit_rate`` (overall bitrate, bits/sec).
    * ``container.size`` → ``size`` (file size in bytes, if available).
    * ``container.metadata`` → ``tags`` (container-level tags such as
      ``title``, ``encoder``, ``creation_time``).

    All numeric fields are stringified to match ffprobe's JSON output, and
    keys whose value is ``None`` / falsy are dropped so the resulting dict
    mirrors ffprobe's "omit unknown fields" behaviour.
    """
    duration_sec = None
    if container.duration is not None:
        duration_sec = container.duration / av.time_base

    fmt = {
        'filename': filepath,
        'nb_streams': len(container.streams),
        'format_name': getattr(container.format, 'name', None),
        'format_long_name': getattr(container.format, 'long_name', None),
        'duration': str(duration_sec) if duration_sec is not None else None,
        'bit_rate': str(container.bit_rate) if container.bit_rate else None,
        'size': str(container.size) if getattr(container, 'size', None) else None,
        'tags': dict(container.metadata or {}),
    }
    return {k: v for k, v in fmt.items() if v is not None}


# ---------------------------------------------------------------------------
# Service runner
# ---------------------------------------------------------------------------

class VideoPreprocess(dl.BaseServiceRunner):

    # ---------------------------------------------------------------- entry points
    def on_create(self, item: dl.Item):
        """Run metadata extraction and/or thumbnail generation on a video item."""

        if not EXTRACT_METADATA and not EXTRACT_THUMBNAIL:
            logger.info('Both stages disabled — skipping item %s', item.id)
            return item

        workdir = item.id
        os.makedirs(workdir, exist_ok=True)
        try:
            filepath = item.download(local_path=workdir)
            if EXTRACT_METADATA:
                item = self._extract_and_write_metadata(item=item, filepath=filepath)
            if EXTRACT_THUMBNAIL:
                item = self._generate_thumbnail(item=item, filepath=filepath, workdir=workdir)
            return item
        except Exception as e:
            # Already recorded by the inner stage handlers, but make sure the
            # top-level failure is visible too.
            tb = traceback.format_exc()
            logger.error('on_create failed for %s: %s\n%s', item.id, e, tb)
            record_etl_error(item, stage='on_create', error=str(e),
                             failed=True)
            item.update(system_metadata=True)
            raise
        finally:
            if os.path.isdir(workdir):
                shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def on_delete(item: dl.Item):
        """Clean up the thumbnail item and any webm modality on item delete."""
        header = f'[video-preprocess][on_delete][{item.id}]'
        thumbnail_id = item.metadata.get('system', {}).get('thumbnailId')
        if thumbnail_id is not None:
            logger.info('%s deleting thumbnail id %s', header, thumbnail_id)
            try:
                dl.items.get(item_id=thumbnail_id).delete()
            except dl.exceptions.NotFound:
                logger.info('%s thumbnail already deleted', header)

        if 'webm' not in (item.mimetype or ''):
            modalities = item.metadata.get('system', {}).get('modalities', []) or []
            expected_name = item.id + '.webm'
            for modality in modalities:
                if (modality.get('type') == 'replace'
                        and modality.get('name') == expected_name):
                    webm_id = modality.get('ref')
                    if webm_id is None:
                        continue
                    logger.info('%s deleting webm id %s', header, webm_id)
                    try:
                        dl.items.delete(item_id=webm_id)
                    except dl.exceptions.NotFound:
                        logger.info('%s webm already deleted', header)
                    break

    # ----------------------------------------------------- metadata extraction
    def _extract_and_write_metadata(self, item: dl.Item, filepath: str) -> dl.Item:
        try:
            metadata = self._extract_metadata(filepath=filepath)
        except Exception as e:
            record_etl_error(item, stage='metadata', error=str(e),
                             failed=True)
            item.update(system_metadata=True)
            raise

        return self._write_metadata_to_item(item=item, metadata=metadata)

    @staticmethod
    def _extract_metadata(filepath: str) -> dict:
        """Read all needed metadata via PyAV."""
        container = av.open(filepath)
        try:
            if not container.streams.video:
                raise ValueError('video stream data is empty')
            video_stream = container.streams.video[0]

            start_time = None
            if video_stream.start_time is not None and video_stream.time_base is not None:
                try:
                    start_time = float(video_stream.start_time * video_stream.time_base)
                except Exception:
                    start_time = None
            if start_time is None:
                start_time = 0.0

            height = video_stream.height
            width = video_stream.width
            fps = _fraction_to_float(video_stream.average_rate)

            # Duration: stream → tags.DURATION → container.duration
            duration = None
            if video_stream.duration is not None and video_stream.time_base is not None:
                try:
                    duration = float(video_stream.duration * video_stream.time_base)
                except Exception:
                    duration = None
            if duration is None:
                duration = _duration_str_to_sec(
                    (video_stream.metadata or {}).get('DURATION'))
            if duration is None and container.duration is not None:
                duration = container.duration / av.time_base

            nb_frames = video_stream.frames or None
            nb_streams = len(container.streams) or 1

            stream_dict = _build_stream_dict(video_stream)
            format_dict = _build_format_dict(container, filepath)

            return {
                'ffmpeg': stream_dict,
                'format': format_dict,
                'startTime': start_time,
                'height': height,
                'width': width,
                'fps': fps,
                'duration': duration,
                'nb_frames': nb_frames,
                'nb_streams': nb_streams,
            }
        finally:
            container.close()

    def _write_metadata_to_item(self, item: dl.Item, metadata: dict) -> dl.Item:
        system = item.metadata.setdefault('system', {})

        if 'ffmpeg' in metadata:
            system['ffmpeg'] = metadata['ffmpeg']
        if 'format' in metadata:
            system['format'] = metadata['format']

        system['startTime'] = metadata.get('startTime', 0)
        # Backward compat fields outside system.
        item.metadata['startTime'] = system['startTime']

        system['height'] = metadata.get('height')
        system['width'] = metadata.get('width')
        system['fps'] = metadata.get('fps')
        item.metadata['fps'] = system['fps']

        system['duration'] = metadata.get('duration')
        # Always include nb_frames (nullable) to match the Rubiks spec.
        system['nb_frames'] = metadata.get('nb_frames')
        system['nb_streams'] = metadata.get('nb_streams', 1)

        # Run validation. On mismatch: record + raise.
        ok, exp_frames, _ = self._validate_video(
            fps=metadata.get('fps'),
            duration=metadata.get('duration'),
            r_frames=metadata.get('nb_frames'),
            default_start_time=metadata.get('startTime'),
        )
        if not ok:
            record_etl_error(item,
                             stage='validation',
                             error='frames count does not match fps * duration',
                             failed=True)
            item.update(system_metadata=True)
            raise ValueError(
                'frames validation failed: expected={} actual={}'.format(
                    exp_frames, metadata.get('nb_frames')))

        # Check that the core values we depend on are populated.
        missing = [k for k in _VALIDATION_FIELDS if not system.get(k)]
        if missing:
            record_etl_error(item,
                             stage='metadata',
                             error='missing metadata values: {}'.format(missing),
                             failed=True)
            item.update(system_metadata=True)
            raise ValueError('missing metadata values: {}'.format(missing))

        return item.update(system_metadata=True)

    @staticmethod
    def _validate_video(fps, duration, r_frames, default_start_time=0):
        """Validate frame count against fps * (duration - start_time).

        Returns ``(ok, expected_frames, detail_dict)``.
        ``ok=True`` when any of the inputs is missing — we only flag mismatches
        when we have all three values.
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
                'delta': abs(full_exp_frames_count - r_frames),
            }
        return True, exp_frames, {}

    # ----------------------------------------------------- thumbnail generation
    def _generate_thumbnail(self, item: dl.Item, filepath: str, workdir: str) -> dl.Item:
        gif_filepath = os.path.join(workdir, '{}.gif'.format(item.id))
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > MAX_GEN_THUMB_SIZE_MB:
                msg = ('file size {:.2f}MB exceeds MAX_GEN_THUMB_SIZE_MB={}'
                       .format(size_mb, MAX_GEN_THUMB_SIZE_MB))
                record_etl_error(item, stage='thumbnail_size', error=msg,
                                 failed=True)
                item.update(system_metadata=True)
                raise ValueError(msg)

            self._build_gif(filepath=filepath, output_filepath=gif_filepath)

            dataset = dl.datasets.get(dataset_id=item.datasetId, fetch=False)
            thumbnail_item = dataset.items.upload(
                local_path=gif_filepath,
                remote_path='/.dataloop/thumbnails',
                overwrite=True,
            )

            # Mark the thumbnail as belonging to the source item.
            thumb_sys = thumbnail_item.metadata.setdefault('system', {})
            thumb_sys['thumbnailOf'] = item.id
            try:
                thumbnail_item.update(system_metadata=True)
            except Exception:
                logger.exception('Failed to set thumbnailOf on %s', thumbnail_item.id)

            item.metadata.setdefault('system', {})['thumbnailId'] = thumbnail_item.id
            return item.update(system_metadata=True)
        except Exception as e:
            # ``thumbnail_size`` already recorded above; only record here when we
            # don't already have an etl entry for this run.
            if not isinstance(e, ValueError) or 'MAX_GEN_THUMB_SIZE_MB' not in str(e):
                record_etl_error(item, stage='thumbnail', error=str(e),
                                 failed=True)
                item.update(system_metadata=True)
            raise

    @staticmethod
    def _build_gif(filepath: str, output_filepath: str) -> None:
        """Decode the first ``THUMB_DURATION_SEC`` of video and write a GIF.

        Resizes each frame so the longest edge equals ``DEFAULT_THUMB_SIZE``,
        preserving aspect ratio (no square crop / stretch).
        """
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise RuntimeError('cv2.VideoCapture could not open: {}'.format(filepath))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

            if fps <= 0:
                raise RuntimeError('invalid fps from cv2: {}'.format(fps))

            length_sec = min(THUMB_DURATION_SEC, frame_count / fps) if frame_count else THUMB_DURATION_SEC
            new_fps = max(fps * 0.25, 1.0)
            frames_in_window = max(int(length_sec * new_fps), 1)

            # Stride so we end up with roughly ``frames_in_window`` frames.
            target_total = max(frame_count * 0.1, frames_in_window) if frame_count else frames_in_window
            interval = round(max(min(int(target_total / frames_in_window),
                                     fps / 3), 4))
            interval = max(int(interval), 1)

            frames = []
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret or len(frames) >= frames_in_window:
                    break
                if i % interval == 0:
                    resized = _resize_keep_aspect(frame, DEFAULT_THUMB_SIZE)
                    frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                i += 1
        finally:
            cap.release()

        if not frames:
            raise RuntimeError('no frames decoded for thumbnail')

        imageio.mimsave(output_filepath, frames, fps=new_fps)


def _resize_keep_aspect(frame: np.ndarray, max_edge: int) -> np.ndarray:
    """Resize ``frame`` so the longest edge equals ``max_edge``.

    Aspect ratio is preserved; small frames are still scaled down only if their
    longest edge exceeds ``max_edge``.
    """
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return frame
    scale = max_edge / float(longest)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
