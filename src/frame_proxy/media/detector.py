from __future__ import annotations

from dataclasses import dataclass

from frame_proxy.domain.errors import ErrorCode, MediaValidationError
from frame_proxy.domain.models import SceneRecord
from frame_proxy.media.fingerprint import FrameFingerprint, fingerprint_distance


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    sampled_frames: tuple[int, ...]
    cut_frames: tuple[int, ...]
    scenes: tuple[SceneRecord, ...]


def _threshold(sensitivity: float) -> float:
    return 0.50 - sensitivity * 0.30


def _nearest_sample(
    midpoint: int,
    left_sample: tuple[int, FrameFingerprint] | None,
    right_sample: tuple[int, FrameFingerprint] | None,
) -> tuple[int, FrameFingerprint]:
    nearest = left_sample
    if nearest is None:
        nearest = right_sample
    elif right_sample is not None and (
        abs(right_sample[0] - midpoint) < abs(nearest[0] - midpoint)
    ):
        nearest = right_sample
    if nearest is None:
        raise MediaValidationError(
            "sampled frames and fingerprints must be non-empty and aligned",
            code=ErrorCode.FRAME_DECODE,
        )
    return nearest


class IncrementalSceneAnalyzer:
    """Consume sampled fingerprints one at a time and emit final scenes.

    A scene is decided as soon as the scan accepts the cut that opens the next
    scene: no later sample can fall inside it, so its range, statistics and
    representative frame are fixed.  The analyzer immediately hands that scene
    back to its caller and drops the fingerprints behind it.

    Only the one sample needed for the next distance check is retained as a
    fingerprint.  The scene still in progress keeps just its sampled frame
    indices and running aggregates, so analysis state does not grow with the
    number of samples already scanned, let alone with delivered scenes.

    Every instance owns its state.  An analysis that is aborted or garbage
    collected cannot mutate another session's results.
    """

    def __init__(
        self,
        frame_count: int,
        *,
        sensitivity: float,
        min_scene_frames: int,
    ) -> None:
        if frame_count <= 0:
            raise MediaValidationError(
                "clip frame count must be positive",
                code=ErrorCode.MEDIA_BOUNDS,
            )
        if min_scene_frames <= 0:
            raise MediaValidationError(
                "minimum scene length must be positive",
                code=ErrorCode.INTERNAL,
            )
        self._frame_count = frame_count
        self._threshold = _threshold(sensitivity)
        self._min_scene_frames = min_scene_frames
        self._sampled_frames: list[int] = []
        self._cut_frames: list[int] = []
        self._scenes: list[SceneRecord] = []
        self._last_cut = 0
        self._open_frames: list[int] = []
        self._open_count = 0
        self._luma_total = 0.0
        self._contrast_total = 0.0
        self._red_total = 0.0
        self._green_total = 0.0
        self._blue_total = 0.0
        self._previous: tuple[int, FrameFingerprint] | None = None
        self._closed = False

    @property
    def retained_fingerprint_count(self) -> int:
        """Fingerprints still held by the session (never inside emitted scenes)."""
        return 1 if self._previous is not None else 0

    def feed(
        self,
        frame_index: int,
        fingerprint: FrameFingerprint,
    ) -> tuple[SceneRecord, ...]:
        """Advance the scan by one sample and return newly final scenes."""
        if self._closed:
            raise MediaValidationError(
                "analysis session is already closed",
                code=ErrorCode.INVALID_STATE,
            )
        if frame_index < 0 or frame_index >= self._frame_count:
            raise MediaValidationError(
                "frame index is outside the clip",
                code=ErrorCode.MEDIA_BOUNDS,
                details={"frame_index": frame_index},
            )
        if self._previous is not None and frame_index <= self._previous[0]:
            raise MediaValidationError(
                "sampled frames must increase monotonically",
                code=ErrorCode.FRAME_DECODE,
            )

        emitted: SceneRecord | None = None
        if self._previous is not None:
            distance = fingerprint_distance(self._previous[1], fingerprint)
            if (
                distance >= self._threshold
                and frame_index - self._last_cut >= self._min_scene_frames
                and self._frame_count - frame_index >= self._min_scene_frames
            ):
                emitted = self._close_scene(
                    end=frame_index,
                    left_sample=self._previous,
                    right_sample=(frame_index, fingerprint),
                )
                self._cut_frames.append(frame_index)
                self._last_cut = frame_index

        self._open_frames.append(frame_index)
        self._open_count += 1
        self._luma_total += fingerprint.mean_luma
        self._contrast_total += fingerprint.contrast
        self._red_total += fingerprint.mean_red
        self._green_total += fingerprint.mean_green
        self._blue_total += fingerprint.mean_blue
        self._previous = (frame_index, fingerprint)
        self._sampled_frames.append(frame_index)
        return (emitted,) if emitted is not None else ()

    def finish(self) -> AnalysisResult:
        """Close the trailing scene and return the complete analysis."""
        if self._closed:
            raise MediaValidationError(
                "analysis session is already closed",
                code=ErrorCode.INVALID_STATE,
            )
        if not self._sampled_frames:
            raise MediaValidationError(
                "sampled frames and fingerprints must be non-empty and aligned",
                code=ErrorCode.FRAME_DECODE,
            )
        self._close_scene(
            end=self._frame_count,
            left_sample=self._previous,
            right_sample=None,
        )
        self._closed = True
        self._previous = None
        return AnalysisResult(
            sampled_frames=tuple(self._sampled_frames),
            cut_frames=tuple(self._cut_frames),
            scenes=tuple(self._scenes),
        )

    def abort(self) -> None:
        """Release retained work without emitting any further scene."""
        self._closed = True
        self._previous = None
        self._open_frames.clear()
        self._open_count = 0

    def _close_scene(
        self,
        *,
        end: int,
        left_sample: tuple[int, FrameFingerprint] | None,
        right_sample: tuple[int, FrameFingerprint] | None,
    ) -> SceneRecord:
        start = self._last_cut
        midpoint = start + (end - start) // 2
        if self._open_count:
            representative_frame = min(
                self._open_frames,
                key=lambda frame_index: abs(frame_index - midpoint),
            )
            count = self._open_count
            mean_luma = self._luma_total / count
            contrast = self._contrast_total / count
            dominant = (
                round(self._red_total / count),
                round(self._green_total / count),
                round(self._blue_total / count),
            )
        else:
            # No sampled frame fell inside this range; fall back to the nearest
            # boundary sample, matching the batch analyzer.
            representative_frame = midpoint
            nearest = _nearest_sample(midpoint, left_sample, right_sample)
            fingerprint = nearest[1]
            mean_luma = fingerprint.mean_luma
            contrast = fingerprint.contrast
            dominant = (
                round(fingerprint.mean_red),
                round(fingerprint.mean_green),
                round(fingerprint.mean_blue),
            )

        scene = SceneRecord(
            index=len(self._scenes),
            start_frame=start,
            end_frame=end,
            representative_frame=representative_frame,
            mean_luma=mean_luma,
            contrast=contrast,
            dominant_rgb=dominant,
        )
        self._scenes.append(scene)
        self._open_frames.clear()
        self._open_count = 0
        self._luma_total = 0.0
        self._contrast_total = 0.0
        self._red_total = 0.0
        self._green_total = 0.0
        self._blue_total = 0.0
        return scene


def analyze_frames(
    frame_count: int,
    sampled_frames: tuple[int, ...],
    fingerprints: tuple[FrameFingerprint, ...],
    *,
    sensitivity: float,
    min_scene_frames: int,
) -> AnalysisResult:
    """Batch entry point kept for callers that already hold all samples."""
    if len(sampled_frames) != len(fingerprints) or not sampled_frames:
        raise MediaValidationError(
            "sampled frames and fingerprints must be non-empty and aligned",
            code=ErrorCode.FRAME_DECODE,
        )
    analyzer = IncrementalSceneAnalyzer(
        frame_count,
        sensitivity=sensitivity,
        min_scene_frames=min_scene_frames,
    )
    for frame_index, fingerprint in zip(sampled_frames, fingerprints):
        analyzer.feed(frame_index, fingerprint)
    return analyzer.finish()
