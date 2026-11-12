from __future__ import annotations

import hashlib

from frame_proxy.config import Settings
from frame_proxy.domain.errors import (
    ApplicationError,
    ErrorCode,
    MediaValidationError,
)
from frame_proxy.domain.models import InspectionRecord, SceneRecord
from frame_proxy.domain.policies import (
    InspectionRequest,
    validate_clip_id,
    validate_identifier,
)
from frame_proxy.domain.transitions import transition_inspection
from frame_proxy.media.avi import probe_avi
from frame_proxy.media.detector import IncrementalSceneAnalyzer
from frame_proxy.media.fingerprint import fingerprint_frame
from frame_proxy.media.frames import decode_frame_at
from frame_proxy.storage.repository import FileRepository


class InspectionService:
    def __init__(self, repository: FileRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    def create(
        self,
        clip_id: str,
        request: InspectionRequest,
    ) -> InspectionRecord:
        normalized_clip_id = validate_clip_id(clip_id)
        clip = self._repository.get_clip(normalized_clip_id)
        if clip.state != "registered":
            raise MediaValidationError(
                "clip is not ready for inspection",
                code=ErrorCode.INVALID_STATE,
            )

        record = InspectionRecord(
            inspection_id=request.inspection_id,
            clip_id=clip.clip_id,
            state="pending",
            sensitivity=request.sensitivity,
            sample_stride=request.sample_stride,
            min_scene_frames=request.min_scene_frames,
        )
        self._repository.create_inspection(record)

        transition_inspection(record, "scanning", reason="frame scan started")
        self._repository.update_inspection(record)
        analyzer: IncrementalSceneAnalyzer | None = None
        try:
            media = self._repository.get_clip_media(clip.clip_id)
            if hashlib.sha256(media).hexdigest() != clip.digest:
                raise MediaValidationError(
                    "stored clip digest does not match its record",
                    code=ErrorCode.FRAME_DECODE,
                )
            descriptor = probe_avi(media)
            analyzer = IncrementalSceneAnalyzer(
                descriptor.frame_count,
                sensitivity=request.sensitivity,
                min_scene_frames=request.min_scene_frames,
            )
            for frame_index in range(
                0,
                descriptor.frame_count,
                request.sample_stride,
            ):
                fingerprint = fingerprint_frame(
                    decode_frame_at(media, descriptor, frame_index)
                )
                for scene in analyzer.feed(frame_index, fingerprint):
                    self._deliver_scene(record, scene)
            result = analyzer.finish()
            for scene in result.scenes[len(record.scenes):]:
                self._deliver_scene(record, scene)
            record.sampled_frames = list(result.sampled_frames)
            record.cut_frames = list(result.cut_frames)
            transition_inspection(record, "indexed", reason="scene ranges verified")
            self._repository.update_inspection(record)
            return record
        except ApplicationError as exc:
            self._abandon_scan(record, analyzer)
            self._mark_failed(record, exc.code.value, exc.message)
            raise
        except Exception as exc:
            self._abandon_scan(record, analyzer)
            self._mark_failed(record, ErrorCode.INTERNAL.value, str(exc))
            raise ApplicationError(
                ErrorCode.INTERNAL,
                "shot indexing failed",
                status=500,
            ) from exc

    def _deliver_scene(
        self,
        record: InspectionRecord,
        scene: SceneRecord,
    ) -> None:
        """Hand an immutable scene to result collection as soon as it is final.

        Finalized scenes are appended to the record immediately, so the
        analysis session does not keep the fingerprints behind them.  A later
        failure still moves the record to `failed`, and `_abandon_scan`
        discards results delivered by an analysis that never completes.
        """
        record.scenes.append(scene)

    def _abandon_scan(
        self,
        record: InspectionRecord,
        analyzer: IncrementalSceneAnalyzer | None,
    ) -> None:
        """Release session work and drop results from a failed analysis.

        This preserves the failed-inspection HTTP shape: a failed record
        carries its error code and no partial scene or sample data.
        """
        if analyzer is not None:
            analyzer.abort()
        record.scenes = []
        record.cut_frames = []
        record.sampled_frames = []

    def _mark_failed(
        self,
        record: InspectionRecord,
        code: str,
        message: str,
    ) -> None:
        record.error_code = code
        record.error_message = message
        if record.state in {"pending", "scanning"}:
            transition_inspection(record, "failed", reason=code)
        try:
            self._repository.update_inspection(record)
        except Exception:
            pass

    def get(self, inspection_id: str) -> InspectionRecord:
        normalized = validate_identifier(inspection_id, "inspection_id")
        return self._repository.get_inspection(normalized)
