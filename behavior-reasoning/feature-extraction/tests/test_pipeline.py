import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_features.pipeline import PipelineConfig, run_pipeline


class FakeDescriber:
    def describe_segment(self, *, frame_paths, segment):
        self.last_frame_count = len(frame_paths)
        return {
            "scene_description": "A person is visible in a simple indoor scene.",
            "scene_confidence": "medium",
            "uncertainty": "synthetic test frames contain limited detail",
            "objects": [
                {
                    "label": "chair",
                    "count": 1,
                    "confidence": 0.7,
                    "description": "visible near the person",
                }
            ],
            "characters": [
                {
                    "temporary_id": "visible-person-1",
                    "gender": "unknown",
                    "age_range": "unknown",
                    "height": "unknown",
                    "clothing": "dark top",
                    "appearance": "person near center of frame",
                    "behavior": "standing or moving slightly in the sampled frames",
                    "confidence": "medium",
                    "uncertainty": "pose landmarks are not available in this test",
                }
            ],
        }


class EmptyDescriber:
    def describe_segment(self, *, frame_paths, segment):
        return {
            "scene_description": "A simple indoor scene.",
            "scene_confidence": "medium",
            "uncertainty": "synthetic test frames contain limited detail",
            "objects": [],
            "characters": [],
        }


class RosterDescriber(EmptyDescriber):
    def __init__(self):
        self.roster = []

    def describe_roster(self, *, frame_paths, expected_subjects=None):
        return {
            "subject_count": 3,
            "subjects": [
                {
                    "id": "person-01",
                    "gender": "male",
                    "age_range": "adult",
                    "clothing": "dark jacket",
                    "appearance": "short hair, no visible jewelry",
                    "confidence": "medium",
                    "evidence": "synthetic roster",
                },
                {
                    "id": "person-02",
                    "gender": "female",
                    "age_range": "adult",
                    "clothing": "light shirt",
                    "appearance": "long hair, earrings",
                    "confidence": "medium",
                    "evidence": "synthetic roster",
                },
                {
                    "id": "person-03",
                    "gender": "unknown",
                    "age_range": "child",
                    "clothing": "dark top",
                    "appearance": "curly hair, glasses",
                    "confidence": "medium",
                    "evidence": "synthetic roster",
                },
            ],
            "uncertainty": "synthetic roster",
        }

    def set_roster(self, subjects):
        self.roster = subjects


class RosterSegmentDescriber(RosterDescriber):
    def describe_segment(self, *, frame_paths, segment):
        payload = super().describe_segment(frame_paths=frame_paths, segment=segment)
        payload["characters"] = [
            {
                "temporary_id": "person-02",
                "gender": "unknown",
                "age_range": "unknown",
                "height": "unknown",
                "clothing": "light shirt",
                "appearance": "second recurring visible person",
                "behavior": "standing in the sampled frames",
                "confidence": "medium",
                "uncertainty": "synthetic test frames contain limited detail",
            }
        ]
        return payload


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_writes_required_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample_clip.mp4"
            _write_synthetic_video(video_path)

            output_root = tmp_path / "outputs"
            result = run_pipeline(
                PipelineConfig(
                    input_path=video_path,
                    output_root=output_root,
                    segment_sec=5.0,
                    stride_sec=4.0,
                    use_vlm=False,
                ),
                describer=FakeDescriber(),
            )

            self.assertEqual(result.video_id, "sample_clip")
            segments_path = output_root / "sample_clip" / "raw" / "segments.json"
            characters_path = output_root / "sample_clip" / "characters" / "characters.json"
            detections_path = output_root / "sample_clip" / "detections" / "detections.json"
            log_path = output_root / "sample_clip" / "logs" / "pipeline.log"

            self.assertTrue(segments_path.exists())
            self.assertTrue(characters_path.exists())
            self.assertTrue(detections_path.exists())
            self.assertTrue(log_path.exists())

            segments = json.loads(segments_path.read_text())
            characters = json.loads(characters_path.read_text())
            detections = json.loads(detections_path.read_text())

            self.assertEqual(segments["segment_length_sec"], 5.0)
            self.assertGreaterEqual(len(segments["segments"]), 2)
            self.assertEqual(list(characters["characters"]), ["person-01"])
            self.assertEqual(detections["characters"][0]["person_id"], "person-01")
            self.assertIsNone(detections["characters"][0]["track_id"])
            self.assertEqual(characters["characters"]["person-01"]["track_ids"], [])
            self.assertIn("vlm", detections["characters"][0]["behavior"]["modalities_used"])

    def test_vlm_character_text_is_not_attached_to_yolo_tracks_by_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample_clip.mp4"
            _write_synthetic_video(video_path)
            output_root = tmp_path / "outputs"

            run_pipeline(
                PipelineConfig(
                    input_path=video_path,
                    output_root=output_root,
                    segment_sec=5.0,
                    stride_sec=4.0,
                    use_vlm=False,
                    max_segments=1,
                ),
                describer=FakeDescriber(),
                yolo_observations_by_segment={
                    "seg_001": [
                        {
                            "track_id": "track-1",
                            "bbox_summary": {
                                "mean_bbox": [10, 10, 50, 80],
                                "visibility": "full",
                                "confidence": 0.8,
                            },
                            "pose": {
                                "available": False,
                                "summary": "Tracked person is mostly stationary.",
                                "moving_body_parts": [],
                                "speed": "stationary",
                                "directionality": "unclear",
                                "confidence": "low",
                            },
                        }
                    ]
                },
            )

            detections = json.loads(
                (output_root / "sample_clip" / "detections" / "detections.json").read_text()
            )

            tracked, vlm_only = detections["characters"]
            self.assertEqual(tracked["track_id"], "track-1")
            self.assertNotIn("standing or moving slightly", tracked["behavior"]["description"])
            self.assertNotIn("vlm", tracked["behavior"]["modalities_used"])
            self.assertIsNone(vlm_only["track_id"])
            self.assertIn("standing or moving slightly", vlm_only["behavior"]["description"])
            self.assertEqual(vlm_only["motion"]["activity_level"], "unclear")

    def test_fragmented_yolo_tracks_are_merged_into_stable_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample_clip.mp4"
            _write_synthetic_video(video_path)
            output_root = tmp_path / "outputs"

            run_pipeline(
                PipelineConfig(
                    input_path=video_path,
                    output_root=output_root,
                    segment_sec=10.0,
                    stride_sec=6.0,
                    use_vlm=False,
                    max_segments=2,
                ),
                describer=EmptyDescriber(),
                yolo_observations_by_segment={
                    "seg_001": [
                        _yolo_row("track-1", [10, 10, 50, 80], 0.90, 0.0, 9.8),
                        _yolo_row("track-2", [70, 10, 110, 80], 0.89, 0.0, 9.8),
                        _yolo_row("track-3", [120, 10, 150, 80], 0.88, 0.0, 9.8),
                        _yolo_row("track-4", [11, 10, 51, 80], 0.70, 0.0, 9.8),
                    ],
                    "seg_002": [
                        _yolo_row("track-10", [12, 10, 52, 80], 0.90, 6.0, 15.8),
                        _yolo_row("track-20", [72, 10, 112, 80], 0.89, 6.0, 15.8),
                        _yolo_row("track-30", [122, 10, 152, 80], 0.88, 6.0, 15.8),
                    ],
                },
            )

            characters = json.loads(
                (output_root / "sample_clip" / "characters" / "characters.json").read_text()
            )

            self.assertEqual(list(characters["characters"]), ["person-01", "person-02", "person-03"])
            self.assertEqual(
                characters["characters"]["person-01"]["track_ids"],
                ["track-1", "track-4", "track-10"],
            )

    def test_roster_mode_does_not_promote_yolo_fragments_to_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample_clip.mp4"
            _write_synthetic_video(video_path)
            output_root = tmp_path / "outputs"

            run_pipeline(
                PipelineConfig(
                    input_path=video_path,
                    output_root=output_root,
                    segment_sec=10.0,
                    stride_sec=6.0,
                    use_vlm=False,
                    max_segments=2,
                    expected_subjects=3,
                ),
                describer=RosterDescriber(),
                yolo_observations_by_segment={
                    "seg_001": [
                        _yolo_row("track-1", [10, 10, 50, 80], 0.90, 0.0, 9.8),
                        _yolo_row("track-2", [70, 10, 110, 80], 0.89, 0.0, 9.8),
                        _yolo_row("track-3", [120, 10, 150, 80], 0.88, 0.0, 9.8),
                        _yolo_row("track-4", [11, 10, 51, 80], 0.70, 0.0, 9.8),
                    ],
                },
            )

            characters = json.loads(
                (output_root / "sample_clip" / "characters" / "characters.json").read_text()
            )
            detections = json.loads(
                (output_root / "sample_clip" / "detections" / "detections.json").read_text()
            )

            self.assertEqual(list(characters["characters"]), ["person-01", "person-02", "person-03"])
            self.assertTrue(all(row["person_id"] is None for row in detections["characters"]))

    def test_roster_mode_uses_vlm_roster_ids_for_character_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "sample_clip.mp4"
            _write_synthetic_video(video_path)
            output_root = tmp_path / "outputs"

            run_pipeline(
                PipelineConfig(
                    input_path=video_path,
                    output_root=output_root,
                    segment_sec=10.0,
                    stride_sec=6.0,
                    use_vlm=False,
                    max_segments=1,
                    expected_subjects=3,
                ),
                describer=RosterSegmentDescriber(),
                yolo_observations_by_segment={},
            )

            characters = json.loads(
                (output_root / "sample_clip" / "characters" / "characters.json").read_text()
            )
            detections = json.loads(
                (output_root / "sample_clip" / "detections" / "detections.json").read_text()
            )

            self.assertEqual(detections["characters"][0]["person_id"], "person-02")
            self.assertEqual(characters["characters"]["person-02"]["example_segments"], ["seg_001"])
            self.assertEqual(characters["characters"]["person-01"]["gender"], "male")
            self.assertEqual(characters["characters"]["person-02"]["gender"], "female")
            self.assertEqual(characters["characters"]["person-03"]["age_range"], "child")
            self.assertEqual(characters["characters"]["person-01"]["appearance"], "short hair, no visible jewelry")
            self.assertEqual(characters["characters"]["person-02"]["appearance"], "long hair, earrings")
            self.assertNotIn("notes", characters["characters"]["person-01"])


def _write_synthetic_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (160, 90))
    for index in range(72):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        x = 20 + index % 80
        cv2.rectangle(frame, (x, 20), (x + 24, 70), (220, 220, 220), -1)
        writer.write(frame)
    writer.release()


def _yolo_row(track_id: str, bbox: list[int], confidence: float, start: float, end: float) -> dict:
    return {
        "track_id": track_id,
        "start_time": start,
        "end_time": end,
        "bbox_summary": {
            "mean_bbox": bbox,
            "visibility": "full",
            "confidence": confidence,
        },
        "pose": {
            "available": False,
            "summary": "Tracked person is mostly stationary.",
            "moving_body_parts": [],
            "speed": "stationary",
            "directionality": "unclear",
            "confidence": "low",
        },
        "motion": {
            "mean_motion": 0.0,
            "peak_motion": 0.0,
            "peak_time": start,
            "activity_level": "low",
        },
    }


if __name__ == "__main__":
    unittest.main()
