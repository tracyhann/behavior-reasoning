import json
import tempfile
import unittest
from pathlib import Path

from video_features.io_utils import ensure_output_tree, write_json
from video_features.segmentation import build_segments
from video_features.subject_library import SubjectLibrary
from video_features.vlm import normalize_roster_payload, normalize_vlm_payload
from video_features.yolo_detection import assign_observations_to_segments, summarize_track


class SegmentationTests(unittest.TestCase):
    def test_builds_overlapping_segments_with_trailing_partial(self):
        segments = build_segments(duration_sec=23.0, segment_sec=10.0, stride_sec=6.0)

        self.assertEqual(
            [segment["segment_id"] for segment in segments],
            ["seg_001", "seg_002", "seg_003", "seg_004"],
        )
        self.assertEqual([segment["start_time"] for segment in segments], [0.0, 6.0, 12.0, 18.0])
        self.assertEqual([segment["end_time"] for segment in segments], [10.0, 16.0, 22.0, 23.0])

    def test_short_video_gets_one_segment(self):
        self.assertEqual(
            build_segments(duration_sec=4.2, segment_sec=10.0, stride_sec=6.0),
            [{"segment_id": "seg_001", "start_time": 0.0, "end_time": 4.2}],
        )


class VlmNormalizationTests(unittest.TestCase):
    def test_normalizes_payload_and_preserves_coarse_demographic_traits(self):
        payload = normalize_vlm_payload(
            json.dumps(
                {
                    "scene_description": "Two people are visible in an indoor room.",
                    "scene_confidence": "high",
                    "objects": [{"label": "chair", "count": 2, "confidence": 0.8}],
                    "characters": [
                        {
                            "temporary_id": "left person",
                            "gender": "male",
                            "age_range": "adult",
                            "clothing": "dark jacket",
                            "appearance": "standing near the left side",
                            "behavior": "standing near a table",
                            "confidence": "medium",
                        }
                    ],
                }
            )
        )

        self.assertEqual(payload["scene_confidence"], "high")
        self.assertEqual(payload["characters"][0]["gender"], "male")
        self.assertEqual(payload["characters"][0]["age_range"], "adult")
        self.assertEqual(payload["characters"][0]["behavior"], "standing near a table")
        self.assertIn("limited", payload["uncertainty"])

    def test_accepts_common_qwen_alternate_keys(self):
        payload = normalize_vlm_payload(
            """
            ```json
            {
              "scene_description": "A person walks down a hallway.",
              "scene_confidence": 0.9,
              "objects": [{"name": "backpack", "location": "on the person's back"}],
              "characters": [{"name": "person", "description": "walking with a backpack"}]
            }
            ```
            """
        )

        self.assertEqual(payload["scene_confidence"], "high")
        self.assertEqual(payload["objects"][0]["label"], "backpack")
        self.assertEqual(payload["objects"][0]["description"], "on the person's back")
        self.assertEqual(payload["characters"][0]["temporary_id"], "person")
        self.assertEqual(payload["characters"][0]["behavior"], "walking with a backpack")

    def test_replaces_identity_names_and_expands_unhelpful_uncertainty(self):
        payload = normalize_vlm_payload(
            {
                "scene_description": "Spider-Man and iron-man are flying near buildings.",
                "scene_confidence": "high",
                "uncertainty": "0.1",
                "objects": [{"label": "spider emblem", "description": "red and gold armor nearby"}],
                "characters": [
                    {
                        "name": "iron-man",
                        "clothing": "red and blue suit with spider emblem",
                        "description": "Spider Man is gliding",
                        "uncertainty": "Low",
                    }
                ],
            }
        )

        dumped = json.dumps(payload)
        self.assertNotIn("Spider-Man", dumped)
        self.assertNotIn("iron-man", dumped)
        self.assertNotIn("Iron Man", dumped)
        self.assertNotIn("Peter Parker", dumped)
        self.assertNotIn("spider emblem", dumped.lower())
        self.assertNotIn("red and gold armor", dumped.lower())
        self.assertEqual(payload["objects"][0]["label"], "person")
        self.assertIn("limited", payload["uncertainty"])
        self.assertIn("limited", payload["characters"][0]["uncertainty"])

    def test_empty_vlm_payload_has_low_confidence(self):
        payload = normalize_vlm_payload({})

        self.assertEqual(payload["scene_confidence"], "low")
        self.assertEqual(payload["scene_description"], "Scene description unavailable from sampled frames.")

    def test_sanitizes_inferred_behavior_language(self):
        payload = normalize_vlm_payload(
            {
                "scene_description": "A young person appears engaged in a conversation and reacting while speaking.",
                "uncertainty": "The facial expression and context suggest the person might be explaining something important.",
                "characters": [
                    {
                        "description": "he is actively communicating, mouth open as if speaking or reacting while talking",
                        "confidence": "high",
                    }
                ],
            }
        )

        dumped = json.dumps(payload).lower()
        self.assertNotIn("engaged", dumped)
        self.assertNotIn("reacting", dumped)
        self.assertNotIn("as if", dumped)
        self.assertNotIn("speaking", dumped)
        self.assertNotIn("talking", dumped)
        self.assertNotIn(" he ", f" {dumped} ")
        self.assertNotIn("actively communicating", dumped)
        self.assertNotIn("young person", dumped)
        self.assertNotIn("might be explaining", dumped)
        self.assertNotIn("something important", dumped)
        self.assertIn("mouth open", dumped)

    def test_roster_normalization_preserves_traits_and_sanitizes_description(self):
        payload = normalize_roster_payload(
            {
                "subjects": [
                    {
                        "id": "person-01",
                        "gender": "male",
                        "age_range": "adult",
                        "visual_description": "man standing in a kitchen with short hair, no visible jewelry, light skin",
                        "confidence": "high",
                    },
                    {
                        "id": "person-02",
                        "gender": "female",
                        "age_range": "adult",
                        "visual_description": "woman sitting on a couch with long hair, earrings, makeup not visible, dark skin",
                        "confidence": "high",
                    },
                ]
            }
        )

        first, second = payload["subjects"]
        self.assertEqual(first["gender"], "male")
        self.assertEqual(first["age_range"], "adult")
        self.assertEqual(second["gender"], "female")
        self.assertEqual(second["age_range"], "adult")
        self.assertNotIn("man", first["appearance"].lower())
        self.assertNotIn("woman", second["appearance"].lower())
        self.assertNotIn("light skin", first["appearance"].lower())
        self.assertNotIn("dark skin", second["appearance"].lower())
        self.assertNotIn("standing", first["appearance"].lower())
        self.assertNotIn("kitchen", first["appearance"].lower())
        self.assertNotIn("sitting", second["appearance"].lower())
        self.assertNotIn("couch", second["appearance"].lower())
        self.assertIn("person with short hair", first["appearance"].lower())
        self.assertIn("no visible jewelry", first["appearance"].lower())
        self.assertIn("long hair", second["appearance"].lower())
        self.assertIn("earrings", second["appearance"].lower())
        self.assertIn("makeup not visible", second["appearance"].lower())
        self.assertNotIn(",,", first["appearance"].lower())
        self.assertFalse(first["appearance"].endswith(","))


class SubjectLibraryTests(unittest.TestCase):
    def test_reuses_existing_person_for_same_track_id(self):
        library = SubjectLibrary()

        first = library.upsert_observation(
            {
                "track_id": "track-4",
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "dark jacket",
                "appearance": "person near left side",
                "crop_path": "raw/crops/person-01_seg_001.jpg",
            }
        )
        second = library.upsert_observation(
            {
                "track_id": "track-4",
                "segment_id": "seg_002",
                "start_time": 6.0,
                "end_time": 16.0,
                "clothing": "dark jacket",
                "appearance": "person near left side",
                "crop_path": "raw/crops/person-01_seg_002.jpg",
            }
        )

        data = library.to_json("video-x")
        self.assertEqual(first, "person-01")
        self.assertEqual(second, "person-01")
        self.assertEqual(list(data["characters"]), ["person-01"])
        self.assertEqual(data["characters"]["person-01"]["last_seen"], 16.0)
        self.assertEqual(data["characters"]["person-01"]["example_segments"], ["seg_001", "seg_002"])
        self.assertNotIn("notes", data["characters"]["person-01"])

    def test_stores_coarse_traits_on_subject_card(self):
        library = SubjectLibrary()

        person_id = library.upsert_observation(
            {
                "track_id": None,
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "gender": "female",
                "age_range": "adult",
                "clothing": "light shirt",
                "appearance": "standing near right side",
            }
        )

        card = library.to_json("video-x")["characters"][person_id]
        self.assertEqual(card["gender"], "female")
        self.assertEqual(card["age_range"], "adult")
        self.assertNotIn("notes", card)

    def test_does_not_merge_overlapping_vlm_only_people_by_time(self):
        library = SubjectLibrary()

        first = library.upsert_observation(
            {
                "track_id": None,
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "dark jacket",
                "appearance": "standing near left side",
            }
        )
        second = library.upsert_observation(
            {
                "track_id": None,
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "light shirt",
                "appearance": "seated near right side",
            }
        )

        self.assertEqual(first, "person-01")
        self.assertEqual(second, "person-02")
        data = library.to_json("video-x")
        self.assertEqual(list(data["characters"]), ["person-01", "person-02"])

    def test_vlm_local_ids_are_not_persisted_as_track_ids(self):
        library = SubjectLibrary()

        person_id = library.upsert_observation(
            {
                "temporary_id": "left person",
                "track_id": None,
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "dark jacket",
                "appearance": "standing",
            }
        )

        data = library.to_json("video-x")
        self.assertEqual(person_id, "person-01")
        self.assertEqual(data["characters"]["person-01"]["track_ids"], [])

    def test_different_real_track_ids_are_not_merged_by_generic_text(self):
        library = SubjectLibrary()

        first = library.upsert_observation(
            {
                "track_id": "track-1",
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "visible clothing unclear",
                "appearance": "tracked person visible in sampled segment",
            }
        )
        second = library.upsert_observation(
            {
                "track_id": "track-2",
                "segment_id": "seg_002",
                "start_time": 6.0,
                "end_time": 16.0,
                "clothing": "visible clothing unclear",
                "appearance": "tracked person visible in sampled segment",
            }
        )

        self.assertEqual(first, "person-01")
        self.assertEqual(second, "person-02")

    def test_different_real_track_ids_can_merge_with_specific_appearance(self):
        library = SubjectLibrary()

        first = library.upsert_observation(
            {
                "track_id": "track-1",
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "dark jacket over white shirt",
                "appearance": "short hair standing in kitchen",
            }
        )
        second = library.upsert_observation(
            {
                "track_id": "track-2",
                "segment_id": "seg_002",
                "start_time": 10.5,
                "end_time": 16.0,
                "clothing": "dark jacket over white shirt",
                "appearance": "short hair standing in kitchen",
            }
        )

        data = library.to_json("video-x")
        self.assertEqual(first, "person-01")
        self.assertEqual(second, "person-01")
        self.assertEqual(data["characters"]["person-01"]["track_ids"], ["track-1", "track-2"])

    def test_different_real_track_ids_can_merge_with_bbox_continuity(self):
        library = SubjectLibrary()

        first = library.upsert_observation(
            {
                "track_id": "track-1",
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 10.0,
                "clothing": "visible clothing unclear",
                "appearance": "tracked person visible in sampled segment",
                "bbox": [10, 10, 50, 90],
            }
        )
        second = library.upsert_observation(
            {
                "track_id": "track-2",
                "segment_id": "seg_002",
                "start_time": 10.5,
                "end_time": 16.0,
                "clothing": "visible clothing unclear",
                "appearance": "tracked person visible in sampled segment",
                "bbox": [12, 11, 52, 91],
            }
        )

        self.assertEqual(first, "person-01")
        self.assertEqual(second, "person-01")


class OutputTreeTests(unittest.TestCase):
    def test_creates_required_output_tree_and_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = ensure_output_tree(Path(tmp), "video-x")
            write_json(root.raw / "segments.json", {"video_id": "video-x", "segments": []})

            self.assertTrue((Path(tmp) / "video-x" / "raw" / "segments.json").exists())
            self.assertTrue(root.characters.exists())
            self.assertTrue(root.detections.exists())
            self.assertTrue(root.logs.exists())


class YoloDetectionTests(unittest.TestCase):
    def test_assigns_observations_to_overlapping_segments(self):
        segments = build_segments(duration_sec=12.0, segment_sec=10.0, stride_sec=6.0)
        observations = [
            {"timestamp": 7.0, "track_id": "track-1", "bbox": [0, 0, 10, 20], "confidence": 0.8}
        ]

        assigned = assign_observations_to_segments(observations, segments)

        self.assertEqual([item["segment_id"] for item in assigned], ["seg_001", "seg_002"])

    def test_summarizes_track_bbox_and_motion(self):
        summary = summarize_track(
            [
                {"timestamp": 0.0, "bbox": [0, 0, 50, 90], "confidence": 0.8},
                {"timestamp": 1.0, "bbox": [20, 0, 70, 90], "confidence": 0.6},
            ],
            frame_width=100,
            frame_height=100,
        )

        self.assertEqual(summary["bbox_summary"]["mean_bbox"], [10, 0, 60, 90])
        self.assertEqual(summary["bbox_summary"]["visibility"], "full")
        self.assertEqual(summary["pose"]["directionality"], "unclear")
        self.assertEqual(summary["pose"]["speed"], "unclear")
        self.assertEqual(summary["motion"]["activity_level"], "high")

    def test_summarizes_pose_keypoint_movement_when_available(self):
        def keypoints(right_wrist_x):
            points = [[0.0, 0.0, 0.0] for _ in range(17)]
            points[6] = [20.0, 20.0, 0.9]
            points[8] = [25.0, 40.0, 0.9]
            points[10] = [right_wrist_x, 60.0, 0.9]
            return points

        summary = summarize_track(
            [
                {
                    "timestamp": 0.0,
                    "bbox": [0, 0, 50, 90],
                    "confidence": 0.8,
                    "keypoints": keypoints(30.0),
                },
                {
                    "timestamp": 1.0,
                    "bbox": [0, 0, 50, 90],
                    "confidence": 0.8,
                    "keypoints": keypoints(55.0),
                },
            ],
            frame_width=100,
            frame_height=100,
        )

        self.assertTrue(summary["pose"]["available"])
        self.assertIn("right_arm", summary["pose"]["moving_body_parts"])
        self.assertEqual(summary["pose"]["directionality"], "unclear")

    def test_groups_only_confident_persistent_track_rows(self):
        from video_features.yolo_detection import group_segment_observations

        rows = group_segment_observations(
            [
                {"track_id": "track-1", "timestamp": 0.0, "bbox": [0, 0, 50, 90], "confidence": 0.8},
                {"track_id": "track-1", "timestamp": 0.1, "bbox": [1, 0, 51, 90], "confidence": 0.7},
                {"track_id": "track-1", "timestamp": 0.2, "bbox": [2, 0, 52, 90], "confidence": 0.7},
                {"track_id": "det-1", "timestamp": 0.0, "bbox": [0, 0, 5, 5], "confidence": 0.9},
                {"track_id": "track-2", "timestamp": 0.0, "bbox": [0, 0, 50, 90], "confidence": 0.1},
                {"track_id": "track-3", "timestamp": 0.0, "bbox": [0, 0, 50, 90], "confidence": 0.9},
            ],
            frame_width=100,
            frame_height=100,
        )

        self.assertEqual([row["track_id"] for row in rows], ["track-1"])


if __name__ == "__main__":
    unittest.main()
