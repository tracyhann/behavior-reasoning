import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_behavior_chains as chains  # noqa: E402


class BehaviorChainTests(unittest.TestCase):
    def test_load_records_accepts_list_and_dict_styles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            list_path = Path(tmp_dir) / "list.json"
            dict_path = Path(tmp_dir) / "dict.json"
            list_path.write_text(json.dumps([{"id": "det_1"}]), encoding="utf-8")
            dict_path.write_text(json.dumps({"characters": {"person-01": {"id": "person-01"}}}), encoding="utf-8")

            self.assertEqual(chains.load_json_file(list_path), [{"id": "det_1"}])
            records = chains.extract_character_records(chains.load_json_file(dict_path))

        self.assertEqual(records, [{"id": "person-01"}])

    def test_normalizes_timestamp_fields_and_aliases(self):
        resolver = chains.CharacterResolver.from_payload(
            {"characters": {"person-01": {"id": "person-01", "aliases": ["P1"]}}}
        )
        raw = {
            "detection_id": "det_a",
            "timestamp": 12.5,
            "label": "P1 waves",
            "type": "gesture",
            "actor": "P1",
            "target_ids": ["unknown"],
            "modality": "rgb",
            "score": "high",
            "source_segment_id": "seg_003",
        }

        event = chains.normalize_detection(raw, resolver, 0, "characters")

        self.assertEqual(event["start_sec"], 12.5)
        self.assertEqual(event["end_sec"], 12.5)
        self.assertEqual(event["event_type"], "gesture")
        self.assertEqual(event["actor_ids"], ["person-01"])
        self.assertEqual(event["target_ids"], ["unknown"])
        self.assertEqual(event["modalities"], ["rgb"])
        self.assertEqual(event["confidence"], 0.9)
        self.assertEqual(event["source_segment_ids"], ["seg_003"])

    def test_resolves_character_aliases_track_ids_and_labels(self):
        resolver = chains.CharacterResolver.from_payload(
            {
                "characters": {
                    "person-01": {
                        "id": "person-01",
                        "name": "speaker",
                        "label": "P1",
                        "track_ids": ["track-7"],
                        "aliases": ["boy"],
                    }
                }
            }
        )

        self.assertEqual(resolver.resolve("speaker"), "person-01")
        self.assertEqual(resolver.resolve("P1"), "person-01")
        self.assertEqual(resolver.resolve("track-7"), "person-01")
        self.assertEqual(resolver.resolve("boy"), "person-01")
        self.assertEqual(resolver.resolve("unmapped"), "unmapped")

    def test_deduplicates_overlapping_duplicate_events(self):
        resolver = chains.CharacterResolver.from_payload({"characters": {"person-01": {"id": "person-01"}}})
        raw_events = [
            {
                "id": "det_1",
                "start_sec": 10,
                "end_sec": 14,
                "event_type": "gesture",
                "actor_ids": ["person-01"],
                "description": "person-01 raises a hand",
                "confidence": 0.8,
            },
            {
                "id": "det_2",
                "start_time": 11,
                "end_time": 15,
                "type": "gesture",
                "person_id": "person-01",
                "caption": "person-01 raises hand",
                "score": 0.6,
            },
        ]
        events = [chains.normalize_detection(e, resolver, i, "characters") for i, e in enumerate(raw_events)]

        canonical = chains.deduplicate_events(events, chains.Params())

        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["source_detection_ids"], ["det_1", "det_2"])
        self.assertEqual(canonical[0]["merge_notes"]["merged_count"], 2)

    def test_does_not_merge_repeated_events_far_apart(self):
        resolver = chains.CharacterResolver.from_payload({"characters": {"person-01": {"id": "person-01"}}})
        raw_events = [
            {"id": "det_1", "start_sec": 1, "end_sec": 2, "event_type": "gesture", "actor_ids": ["person-01"]},
            {"id": "det_2", "start_sec": 20, "end_sec": 21, "event_type": "gesture", "actor_ids": ["person-01"]},
        ]
        events = [chains.normalize_detection(e, resolver, i, "characters") for i, e in enumerate(raw_events)]

        canonical = chains.deduplicate_events(events, chains.Params())

        self.assertEqual(len(canonical), 2)

    def test_builds_cue_response_chain(self):
        canonical = [
            make_event(
                "evt_0001",
                10,
                11,
                "gesture",
                ["person-01"],
                ["person-02"],
                "person-01 gestures toward person-02.",
                ["det_1"],
            ),
            make_event(
                "evt_0002",
                13,
                14,
                "response",
                ["person-02"],
                ["person-01"],
                "person-02 turns toward person-01.",
                ["det_2"],
            ),
        ]

        payload = chains.build_behavior_chains(canonical, {}, chains.Params())

        self.assertEqual(len(payload["chains"]), 1)
        chain = payload["chains"][0]
        self.assertEqual(chain["event_ids"], ["evt_0001", "evt_0002"])
        self.assertEqual(chain["participant_ids"], ["person-01", "person-02"])
        self.assertEqual(chain["interaction_pattern"], "cue_response")
        self.assertFalse(payload["orphan_event_ids"])

    def test_orphans_events_missing_time(self):
        canonical = [
            make_event(
                "evt_0001",
                None,
                None,
                "gesture",
                ["person-01"],
                [],
                "missing timestamp",
                ["det_1"],
            )
        ]

        payload = chains.build_behavior_chains(canonical, {}, chains.Params())

        self.assertEqual(payload["chains"], [])
        self.assertEqual(payload["orphan_event_ids"], ["evt_0001"])
        self.assertTrue(payload["warnings"])

    def test_preserves_chronological_order(self):
        canonical = []
        for idx, start in enumerate([20, 5, 12], start=1):
            canonical.append(
                make_event(
                    f"evt_{idx:04d}",
                    start,
                    start + 1,
                    "motion",
                    ["person-01"],
                    [],
                    f"event {start}",
                    [f"det_{idx}"],
                )
            )

        payload = chains.build_behavior_chains(canonical, {}, chains.Params(max_chain_gap_sec=30))

        self.assertEqual(payload["chains"][0]["event_ids"], ["evt_0002", "evt_0003", "evt_0001"])
        self.assertEqual([e["event_id"] for e in payload["chains"][0]["events"]], ["evt_0002", "evt_0003", "evt_0001"])

    def test_missing_optional_fields_no_crash(self):
        resolver = chains.CharacterResolver.from_payload({})
        event = chains.normalize_detection({"description": "A person walks."}, resolver, 3, "freeform")
        canonical = chains.deduplicate_events([event], chains.Params())
        payload = chains.build_behavior_chains(canonical, {}, chains.Params())

        self.assertEqual(canonical[0]["event_id"], "evt_0001")
        self.assertEqual(canonical[0]["actor_ids"], [])
        self.assertEqual(canonical[0]["start_sec"], None)
        self.assertEqual(payload["chains"], [])
        self.assertEqual(payload["orphan_event_ids"], ["evt_0001"])

    def test_quality_checks_repeat_five_iterations(self):
        canonical = [
            make_event(
                "evt_0001",
                1,
                2,
                "gesture",
                ["person-01"],
                ["person-02"],
                "person-01 gestures toward person-02.",
                ["det_1"],
            ),
            make_event(
                "evt_0002",
                3,
                4,
                "response",
                ["person-02"],
                ["person-01"],
                "person-02 turns toward person-01.",
                ["det_2"],
            ),
        ]
        payload = chains.build_behavior_chains(canonical, {}, chains.Params())

        report = chains.run_quality_checks(canonical, payload, qa_iterations=5)

        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual(len(report["iterations"]), 5)
        self.assertTrue(all(item["status"] == "pass" for item in report["iterations"]))


def make_event(event_id, start, end, event_type, actors, targets, description, source_ids):
    return {
        "event_id": event_id,
        "start_sec": start,
        "end_sec": end,
        "event_type": event_type,
        "actor_ids": actors,
        "target_ids": targets,
        "object_ids": [],
        "modalities": ["rgb"],
        "description": description,
        "confidence": 0.8,
        "source_detection_ids": source_ids,
        "source_segment_ids": ["seg_001"],
        "merge_notes": {"merged_count": 1, "conflicts": [], "dropped_interpretive_claims": []},
    }


if __name__ == "__main__":
    unittest.main()
