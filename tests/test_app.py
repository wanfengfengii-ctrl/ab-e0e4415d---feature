"""HTTP 层测试：直接调用处理函数与真实 HTTP 端到端冒烟。"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, Dict, Tuple

from nanopore_align.app import align_from_payload, build_server


FEASIBLE_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [15, 25, 35, 45, 55, 65, 75, 85],
    "drift_min": -10,
    "drift_max": 10,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}

INFEASIBLE_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [900] * 8,
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 1,
}


class AlignFromPayloadTests(unittest.TestCase):
    def test_feasible(self) -> None:
        status, body = align_from_payload(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertEqual(body["drift"], 5)
        self.assertEqual(body["residual_sum"], 0)
        self.assertEqual(len(body["levels"]), 8)

    def test_infeasible_is_explicit_conclusion(self) -> None:
        status, body = align_from_payload(INFEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "no_alignment_exists")
        self.assertIn("message", body)

    def test_missing_field_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        del bad["drift_max"]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["error"], "missing_fields")

    def test_unknown_field_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["extra"] = 1
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "unknown_fields")

    def test_bad_size_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["reference_levels"] = list(range(7))
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")

    def test_too_many_observations_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["observations"] = list(range(61))
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)

    def test_wrong_type_rejected(self) -> None:
        status, body = align_from_payload("not-a-dict")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_body")


class AnchoredPayloadTests(unittest.TestCase):
    """anchors 字段：可行证据、不相容结论与字段级录入错误。"""

    def _anchored(self, anchors: Any) -> Dict[str, Any]:
        payload = dict(FEASIBLE_PAYLOAD)
        payload["anchors"] = anchors
        return payload

    def test_anchored_feasible_with_assignments(self) -> None:
        payload = self._anchored(
            [
                {"reference_index": 0, "sample_index": 0},
                {"reference_index": 7, "sample_index": 7},
            ]
        )
        status, body = align_from_payload(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        assignments = body["anchor_assignments"]
        self.assertEqual(len(assignments), 2)
        first, last = assignments
        self.assertEqual((first["reference_index"], first["sample_index"]), (0, 0))
        self.assertEqual(first["level_order"], 0)
        self.assertEqual((first["sample_start"], first["sample_end"]), (0, 1))
        self.assertEqual(first["observed"], 15)
        self.assertEqual(first["adopted_level"], 15)
        self.assertEqual(first["residual"], 0)
        self.assertEqual((last["reference_index"], last["sample_index"]), (7, 7))
        self.assertEqual(last["level_order"], 7)

    def test_omitted_anchors_response_unchanged(self) -> None:
        status, body = align_from_payload(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertNotIn("anchor_assignments", body)

    def test_anchors_null_treated_as_omitted(self) -> None:
        payload = self._anchored(None)
        status, body = align_from_payload(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertNotIn("anchor_assignments", body)

    def test_anchor_incompatible_is_explicit_conclusion(self) -> None:
        # dwell=1 下样本 7 只能归属末级，锚定到首级格式合法但不相容。
        payload = self._anchored([{"reference_index": 0, "sample_index": 7}])
        status, body = align_from_payload(payload)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "no_alignment_with_anchors")
        self.assertIn("message", body)

    def test_anchor_out_of_range_is_field_error(self) -> None:
        payload = self._anchored([{"reference_index": 8, "sample_index": 0}])
        status, body = align_from_payload(payload)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["error"], "invalid_anchors")
        self.assertEqual(body["field"], "anchors[0].reference_index")
        self.assertIn("message", body)

    def test_anchor_duplicate_is_field_error(self) -> None:
        payload = self._anchored(
            [
                {"reference_index": 2, "sample_index": 2},
                {"reference_index": 2, "sample_index": 5},
            ]
        )
        status, body = align_from_payload(payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_anchors")
        self.assertEqual(body["field"], "anchors[1].reference_index")

    def test_anchor_order_conflict_is_field_error(self) -> None:
        payload = self._anchored(
            [
                {"reference_index": 2, "sample_index": 5},
                {"reference_index": 4, "sample_index": 2},
            ]
        )
        status, body = align_from_payload(payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_anchors")
        self.assertEqual(body["field"], "anchors[1].sample_index")

    def test_anchor_shape_errors(self) -> None:
        for bad in (
            [],
            [{"reference_index": i, "sample_index": i} for i in range(4)],
            [{"reference_index": 0}],
            [{"reference_index": 0, "sample_index": 0, "x": 1}],
            [{"reference_index": 0, "sample_index": "0"}],
            ["not-a-dict"],
        ):
            with self.subTest(anchors=bad):
                status, body = align_from_payload(self._anchored(bad))
                self.assertEqual(status, 400)
                self.assertEqual(body["error"], "invalid_anchors")
                self.assertTrue(body["field"].startswith("anchors"))


class HttpEndToEndTests(unittest.TestCase):
    server: ThreadingHTTPServer
    thread: threading.Thread
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = build_server("127.0.0.1", 0)
        port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _post(self, payload: Any) -> Tuple[int, Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/current-traces/align",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self) -> None:
        with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["status"], "ok")

    def test_feasible_roundtrip(self) -> None:
        status, body = self._post(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        covered = [
            s["index"]
            for lv in body["levels"]
            for s in lv["samples"]
        ]
        self.assertEqual(covered, list(range(8)))

    def test_infeasible_roundtrip(self) -> None:
        status, body = self._post(INFEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])

    def test_anchored_roundtrip(self) -> None:
        payload = dict(FEASIBLE_PAYLOAD)
        payload["anchors"] = [{"reference_index": 3, "sample_index": 3}]
        status, body = self._post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        (ev,) = body["anchor_assignments"]
        self.assertEqual((ev["reference_index"], ev["sample_index"]), (3, 3))
        self.assertLessEqual(ev["sample_start"], 3)
        self.assertLess(3, ev["sample_end"])
        self.assertEqual(ev["residual"], ev["observed"] - ev["adopted_level"])

    def test_anchor_validation_error_roundtrip(self) -> None:
        payload = dict(FEASIBLE_PAYLOAD)
        payload["anchors"] = [
            {"reference_index": 0, "sample_index": 0},
            {"reference_index": 1, "sample_index": 0},  # 同一样本重复
        ]
        status, body = self._post(payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_anchors")

    def test_anchor_infeasible_roundtrip(self) -> None:
        payload = dict(FEASIBLE_PAYLOAD)
        payload["anchors"] = [{"reference_index": 7, "sample_index": 0}]
        status, body = self._post(payload)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "no_alignment_with_anchors")

    def test_invalid_roundtrip(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["drift_min"] = 99
        bad["drift_max"] = 0
        status, body = self._post(bad)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])

    def test_malformed_json(self) -> None:
        req = urllib.request.Request(
            f"{self.base}/api/current-traces/align",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("应当返回 400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_unknown_route_404(self) -> None:
        try:
            urllib.request.urlopen(f"{self.base}/nope", timeout=5)
            self.fail("应当返回 404")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)


if __name__ == "__main__":
    unittest.main()
