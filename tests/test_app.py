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

# N=15、dwell_min=2 迫使 1 个内部跳过；无锚点时跳 R3，锚定 (R3, 采样6)
# 后必须改跳等价电平 R4（R3=R4=300）。
ANCHORED_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [0, 100, 200, 300, 300, 400, 500, 600],
    "observations": [0, 0, 100, 100, 200, 200, 300, 300,
                     400, 400, 500, 500, 600, 600, 600],
    "drift_min": 0,
    "drift_max": 0,
    "residual_limit": 0,
    "dwell_min": 2,
    "dwell_max": 3,
    "anchors": [[3, 6]],
}

# 同一轨迹锚定一个无法被满足的归属（采样 6 归 R2，停留约束下不可达）。
ANCHORED_INCOMPATIBLE_PAYLOAD: Dict[str, Any] = {
    **{
        k: v for k, v in ANCHORED_PAYLOAD.items() if k != "anchors"
    },
    "anchors": [[2, 6]],
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

    def test_anchored_request_returns_evidence(self) -> None:
        status, body = align_from_payload(ANCHORED_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        # 锚点禁止跨过 R3：无锚点时本会跳过 R3。
        self.assertEqual(body["skipped_reference_indices"], [4])
        assignments = body["anchor_assignments"]
        self.assertEqual(len(assignments), 1)
        ev = assignments[0]
        self.assertEqual(ev["reference_index"], 3)
        self.assertEqual(ev["sample_index"], 6)
        self.assertLessEqual(ev["sample_start"], 6)
        self.assertLess(6, ev["sample_end"])
        self.assertEqual(ev["assigned_level_order"], 3)
        anchored = [lv for lv in body["levels"] if lv["anchored"]]
        self.assertEqual(len(anchored), 1)
        self.assertEqual(anchored[0]["reference_index"], 3)

    def test_anchored_incompatible_is_explicit_conclusion(self) -> None:
        status, body = align_from_payload(ANCHORED_INCOMPATIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "anchors_incompatible")
        self.assertIn("message", body)

    def test_anchor_out_of_range_field_error(self) -> None:
        bad = dict(ANCHORED_PAYLOAD)
        bad["anchors"] = [[8, 0]]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_anchors")
        fields = [e["field"] for e in body["errors"]]
        self.assertIn("anchors[0].reference_index", fields)

        bad = dict(ANCHORED_PAYLOAD)
        bad["anchors"] = [[0, 15]]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        fields = [e["field"] for e in body["errors"]]
        self.assertIn("anchors[0].sample_index", fields)

    def test_anchor_duplicate_and_order_field_errors(self) -> None:
        bad = dict(ANCHORED_PAYLOAD)
        bad["anchors"] = [[3, 1], [3, 6]]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_anchors")
        messages = " ".join(e["message"] for e in body["errors"])
        self.assertIn("reference_index", messages)

        bad = dict(ANCHORED_PAYLOAD)
        bad["anchors"] = [[4, 6], [3, 8]]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        messages = " ".join(e["message"] for e in body["errors"])
        self.assertIn("reference_index", messages)

        bad = dict(ANCHORED_PAYLOAD)
        bad["anchors"] = [[3, 9], [4, 8]]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        messages = " ".join(e["message"] for e in body["errors"])
        self.assertIn("sample_index", messages)

    def test_anchor_shape_field_errors(self) -> None:
        for bad_anchors in [[], [[1]], [[1, 2, 3]],
                            [[1, "x"]], "nope", [[1, 2], [3, 4],
                                                  [5, 6], [7, 0]]]:
            bad = dict(ANCHORED_PAYLOAD)
            bad["anchors"] = bad_anchors
            status, body = align_from_payload(bad)
            self.assertEqual(
                status, 400, msg=f"anchors={bad_anchors} body={body}"
            )
            self.assertEqual(body["error"], "invalid_anchors")
            self.assertTrue(body["errors"])

    def test_anchor_field_error_does_not_swallow_other_shape(self) -> None:
        # anchors 结构错误时直接报字段错误，不依赖其他字段是否合法。
        status, body = align_from_payload({"anchors": [[1, 2]]})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "missing_fields")

    def test_explicit_null_anchors_equivalent_to_omitted(self) -> None:
        payload = dict(FEASIBLE_PAYLOAD)
        payload["anchors"] = None
        status, body = align_from_payload(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertNotIn("anchor_assignments", body)


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
        status, body = self._post(ANCHORED_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertEqual(body["skipped_reference_indices"], [4])
        ev = body["anchor_assignments"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["reference_index"], 3)
        self.assertEqual(ev[0]["sample_index"], 6)
        self.assertLessEqual(ev[0]["sample_start"], 6)
        self.assertLess(6, ev[0]["sample_end"])

    def test_anchored_incompatible_roundtrip(self) -> None:
        status, body = self._post(ANCHORED_INCOMPATIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "anchors_incompatible")

    def test_anchors_field_error_roundtrip(self) -> None:
        bad = dict(ANCHORED_PAYLOAD)
        bad["anchors"] = [[3, 1], [3, 2]]
        status, body = self._post(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_anchors")
        self.assertTrue(body["errors"])

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
