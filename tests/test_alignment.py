"""联合对齐算法的单元测试与暴力对照测试。"""

from __future__ import annotations

import itertools
import random
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

from nanopore_align.alignment import (
    AlignmentError,
    AnchorValidationError,
    solve_alignment,
)


def brute_force(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    anchors: Optional[Sequence[dict]] = None,
) -> Optional[dict]:
    """穷举所有漂移 / 含首尾子序列 / 停留组合，返回与 solve 同口径的最优解。"""
    R = len(reference)
    N = len(observations)
    anchor_by_ref = (
        {a["reference_index"]: a["sample_index"] for a in anchors}
        if anchors
        else {}
    )
    best: Optional[Tuple] = None

    for d in range(drift_min, drift_max + 1):
        # 枚举被跳过的内部参考索引集合（大小 <= max_skips）。
        inner = list(range(1, R - 1))
        for skip_count in range(0, min(max_skips, len(inner)) + 1):
            for skipped in itertools.combinations(inner, skip_count):
                used = [i for i in range(R) if i not in set(skipped)]
                k = len(used)
                if k > N or k * dwell_min > N or k * dwell_max < N:
                    continue
                # 锚定参考级不得被跳过。
                if any(ri in set(skipped) for ri in anchor_by_ref):
                    continue
                for dwells in itertools.product(
                    range(dwell_min, dwell_max + 1), repeat=k
                ):
                    if sum(dwells) != N:
                        continue
                    boundaries = tuple(itertools.accumulate(dwells))
                    # 锚定观测必须归属锚定级的连续采样区间。
                    if anchor_by_ref:
                        pos = {ri: slot for slot, ri in enumerate(used)}
                        starts = (0,) + boundaries[:-1]
                        if any(
                            not (
                                starts[pos[ri]]
                                <= t
                                < starts[pos[ri]] + dwells[pos[ri]]
                            )
                            for ri, t in anchor_by_ref.items()
                        ):
                            continue
                    total = 0
                    worst = 0
                    ok = True
                    s = 0
                    for ri, L in zip(used, dwells):
                        level = reference[ri] + d
                        for t in range(s, s + L):
                            r = abs(observations[t] - level)
                            if r > residual_limit:
                                ok = False
                                break
                            total += r
                            worst = max(worst, r)
                        if not ok:
                            break
                        s += L
                    if not ok:
                        continue
                    cost = (skip_count, total, worst, d, boundaries)
                    if best is None or cost < best[0]:
                        best = (cost, used, d)

    if best is None:
        return None
    (skipped_n, total, worst, d, boundaries), used, drift = best
    return {
        "feasible": True,
        "drift": drift,
        "num_skips": skipped_n,
        "residual_sum": total,
        "max_abs_residual": worst,
        "boundaries": list(boundaries[:-1]),
        "used_indices": list(used),
    }


def _assert_matches_brute(
    testcase: unittest.TestCase,
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int = 1,
    dwell_max: int = 3,
    max_skips: int = 2,
    anchors: Optional[Sequence[dict]] = None,
) -> None:
    got = solve_alignment(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
        anchors=anchors,
    )
    want = brute_force(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
        anchors=anchors,
    )
    if want is None:
        testcase.assertFalse(got["feasible"], msg=f"意外可行: {got}")
        return
    testcase.assertTrue(got["feasible"], msg="意外无解")
    testcase.assertEqual(got["drift"], want["drift"])
    testcase.assertEqual(got["num_skips"], want["num_skips"])
    testcase.assertEqual(got["residual_sum"], want["residual_sum"])
    testcase.assertEqual(got["max_abs_residual"], want["max_abs_residual"])
    testcase.assertEqual(got["boundaries"], want["boundaries"])
    got_used = [lv["reference_index"] for lv in got["levels"]]
    testcase.assertEqual(got_used, want["used_indices"])
    if anchors:
        _assert_anchor_evidence(testcase, got, anchors)


def _assert_anchor_evidence(
    testcase: unittest.TestCase, got: dict, anchors: Sequence[dict]
) -> None:
    """逐项锚定归属证据必须与请求锚点一一对应且落在采样区间内。"""
    assignments = got.get("anchor_assignments")
    testcase.assertIsInstance(assignments, list)
    testcase.assertEqual(len(assignments), len(anchors))
    level_by_order = {lv["level_order"]: lv for lv in got["levels"]}
    for want, ev in zip(anchors, assignments):
        testcase.assertEqual(ev["reference_index"], want["reference_index"])
        testcase.assertEqual(ev["sample_index"], want["sample_index"])
        lv = level_by_order[ev["level_order"]]
        testcase.assertEqual(lv["reference_index"], want["reference_index"])
        testcase.assertEqual(ev["sample_start"], lv["sample_start"])
        testcase.assertEqual(ev["sample_end"], lv["sample_end"])
        testcase.assertLessEqual(lv["sample_start"], want["sample_index"])
        testcase.assertLess(want["sample_index"], lv["sample_end"])
        testcase.assertEqual(ev["adopted_level"], lv["adopted_level"])
        sample_ev = {
            s["index"]: s for s in lv["samples"]
        }[want["sample_index"]]
        testcase.assertEqual(ev["observed"], sample_ev["observed"])
        testcase.assertEqual(ev["residual"], sample_ev["residual"])
        testcase.assertEqual(ev["residual"], ev["observed"] - ev["adopted_level"])


class ExactAlignmentTests(unittest.TestCase):
    def test_wide_drift_interval_finds_far_drift(self) -> None:
        # 真实漂移远离 0，且首级/末级可行漂移区间很窄：
        # 验证首末级预筛不会把正确漂移漏掉。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 432 for x in ref]
        res = solve_alignment(ref, obs, -1000, 1000, 0)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 432)
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)

    def test_first_last_prefilter_conflict_is_infeasible(self) -> None:
        # 首级前几个观测要求漂移 -7 附近，末级观测要求漂移 +7 附近，
        # 首末级可行区间交集为空 -> 明确无解（limit=0，无折中可能）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [-7, -7, -7, 10, 20, 30, 40, 50, 60, 70, 77, 77]
        res = solve_alignment(ref, obs, -100, 100, 0,
                              dwell_min=1, dwell_max=3)
        self.assertFalse(res["feasible"])

    def test_straight_no_skip(self) -> None:
        # 8 个参考电平，每级恰好 1 个观测，漂移 +5，残差全 0。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 5 for x in ref]
        res = solve_alignment(ref, obs, -10, 10, 2)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 5)
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)
        self.assertEqual(res["max_abs_residual"], 0)
        self.assertEqual(res["boundaries"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(res["levels"]), 8)
        for slot, lv in enumerate(res["levels"]):
            self.assertEqual(lv["sample_start"], slot)
            self.assertEqual(lv["sample_end"], slot + 1)
            self.assertEqual(lv["dwell"], 1)
            self.assertEqual(lv["samples"][0]["residual"], 0)

    def test_dwell_two_three_and_skip(self) -> None:
        # 8 个参考电平；跳过索引 3；停留：2,3,2,... 凑 14 个观测。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        used = [0, 1, 2, 4, 5, 6, 7]
        dwells = [2, 3, 2, 2, 2, 2, 1]
        self.assertEqual(sum(dwells), 14)
        obs: List[int] = []
        for ri, L in zip(used, dwells):
            obs.extend([ref[ri] - 3] * L)
        res = solve_alignment(ref, obs, -5, 5, 1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], -3)
        self.assertEqual(res["num_skips"], 1)
        self.assertEqual(res["skipped_reference_indices"], [3])
        self.assertEqual(
            [lv["reference_index"] for lv in res["levels"]], used
        )
        self.assertEqual([lv["dwell"] for lv in res["levels"]], dwells)
        self.assertEqual(res["boundaries"], [2, 5, 7, 9, 11, 13])
        for lv in res["levels"]:
            for s in lv["samples"]:
                self.assertEqual(abs(s["residual"]), 0)

    def test_first_and_last_mandatory(self) -> None:
        # 首电平与尾电平与观测差距巨大，任何跳过都救不了 -> 无解。
        ref = [0, 100, 100, 100, 100, 100, 100, 1000]
        obs = [100] * 8
        res = solve_alignment(ref, obs, 0, 0, 1)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_infeasible_when_residual_limit_tight(self) -> None:
        ref = [0, 1, 2, 3, 4, 5, 6, 7]
        # 每个观测比对应电平高 2；limit=1、漂移只能 0 -> 无解。
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, 0, 0, 1)
        self.assertFalse(res["feasible"])

    def test_infeasible_when_too_many_levels(self) -> None:
        # 8 个观测、8 个必到级数（首尾+内部不允许跳过），每级最多 1 采样可行；
        # 但若禁用跳过且观测只有 8 个、停留最小 2 -> 无解。
        ref = list(range(8))
        obs = list(range(8))
        res = solve_alignment(ref, obs, 0, 0, 0, dwell_min=2, dwell_max=3)
        self.assertFalse(res["feasible"])

    def test_skip_cap_enforced(self) -> None:
        # 需要 3 个跳过才可行，但上限为 2 -> 无解。
        ref = [0, 100, 200, 300, 400, 500, 600, 700]
        obs = [0, 0, 700, 700]  # 仅首尾附近有观测；N 最少为 8，构造 8 个
        obs = [0, 0, 0, 0, 700, 700, 700, 700]
        # 合法对齐至少要跳过中间 6 个内部电平，> 2。
        res = solve_alignment(ref, obs, 0, 0, 0, max_skips=2)
        self.assertFalse(res["feasible"])
        res2 = solve_alignment(ref, obs, 0, 0, 0, max_skips=2,
                               dwell_min=1, dwell_max=3)
        self.assertFalse(res2["feasible"])


class ObjectiveOrderTests(unittest.TestCase):
    def test_minimize_skips_first(self) -> None:
        # 无跳过对齐需要较大残差；带 1 跳过残差为 0。
        # 仍应选择无跳过（跳过数优先级最高）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        # 8 观测每级 1 个，全部偏移 2（limit=2，无跳过，残差和 16）。
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, 0, 0, 5)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 16)

    def test_then_residual_sum(self) -> None:
        # 漂移 -1 / 0 / +1 都可行，残差和不同 -> 选和最小者。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [r - 1 for r in ref]  # drift=-1 时残差为 0
        res = solve_alignment(ref, obs, -2, 2, 5)
        self.assertEqual(res["drift"], -1)
        self.assertEqual(res["residual_sum"], 0)

    def test_then_max_residual(self) -> None:
        # 构造两个漂移残差和相同但最大残差不同：用对称扰动。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        # drift=0: 残差 +1,-1,...,0 -> 和 0（带符号不影响，这里用绝对值）
        obs = [1, 9, 21, 29, 41, 39, 61, 69]
        # drift=0 时绝对残差全 1；和 8、最大 1
        # drift=1 时残差 [0,-2,0,-2,...,±2] 和 16 更大 -> 0 胜
        res = solve_alignment(ref, obs, -1, 1, 3)
        self.assertEqual(res["drift"], 0)
        self.assertEqual(res["max_abs_residual"], 1)
        self.assertEqual(res["residual_sum"], 8)

    def test_then_drift_tie_break(self) -> None:
        # 所有观测恰好在两个漂移下都零残差（参考电平差偶数且观测居中无法同时为0；
        # 改用只有 1 级停留长度结构使得 +0 与 +1 不可能同残差，因此直接构造
        # limit 宽松、残差相同的情形：参考全部相同间距无法做到——
        # 这里验证漂移平局取较小漂移：令观测 = ref + 1，并允许 drift=1 与
        # 跳过路径下 drift=0 残差结构一致较难构造，改为直接校验纯平局：
        # 参考电平全部为偶数，观测相对 ref：+1，此时仅 drift=1 零残差，
        # 退而求其次，校验候选漂移中较小者获胜的代码路径由对照测试覆盖。
        ref = [0, 2, 4, 6, 8, 10, 12, 14]
        obs = [r + 1 for r in ref]
        res = solve_alignment(ref, obs, 0, 2, 1)
        self.assertEqual(res["drift"], 1)

    def test_boundaries_lexicographic_tie_break(self) -> None:
        # 全部相同参考电平：任意边界划分残差相同；应取字典序最小边界
        # （尽早结束第一级：在 dwell_min=1 下首条边界最小）。
        ref = [5] * 8
        obs = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # 10 观测，8 级
        res = solve_alignment(ref, obs, 0, 0, 0, dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        # 额外的 2 个采样尽量后置 -> 停留 (1,1,1,1,1,1,1,3)，
        # 内部边界字典序最小。
        self.assertEqual(res["boundaries"], [1, 2, 3, 4, 5, 6, 7])

    def test_boundary_tie_break_not_shadowed_by_prefix_max(self) -> None:
        # 回归：两条前缀的 (跳过数, 残差和) 相同但前缀最大残差不同
        # （2 vs 3），而末级残差 8 把两者最终最大残差都拉平到 8；
        # 此时必须按边界字典序裁决，不能因前缀最大残差较差而提前
        # 剪掉字典序更小的分支。
        ref = [10, 38, -24, -22, -20, -21, 13, 40, 48]
        obs = [14, 44, -20, -18, -17, -19, -15, -16, 42, 42, 44, 51]
        res = solve_alignment(ref, obs, 2, 5, 8,
                              dwell_min=1, dwell_max=2, max_skips=1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["num_skips"], 1)
        self.assertEqual(res["residual_sum"], 21)
        self.assertEqual(res["max_abs_residual"], 8)
        self.assertEqual(res["drift"], 4)
        self.assertEqual(res["boundaries"], [1, 2, 3, 5, 7, 8, 10])


class ValidationTests(unittest.TestCase):
    def _base(self) -> dict:
        return dict(
            reference=list(range(8)),
            observations=list(range(8)),
            drift_min=0,
            drift_max=0,
            residual_limit=0,
        )

    def test_reference_size_bounds(self) -> None:
        kw = self._base()
        kw["reference"] = list(range(7))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["reference"] = list(range(25))
        kw["observations"] = list(range(25))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_observation_size_bounds(self) -> None:
        kw = self._base()
        kw["observations"] = list(range(7))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["observations"] = list(range(61))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_drift_interval(self) -> None:
        kw = self._base()
        kw["drift_min"], kw["drift_max"] = 5, 4
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_dwell_range(self) -> None:
        kw = self._base()
        kw["dwell_min"], kw["dwell_max"] = 0, 3
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["dwell_min"], kw["dwell_max"] = 2, 1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["dwell_min"], kw["dwell_max"] = 1, 4
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_max_skips_range(self) -> None:
        kw = self._base()
        kw["max_skips"] = 3
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["max_skips"] = -1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_drift_width_cap(self) -> None:
        from nanopore_align.alignment import DRIFT_WIDTH_MAX

        kw = self._base()
        kw["drift_min"] = 0
        kw["drift_max"] = DRIFT_WIDTH_MAX
        solve_alignment(**kw)  # 边界宽度合法
        kw["drift_max"] = DRIFT_WIDTH_MAX + 1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_negative_residual_limit_rejected(self) -> None:
        kw = self._base()
        kw["residual_limit"] = -1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_type_checks(self) -> None:
        kw = self._base()
        kw["reference"] = [1, 2, 3, 4, 5, 6, 7, "8"]  # type: ignore[list-item]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["drift_min"] = 0.5  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["observations"] = True  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["reference"] = []  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)


class ResidualEvidenceTests(unittest.TestCase):
    def test_evidence_covers_every_observation_once(self) -> None:
        ref = [3, 7, 11, 15, 19, 23, 27, 31]
        obs = [3, 3, 8, 10, 16, 19, 22, 24, 28, 30, 30, 31]
        res = solve_alignment(ref, obs, -2, 2, 2,
                              dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        covered: List[int] = []
        for lv in res["levels"]:
            self.assertEqual(lv["sample_end"] - lv["sample_start"], lv["dwell"])
            self.assertEqual(len(lv["samples"]), lv["dwell"])
            for s in lv["samples"]:
                self.assertEqual(
                    s["residual"], s["observed"] - lv["adopted_level"]
                )
                self.assertLessEqual(abs(s["residual"]), 2)
                covered.append(s["index"])
        self.assertEqual(covered, list(range(len(obs))))
        self.assertEqual(
            res["residual_sum"],
            sum(abs(s["residual"]) for lv in res["levels"] for s in lv["samples"]),
        )
        self.assertEqual(
            res["max_abs_residual"],
            max(abs(s["residual"]) for lv in res["levels"] for s in lv["samples"]),
        )


class AnchoredAlignmentTests(unittest.TestCase):
    """硬锚点：锚定级不可跳过、锚定观测必须归属该级采样区间。"""

    def test_anchor_satisfied_keeps_result_and_gives_evidence(self) -> None:
        # 锚点被无锚点最优解自然满足：裁决结果不变，且给出逐项归属证据。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [12, 19, 31, 38, 52, 58, 71, 79]
        anchors = [
            {"reference_index": 2, "sample_index": 2},
            {"reference_index": 5, "sample_index": 5},
        ]
        base = solve_alignment(ref, obs, -5, 5, 2, dwell_min=1, dwell_max=1)
        res = solve_alignment(
            ref, obs, -5, 5, 2, dwell_min=1, dwell_max=1, anchors=anchors
        )
        self.assertTrue(res["feasible"])
        # 无锚点响应结构保持不变；有锚点仅追加 anchor_assignments。
        self.assertNotIn("anchor_assignments", base)
        for key in (
            "drift",
            "num_skips",
            "skipped_reference_indices",
            "residual_sum",
            "max_abs_residual",
            "boundaries",
            "num_levels_used",
        ):
            self.assertEqual(res[key], base[key])
        ev0, ev1 = res["anchor_assignments"]
        self.assertEqual((ev0["reference_index"], ev0["sample_index"]), (2, 2))
        self.assertEqual(ev0["level_order"], 2)
        self.assertEqual((ev0["sample_start"], ev0["sample_end"]), (2, 3))
        self.assertEqual(ev0["adopted_level"], 30)
        self.assertEqual(ev0["observed"], 31)
        self.assertEqual(ev0["residual"], 1)
        self.assertEqual((ev1["reference_index"], ev1["sample_index"]), (5, 5))
        self.assertEqual(ev1["level_order"], 5)
        self.assertEqual(ev1["residual"], -2)

    def test_anchor_overrides_lexicographic_boundary_tie_break(self) -> None:
        # 无锚点最优取字典序最小边界 [1..7]；锚点 (3, 5) 迫使第 4 级
        # 区间覆盖样本 5，边界变为 [1,2,3,6,7,8,9]，五级裁决顺序不变。
        ref = [5] * 8
        obs = [5] * 10
        base = solve_alignment(ref, obs, 0, 0, 0, dwell_min=1, dwell_max=3)
        self.assertEqual(base["boundaries"], [1, 2, 3, 4, 5, 6, 7])
        res = solve_alignment(
            ref,
            obs,
            0,
            0,
            0,
            dwell_min=1,
            dwell_max=3,
            anchors=[{"reference_index": 3, "sample_index": 5}],
        )
        self.assertTrue(res["feasible"])
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)
        self.assertEqual(res["boundaries"], [1, 2, 3, 6, 7, 8, 9])
        ev = res["anchor_assignments"][0]
        self.assertEqual((ev["sample_start"], ev["sample_end"]), (3, 6))
        self.assertLessEqual(ev["sample_start"], 5)
        self.assertLess(5, ev["sample_end"])

    def test_anchored_level_cannot_be_skipped(self) -> None:
        # 无锚点最优（残差 0）恰好跳过参考级 2；锚定级 2 后该方案被
        # 排除——锚点优先于残差最优，本例中不存在其他合法对齐。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [0, 10, 30, 30, 40, 50, 60, 70, 70]
        base = solve_alignment(ref, obs, 0, 0, 2, dwell_min=1, dwell_max=2)
        self.assertTrue(base["feasible"])
        self.assertEqual(base["num_skips"], 1)
        self.assertEqual(base["skipped_reference_indices"], [2])
        res = solve_alignment(
            ref,
            obs,
            0,
            0,
            2,
            dwell_min=1,
            dwell_max=2,
            anchors=[{"reference_index": 2, "sample_index": 2}],
        )
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_with_anchors")
        self.assertIn("message", res)

    def test_anchor_forces_level_and_alignment_stays_feasible(self) -> None:
        # 无锚点最优跳过级 2（残差 0）；锚点 (2, 2) 后级 2 必须采用，
        # 最优解改为跳过级 3（残差和 0 -> 1），跳过数与裁决顺序不变。
        ref = [0, 10, 20, 21, 40, 50, 60, 70]
        obs = [0, 10, 21, 40, 50, 60, 70, 70]
        base = solve_alignment(ref, obs, 0, 0, 2, dwell_min=1, dwell_max=2)
        self.assertTrue(base["feasible"])
        self.assertEqual(base["num_skips"], 1)
        self.assertEqual(base["skipped_reference_indices"], [2])
        self.assertEqual(base["residual_sum"], 0)
        res = solve_alignment(
            ref,
            obs,
            0,
            0,
            2,
            dwell_min=1,
            dwell_max=2,
            anchors=[{"reference_index": 2, "sample_index": 2}],
        )
        self.assertTrue(res["feasible"])
        self.assertEqual(res["num_skips"], 1)
        self.assertEqual(res["skipped_reference_indices"], [3])
        used = [lv["reference_index"] for lv in res["levels"]]
        self.assertIn(2, used)
        ev = res["anchor_assignments"][0]
        self.assertEqual((ev["reference_index"], ev["sample_index"]), (2, 2))
        self.assertEqual((ev["sample_start"], ev["sample_end"]), (2, 3))
        self.assertEqual(ev["adopted_level"], 20)
        self.assertEqual(ev["observed"], 21)
        self.assertEqual(ev["residual"], 1)
        # 与暴力枚举的带锚最优完全一致。
        want = brute_force(ref, obs, 0, 0, 2, 1, 2, 2,
                           anchors=[{"reference_index": 2, "sample_index": 2}])
        self.assertIsNotNone(want)
        self.assertEqual(used, want["used_indices"])
        self.assertEqual(res["num_skips"], want["num_skips"])
        self.assertEqual(res["residual_sum"], want["residual_sum"])

    def test_anchor_on_first_and_last_level(self) -> None:
        # 首级与末级锚点：区间分别须覆盖样本 0 与样本 N-1。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [10, 10, 20, 30, 40, 50, 60, 70, 80, 80]
        anchors = [
            {"reference_index": 0, "sample_index": 1},
            {"reference_index": 7, "sample_index": 8},
        ]
        res = solve_alignment(
            ref, obs, 0, 0, 0, dwell_min=1, dwell_max=2, anchors=anchors
        )
        self.assertTrue(res["feasible"])
        ev_first, ev_last = res["anchor_assignments"]
        self.assertEqual(ev_first["level_order"], 0)
        self.assertEqual((ev_first["sample_start"], ev_first["sample_end"]), (0, 2))
        self.assertEqual(ev_last["reference_index"], 7)
        self.assertEqual(
            (ev_last["sample_start"], ev_last["sample_end"]), (8, 10)
        )

    def test_three_anchors(self) -> None:
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [12, 19, 31, 38, 52, 58, 71, 79]
        anchors = [
            {"reference_index": 0, "sample_index": 0},
            {"reference_index": 3, "sample_index": 3},
            {"reference_index": 7, "sample_index": 7},
        ]
        res = solve_alignment(
            ref, obs, -5, 5, 2, dwell_min=1, dwell_max=1, anchors=anchors
        )
        self.assertTrue(res["feasible"])
        self.assertEqual(len(res["anchor_assignments"]), 3)

    def test_infeasible_anchor_sample_out_of_level_reach(self) -> None:
        # dwell 固定为 1 时样本 7 只能归属末级，锚定到首级必然不相容。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [12, 19, 31, 38, 52, 58, 71, 79]
        res = solve_alignment(
            ref,
            obs,
            -5,
            5,
            2,
            dwell_min=1,
            dwell_max=1,
            anchors=[{"reference_index": 0, "sample_index": 7}],
        )
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_with_anchors")


class AnchorValidationTests(unittest.TestCase):
    """锚点格式错误：越界、重复、次序冲突均为字段级 AlignmentError。"""

    def _base(self) -> dict:
        return dict(
            reference=list(range(8)),
            observations=list(range(8)),
            drift_min=0,
            drift_max=0,
            residual_limit=0,
        )

    def test_reference_index_out_of_range(self) -> None:
        kw = self._base()
        kw["anchors"] = [{"reference_index": 8, "sample_index": 0}]
        with self.assertRaises(AnchorValidationError) as ctx:
            solve_alignment(**kw)
        self.assertEqual(ctx.exception.field, "anchors[0].reference_index")
        kw["anchors"] = [{"reference_index": -1, "sample_index": 0}]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)

    def test_sample_index_out_of_range(self) -> None:
        kw = self._base()
        kw["anchors"] = [{"reference_index": 0, "sample_index": 8}]
        with self.assertRaises(AnchorValidationError) as ctx:
            solve_alignment(**kw)
        self.assertEqual(ctx.exception.field, "anchors[0].sample_index")
        kw["anchors"] = [{"reference_index": 0, "sample_index": -2}]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)

    def test_duplicate_indices_rejected(self) -> None:
        kw = self._base()
        kw["anchors"] = [
            {"reference_index": 2, "sample_index": 2},
            {"reference_index": 2, "sample_index": 5},
        ]
        with self.assertRaises(AnchorValidationError) as ctx:
            solve_alignment(**kw)
        self.assertEqual(ctx.exception.field, "anchors[1].reference_index")
        kw["anchors"] = [
            {"reference_index": 2, "sample_index": 2},
            {"reference_index": 4, "sample_index": 2},
        ]
        with self.assertRaises(AnchorValidationError) as ctx:
            solve_alignment(**kw)
        self.assertEqual(ctx.exception.field, "anchors[1].sample_index")

    def test_order_conflicts_rejected(self) -> None:
        kw = self._base()
        kw["anchors"] = [
            {"reference_index": 4, "sample_index": 2},
            {"reference_index": 2, "sample_index": 5},
        ]
        with self.assertRaises(AnchorValidationError) as ctx:
            solve_alignment(**kw)
        self.assertEqual(ctx.exception.field, "anchors[1].reference_index")
        kw["anchors"] = [
            {"reference_index": 2, "sample_index": 5},
            {"reference_index": 4, "sample_index": 2},
        ]
        with self.assertRaises(AnchorValidationError) as ctx:
            solve_alignment(**kw)
        self.assertEqual(ctx.exception.field, "anchors[1].sample_index")

    def test_shape_and_type_errors(self) -> None:
        kw = self._base()
        kw["anchors"] = []
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = [
            {"reference_index": i, "sample_index": i} for i in range(4)
        ]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = "not-a-list"  # type: ignore[assignment]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = [5]  # type: ignore[list-item]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = [{"reference_index": 0}]  # 缺 sample_index
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = [
            {"reference_index": 0, "sample_index": 0, "extra": 1}
        ]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = [{"reference_index": 0.5, "sample_index": 0}]  # type: ignore[dict-item]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)
        kw["anchors"] = [{"reference_index": 0, "sample_index": True}]  # type: ignore[dict-item]
        with self.assertRaises(AnchorValidationError):
            solve_alignment(**kw)

    def test_anchor_error_is_alignment_error(self) -> None:
        kw = self._base()
        kw["anchors"] = [{"reference_index": 99, "sample_index": 0}]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)


class AnchoredBruteForceComparisonTests(unittest.TestCase):
    """随机带锚实例：DP 必须与穷举结果完全一致（含锚点归属证据）。"""

    def test_random_anchored_cases(self) -> None:
        rng = random.Random(20261005)
        for trial in range(160):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
            dwells = BruteForceComparisonTests._random_composition(
                rng, k, N, 1, 3
            )
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 0, 1, -1, 2, -2, 5])
                    obs.append(level + noise)
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            anchors = self._random_anchors(rng, used, dwells, R, N)
            with self.subTest(trial=trial, ref=ref, obs=obs, anchors=anchors):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2,
                    anchors=anchors,
                )

    @staticmethod
    def _random_anchors(
        rng: random.Random,
        used: List[int],
        dwells: List[int],
        R: int,
        N: int,
    ) -> List[dict]:
        """一半概率从真值对齐采锚点（必相容），一半纯随机（可能不相容）。"""
        k = rng.randint(1, 3)
        if rng.random() < 0.5:
            # 真值各级区间互不重叠且递增，样本天然互异且随级递增。
            slots = rng.sample(range(len(used)), min(k, len(used)))
            slots.sort()
            anchors = []
            start = 0
            starts: List[int] = []
            for L in dwells:
                starts.append(start)
                start += L
            for slot in slots:
                t = rng.randrange(starts[slot], starts[slot] + dwells[slot])
                anchors.append(
                    {"reference_index": used[slot], "sample_index": t}
                )
            return anchors
        refs = sorted(rng.sample(range(R), k))
        samples = sorted(rng.sample(range(N), k))
        return [
            {"reference_index": ri, "sample_index": si}
            for ri, si in zip(refs, samples)
        ]


class BruteForceComparisonTests(unittest.TestCase):
    """随机小规模实例：DP 必须与穷举结果完全一致（含全部平局裁决）。"""

    def test_random_cases(self) -> None:
        rng = random.Random(20261004)
        for trial in range(120):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            # 先随机生成一个“真值”对齐，再对部分观测加入噪声，
            # 保证可行与不可行实例混合出现。
            used = [0]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            # 随机跳过 0..2 个内部电平
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
                skipped_set = set()
            # 随机停留组合，和为 N，每段 1..3
            dwells = self._random_composition(rng, k, N, 1, 3)
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 0, 1, -1, 2, -2, 5])
                    obs.append(level + noise)
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            dwell_min = rng.randint(1, 2)
            dwell_max = rng.randint(max(dwell_min, 2), 3)
            max_skips = rng.randint(0, 2)
            with self.subTest(trial=trial, ref=ref, obs=obs,
                              lo=d_lo, hi=d_hi, limit=limit,
                              dwell=(dwell_min, dwell_max),
                              max_skips=max_skips):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit,
                    dwell_min, dwell_max, max_skips,
                )

    @staticmethod
    def _random_composition(
        rng: random.Random, k: int, n: int, lo: int, hi: int
    ) -> Optional[List[int]]:
        choices: List[List[int]] = []
        total_lo = k * lo
        total_hi = k * hi
        if not total_lo <= n <= total_hi:
            return None
        # 在合法空间内简单拒绝采样。
        for _ in range(500):
            parts = [rng.randint(lo, hi) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


class WideDriftRandomTests(unittest.TestCase):
    """宽漂移区间下随机真值实例：预筛与枚举必须找回唯一真值对齐。"""

    def test_wide_interval_random_truth(self) -> None:
        rng = random.Random(424242)
        for _ in range(40):
            R = rng.randint(8, 24)
            k = R - rng.randint(0, 2)
            skip_set = set(rng.sample(range(1, R - 1), R - k))
            used = [i for i in range(R) if i not in skip_set]
            dwells = self._composition(rng, k, rng.randint(k, 3 * k))
            if dwells is None:
                continue
            N = sum(dwells)
            if N > 60:
                continue
            ref = sorted(rng.sample(range(-20000, 20000), R))
            truth_d = rng.randint(-500, 500)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                obs.extend([ref[ri] + truth_d] * L)
            with self.subTest(R=R, N=N, d=truth_d):
                res = solve_alignment(
                    ref, obs, -1000, 1000, 0, 1, 3, 2
                )
                self.assertTrue(res["feasible"], msg="漏掉可行真值对齐")
                self.assertEqual(res["drift"], truth_d)
                self.assertEqual(res["residual_sum"], 0)
                self.assertEqual(
                    [lv["reference_index"] for lv in res["levels"]], used
                )
                self.assertEqual(
                    [lv["dwell"] for lv in res["levels"]], dwells
                )

    @staticmethod
    def _composition(
        rng: random.Random, k: int, n: int
    ) -> Optional[List[int]]:
        for _ in range(800):
            parts = [rng.randint(1, 3) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


if __name__ == "__main__":
    unittest.main()
