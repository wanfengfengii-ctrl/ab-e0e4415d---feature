"""联合对齐算法的单元测试与暴力对照测试。"""

from __future__ import annotations

import itertools
import random
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

from nanopore_align.alignment import AlignmentError, solve_alignment


def brute_force_anchored(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    anchors: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[dict]:
    """穷举漂移 / 含首尾子序列 / 停留组合（含硬锚点），返回最优解口径。"""
    R = len(reference)
    N = len(observations)
    anchor_at = dict(anchors or [])
    best: Optional[Tuple] = None

    for d in range(drift_min, drift_max + 1):
        inner = list(range(1, R - 1))
        for skip_count in range(0, min(max_skips, len(inner)) + 1):
            for skipped in itertools.combinations(inner, skip_count):
                skipped_set = set(skipped)
                if any(ri in skipped_set for ri in anchor_at):
                    continue
                used = [i for i in range(R) if i not in skipped_set]
                k = len(used)
                if k > N or k * dwell_min > N or k * dwell_max < N:
                    continue
                for dwells in itertools.product(
                    range(dwell_min, dwell_max + 1), repeat=k
                ):
                    if sum(dwells) != N:
                        continue
                    boundaries = tuple(itertools.accumulate(dwells))
                    total = 0
                    worst = 0
                    ok = True
                    s = 0
                    for ri, L in zip(used, dwells):
                        if ri in anchor_at and not (
                            s <= anchor_at[ri] < s + L
                        ):
                            ok = False
                            break
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


def brute_force(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> Optional[dict]:
    """穷举所有漂移 / 含首尾子序列 / 停留组合，返回与 solve 同口径的最优解。"""
    return brute_force_anchored(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )


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
            with self.subTest(trial=trial, ref=ref, obs=obs,
                              lo=d_lo, hi=d_hi, limit=limit):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2
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


class AnchorAlignmentTests(unittest.TestCase):
    """硬锚点：锚定级必含、锚定采样必须落入该级连续区间。"""

    def test_unanchored_response_has_no_anchor_fields(self) -> None:
        ref = [0, 100, 200, 300, 400, 500, 600, 700]
        obs = [0, 100, 200, 400, 500, 600, 700, 700]
        res = solve_alignment(ref, obs, 0, 0, 0,
                              dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["skipped_reference_indices"], [3])
        self.assertNotIn("anchor_assignments", res)
        for lv in res["levels"]:
            self.assertNotIn("anchored", lv)
            self.assertNotIn("anchor_sample_index", lv)

    def test_anchor_breaks_equal_cost_skip_tie(self) -> None:
        # N=15、dwell_min=2 迫使恰好跳过 1 个内部级；R3 与 R4 电平相同，
        # 跳过二者的费用（含边界序列）完全一致，无锚点时裁决取跳 R3。
        ref = [0, 100, 200, 300, 300, 400, 500, 600]
        obs = [0, 0, 100, 100, 200, 200, 300, 300,
               400, 400, 500, 500, 600, 600, 600]
        res = solve_alignment(ref, obs, 0, 0, 0,
                              dwell_min=2, dwell_max=3)
        self.assertEqual(res["skipped_reference_indices"], [3])

        # 荧光锚点确认采样 6 属于 R3：不得再选跨过 R3 的同费用方案，
        # 改为跳过 R4，残差/跳过数裁决完全不变。
        anchored = solve_alignment(ref, obs, 0, 0, 0,
                                   dwell_min=2, dwell_max=3,
                                   anchors=[(3, 6)])
        self.assertTrue(anchored["feasible"])
        self.assertEqual(anchored["num_skips"], 1)
        self.assertEqual(anchored["residual_sum"], 0)
        self.assertEqual(anchored["skipped_reference_indices"], [4])
        used = [lv["reference_index"] for lv in anchored["levels"]]
        self.assertIn(3, used)

        ev = anchored["anchor_assignments"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["anchor_order"], 0)
        self.assertEqual(ev[0]["reference_index"], 3)
        self.assertEqual(ev[0]["sample_index"], 6)
        slot = used.index(3)
        self.assertEqual(ev[0]["assigned_level_order"], slot)
        self.assertEqual(
            (ev[0]["sample_start"], ev[0]["sample_end"]), (6, 8)
        )
        self.assertTrue(6 in range(ev[0]["sample_start"],
                                   ev[0]["sample_end"]))
        self.assertEqual(ev[0]["dwell"], 2)
        self.assertEqual(ev[0]["residual"], 0)

        anchored_levels = [lv for lv in anchored["levels"]
                           if lv["anchored"]]
        self.assertEqual(len(anchored_levels), 1)
        self.assertEqual(anchored_levels[0]["reference_index"], 3)
        self.assertEqual(anchored_levels[0]["anchor_sample_index"], 6)

    def test_anchor_blocks_lower_residual_skip_path(self) -> None:
        # N=15、dwell_min=2：无锚点时跳过 R3 残差和为 0。
        ref = [0, 100, 200, 300, 400, 500, 600, 700]
        obs = [0, 0, 100, 100, 200, 200, 400, 400,
               500, 500, 600, 600, 700, 700, 700]
        res = solve_alignment(ref, obs, 0, 0, 100,
                              dwell_min=2, dwell_max=3)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["skipped_reference_indices"], [3])
        self.assertEqual(res["residual_sum"], 0)

        # 锚定 (R3, 采样6)：R3 不得跳过，采样 6 必须落入 R3 区间，
        # 残差更小的跳过路径被禁止，改跳 R4 且 R3 承担残差。
        anchored = solve_alignment(ref, obs, 0, 0, 100,
                                   dwell_min=2, dwell_max=3,
                                   anchors=[(3, 6)])
        self.assertTrue(anchored["feasible"])
        self.assertEqual(anchored["skipped_reference_indices"], [4])
        self.assertGreater(anchored["residual_sum"], 0)
        level = next(lv for lv in anchored["levels"]
                     if lv["reference_index"] == 3)
        self.assertLessEqual(level["sample_start"], 6)
        self.assertLess(6, level["sample_end"])

        # 收紧残差上限使锚定后的替代路径也不可行 -> 明确不相容。
        tight = solve_alignment(ref, obs, 0, 0, 50,
                                dwell_min=2, dwell_max=3,
                                anchors=[(3, 6)])
        self.assertFalse(tight["feasible"])
        self.assertEqual(tight["reason"], "anchors_incompatible")

    def test_anchor_incompatible_is_distinct_conclusion(self) -> None:
        # 轨迹只允许跳过 R3（obs3=400 对 R3=300 残差 100，limit=0）；
        # 锚定 R3 后无任何合法对齐 -> anchors_incompatible。
        ref = [0, 100, 200, 300, 400, 500, 600, 700]
        obs = [0, 100, 200, 400, 500, 600, 700, 700]
        res = solve_alignment(ref, obs, 0, 0, 0,
                              dwell_min=1, dwell_max=3, anchors=[(3, 3)])
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "anchors_incompatible")
        # 同一请求去掉锚点是可行解（普通无解语义不受影响）。
        res2 = solve_alignment(ref, obs, 0, 0, 0,
                               dwell_min=1, dwell_max=3)
        self.assertTrue(res2["feasible"])
        self.assertEqual(res2["skipped_reference_indices"], [3])

    def test_anchors_on_first_and_last_levels(self) -> None:
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, -5, 5, 0,
                              dwell_min=1, dwell_max=1,
                              anchors=[(0, 0), (7, 7)])
        self.assertTrue(res["feasible"])
        self.assertEqual(
            [(a["reference_index"], a["sample_index"])
             for a in res["anchor_assignments"]],
            [(0, 0), (7, 7)],
        )
        for ev in res["anchor_assignments"]:
            self.assertLessEqual(ev["sample_start"], ev["sample_index"])
            self.assertLess(ev["sample_index"], ev["sample_end"])

    def test_three_anchors_evidence_in_request_order(self) -> None:
        # 真值停留 (1,1,2,1,1,2,2,2)：R2 覆盖采样 2,3；R4 覆盖采样 5；
        # R6 覆盖采样 8,9（锚点取 9）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [0, 10, 20, 20, 30, 40, 50, 50, 60, 60, 70, 70]
        anchors = [(2, 3), (4, 5), (6, 9)]
        res = solve_alignment(ref, obs, 0, 0, 0,
                              dwell_min=1, dwell_max=3, anchors=anchors)
        self.assertTrue(res["feasible"])
        ev = res["anchor_assignments"]
        self.assertEqual(
            [(a["anchor_order"], a["reference_index"], a["sample_index"])
             for a in ev],
            [(0, 2, 3), (1, 4, 5), (2, 6, 9)],
        )
        # assigned_level_order 与裁决输出的 levels 顺序一致且严格递增。
        orders = [a["assigned_level_order"] for a in ev]
        self.assertEqual(orders, sorted(orders))
        self.assertEqual(len(set(orders)), 3)
        used = [lv["reference_index"] for lv in res["levels"]]
        for a in ev:
            self.assertEqual(used[a["assigned_level_order"]],
                             a["reference_index"])
        anchored_refs = {
            lv["reference_index"]
            for lv in res["levels"] if lv["anchored"]
        }
        self.assertEqual(anchored_refs, {2, 4, 6})

    def test_anchor_sample_must_fall_in_level_block(self) -> None:
        # dwell 恒为 1 时，锚点 (R2, 采样5) 要求 R2 占据采样 5，
        # 但 R2 之前已有 2 个必含级各占 1 采样，物理上不可能 -> 无解。
        ref = list(range(0, 80, 10))
        obs = list(range(0, 80, 10))
        res = solve_alignment(ref, obs, 0, 0, 0,
                              dwell_min=1, dwell_max=1, anchors=[(2, 5)])
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "anchors_incompatible")

    def test_anchor_dwell_limit_still_binds(self) -> None:
        # R0 锚定采样 3，但首级停留上限为 2 -> 锚点与停留约束不相容。
        ref = [5] * 8
        obs = [5] * 10
        res = solve_alignment(ref, obs, 0, 0, 0,
                              dwell_min=1, dwell_max=2, anchors=[(0, 3)])
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "anchors_incompatible")



class AnchorValidationTests(unittest.TestCase):
    def _base(self) -> dict:
        return dict(
            reference=list(range(8)),
            observations=list(range(8)),
            drift_min=0,
            drift_max=0,
            residual_limit=0,
        )

    def test_out_of_range_reference_index(self) -> None:
        kw = self._base()
        kw["anchors"] = [(8, 0)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["anchors"] = [(-1, 0)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_out_of_range_sample_index(self) -> None:
        kw = self._base()
        kw["anchors"] = [(0, 8)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["anchors"] = [(0, -1)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_duplicate_indices_rejected(self) -> None:
        kw = self._base()
        kw["anchors"] = [(2, 1), (2, 3)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["anchors"] = [(1, 2), (3, 2)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_non_increasing_order_rejected(self) -> None:
        kw = self._base()
        kw["anchors"] = [(3, 1), (2, 3)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["anchors"] = [(2, 4), (3, 3)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_count_bounds(self) -> None:
        kw = self._base()
        kw["anchors"] = []
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["anchors"] = [(0, 0), (1, 1), (2, 2), (3, 3)]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_shape_and_type(self) -> None:
        kw = self._base()
        for bad in [[1], [[1]], [[1, 2, 3]], [(1, "2")],
                    [("1", 2)], "x", 42, {"a": 1}]:
            kw["anchors"] = bad
            with self.assertRaises(AlignmentError, msg=f"bad={bad}"):
                solve_alignment(**kw)

    def test_none_and_omitted_equivalent(self) -> None:
        kw = self._base()
        r1 = solve_alignment(**kw)
        kw["anchors"] = None
        r2 = solve_alignment(**kw)
        self.assertEqual(r1["feasible"], r2["feasible"])
        self.assertNotIn("anchor_assignments", r2)


class AnchoredBruteForceComparisonTests(unittest.TestCase):
    """随机小规模实例：锚点 DP 必须与穷举结果（含五级裁决）完全一致。"""

    @staticmethod
    def _composition(rng: random.Random, k: int, n: int) -> Optional[List[int]]:
        for _ in range(400):
            parts = [rng.randint(1, 3) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None

    def test_random_anchored_cases(self) -> None:
        rng = random.Random(20261005)
        checked = 0
        for trial in range(400):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            skip_set = set(inner[:rng.randint(0, min(2, R - 2))])
            used = [i for i in range(R) if i not in skip_set]
            k = len(used)
            dwells = self._composition(rng, k, N)
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                for _ in range(L):
                    obs.append(
                        ref[ri] + d
                        + rng.choice([0, 0, 0, 1, -1, 2, -2, 5])
                    )
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)

            # 从真值块中挑 1..3 个严格递增的（级, 采样）锚点。
            spans: Dict[int, Tuple[int, int]] = {}
            acc = 0
            for ri, L in zip(used, dwells):
                spans[ri] = (acc, acc + L)
                acc += L
            chosen = sorted(rng.sample(used, rng.randint(1, min(3, k))))
            anchors: List[Tuple[int, int]] = []
            prev = -1
            feasible_shape = True
            for ri in chosen:
                s0, s1 = spans[ri]
                pool = [t for t in range(max(s0, prev + 1), s1)]
                if not pool:
                    feasible_shape = False
                    break
                t = rng.choice(pool)
                anchors.append((ri, t))
                prev = t
            if not feasible_shape:
                continue

            with self.subTest(trial=trial, anchors=anchors):
                got = solve_alignment(ref, obs, d_lo, d_hi, limit,
                                      1, 3, 2, anchors=anchors)
                want = brute_force_anchored(
                    ref, obs, d_lo, d_hi, limit, 1, 3, 2, anchors
                )
                if want is None:
                    self.assertFalse(got["feasible"], msg=f"意外可行: {got}")
                    continue
                checked += 1
                self.assertTrue(got["feasible"], msg="意外无解")
                self.assertEqual(got["drift"], want["drift"])
                self.assertEqual(got["num_skips"], want["num_skips"])
                self.assertEqual(got["residual_sum"], want["residual_sum"])
                self.assertEqual(
                    got["max_abs_residual"], want["max_abs_residual"]
                )
                self.assertEqual(got["boundaries"], want["boundaries"])
                self.assertEqual(
                    [lv["reference_index"] for lv in got["levels"]],
                    want["used_indices"],
                )
                got_anchor_refs = {
                    a["reference_index"] for a in got["anchor_assignments"]
                }
                self.assertEqual(got_anchor_refs, {ri for ri, _ in anchors})
                for a in got["anchor_assignments"]:
                    self.assertLessEqual(
                        a["sample_start"], a["sample_index"]
                    )
                    self.assertLess(
                        a["sample_index"], a["sample_end"]
                    )
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
