"""联合对齐算法。

问题定义
========

给定：

* ``reference``: R 个参考整数电平 ``R[0..R-1]``（8 <= R <= 24）；
* ``observations``: N 个观测整数 ``O[0..N-1]``（8 <= N <= 60）；
* 统一漂移闭区间 ``[drift_min, drift_max]``，漂移 d 必须为其中的整数，
  区间宽度不超过 :data:`DRIFT_WIDTH_MAX`；
* 残差上限 ``residual_limit``，每个采用电平 L = R[i] + d 必须满足
  ``|O[t] - L| <= residual_limit``；
* 每级停留采样数范围 ``[dwell_min, dwell_max]``（1 <= ... <= 3）；
* 最多 ``max_skips`` 个内部跳过（0..2，默认 2；跳过的是参考电平，
  且只能发生在首尾之间）。

求解：联合选择一个整数漂移 d、一个**必含首尾**的参考子序列
``R[i_0]=R[0], R[i_1], ..., R[i_{k-1}]=R[R-1]``（相邻索引严格递增）
以及连续采样边界 ``0 = s_0 < s_1 < ... < s_k = N``，令第 j 级占据
连续采样区间 ``[s_j, s_{j+1})``，每个观测恰好归属一个采用电平，
停留长度 ``s_{j+1} - s_j in [dwell_min, dwell_max]``，区间内每个观测
相对该电平残差不越限。

目标按顺序最小化：

1. 跳过数 ``(R-1) - (k-1)``（等价于最小化级数 k）；
2. 残差绝对值总和；
3. 最大残差；
4. 漂移 d；
5. 边界序列（``s_1, ..., s_{k-1}``）字典序。

硬锚点（可选，1..3 个）
========================

``anchors`` 给出若干 ``(reference_index, sample_index)`` 标记（零基），
来自同步荧光通道的可信事件：

* ``reference_index`` 两两互异且严格递增，``sample_index`` 同样两两互异且
  严格递增，标记顺序即参考级与采样的共同次序；
* 每个被锚定的参考级**必含**（不得被跳过），其连续采样区间
  ``[s_j, s_{j+1})`` 必须包含被锚定的观测 ``sample_index``
  （即 ``s_j <= sample_index < s_{j+1}``）；
* 漂移、停留、残差与内部跳过上限等原有约束仍然全部生效；
* 标记本身只合法但无任何对齐能同时满足锚点与原约束时，返回
  ``feasible=False``、``reason="anchors_incompatible"`` 的明确无解结论，
  与普通无解（``no_alignment_exists``）区分，便于质控人员分辨标记录入
  错误与实验轨迹不相容。

实现要点
========

* **逐漂移 DP**：漂移必须沿整条路径一致，因此对每个候选整数漂移
  独立做一次 DP。``grid[j][i]`` 表示观测前缀 ``O[0..j)`` 恰好在参考
  电平 R[i] 结束时的最优代价；首级必须是 R[0]，末级必须是 R[R-1]，
  首尾因此强制必用。因最多 2 个内部跳过，前驱只需考虑 i-1/i-2/i-3。
* **锚点 DP**：锚定参考级被标记为必含（其前驱/本级枚举被相应收紧），
  且只有覆盖其锚定采样的本级块才允许落状态；首级与末级锚点进一步
  收紧首块长度与末块边界；首末级可行漂移预筛也与锚点块区间取交。
* **块可行区间预计算**：对采样块 [j-L, j) 归属 R[i]，令
  c_t = O[t] - R[i]，可行漂移为整数闭区间
  ``[max c_t - lim, min c_t + lim]``，与漂移无关，只算一次；
  每漂移仅做一次整数包含判断与至多 3 个残差的统计。
* **首末级预筛**：任何完整对齐都必须让首级（含前 dwell_min 个观测）
  与末级块可行，取两者可行区间并集的交集再枚举漂移。
* **边界序列编码**：边界均在 1..60，以 64 为基编码为单个整数，
  同级数（同跳过数）下整数大小次序恰为边界序列字典序，
  追加边界即 ``code = code * 64 + j``。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class AlignmentError(ValueError):
    """输入不合法（字段越界或序列规模非法）。"""


# 漂移闭区间允许的最大宽度（drift_max - drift_min）。
DRIFT_WIDTH_MAX = 2000

# (skipped, residual_sum, residual_max, drift, boundary_code)
Cost = Tuple[int, int, int, int, int]

# 状态：(代价, 前驱参考索引, 本级停留长度)；首级前驱为 -1。
State = Optional[Tuple[Cost, int, int]]

_BOUNDARY_BASE = 64  # 必须 > 最大观测数 60


def solve_alignment(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int = 1,
    dwell_max: int = 3,
    max_skips: int = 2,
    anchors: Optional[Sequence[Tuple[int, int]]] = None,
) -> dict:
    """求最优联合对齐。

    成功返回含 ``feasible=True`` 的结果字典（漂移、逐级采样区间、
    残差证据、残差和/最大残差、跳过信息，启用锚点时另含逐项锚定归属
    证据）；任何合法对齐都不存在时返回 ``{"feasible": False, ...}``，
    其中无锚点或普通无解说 ``no_alignment_exists``，合法锚点与原约束
    无法同时满足时说 ``anchors_incompatible``；非法输入抛
    :class:`AlignmentError`。
    """
    anchor_list = _validate_anchors(anchors)
    _validate_inputs(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )

    R = len(reference)
    N = len(observations)
    lim = residual_limit

    # 序列规模已知后，再做锚点索引越界检查（结构已在 _validate_anchors
    # 中校验：数量、类型、互异、严格递增）。
    for pos, (ref_i, sample_t) in enumerate(anchor_list):
        if not 0 <= ref_i < R:
            raise AlignmentError(
                f"anchors[{pos}].reference_index={ref_i} 越界，"
                f"合法范围为 0 至 {R - 1}"
            )
        if not 0 <= sample_t < N:
            raise AlignmentError(
                f"anchors[{pos}].sample_index={sample_t} 越界，"
                f"合法范围为 0 至 {N - 1}"
            )

    # 锚定参考级 -> 被锚定的采样索引（每个参考级至多一个锚点）。
    # 锚点已校验为按 reference_index（随之按 sample_index）严格递增。
    anchor_at: Dict[int, int] = {}
    for ref_i, sample_t in anchor_list:
        anchor_at[ref_i] = sample_t
    anchor_order = list(anchor_list)

    # 级数上下界（首尾必用、锚定级必含、至多 max_skips 个内部跳过）。
    mandatory_inner = sum(
        1 for i in range(1, R - 1) if i in anchor_at
    )
    min_levels = max(2 + mandatory_inner, R - max_skips)
    max_levels = R
    if min_levels * dwell_min > N or max_levels * dwell_max < N:
        return _infeasible_result(bool(anchor_list))

    # 必含参考级集合：首尾级加上每个锚点所在参考级。
    mandatory = {0, R - 1} | set(anchor_at)
    # 严格位于 i 之前/之后的必含级数，用于 DP 边界剪枝。
    mandatory_before = [0] * R
    mandatory_after = [0] * R
    seen = 0
    for i in range(R):
        mandatory_before[i] = seen
        if i in mandatory:
            seen += 1
    seen = 0
    for i in range(R - 1, -1, -1):
        mandatory_after[i] = seen
        if i in mandatory:
            seen += 1

    # 锚点隐含的采样顺序可行性完全交给 DP 统一裁决
    # （锚定采样必须落在该级连续区间内，锚点间次序随参考级严格递增），
    # 这里只保留最便宜的级数/观测数总量裁断。

    # 预计算每个 (i, j, L) 块的可行漂移区间与中心化残差 c_t = O[t]-R[i]。
    # blocks[(i, j, L)] = (feas_lo, feas_hi, (c_{j-L}, ..., c_{j-1}))
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]] = {}
    for i in range(R):
        ri = reference[i]
        for j in range(N + 1):
            for L in range(dwell_min, dwell_max + 1):
                s = j - L
                if s < 0:
                    continue
                cs = tuple(observations[t] - ri for t in range(s, j))
                blocks[(i, j, L)] = (max(cs) - lim, min(cs) + lim, cs)

    def _block_covers_anchor(i: int, s: int, j: int) -> bool:
        """块 [s, j) 是否覆盖参考级 i 的锚定采样。"""
        t = anchor_at.get(i)
        return t is None or s <= t < j

    def first_block_intervals() -> List[Tuple[int, int]]:
        # 首级停留 j 个采样后，余下观测必须还能铺够最少后续级数：
        # 跳过上限与后续必含级数（末级 + 内部锚定级）两者取大。
        later_min = max(1, R - 1 - max_skips, mandatory_after[0])
        out = []
        for j in range(dwell_min, dwell_max + 1):
            if N - j < later_min * dwell_min:
                continue
            if 0 in anchor_at and not _block_covers_anchor(0, 0, j):
                continue
            lo, hi, _ = blocks[(0, j, j)]
            out.append((lo, hi))
        return _merge_intervals(out)

    def last_block_intervals() -> List[Tuple[int, int]]:
        # 末级之前至少需要的级数同样受跳过上限与必含级数双重约束。
        earlier_min = max(
            1, R - 1 - max_skips, mandatory_before[R - 1]
        )
        out = []
        for L in range(dwell_min, dwell_max + 1):
            start = N - L
            if start < earlier_min * dwell_min:
                continue
            if R - 1 in anchor_at and not _block_covers_anchor(
                R - 1, start, N
            ):
                continue
            lo, hi, _ = blocks[(R - 1, N, L)]
            out.append((lo, hi))
        return _merge_intervals(out)

    # 内部锚点级的块可行漂移区间同样必须与最终漂移相容：
    # 取每个锚点所有“覆盖锚定采样的块”区间并集，再整体取交。
    anchor_block_intervals: List[List[Tuple[int, int]]] = []
    for ref_i, sample_t in anchor_order:
        if ref_i in (0, R - 1):
            continue
        ivs: List[Tuple[int, int]] = []
        for L in range(dwell_min, dwell_max + 1):
            s_lo = max(0, sample_t - L + 1)
            s_hi = min(sample_t, N - L)
            for s in range(s_lo, s_hi + 1):
                lo, hi, _ = blocks[(ref_i, s + L, L)]
                ivs.append((lo, hi))
        anchor_block_intervals.append(_merge_intervals(ivs))

    intervals = _intersect_interval_lists(
        first_block_intervals(), last_block_intervals()
    )
    for ivs in anchor_block_intervals:
        intervals = _intersect_interval_lists(intervals, ivs)
    drift_ranges = _enumerate_int_intervals(
        intervals, drift_min, drift_max
    )

    best_cost: Optional[Cost] = None
    best_solution: Optional[Tuple[List[int], List[int]]] = None

    # 每个必含级（首尾 + 锚定级）前一个必含级的参考索引，
    # 用于禁止 DP 跨过必含级。
    prev_mandatory: List[int] = []
    last_m = 0
    for i in range(R):
        prev_mandatory.append(last_m)
        if i in mandatory:
            last_m = i

    for drift_range in drift_ranges:
        for d in drift_range:
            grid: List[List[State]] = [
                [None] * R for _ in range(N + 1)
            ]

            # 首级 i = 0（必含）。
            first_later_min = max(
                1, R - 1 - max_skips, mandatory_after[0]
            )
            first_j_hi = min(
                dwell_max, N - first_later_min * dwell_min
            )
            for j in range(dwell_min, first_j_hi + 1):
                if 0 in anchor_at and not _block_covers_anchor(0, 0, j):
                    continue
                lo, hi, cs = blocks[(0, j, j)]
                if not (lo <= d <= hi):
                    continue
                bsum, bmax = _block_stats(cs, d)
                cand: Cost = (0, bsum, bmax, d, j)
                if grid[j][0] is None or cand < grid[j][0][0]:  # type: ignore[index]
                    grid[j][0] = (cand, -1, j)

            # 后续级。
            for i in range(1, R):
                # 前驱下界：受跳过上限与“不得跨过前一个必含级”双重约束。
                p_lo = max(
                    0, i - max_skips - 1, prev_mandatory[i]
                )
                used_before_min = i - min(max_skips, max(0, i - 1))
                used_before_max = i
                # 本级之后至少需要的级数：跳过上限决定的下界与必含级
                # （末级 + 后续锚定级）两者取大。
                later_min = max(
                    0 if i == R - 1 else 1,
                    R - 1 - i - max_skips,
                    mandatory_after[i],
                )
                # 含本级在内至少/至多消耗的观测数；同时保证剩余观测
                # 足以容纳后续最少级数。
                j_lo = max(
                    (used_before_min + 1) * dwell_min,
                    (mandatory_before[i] + 1) * dwell_min,
                )
                j_hi = min(
                    N,
                    (used_before_max + 1) * dwell_max,
                    N - later_min * dwell_min,
                )
                if j_lo > j_hi:
                    continue
                anchored_t = anchor_at.get(i)
                for j in range(j_lo, j_hi + 1):
                    # 锚定级的本级区间必须覆盖锚定采样：s <= t < j。
                    if anchored_t is not None and not (j > anchored_t):
                        continue
                    best: Optional[Tuple[Cost, int, int]] = None
                    L_lo = dwell_min
                    L_hi = min(dwell_max, j)
                    if anchored_t is not None:
                        # 区间 [j-L, j) 覆盖 t 要求 L >= j - t。
                        L_lo = max(L_lo, j - anchored_t)
                    for L in range(L_lo, L_hi + 1):
                        prev_j = j - L
                        if prev_j <= 0:
                            continue
                        lo, hi, cs = blocks[(i, j, L)]
                        if not (lo <= d <= hi):
                            continue
                        bsum, bmax = _block_stats(cs, d)
                        for p in range(i - 1, p_lo - 1, -1):
                            prev = grid[prev_j][p]
                            if prev is None:
                                continue
                            pcost = prev[0]
                            new_skipped = pcost[0] + (i - p - 1)
                            if new_skipped > max_skips:
                                continue
                            cand = (
                                new_skipped,
                                pcost[1] + bsum,
                                pcost[2] if pcost[2] >= bmax else bmax,
                                d,
                                pcost[4] * _BOUNDARY_BASE + j,
                            )
                            if best is None or cand < best[0]:
                                best = (cand, p, L)
                    if best is not None:
                        cur = grid[j][i]
                        if cur is None or best[0] < cur[0]:
                            grid[j][i] = best

            final_state = grid[N][R - 1]
            if final_state is not None and (
                best_cost is None or final_state[0] < best_cost
            ):
                best_cost = final_state[0]
                best_solution = _backtrack(grid, N, R - 1)

    if best_cost is None or best_solution is None:
        return _infeasible_result(bool(anchor_list))

    used_indices, dwells = best_solution
    return _build_result(
        reference,
        observations,
        best_cost,
        used_indices,
        dwells,
        anchor_order,
        anchor_at,
    )


def _block_stats(cs: Tuple[int, ...], drift: int) -> Tuple[int, int]:
    """块内残差绝对和与最大绝对残差（块长 <= 3）。"""
    total = 0
    worst = 0
    for c in cs:
        ar = abs(c - drift)
        total += ar
        if ar > worst:
            worst = ar
    return total, worst


def _merge_intervals(
    intervals: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """合并相交的整数闭区间。"""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        mlo, mhi = merged[-1]
        if lo <= mhi + 1:
            if hi > mhi:
                merged[-1] = (mlo, hi)
        else:
            merged.append((lo, hi))
    return merged


def _intersect_interval_lists(
    a: List[Tuple[int, int]], b: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """两个已合并区间列表的交集。"""
    out: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _enumerate_int_intervals(
    intervals: List[Tuple[int, int]], lo: int, hi: int
) -> List[range]:
    """区间列表与 [lo, hi] 取交后返回各段整数 range。"""
    result: List[range] = []
    for ilo, ihi in intervals:
        clo = max(ilo, lo)
        chi = min(ihi, hi)
        if clo <= chi:
            result.append(range(clo, chi + 1))
    return result


def _backtrack(
    grid: List[List[State]], end_j: int, end_i: int
) -> Tuple[List[int], List[int]]:
    """沿状态指针回溯，返回 (采用参考索引序列, 逐级停留长度)。"""
    j = end_j
    i = end_i
    rev_idx: List[int] = []
    rev_dwell: List[int] = []
    while True:
        cur = grid[j][i]
        assert cur is not None
        _, prev_i, dwell = cur
        rev_idx.append(i)
        rev_dwell.append(dwell)
        if prev_i < 0:
            break
        j -= dwell
        i = prev_i
    rev_idx.reverse()
    rev_dwell.reverse()
    return rev_idx, rev_dwell


def _build_result(
    reference: Sequence[int],
    observations: Sequence[int],
    cost: Cost,
    used_indices: List[int],
    dwells: List[int],
    anchor_order: Optional[List[Tuple[int, int]]] = None,
    anchor_at: Optional[Dict[int, int]] = None,
) -> dict:
    skipped, residual_sum, max_abs, drift, _code = cost
    boundaries: List[int] = []
    acc = 0
    for L in dwells:
        acc += L
        boundaries.append(acc)
    assert acc == len(observations)

    levels_out = []
    residual_sum_check = 0
    max_abs_check = 0
    sample_t = 0
    # 参考级索引 -> 输出中的级序号与采样区间，供锚点证据回填。
    level_slot: Dict[int, int] = {}
    level_span: Dict[int, Tuple[int, int]] = {}
    for slot, (ri, L) in enumerate(zip(used_indices, dwells)):
        adopted = reference[ri] + drift
        s = sample_t
        e = sample_t + L
        samples = []
        for t in range(s, e):
            r = observations[t] - adopted
            ar = abs(r)
            residual_sum_check += ar
            if ar > max_abs_check:
                max_abs_check = ar
            samples.append(
                {"index": t, "observed": observations[t], "residual": r}
            )
        level_out = {
            "level_order": slot,
            "reference_index": ri,
            "reference_level": reference[ri],
            "adopted_level": adopted,
            "sample_start": s,
            "sample_end": e,
            "dwell": L,
            "samples": samples,
        }
        if anchor_order:
            # 仅在启用锚点时附加标记，保证无锚点响应结构不变。
            anchored_t = anchor_at.get(ri)
            level_out["anchored"] = anchored_t is not None
            if anchored_t is not None:
                level_out["anchor_sample_index"] = anchored_t
        levels_out.append(level_out)
        level_slot[ri] = slot
        level_span[ri] = (s, e)
        sample_t = e

    used_set = set(used_indices)
    skipped_indices = [i for i in range(len(reference)) if i not in used_set]

    # 防御性自检：DP 代价必须与重建结果完全一致。
    assert residual_sum_check == residual_sum
    assert max_abs_check == max_abs
    assert skipped == len(skipped_indices)
    assert used_indices[0] == 0
    assert used_indices[-1] == len(reference) - 1

    result = {
        "feasible": True,
        "drift": drift,
        "num_skips": skipped,
        "skipped_reference_indices": skipped_indices,
        "residual_sum": residual_sum,
        "max_abs_residual": max_abs,
        "boundaries": boundaries[:-1],
        "num_levels_used": len(used_indices),
        "levels": levels_out,
    }

    if anchor_order:
        anchor_evidence = []
        for order, (ri, sample_t_anchor) in enumerate(anchor_order):
            assert ri in level_slot, "锚定参考级未被采用"
            slot = level_slot[ri]
            s, e = level_span[ri]
            assert s <= sample_t_anchor < e, "锚定采样未落入该级区间"
            anchor_evidence.append(
                {
                    "anchor_order": order,
                    "reference_index": ri,
                    "sample_index": sample_t_anchor,
                    "assigned_level_order": slot,
                    "sample_start": s,
                    "sample_end": e,
                    "dwell": e - s,
                    "residual": (
                        observations[sample_t_anchor]
                        - (reference[ri] + drift)
                    ),
                }
            )
            # 防御性交叉核对：逐级输出的锚点标记与证据一致。
            assert levels_out[slot]["anchored"] is True
            assert levels_out[slot]["anchor_sample_index"] == sample_t_anchor
        result["anchor_assignments"] = anchor_evidence

    return result


def _infeasible_result(anchors_present: bool = False) -> dict:
    if anchors_present:
        return {
            "feasible": False,
            "reason": "anchors_incompatible",
            "message": (
                "锚点标记本身合法，但不存在同时满足全部硬锚点与漂移区间、"
                "残差上限、停留范围、跳过上限约束的对齐；请核对荧光事件"
                "标记或确认实验轨迹不相容。"
            ),
        }
    return {
        "feasible": False,
        "reason": "no_alignment_exists",
        "message": (
            "在给定漂移区间、残差上限、停留范围与跳过上限下，"
            "不存在任何合法对齐。"
        ),
    }


def _validate_anchors(
    anchors: Optional[Sequence[Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    """校验并规范化硬锚点；非法时抛 :class:`AlignmentError`。

    注意：本函数在参考/观测长度校验之前调用，因此索引越界由
    :func:`solve_alignment` 在校验完序列规模后二次检查，这里只做
    与规模无关的结构校验。
    """
    if anchors is None:
        return []
    if not isinstance(anchors, (list, tuple)):
        raise AlignmentError("anchors 必须是包含 1 至 3 个标记的数组")
    if not 1 <= len(anchors) <= 3:
        raise AlignmentError("anchors 必须包含 1 至 3 个标记")

    normalized: List[Tuple[int, int]] = []
    for pos, item in enumerate(anchors):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise AlignmentError(
                f"anchors[{pos}] 必须是 [reference_index, sample_index] "
                "两个整数组成的数组"
            )
        ref_i, sample_t = item
        if isinstance(ref_i, bool) or not isinstance(ref_i, int):
            raise AlignmentError(
                f"anchors[{pos}].reference_index 必须为整数"
            )
        if isinstance(sample_t, bool) or not isinstance(sample_t, int):
            raise AlignmentError(
                f"anchors[{pos}].sample_index 必须为整数"
            )
        normalized.append((ref_i, sample_t))

    ref_indices = [ri for ri, _ in normalized]
    sample_indices = [st for _, st in normalized]
    if len(set(ref_indices)) != len(ref_indices):
        raise AlignmentError("anchors 的 reference_index 必须两两互异")
    if len(set(sample_indices)) != len(sample_indices):
        raise AlignmentError("anchors 的 sample_index 必须两两互异")
    if any(ref_indices[i] >= ref_indices[i + 1]
           for i in range(len(ref_indices) - 1)):
        raise AlignmentError(
            "anchors 的 reference_index 必须严格递增"
        )
    if any(sample_indices[i] >= sample_indices[i + 1]
           for i in range(len(sample_indices) - 1)):
        raise AlignmentError(
            "anchors 的 sample_index 必须严格递增"
        )
    return normalized



def _validate_inputs(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> None:
    def _is_int_list(v: object, name: str) -> None:
        if not isinstance(v, list) or not v:
            raise AlignmentError(f"{name} 必须是非空整数数组")
        for x in v:
            if isinstance(x, bool) or not isinstance(x, int):
                raise AlignmentError(f"{name} 的元素必须为整数")

    def _is_int(v: object, name: str) -> None:
        if isinstance(v, bool) or not isinstance(v, int):
            raise AlignmentError(f"{name} 必须为整数")

    _is_int_list(reference, "reference_levels")
    _is_int_list(observations, "observations")
    _is_int(drift_min, "drift_min")
    _is_int(drift_max, "drift_max")
    _is_int(residual_limit, "residual_limit")
    _is_int(dwell_min, "dwell_min")
    _is_int(dwell_max, "dwell_max")
    _is_int(max_skips, "max_skips")

    R = len(reference)
    N = len(observations)

    if not 8 <= R <= 24:
        raise AlignmentError("参考电平数量必须在 8 至 24 之间")
    if not 8 <= N <= 60:
        raise AlignmentError("观测值数量必须在 8 至 60 之间")
    if drift_min > drift_max:
        raise AlignmentError("漂移闭区间要求 drift_min <= drift_max")
    if abs(drift_min) > 1_000_000 or abs(drift_max) > 1_000_000:
        raise AlignmentError("漂移区间超出允许范围 (+/-1000000)")
    if drift_max - drift_min > DRIFT_WIDTH_MAX:
        raise AlignmentError(
            f"漂移闭区间宽度不得超过 {DRIFT_WIDTH_MAX}"
        )
    if residual_limit < 0 or residual_limit > 1_000_000:
        raise AlignmentError("残差上限必须为非负整数且不超过 1000000")
    if not 1 <= dwell_min <= dwell_max <= 3:
        raise AlignmentError("停留范围要求 1 <= dwell_min <= dwell_max <= 3")
    if not 0 <= max_skips <= 2:
        raise AlignmentError("内部跳过数上限必须在 0 至 2 之间")
    for x in list(reference) + list(observations):
        if abs(x) > 1_000_000_000:
            raise AlignmentError("电平/观测值超出允许范围 (+/-1e9)")
