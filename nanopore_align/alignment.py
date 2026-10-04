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

实现要点
========

* **两阶段逐漂移 DP**：漂移必须沿整条路径一致，因此对每个候选整数
  漂移独立求解。阶段一逐漂移最小化 ``(跳过数, 残差和, 最大残差)``：
  三个分量均可加或对前缀单调，逐 ``(j, i)`` 只保留最优状态是精确
  的；随后跨漂移取 ``(跳过数, 残差和, 最大残差, 漂移)`` 最小者。
  阶段二仅在最优漂移下把残差上限收紧到最优最大残差，再最小化
  ``(跳过数, 残差和, 边界编码)``：三者全可加，边界字典序裁决精确。
  （若单趟把最大残差放进代价元组，该非可加分量会把"前缀最大残差
  较差、但后缀能将其拉平且边界字典序更小"的状态错误剪掉。）
  ``grid[j][i]`` 表示观测前缀 ``O[0..j)`` 恰好在参考电平 R[i]
  结束时的最优代价；首级必须是 R[0]，末级必须是 R[R-1]，首尾因此
  强制必用。因最多 2 个内部跳过，前驱只需考虑 i-1/i-2/i-3。
* **块可行区间预计算**：对采样块 [j-L, j) 归属 R[i]，令
  c_t = O[t] - R[i]，可行漂移为整数闭区间
  ``[max c_t - lim, min c_t + lim]``，与漂移无关，只算一次；
  每漂移仅做一次整数包含判断与至多 3 个残差的统计。
* **首末级预筛**：任何完整对齐都必须让首级（含前 dwell_min 个观测）
  与末级块可行，取两者可行区间并集的交集再枚举漂移。
* **边界序列编码**：边界均在 1..60，以 64 为基编码为单个整数，
  同级数（同跳过数）下整数大小次序恰为边界序列字典序，
  追加边界即 ``code = code * 64 + j``。

锚点标记（硬锚点）
==================

质控人员可传入 1 至 3 个来自同步荧光通道的可信事件标记
``anchors = [{"reference_index": i, "sample_index": t}, ...]``（零基，
两类索引各自互异且严格递增），作为对齐的硬约束：

* 锚定参考级 ``i`` 不得被跳过（必在采用子序列中）；
* 观测 ``O[t]`` 必须归属采用 ``R[i]`` 那一级的连续采样区间。

漂移、停留、残差与内部跳过限制不变，目标五级裁决顺序不变——
锚点只收缩合法解空间，残差更小的方案不得跨过已确认的碱基事件。

锚点格式非法（越界、重复、次序冲突等）抛
:class:`AnchorValidationError`（携带字段级定位）；格式合法但与
原约束不相容时返回 ``reason="no_alignment_with_anchors"`` 的明确
无解结论，与无锚点时的 ``no_alignment_exists`` 区分，使质控人员
能够分辨标记录入错误与实验轨迹不相容。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class AlignmentError(ValueError):
    """输入不合法（字段越界或序列规模非法）。"""


class AnchorValidationError(AlignmentError):
    """锚点标记非法（越界、重复或次序冲突），携带字段级定位。"""

    def __init__(self, message: str, field: str = "anchors") -> None:
        super().__init__(message)
        self.field = field


# 漂移闭区间允许的最大宽度（drift_max - drift_min）。
DRIFT_WIDTH_MAX = 2000

# 阶段一代价：(skipped, residual_sum, residual_max)，全部分量可加或
# 对前缀单调，逐状态只保留最优是精确的。
Metrics = Tuple[int, int, int]

# 阶段二代价：(skipped, residual_sum, boundary_code)，全部可加。
BoundaryCost = Tuple[int, int, int]

# 阶段二状态：(代价, 前驱参考索引, 本级停留长度)；首级前驱为 -1。
State = Optional[Tuple[BoundaryCost, int, int]]

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
    anchors: Optional[Sequence[dict]] = None,
) -> dict:
    """求最优联合对齐。

    成功返回含 ``feasible=True`` 的结果字典（漂移、逐级采样区间、
    残差证据、残差和/最大残差、跳过信息）；任何合法对齐都不存在时
    返回 ``{"feasible": False, ...}``；非法输入抛 :class:`AlignmentError`。

    可选 ``anchors``：1 至 3 个 ``{"reference_index", "sample_index"}``
    硬锚点。启用后锚定参考级不得被跳过，且锚定观测必须归属该级的
    连续采样区间；成功结果额外携带 ``anchor_assignments`` 逐项归属
    证据。锚点非法抛 :class:`AnchorValidationError`。
    """
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
    anchor_by_ref = _validate_anchors(anchors, len(reference), len(observations))

    R = len(reference)
    N = len(observations)
    lim = residual_limit

    # 锚定参考级前缀计数：DP 转移 (p, i) 会跳过开区间 (p, i) 内的
    # 全部参考级，O(1) 判断其中是否含锚定级（含则转移非法）。
    anchored_prefix = [0] * (R + 1)
    for i in range(R):
        anchored_prefix[i + 1] = anchored_prefix[i] + (
            1 if i in anchor_by_ref else 0
        )

    # 级数上下界（首尾必用、至多 max_skips 个内部跳过）。
    min_levels = R - max_skips
    max_levels = R
    if min_levels * dwell_min > N or max_levels * dwell_max < N:
        return _infeasible_result(bool(anchor_by_ref))

    # 预计算每个 (i, j, L) 块的中心化残差 c_t = O[t]-R[i] 及其上下界。
    # 块可行漂移区间为 [max_c - lim, min_c + lim]，随 lim 即时推导。
    # blocks[(i, j, L)] = (max_c, min_c, (c_{j-L}, ..., c_{j-1}))
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]] = {}
    for i in range(R):
        ri = reference[i]
        for j in range(N + 1):
            for L in range(dwell_min, dwell_max + 1):
                s = j - L
                if s < 0:
                    continue
                cs = tuple(observations[t] - ri for t in range(s, j))
                blocks[(i, j, L)] = (max(cs), min(cs), cs)

    def first_block_intervals() -> List[Tuple[int, int]]:
        # 首级停留 j 个采样后，余下观测必须还能铺够最少末前级数。
        later_min = max(1, R - 1 - max_skips)
        out = []
        for j in range(dwell_min, dwell_max + 1):
            if N - j < later_min * dwell_min:
                continue
            max_c, min_c, _ = blocks[(0, j, j)]
            out.append((max_c - lim, min_c + lim))
        return _merge_intervals(out)

    def last_block_intervals() -> List[Tuple[int, int]]:
        earlier_min = max(1, R - 1 - max_skips)
        out = []
        for L in range(dwell_min, dwell_max + 1):
            start = N - L
            if start < earlier_min * dwell_min:
                continue
            max_c, min_c, _ = blocks[(R - 1, N, L)]
            out.append((max_c - lim, min_c + lim))
        return _merge_intervals(out)

    intervals = _intersect_interval_lists(
        first_block_intervals(), last_block_intervals()
    )
    drift_ranges = _enumerate_int_intervals(
        intervals, drift_min, drift_max
    )

    # 每个 (i, j) 的合法级数范围推导用的观测数上下界。
    later_levels_min: List[int] = []
    later_levels_max: List[int] = []
    for i in range(R):
        if i == R - 1:
            later_levels_min.append(0)
        else:
            later_levels_min.append(max(1, R - 1 - i - max_skips))
        later_levels_max.append(R - 1 - i)

    # 阶段一：逐漂移最小化 (跳过数, 残差和, 最大残差)，
    # 跨漂移按 (跳过数, 残差和, 最大残差, 漂移) 取最优。
    best_key: Optional[Tuple[int, int, int, int]] = None
    for drift_range in drift_ranges:
        for d in drift_range:
            metrics = _dp_metrics(
                blocks,
                R,
                N,
                dwell_min,
                dwell_max,
                max_skips,
                d,
                lim,
                anchor_by_ref,
                anchored_prefix,
                later_levels_min,
                later_levels_max,
            )
            if metrics is None:
                continue
            key = (metrics[0], metrics[1], metrics[2], d)
            if best_key is None or key < best_key:
                best_key = key

    if best_key is None:
        return _infeasible_result(bool(anchor_by_ref))

    skipped, residual_sum, residual_max, drift = best_key

    # 阶段二：在最优漂移下把残差上限收紧到最优最大残差（不损失任何
    # 最优解），再最小化 (跳过数, 残差和, 边界编码)，取得字典序最小
    # 的边界序列并回溯。
    solution = _dp_boundaries(
        blocks,
        R,
        N,
        dwell_min,
        dwell_max,
        max_skips,
        drift,
        residual_max,
        anchor_by_ref,
        anchored_prefix,
        later_levels_min,
        later_levels_max,
    )
    assert solution is not None  # 阶段一已证该漂移下可行
    (skipped2, residual_sum2, _code), used_indices, dwells = solution
    assert skipped2 == skipped and residual_sum2 == residual_sum

    return _build_result(
        reference,
        observations,
        skipped,
        residual_sum,
        residual_max,
        drift,
        used_indices,
        dwells,
        anchor_by_ref,
    )


def _dp_metrics(
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]],
    R: int,
    N: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    drift: int,
    lim: int,
    anchor_by_ref: Dict[int, int],
    anchored_prefix: List[int],
    later_levels_min: List[int],
    later_levels_max: List[int],
) -> Optional[Metrics]:
    """单漂移阶段一 DP：最小化 (跳过数, 残差和, 最大残差)。"""
    grid: List[List[Optional[Metrics]]] = [
        [None] * R for _ in range(N + 1)
    ]

    # 首级 i = 0，占据区间 [0, j)；若为首级锚点则区间须覆盖锚定样本。
    first_anchor_t = anchor_by_ref.get(0)
    for j in range(dwell_min, min(dwell_max, N) + 1):
        if first_anchor_t is not None and j <= first_anchor_t:
            continue
        max_c, min_c, cs = blocks[(0, j, j)]
        if not (max_c - lim <= drift <= min_c + lim):
            continue
        bsum, bmax = _block_stats(cs, drift)
        cand: Metrics = (0, bsum, bmax)
        if grid[j][0] is None or cand < grid[j][0]:  # type: ignore[index]
            grid[j][0] = cand

    # 后续级。
    for i in range(1, R):
        anchor_t = anchor_by_ref.get(i)
        p_lo = max(0, i - max_skips - 1)
        used_before_min = i - min(max_skips, max(0, i - 1))
        used_before_max = i
        # 含本级在内至少/至多消耗的观测数，同时保证余下观测数
        # 足以容纳后续最少/最多级数。
        j_lo = max(
            (used_before_min + 1) * dwell_min,
            N - later_levels_max[i] * dwell_max,
        )
        j_hi = min(
            N,
            (used_before_max + 1) * dwell_max,
            N - later_levels_min[i] * dwell_min,
        )
        if j_lo > j_hi:
            continue
        for j in range(j_lo, j_hi + 1):
            best: Optional[Metrics] = None
            for L in range(dwell_min, min(dwell_max, j) + 1):
                prev_j = j - L
                if prev_j <= 0:
                    continue
                # 锚定级：本级块 [prev_j, j) 必须覆盖锚定样本。
                if anchor_t is not None and not (prev_j <= anchor_t < j):
                    continue
                max_c, min_c, cs = blocks[(i, j, L)]
                if not (max_c - lim <= drift <= min_c + lim):
                    continue
                bsum, bmax = _block_stats(cs, drift)
                for p in range(i - 1, p_lo - 1, -1):
                    # 转移会跳过 (p, i) 内的参考级，锚定级不可跳过。
                    if anchored_prefix[i] - anchored_prefix[p + 1] > 0:
                        continue
                    prev = grid[prev_j][p]
                    if prev is None:
                        continue
                    new_skipped = prev[0] + (i - p - 1)
                    if new_skipped > max_skips:
                        continue
                    cand = (
                        new_skipped,
                        prev[1] + bsum,
                        prev[2] if prev[2] >= bmax else bmax,
                    )
                    if best is None or cand < best:
                        best = cand
            if best is not None:
                cur = grid[j][i]
                if cur is None or best < cur:
                    grid[j][i] = best

    return grid[N][R - 1]


def _dp_boundaries(
    blocks: Dict[Tuple[int, int, int], Tuple[int, int, Tuple[int, ...]]],
    R: int,
    N: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
    drift: int,
    lim: int,
    anchor_by_ref: Dict[int, int],
    anchored_prefix: List[int],
    later_levels_min: List[int],
    later_levels_max: List[int],
) -> Optional[Tuple[BoundaryCost, List[int], List[int]]]:
    """单漂移阶段二 DP：最小化 (跳过数, 残差和, 边界编码) 并回溯。

    调用方须把 ``lim`` 收紧到阶段一最优最大残差；此时三个代价分量
    全可加，逐 ``(j, i)`` 只保留最优状态是精确的。返回
    ``((skipped, residual_sum, boundary_code), used_indices, dwells)``。
    """
    grid: List[List[State]] = [[None] * R for _ in range(N + 1)]

    # 首级 i = 0，占据区间 [0, j)；若为首级锚点则区间须覆盖锚定样本。
    first_anchor_t = anchor_by_ref.get(0)
    for j in range(dwell_min, min(dwell_max, N) + 1):
        if first_anchor_t is not None and j <= first_anchor_t:
            continue
        max_c, min_c, cs = blocks[(0, j, j)]
        if not (max_c - lim <= drift <= min_c + lim):
            continue
        bsum, _bmax = _block_stats(cs, drift)
        cand: BoundaryCost = (0, bsum, j)
        if grid[j][0] is None or cand < grid[j][0][0]:  # type: ignore[index]
            grid[j][0] = (cand, -1, j)

    # 后续级。
    for i in range(1, R):
        anchor_t = anchor_by_ref.get(i)
        p_lo = max(0, i - max_skips - 1)
        used_before_min = i - min(max_skips, max(0, i - 1))
        used_before_max = i
        j_lo = max(
            (used_before_min + 1) * dwell_min,
            N - later_levels_max[i] * dwell_max,
        )
        j_hi = min(
            N,
            (used_before_max + 1) * dwell_max,
            N - later_levels_min[i] * dwell_min,
        )
        if j_lo > j_hi:
            continue
        for j in range(j_lo, j_hi + 1):
            best: Optional[Tuple[BoundaryCost, int, int]] = None
            for L in range(dwell_min, min(dwell_max, j) + 1):
                prev_j = j - L
                if prev_j <= 0:
                    continue
                # 锚定级：本级块 [prev_j, j) 必须覆盖锚定样本。
                if anchor_t is not None and not (prev_j <= anchor_t < j):
                    continue
                max_c, min_c, cs = blocks[(i, j, L)]
                if not (max_c - lim <= drift <= min_c + lim):
                    continue
                bsum, _bmax = _block_stats(cs, drift)
                for p in range(i - 1, p_lo - 1, -1):
                    # 转移会跳过 (p, i) 内的参考级，锚定级不可跳过。
                    if anchored_prefix[i] - anchored_prefix[p + 1] > 0:
                        continue
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
                        pcost[2] * _BOUNDARY_BASE + j,
                    )
                    if best is None or cand < best[0]:
                        best = (cand, p, L)
            if best is not None:
                cur = grid[j][i]
                if cur is None or best[0] < cur[0]:
                    grid[j][i] = best

    final_state = grid[N][R - 1]
    if final_state is None:
        return None
    used_indices, dwells = _backtrack(grid, N, R - 1)
    return final_state[0], used_indices, dwells


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
    skipped: int,
    residual_sum: int,
    max_abs: int,
    drift: int,
    used_indices: List[int],
    dwells: List[int],
    anchor_by_ref: Dict[int, int],
) -> dict:
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
        levels_out.append(
            {
                "level_order": slot,
                "reference_index": ri,
                "reference_level": reference[ri],
                "adopted_level": adopted,
                "sample_start": s,
                "sample_end": e,
                "dwell": L,
                "samples": samples,
            }
        )
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

    if anchor_by_ref:
        # 逐项锚定归属证据，顺序与请求锚点一致（参考索引递增）。
        level_by_ref = {lv["reference_index"]: lv for lv in levels_out}
        assignments = []
        for ri, t in anchor_by_ref.items():
            lv = level_by_ref[ri]  # 锚定级必被采用（DP 已强制）
            assert lv["sample_start"] <= t < lv["sample_end"]
            adopted = lv["adopted_level"]
            assignments.append(
                {
                    "reference_index": ri,
                    "sample_index": t,
                    "level_order": lv["level_order"],
                    "sample_start": lv["sample_start"],
                    "sample_end": lv["sample_end"],
                    "adopted_level": adopted,
                    "observed": observations[t],
                    "residual": observations[t] - adopted,
                }
            )
        result["anchor_assignments"] = assignments

    return result


def _infeasible_result(anchored: bool = False) -> dict:
    if anchored:
        return {
            "feasible": False,
            "reason": "no_alignment_with_anchors",
            "message": (
                "锚点标记格式合法，但在给定漂移区间、残差上限、停留范围"
                "与跳过上限下，不存在同时满足全部锚点的合法对齐；"
                "实验轨迹与标记不相容。"
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
    anchors: Optional[Sequence[dict]], R: int, N: int
) -> Dict[int, int]:
    """校验锚点标记，返回 {参考索引: 样本索引}（按键插入序即参考索引递增）。

    ``anchors=None`` 表示未启用锚点。启用时必须是 1 至 3 个
    ``{"reference_index", "sample_index"}`` 对象，两类零基索引各自
    互异且严格递增；任何违规抛 :class:`AnchorValidationError`。
    """
    if anchors is None:
        return {}
    if not isinstance(anchors, list) or not 1 <= len(anchors) <= 3:
        raise AnchorValidationError("anchors 必须是包含 1 至 3 个标记的数组")

    anchor_by_ref: Dict[int, int] = {}
    prev_ref = -1
    prev_sample = -1
    for pos, mark in enumerate(anchors):
        field = f"anchors[{pos}]"
        if not isinstance(mark, dict):
            raise AnchorValidationError(f"{field} 必须是对象", field)
        if set(mark) != {"reference_index", "sample_index"}:
            raise AnchorValidationError(
                f"{field} 必须恰好包含 reference_index 与 sample_index",
                field,
            )
        ri = mark["reference_index"]
        si = mark["sample_index"]
        if isinstance(ri, bool) or not isinstance(ri, int):
            raise AnchorValidationError(
                f"{field}.reference_index 必须为整数",
                f"{field}.reference_index",
            )
        if isinstance(si, bool) or not isinstance(si, int):
            raise AnchorValidationError(
                f"{field}.sample_index 必须为整数",
                f"{field}.sample_index",
            )
        if not 0 <= ri < R:
            raise AnchorValidationError(
                f"{field}.reference_index 越界: {ri}"
                f"（参考电平共 {R} 级，允许 0..{R - 1}）",
                f"{field}.reference_index",
            )
        if not 0 <= si < N:
            raise AnchorValidationError(
                f"{field}.sample_index 越界: {si}"
                f"（观测共 {N} 个，允许 0..{N - 1}）",
                f"{field}.sample_index",
            )
        if ri == prev_ref:
            raise AnchorValidationError(
                f"{field}.reference_index 与前一项重复: {ri}",
                f"{field}.reference_index",
            )
        if ri < prev_ref:
            raise AnchorValidationError(
                f"{field}.reference_index 次序冲突: {ri}"
                f" 出现在 {prev_ref} 之后，须严格递增",
                f"{field}.reference_index",
            )
        if si == prev_sample:
            raise AnchorValidationError(
                f"{field}.sample_index 与前一项重复: {si}",
                f"{field}.sample_index",
            )
        if si < prev_sample:
            raise AnchorValidationError(
                f"{field}.sample_index 次序冲突: {si}"
                f" 出现在 {prev_sample} 之后，须严格递增",
                f"{field}.sample_index",
            )
        anchor_by_ref[ri] = si
        prev_ref = ri
        prev_sample = si
    return anchor_by_ref


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
