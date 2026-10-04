# Nanopore Current-Trace Alignment

纳米孔测序质控用的离子电流观测—参考电平**联合对齐**服务。与“先估计基线
漂移、再分段”的两阶段做法不同，本服务在一次动态规划中**联合**选择：

1. 一个整数漂移 `d`（统一作用于所有采用电平）；
2. 一个**首尾必用**的参考电平子序列（最多跳过 2 个内部电平，
   避免把短暂停留误判成碱基跳过）；
3. 连续采样边界（每段归属一个采用电平，每个观测恰好归属一段，
   每级停留 1–3 个采样）。

并依次最小化：**跳过数 → 残差绝对值总和 → 最大残差 → 漂移 → 边界序列字典序**。

零三方运行时依赖（仅 Python 标准库），可完全离线构建镜像。

## API

### `POST /api/current-traces/align`

请求字段：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `reference_levels` | int[] | 8–24 个整数参考电平 |
| `observations` | int[] | 8–60 个整数观测 |
| `drift_min` / `drift_max` | int | 统一漂移闭区间，宽度 ≤ 2000 |
| `residual_limit` | int | 残差上限（非负） |
| `dwell_min` / `dwell_max` | int，可选 | 每级停留采样范围，默认 1–3（1 ≤ … ≤ 3） |
| `max_skips` | int，可选 | 内部跳过上限，默认 2（0–2） |
| `anchors` | array，可选 | 1–3 个硬锚点 `[reference_index, sample_index]`（零基），两类索引各自互异且严格递增 |

硬锚点（可选）
--------------

质控人员可将同步荧光通道确认的可信事件作为现有电流轨迹对齐的**硬约束**：

* `reference_index`、`sample_index` 均为零基；同一次请求内
  `reference_index` 两两互异且严格递增，`sample_index` 同样两两互异且
  严格递增（两类索引的标记顺序一致）；
* 每个被锚定的参考级**不得被跳过**，且被锚定的观测必须归属该级的
  连续采样区间 `[sample_start, sample_end)`；
* 漂移闭区间、停留范围、残差上限与内部跳过上限等原有约束仍然全部生效；
* 五级裁决顺序（跳过数 → 残差和 → 最大残差 → 漂移 → 边界字典序）不变，
  锚点通过收紧可行域生效——残差更小但跨过已确认事件的方案不再可选；
* 省略 `anchors`（或为 `null`）时，请求/响应结构与裁决、无解语义完全不变。

启用锚点的成功响应额外返回逐项归属证据 `anchor_assignments`（顺序与请求
中的锚点一致），且 `levels[]` 中被锚定级带 `anchored` /
`anchor_sample_index` 标记：

```json
{
  "anchor_order": 0,
  "reference_index": 3,
  "sample_index": 6,
  "assigned_level_order": 3,
  "sample_start": 6,
  "sample_end": 8,
  "dwell": 2,
  "residual": 0
}
```

错误区分：

* **标记录入错误**（索引越界、重复、次序冲突、形状/类型错误、数量不是
  1–3 个）返回 HTTP 400，`error="invalid_anchors"`，并在 `errors[]` 中
  给出逐项字段级错误（`field` + `message`）；
* **实验轨迹不相容**——锚点格式合法但漂移/残差/停留/跳过约束无法同时
  满足——返回 HTTP 200 与明确无解结论
  `{"feasible": false, "reason": "anchors_incompatible", ...}`，
  与普通无解 `no_alignment_exists` 区分。

成功（HTTP 200）返回：

```json
{
  "feasible": true,
  "drift": 5,
  "num_skips": 0,
  "skipped_reference_indices": [],
  "residual_sum": 0,
  "max_abs_residual": 0,
  "boundaries": [1, 2, 3],
  "num_levels_used": 8,
  "levels": [
    {
      "level_order": 0,
      "reference_index": 0,
      "reference_level": 10,
      "adopted_level": 15,
      "sample_start": 0,
      "sample_end": 1,
      "dwell": 1,
      "samples": [{"index": 0, "observed": 15, "residual": 0}]
    }
  ]
}
```

* `boundaries`：内部连续采样边界（不含 0 与末尾 N）；
* `levels[].samples[]`：逐级残差证据，覆盖每个观测恰好一次。

任何合法对齐都不存在时仍返回 HTTP 200，但给出明确无解结论：

```json
{"feasible": false, "reason": "no_alignment_exists", "message": "..."}
```

字段越界或序列规模非法返回 HTTP 400 与错误原因；另有
`GET /health` 健康检查。

## 本地运行（无需容器）

```bash
python -m nanopore_align.app --host 0.0.0.0 --port 8000
python -m unittest discover -s tests -v
python verify/verify.py
```

## Docker / Compose

```bash
# 构建并启动带健康检查的 API（宿主机端口可配置）
NANOPORE_HOST_PORT=9090 docker compose up -d --build api

# 一次性验证服务：包构建 → 代码测试 → 可行/无解/有锚点可行/锚点不相容/
# 锚点字段错误/非法 HTTP 冒烟，以退出码汇报（0 全部通过）
docker compose up --build verify
```

## 算法概要

见 `nanopore_align/alignment.py` 模块文档字符串。要点：逐整数漂移做
动态规划，`grid[j][i]` 为观测前缀在参考电平 `R[i]` 结束时的最优代价；
首级强制为 `R[0]`、末级强制为 `R[R-1]`；采样块的可行漂移区间
`[max(O−R) − lim, min(O−R) + lim]` 与漂移无关、一次性预计算，
并以首末级可行区间交集预筛漂移；边界序列以 64 为基编码为整数，
字典序平局裁决即整数比较。启用锚点时，锚定参考级被加入必含集合
（DP 前驱不得跨过前一个必含级），只有覆盖锚定采样的本级块才允许落
状态，锚点块的可行漂移区间也参与漂移预筛。
