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
| `anchors` | object[]，可选 | 1–3 个硬锚点，见下节 |

### 硬锚点（可选）

质控人员可把来自同步荧光通道的可信事件标记作为硬锚点传入，
避免残差更小的方案跨过已确认的碱基事件：

```json
"anchors": [
  {"reference_index": 2, "sample_index": 5},
  {"reference_index": 4, "sample_index": 9}
]
```

* 1–3 个标记，均由零基 `reference_index` 与 `sample_index` 组成，
  两类索引各自互异且严格递增；
* 启用后每个锚定参考级**不得被跳过**，且对应观测必须归属该级的
  连续采样区间；漂移、停留、残差与内部跳过限制照常共同生效，
  目标裁决顺序（跳过数 → 残差和 → 最大残差 → 漂移 → 边界字典序）
  不变；
* 省略 `anchors` 时，请求、响应、裁决与无解语义与此前完全一致。

成功响应额外携带逐项锚定归属证据（顺序与请求锚点一致）：

```json
"anchor_assignments": [
  {
    "reference_index": 2,
    "sample_index": 5,
    "level_order": 2,
    "sample_start": 4,
    "sample_end": 7,
    "adopted_level": 35,
    "observed": 34,
    "residual": -1
  }
]
```

锚点索引越界、重复或次序冲突属于**标记录入错误**，返回 HTTP 400
与字段级定位：

```json
{
  "feasible": false,
  "error": "invalid_anchors",
  "field": "anchors[1].reference_index",
  "message": "..."
}
```

锚点格式合法但与漂移/残差/停留/跳过约束无法同时满足时，属于
**实验轨迹不相容**，仍返回 HTTP 200 与明确无解结论：

```json
{"feasible": false, "reason": "no_alignment_with_anchors", "message": "..."}
```

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

# 一次性验证服务：包构建 → 代码测试 → 可行/无解/非法及
# 有锚点（可行/不相容/录入错误）HTTP 冒烟，以退出码汇报（0 全部通过）
docker compose up --build verify
```

## 算法概要

见 `nanopore_align/alignment.py` 模块文档字符串。要点：逐整数漂移做
两阶段动态规划——阶段一逐漂移最小化（跳过数 → 残差和 → 最大残差），
跨漂移再比漂移；阶段二在最优漂移下把残差上限收紧到最优最大残差，
最小化（跳过数 → 残差和 → 边界编码），精确取得字典序最小边界。
`grid[j][i]` 为观测前缀在参考电平 `R[i]` 结束时的最优代价；首级
强制为 `R[0]`、末级强制为 `R[R-1]`；采样块的可行漂移区间
`[max(O−R) − lim, min(O−R) + lim]` 与漂移无关、一次性预计算，
并以首末级可行区间交集预筛漂移；边界序列以 64 为基编码为整数，
字典序平局裁决即整数比较。启用锚点时，DP 转移额外要求：锚定
参考级不被跳过、锚定观测落在该级采样区间内——锚点只收缩合法
解空间，不改变五级裁决顺序。
