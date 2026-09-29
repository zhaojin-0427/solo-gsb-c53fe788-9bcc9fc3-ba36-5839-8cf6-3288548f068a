# 地震台网震相关联 API

基于 **Python + FastAPI + PostgreSQL + ObsPy + NumPy** 的震相关联与定位服务。

客户端上传包含台站经纬高、采样率、UTC 起点的 **miniSEED** 波形窗口（可附人工 P/S 拾取与分层 P/S 速度模型），服务对波形做固定 **1–15 Hz** 带通滤波与 **0.5/5 s STA/LTA** 检测生成候选拾取，按请求给定的**最大旅行时残差**与**最少台站数**筛选，按（残差， 台站代码）稳定排序关联为事件，并以**加权最小二乘**联合求解发震 UTC 时刻与经纬深度，返回每个拾取的相位、残差、未关联原因及质量等级。

此外可从指定速度模型的**历史事件**估计**逐台站分相位的到时校准**（Huber 鲁棒估计、每相位零和约束、留出 RMS 严格把关），激活后的校准版本可在处理/重放时先修正到时再定位。

针对多个**已完成的 run**，还可构建**跨运行震情目录（cross-run catalog）**：按发震时刻、震中/深度半径与台站证据对同一地震的多个 Event 做**确定性聚类**，合并为内容**不可变**的目录条目，保留全部来源 run/event 与版本链；条目在**草稿 → 发布 → 撤回**生命周期中流转，发布用 `expected_revision` 做乐观并发校验。

最后，针对任一**已完成 run 的 event**可生成**台站影响审计（station impact audit / leave-one-station-out）**：冻结该事件的原始拾取、速度模型、校准版本与处理参数，在原始关联拾取子集上逐一移除每个参与台站的**全部 P/S 拾取**后重新关联定位，比较发震时刻、经纬深度、RMS、质量等级、关联拾取集合与不可解原因，按**事件消失 → 质量降级 → RMS 增幅 → 台站码**稳定排序并标出**关键台站**。审计报告绑定 event、run 与参数哈希，重复请求幂等，列表与逐台结果均支持分页；审计只读，**原事件、run 与目录内容绝不改变**。

针对**已完成事件**还可做 **P 波初动震源机制反演（first-motion focal mechanism inversion）**：客户端提交该事件关联 P 拾取的**正负极性与权重**，服务用事件震源位置、台站坐标与该 run 的分层速度模型求每条射线的**方位角与出射角**，在 **10° 网格**上搜索双力偶（走向、倾角、滑动角），以**加权极性误判最少**选解（预测零幅值的节面台站按误判计）；网格遍历**仅合并互换节面的同一矩张量**，**整体反号的矩张量（滑动反向机制，预测极性全部相反）作为独立候选保留**，每组解返回**两个节面、误判权重与逐台站观测/预测**，同分最优解按（走向， 倾角， 滑动角）稳定排序并支持游标分页。非关联拾取、同站矛盾极性、少于六个独立台站或最大方位空档 ≥ 180° 均被明确拒绝；反演按（事件， 模型版本， 规范化极性集）幂等保存，并发同请求只产生一份结果，且**不改动原事件、run 与目录**。早期版本曾把 `M` 与 `-M` 错误合并；按旧规则保存的报告在**重复提交或按原 ID 查询时惰性修正**（幂等、并发安全，ID/请求哈希/冻结快照不变）。

在此之上还有**目录告警编排（catalog alert orchestration）**：管理员针对**已发布目录**定义**不可变的告警规则版本**（质量等级、台站数、RMS、发震时刻时间窗、圆形区域条件可任意组合，另含合并时间间隔），以 `expected_revision` 做 CAS 发布（草稿 → 发布 → 停用）；对指定**目录编译批次**做评估时，按规则对每个（规则， 条目）匹配生成**确定性的命中记录**，同一规则的连续命中在合并间隔内合并为一条告警（抑制），命中已确认/已解除的告警则将其**升级**回打开；告警支持**确认**、**解除**与**幂等重评估**；目录条目撤回后只自动关闭**未确认**告警，命中与状态迁移全程留痕（审计链）；并发评估在唯一约束与序列化临界区保证下**不重复、不漏报**，查询均提供稳定游标。

最后，针对任一**已完成事件**可做**事件定位不确定性评估（location uncertainty assessment / 台站自助法 bootstrap）**：客户端提交 `event_id`、重采样次数 `N`（200–2000）、置信水平 `α`（0.8–0.99）与随机种子；服务冻结该事件的关联 P/S 拾取（含权重）、速度模型、校准版本与处理参数，按种子**有放回**抽取与原台站数相同的台站并保留其全部关联拾取（重复台站按出现次数计权），用同一定位器重算 N 次，逐次记录可解状态与结构化失败原因；按**最近秩**取两侧 `(1−α)/2` 分位点输出发震时刻、纬度、经度、深度的 α 置信区间，仅用可解样本计算样本协方差并给出可解率；**有效样本少于 N 的 80% 明确拒绝（422）且不留状态**。评估按（事件， 参数， 种子）幂等，并发只生成一份，报告与样本只读并支持稳定游标分页，**不改动原事件、run 与目录**。

## 功能特性

- **波形摄取**：base64 miniSEED + 台站元数据（经纬高、采样率、UTC 起点）；头部与声明元数据一致性校验
- **候选拾取**：固定 1–15 Hz 带通 + 0.5/5 s STA/LTA + AIC 精化，支持人工 P/S 拾取合并
- **震相关联**：台网孔径时间窗聚类 → 首拾取 P 引导定位 → 相位自动判定 → 残差/台站数筛选 → （残差， 台站代码）稳定排序，全程确定性可复现
- **事件定位**：Geiger 加权最小二乘（Levenberg 阻尼），联合求解发震时刻、经纬度、深度；走时来自水平分层 1-D 模型的射线参数精确解
- **质量等级**：由残差 RMS 与台站数确定（A/B/C/D）
- **台站到时校准**：按训练/留出时间段与最低质量、最少事件数、最短时间跨度、最低台站覆盖率，从历史事件提取 P/S 残差；以每个相位校正量总和为零的约束做 Huber 鲁棒估计；逐站分相位返回样本量、偏移、不确定度及训练/留出 RMS；**仅当留出 RMS 严格低于未校准基线才允许原子激活**；已发布（published）版本不可修改
- **幂等与重放**：同一波形窗口重复提交返回同一结果；模型版本、校准版本不可变；事件可按模型/校准版本重放并比较
- **跨运行震情目录**：对多个已完成 run 的事件按时间窗/空间半径/最小质量/版本条件确定性聚类；同一地震合并为不可变条目，质量最高且 RMS 最小者为主解（同质量同 RMS 按 run/event ID 决胜），冲突解作为候选并附相对主解偏差；草稿/发布/撤回状态机 + `expected_revision` 乐观锁，重复请求幂等，撤回不改写已发布内容；查询支持时间/区域/质量/状态过滤与稳定游标分页
- **台站影响审计**：冻结已完成事件的原始关联拾取、速度模型、校准版本（含逐站 P/S 偏移）与处理参数；逐一移除每个参与台站的全部 P/S 拾取并用同一确定性关联/定位重算；输出事件是否消失、发震时刻/经纬深度/RMS 增量、质量等级、关联（保留/丢弃）拾取集合与不可解原因；按事件消失 → 质量降级深度 → RMS 增幅 → 台站码稳定排序，关键台站（事件消失或质量降级）显式标注；报告绑定 event/run/参数哈希，幂等可重放，列表与逐台结果均支持游标分页，且不改动任何 run/event/目录内容
- **目录告警编排**：针对已发布目录的不可变告警规则版本（质量/台站数/RMS/时间窗/圆形区域 + 合并间隔），`expected_revision` CAS 发布与停用；按目录编译批次做确定性评估，每个（规则， 条目）匹配恰好记录一次命中；连续命中按规则与合并间隔合并（抑制），命中已确认/已解除告警则升级回打开；支持确认、解除与幂等重评估；条目撤回只关闭未确认告警且审计链完整保留；并发评估不重复不漏报；全部列表查询提供稳定游标
- **P 波初动震源机制反演**：针对已完成事件的关联 P 拾取提交正负极性与权重；用事件震源位置、台站坐标与该 run 的分层速度模型求射线方位角/出射角，10° 网格搜索双力偶（走向/倾角/滑动角），按完整候选集计算加权极性误判，最少者入选（零幅值按误判计）；去重仅合并互换节面的同一矩张量，**整体反号机制（滑动反向）保留为独立候选**；每组解返回两个节面、误判权重与逐台站观测/预测；同分最优解稳定排序 + 游标分页；非关联拾取、同站矛盾极性、< 6 独立台站、最大方位空档 ≥ 180° 明确拒绝；按（事件， 模型版本， 规范化极性集）幂等，并发同请求折叠为一份，只写独立反演表；旧版 M==-M 误去重的已存报告在重复提交或按 ID 查询时幂等、并发安全地惰性修正，原 ID/请求哈希/冻结快照不变
- **事件定位不确定性评估**：冻结已完成事件的关联 P/S 拾取（含权重）、速度模型、校准版本与处理参数；按客户端种子有放回重采样与原台站数相同的台站（保留全部关联拾取，重复台站按出现次数计权），用同一加权最小二乘定位器重算 N 次（200–2000）；逐样本记录可解状态与结构化失败原因（`insufficient_stations`/`location_failed`/`not_converged`）；按最近秩两侧 `(1−α)/2` 分位点输出发震时刻/纬度/经度/深度的 α（0.8–0.99）置信区间，仅用可解样本计算样本协方差（ddof=1）与可解率；可解样本不足 80% 明确拒绝且不留状态；按（事件， 参数， 种子）幂等，并发折叠为一份，报告与样本只读 + 稳定游标分页，不改动原事件、run 与目录
- **明确拒绝**：缺采样率、坐标越界、数据不足、校准门槛不达标等均以结构化错误码拒绝

## 快速启动（Docker Compose）

```bash
docker compose up --build
```

启动两个服务：`db`（PostgreSQL 16）与 `api`（FastAPI）。表结构自动创建，并播种默认速度模型 `default@1.0`（15 km/5.8、20 km/6.5、半空间 8.0 km/s）。

## 访问

| 地址 | 说明 |
|---|---|
| `http://localhost:8000/docs` | Swagger 交互文档 |
| `http://localhost:8000/health` | 健康检查 |
| `http://localhost:8000/v1/models` | 速度模型列表 |
| `http://localhost:8000/v1/catalog/entries` | 跨运行目录条目 |
| `http://localhost:8000/v1/alert-rules`、`http://localhost:8000/v1/alerts` | 告警规则与告警事件 |
| `http://localhost:8000/v1/focal-mechanisms` | P 波初动震源机制反演 |
| `http://localhost:8000/v1/location-uncertainties` | 事件定位不确定性评估（台站 bootstrap） |

## 示例请求

生成合成示例（8 台站、1 个事件、含 3 个不相关噪声脉冲）：

```bash
python scripts/make_example.py > example_request.json   # 依赖 requirements.txt
curl -s -X POST http://localhost:8000/v1/process \
  -H 'Content-Type: application/json' -d @example_request.json | python -m json.tool
```

响应节选：

```json
{
  "run_id": "87dc643902b44fa986a71197d992017e",
  "ingestion_id": "5dac4243c24a455ea551bff5b5bec443",
  "created": true,
  "model": {"name": "default", "version": "1.0"},
  "params": {"max_residual_s": 1.0, "min_stations": 4},
  "events": [
    {
      "event_id": "0f535f17cfef459db99bd1adad32dc99",
      "origin_time": "2026-09-22T00:00:29.828Z",
      "latitude": 35.05021, "longitude": -117.95015, "depth_km": 11.96,
      "rms_residual_s": 0.0056, "station_count": 8, "quality": "A",
      "picks": [
        {"pick_id": "…", "station": "EX01", "phase": "P",
         "time": "2026-09-22T00:00:33.950Z", "source": "stalta",
         "residual_s": 0.004, "weight": 1.0}
      ]
    }
  ],
  "unassociated_picks": [
    {"pick_id": "…", "station": "EX00", "phase": null,
     "time": "2026-09-22T00:01:34.850Z", "source": "stalta",
     "reason": "insufficient_stations"}
  ]
}
```

### 按版本重放与比较

```bash
# 注册新的（不可变）模型版本
curl -s -X POST http://localhost:8000/v1/models -H 'Content-Type: application/json' -d '{
  "name": "default", "version": "2.0",
  "layers": [
    {"thickness_km": 15.0, "vp_kms": 5.2, "vs_kms": 3.0},
    {"thickness_km": 20.0, "vp_kms": 5.9, "vs_kms": 3.4},
    {"thickness_km": null,  "vp_kms": 7.2, "vs_kms": 4.2}
  ]}'

# 用同一波形窗口（ingestion_id）按 v2.0 重放
curl -s -X POST http://localhost:8000/v1/runs -H 'Content-Type: application/json' -d '{
  "ingestion_id": "<ingestion_id>",
  "model": {"name": "default", "version": "2.0"},
  "params": {"max_residual_s": 1.0, "min_stations": 4}}'

# 比较两个 run 的事件差异
curl -s "http://localhost:8000/v1/runs/compare?run_a=<run_a>&run_b=<run_b>"
```

## 台站到时校准：从创建到激活

校准以某一**速度模型**下的**未校准历史事件**为输入，输出逐站分相位的偏移量。流程为：**创建候选（门槛校验 + Huber 估计 + 留出评估）→ 查看候选指标 → 原子激活 → 在处理/重放中选用**。

### 1. 创建候选版本

```bash
curl -s -X POST http://localhost:8000/v1/models/default/1.0/calibrations \
  -H 'Content-Type: application/json' -d '{
    "version": "cal-2026-09",
    "train_start":   "2026-01-01T00:00:00Z",
    "train_end":     "2026-08-01T00:00:00Z",
    "holdout_start": "2026-08-01T00:00:00Z",
    "holdout_end":   "2026-09-01T00:00:00Z",
    "min_quality": "B",
    "min_events": 20,
    "min_span_days": 30.0,
    "min_station_coverage": 0.8,
    "huber_delta_s": 0.5,
    "label": "2026 夏季台站钟差修正"
  }' | python -m json.tool
```

响应（节选）：`status` 为 `candidate`；`metrics.phases` 给出 P/S 各自的训练与留出 RMS；`stations` 给出**逐站分相位的样本量、偏移（秒）、不确定度（秒）**：

```json
{
  "calibration_id": "…", "version": "cal-2026-09", "status": "candidate",
  "metrics": {
    "train_event_count": 42, "holdout_event_count": 6, "train_span_days": 210.4,
    "station_universe": ["ST01", "ST02"],
    "phases": {
      "P": {"sample_count": 310, "station_count": 8, "coverage": 1.0,
            "train_rms_baseline_s": 0.31, "train_rms_calibrated_s": 0.11,
            "holdout_rms_baseline_s": 0.28, "holdout_rms_calibrated_s": 0.12},
      "S": {"sample_count": 305, "station_count": 8, "coverage": 1.0,
            "holdout_rms_baseline_s": 0.40, "holdout_rms_calibrated_s": 0.18}
    }
  },
  "stations": [
    {"station": "ST01", "phase": "P", "sample_count": 42,
     "offset_s": 0.182, "uncertainty_s": 0.021}
  ]
}
```

门槛不达标时请求被**明确拒绝（422）且不产生任何状态**：训练/留出窗口非法（`invalid_time_window`）、训练合格事件不足（`insufficient_events`）、无留出事件（`insufficient_holdout_events`）、训练时间跨度过短（`time_span_too_short`）、台站覆盖率不足（`station_coverage_too_low`）。相同请求体重复创建幂等返回原候选；同 `version` 不同参数返回 `409 calibration_version_conflict`。

### 2. 原子激活

```bash
curl -s -X POST http://localhost:8000/v1/models/default/1.0/calibrations/cal-2026-09/activate
```

激活在单个事务内完成，服务端会**再次从持久化的候选指标严格复核：每个相位的留出 RMS 必须严格低于未校准基线**（`holdout_rms_calibrated_s < holdout_rms_baseline_s`），否则返回 `409 holdout_gate_failed`，候选保持不激活。每个模型至多一个 `active` 校准（数据库部分唯一索引 + 每模型激活锁双重保证）；激活新版本时旧版本原子转为 `published`，其偏移与指标**永久不可修改**，再次激活已发布版本返回 `409 calibration_published_immutable`。对同一版本重复激活、并发激活均幂等，不会产生部分状态。

### 3. 在处理与重放中选用

处理和重放接口均可携带 `calibration: {"version": …}`；服务在关联与定位**之前**按（台站， 相位）从到时中减去偏移量，响应中回显所用校准版本与当前状态（`active`/`published`）。不带该字段时行为与旧版完全一致（未校准基线）。

```bash
# 处理时指定
curl -s -X POST http://localhost:8000/v1/process -H 'Content-Type: application/json' -d '{
  …, "calibration": {"version": "cal-2026-09"}}'

# 重放时指定
curl -s -X POST http://localhost:8000/v1/runs -H 'Content-Type: application/json' -d '{
  "ingestion_id": "<ingestion_id>",
  "model": {"name": "default", "version": "1.0"},
  "params": {"max_residual_s": 1.0, "min_stations": 4},
  "calibration": {"version": "cal-2026-09"}}'
```

响应中：`"calibration": {"version": "cal-2026-09", "status": "active"}`（未校准则为 `null`）。只能引用属于同一速度模型且非候选的版本，否则返回 `404 calibration_not_found` 或 `409 calibration_not_active`。

校准估计说明：对每个相位独立建模残差 `r = a_事件 + b_台站`（事件截距吸收发震时刻自由度），台站项以 **Σb = 0** 硬约束（KKT 方程组）保证可辨识性；Huber 鲁棒性由 IRLS 权重实现；部署时只应用台站项（发震时刻重新估计），因此训练与留出 RMS 均按事件中心化后再汇总，与部署口径一致。

## 跨运行震情目录（cross-run catalog）

目录把**多个已完成 run** 中属于同一地震的 `Event` 合并成**一条不可变目录条目**。它不改动任何 run/event：现有 `process`、`replay`、`compare` 语义完全不变，目录只是新增的只读聚合层与生命周期层。

### 聚类规则（确定性）

对满足时间窗（左闭右开）、最低质量、版本条件（可选中心圆区域）的事件，服务按总序 `(origin_time, run_id, event_id)` 依次扫描，满足以下**全部**条件即并入最早的兼容簇：

1. **台站证据**：候选与簇锚点至少共有 `min_shared_stations` 个观测台站（默认 1，设为 0 可关闭该闸门）；
2. **发震时刻**：与簇内任一成员相差 ≤ `origin_tolerance_s`（默认 5 s）；
3. **空间**：与簇内任一成员震中距 ≤ `radius_km`（默认 50 km）、深度差 ≤ `depth_tolerance_km`（默认 15 km）。

锚点共享台站闸门保证成员可追溯到共同观测集合，时间/空间采用单连接扩展，整个结果与写入顺序、run 数量、数据库返回顺序无关——重复编译逐字节稳定。

每个簇内，**主解**按 `(质量等级 A<B<C<D, RMS, run_id, event_id)` 选择：质量最高优先，其次 RMS 最小，再以 run/event ID 字典序稳定决胜；其余解全部保留为**候选（conflicting candidates）**，并给出相对主解的 `delta`（发震时刻差/纬经差/深度差/RMS 差）。条目还返回台站证据并集 `stations` 与全部成员的交集 `shared_stations`。`min_sources`（默认 1）可丢弃来源不足的簇（设为 2 即只保留跨 run 合并条目）。

### 1. 提交编译（幂等）

```bash
curl -s -X POST http://localhost:8000/v1/catalog/compilations \
  -H 'Content-Type: application/json' -d '{
    "start_time": "2026-09-01T00:00:00Z",
    "end_time":   "2026-10-01T00:00:00Z",
    "radius_km": 50.0,
    "origin_tolerance_s": 5.0,
    "depth_tolerance_km": 15.0,
    "min_quality": "C",
    "min_sources": 1,
    "min_shared_stations": 1,
    "version_conditions": {
      "model_versions": [{"name": "default", "version": "1.0"}],
      "calibration_versions": [],
      "include_uncalibrated": true
    }
  }' | python -m json.tool
```

- 同一请求体（规范化哈希，与字段顺序无关）重复提交返回 **200 + 同一 `compilation_id`/条目 ID**（首次 201，`created=true`）；并发相同请求在唯一索引上折叠为一个结果。
- 空结果也会持久化并幂等（便于固定批次窗口）。
- `version_conditions` 全部可选：`model_versions` 限定模型版本；`calibration_versions` 限定校准版本；`include_uncalibrated`（默认 true）决定是否纳入未校准基线 run。引用不存在的模型/校准返回 404。
- 可选中心圆：`center_latitude` + `center_longitude` + `region_radius_km` 必须同时给出，否则 `422 invalid_spatial_filter`。

响应节选（`sources` 即完整来源链，主解 `is_master=true` 排在首位）：

```json
{
  "compilation_id": "…", "created": true, "entry_count": 1,
  "entries": [{
    "entry_id": "…", "revision": 1, "status": "draft",
    "origin_time": "2026-09-22T00:00:29.764Z",
    "latitude": 35.04912, "longitude": -117.94737, "depth_km": 12.83,
    "rms_residual_s": 0.0726, "station_count": 8, "quality": "A",
    "master": {"run_id": "…", "event_id": "…"},
    "stations": ["CC00", "…"], "shared_stations": ["CC00", "…"],
    "source_count": 3, "distinct_run_count": 3,
    "sources": [{
      "run_id": "…", "event_id": "…", "is_master": true,
      "origin_time": "…", "latitude": 35.04912, "…": "…",
      "model": {"name": "default", "version": "1.0"},
      "calibration": null,
      "delta": {"origin_time_s": 0.0, "latitude": 0.0, "longitude": 0.0,
                "depth_km": 0.0, "rms_s": 0.0}
    }, {
      "run_id": "…", "event_id": "…", "is_master": false,
      "model": {"name": "default", "version": "2.0"},
      "calibration": {"version": "cal-2026-09", "status": "active"},
      "delta": {"origin_time_s": 0.4073, "latitude": -0.07519, "…": "…"}
    }]
  }]
}
```

### 2. 发布与撤回（乐观并发 + 幂等）

```bash
# draft -> published，expected_revision 必须等于当前 revision（草稿初始为 1）
curl -s -X POST http://localhost:8000/v1/catalog/entries/<entry_id>/publish \
  -H 'Content-Type: application/json' -d '{"expected_revision": 1}'

# published -> withdrawn（墓碑，不修改任何内容列与来源链）
curl -s -X POST http://localhost:8000/v1/catalog/entries/<entry_id>/withdraw \
  -H 'Content-Type: application/json' \
  -d '{"expected_revision": 2, "reason": "superseded by Q3 review"}'
```

- 每次**真实状态迁移** `revision` 加 1；`expected_revision` 不匹配返回 `409 revision_conflict`，状态不变（PostgreSQL 行锁 + SQLite 写串行保证并发安全）。
- 对**已是目标状态**的条目重复调用是幂等 no-op：返回 200、不改时间戳、不再增加 revision（客户端重试安全，即使携带旧 revision）。
- 撤回只写 `status/revision/withdrawn_at/withdraw_reason`，**已发布内容（主解、坐标、来源链）原样保留**；撤回后不可再发布（`409 catalog_entry_withdrawn`）；草稿不可直接撤回（`409 catalog_entry_not_published`）。

### 3. 查询：时间 / 区域 / 质量 / 状态 + 稳定游标

```bash
curl -s "http://localhost:8000/v1/catalog/entries?\
start_time=2026-09-01T00:00:00Z&end_time=2026-10-01T00:00:00Z&\
min_latitude=34.5&max_latitude=35.5&min_longitude=-118.5&max_longitude=-117.5&\
min_quality=B&status=published&limit=100"
```

- 结果按 `(origin_time, entry_id)` 键集分页，返回不透明 `next_cursor`；游标携带查询指纹，换时间/区域/质量/状态条件复用旧游标返回 `422 invalid_cursor`。
- `status` 取 `draft` / `published` / `withdrawn` / `all`（默认 `all`）；`min_quality` 默认 `D`（即不过滤），时间窗左闭右开；矩形区域四个边界必须同时给出且 min ≤ max。
- 条目载荷始终包含完整 `sources` 来源链（run/event、模型与校准版本、相对主解偏差）。
- 另可 `GET /v1/catalog/compilations`（列表）与 `GET /v1/catalog/compilations/{id}`（查看某次编译的全部条目）。

## 台站影响审计（leave-one-station-out）

审计是叠加在已完成事件之上的**只读分析层**，与目录一样不写 ingestions/runs/events/picks，也不触碰目录内容。提交时给出**已完成 run 的 `event_id`** 与**最少保留台站数** `min_keep_stations`，服务即冻结分析上下文并逐一做 leave-one-station-out 重算。

### 冻结内容

报告（`frozen` / `baseline`）在创建时固化以下信息，保证审计可复现：

- **原始拾取子集**：该事件关联到的全部 P/S 拾取（pick_id、台站、相位、到时、来源、残差、权重）；
- **速度模型**：原 run 的 `(name, version)` 与完整分层；
- **校准版本**：原 run 若使用校准，固化校准版本号、状态及其逐站分相位偏移项（重算时同样先修正到时）；未校准则为 `null`；
- **处理参数**：原 run 的 `max_residual_s` / `min_stations`，以及本次审计的 `min_keep_stations`；
- 台站几何（经纬高）与原事件基线解（发震时刻、经纬深度、RMS、台站数、质量）。

### 重算与比较

对每个参与台站，仅从**冻结的原始拾取子集**中移除该台站的**全部 P 与 S 拾取**（不引入该 ingestion 上的其它噪声拾取），再以冻结的模型/校准/参数重新执行与 `process`/`replay` **完全相同的确定性关联与定位**，其中最小台站数门槛取 `min_keep_stations`。每个逐台结果比较：

- **事件是否存活**：`survived` 或 `event_lost`；
- **解的变化**：相对基线的发震时刻差（s）、纬经差、深度差（km）、RMS 差（s）、台站数差；
- **质量等级**：是否降级（如 A 需要 ≥8 台，移除任一台站即降为 B）；
- **关联拾取集合**：`removed_pick_ids`（该台 P/S）、`retained_pick_ids`（基线中仍属于重定位事件）、`dropped_pick_ids`（仍在但不再属于该事件）与 `associated_pick_ids`（重定位事件实际关联集合）；四者与基线拾取集严格划分；
- **不可解原因**：事件消失时给出主导原因 `unresolvable_reason`（`isolated` / `insufficient_stations` / `location_failed` / `residual_exceeded`）及各原因的 `reason_counts`。

### 稳定排序与关键台站

逐台结果按以下全序稳定排序（持久化为连续的 `sort_index`，也用作分页顺序）：

1. **事件消失**（`event_lost` 在前）；
2. **质量降级深度**（降级越多越靠前）；
3. **RMS 增幅**（`rms_increase_s` 降序）；
4. **台站码**（字典序，最终决胜）。

只要移除某台站会导致**事件消失或质量降级**，该台站即标注 `is_key_station=true`，报告顶层给出 `key_stations` 与 `key_station_count`。

### 1. 创建审计（幂等）

```bash
curl -s -X POST http://localhost:8000/v1/station-audits \
  -H 'Content-Type: application/json' \
  -d '{"event_id": "<event_id>", "min_keep_stations": 4}' | python -m json.tool
```

- 首次 `201 created=true`；同一 `(event, min_keep_stations)` 重复提交返回 `200` 与**同一 `audit_id`/结果**（规范化 `request_hash`，并发相同请求在唯一索引上折叠）。
- 不同 `min_keep_stations` 是**不同审计**（不同 `params_hash`）。`params_hash` 绑定 run、模型、校准版本、原参数与本次 `min_keep_stations`。
- 校验：事件不存在返回 `404 event_not_found`；若移除任一台站后剩余台站数已不足 `min_keep_stations`（即 `N-1 < min_keep_stations`）返回 `422 min_keep_stations_too_high`。

响应节选（8 台 A 级事件：移除任一台站降为 B，故 8 台均为关键台站；4 台事件移除任一台只剩 3 台、四未知定位欠定，全部 `event_lost`）：

```json
{
  "audit_id": "…", "created": true,
  "event_id": "…", "run_id": "…", "ingestion_id": "…",
  "params_hash": "…",
  "frozen": {
    "model": {"name": "default", "version": "1.0", "layers": [ … ]},
    "calibration": null,
    "params": {"max_residual_s": 1.0, "min_stations": 4},
    "min_keep_stations": 4,
    "stations": [{"station": "EX01", "latitude": …, "longitude": …, "elevation_m": 0.0}]
  },
  "baseline": {"quality": "A", "station_count": 8, "rms_residual_s": 0.0076, "picks": [ … ]},
  "station_count": 8, "result_count": 8, "key_station_count": 8,
  "key_stations": ["EX02", "EX04", …],
  "results": [{
    "station": "EX02", "sort_index": 0,
    "outcome": "survived", "is_key_station": true, "quality_downgraded": true,
    "rms_increase_s": 0.0005,
    "solution": {"origin_time": "…", "latitude": …, "depth_km": …,
                 "rms_residual_s": 0.0081, "station_count": 7, "quality": "B"},
    "delta": {"origin_time_s": 0.02, "latitude": 0.001, "longitude": -0.002,
              "depth_km": 0.3, "rms_s": 0.0005, "station_count": -1},
    "removed_pick_ids": ["…", "…"],
    "associated_pick_ids": [ … ], "retained_pick_ids": [ … ], "dropped_pick_ids": [ … ],
    "stations": ["EX00", "EX01", "EX03", …],
    "unresolvable_reason": null, "reason_counts": {}
  }]
}
```

事件消失的结果：`"outcome": "event_lost"`、`"solution": null`、`"delta": null`、`"rms_increase_s": null`、`"unresolvable_reason": "location_failed"`、`"reason_counts": {"location_failed": 6}`。

### 2. 查询与分页

```bash
# 审计列表（可按 event_id / run_id 过滤），稳定键集分页 (created_at, audit_id)
curl -s "http://localhost:8000/v1/station-audits?event_id=<event_id>&limit=50"

# 单个报告（含全部逐台结果与冻结/基线快照）
curl -s http://localhost:8000/v1/station-audits/<audit_id>

# 逐台结果单独分页（按稳定排序 sort_index），不透明 next_cursor
curl -s "http://localhost:8000/v1/station-audits/<audit_id>/results?limit=20"
```

- 列表游标携带 `(event_id, run_id, limit)` 查询指纹，换过滤条件复用旧游标返回 `422 invalid_cursor`；逐台结果游标绑定具体审计，跨审计复用同样 `422 invalid_cursor`。
- 审计/结果不存在返回 `404 station_audit_not_found`；列表按未知 `event_id`/`run_id` 过滤返回对应 404。
- 审计创建后内容不可变；审计**不修改原事件与目录**，`process`、`replay`、`compare`、目录编译/发布/撤回的请求、响应与幂等键均保持原样。

## P 波初动震源机制反演（first-motion focal mechanism）

反演是叠加在已完成事件之上的**只读分析层**（与目录、审计同级）：客户端针对已完成 run 的某个 event，提交其**关联 P 拾取**的初动极性（`positive`/`negative`）与权重；服务用**事件震源位置、台站坐标和该 run 的分层速度模型**求每条 P 射线的**方位角**（自北顺时针）与**出射角**（自向下垂直方向量起，> 90° 为向上出射，由射线参数 `sin(i) = p·v_震源` 精确求取），然后在 **10° 网格**上搜索双力偶（走向 0–350°、倾角 10–90°、滑动角 −180–170°），以**加权极性误判最少**选出全部最优解。

### 选解规则（确定性）

- **误判**：某台站的预测初动符号（P 辐射花样 `(x·n)(x·u)` 的符号）与观测极性不一致即计一次误判，误判代价为该台站权重；**预测零幅值（节面上，|A| ≤ 1e-6）一律按误判计**。
- **等价解去重（仅互换节面）**：互换两个节面（法向量 ↔ 滑动矢量）得到的是同一矩张量 `M = u·nᵀ + n·uᵀ`；网格搜索用**仅做幅值规范化、不做符号规范化**的矩张量键去重，每个矩张量只保留一个代表解。**整体反号 `(n,u) → (−n,−u)` 得到的是 `−M`（滑动反向机制）**：其 P 辐射花样处处反号、预测极性全部相反，是另一个独立候选，**必须保留并参与最优比较**；`M` 与 `−M` 的加权误判按完整候选集分别计算，因此真实最优解不会因先遍历到反向机制而被漏掉。
- **同分最优解**：所有加权误判等于最小值的机制全部保留，按 **（走向， 倾角， 滑动角）** 全序稳定排序（持久化为连续 `solution_index`，即分页顺序）；每组解返回**两个节面**（`plane_1`/`plane_2`）、`misfit_weight`、`misfit_count` 与**逐台站观测/预测**（方位角、出射角、观测极性、预测极性/幅值、是否匹配）。

### 1. 提交反演（幂等）

```bash
curl -s -X POST http://localhost:8000/v1/focal-mechanisms \
  -H 'Content-Type: application/json' -d '{
    "event_id": "<event_id>",
    "observations": [
      {"pick_id": "<p_pick_id_1>", "polarity": "positive", "weight": 1.0},
      {"pick_id": "<p_pick_id_2>", "polarity": "negative", "weight": 0.5}
    ]}' | python -m json.tool
```

- 首次 `201 created=true`；按（**事件， 模型版本， 规范化极性集**）的规范哈希幂等：同一事件同一极性集（与提交顺序无关；同站同极性的重复提交按最大权重合并）重复提交返回 `200` 与**同一 `inversion_id`/解集**；并发相同请求在唯一索引上折叠为一份结果。
- **旧报告修正**：早期版本的矩张量键额外做了符号规范化，会把滑动反向的 `M` 与 `−M` 合并、漏掉预测极性相反的候选。按旧规则保存的报告带 `dedup_rule_version=1` 标记；服务在**重复提交同一请求或按原 `inversion_id` 查询**（含 `/solutions` 与列表聚合字段）时，用不可变的事件/run/台站/冻结模型分层重算该报告的解集并升级为 `dedup_rule_version=2`：`inversion_id`、`request_hash`、`request`、事件/run 关联与冻结输入保持不变，仅派生的解集与 `solution_count`/`min_misfit_weight` 被重写。重算是**幂等**的（已是 v2 的报告直接返回）且**并发安全**（先做只读重算，再以单条带版本条件的 `UPDATE` 原子认领：SQLite 上为写锁、PostgreSQL 上为行锁，并发修正中恰有一个工作者重写解集，其余直接读取修正后的同一报告）。
- 校验（明确拒绝，不产生任何状态）：事件不存在 `404 event_not_found`；拾取不存在 `404 pick_not_found`；拾取不是该事件的关联 P 拾取（含 S 拾取、他事件拾取、未关联拾取）`422 pick_not_associated`；同一台站提交矛盾极性 `422 contradictory_polarity`；独立台站少于 6 个 `422 insufficient_polarity_stations`；最大方位空档 ≥ 180° `422 azimuthal_gap_too_large`。

响应节选（8 台站、最小误判 0、68 组同分最优解中的第一组）：

```json
{
  "inversion_id": "…", "created": true,
  "event_id": "…", "run_id": "…",
  "grid_step_deg": 10.0, "station_count": 8,
  "min_misfit_weight": 0.0, "solution_count": 68,
  "frozen": {
    "model": {"name": "default", "version": "1.0", "layers": [ … ]},
    "hypocenter": {"origin_time": "…", "latitude": 35.05, "longitude": -117.95, "depth_km": 12.0},
    "stations": [{"station": "EX00", "latitude": 35.25, "…": "…",
                  "azimuth_deg": 348.34, "takeoff_deg": 119.39}],
    "polarities": [{"station": "EX00", "pick_id": "…", "polarity": "negative", "weight": 1.0}],
    "azimuthal_gap_deg": 58.07
  },
  "solutions": [{
    "solution_index": 0,
    "plane_1": {"strike_deg": 20.0, "dip_deg": 40.0, "rake_deg": -90.0},
    "plane_2": {"strike_deg": 200.0, "dip_deg": 50.0, "rake_deg": -90.0},
    "misfit_weight": 0.0, "misfit_count": 0,
    "observations": [{
      "station": "EX00", "pick_id": "…",
      "azimuth_deg": 348.34, "takeoff_deg": 119.39,
      "observed": "negative", "weight": 1.0,
      "predicted": "negative", "amplitude": -0.0545, "match": true
    }]
  }],
  "next_cursor": "…"
}
```

`frozen` 在创建时固化模型分层、震源位置、逐台站射线几何与规范化极性集，保证反演可复现；`solutions` 为同分最优解的第一页（默认 50 条），`next_cursor` 继续翻页。

### 2. 查询与分页

```bash
# 反演列表（可按 event_id 过滤），稳定键集分页 (created_at, inversion_id)
curl -s "http://localhost:8000/v1/focal-mechanisms?event_id=<event_id>&limit=50"

# 单个反演（含冻结快照与同分最优解第一页）
curl -s http://localhost:8000/v1/focal-mechanisms/<inversion_id>

# 同分最优解逐页遍历（稳定 (strike, dip, rake) 顺序），不透明 next_cursor
curl -s "http://localhost:8000/v1/focal-mechanisms/<inversion_id>/solutions?limit=50"
```

- 解分页游标绑定具体反演，跨反演复用返回 `422 invalid_cursor`；列表游标携带 `(event_id, limit)` 查询指纹，换过滤条件复用同样 `422 invalid_cursor`。
- 反演不存在返回 `404 focal_mechanism_not_found`；列表按未知 `event_id` 过滤返回 `404 event_not_found`。
- 除旧规则报告的一次性惰性修正（见上，ID/请求哈希/冻结输入不变）外，反演创建后内容不可变；反演**不修改原事件、run、拾取或目录内容**，`process`、`replay`、`compare`、目录与审计接口语义保持原样。

## 事件定位不确定性评估（station bootstrap）

评估是叠加在已完成事件之上的**只读分析层**，与目录、台站审计、震源机制反演一样不写 ingestions/runs/events/picks，也不触碰目录内容。提交时给出**已完成 run 的 `event_id`**、**重采样次数 `n_resamples`**（200–2000）、**置信水平 `alpha`**（0.8–0.99）与**随机种子 `seed`**，服务即冻结分析上下文并做 N 次台站自助法（bootstrap）重定位。

### 冻结内容

报告（`frozen` / `baseline`）在创建时固化以下信息，保证评估可复现：

- **原始拾取子集**：该事件关联到的全部 P/S 拾取（pick_id、台站、相位、到时、来源、**权重**、残差）；
- **速度模型**：原 run 的 `(name, version)` 与完整分层；
- **校准版本**：原 run 若使用校准，固化校准版本号、状态及其逐站分相位偏移项（重定位时同样先修正到时）；未校准则为 `null`；
- **处理参数**：原 run 的 `max_residual_s` / `min_stations`（后者作为可解性台站数门槛）；
- 台站几何（经纬高）与原事件基线解（发震时刻、经纬深度、RMS、台站数、质量）。

### 重采样与重定位

每次重采样以 `numpy.random.default_rng(seed)` **有放回**抽取**与原参与台站数相同**的台站，保留被抽中台站的**全部冻结关联拾取**；同一台站被抽中 `k` 次时其拾取权重乘以 `k`（与物理重复该台站 `k` 次的正规方程完全等价）。随后用与 `process`/`replay` **完全相同的加权最小二乘定位器**（同一初值与阻尼）重算发震时刻与经纬深度。逐样本记录：

- `status`：`solved` / `failed`；
- `failure_reason`（结构化）：`insufficient_stations`（本次抽样的独立台站数低于冻结的 `min_stations`）、`location_failed`（定位系统欠定/奇异）、`not_converged`（迭代未收敛）；
- `stations`：本次抽样的台站多重集（台站码 → 出现次数），其总数恒等于原台站数；
- 可解样本的完整解（发震时刻、经纬深度、RMS、迭代次数）。

### 统计输出

- **置信区间**：对发震时刻、纬度、经度、深度分别按**最近秩（nearest-rank）**取两侧 `(1−α)/2` 分位点（秩为 `ceil(p·n)`，n 为可解样本数），上下界均为实际样本值；
- **样本协方差**：仅用可解样本，按 `[origin_time_s, latitude, longitude, depth_km]` 顺序计算无偏（ddof=1）协方差矩阵（发震时刻以 epoch 秒计）；
- **可解率**：`n_solved / n_resamples`，附各失败原因聚合计数 `failure_reasons`；
- **明确拒绝**：可解样本少于 `n_resamples` 的 80% 时返回 `422 insufficient_solvable_samples`（消息含可解数、所需下限与失败原因计数），**不持久化任何报告或样本**；同一（事件， 参数， 种子）重复提交得到同一拒绝，效果幂等。

### 1. 创建评估（幂等）

```bash
curl -s -X POST http://localhost:8000/v1/location-uncertainties \
  -H 'Content-Type: application/json' -d '{
    "event_id": "<event_id>",
    "n_resamples": 500,
    "alpha": 0.9,
    "seed": 42
  }' | python -m json.tool
```

- 首次 `201 created=true`；同一 `(event, n_resamples, alpha, seed)` 重复提交返回 `200` 与**同一 `assessment_id`/报告**（规范化 `request_hash`，并发相同请求在唯一索引上折叠为一份）。
- 任一参数（含种子）不同即为**不同评估**；同一种子下逐样本的重采样与解完全确定（可复现）。
- 校验：事件不存在返回 `404 event_not_found`；参数越界（N 不在 200–2000、α 不在 0.8–0.99、种子为负）返回 `422 validation_error`。

响应节选：

```json
{
  "assessment_id": "…", "created": true,
  "event_id": "…", "run_id": "…", "ingestion_id": "…",
  "request": {"event_id": "…", "run_id": "…", "n_resamples": 500, "alpha": 0.9, "seed": 42},
  "n_resamples": 500, "alpha": 0.9, "seed": 42,
  "n_solved": 491, "n_failed": 9, "solvable_rate": 0.982,
  "min_solvable_required": 400,
  "failure_reasons": {"insufficient_stations": 9},
  "confidence_intervals": {
    "alpha": 0.9, "method": "nearest_rank", "sample_count": 491,
    "origin_time": {"lower": "2026-09-22T00:00:29.820Z", "upper": "2026-09-22T00:00:29.851Z"},
    "latitude":  {"lower": 35.04991, "upper": 35.05072},
    "longitude": {"lower": -117.95061, "upper": -117.94988},
    "depth_km":  {"lower": 11.74, "upper": 11.97}
  },
  "covariance": {
    "parameters": ["origin_time_s", "latitude", "longitude", "depth_km"],
    "sample_count": 491,
    "matrix": [[2.1e-05, "…", "…", "…"], ["…", 4.2e-08, "…", "…"], ["…", "…", 3.4e-08, "…"], ["…", "…", "…", 2.3e-03]]
  },
  "frozen": {"model": {"name": "default", "version": "1.0", "layers": ["…"]},
             "calibration": null,
             "params": {"max_residual_s": 1.0, "min_stations": 4},
             "stations": ["…"], "picks": ["…"]},
  "baseline": {"origin_time": "2026-09-22T00:00:29.828Z", "latitude": 35.05021,
               "longitude": -117.95015, "depth_km": 11.96, "quality": "A", "…": "…"}
}
```

### 2. 查询与分页

```bash
# 评估列表（可按 event_id / run_id 过滤），稳定键集分页 (created_at, assessment_id)
curl -s "http://localhost:8000/v1/location-uncertainties?event_id=<event_id>&limit=50"

# 单个报告（含冻结/基线快照、置信区间、协方差与可解率；不含逐样本列表）
curl -s "http://localhost:8000/v1/location-uncertainties/<assessment_id>"

# 逐样本分页（按 sample_index 稳定顺序），可按 status=solved|failed 过滤，不透明 next_cursor
curl -s "http://localhost:8000/v1/location-uncertainties/<assessment_id>/samples?limit=100&status=failed"
```

游标携带查询指纹：把某过滤条件下发的游标用于其他过滤条件（或损坏的游标）返回 `422 invalid_cursor`。报告与样本**只读**，创建后不可变。

## 目录告警编排（catalog alert orchestration）

告警是叠加在**已发布目录**之上的规则驱动层：管理员定义**不可变的告警规则版本**并以 CAS 发布；对任一**目录编译批次**做评估时，服务按规则对批次内**已发布条目**逐条匹配，为每个（规则， 条目）匹配生成**恰好一次**的确定性命中记录，并把连续命中按规则与合并间隔编排成告警事件。该层只读目录、只写自己的表：`process`/`replay`/`compare`、校准、目录与审计接口语义完全不变。

### 规则：不可变版本 + CAS 生命周期

规则条件（全部可选，合取组合）：

| 字段 | 含义 |
|---|---|
| `min_quality` | 条目质量等级不低于该等级：`A`/`B`/`C`/`D`（默认 `D`，即不过滤） |
| `min_stations` | 条目台站数下限（默认不限） |
| `max_rms_s` | 条目 RMS 上限（秒，默认不限） |
| `window_start` / `window_end` | 发震时刻窗（左闭右开，UTC；必须同时给出且 start < end） |
| `center_latitude` / `center_longitude` / `region_radius_km` | 圆形区域，三者必须同时给出 |
| `merge_interval_s` | 连续命中合并间隔（秒，默认 1800，(0, 86400]） |
| `label` | 可选说明文本 |

生命周期：**草稿 → 发布 → 停用**。创建即为草稿（`revision=1`），内容自创建起不可变；同 `(name, version)` 同内容的重复创建幂等返回（200，`created=false`），内容不同返回 `409 alert_rule_version_conflict`。发布与停用都携带 `expected_revision` 做 CAS：

```bash
# 创建（草稿）
curl -s -X POST http://localhost:8000/v1/alert-rules \
  -H 'Content-Type: application/json' -d '{
    "name": "strong-local", "version": "1.0",
    "min_quality": "B", "min_stations": 6, "max_rms_s": 0.5,
    "center_latitude": 35.0, "center_longitude": -118.0, "region_radius_km": 80.0,
    "merge_interval_s": 900.0, "label": "强本地事件"}'

# 发布（expected_revision 必须等于当前 revision）
curl -s -X POST http://localhost:8000/v1/alert-rules/<rule_id>/publish \
  -H 'Content-Type: application/json' -d '{"expected_revision": 1}'

# 停用（不再参与后续评估；既有告警与审计链保留）
curl -s -X POST http://localhost:8000/v1/alert-rules/<rule_id>/retire \
  -H 'Content-Type: application/json' -d '{"expected_revision": 2}'
```

- 每次真实状态迁移 `revision` 加 1；`expected_revision` 不匹配返回 `409 revision_conflict`。
- 对已是目标状态的规则重复调用是幂等 no-op（200，不再加 revision）；草稿不可停用（`409 alert_rule_not_published`），停用后不可再发布（`409 alert_rule_retired`）。
- 只有**已发布**规则参与评估；评估时显式引用草稿/停用规则分别返回 `409 alert_rule_not_published` / `409 alert_rule_retired`。

### 评估：确定性命中 + 幂等重评估

```bash
curl -s -X POST http://localhost:8000/v1/alerts/evaluations \
  -H 'Content-Type: application/json' -d '{
    "compilation_id": "<compilation_id>",
    "rule_ids": null}' | python -m json.tool
```

- `rule_ids` 省略/`null` 表示评估**当前全部已发布规则**；显式列表则只评估这些规则（引用不存在的规则返回 `404 alert_rule_not_found`）。
- 评估按 `(origin_time, entry_id)` 顺序扫描批次内**已发布**条目（草稿与已撤回条目不参与），规则按 `(name, version, id)` 排序，命中判定与合并完全确定，与请求时机、并发无关。
- 评估按规范请求哈希幂等：同一请求首次 `201 created=true`；**重复同一请求即幂等重评估**（200，`created=false`）——只补录新出现的匹配（例如评估后才发布的条目、或 `rule_ids=null` 时新发布的规则），既有命中原样保留，`new_hit_count` 报告本次新录数量，`evaluation_count` 累计评估次数。显式规则集在创建时冻结；规则停用后的重评估会自动跳过它。
- 响应含 `rules`（本次实际评估的规则）、`entry_count`（纳入的已发布条目数）、`matched_count`（范围内全部命中）、`new_hit_count`、`hits`（含每条命中的 `disposition`/`reason`/条目快照/所属告警）。

### 命中合并：抑制与升级

每条命中记录 `disposition` 与 `reason`：

- **`opened`**：该命中新开一条告警（`reason=null`）。
- **`merged`**（抑制）：同规则存在未关闭告警，且该命中与其最近一次命中的发震时刻间隔 ≤ `merge_interval_s`，则并入该告警而不是新开告警，`reason=within_merge_interval`；告警的命中数、首末命中时刻、最优质量、最大台站数、最小 RMS 随之确定性更新。合并且可以**跨编译批次**（同一规则的连续命中跨批次仍合并）。
- **`escalated`**（升级）：合并目标告警已被确认（`new_hit_after_acknowledge`）或已解除（`new_hit_after_resolve`）时，命中将其**升级回 `open`**，提醒运维情况仍在发展。
- 已**关闭**（closed）的告警是终态，绝不吸收新命中——间隔内的新命中会新开告警。

### 确认、解除与审计链

```bash
curl -s -X POST http://localhost:8000/v1/alerts/<alert_id>/acknowledge \
  -H 'Content-Type: application/json' -d '{"note": "oncall 已接手"}'
curl -s -X POST http://localhost:8000/v1/alerts/<alert_id>/resolve \
  -H 'Content-Type: application/json' -d '{"note": "已核实为标定脉冲"}'
```

- 状态机：`open → acknowledged → resolved`（确认/解除均为幂等迁移，重复调用是 no-op，不重复写审计链）；已解除不可再确认（`409 alert_already_resolved`）；已关闭不可确认/解除（`409 alert_closed`）。请求体可整体省略。
- 每条告警的完整审计链（`events`，按 `seq` 全序）记录：每次命中（opened/merged/escalated，关联 hit 与原因）与每次状态迁移（acknowledged/resolved/closed，含操作备注或关闭原因）。`GET /v1/alerts/{alert_id}` 返回命中明细 + 审计链 + 当前状态。

### 目录条目撤回的联动

目录条目被撤回时（同一事务内）：凡命中过该条目且仍为 **`open`（未确认）** 的告警被自动关闭（`closed`），审计链追加 `closed` 事件（`reason=catalog_entry_withdrawn`，附条目 ID 与撤回原因）；**已确认/已解除的告警保持不变**，全部命中记录与审计链永久保留。撤回联动与评估通过同一临界区（进程内锁 + PostgreSQL 咨询锁）序列化：评估要么看到条目仍已发布（其告警随后被关闭），要么看到条目已撤回（不再对其产生命中），不会漏关。

### 并发与一致性保证

- **不重复**：命中以 `(rule_id, entry_id)` 数据库唯一约束兜底，任何并发/重试下每个（规则， 条目）匹配至多记录一次。
- **不漏报**：评估覆盖批次内全部已发布条目；重评估补录新发布条目；`rule_ids=null` 时新发布的规则也随下次重评估生效。
- **序列化**：所有告警状态变更（评估、确认、解除、撤回联动）进入同一临界区——进程内锁（SQLite/单进程）+ PostgreSQL `pg_advisory_xact_lock`（多进程部署），并发相同评估在唯一索引上折叠为一次创建。

### 查询（稳定游标）

```bash
curl -s "http://localhost:8000/v1/alerts?status=open&rule_id=<rule_id>&limit=50"
curl -s "http://localhost:8000/v1/alerts?compilation_id=<compilation_id>"
curl -s "http://localhost:8000/v1/alert-rules?status=published&limit=50"
curl -s "http://localhost:8000/v1/alerts/evaluations?compilation_id=<compilation_id>"
```

- 告警/规则/评估列表均按 `(created_at, id)` 键集分页，返回不透明 `next_cursor`；游标携带查询指纹，换过滤条件复用旧游标返回 `422 invalid_cursor`。
- 告警过滤：`rule_id`、`status`（`open`/`acknowledged`/`resolved`/`closed`/`all`）、`compilation_id`（在该批次有命中的告警）；按未知 `rule_id`/`compilation_id` 过滤返回对应 404。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/v1/models` | 创建不可变模型版本（同名同版本同内容幂等，内容不同返回 409） |
| GET | `/v1/models`、`/v1/models/{name}/{version}` | 查询模型 |
| POST | `/v1/models/{name}/{version}/calibrations` | 创建校准候选（同请求幂等；门槛不达标 422 拒绝，不留状态） |
| GET | `/v1/models/{name}/{version}/calibrations`、`.../calibrations/{cal_version}` | 列出/查看校准版本与逐站分相位指标 |
| POST | `/v1/models/{name}/{version}/calibrations/{cal_version}/activate` | 原子激活（复核留出 RMS；旧激活版本转为已发布且不可变） |
| POST | `/v1/process` | 上传波形窗口并关联定位（同一窗口幂等，可选 `calibration`） |
| POST | `/v1/runs` | 按模型/校准版本重放已有 ingestion（幂等） |
| GET | `/v1/runs/{run_id}` | 查询 run 完整结果 |
| GET | `/v1/runs/compare?run_a=&run_b=` | 比较两个 run 的事件（按发震时刻 ±2 s 匹配） |
| GET | `/v1/events/{event_id}` | 查询单个事件 |
| POST | `/v1/catalog/compilations` | 跨 run 确定性聚类编译（同请求幂等；首次 201，重复 200） |
| GET | `/v1/catalog/compilations`、`/v1/catalog/compilations/{id}` | 列出/查看编译与其条目 |
| GET | `/v1/catalog/entries` | 时间/矩形区域/质量/状态查询 + 稳定游标分页 |
| GET | `/v1/catalog/entries/{entry_id}` | 查询单条目录条目（含完整来源链） |
| POST | `/v1/catalog/entries/{entry_id}/publish` | 发布草稿（`expected_revision` 乐观锁，重复幂等） |
| POST | `/v1/catalog/entries/{entry_id}/withdraw` | 撤回已发布条目（不改写内容，重复幂等） |
| POST | `/v1/station-audits` | 对已完成事件做台站影响审计（冻结拾取/模型/校准/参数，逐台移除 P/S 重算；同请求幂等，首次 201，重复 200） |
| GET | `/v1/station-audits` | 审计列表（可按 `event_id`/`run_id` 过滤，稳定游标分页） |
| GET | `/v1/station-audits/{audit_id}` | 查询单个审计（含冻结快照、基线与全部逐台结果） |
| GET | `/v1/station-audits/{audit_id}/results` | 逐台审计结果分页（稳定排序顺序，游标分页） |
| POST | `/v1/focal-mechanisms` | P 波初动震源机制反演（10° 网格搜索，仅互换节面去重、保留滑动反向候选；同请求幂等，首次 201，重复 200） |
| GET | `/v1/focal-mechanisms` | 反演列表（可按 `event_id` 过滤，稳定游标分页） |
| GET | `/v1/focal-mechanisms/{inversion_id}` | 查询单个反演（含冻结快照与同分最优解第一页） |
| GET | `/v1/focal-mechanisms/{inversion_id}/solutions` | 同分最优解分页（稳定 (strike, dip, rake) 顺序，游标分页） |
| POST | `/v1/location-uncertainties` | 事件定位不确定性评估（冻结拾取/权重/模型/校准，种子化台站 bootstrap 重定位 N 次；同请求幂等，首次 201，重复 200；可解样本 < 80% 时 422 拒绝不留状态） |
| GET | `/v1/location-uncertainties` | 评估列表（可按 `event_id`/`run_id` 过滤，稳定游标分页） |
| GET | `/v1/location-uncertainties/{assessment_id}` | 查询单个评估（冻结快照、基线、置信区间、协方差、可解率） |
| GET | `/v1/location-uncertainties/{assessment_id}/samples` | 逐样本分页（重采样顺序，可按 `status=solved|failed` 过滤，游标分页） |
| POST | `/v1/alert-rules` | 创建不可变告警规则版本（同 `(name, version)` 同内容幂等；内容不同 409） |
| GET | `/v1/alert-rules`、`/v1/alert-rules/{rule_id}` | 列出（`name`/`status` 过滤 + 游标分页）/查看告警规则 |
| POST | `/v1/alert-rules/{rule_id}/publish` | 发布草稿规则（`expected_revision` CAS，重复幂等） |
| POST | `/v1/alert-rules/{rule_id}/retire` | 停用已发布规则（CAS，重复幂等；不再参与评估） |
| POST | `/v1/alerts/evaluations` | 评估目录编译批次（同请求幂等：首次 201，重复 200 为幂等重评估，只补录新匹配） |
| GET | `/v1/alerts/evaluations`、`/v1/alerts/evaluations/{evaluation_id}` | 列出（可按 `compilation_id` 过滤，游标分页）/查看评估与命中 |
| GET | `/v1/alerts` | 告警列表（`rule_id`/`status`/`compilation_id` 过滤 + 稳定游标） |
| GET | `/v1/alerts/{alert_id}` | 告警详情（命中明细 + 完整审计链） |
| POST | `/v1/alerts/{alert_id}/acknowledge` | 确认告警（幂等；已解除 409，已关闭 409） |
| POST | `/v1/alerts/{alert_id}/resolve` | 解除告警（幂等；已关闭 409） |

### `POST /v1/process` 请求体

```json
{
  "waveforms": [{
    "station": "EX00",
    "latitude": 35.25, "longitude": -118.0, "elevation_m": 0.0,
    "sampling_rate": 100.0,
    "start_time": "2026-09-22T00:00:00Z",
    "data_b64": "<base64 miniSEED>"
  }],
  "manual_picks": [{"station": "EX00", "phase": "P", "time": "2026-09-22T00:00:34.260Z", "weight": 1.0}],
  "model": {"name": "default", "version": "1.0"},
  "params": {"max_residual_s": 1.0, "min_stations": 4},
  "calibration": {"version": "cal-2026-09"}
}
```

`calibration` 可省略（未校准基线）；`POST /v1/runs` 接受同样的可选字段。

### `POST /v1/models/{name}/{version}/calibrations` 请求体

| 字段 | 含义 |
|---|---|
| `version` | 校准版本号（同模型下唯一、不可变） |
| `train_start` / `train_end` | 训练时间段（左闭右开，UTC） |
| `holdout_start` / `holdout_end` | 留出时间段（左闭右开，必须晚于训练结束） |
| `min_quality` | 可纳入事件的最低质量等级：`A` / `B`（默认）/ `C` |
| `min_events` | 训练窗最少合格事件数（默认 5） |
| `min_span_days` | 训练事件最短时间跨度（天，默认 1） |
| `min_station_coverage` | 每个相位的最低台站覆盖率（0–1，默认 0.6；相对训练+留出台站全集） |
| `huber_delta_s` | Huber 门槛（秒，默认 0.5） |
| `label` | 可选说明文本 |

### `POST /v1/catalog/compilations` 请求体

| 字段 | 含义 |
|---|---|
| `start_time` / `end_time` | 事件发震时刻窗（左闭右开，UTC；必须 start < end） |
| `radius_km` | 震中链接半径（km，默认 50） |
| `origin_tolerance_s` | 发震时刻链接容差（秒，默认 5） |
| `depth_tolerance_km` | 深度链接容差（km，默认 15） |
| `min_quality` | 纳入事件的最低质量：`A`/`B`/`C`（默认）/`D` |
| `min_sources` | 成条所需最少来源事件数（默认 1；设 2 即只保留跨 run 合并条目） |
| `min_shared_stations` | 候选与簇锚点至少共有的台站数（默认 1；0 关闭台站证据闸门） |
| `center_latitude` / `center_longitude` / `region_radius_km` | 可选中心圆过滤，三者必须同时给出 |
| `version_conditions` | 可选版本条件：`model_versions`（模型版本白名单）、`calibration_versions`（校准版本白名单）、`include_uncalibrated`（默认 true，是否纳入未校准基线 run） |

发布/撤回请求体：`{"expected_revision": <当前 revision>}`，撤回额外要求 `reason`（1–500 字符）。

### `POST /v1/station-audits` 请求体

| 字段 | 含义 |
|---|---|
| `event_id` | 已完成 run 中的事件 ID（必填） |
| `min_keep_stations` | 逐台移除后重定位事件需满足的**最少保留台站数**（默认 4，范围 2–200）；必须满足 `min_keep_stations <= N-1`（N 为该事件参与台站数），否则 `422 min_keep_stations_too_high`。重算仍沿用原 run 的模型、校准与 `max_residual_s`/`min_stations` 语义，仅最小台站门槛取本字段 |

### `POST /v1/focal-mechanisms` 请求体

| 字段 | 含义 |
|---|---|
| `event_id` | 已完成 run 中的事件 ID（必填） |
| `observations` | 极性观测列表（1–512 条）：`pick_id`（必须是该事件的**关联 P 拾取**）、`polarity`（`positive`/`negative`）、`weight`（0–10，默认 1.0）。同一台站只接受一种极性：同站同极性的重复提交按最大权重合并，同站矛盾极性返回 `422 contradictory_polarity`；规范化后独立台站须 ≥ 6 且最大方位空档 < 180° |

### `POST /v1/location-uncertainties` 请求体

| 字段 | 含义 |
|---|---|
| `event_id` | 已完成 run 中的事件 ID（必填） |
| `n_resamples` | 台站 bootstrap 重采样次数（200–2000，必填） |
| `alpha` | 置信水平（0.8–0.99，必填）；置信区间按最近秩取两侧 `(1−α)/2` 分位点 |
| `seed` | 随机种子（0 到 2³¹−1，必填）；同一（事件， 参数， 种子）的重采样序列与逐样本解完全确定 |

### `POST /v1/alert-rules` 请求体
| 字段 | 含义 |
|---|---|
| `name` / `version` | 规则名与版本号（`(name, version)` 唯一、不可变） |
| `min_quality` | 最低质量等级 `A`/`B`/`C`/`D`（默认 `D`） |
| `min_stations` / `max_rms_s` | 台站数下限 / RMS 上限（秒；默认不限） |
| `window_start` / `window_end` | 发震时刻窗（左闭右开，UTC；须同时给出且 start < end） |
| `center_latitude` / `center_longitude` / `region_radius_km` | 圆形区域，三者须同时给出 |
| `merge_interval_s` | 连续命中合并间隔（秒，默认 1800） |
| `label` | 可选说明文本 |

发布/停用请求体：`{"expected_revision": <当前 revision>}`。

### `POST /v1/alerts/evaluations` 请求体

| 字段 | 含义 |
|---|---|
| `compilation_id` | 待评估的目录编译批次 ID（必填） |
| `rule_ids` | 可选规则 ID 列表；省略/`null` 表示评估当前全部已发布规则 |

确认/解除请求体可省略或携带 `{"note": "<=500 字符"}`。

## 处理流程

1. **校验**：逐台站检查采样率、坐标/高程边界，解码 miniSEED 并交叉校验头部采样率/起点/数据量。
2. **拾取**：去趋势 + 1–15 Hz 零相位带通，0.5/5 s STA/LTA（触发 3.0 / 解除 1.0），AIC 变化点精化到时；人工拾取直接并入（默认权重 1.0）。
3. **关联**：按时间顺序以未使用拾取为种子，在台网孔径时间窗内聚类；每台站首拾取按 P 引导定位；迭代“定位 → 相位判定（未标注拾取取 P/S 残差较小者）→ 按（|残差|, 台站代码）稳定排序 → 按最大残差筛选 → 每（台站， 相位）保留最优 → 重定位”直至集合稳定；台站数达标则成事件，每台站 P、S 各取一个。
4. **定位**：加权最小二乘（Geiger 法），未知数为发震时刻与经纬深度（局部直角坐标系，含台站高程改正），水平偏导数用射线参数精确值（dT/dX = p），深度偏导数中心差分，Levenberg 阻尼，深度约束 0–700 km。
5. **定级**：A（RMS ≤ 0.5 s 且 ≥ 8 台）、B（≤ 1.0 s 且 ≥ 6 台）、C（≤ 2.0 s 且 ≥ 4 台）、其余 D。
6. **校准修正（可选）**：当请求指定了已发布/激活的校准版本，每个到时在关联与定位前按（台站， 相位）减去对应偏移；返回的残差为修正后残差，响应回显校准版本。

## 幂等与版本化

- **同一波形窗口幂等**：窗口哈希 = 全部波形（台站/采样率/起点/数据 SHA-256）+ 人工拾取的规范哈希；重复提交返回同一 `run_id` 与事件 ID（`created=false`）。
- **模型版本不可变**：`(name, version)` 唯一；内容不同的重复创建返回 `409 model_version_conflict`，内容相同则幂等返回。
- **按版本重放**：`POST /v1/runs` 以 `(ingestion_id, model_id, calibration_id, params_hash)` 去重，可用不同模型/校准版本重放同一窗口，`/v1/runs/compare` 输出发震时刻/位置/深度/RMS/台站数差异。
- **校准版本生命周期**：`candidate`（创建后）→ `active`（唯一，激活后）→ `published`（被新版本替换后，永久只读）。创建按完整请求哈希与 `(模型, 版本号)` 双重幂等；激活是单事务原子操作（旧版本降级 + 新版本生效 + 模型激活指针切换），重复与并发激活均不会产生部分状态。
- **校准只取未校准基线事件**：估计残差来自该模型 `calibration_id IS NULL` 的历史 run，避免用校准结果再训练校准造成反馈。
- **跨运行编译幂等**：编译按完整请求的规范哈希去重（唯一索引），重复与并发相同请求返回同一 `compilation_id` 与条目 ID；条目内容在创建后不可变，仅状态机迁移（草稿→发布→撤回）更新 `revision`/时间戳/撤回原因；同一目标状态的重复发布/撤回为 no-op，不改时间戳也不增加 revision。
- **目录不影响既有语义**：编译/发布/撤回只新增目录表，不修改 ingestions/runs/events/picks；`process`、`replay`、`compare` 与校准接口的请求、响应与幂等键均保持原样。
- **台站影响审计幂等且只读**：审计按 `(event_id, run_id, min_keep_stations)` 的规范哈希（`request_hash`）去重，重复/并发相同请求返回同一 `audit_id` 与逐台结果；不同 `min_keep_stations` 是不同审计（`params_hash` 同时绑定 run、模型/校准版本、原参数与本次门槛）。审计在内存中重算关联/定位，只写独立的审计表，冻结的拾取/模型/校准/参数随报告固化；它**不修改原事件、run、拾取或目录内容**，`process`、`replay`、`compare` 与目录接口语义保持原样。
- **震源机制反演幂等且只读**：反演按 `(事件, 模型版本, 规范化极性集)` 的规范哈希（`request_hash`）去重——极性集按台站排序、同站同极性重复项按最大权重合并，因此与提交顺序无关；重复/并发相同请求在唯一索引上折叠为同一 `inversion_id` 与解集。模型分层、震源位置、逐台站射线几何与极性集随报告冻结，同分最优解一次性持久化并按（走向， 倾角， 滑动角）稳定排序分页；去重只合并互换节面的同一矩张量，滑动反向（`−M`）候选单独参与比较。反演只写独立的反演表，**不修改原事件、run、拾取或目录内容**，既有接口的请求、响应与幂等键均保持原样。旧版把 `M`/`−M` 合并的报告在重复提交或按原 ID 查询时惰性重算（幂等、单条条件 UPDATE 原子认领的并发安全修正），仅重写派生出的解集。
- **目录告警编排幂等且只增不改**：规则版本内容自创建起不可变，仅生命周期迁移（草稿→发布→停用）更新 `revision`/时间戳；评估按规范请求哈希去重（唯一索引），重复请求即幂等重评估，只补录新匹配；命中以 `(rule_id, entry_id)` 唯一约束保证任何并发下不重复，序列化临界区（进程内锁 + PostgreSQL 咨询锁）保证评估/确认/解除/撤回联动互不交错、不漏报；告警确认/解除是幂等迁移，条目撤回只关闭未确认告警且不改写任何命中与审计链；告警层不写 ingestions/runs/events/picks 与目录内容，既有接口的请求、响应与幂等键均保持原样。
- **定位不确定性评估幂等且只读**：评估按 `(event_id, run_id, n_resamples, alpha, seed)` 的规范哈希（`request_hash`）去重，重复/并发相同请求在唯一索引上折叠为同一 `assessment_id` 与报告；任一参数（含种子）不同即为不同评估。冻结的拾取（含权重）/模型/校准/参数随报告固化，重采样序列由种子完全确定（`numpy.random.default_rng(seed)` 逐次有放回抽取），逐样本结果一次性持久化且创建后不可变；可解样本不足 80% 的请求被 422 拒绝且不留任何状态（重复提交得到同一拒绝）。评估只写独立的评估/样本表，**不修改原事件、run、拾取或目录内容**，既有接口语义保持原样。

## 错误码

所有领域错误返回 `{"detail": {"code": …, "message": …}}`：

| HTTP | code | 含义 |
|---|---|---|
| 422 | `missing_sampling_rate` | 缺采样率 |
| 422 | `invalid_sampling_rate` / `sampling_rate_too_low` | 采样率非法 / 低于 40 Hz（无法支撑 1–15 Hz 固定频带） |
| 422 | `sampling_rate_mismatch` / `start_time_mismatch` | 声明值与 miniSEED 头部不一致 |
| 422 | `coordinate_out_of_bounds` | 经纬度或高程越界 |
| 422 | `insufficient_data` | 数据时长 < 10 s（无法支撑 5 s LTA） |
| 422 | `invalid_base64` / `invalid_miniseed` | 负载无法解码 |
| 422 | `duplicate_station` / `unknown_pick_station` | 台站重复 / 人工拾取引用未知台站 |
| 404 | `model_not_found` / `ingestion_not_found` / `run_not_found` / `event_not_found` / `calibration_not_found` | 资源不存在 |
| 409 | `model_version_conflict` | 模型版本不可变冲突 |
| 409 | `calibration_version_conflict` | 校准版本号已存在但请求参数不同 |
| 409 | `holdout_gate_failed` | 激活时留出 RMS 未严格低于未校准基线 |
| 409 | `calibration_not_active` | 候选版本未激活，不能用于处理/重放 |
| 409 | `calibration_published_immutable` | 已发布版本不可再次激活或修改 |
| 409 | `calibration_activation_conflict` | 并发激活冲突（部分唯一索引兜底） |
| 422 | `invalid_time_window` | 训练/留出时间段非法或重叠 |
| 422 | `insufficient_events` / `insufficient_holdout_events` | 训练/留出合格事件不足 |
| 422 | `time_span_too_short` / `station_coverage_too_low` | 训练时间跨度 / 台站覆盖率不达标 |
| 422 | `invalid_catalog_time_window` | 目录编译/查询时间窗非法或只给一侧 |
| 422 | `invalid_spatial_filter` | 中心圆参数不完整，或矩形边界不完整/越界 |
| 422 | `invalid_cursor` | 分页游标损坏，或与当前查询条件不匹配 |
| 404 | `catalog_compilation_not_found` / `catalog_entry_not_found` | 目录编译/条目不存在 |
| 409 | `revision_conflict` | 发布/撤回的 `expected_revision` 与当前 revision 不符 |
| 409 | `catalog_entry_not_published` / `catalog_entry_withdrawn` | 草稿不可撤回 / 撤回条目不可再发布 |
| 404 | `station_audit_not_found` | 台站影响审计不存在 |
| 422 | `min_keep_stations_too_high` | 最少保留台站数不小于事件参与台站数（移除一台后无法满足） |
| 422 | `audit_event_has_no_picks` | 审计目标事件没有关联拾取 |
| 404 | `pick_not_found` / `focal_mechanism_not_found` | 拾取 / 震源机制反演不存在 |
| 422 | `pick_not_associated` | 拾取不是该事件的关联 P 拾取（含 S 拾取、他事件拾取、未关联拾取） |
| 422 | `contradictory_polarity` | 同一台站提交了矛盾的正负极性 |
| 422 | `insufficient_polarity_stations` / `azimuthal_gap_too_large` | 独立台站少于 6 个 / 最大方位空档不小于 180° |
| 404 | `alert_rule_not_found` / `alert_not_found` / `alert_evaluation_not_found` | 告警规则 / 告警 / 评估不存在 |
| 409 | `alert_rule_version_conflict` | 同 `(name, version)` 规则内容不同（规则版本不可变） |
| 409 | `alert_rule_not_published` / `alert_rule_retired` | 草稿规则不可停用/评估 / 规则已停用不可再发布或评估 |
| 409 | `alert_already_resolved` / `alert_closed` | 已解除告警不可确认 / 已关闭告警不可确认或解除 |
| 422 | `invalid_rule_time_window` | 规则发震时刻窗只给一侧或 start ≥ end |
| 404 | `location_uncertainty_not_found` | 定位不确定性评估不存在 |
| 422 | `insufficient_solvable_samples` | bootstrap 可解样本不足重采样次数的 80%（拒绝且不留状态） |

未关联拾取原因：`isolated`（窗内无其他拾取）、`insufficient_stations`（台站数不足）、`location_failed`（定位不收敛/欠定）、`residual_exceeded`（残差超限）。

bootstrap 样本失败原因：`insufficient_stations`（本次抽样的独立台站数低于冻结的 `min_stations`）、`location_failed`（定位系统欠定/奇异）、`not_converged`（定位迭代未收敛）。

## 本地开发与测试

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -q                 # 169 个测试（走时/定位/关联/Huber/校准生命周期/校准幂等/跨运行目录聚类/生命周期/分页/台站影响审计 leave-one-out/幂等/分页/只读/目录告警编排规则生命周期/评估幂等/合并抑制升级/确认解除/撤回联动/并发/游标/P 波初动震源机制反演网格搜索/仅互换节面去重与保留滑动反向候选/旧报告幂等并发安全惰性修正/拒绝/分页/只读/定位不确定性台站 bootstrap/最近秩分位/协方差/可解率门槛/幂等/并发/游标/只读，端到端 API，SQLite）
DATABASE_URL=sqlite:////tmp/dev.db uvicorn app.main:app --reload   # 无 PostgreSQL 时本地调试
```

生产部署通过 `DATABASE_URL` 注入 PostgreSQL 连接串（compose 已配置）。

## 目录结构

```
app/
  main.py               # FastAPI 入口、建表、默认模型播种、异常处理
  config.py             # 固定处理参数与阈值（环境变量 DATABASE_URL）
  database.py           # 引擎/会话
  models.py             # SQLAlchemy 表（模型/摄取/波形/拾取/run/事件/校准/目录/审计/震源机制反演/告警/定位不确定性评估）
  schemas.py            # Pydantic 请求模型
  api/routes.py         # HTTP 路由
  services/
    traveltime.py       # 分层模型走时（射线参数二分）
    picking.py          # 1–15 Hz + STA/LTA + AIC 拾取
    location.py         # 加权最小二乘定位
    association.py      # 确定性关联（支持校准到时修正）
    quality.py          # 质量等级
    huber.py            # 零和约束 Huber 台站项估计（IRLS + KKT）
    waveform.py         # miniSEED 解码与校验
    catalog.py          # 跨 run 确定性聚类（时间/空间/台站证据）与主解决胜
    catalog_pipeline.py # 目录编译/发布撤回（乐观锁）/游标分页/来源链编排/撤回联动关闭告警
    station_audit.py    # 台站影响审计：冻结基线、逐台移除 P/S 重关联定位、关键台站与稳定排序/分页
    focal_mechanism.py  # P 波初动震源机制反演：射线方位/出射角、10° 网格搜索双力偶、
                        # 矩张量键仅对互换节面去重（保留滑动反向 −M 候选）、同分最优解稳定分页、
                        # 幂等冻结保存、旧版 M==−M 报告的幂等并发安全惰性修正
    alerting.py         # 目录告警编排：不可变规则版本与 CAS 生命周期、确定性评估与命中合并（抑制/升级）、
                        # 确认/解除、撤回联动关闭、序列化临界区与游标分页
    location_uncertainty.py # 定位不确定性评估：冻结拾取/权重/模型/校准，种子化台站 bootstrap 重定位、
                        # 最近秩置信区间、可解样本协方差与可解率门槛、幂等冻结保存、游标分页
    pipeline.py         # 摄取/关联/重放/校准创建与原子激活/比较编排与持久化
scripts/make_example.py # 生成合成示例请求
tests/                  # pytest（单元 + 端到端，169 个测试，SQLite）
```
