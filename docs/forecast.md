# 交付时长区间与覆盖回放

本模块在既有资料读取与摘要之上，提供**按预测发生时点冻结**的截顶交付时长
点预测、批组共形区间，以及历史覆盖回放。既有 `app.service.inspect`、
`app.catalog.load` 与 `app.baseline.analyze` 的语义保持不变。

## 入口

    python -m app.forecast_cli forecast fixtures/synthetic.json [CONFIG.json]
    python -m app.forecast_cli replay W1.json W2.json W3.json --config CONFIG.json
    python -m app.forecast_cli settle FROZEN.json LATER.json

Python 入口：

* `app.forecast.inspect_forecast(payload, config=None, store=None)` 冻结一次预测。
* `app.forecast.settle(record, payload, store=None)` 用后到资料结算旧预测。
* `app.forecast.replay(payloads, config=None)` 逐时点冻结并结算，输出覆盖回放。

## 目标与截顶

预测目标是**截顶时长**：

    target = min(completed - shipped, horizon)

超过 `metadata.horizon` 的完成只记录到上限；没有已知完成且已过
`shipped + horizon + reporting_lag` 的发运记为目标 `horizon`，标签类型为
`cap_censored`，与恰好等于上限的观察标签区分。所有覆盖承诺仅针对该截顶
目标，**不解释为真实长尾交付时长的覆盖**。

区间端点按目标定义域 `[0, horizon]` 收拢；收拢前后的端点、是否收拢都逐行
保留（`raw_lower`、`raw_upper`、`bounds_clipped_to_target_range`）。

## 预测时点与可见资料

* 预测发生时点是该发运的 `shipped`（发运日）；不是 `as_of`。
* 特征仅当 `feature_known <= shipped` 且该修订在预测时点已到达时可用。
* 发运修订的到达时间取 `recorded_at`，缺省等于 `feature_known`。完成记录的
  到达时间取 `recorded_at`，缺省等于 `known`。撤回记录必须显式给出整数
  `recorded_at`，否则无法判断其是否在预测时点已发生，直接拒绝。
* 发运修订的到达时间取 `recorded_at`，缺省等于 `feature_known`。完成记录的
  到达时间取 `recorded_at`，缺省等于 `known`。撤回记录必须显式给出整数
  `recorded_at`，否则无法判断其是否在预测时点已发生，直接拒绝。
* 预测时点的可见视图按 `recorded_at <= as_of` 重建后再取每个 `id` 的最高
  可见修订。完成记录还须满足 `known <= as_of`（时序有效性）。这条规则的
  必要性在于：`catalog.load` 只保留当前最高修订，如果更高修订或补录标签在
  预测之后才到，直接使用它会把未来资料回写旧预测。
* 缺失特征用**训练批组**的中位数填补并附加缺失标记；未完成、缺失和未来值
  都不会被当作 0。若某发运在预测时点特征不可用，该行标为
  `feature_unavailable_at_forecast_time`，不给点值也不参与校准。
* 未成熟发运的标签是 `None`（`kind = immature`），不是 0。

## 批组隔离

分派单位是 `batch_group`，组内不拆行：

* 训练、验证、校准、留出按已成熟批组的首个发运日排序，用配置的比例切分
  （`split_fractions`，缺省 `0.5 / 0.65 / 0.85`）。
* 也可用 `group_split` 提供事前清单。清单是校准队列的冻结来源；清单里未
  成熟的批组整批降为 `pending` 并写入 `manifest_demotions`，不会因为“已有
  完成标签”被提前放行。
* 拟合时若同批任一行特征不可用，整批排除并记录，不保留组内“好”的行。
* 输出逐批列出 `evaluation_role`、`roster` 与 `roster_digest`，可直接核对
  没有批组跨角色、没有同行拆分。

## 校准资格（不按标签挑快单）

校准队列在发布时确定，资格只看日历：

    shipped + horizon + reporting_lag <= as_of

某批组只有**整批**都过该时点才可进入校准；已收到完成标签但未过观察期的
快单批组保持 `pending`。`calibration_eligibility` 逐批列出是否合格与原因，
`split.excluded_from_calibration` 给出被排除的校准批组。

## 点预测、分位基线与模型选择

* 特征空间（键、中位数、均值、标准差）只在训练批组上拟合。
* 点预测是标准化后的岭回归；目标按训练均值和标准差标准化。
* 基线区间有两种候选：残差分位策略与线性分位回归策略。
* 正则强度候选 `lambdas` 和区间策略都在**验证批组**上按平均绝对误差 /
  分位损失选择；校准批组不参与选择。没有成熟验证批组时，选择结果明确记为
  `fallback_no_validation_groups`，并声明不据此声称模型选择。
* 所有训练、验证、校准、留出批组及组内行都按稳定顺序遍历，输入顺序与相同
  记录重试不改变结果。

## 批组共形

* 交换单位是批组。批组得分是组内逐发运不符合分数的最大值：

      score(group) = max_i max(lower_i - y_i, y_i - upper_i, 0)

  因此承诺的是**同一新批组冻结名单的联合覆盖**，不是逐行的边际覆盖。
* 有限样本秩：`k = ceil((n + 1) * (1 - alpha))`，阈值取第 k 个最小批组得分。
  秩与 n 一同输出；`k > n` 时 `rank_sufficient = false`，`threshold = null`，
  状态为 `insufficient_calibration_groups`，不编造区间。
* 并列得分按稳定顺序取第 k 个；这不会低于名义覆盖，属于保守处理。
* `calibration_scores` 逐行保留点值、基线端点、截顶标签、标签身份和不符合
  分数；`calibration` 保留每个批组的得分与秩、阈值是否被选中。

## 交换性假设与承诺边界

输出中的 `exchangeability` 明确写明：

* 校准批组与新预测批组可交换，批组名单在预测时点已冻结；
* 同批组的后续发运不回溯进入旧名单；
* 承诺对象是截顶时长，不是真实长尾时长；
* 并列处理规则。

只有在批组可交换、无漂移、且校准秩足够时，才存在上述有限样本覆盖解释。

## 市场

逐市场单独校准并报告 `markets`。某市场校准批组不足以支撑其自身的秩时：

* 该市场 `status = insufficient_market_samples`；
* 该市场行仍可使用总体池阈值，但 `guarantee_scope` 明确标为
  `pooled_group_guarantee_no_market_guarantee`；
* 文案明确说明“总体池覆盖不自动等于本市场覆盖”。

市场足够时，该市场行使用自己的阈值并标为 `market_group_guarantee`。批组
跨市场会直接拒绝，因为无法给出市场级报告。

## 结算、缓存与不可回写

* 预测身份 = 稳定摘要（配置版本、模型版本、预测时点可见资料的摘要、`as_of`、
  `horizon`、`reporting_lag`、配置、批组名单摘要）。
* `FrozenStore` 只追加：相同身份同内容幂等返回，同身份不同内容直接报错，
  拒绝回写。
* 结算记录以“预测身份 + 结算时点 + 结算输入摘要”为身份追加，不修改预测
  记录。结算逐行保留 `shipment_revision_at_forecast`、
  `shipment_revision_at_settlement` 和 `feature_revised_after_forecast`。
* 结算成熟判定只看结算时点的日历规则；只要同批仍有行未过观察期，整批保持
  `pending`，即使该行已有完成标签。
* 逐批预测、校准资格、校准得分、区间命中与 pending 原因都保留，不只给全期
  平均误差。

## 覆盖回放与漂移

`replay` 对每个时点的资料独立冻结预测，再用**最终资料**结算：

* 每个窗口都有独立身份；后到资料不会改写旧窗口。
* 覆盖率只统计整批已成熟的样本外批组（`holdout`、`pending`、`unassigned`），
  在途批组列在 `pending_batches`。
* 每个窗口输出 `settled_coverage`（行命中率、批组联合命中率）与
  `settled_drift`（前后半段 MAE 比、观察命中率与名义下界）。
* 出现漂移迹象或观察命中率低于名义下界时，`distribution_free_claim` 变为
  `withdrawn_under_drift`；其余情况最多为 `retrospective_only`。回放覆盖率
  是历史观察值，不构成未来分布无关覆盖承诺。

## 失败状态

| status | 含义 |
| --- | --- |
| `ok` | 总体与各市场都有足够的校准批组 |
| `partial_market_guarantee` | 总体足够，但至少一个市场不足 |
| `pooled_only_no_market_guarantee` | 总体足够，但没有任何市场单独足够 |
| `insufficient_calibration_groups` | 总体秩不足，不提供区间 |
| `model_fit_failed` | 无成熟训练批组 / 特征不可用 / 拟合失败，记录 `failure` |

其余可逐行检查的状态：`feature_unavailable_at_forecast_time`、
`no_group_guarantee`、`interval`，以及 pending 原因
`not_all_rows_matured_by_settlement_as_of`、`no_group_guarantee`、
`roster_not_in_settlement_frame`。

## 配置

缺省配置在 `app.forecast.DEFAULT_CONFIG`，可用 JSON 覆盖：

* `alpha`（缺省 0.1）
* `split_fractions`（缺省 `[0.5, 0.65, 0.85]`）
* `group_split`（批组到角色的清单，角色为 train/validation/calibration/holdout/pending）
* `lambdas`（岭正则候选）
* `interval_strategy`（`auto` / `residual` / `quantile`）
* `drift_mae_ratio`、`min_drift_groups`、`replay_drop_tolerance`（缺省 0.05）

未知键、非正则、非法角色、非递增比例等一律拒绝。

## 运行

    python -m app.service fixtures/sample.json
    python -m app.forecast_cli forecast fixtures/synthetic.json
    python -m unittest discover -s tests -v
    python -m compileall -q app tests
