# 资料与计算约定

## 修订快照

schema_version为整数1，metadata为业务方案字典，collections为业务集合字典。每条记录有非空字符串id和正整数revision。
同集合相同id取最高revision；同id同revision内容不同即冲突，即便存在更高修订也拒绝。deleted=true为撤回，之后更高修订可恢复。
撤回记录不要求携带业务字段。输入顺序与相同记录重试不改变语义；非有限数字拒绝。
Snapshot.rows返回独立副本；input_digest绑定有效修订和全部metadata。输出data_version沿用metadata，不由程序自造资料版本。
既有摘要入口为app.service.inspect(payload)，输出schema_version、input_digest、data_version和summary。

## 业务资料

shipments含batch_group、market、shipped、features、feature_known；completions含shipment、completed、known。时间用整数天。horizon=metadata.horizon，目标min(completed-shipped,horizon)；每条完成记录必须满足shipped<=completed<=known<=completed+reporting_lag，reporting_lag为已知非负晚报上限，违反即不适用而非作完整覆盖承诺。仅as_of>=shipped+horizon+reporting_lag的队列完整成熟；成熟且尚无已知完成时目标为horizon。已完成快单可提前显示标签，但不得提前进入校准。批组内可相关，覆盖的有限样本单位为交换批组；组内最大不符合分数可给整批同时覆盖，或声明其他可证明的组级方案。预测截止只取feature_known<=cutoff；baseline.analyze按as_of返回当时可知截顶标签及成熟资格。

各集合必要字段：{"shipments": ["batch_group", "market", "shipped", "features", "feature_known"], "completions": ["shipment", "completed", "known"]}。

### 可选的到达时间

目录只保留每个id的最高修订，不保留旧修订内容。为了让“预测之后才到达的修订”不能回写旧预测，新增分析可接受可选整数字段：

* shipments：`recorded_at`，表示当前修订到达资料系统的时间，缺省等于`feature_known`；不得早于`feature_known`。
* completions：`recorded_at`，缺省等于`known`；表示该完成记录到达时间。
* 撤回记录没有业务字段，`recorded_at`必须显式给出，否则新增分析拒绝该资料。

这些字段不改变既有`catalog.load`、`baseline.analyze`与`service.inspect`的语义；缺省时旧资料行为不变，但只有记录的`known`/`feature_known`到达时间可作为可见性依据。若某修订在预测时点之后到达却缺少`recorded_at`，新增分析只能按`feature_known`/`known`视为当时可见——企业若需要严格不可回写保证，应在资料中提供`recorded_at`。

## 当前能力

app.baseline.analyze(snapshot)产生已有摘要，app.service.inspect负责绑定输入身份；新增分析可以增设独立入口，已有接口及上述数据语义保持兼容。

按预测时点冻结的交付时长区间、批组共形校准、结算与覆盖回放的语义见 [区间与回放说明](forecast.md)。
