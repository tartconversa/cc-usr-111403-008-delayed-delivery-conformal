# 外贸交付时长的延迟标签共形区间与覆盖回放

本项目在既有资料修订读取与业务摘要之上，提供按预测发生时点冻结的交付时长
点预测、按发运批组为交换单位的有限样本共形区间，以及历史覆盖回放。

既有能力保持不变：

* app/catalog.py 按集合和记录键解析修订，拒绝同修订冲突；
* app/baseline.py 承载当前业务计算（截顶标签与成熟资格）；
* app/service.py 输出摘要及输入摘要哈希。

新增能力：

* app/forecast.py — 时点冻结预测、批组隔离、组级共形校准、结算与回放；
* app/model.py — 点预测与分位区间基线，模型选择只看验证批组；
* app/conformal.py — 批组最大不符合分数的有限样本秩与阈值；
* app/features.py — 只用预测时点已知特征，缺失显式中位数填补；
* app/timing.py — 截顶标签、成熟规则、可见修订与身份摘要；
* app/store.py — 只追加的冻结预测/结算存储；
* app/forecast_cli.py — 命令行入口。

数据字段见 [资料契约](docs/data-contract.md)，预测与区间语义、交换性假设、
有限样本秩、失败状态与回放口径见 [区间与回放说明](docs/forecast.md)。
fixtures/sample.json 是既有接口的合成小数据；fixtures/synthetic.json 是用于
端到端验证的更完整合成数据，均不对应真实企业或个人。

运行环境为 Python 3.14（本仓库测试在 3.11 亦可运行），仅使用标准库。

读取并计算当前摘要：

    python -m app.service fixtures/sample.json

冻结一次预测：

    python -m app.forecast_cli forecast fixtures/synthetic.json

历史覆盖回放（按时点严格递增）：

    python -m app.forecast_cli replay fixtures/replay/w1.json \
        fixtures/replay/w2.json fixtures/replay/w3.json

运行检查：

    python -m unittest discover -s tests -v

检查源码编译：

    python -m compileall -q app tests
