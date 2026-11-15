"""冻结预测与追加式结算记录。

预测身份绑定预测时点可见资料、配置、模型版本和批组名单。存储只追加：
之后到达的完成、修订和结算以新记录表达，不修改或回写既有预测。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .timing import digest


def prediction_identity(payload):
    return digest(payload)


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


@dataclass
class FrozenStore:
    """内存中的只追加存储，按预测身份索引。"""

    forecasts: dict = field(default_factory=dict)
    records: list = field(default_factory=list)

    def get(self, identity):
        frozen = self.forecasts.get(identity)
        return None if frozen is None else _copy(frozen)

    def freeze(self, identity, record):
        existing = self.forecasts.get(identity)
        if existing is not None:
            if existing != record:
                raise ValueError('同一预测身份出现不同内容，拒绝回写')
            return _copy(existing)
        frozen = _copy(record)
        self.forecasts[identity] = frozen
        self.records.append({'kind': 'frozen_forecast', 'identity': identity, 'record': frozen})
        return _copy(frozen)

    def append_outcome(self, identity, outcome):
        """追加后续资料结算；不修改已冻结预测本身。"""
        if identity not in self.forecasts:
            raise KeyError('预测身份不存在')
        entry = _copy(outcome)
        entry_identity = digest(
            {
                'prediction_identity': identity,
                'settled_at_as_of': entry.get('settled_at_as_of'),
                'settlement_input_digest': entry.get('settlement_input_digest'),
            }
        )
        for existing in self.records:
            if existing.get('kind') == 'frozen_outcome' and existing.get('identity') == entry_identity:
                if existing['record'] != entry:
                    raise ValueError('同一结算身份出现不同内容')
                return _copy(existing['record'])
        record = {'kind': 'frozen_outcome', 'identity': entry_identity, 'record': entry}
        self.records.append(record)
        return _copy(entry)

    def outcomes(self, identity):
        return [
            _copy(record['record'])
            for record in self.records
            if record.get('kind') == 'frozen_outcome'
            and record['record'].get('prediction_identity') == identity
        ]

    def export(self):
        return _copy(self.records)
