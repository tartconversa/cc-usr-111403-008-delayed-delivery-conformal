"""时点、资格与截顶标签的共同定义。

所有函数只做时序与字段校验，不读取完成标签来决定校准资格。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


def digest(value):
    """对 JSON 安全结构做稳定摘要。"""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False
    )
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def _is_int(value):
    return type(value) is int


@dataclass(frozen=True)
class Timing:
    as_of: int
    horizon: int
    reporting_lag: int

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            raise ValueError('metadata应为对象')
        for key in ('as_of', 'horizon', 'reporting_lag'):
            if key not in metadata:
                raise ValueError(f'metadata缺少{key}')
        as_of, horizon, lag = (
            metadata['as_of'], metadata['horizon'], metadata['reporting_lag']
        )
        if not _is_int(as_of):
            raise ValueError('as_of应为整数')
        if not _is_int(horizon) or horizon <= 0:
            raise ValueError('horizon应为正整数')
        if not _is_int(lag) or lag < 0:
            raise ValueError('reporting_lag应为非负整数')
        return cls(as_of, horizon, lag)

    def mature_at(self, shipped):
        return shipped + self.horizon + self.reporting_lag

    def is_mature(self, shipped, cutoff):
        return cutoff >= self.mature_at(shipped)

    def target(self, duration):
        """业务目标：截顶时长。"""
        return min(duration, self.horizon)


def normalize_shipment(row):
    """校验并提取发运行的时点与身份信息。

    ``recorded_at`` 可选：表示当前修订到达资料系统的时间。缺省取
    ``feature_known``，即特征自那时已知。晚于预测截止到达的修订不得用于
    旧预测，也不得改变旧预测的可见身份。
    """
    if not isinstance(row, dict):
        raise ValueError('发运行应为对象')
    group = row.get('batch_group')
    market = row.get('market')
    shipped = row.get('shipped')
    known = row.get('feature_known')
    features = row.get('features')
    recorded = row.get('recorded_at', known)
    if not isinstance(group, str) or not group:
        raise ValueError('batch_group应为非空字符串')
    if not isinstance(market, str) or not market:
        raise ValueError('market应为非空字符串')
    if not _is_int(shipped):
        raise ValueError('shipped应为整数')
    if not _is_int(known):
        raise ValueError('feature_known应为整数')
    if not _is_int(recorded):
        raise ValueError('recorded_at应为整数')
    if recorded < known:
        raise ValueError('recorded_at不得早于feature_known')
    if not isinstance(features, dict):
        raise ValueError('features应为对象')
    return {
        'id': row['id'],
        'revision': row['revision'],
        'batch_group': group,
        'market': market,
        'shipped': shipped,
        'feature_known': known,
        'recorded_at': recorded,
        'features': features,
    }


def normalize_completions(rows, shipment_ids, lag):
    """按现有摘要的语义校验完成记录，返回按发运索引的映射。"""
    completions = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('完成记录应为对象')
        shipment = row.get('shipment')
        completed = row.get('completed')
        known = row.get('known')
        if shipment not in shipment_ids:
            raise ValueError('完成身份未知或重复')
        if shipment in completions:
            raise ValueError('完成身份未知或重复')
        if not _is_int(completed) or not _is_int(known):
            raise ValueError('完成或已知时间应为整数')
        shipped = shipment_ids[shipment]['shipped']
        if not shipped <= completed <= known <= completed + lag:
            raise ValueError('完成时间或晚报上限无效')
        completions[shipment] = {
            'id': row['id'],
            'revision': row['revision'],
            'shipment': shipment,
            'completed': completed,
            'known': known,
        }
    return completions


def completion_identity(completion):
    return {
        'completion_id': completion['id'],
        'completion_revision': completion['revision'],
        'completed': completion['completed'],
        'known': completion['known'],
    }


def label_at(shipment, completion, cutoff, timing):
    """返回截止时点可知的截顶标签。

    ``label`` 为 ``None`` 表示未成熟；这不是把未完成当成 0。
    """
    shipped = shipment['shipped']
    if completion is not None and completion['known'] <= cutoff:
        duration = completion['completed'] - shipped
        target = timing.target(duration)
        kind = 'observed' if duration <= timing.horizon else 'observed_capped'
        identity = {
            'kind': kind,
            'target': target,
            'horizon': timing.horizon,
            'duration': duration,
            **completion_identity(completion),
        }
        return {
            'label': target,
            'kind': kind,
            'duration': duration,
            'completion': completion,
            'label_identity': {**identity, 'label_id': digest(identity)},
        }
    if timing.is_mature(shipped, cutoff):
        identity = {
            'kind': 'cap_censored',
            'target': timing.horizon,
            'horizon': timing.horizon,
            'mature_at': timing.mature_at(shipped),
        }
        return {
            'label': timing.horizon,
            'kind': 'cap_censored',
            'duration': None,
            'completion': None,
            'label_identity': {**identity, 'label_id': digest(identity)},
        }
    return {
        'label': None,
        'kind': 'immature',
        'duration': None,
        'completion': None,
        'label_identity': None,
    }


def shipment_identity(shipment):
    """发运行的可见身份。

    调用方只传入在预测时点已经可见的修订，因此这里不再重复判到达时间；
    可见性由预测模块的可见修订视图统一保证。
    """
    return {
        'shipment': shipment['id'],
        'revision': shipment['revision'],
        'batch_group': shipment['batch_group'],
        'market': shipment['market'],
        'shipped': shipment['shipped'],
        'feature_known': shipment['feature_known'],
        'recorded_at': shipment['recorded_at'],
        'features_digest': digest(shipment['features']),
    }


def roster_digest(shipments):
    ordered = sorted(shipments, key=lambda row: row['id'])
    return digest([shipment_identity(row) for row in ordered])
