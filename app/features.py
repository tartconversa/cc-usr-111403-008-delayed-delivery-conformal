"""预测时点可用的特征编码。

只使用 ``feature_known <= cutoff`` 的发运记录。缺失的特征键用训练期中位数
填补并保留显式缺失标记；未知特征键被忽略并记录，不会把缺失、未完成或未来值
当作 0。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _number(value, label):
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f'特征{label}应为有限数字')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'特征{label}应为有限数字')
    return value


@dataclass(frozen=True)
class Encoded:
    """一行标准化后的特征。"""

    values: tuple[float, ...]
    missing: tuple[float, ...]
    ignored: tuple[str, ...]

    def vector(self):
        return self.values + self.missing


@dataclass(frozen=True)
class FeatureSpace:
    """由训练批组拟合的固定特征空间。"""

    names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    @classmethod
    def fit(cls, rows):
        rows = list(rows)
        if not rows:
            raise ValueError('拟合特征空间需要至少一行训练记录')
        names = sorted({key for row in rows for key in row.get('features', {})})
        medians, means, stds = [], [], []
        for name in names:
            present = [
                _number(row['features'][name], name)
                for row in rows
                if name in row.get('features', {})
            ]
            present.sort()
            median = quantile(present, 0.5) if present else 0.0
            imputed = [
                _number(row['features'][name], name)
                if name in row.get('features', {})
                else median
                for row in rows
            ]
            mean = sum(imputed) / len(imputed)
            variance = sum((value - mean) ** 2 for value in imputed) / len(imputed)
            std = math.sqrt(variance) if variance > 0 else 1.0
            medians.append(median)
            means.append(mean)
            stds.append(std)
        return cls(tuple(names), tuple(medians), tuple(means), tuple(stds))

    def encode(self, row, cutoff):
        """返回 :class:`Encoded`；该修订在截止时点不可见或特征未知则返回 ``None``。

        ``recorded_at`` 缺省等于 ``feature_known``。修订到达晚于截止时点
        时返回 ``None``：旧预测不能使用之后才出现的特征值，也不能用后来
        的修订回写旧预测。
        """
        if type(row.get('feature_known')) is not int:
            raise ValueError('feature_known应为整数')
        recorded = row.get('recorded_at', row['feature_known'])
        if type(recorded) is not int:
            raise ValueError('recorded_at应为整数')
        if row['feature_known'] > cutoff or recorded > cutoff:
            return None
        features = row.get('features')
        if not isinstance(features, dict):
            raise ValueError('features应为对象')
        ignored = tuple(sorted(set(features) - set(self.names)))
        values, missing = [], []
        for name, median, mean, std in zip(
            self.names, self.medians, self.means, self.stds
        ):
            if name in features:
                value = _number(features[name], name)
                missing.append(0.0)
            else:
                value = median
                missing.append(1.0)
            values.append((value - mean) / std)
        return Encoded(tuple(values), tuple(missing), ignored)

    def describe(self):
        return {
            'names': list(self.names),
            'medians': list(self.medians),
            'means': list(self.means),
            'stds': list(self.stds),
            'missing_policy': '训练期中位数填补并附加缺失标记；未完成不填补为0',
            'unknown_key_policy': '忽略训练中未出现的特征键并逐行记录',
        }


def _quantile(ordered, q):
    """线性插值分位数；``ordered`` 必须已排序。"""
    if not ordered:
        raise ValueError('空序列没有分位数')
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def quantile(ordered, q):
    if not 0.0 <= q <= 1.0:
        raise ValueError('分位点应在[0,1]')
    return _quantile(sorted(ordered), q)
