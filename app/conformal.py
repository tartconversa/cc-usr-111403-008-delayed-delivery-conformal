"""按发运批组为交换单位的有限样本共形校准。

批组得分取组内逐发运行不符合分数的最大值，因此承诺的是同一新批组内
全部（冻结名单）发运的联合覆盖，而不是逐行的边际覆盖。分位秩为
``k = ceil((n + 1)(1 - alpha))``，取第 k 个最小批组得分；若 k > n 则
无法给出有限样本保证。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .model import ModelFailure

EXCHANGEABILITY = {
    'unit': 'batch_group',
    'assumption': (
        '校准批组与新预测批组可交换：各批组得分同分布，且批组名单在预测时点已冻结；'
        '同一批组的多行只作为一个交换单位。'
    ),
    'roster': '校准名单只含窗口截止时点已发运的发运行，之后加入同批组的发运不回溯进入该名单。',
    'scope': '承诺对象为截顶时长 min(completed-shipped, horizon)，不是真实长尾交付时长。',
    'capping': '超过观察上限按项目定义截顶；上限以上不区分。',
    'tie_policy': '取第 k 个最小得分；并列时仍使用该阶统计量，可能偏保守，不降低承诺。',
}


def required_rank(n, alpha):
    """有限样本分位秩；返回 ``(k, sufficient)``。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha应在(0,1)')
    if n < 1:
        return 0, False
    exact = Fraction(n + 1) * (1 - Fraction(str(alpha)))
    k = -((-exact.numerator) // exact.denominator)
    return k, k <= n


def nonconformity(prediction, label):
    """基线区间外的不符合分数；落在区间内为 0。"""
    return max(
        prediction['lower'] - label, label - prediction['upper'], 0.0
    )


@dataclass(frozen=True)
class Calibration:
    alpha: float
    n_groups: int
    rank: int
    rank_sufficient: bool
    threshold: float | None
    groups: tuple[dict, ...]
    status: str

    def margin(self):
        return self.threshold if self.rank_sufficient else None

    def describe(self):
        return {
            'method': 'group_max_split_conformal',
            'alpha': self.alpha,
            'n_groups': self.n_groups,
            'rank': self.rank,
            'rank_sufficient': self.rank_sufficient,
            'threshold': self.threshold,
            'status': self.status,
            'guarantee': (
                f'在批组可交换下，新批组冻结名单的联合覆盖至少 {self.rank}/{self.n_groups + 1}'
                if self.rank_sufficient
                else '校准批组不足，无法提供有限样本组级覆盖保证'
            ),
            'rank_definition': 'k = ceil((n+1)(1-alpha))，阈值取第 k 个最小批组得分',
            'exchangeability': dict(EXCHANGEABILITY),
        }


def calibrate(alpha, groups):
    """``groups`` 为已通过完整成熟检查的批组得分记录。"""
    ordered = sorted(groups, key=lambda item: (item['score'], item['batch_group']))
    n = len(ordered)
    k, sufficient = required_rank(n, alpha)
    threshold = None
    if sufficient:
        threshold = ordered[k - 1]['score']
    ranked = []
    for position, item in enumerate(ordered, start=1):
        ranked.append(
            {
                'batch_group': item['batch_group'],
                'market': item['market'],
                'score': item['score'],
                'rank': position,
                'selected_as_threshold': sufficient and position == k,
            }
        )
    status = 'ok' if sufficient else 'insufficient_calibration_groups'
    return Calibration(
        alpha=alpha,
        n_groups=n,
        rank=k,
        rank_sufficient=sufficient,
        threshold=threshold,
        groups=tuple(ranked),
        status=status,
    )


def score_group(group, predictions, labels):
    """``predictions``/``labels`` 与 ``group['rows']`` 一一对应。"""
    if not group['rows']:
        raise ValueError('空批组不能成为校准单位')
    scores = [
        nonconformity(prediction, label)
        for prediction, label in zip(predictions, labels)
    ]
    if any(not math.isfinite(value) for value in scores):
        raise ModelFailure('不符合分数非有限')
    return max(scores)
