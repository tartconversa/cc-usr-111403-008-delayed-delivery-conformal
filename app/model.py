"""点预测与分位区间基线。

模型只在训练批组上拟合，在验证批组上选择；校准批组不参与拟合或选择。
特征和时间均已标准化，目标按训练期均值/标准差标准化。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .features import FeatureSpace, quantile

MODEL_VERSION = 'ridge-quantile-v1'
DEFAULT_LAMBDAS = (0.01, 0.1, 1.0, 10.0)


class ModelFailure(RuntimeError):
    """模型无法给出可用基线，需作为失败状态上报。"""


def _solve(matrix, rhs):
    """带部分主元的高斯消元；矩阵为对称正定（岭正则）。"""
    size = len(matrix)
    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ModelFailure('设计矩阵奇异')
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                for index in range(column, size + 1):
                    augmented[row][index] -= factor * augmented[column][index]
    return [augmented[i][size] for i in range(size)]


def _fit_linear(xs, ys, penalty, weights=None):
    """加权岭回归，含不惩罚截距。返回系数（长度 d+1）。"""
    if not xs:
        raise ModelFailure('训练集为空')
    dimension = len(xs[0])
    weights = weights or [1.0] * len(xs)
    normal = [[0.0] * (dimension + 1) for _ in range(dimension + 1)]
    rhs = [0.0] * (dimension + 1)
    for x, y, weight in zip(xs, ys, weights):
        row = tuple(x) + (1.0,)
        for i in range(dimension + 1):
            rhs[i] += weight * row[i] * y
            for j in range(dimension + 1):
                normal[i][j] += weight * row[i] * row[j]
    for i in range(dimension):
        normal[i][i] += penalty
    return _solve(normal, rhs)


def _predict(coef, x):
    return sum(c * value for c, value in zip(coef, x)) + coef[-1]


def _pinball(residual, q):
    return q * residual if residual >= 0 else (q - 1.0) * residual


def _fit_quantile(xs, ys, q, penalty, rounds=80):
    """IRLS 线性分位回归；初值为同惩罚下的均值回归。"""
    if not 0.0 < q < 1.0:
        raise ValueError('分位点应在(0,1)')
    coef = _fit_linear(xs, ys, penalty)
    for _ in range(rounds):
        residuals = [y - _predict(coef, x) for x, y in zip(xs, ys)]
        weights = []
        for residual in residuals:
            magnitude = max(abs(residual), 1e-6)
            weights.append((q if residual >= 0 else 1.0 - q) / magnitude)
        updated = _fit_linear(xs, ys, penalty, weights)
        if max(abs(a - b) for a, b in zip(coef, updated)) < 1e-9:
            coef = updated
            break
        coef = updated
    if not all(math.isfinite(value) for value in coef):
        raise ModelFailure('分位回归系数非有限')
    return coef


@dataclass(frozen=True)
class Baseline:
    """可冻结的基线模型。"""

    feature_space: FeatureSpace
    point_coef: tuple[float, ...]
    interval_strategy: str
    alpha: float
    lambdas: tuple[float, ...]
    selected_lambda: float
    y_mean: float
    y_std: float
    lower_residual: float | None
    upper_residual: float | None
    lower_coef: tuple[float, ...] | None
    upper_coef: tuple[float, ...] | None
    selection: dict

    def predict(self, encoded):
        x = encoded.vector()
        point_std = _predict(self.point_coef, x)
        point = point_std * self.y_std + self.y_mean
        if self.interval_strategy == 'residual':
            lower = point + self.lower_residual
            upper = point + self.upper_residual
        else:
            lower = _predict(self.lower_coef, x) * self.y_std + self.y_mean
            upper = _predict(self.upper_coef, x) * self.y_std + self.y_mean
        if not all(math.isfinite(value) for value in (point, lower, upper)):
            raise ModelFailure('预测值非有限')
        if lower > upper:
            # 分位交叉：保守地取排序后的端点，并不隐藏该失败。
            lower, upper = upper, lower
        return {'point': point, 'lower': lower, 'upper': upper}

    def describe(self):
        return {
            'model_version': MODEL_VERSION,
            'feature_space': self.feature_space.describe(),
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'selected_lambda': self.selected_lambda,
            'interval_strategy': self.interval_strategy,
            'point_coefficients': list(self.point_coef),
            'lower_residual': self.lower_residual,
            'upper_residual': self.upper_residual,
            'selection': self.selection,
        }


def _prepare(rows, labels, features):
    xs, ys = [], []
    for row, label in zip(rows, labels):
        # 预测发生时点为该发运的发运日；只用当时已知的特征。
        encoded = features.encode(row, cutoff=row['shipped'])
        if encoded is None:
            raise ModelFailure('训练特征在发运时点不可用')
        xs.append(encoded.vector())
        ys.append(float(label))
    return xs, ys


def _standardize_target(ys):
    mean = sum(ys) / len(ys)
    variance = sum((y - mean) ** 2 for y in ys) / len(ys)
    std = math.sqrt(variance) if variance > 1e-12 else 1.0
    return mean, std, [(y - mean) / std for y in ys]


def fit_baseline(train_rows, train_labels, validation_rows, validation_labels, alpha,
                 lambdas=DEFAULT_LAMBDAS, interval_strategy='auto'):
    """拟合点预测，按验证批组选择正则与区间策略。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha应在(0,1)')
    if not train_rows:
        raise ModelFailure('没有可用的训练批组')
    features = FeatureSpace.fit(train_rows)
    xs, ys = _prepare(train_rows, train_labels, features)
    y_mean, y_std, ys_std = _standardize_target(ys)

    validation_xs, validation_ys = [], []
    if validation_rows:
        validation_xs, validation_ys = _prepare(validation_rows, validation_labels, features)

    lambdas = tuple(float(value) for value in lambdas)
    if not lambdas or any(value <= 0 for value in lambdas):
        raise ValueError('正则候选应为正数')

    selection = {'candidates': [], 'criterion': '验证批组平均绝对误差；并列取较大正则'}
    point_scores = {}
    for penalty in lambdas:
        coef = _fit_linear(xs, ys_std, penalty)
        if validation_xs:
            errors = [
                abs((_predict(coef, x) * y_std + y_mean) - y)
                for x, y in zip(validation_xs, validation_ys)
            ]
            score = sum(errors) / len(errors)
        else:
            score = 0.0
        point_scores[penalty] = score
        selection['candidates'].append(
            {'lambda': penalty, 'validation_mae': score if validation_xs else None}
        )
    if validation_xs:
        selected_lambda = min(lambdas, key=lambda value: (point_scores[value], -value))
        selection['kind'] = 'validation'
    else:
        selected_lambda = 1.0 if 1.0 in lambdas else lambdas[0]
        selection['kind'] = 'fallback_no_validation_groups'
        selection['fallback_reason'] = '没有完整成熟的验证批组，固定正则且不据此声称模型选择'
    point_coef = _fit_linear(xs, ys_std, selected_lambda)

    residuals = sorted(
        y - (_predict(point_coef, x) * y_std + y_mean) for x, y in zip(xs, ys)
    )
    residual_lower = quantile(residuals, alpha / 2.0)
    residual_upper = quantile(residuals, 1.0 - alpha / 2.0)

    strategies = ['residual', 'quantile'] if interval_strategy == 'auto' else [interval_strategy]
    if interval_strategy not in ('auto', 'residual', 'quantile'):
        raise ValueError('interval_strategy应为auto、residual或quantile')
    strategy_scores = {}
    strategy_coefs = {}
    for strategy in strategies:
        if strategy == 'residual':
            strategy_scores[strategy] = None
            strategy_coefs[strategy] = (None, None)
            continue
        try:
            low = _fit_quantile(xs, ys_std, alpha / 2.0, selected_lambda)
            high = _fit_quantile(xs, ys_std, 1.0 - alpha / 2.0, selected_lambda)
        except ModelFailure:
            strategy_scores[strategy] = float('inf')
            strategy_coefs[strategy] = (None, None)
            continue
        strategy_coefs[strategy] = (low, high)
        if validation_xs:
            loss = 0.0
            for x, y in zip(validation_xs, validation_ys):
                low_pred = _predict(low, x) * y_std + y_mean
                high_pred = _predict(high, x) * y_std + y_mean
                low_pred, high_pred = min(low_pred, high_pred), max(low_pred, high_pred)
                loss += _pinball(y - low_pred, alpha / 2.0)
                loss += _pinball(y - high_pred, 1.0 - alpha / 2.0)
            strategy_scores[strategy] = loss / len(validation_xs)
        else:
            strategy_scores[strategy] = 0.0

    if interval_strategy == 'auto':
        # residual 策略没有可比的验证分位损失时，用同一验证口径重算。
        if validation_xs:
            loss = 0.0
            for x, y in zip(validation_xs, validation_ys):
                point = _predict(point_coef, x) * y_std + y_mean
                low_pred, high_pred = point + residual_lower, point + residual_upper
                loss += _pinball(y - low_pred, alpha / 2.0)
                loss += _pinball(y - high_pred, 1.0 - alpha / 2.0)
            strategy_scores['residual'] = loss / len(validation_xs)
        else:
            strategy_scores['residual'] = 0.0
        selected_strategy = min(
            strategies, key=lambda value: (strategy_scores[value], value != 'residual')
        )
        selection['interval_kind'] = 'validation' if validation_xs else 'fallback_no_validation_groups'
    else:
        selected_strategy = interval_strategy
        selection['interval_kind'] = 'fixed_by_config'
    selection['interval_candidates'] = [
        {'strategy': strategy, 'validation_pinball': strategy_scores[strategy]}
        for strategy in strategies
    ]

    lower_residual = upper_residual = None
    lower_coef = upper_coef = None
    if selected_strategy == 'quantile':
        lower_coef, upper_coef = strategy_coefs['quantile']
        if lower_coef is None:
            raise ModelFailure('线性分位回归不可用')
    else:
        lower_residual, upper_residual = residual_lower, residual_upper

    return Baseline(
        feature_space=features,
        point_coef=tuple(point_coef),
        interval_strategy=selected_strategy,
        alpha=alpha,
        lambdas=lambdas,
        selected_lambda=selected_lambda,
        y_mean=y_mean,
        y_std=y_std,
        lower_residual=None if lower_residual is None else float(lower_residual),
        upper_residual=None if upper_residual is None else float(upper_residual),
        lower_coef=None if lower_coef is None else tuple(lower_coef),
        upper_coef=None if upper_coef is None else tuple(upper_coef),
        selection=selection,
    )
