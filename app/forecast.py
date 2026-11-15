"""按预测时点冻结的交付时长点预测、批组共形区间与覆盖回放。

设计要点（均可从输出核查）：

* 预测发生时点为发运日 ``shipped``；特征只在 ``feature_known <= shipped``
  时可用。缺失特征用训练期中位数加缺失标记；未完成决不是 0。
* 目标是项目截顶时长 ``min(completed - shipped, horizon)``。超过观察上限只
  区分到上限；覆盖承诺不解释为真实长尾时长覆盖。
* 校准资格由日历成熟度 ``shipped + horizon + reporting_lag <= as_of`` 决定，
  不按“是否已收到完成标签”挑行。队列在预测发布时冻结。
* 训练、验证、校准、留出均以发运批组为整体单位，组内不拆行。
* 交换单位是批组；组得分取组内逐行不符合分数的最大值，承诺同一批组冻结
  名单的联合覆盖。有限样本秩 ``k = ceil((n+1)(1-alpha))``，``k > n`` 时
  明确报告无法提供保证。
* 市场覆盖单独报告。某市场校准批组不足时，只能声明该市场无法提供组别
  保证；总体池覆盖不自动等于各市场覆盖。
* 预测身份绑定预测时点可见资料、配置、模型版本和批组名单；记录只追加。
  之后到达的完成/修订只作为新增结果记录，不回写旧预测。
* 样本外评估覆盖留出与在途批组；在途批组在预测时点保留预测、在结算时点
  按日历成熟度进入评估，不因尚未收到完成标签而被漏掉。
"""
from __future__ import annotations

import math

from .catalog import load
from .conformal import EXCHANGEABILITY, calibrate, nonconformity, score_group
from .model import MODEL_VERSION, ModelFailure, fit_baseline
from .service import SCHEMAS
from .store import FrozenStore, prediction_identity
from .timing import (
    Timing,
    digest,
    label_at,
    normalize_completions,
    normalize_shipment,
    roster_digest,
    shipment_identity,
)

CONFIG_VERSION = 1
SPLIT_ROLES = ('train', 'validation', 'calibration', 'holdout', 'pending', 'unassigned')
FIT_ROLES = ('train', 'validation', 'calibration')
OUT_OF_SAMPLE_ROLES = ('holdout', 'pending', 'unassigned')
DEFAULT_CONFIG = {
    'alpha': 0.1,
    'split_fractions': [0.5, 0.65, 0.85],
    'group_split': None,
    'lambdas': [0.01, 0.1, 1.0, 10.0],
    'interval_strategy': 'auto',
    'drift_mae_ratio': 1.5,
    'min_drift_groups': 3,
    'replay_drop_tolerance': 0.05,
}


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

def _config(config):
    merged = dict(DEFAULT_CONFIG)
    if config:
        if not isinstance(config, dict):
            raise ValueError('config应为对象')
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError('未知配置键: ' + ','.join(sorted(unknown)))
        merged.update(config)
    return merged


def _validate_config(config):
    alpha = config['alpha']
    if type(alpha) is bool or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError('alpha应在(0,1)')
    fractions = config['split_fractions']
    if not isinstance(fractions, (list, tuple)) or len(fractions) != 3:
        raise ValueError('split_fractions应为3个累计边界')
    if any(
        type(value) is bool or not isinstance(value, (int, float)) or not 0 < value < 1
        for value in fractions
    ):
        raise ValueError('split_fractions应在(0,1)')
    if not fractions[0] < fractions[1] < fractions[2]:
        raise ValueError('split_fractions应严格递增')
    lambdas = config['lambdas']
    if not isinstance(lambdas, (list, tuple)) or not lambdas:
        raise ValueError('lambdas应为非空数组')
    if any(
        type(value) is bool or not isinstance(value, (int, float)) or value <= 0
        for value in lambdas
    ):
        raise ValueError('lambdas应为正数')
    if config['interval_strategy'] not in ('auto', 'residual', 'quantile'):
        raise ValueError('interval_strategy应为auto、residual或quantile')
    drift = config['drift_mae_ratio']
    if type(drift) is bool or not isinstance(drift, (int, float)) or drift <= 0:
        raise ValueError('drift_mae_ratio应为正数')
    if type(config['min_drift_groups']) is not int or config['min_drift_groups'] < 1:
        raise ValueError('min_drift_groups应为正整数')
    tolerance = config['replay_drop_tolerance']
    if type(tolerance) is bool or not isinstance(tolerance, (int, float)):
        raise ValueError('replay_drop_tolerance应为数字')
    if not 0 <= tolerance < 1:
        raise ValueError('replay_drop_tolerance应在[0,1)')
    manifest = config['group_split']
    if manifest is not None:
        if not isinstance(manifest, dict):
            raise ValueError('group_split应为批组到角色的映射')
        for name, role in manifest.items():
            if not isinstance(name, str) or not name:
                raise ValueError('group_split的批组应为非空字符串')
            if role not in SPLIT_ROLES:
                raise ValueError('group_split角色应为: ' + ','.join(SPLIT_ROLES))


# --------------------------------------------------------------------------
# 资料准备
# --------------------------------------------------------------------------

def _preview_record(row, scene):
    """读取记录的到达时间；发运用recorded_at（缺省feature_known），完成用known。

    撤回记录没有业务字段，必须显式给出 ``recorded_at``，否则无法判断它
    在预测时点是否已经发生，直接拒绝而不是猜测。
    """
    if not isinstance(row, dict):
        raise ValueError('记录应为对象')
    if scene == 'shipments':
        value = row.get('recorded_at', row.get('feature_known'))
    else:
        value = row.get('recorded_at', row.get('known'))
    if type(value) is not int:
        raise ValueError('撤回或修订记录缺少整数recorded_at')
    return value


def _visible_rows(raw_rows, cutoff, scene):
    """按截止时点重建可见修订视图，再按 id 取最高可见修订。

    ``catalog.load`` 只保留当前最高修订；若更高修订在预测之后到达，直接
    使用它会把未来修订回写旧预测。这里先按到达时间遮蔽后到修订，再在
    可见集合内选最高修订。撤回同样受到达时间约束。
    """
    visible = []
    for row in raw_rows:
        arrived = _preview_record(row, scene)
        if arrived <= cutoff:
            visible.append(row)
    winners = {}
    for row in visible:
        identity = row['id']
        current = winners.get(identity)
        if current is None or current['revision'] < row['revision']:
            winners[identity] = row
    return sorted(
        (row for row in winners.values() if not row.get('deleted', False)),
        key=lambda row: row['id'],
    )


def _prepare_frame(payload, timing):
    """从原始集合重建预测时点可见视图。

    先用 ``load`` 校验修订冲突与撤回语义（保留既有读取契约），再用到达
    时间过滤出预测时点可见的修订。完成记录只暴露 ``known <= as_of`` 的
    版本；此后到达的标签只用于结算。
    """
    raw = payload.get('collections')
    if not isinstance(raw, dict):
        raise ValueError('业务集合不完整或包含未知集合')
    shipments_raw = _visible_rows(raw.get('shipments', []), timing.as_of, 'shipments')
    completions_raw = _visible_rows(
        raw.get('completions', []), timing.as_of, 'completions'
    )
    shipments = [normalize_shipment(row) for row in shipments_raw]
    all_ids = {row['id']: row for row in shipments}
    known_shipment_ids = {
        row['id'] for row in raw.get('shipments', []) if isinstance(row, dict)
    }
    # 发运名单在预测时点尚未可见（如当前修订晚于as_of到达）时，其完成记录
    # 也不可用；这是时序导致的遮蔽，不是资料错误。真正未知的身份仍然拒绝。
    visible_completions = []
    for row in completions_raw:
        shipment = row.get('shipment')
        if shipment in all_ids:
            visible_completions.append(row)
        elif shipment not in known_shipment_ids:
            raise ValueError('完成身份未知或重复')
    completions = normalize_completions(
        visible_completions, all_ids, timing.reporting_lag
    )
    roster = [row for row in shipments if row['shipped'] <= timing.as_of]
    return roster, completions


def _group_frame(shipments):
    groups = {}
    for row in shipments:
        groups.setdefault(row['batch_group'], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row['shipped'], row['id']))
    return groups


def _market_of(rows):
    markets = {row['market'] for row in rows}
    if len(markets) != 1:
        raise ValueError('批组跨市场，无法按市场报告；请先修正业务分组')
    return markets.pop()


def _is_mature(rows, timing, cutoff=None):
    cutoff = timing.as_of if cutoff is None else cutoff
    return all(timing.is_mature(row['shipped'], cutoff) for row in rows)


def _labels(rows, completions, cutoff, timing):
    return [label_at(row, completions.get(row['id']), cutoff, timing) for row in rows]


# --------------------------------------------------------------------------
# 批组分派
# --------------------------------------------------------------------------

def _derive_roles(mature_names, config):
    """已成熟批组按首个发运日排序，按事前比例切分。

    比例只依赖批组名单与发运日，不依赖完成标签或覆盖结果。批组过少时
    某些角色为空，由失败状态显式报告，不静默改规则。
    """
    count = len(mature_names)
    boundaries = [
        int(math.floor(count * fraction)) for fraction in config['split_fractions']
    ]
    roles = {}
    for position, name in enumerate(mature_names):
        if position < boundaries[0]:
            roles[name] = 'train'
        elif position < boundaries[1]:
            roles[name] = 'validation'
        elif position < boundaries[2]:
            roles[name] = 'calibration'
        else:
            roles[name] = 'holdout'
    return roles


def _assign_roles(groups, timing, config):
    """返回批组角色和清单降级记录。

    清单是事前冻结的队列。若某批组在 ``as_of`` 尚未整批成熟，不按它是否
    已收到完成标签挑选，而是整批降为 pending 并记录原因；降级只看日历。
    """
    cutoff = timing.as_of
    mature = sorted(
        (name for name, rows in groups.items() if _is_mature(rows, timing, cutoff)),
        key=lambda name: (groups[name][0]['shipped'], name),
    )
    roles = _derive_roles(mature, config)
    demotions = []
    manifest = config['group_split']
    if manifest is not None:
        unknown = sorted(set(manifest) - set(groups))
        if unknown:
            raise ValueError('group_split包含未知批组: ' + ','.join(unknown))
        roles = {}
        for name in groups:
            if name in manifest:
                roles[name] = manifest[name]
            elif name in mature:
                roles[name] = 'unassigned'
            else:
                roles[name] = 'pending'
        for name, role in sorted(manifest.items()):
            if role in FIT_ROLES and not _is_mature(groups[name], timing, cutoff):
                roles[name] = 'pending'
                demotions.append(
                    {
                        'batch_group': name,
                        'requested_role': role,
                        'assigned_role': 'pending',
                        'reason': 'manifest_group_not_matured_at_as_of',
                    }
                )
    for name in groups:
        role = roles.get(name, 'pending')
        if role in FIT_ROLES and not _is_mature(groups[name], timing, cutoff):
            role = 'pending'
        roles[name] = role
    return roles, demotions


# --------------------------------------------------------------------------
# 模型、校准
# --------------------------------------------------------------------------

def _packets(baseline, groups, labels_by_group):
    packets = {}
    for name, rows in groups.items():
        encoded = [
            baseline.feature_space.encode(row, cutoff=row['shipped']) for row in rows
        ]
        packets[name] = {
            'batch_group': name,
            'market': _market_of(rows),
            'rows': rows,
            'encoded': encoded,
            'predictions': [
                None if vector is None else baseline.predict(vector) for vector in encoded
            ],
            'labels': labels_by_group[name],
        }
    return packets


def _score_packet(packet):
    predictions, label_infos = [], []
    for prediction, info in zip(packet['predictions'], packet['labels']):
        if prediction is None:
            raise ModelFailure('校准批组存在预测时点不可用特征，不能作为交换单位')
        if info['label'] is None:
            raise ModelFailure('成熟批组出现空标签，资料成熟规则不一致')
        predictions.append(prediction)
        label_infos.append(info)
    return {
        'batch_group': packet['batch_group'],
        'market': packet['market'],
        'score': score_group(
            packet, predictions, [info['label'] for info in label_infos]
        ),
        'rows': [
            {
                'shipment': row['id'],
                'shipment_revision': row['revision'],
                'point': prediction['point'],
                'baseline_lower': prediction['lower'],
                'baseline_upper': prediction['upper'],
                'label': info['label'],
                'label_kind': info['kind'],
                'label_identity': info['label_identity'],
                'nonconformity': nonconformity(prediction, info['label']),
            }
            for row, prediction, info in zip(packet['rows'], predictions, label_infos)
        ],
    }


def _calibrate_all(alpha, calibration_packets, all_markets):
    scored = [_score_packet(packet) for _, packet in sorted(calibration_packets.items())]
    overall = calibrate(alpha, scored)
    pools = {}
    for record in scored:
        pools.setdefault(record['market'], []).append(record)
    market_calibrations = {
        market: calibrate(alpha, records) for market, records in sorted(pools.items())
    }
    market_status = {}
    for market in all_markets:
        calibration = market_calibrations.get(market)
        if calibration is None:
            market_status[market] = {
                'calibration_groups': 0,
                'rank': 0,
                'rank_sufficient': False,
                'threshold': None,
                'status': 'insufficient_market_samples',
                'note': (
                    '该市场没有校准批组，不能提供市场级组覆盖保证；'
                    '总体池覆盖不自动等于本市场覆盖。'
                ),
            }
            continue
        market_status[market] = {
            'calibration_groups': calibration.n_groups,
            'rank': calibration.rank,
            'rank_sufficient': calibration.rank_sufficient,
            'threshold': calibration.threshold,
            'status': 'group_guarantee'
            if calibration.rank_sufficient
            else 'insufficient_market_samples',
            'note': (
                '该市场有独立的有限样本组覆盖保证；总体池覆盖仍不自动等于逐市场覆盖。'
                if calibration.rank_sufficient
                else '该市场校准批组不足，不能提供市场级组覆盖保证；'
                '总体池覆盖不自动等于本市场覆盖。'
            ),
        }
    return overall, market_calibrations, market_status, scored


def _margin_for(market, overall, market_calibrations):
    """事前声明的取阈值顺序：市场自身保证优先，否则总体池并标明范围。"""
    market_calibration = market_calibrations.get(market)
    if market_calibration is not None and market_calibration.rank_sufficient:
        return market_calibration.threshold, 'market_group_guarantee'
    if overall.rank_sufficient:
        return overall.threshold, 'pooled_group_guarantee_no_market_guarantee'
    return None, 'no_group_guarantee'


def _prediction_batches(packets, overall, market_calibrations, roles, timing):
    """生成逐发运预测。区间端点按目标定义域 [0, horizon] 截断。

    截断不改变对截顶目标的覆盖：目标总在 [0, horizon] 内，所以低于0的
    下界或高于上限的上界本来就不可能漏掉目标。它也不表示对上限以上真实
    长尾时长的判断。
    """
    batches = []
    horizon = timing.horizon
    for name, packet in sorted(packets.items()):
        margin, scope = _margin_for(packet['market'], overall, market_calibrations)
        rows = []
        for row, encoded, prediction, info in zip(
            packet['rows'], packet['encoded'], packet['predictions'], packet['labels']
        ):
            base = {
                'shipment': row['id'],
                'shipment_revision': row['revision'],
                'shipment_identity': shipment_identity(row),
                'shipped': row['shipped'],
                'feature_known': row['feature_known'],
                'market': packet['market'],
                'evaluation_role': roles[name],
                'in_sample_for_model': roles[name] in FIT_ROLES,
                'label_at_as_of': info['label'],
                'label_kind_at_as_of': info['kind'],
                'label_identity_at_as_of': info['label_identity'],
                'calendar_mature_at_as_of': timing.is_mature(row['shipped'], timing.as_of),
                'ignored_feature_keys': (
                    [] if encoded is None else list(encoded.ignored)
                ),
            }
            if prediction is None:
                base['status'] = 'feature_unavailable_at_forecast_time'
                base['feature_timing_reason'] = 'feature_known晚于发运日'
                rows.append(base)
                continue
            point = min(max(prediction['point'], 0.0), float(horizon))
            base.update(
                {
                    'point': point,
                    'point_unclamped': prediction['point'],
                    'baseline_lower': prediction['lower'],
                    'baseline_upper': prediction['upper'],
                    'guarantee_scope': scope,
                }
            )
            if margin is None:
                base['status'] = 'no_group_guarantee'
            else:
                raw_lower = prediction['lower'] - margin
                raw_upper = prediction['upper'] + margin
                lower = max(0.0, raw_lower)
                upper = min(float(horizon), raw_upper)
                base['status'] = 'interval'
                base['lower'] = lower
                base['upper'] = upper
                base['raw_lower'] = raw_lower
                base['raw_upper'] = raw_upper
                base['bounds_clipped_to_target_range'] = (
                    lower != raw_lower or upper != raw_upper
                )
                base['interval_target'] = 'capped_duration_min_duration_horizon'
                base['hit_at_as_of'] = (
                    None
                    if info['label'] is None
                    else lower <= info['label'] <= upper
                )
            rows.append(base)
        batches.append(
            {
                'batch_group': name,
                'market': packet['market'],
                'evaluation_role': roles[name],
                'first_shipped': packet['rows'][0]['shipped'],
                'roster': [row['id'] for row in packet['rows']],
                'roster_digest': roster_digest(packet['rows']),
                'rows': rows,
            }
        )
    return batches


def _coverage(entries):
    if not entries:
        return {
            'evaluated': 0,
            'hit_rate': None,
            'group_hit_rate': None,
            'note': '没有已结算标签，不报告覆盖率。',
        }
    groups = {}
    for entry in entries:
        groups.setdefault(entry['batch_group'], []).append(entry['hit'])
    group_hits = sum(1 for values in groups.values() if all(values))
    hits = sum(1 for entry in entries if entry['hit'])
    return {
        'evaluated': len(entries),
        'interval_hits': hits,
        'interval_misses': len(entries) - hits,
        'hit_rate': hits / len(entries),
        'evaluated_groups': len(groups),
        'group_hits': group_hits,
        'group_hit_rate': group_hits / len(groups),
    }


def _settle_batches(batches, final_shipments, final_labels, final_revision, timing,
                    role_filter=None):
    """用最终资料结算冻结预测；成熟与否只看结算时点的日历规则。

    即使某成熟发运早已收到完成标签，只要同批仍有发运未过观察期，整批
    保持 pending；绝不按“是否收到完成标签”挑快单。结算不修改预测记录。
    """
    entries, pending, excluded = [], [], []
    cutoff = timing.as_of
    for batch in batches:
        name = batch['batch_group']
        if role_filter is not None and batch['evaluation_role'] not in role_filter:
            continue
        roster = batch['roster']
        missing = [shipment for shipment in roster if shipment not in final_shipments]
        if missing:
            pending.append(
                {
                    'batch_group': name,
                    'reason': 'roster_not_in_settlement_frame',
                    'shipments': missing,
                }
            )
            continue
        immature = [
            shipment
            for shipment in roster
            if not timing.is_mature(final_shipments[shipment]['shipped'], cutoff)
        ]
        if immature:
            pending.append(
                {
                    'batch_group': name,
                    'reason': 'not_all_rows_matured_by_settlement_as_of',
                    'immature_shipments': immature,
                    'maturity_rule': 'shipped + horizon + reporting_lag <= 结算as_of',
                }
            )
            continue
        for row in batch['rows']:
            shipment = row['shipment']
            if row['status'] == 'feature_unavailable_at_forecast_time':
                excluded.append(
                    {'shipment': shipment, 'reason': 'feature_unavailable_at_forecast_time'}
                )
                continue
            if row['status'] == 'no_group_guarantee':
                pending.append({'shipment': shipment, 'reason': 'no_group_guarantee'})
                continue
            info = final_labels[shipment]
            revision = final_revision.get(shipment)
            entries.append(
                {
                    'shipment': shipment,
                    'batch_group': name,
                    'market': row['market'],
                    'batch_first_shipped': batch['first_shipped'],
                    'point': row['point'],
                    'lower': row['lower'],
                    'upper': row['upper'],
                    'label': info['label'],
                    'label_kind': info['kind'],
                    'label_identity': info['label_identity'],
                    'hit': row['lower'] <= info['label'] <= row['upper'],
                    'shipment_revision_at_forecast': row['shipment_revision'],
                    'shipment_revision_at_settlement': revision,
                    'feature_revised_after_forecast': revision != row['shipment_revision'],
                    'forecast_shipment_identity': row['shipment_identity'],
                }
            )
    return entries, pending, excluded


def _drift(entries, nominal_alpha, config):
    if not entries:
        return {
            'status': 'not_evaluated',
            'coverage_promise': 'none',
            'note': '没有已结算样本，未作漂移判断。',
        }
    groups = sorted(
        {entry['batch_group'] for entry in entries},
        key=lambda name: (
            next(e['batch_first_shipped'] for e in entries if e['batch_group'] == name),
            name,
        ),
    )
    if len(groups) < config['min_drift_groups']:
        return {
            'status': 'insufficient_groups',
            'coverage_promise': 'calibration_exchangeability_only',
            'evaluated_groups': len(groups),
            'note': '结算批组过少，不能据此判断漂移；兑现仍以校准交换性为前提。',
        }
    half = len(groups) // 2 or 1
    early_groups = set(groups[:half])
    early = [entry for entry in entries if entry['batch_group'] in early_groups]
    late = [entry for entry in entries if entry['batch_group'] not in early_groups]
    early_mae = sum(abs(e['point'] - e['label']) for e in early) / len(early) if early else None
    late_mae = sum(abs(e['point'] - e['label']) for e in late) / len(late) if late else None
    coverage = _coverage(entries)
    ratio = None
    if early_mae and late_mae is not None:
        ratio = late_mae / early_mae
    drifted = bool(ratio is not None and ratio > config['drift_mae_ratio'])
    warning = (
        'settled_hit_rate_below_nominal'
        if coverage['hit_rate'] is not None and coverage['hit_rate'] < 1 - nominal_alpha
        else None
    )
    return {
        'status': 'drift_suspected' if drifted else 'no_clear_drift',
        'coverage_promise': 'retrospective_observation_only',
        'distribution_free_claim': 'withdrawn_under_drift'
        if (drifted or warning)
        else 'not_established_beyond_calibration',
        'early_mae': early_mae,
        'late_mae': late_mae,
        'late_to_early_mae_ratio': ratio,
        'threshold_ratio': config['drift_mae_ratio'],
        'evaluated_groups': len(groups),
        'observed_hit_rate': coverage['hit_rate'],
        'nominal_minimum': 1 - nominal_alpha,
        'warning': warning,
        'note': '结算覆盖率是历史观察值；出现漂移后不得继续声称分布无关覆盖。',
    }


# --------------------------------------------------------------------------
# 身份
# --------------------------------------------------------------------------

def _visible_inputs(shipments, completions, timing):
    visible_completions = [
        {
            'id': completion['id'],
            'revision': completion['revision'],
            'shipment': completion['shipment'],
            'completed': completion['completed'],
            'known': completion['known'],
        }
        for completion in sorted(
            completions.values(), key=lambda value: value['shipment']
        )
        if completion['known'] <= timing.as_of
        and completion['shipment'] in {row['id'] for row in shipments}
    ]
    return {
        'shipments': roster_digest(shipments),
        'completions': digest(visible_completions),
    }


def _identity(timing, config, roster, visible):
    inputs = {
        'config_version': CONFIG_VERSION,
        'model_version': MODEL_VERSION,
        'visible_input_digest': digest(visible),
        'as_of': timing.as_of,
        'horizon': timing.horizon,
        'reporting_lag': timing.reporting_lag,
        'config': config,
        'roster_digest': roster,
    }
    return prediction_identity(inputs), inputs


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def inspect_forecast(payload, config=None, store=None):
    """按 ``metadata.as_of`` 计算并冻结预测；同一身份返回冻结记录。"""
    config = _config(config)
    _validate_config(config)
    snapshot = load(payload, SCHEMAS)
    timing = Timing.from_metadata(snapshot.metadata)
    shipments, completions = _prepare_frame(payload, timing)
    groups = _group_frame(shipments)
    roles, demotions = _assign_roles(groups, timing, config)
    roster = roster_digest(shipments)
    identity, identity_inputs = _identity(
        timing, config, roster, _visible_inputs(shipments, completions, timing)
    )
    store = store if store is not None else FrozenStore()
    cached = store.get(identity)
    if cached is not None:
        return cached

    labels_by_group = {
        name: _labels(rows, completions, timing.as_of, timing)
        for name, rows in groups.items()
    }
    timeline = {
        'as_of': timing.as_of,
        'horizon': timing.horizon,
        'reporting_lag': timing.reporting_lag,
        'maturity_rule': 'shipped + horizon + reporting_lag <= as_of',
        'forecast_time_rule': '预测发生时点取shipped；仅feature_known <= shipped的特征可用',
        'queue_rule': '校准队列在发布时由批组名单与发运日固定；完成标签不参与资格判断',
        'label_delay_rule': (
            '完成记录须known <= as_of且到达时间recorded_at <= as_of才在预测时点可见；'
            '此后到达的标签只作结算'
        ),
        'evaluation_rule': '样本外覆盖评估留出与在途批组，按结算时点日历成熟度结算',
    }
    split = {
        'unit': 'batch_group',
        'assignment': dict(sorted(roles.items())),
        'group_counts': {
            role: sum(1 for value in roles.values() if value == role)
            for role in SPLIT_ROLES
        },
        'no_row_splitting': True,
        'eligibility_basis': timeline['maturity_rule'],
        'selection_basis': '按批组与日历成熟度分派；不看完成标签是否已收到',
        'calibration_manifest': config['group_split'],
        'manifest_demotions': demotions,
        'derived_fractions': config['split_fractions'],
        'excluded_from_calibration': [],
    }
    fit_groups = {
        role: {name: groups[name] for name in groups if roles[name] == role}
        for role in FIT_ROLES
    }
    if not fit_groups['train']:
        return store.freeze(
            identity,
            _failure_record(
                identity,
                identity_inputs,
                timeline,
                split,
                ModelFailure('没有完整成熟且特征可用的训练批组'),
            ),
        )

    def usable(role):
        """整批可用才纳入拟合；同批任一行不可用则整批排除并记录。

        这样批组不会在训练/验证/校准之间被拆开，也不会只保留组内“好”的行。
        批组与组内行都按稳定顺序遍历，保证输入顺序不影响拟合结果。
        """
        rows, labels, dropped = [], [], []
        for name in sorted(fit_groups[role]):
            group_rows = fit_groups[role][name]
            if not _is_mature(group_rows, timing, timing.as_of):
                dropped.append({'batch_group': name, 'reason': 'immature'})
                continue
            unavailable = [
                row['id'] for row in group_rows if row['feature_known'] > row['shipped']
            ]
            if unavailable:
                dropped.append(
                    {
                        'batch_group': name,
                        'reason': 'feature_known晚于发运日',
                        'shipments': unavailable,
                    }
                )
                continue
            for row, info in zip(group_rows, labels_by_group[name]):
                if info['label'] is None:
                    raise ValueError('成熟批组出现空标签，资料成熟规则不一致')
                rows.append(row)
                labels.append(info['label'])
        return rows, labels, dropped

    train_rows, train_labels, train_dropped = usable('train')
    validation_rows, validation_labels, _ = usable('validation')
    if not train_rows:
        return store.freeze(
            identity,
            _failure_record(
                identity,
                identity_inputs,
                timeline,
                {**split, 'excluded_from_fit': {'train': train_dropped}},
                ModelFailure('训练批组在预测时点没有可用特征'),
            ),
        )

    try:
        baseline = fit_baseline(
            train_rows,
            train_labels,
            validation_rows,
            validation_labels,
            alpha=config['alpha'],
            lambdas=tuple(config['lambdas']),
            interval_strategy=config['interval_strategy'],
        )
    except ModelFailure as failure:
        return store.freeze(
            identity,
            _failure_record(
                identity,
                identity_inputs,
                timeline,
                {**split, 'excluded_from_fit': {'train': train_dropped}},
                failure,
            ),
        )

    packets = _packets(baseline, groups, labels_by_group)

    calibration_packets = {}
    calibration_excluded = []
    for name in sorted(groups):
        if roles[name] != 'calibration':
            continue
        packet = packets[name]
        if not _is_mature(packet['rows'], timing, timing.as_of):
            calibration_excluded.append(
                {'batch_group': name, 'reason': 'not_all_rows_matured_at_as_of'}
            )
        elif any(vector is None for vector in packet['encoded']):
            calibration_excluded.append(
                {'batch_group': name, 'reason': 'feature_unavailable_for_some_row'}
            )
        else:
            calibration_packets[name] = packet
    split['excluded_from_calibration'] = calibration_excluded

    all_markets = sorted({_market_of(rows) for rows in groups.values()})
    overall, market_calibrations, market_status, scored = _calibrate_all(
        config['alpha'], calibration_packets, all_markets
    )
    predictions = _prediction_batches(
        packets, overall, market_calibrations, roles, timing
    )

    final_labels = {}
    final_shipments = {row['id']: row for row in shipments}
    final_revision = {row['id']: row['revision'] for row in shipments}
    for name, rows in groups.items():
        for row, info in zip(rows, labels_by_group[name]):
            final_labels[row['id']] = info
    entries, pending, excluded_rows = _settle_batches(
        predictions, final_shipments, final_labels, final_revision, timing,
        role_filter=OUT_OF_SAMPLE_ROLES,
    )
    coverage_at_as_of = _coverage(entries)
    drift_at_as_of = _drift(entries, config['alpha'], config)

    status = _status(overall, market_calibrations, market_status)

    record = {
        'schema_version': 1,
        'kind': 'frozen_forecast',
        'prediction_identity': identity,
        'identity_inputs': identity_inputs,
        'visible_input_digest': identity_inputs['visible_input_digest'],
        'data_version_at_forecast': snapshot.metadata.get('data_version'),
        'status': status,
        'timeline': timeline,
        'target_definition': (
            '截顶交付时长 target = min(completed - shipped, horizon)；'
            '区间承诺不覆盖真实长尾交付时长。'
        ),
        'horizon_cap': timing.horizon,
        'config': config,
        'split': split,
        'baseline': baseline.describe(),
        'calibration': overall.describe(),
        'calibration_eligibility': _calibration_eligibility(calibration_packets, split),
        'calibration_scores': [
            {
                'batch_group': item['batch_group'],
                'market': item['market'],
                'score': item['score'],
                'rows': [
                    {
                        'shipment': row['shipment'],
                        'shipment_revision': row['shipment_revision'],
                        'point': row['point'],
                        'baseline_lower': row['baseline_lower'],
                        'baseline_upper': row['baseline_upper'],
                        'label': row['label'],
                        'label_kind': row['label_kind'],
                        'label_identity': row['label_identity'],
                        'nonconformity': row['nonconformity'],
                    }
                    for row in item['rows']
                ],
            }
            for item in scored
        ],
        'market_calibrations': {
            market: calibration.describe()
            for market, calibration in sorted(market_calibrations.items())
        },
        'markets': market_status,
        'predictions': predictions,
        'settlement_at_as_of': {
            'coverage': coverage_at_as_of,
            'drift': drift_at_as_of,
            'entries': entries,
            'pending': pending,
            'excluded': excluded_rows,
            'note': (
                '发布时点结算样本外且已成熟批组；未成熟批组列在pending，'
                '待后续资料按同一预测身份追加结算，不按完成标签临时挑选。'
            ),
        },
        'roster_digest': roster,
        'roster_source': 'as_of时点已发运的发运行；之后发运或修订不进入该冻结名单。',
        'exchangeability': dict(EXCHANGEABILITY),
    }
    return store.freeze(identity, record)


def _calibration_eligibility(calibration_packets, split):
    """逐批列出校准队列、资格与得分秩，失败原因可查。"""
    excluded = {item['batch_group']: item['reason'] for item in split['excluded_from_calibration']}
    entries = []
    for name in sorted(split['assignment']):
        role = split['assignment'][name]
        if role != 'calibration':
            continue
        packet = calibration_packets.get(name)
        if packet is None:
            entries.append(
                {
                    'batch_group': name,
                    'eligible': False,
                    'reason': excluded.get(name, 'unknown'),
                }
            )
        else:
            entries.append(
                {
                    'batch_group': name,
                    'eligible': True,
                    'reason': 'all_rows_matured_and_encoded_at_as_of',
                }
            )
    return {
        'queue_rule': split['selection_basis'],
        'entries': entries,
    }


def _status(overall, market_calibrations, market_status):
    if not overall.rank_sufficient:
        # 总体池不少于任何单个市场，总体不足则市场也不可能充足。
        return 'insufficient_calibration_groups'
    insufficient = [
        market for market, report in market_status.items()
        if not report['rank_sufficient']
    ]
    if not insufficient:
        return 'ok'
    if any(calibration.rank_sufficient for calibration in market_calibrations.values()):
        return 'partial_market_guarantee'
    return 'pooled_only_no_market_guarantee'


def _failure_record(identity, identity_inputs, timeline, split, failure):
    return {
        'schema_version': 1,
        'kind': 'frozen_forecast',
        'prediction_identity': identity,
        'identity_inputs': identity_inputs,
        'status': 'model_fit_failed',
        'failure': {'stage': 'baseline_fit', 'message': str(failure)},
        'timeline': timeline,
        'target_definition': (
            '截顶交付时长 target = min(completed - shipped, horizon)；'
            '区间承诺不覆盖真实长尾交付时长。'
        ),
        'split': split,
        'config': None,
        'baseline': None,
        'calibration': None,
        'calibration_eligibility': None,
        'market_calibrations': {},
        'markets': {},
        'predictions': [],
        'settlement_at_as_of': {
            'coverage': _coverage([]),
            'drift': {'status': 'not_evaluated', 'coverage_promise': 'none'},
            'entries': [],
            'pending': [],
            'excluded': [],
        },
        'roster_digest': None,
        'exchangeability': dict(EXCHANGEABILITY),
    }


# --------------------------------------------------------------------------
# 事后结算与回放
# --------------------------------------------------------------------------

def settle(record, payload, store=None):
    """用之后到达的资料结算一个冻结预测；不修改预测本身。"""
    if record.get('status') == 'model_fit_failed' or record.get('calibration') is None:
        raise ValueError('该预测没有可结算的区间预测')
    snapshot = load(payload, SCHEMAS)
    timing = Timing.from_metadata(snapshot.metadata)
    forecast_as_of = record['identity_inputs']['as_of']
    if timing.as_of < forecast_as_of:
        raise ValueError('结算资料不得早于预测时点')
    shipments, completions = _prepare_frame(payload, timing)
    groups = _group_frame(shipments)
    final_labels = {}
    for name, rows in groups.items():
        for row, info in zip(rows, _labels(rows, completions, timing.as_of, timing)):
            final_labels[row['id']] = info
    final_shipments = {row['id']: row for row in shipments}
    final_revision = {row['id']: row['revision'] for row in shipments}
    entries, pending, excluded = _settle_batches(
        record['predictions'], final_shipments, final_labels, final_revision, timing,
        role_filter=OUT_OF_SAMPLE_ROLES,
    )
    coverage = _coverage(entries)
    drift = _drift(entries, record['calibration']['alpha'], record['config'])
    outcome = {
        'schema_version': 1,
        'kind': 'frozen_outcome',
        'prediction_identity': record['prediction_identity'],
        'forecast_as_of': forecast_as_of,
        'settled_at_as_of': timing.as_of,
        'settlement_input_digest': snapshot.digest,
        'coverage': coverage,
        'drift': drift,
        'entries': entries,
        'pending': pending,
        'excluded': excluded,
        'claim': (
            '只结算整批已成熟的样本外批组；未成熟批组保持pending。'
            '命中针对截顶时长，不解释为真实长尾时长覆盖。'
        ),
    }
    if store is not None:
        store.append_outcome(record['prediction_identity'], outcome)
    return outcome


def replay(payloads, config=None):
    """对按时点排列的资料快照逐一冻结，再用最终资料结算旧预测。

    每个窗口的身份只绑定该时点可见资料，后到资料不会改写旧预测。覆盖率
    用最终资料结算，未成熟批组列 pending；出现漂移或交换性不成立时只报告
    历史观察值并撤回分布无关承诺。
    """
    if not isinstance(payloads, (list, tuple)) or not payloads:
        raise ValueError('payloads应为非空数组')
    snapshots = []
    previous = None
    for payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError('回放快照应为对象')
        timing = Timing.from_metadata(load(payload, SCHEMAS).metadata)
        if previous is not None and timing.as_of <= previous:
            raise ValueError('回放快照as_of应严格递增')
        previous = timing.as_of
        snapshots.append(payload)

    store = FrozenStore()
    records = [inspect_forecast(payload, config=config, store=store) for payload in snapshots]
    final = snapshots[-1]
    windows = []
    for record in records:
        if record.get('status') == 'model_fit_failed' or record.get('calibration') is None:
            outcome = {
                'coverage': {
                    'evaluated': 0,
                    'hit_rate': None,
                    'group_hit_rate': None,
                    'note': '该窗口模型不可用，未结算。',
                },
                'drift': {'status': 'not_evaluated', 'coverage_promise': 'none'},
                'pending': [],
            }
        else:
            outcome = settle(record, final, store)
        windows.append(
            {
                'as_of': record['identity_inputs']['as_of'],
                'prediction_identity': record['prediction_identity'],
                'status': record['status'],
                'calibration_groups': (
                    None
                    if record['calibration'] is None
                    else record['calibration']['n_groups']
                ),
                'calibration_rank': (
                    None if record['calibration'] is None else record['calibration']['rank']
                ),
                'calibration_rank_sufficient': (
                    None
                    if record['calibration'] is None
                    else record['calibration']['rank_sufficient']
                ),
                'market_status': record['markets'],
                'settled_coverage': outcome['coverage'],
                'settled_drift': outcome['drift'],
                'pending_batches': [
                    item for item in outcome['pending'] if 'batch_group' in item
                ],
            }
        )
    return {
        'schema_version': 1,
        'kind': 'coverage_replay',
        'windows': windows,
        'replay_drift': _replay_drift(windows, _config(config)),
        'store': store.export(),
        'claim': (
            '各窗口覆盖率是用最终资料对冻结预测的观察值；未成熟批组保持pending。'
            '只有批组可交换且无漂移时可沿用有限样本承诺；漂移迹象出现后只报告历史值。'
        ),
        'exchangeability': dict(EXCHANGEABILITY),
    }


def _replay_drift(windows, config):
    scored = [
        window
        for window in windows
        if window['settled_coverage']['evaluated']
        and window['settled_coverage']['hit_rate'] is not None
    ]
    if len(scored) < 2:
        return {
            'status': 'insufficient_windows',
            'distribution_free_claim': 'not_established',
            'note': '可结算窗口少于2个，只报告逐窗口观察覆盖率，不作漂移结论。',
        }
    rates = [window['settled_coverage']['hit_rate'] for window in scored]
    flagged = [
        window['as_of']
        for window in scored
        if window['settled_drift'].get('status') == 'drift_suspected'
        or window['settled_drift'].get('warning')
    ]
    tolerance = config['replay_drop_tolerance']
    drifted = bool(flagged) or rates[-1] < rates[0] - tolerance
    return {
        'status': 'drift_suspected' if drifted else 'no_clear_drift',
        'distribution_free_claim': 'withdrawn_under_drift'
        if drifted
        else 'retrospective_only',
        'evaluated_windows': len(scored),
        'hit_rate_series': rates,
        'first_window_hit_rate': rates[0],
        'last_window_hit_rate': rates[-1],
        'flagged_windows': flagged,
        'note': '回放不能证明未来分布无关覆盖；漂移迹象出现后只能报告历史观察值。',
    }
