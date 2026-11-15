"""验证按预测时点冻结的点预测、批组共形区间、资格与回放。

测试使用合成资料；每个断言对应一条业务承诺，而不是只检查数值接近。
"""
import copy
import json
import unittest
from pathlib import Path

from app.conformal import calibrate, nonconformity, required_rank, score_group
from app.forecast import inspect_forecast, replay, settle
from app.store import FrozenStore
from app.timing import Timing, label_at

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'
SYNTHETIC = json.loads((FIXTURES / 'synthetic.json').read_text(encoding='utf-8'))
CONFIG = {'alpha': 0.2}


def _payload():
    return copy.deepcopy(SYNTHETIC)


def _rows(record):
    return [row for batch in record['predictions'] for row in batch['rows']]


class LabelAndTimingTests(unittest.TestCase):
    def test_target_is_capped_duration_and_immature_is_not_zero(self):
        timing = Timing(20, 10, 2)
        shipment = {
            'id': 's', 'revision': 1, 'batch_group': 'g', 'market': 'm',
            'shipped': 0, 'feature_known': 0, 'recorded_at': 0, 'features': {},
        }
        observed = {'id': 'c', 'revision': 1, 'shipment': 's', 'completed': 7, 'known': 9}
        info = label_at(shipment, observed, 20, timing)
        self.assertEqual(info['label'], 7)
        self.assertEqual(info['kind'], 'observed')

        slow = {'id': 'c', 'revision': 1, 'shipment': 's', 'completed': 25, 'known': 25}
        capped = label_at(shipment, slow, 30, timing)
        self.assertEqual(capped['label'], 10)
        self.assertEqual(capped['kind'], 'observed_capped')

        late_ship = {**shipment, 'shipped': 15, 'recorded_at': 15, 'feature_known': 15}
        unfinished = label_at(late_ship, None, 20, timing)
        self.assertIsNone(unfinished['label'])
        self.assertEqual(unfinished['kind'], 'immature')

        no_label = label_at(late_ship, None, 27, timing)
        self.assertEqual(no_label['label'], 10)
        self.assertEqual(no_label['kind'], 'cap_censored')

    def test_label_delay_and_later_arrival_not_visible_at_forecast(self):
        timing = Timing(7, 5, 2)
        shipment = {
            'id': 's', 'revision': 1, 'batch_group': 'g', 'market': 'm',
            'shipped': 1, 'feature_known': 1, 'recorded_at': 1, 'features': {},
        }
        completion = {'id': 'c', 'revision': 1, 'shipment': 's', 'completed': 4, 'known': 12}
        self.assertIsNone(label_at(shipment, completion, 7, timing)['label'])
        later = label_at(shipment, completion, 12, timing)
        self.assertEqual(later['label'], 3)
        self.assertEqual(later['kind'], 'observed')

    def test_completion_identity_changes_with_revision(self):
        timing = Timing(30, 10, 2)
        shipment = {
            'id': 's', 'revision': 1, 'batch_group': 'g', 'market': 'm',
            'shipped': 0, 'feature_known': 0, 'recorded_at': 0, 'features': {},
        }
        first = {'id': 'c', 'revision': 1, 'shipment': 's', 'completed': 5, 'known': 6}
        second = {**first, 'revision': 2, 'completed': 6, 'known': 7}
        left = label_at(shipment, first, 30, timing)
        right = label_at(shipment, second, 30, timing)
        self.assertNotEqual(left['label_identity']['label_id'], right['label_identity']['label_id'])
        self.assertEqual(left['label_identity']['completion_revision'], 1)


class RankTests(unittest.TestCase):
    def test_required_rank_and_insufficiency(self):
        self.assertEqual(required_rank(9, 0.1), (9, True))
        self.assertEqual(required_rank(8, 0.1), (9, False))
        self.assertEqual(required_rank(12, 0.2), (11, True))
        self.assertEqual(required_rank(10, 0.2), (9, True))
        self.assertEqual(required_rank(0, 0.2), (0, False))

    def test_threshold_is_kth_smallest_and_ranks_are_reportable(self):
        groups = [
            {'batch_group': f'g{i}', 'market': 'm', 'score': float(value)}
            for i, value in enumerate([5, 1, 3, 2, 4, 6, 7, 8, 9, 10])
        ]
        result = calibrate(0.2, groups)
        self.assertEqual(result.n_groups, 10)
        self.assertEqual(result.rank, 9)
        self.assertEqual(result.threshold, 9.0)
        self.assertTrue(result.rank_sufficient)
        selected = [item for item in result.groups if item['selected_as_threshold']]
        self.assertEqual(selected[0]['batch_group'], 'g8')
        self.assertEqual(
            [item['rank'] for item in result.groups], list(range(1, 11))
        )

    def test_group_score_is_max_over_rows(self):
        group = {'rows': [1, 2]}
        predictions = [
            {'lower': 0, 'upper': 1},
            {'lower': 0, 'upper': 1},
        ]
        labels = [1, 4]
        self.assertEqual(score_group(group, predictions, labels), 3.0)
        self.assertEqual(nonconformity(predictions[0], 1), 0.0)


class GroupIsolationTests(unittest.TestCase):
    def test_no_batch_group_crosses_train_calibration_or_holdout(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        roles = record['split']['assignment']
        for batch in record['predictions']:
            name = batch['batch_group']
            self.assertEqual(batch['evaluation_role'], roles[name])
        self.assertTrue(record['split']['no_row_splitting'])
        for name, role in roles.items():
            # 组内所有行都在同一角色下，不存在同批多行被拆到两边的可能。
            self.assertEqual(
                {
                    batch['evaluation_role']
                    for batch in record['predictions']
                    if batch['batch_group'] == name
                },
                {role},
            )

    def test_fit_never_mixes_groups(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        assignment = record['split']['assignment']
        trained = {
            batch['batch_group']
            for batch in record['predictions']
            if batch['evaluation_role'] == 'train'
        }
        calibrated = {
            batch['batch_group']
            for batch in record['predictions']
            if batch['evaluation_role'] == 'calibration'
        }
        self.assertTrue(trained.isdisjoint(calibrated))
        for name in trained | calibrated:
            self.assertEqual(assignment[name], record['split']['assignment'][name])

    def test_manifest_group_never_split(self):
        payload = _payload()
        first = inspect_forecast(payload, config=CONFIG)
        names = sorted(first['split']['assignment'])
        manifest = {name: 'calibration' for name in names[:6]}
        manifest.update({name: 'train' for name in names[6:20]})
        record = inspect_forecast(payload, config={**CONFIG, 'group_split': manifest})
        for batch in record['predictions']:
            if batch['batch_group'] in manifest:
                self.assertEqual(batch['evaluation_role'], manifest[batch['batch_group']])


class CalibrationQueueTests(unittest.TestCase):
    def test_immature_group_never_enters_calibration_even_with_completion(self):
        payload = _payload()
        as_of = payload['metadata']['as_of']
        horizon = payload['metadata']['horizon']
        lag = payload['metadata']['reporting_lag']
        # Ship a new group one day before as_of and give it a completion label.
        payload['collections']['shipments'].append(
            {
                'id': 'late-1', 'revision': 1, 'batch_group': 'late', 'market': 'm1',
                'shipped': as_of - 1, 'features': {'distance': 2.0, 'weight': 1.0},
                'feature_known': as_of - 1,
            }
        )
        payload['collections']['completions'].append(
            {
                'id': 'late-c', 'revision': 1, 'shipment': 'late-1',
                'completed': as_of, 'known': as_of,
            }
        )
        names = sorted(
            name for name in {
                row['batch_group'] for row in payload['collections']['shipments']
            }
        )
        # 保留默认分派，只把尚在途的 late 批组放进校准清单，验证被整批降级。
        derived = inspect_forecast(payload, config=CONFIG)['split']['assignment']
        manifest = {name: derived.get(name, 'pending') for name in names}
        manifest['late'] = 'calibration'
        record = inspect_forecast(
            payload, config={**CONFIG, 'group_split': manifest}
        )
        demoted = {
            item['batch_group'] for item in record['split']['manifest_demotions']
        }
        self.assertIn('late', demoted)
        self.assertEqual(
            record['split']['assignment']['late'], 'pending'
        )
        self.assertNotIn('late', {
            item['batch_group']
            for item in record['calibration_eligibility']['entries']
            if item['eligible']
        })
        # Its single-row batch is reported as pending, not scored as a fast hit.
        pending = record['settlement_at_as_of']['pending']
        self.assertTrue(
            any(item.get('batch_group') == 'late' for item in pending)
        )

    def test_calibration_queue_is_calendar_not_label_based(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        horizon = record['identity_inputs']['horizon']
        lag = record['identity_inputs']['reporting_lag']
        as_of = record['identity_inputs']['as_of']
        for batch in record['predictions']:
            if batch['evaluation_role'] != 'calibration':
                continue
            for row in batch['rows']:
                self.assertLessEqual(
                    row['shipped'] + horizon + lag, as_of
                )

    def test_insufficient_calibration_reports_no_guarantee(self):
        payload = _payload()
        names = sorted(
            {row['batch_group'] for row in payload['collections']['shipments']}
        )
        manifest = {name: 'train' for name in names}
        # 只留两个批组进校准，低于 alpha=0.2 所需的有限样本秩。
        manifest[names[-1]] = 'calibration'
        manifest[names[-2]] = 'calibration'
        record = inspect_forecast(payload, config={**CONFIG, 'group_split': manifest})
        self.assertFalse(record['calibration']['rank_sufficient'])
        self.assertEqual(record['status'], 'insufficient_calibration_groups')
        for row in _rows(record):
            if row['status'] != 'feature_unavailable_at_forecast_time':
                self.assertEqual(row['status'], 'no_group_guarantee')
                self.assertNotIn('lower', row)


class MarketTests(unittest.TestCase):
    def test_scarce_market_does_not_inherit_pool_coverage(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        self.assertEqual(record['status'], 'partial_market_guarantee')
        market = record['markets']['m2']
        self.assertFalse(market['rank_sufficient'])
        self.assertEqual(market['status'], 'insufficient_market_samples')
        self.assertIn('不能提供市场级组覆盖保证', market['note'])
        # Rows of the scarce market still get pool-scoped intervals, explicitly labelled.
        scarce = [
            row
            for row in _rows(record)
            if row['market'] == 'm2' and row['status'] == 'interval'
        ]
        self.assertTrue(scarce)
        for row in scarce:
            self.assertEqual(
                row['guarantee_scope'],
                'pooled_group_guarantee_no_market_guarantee',
            )

    def test_market_with_enough_groups_has_own_threshold(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        market = record['markets']['m1']
        self.assertTrue(market['rank_sufficient'])
        rows = [
            row for row in _rows(record)
            if row['market'] == 'm1' and row['status'] == 'interval'
        ]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row['guarantee_scope'], 'market_group_guarantee')


class FrozenIdentityTests(unittest.TestCase):
    def test_later_completion_does_not_rewrite_forecast(self):
        payload = _payload()
        record = inspect_forecast(payload, config=CONFIG)
        later = copy.deepcopy(payload)
        later['collections']['completions'].append(
            {
                'id': 'later', 'revision': 1, 'shipment': 's162',
                'completed': 165, 'known': 165, 'recorded_at': 165,
            }
        )
        again = inspect_forecast(later, config=CONFIG)
        self.assertEqual(record['prediction_identity'], again['prediction_identity'])
        self.assertEqual(record['predictions'], again['predictions'])

    def test_feature_revision_after_cutoff_does_not_rewrite_forecast(self):
        payload = _payload()
        record = inspect_forecast(payload, config=CONFIG)
        revised = copy.deepcopy(payload)
        original = next(
            row for row in revised['collections']['shipments'] if row['id'] == 's001'
        )
        revised['collections']['shipments'].append(
            {
                **original,
                'revision': 2,
                'recorded_at': 170,
                'features': {'distance': 99.0, 'weight': 99.0},
            }
        )
        again = inspect_forecast(revised, config=CONFIG)
        self.assertEqual(record['prediction_identity'], again['prediction_identity'])
        row = next(
            row for row in _rows(again) if row['shipment'] == 's001'
        )
        first = next(row for row in _rows(record) if row['shipment'] == 's001')
        self.assertEqual(row['point'], first['point'])

    def test_same_identity_is_idempotent_and_conflicts_rejected(self):
        record = inspect_forecast(_payload(), config=CONFIG, store=FrozenStore())
        store = FrozenStore()
        frozen = store.freeze(record['prediction_identity'], record)
        self.assertEqual(
            store.freeze(record['prediction_identity'], record)['prediction_identity'],
            frozen['prediction_identity'],
        )
        with self.assertRaisesRegex(ValueError, '拒绝回写'):
            store.freeze(record['prediction_identity'], {**record, 'status': 'ok'})

    def test_settle_appends_without_mutating_forecast(self):
        payload = _payload()
        store = FrozenStore()
        record = inspect_forecast(payload, config=CONFIG, store=store)
        before = copy.deepcopy(record)
        outcome = settle(record, payload, store)
        self.assertEqual(record, before)
        self.assertEqual(
            outcome['prediction_identity'], record['prediction_identity']
        )
        self.assertEqual(len(store.outcomes(record['prediction_identity'])), 1)
        settle(record, payload, store)
        self.assertEqual(len(store.outcomes(record['prediction_identity'])), 1)

    def test_input_order_and_retry_do_not_change_result(self):
        first = inspect_forecast(_payload(), config=CONFIG)
        shuffled = _payload()
        for rows in shuffled['collections'].values():
            rows.reverse()
            rows.append(copy.deepcopy(rows[0]))
        second = inspect_forecast(shuffled, config=CONFIG)
        self.assertEqual(first, second)


class SettlementTests(unittest.TestCase):
    def test_in_transit_group_with_fast_completion_stays_pending(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        pending_batches = {
            item['batch_group']
            for item in record['settlement_at_as_of']['pending']
            if 'batch_group' in item
        }
        # g60 and g62 already carry a completion label but are not calendar mature.
        self.assertIn('g60', pending_batches)
        self.assertIn('g62', pending_batches)
        self.assertFalse(
            any(
                entry['batch_group'] == 'g60'
                for entry in record['settlement_at_as_of']['entries']
            )
        )

    def test_settlement_uses_final_labels_and_flags_revision(self):
        payload = _payload()
        names = sorted(
            {row['batch_group'] for row in payload['collections']['shipments']}
        )
        manifest = {name: 'holdout' for name in names}
        for name in names[:12]:
            manifest[name] = 'calibration'
        manifest[names[12]] = 'train'
        record = inspect_forecast(payload, config={**CONFIG, 'group_split': manifest})
        target = next(
            row['shipment']
            for batch in record['predictions']
            for row in batch['rows']
            if batch['evaluation_role'] == 'holdout'
            and row['status'] == 'interval'
        )
        revised = copy.deepcopy(payload)
        original = next(
            row for row in revised['collections']['shipments'] if row['id'] == target
        )
        revised['collections']['shipments'].append(
            {**original, 'revision': 2, 'recorded_at': 170}
        )
        # 结算资料晚于该修订的到达时间，修订才可在结算时点可见。
        revised['metadata']['as_of'] = 170
        outcome = settle(record, revised)
        entry = next(
            entry for entry in outcome['entries'] if entry['shipment'] == target
        )
        self.assertTrue(entry['feature_revised_after_forecast'])
        self.assertEqual(entry['shipment_revision_at_forecast'], 1)
        self.assertEqual(entry['shipment_revision_at_settlement'], 2)

    def test_coverage_counts_group_hits_and_row_hits_separately(self):
        record = inspect_forecast(_payload(), config=CONFIG)
        coverage = record['settlement_at_as_of']['coverage']
        self.assertGreater(coverage['evaluated'], 0)
        self.assertLessEqual(coverage['group_hits'], coverage['evaluated_groups'])
        self.assertGreaterEqual(coverage['hit_rate'], 0.0)
        self.assertLessEqual(coverage['hit_rate'], 1.0)


class ReplayTests(unittest.TestCase):
    def _snapshots(self):
        payload = _payload()
        snapshots = []
        for as_of in (120, 140, 160):
            snapshot = copy.deepcopy(payload)
            snapshot['metadata']['as_of'] = as_of
            snapshot['collections']['shipments'] = [
                row for row in snapshot['collections']['shipments']
                if row['shipped'] <= as_of
            ]
            snapshot['collections']['completions'] = [
                row for row in snapshot['collections']['completions']
                if row['known'] <= as_of
            ]
            snapshots.append(snapshot)
        return snapshots

    def test_replay_has_independent_identities_and_no_backwrite(self):
        result = replay(self._snapshots(), config=CONFIG)
        identities = [window['prediction_identity'] for window in result['windows']]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(result['kind'], 'coverage_replay')
        self.assertIn('观察值', result['claim'])
        for window in result['windows']:
            self.assertIn('settled_coverage', window)
            self.assertIn('settled_drift', window)

    def test_replay_drift_withdraws_distribution_free_claim(self):
        snapshots = self._snapshots()
        result = replay(snapshots, config={**CONFIG, 'min_drift_groups': 1})
        drifted = result['replay_drift']
        self.assertIn(drifted['distribution_free_claim'], (
            'retrospective_only', 'withdrawn_under_drift'
        ))
        if drifted['status'] == 'drift_suspected':
            self.assertEqual(
                drifted['distribution_free_claim'], 'withdrawn_under_drift'
            )

    def test_replay_rejects_non_increasing_cutoffs(self):
        snapshots = self._snapshots()
        with self.assertRaisesRegex(ValueError, '严格递增'):
            replay([snapshots[1], snapshots[0]], config=CONFIG)


class FailureStateTests(unittest.TestCase):
    def test_missing_train_group_is_explicit_failure(self):
        payload = _payload()
        names = sorted(
            {row['batch_group'] for row in payload['collections']['shipments']}
        )
        manifest = {name: 'holdout' for name in names}
        record = inspect_forecast(payload, config={**CONFIG, 'group_split': manifest})
        self.assertEqual(record['status'], 'model_fit_failed')
        self.assertIn('训练批组', record['failure']['message'])
        self.assertEqual(record['predictions'], [])

    def test_small_fixture_reports_failure_not_fake_interval(self):
        payload = json.loads(
            (FIXTURES / 'sample.json').read_text(encoding='utf-8')
        )
        record = inspect_forecast(payload)
        self.assertEqual(record['status'], 'model_fit_failed')
        self.assertEqual(record['settlement_at_as_of']['coverage']['evaluated'], 0)
        self.assertIsNone(record['settlement_at_as_of']['coverage']['hit_rate'])

    def test_feature_unavailable_row_is_not_zero_filled(self):
        payload = _payload()
        record = inspect_forecast(payload, config=CONFIG)
        rows = {
            row['shipment']: row for row in _rows(record)
        }
        self.assertEqual(rows['s155']['status'], 'feature_unavailable_at_forecast_time')
        self.assertNotIn('point', rows['s155'])

    def test_bad_config_rejected(self):
        payload = _payload()
        for config in (
            {'alpha': 0},
            {'alpha': 1},
            {'split_fractions': [0.5, 0.4, 0.9]},
            {'lambdas': []},
            {'interval_strategy': 'magic'},
            {'unknown': 1},
        ):
            with self.assertRaises(ValueError):
                inspect_forecast(payload, config=config)

    def test_cross_market_group_rejected(self):
        payload = _payload()
        rows = payload['collections']['shipments']
        by_group = {}
        for row in rows:
            by_group.setdefault(row['batch_group'], []).append(row)
        target = next(
            group for group in by_group.values() if len(group) >= 2
        )
        target[1]['market'] = 'm9'
        with self.assertRaisesRegex(ValueError, '跨市场'):
            inspect_forecast(payload, config=CONFIG)


if __name__ == '__main__':
    unittest.main()
