"""T48 alarm-persistence helpers. Synthetic stays only (ids far from 3xxxxxxx)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from persistence import (catch_at_quiet_share, hysteresis_state, run_min,
                         share_at_or_above, stay_maxima)
from t35_eicu_operating_points import catch_at_quiet_share as t38_catch
from t35_eicu_operating_points import stay_max_scores


def _frame(seed=0, n_stays=60):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_stays):
        length = int(rng.integers(3, 30))
        event = float(6 + rng.integers(0, length)) if rng.random() < 0.3 else np.nan
        for h in range(6, 6 + length):
            rows.append((1000 + k, h, event))
    df = pd.DataFrame(rows, columns=["stay_id", "hour", "event_h"])
    return df, rng.random(len(df))


def test_run_min_by_hand():
    stay = [1010] * 5 + [1020] * 3
    hour = [6, 7, 8, 9, 10, 6, 7, 8]
    s = [0.5, 0.9, 0.8, 0.2, 0.7, 0.9, 0.95, 0.1]
    assert np.array_equal(run_min(s, stay, hour, 1), s)
    assert run_min(s, stay, hour, 2).tolist() == [-np.inf, 0.5, 0.8, 0.2, 0.2, -np.inf, 0.9, 0.1]
    assert run_min(s, stay, hour, 3).tolist() == [-np.inf, -np.inf, 0.5, 0.2, 0.2,
                                                  -np.inf, -np.inf, 0.1]


def test_run_min_restarts_at_an_hour_gap():
    out = run_min([0.9, 0.9, 0.9, 0.9], [1010] * 4, [6, 7, 9, 10], 2)
    assert out.tolist() == [-np.inf, 0.9, -np.inf, 0.9]


def test_run_min_reads_only_hours_up_to_t():
    df, s = _frame(1)
    base = run_min(s, df.stay_id, df.hour, 3)
    for t in (8, 15):
        later = s.copy()
        later[df.hour.to_numpy() > t] = 0.0
        out = run_min(later, df.stay_id, df.hour, 3)
        keep = df.hour.to_numpy() <= t
        assert np.array_equal(out[keep], base[keep])


def test_k_rule_alarms_exactly_after_k_consecutive_hours_above():
    df, s = _frame(2)
    thr = 0.6
    for k in (2, 3):
        rule = run_min(s, df.stay_id, df.hour, k) >= thr
        above = s >= thr
        brute = np.zeros(len(s), dtype=bool)
        st = df.stay_id.to_numpy()
        for i in range(len(s)):
            lo = i - k + 1
            brute[i] = lo >= 0 and (st[lo:i + 1] == st[i]).all() and above[lo:i + 1].all()
        assert np.array_equal(rule, brute)


def test_hysteresis_turns_on_at_thr_on_and_off_below_thr_off():
    s = [0.1, 0.7, 0.5, 0.35, 0.29, 0.5, 0.65, 0.9]
    st = [1010] * 6 + [1020] * 2
    on = hysteresis_state(s, st, 0.6, 0.3)
    assert on.tolist() == [False, True, True, True, False, False, True, True]


def test_hysteresis_first_alarms_equal_the_k1_rule():
    df, s = _frame(3)
    on = hysteresis_state(s, df.stay_id.to_numpy(), 0.7, 0.4)
    k1 = s >= 0.7
    first = pd.DataFrame({"stay": df.stay_id, "h": df.hour, "a": on, "b": k1})
    fa = first[first.a].groupby("stay").h.min()
    fb = first[first.b].groupby("stay").h.min()
    pd.testing.assert_series_equal(fa, fb, check_names=False)
    assert on.sum() >= k1.sum()                    # hysteresis only lengthens alarms


def test_stay_maxima_and_unit_weight_catch_equal_t38():
    df, s = _frame(4)
    em, qm, _, _ = stay_maxima(df.stay_id, df.hour, df.event_h, s)
    e38, q38 = stay_max_scores(df, s)
    assert np.array_equal(em, e38) and np.array_equal(qm, q38)
    for q in (0.0, 0.1, 0.3, 0.75, 1.0):
        assert np.isclose(catch_at_quiet_share(em, qm, q), t38_catch(e38, q38, q))


def test_weighted_catch_equals_catch_on_replicated_stays():
    df, s = _frame(5)
    em, qm, ep, qp = stay_maxima(df.stay_id, df.hour, df.event_h, s)
    rng = np.random.default_rng(9)
    w = rng.integers(0, 4, ep.max(initial=0) + qp.max(initial=0) + 2).astype(float)
    we, wq = w[ep], w[qp]
    rep_e, rep_q = np.repeat(em, we.astype(int)), np.repeat(qm, wq.astype(int))
    for q in (0.05, 0.2, 0.5):
        assert np.isclose(catch_at_quiet_share(em, qm, q, we, wq),
                          catch_at_quiet_share(rep_e, rep_q, q))
    thr = 0.8
    assert np.isclose(share_at_or_above(qm, thr, wq), (rep_q >= thr).mean())


def _hysteresis_loop(s, st, on_thr, off_thr):
    out, state = [], False
    for i in range(len(s)):
        if i == 0 or st[i] != st[i - 1]:
            state = False
        state = s[i] >= on_thr if not state else s[i] >= off_thr
        out.append(state)
    return np.array(out)


def test_vectorised_hysteresis_equals_the_loop():
    for seed in range(5):
        df, s = _frame(10 + seed)
        st = df.stay_id.to_numpy()
        for on_thr, off_thr in ((0.7, 0.4), (0.9, 0.2), (0.5, 0.5)):
            assert np.array_equal(hysteresis_state(s, st, on_thr, off_thr),
                                  _hysteresis_loop(s, st, on_thr, off_thr))
