"""Regression checks for fees, signal timing, RSI parity and strategy equity."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from jpstock_agent import backtest, ta


def bars(signals, opens=None, closes=None):
    count = len(signals)
    closes = closes if closes is not None else [100.] * count
    opens = opens if opens is not None else [100.] * count
    return pd.DataFrame({
        'open': opens, 'high': np.maximum(opens, closes) + 1,
        'low': np.minimum(opens, closes) - 1, 'close': closes,
        'volume': [10000.] * count, 'signal': signals,
    }, index=pd.bdate_range('2026-01-05', periods=count))


@pytest.mark.parametrize('min_fee', [0., 500.])
def test_commission_charged_once_and_cash_conserved(min_fee):
    cost = backtest.CostModel(.1, 0, 0, min_commission=min_fee)
    df = bars([1, -1, 0])
    trades, final, paid = backtest._execute_trades(df, 100000., cost)
    notional = min(100000 / 1.001, 100000 - min_fee)
    expected = notional - max(notional * .001, min_fee)
    assert final == pytest.approx(expected)
    assert paid == pytest.approx(100000 - final)
    assert trades[-1]['return_pct'] == pytest.approx(round((final / 100000 - 1) * 100, 2))
    assert min(df.attrs['equity_curve']) >= 0


def test_signal_executes_next_open_including_overnight_gap():
    df = bars([1, -1, 0], opens=[90., 120., 80.], closes=[100., 110., 85.])
    trades, final, _ = backtest._execute_trades(df, 120000.)
    assert trades[0]['date'] == '2026-01-06'
    assert trades[0]['signal_date'] == '2026-01-05'
    assert trades[0]['price'] == 120.
    assert trades[1]['date'] == '2026-01-07'
    assert trades[1]['price'] == 80.
    assert final == pytest.approx(80000.)


def test_final_signal_not_executed_and_liquidation_is_explicit():
    trades, final, _ = backtest._execute_trades(bars([0, 0, 1]), 100000.)
    assert trades == []
    assert final == 100000.
    trades, _, _ = backtest._execute_trades(bars([1, 0, 0]), 100000.)
    assert trades[-1]['reason'] == 'end_of_sample'


def test_rsi_matches_analysis_and_independent_wilder_recurrence():
    close = pd.Series(100 + np.sin(np.arange(80)) * 7 + np.arange(80) / 10)
    frame = bars([0] * 80, closes=close.to_numpy())
    gain = loss = 0.
    expected = []
    for change in close.diff().fillna(0):
        gain = gain * 13 / 14 + max(change, 0) / 14
        loss = loss * 13 / 14 + max(-change, 0) / 14
        expected.append(100 if loss == 0 else 100 - 100 / (1 + gain / loss))
    actual = backtest._calculate_rsi(frame.close)
    np.testing.assert_allclose(actual.iloc[13:], expected[13:], atol=1e-10)
    with patch.object(ta, '_get_ohlcv_df', return_value=frame.copy()):
        analysis = ta.ta_rsi('3940')
    for i, record in enumerate(analysis):
        if record['RSI_14'] is not None:
            assert record['RSI_14'] == pytest.approx(actual.iloc[i], abs=1e-10)


def test_flat_strategy_has_zero_risk_even_when_stock_falls():
    frame = bars([0] * 5, closes=[100., 90., 80., 70., 60.])
    with patch.object(backtest, '_get_ohlcv_df', return_value=frame), patch.object(
        backtest, '_generate_signals', return_value=frame.copy()
    ):
        result = backtest.backtest_strategy('3940', 'rsi_reversal')
    assert result['total_return_pct'] == 0
    assert result['max_drawdown_pct'] == 0
    assert result['sharpe_ratio'] == 0
    assert result['execution_model'] == 'previous_close_signal_next_open'


def test_data_source_error_preserved():
    with patch.object(backtest, '_get_ohlcv_df', return_value={'error': 'source unavailable'}):
        assert backtest.backtest_strategy('3940', 'sma_crossover') == {'error': 'source unavailable'}
