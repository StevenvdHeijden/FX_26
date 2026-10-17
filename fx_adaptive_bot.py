"""
fx_adaptive_bot.py
==================
Walk-forward ensemble of optimized FX indicator strategies with adaptive weights.

Pipeline
--------
1. Every strategy maps (OHLC data, params) -> position in {-1, 0, +1}, decided at bar close
   and held over the NEXT bar (same convention as SMABacktester: position.shift(1) * return).
2. Walk-forward: at each fold, parameters are re-optimized on the trailing `train_bars`
   (net of costs, Sharpe objective, optional neighbourhood smoothing to prefer plateaus over
   peaks) and then frozen for the next `step_bars` (out-of-sample).
3. "Shadow" net returns are recorded per strategy (what each would have earned alone,
   including transaction costs). These, not the bot's own fills, drive the weights.
4. Weights = softmax of EWMA Sharpe, smoothed over time and clipped to [w_min, w_max].
   Ensemble signal S = sum_i w_i * pos_i; trade sign(S) when |S| > threshold.

All strategies are vectorised re-implementations of the *Backtester classes (SMA, SMAEMA, MACD,
RSI, BB, Stochastic, FRL), checked bar-by-bar against copies of the original code. RSI is
reproduced AS WRITTEN (it has two bugs, see sig_rsi); use --fix-rsi for the corrected version.

Usage
-----
    python fx_adaptive_bot.py --synthetic
    python fx_adaptive_bot.py --data "C:/data/AUDJPY/**/DAT_MT_AUDJPY_M1_2023*.csv" --tf 30min --cost 0.000035
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_histdata(pattern: str, timeframe: str = "30min") -> pd.DataFrame:
    """Load histdata.com MT-format M1 csv files (several months allowed) and resample to OHLC."""
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    cols = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    raw = pd.concat(pd.read_csv(f, header=None, names=cols) for f in files)
    raw.index = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%Y.%m.%d %H:%M")
    raw = raw.sort_index()[["Open", "High", "Low", "Close"]]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    return raw.resample(timeframe).agg(agg).dropna()


def synthetic_data(n: int = 40000, seed: int = 0, timeframe: str = "30min") -> pd.DataFrame:
    """Driftless random walk with mild vol clustering: a sanity check, there should be NO edge."""
    rng = np.random.default_rng(seed)
    vol = 4e-4 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.2)
    r = rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 1, n)) * vol * close * 0.5
    idx = pd.date_range("2021-01-01", periods=n, freq=timeframe)
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum(open_, close) + wick,
         "Low": np.minimum(open_, close) - wick, "Close": close}, index=idx)


def bars_per_year(timeframe: str) -> float:
    return pd.Timedelta("365D") / pd.Timedelta(timeframe) * 5 / 7  # approx. FX week


# ----------------------------------------------------------------------------------------
# Signals: causal functions, position decided at bar close
# ----------------------------------------------------------------------------------------
def _hold(long_cond, short_cond, flat_cond=None, flat_last=False) -> pd.Series:
    """Stateful position: set on conditions (later assignments override earlier ones),
    otherwise hold the previous position."""
    raw = pd.Series(np.nan, index=long_cond.index)
    if flat_cond is not None and not flat_last:
        raw[flat_cond] = 0.0
    raw[long_cond] = 1.0
    raw[short_cond] = -1.0
    if flat_cond is not None and flat_last:
        raw[flat_cond] = 0.0
    return raw.ffill().fillna(0.0)


def _side(cond_long: pd.Series, *valid: pd.Series) -> pd.Series:
    """+1 where cond_long else -1 (always in the market); 0 while any input is still NaN."""
    ok = valid[0].notna()
    for v in valid[1:]:
        ok &= v.notna()
    return pd.Series(np.where(cond_long, 1.0, -1.0), index=cond_long.index).where(ok, 0.0)


# --- Signals below replicate your *Backtester classes (checked bar-by-bar in the test script) ---
def sig_sma(df, s, l):                       # SMABacktester: SMA_S > SMA_L -> long else short
    c = df["Close"]
    a, b = c.rolling(s).mean(), c.rolling(l).mean()
    return _side(a > b, a, b)


def sig_smaema(df, sma, ema):                # SMAEMABacktester: EMA > SMA -> long else short
    c = df["Close"]
    a, b = c.ewm(span=ema, min_periods=ema).mean(), c.rolling(sma).mean()
    return _side(a > b, a, b)


def sig_macd(df, ema_s, ema_l, signal_mw):   # MACDBacktester: MACD > signal -> long else short
    c = df["Close"]
    m = c.ewm(span=ema_s, min_periods=ema_s).mean() - c.ewm(span=ema_l, min_periods=ema_l).mean()
    d = m - m.ewm(span=signal_mw, min_periods=signal_mw).mean()
    return _side(d > 0, d)


def sig_rsi(df, periods, lower):
    """RSIBacktester AS WRITTEN. Two things in the original make it behave differently from a
    textbook RSI (see the notes in the reply): D is negative, so RSI = U/(U+D)*100 is not bounded
    in [0,100]; and the second np.where overwrites the first, so the short side never fires and
    `rsi_upper` has no effect. Net effect: long when price < price `periods` bars ago, else flat."""
    c = df["Close"]
    diff = c.diff()
    U = pd.Series(np.where(diff > 0, diff, 0.0), index=c.index)
    D = pd.Series(np.where(diff < 0, diff, 0.0), index=c.index)
    ma_u, ma_d = U.rolling(periods).mean(), D.rolling(periods).mean()
    rsi = ma_u / (ma_u + ma_d) * 100
    return pd.Series(np.where(rsi < lower, 1.0, 0.0), index=c.index)


def sig_rsi_fixed(df, periods, lower, upper):
    """What the RSI backtester presumably intends: bounded RSI (simple-average / Cutler form),
    long below `lower`, short above `upper`, flat in between."""
    c = df["Close"]
    diff = c.diff()
    ma_u = diff.clip(lower=0).rolling(periods).mean()
    ma_d = (-diff.clip(upper=0)).rolling(periods).mean()
    rsi = 100 * ma_u / (ma_u + ma_d)
    return pd.Series(np.where(rsi > upper, -1.0, np.where(rsi < lower, 1.0, 0.0)), index=c.index)


def sig_bb(df, window, dev):                 # BBBacktester: mean reversion, flat on SMA cross
    c = df["Close"]
    ma, sd = c.rolling(window).mean(), c.rolling(window).std()
    dist = c - ma
    cross = (dist * dist.shift(1)) < 0       # applied last, as in the original
    return _hold(c < ma - sd * dev, c > ma + sd * dev, cross, flat_last=True)


def sig_stoch(df, periods, D_mw):            # SOBacktester: %K > %D -> long else short
    low, high = df["Low"].rolling(periods).min(), df["High"].rolling(periods).max()
    K = (df["Close"] - low) / (high - low) * 100
    D = K.rolling(D_mw).mean()
    return _side(K > D, K, D)


def sig_ema(df, s, l):
    """Plain EMA/EMA crossover (NOT in the registry: your SMAEMABacktester is EMA vs SMA)."""
    c = df["Close"]
    a, b = c.ewm(span=s, min_periods=s).mean(), c.ewm(span=l, min_periods=l).mean()
    return _side(a > b, a, b)


_EXT_CACHE: dict = {}


def _last_extreme_idx(x: pd.Series, order: int, is_max: bool) -> np.ndarray:
    """For every bar t, the index of the most recent local extremum of x[:t+1], identical to
    argrelextrema(x[:t+1], greater_equal / less_equal, order=order)[0][-1] (mode='clip'), but
    O(n * order) instead of recomputing from scratch at each bar. -1 if none.

    Extrema older than `order` bars are 'confirmed' (compared with `order` bars on both sides);
    the last `order` bars are provisional and are compared with whatever data exists up to t,
    exactly as the original per-bar recomputation does. Nothing after t is ever used."""
    n, k = len(x), order
    cmp = np.greater_equal if is_max else np.less_equal
    xv, ar = x.to_numpy(), np.arange(len(x))
    cen = x.rolling(2 * k + 1, center=True, min_periods=1)
    confirmed = cmp(xv, (cen.max() if is_max else cen.min()).to_numpy())
    last_conf = np.maximum.accumulate(np.where(confirmed, ar, -1))
    out = np.full(n, -1)
    if n > k:
        out[k:] = last_conf[: n - k]               # latest confirmed extremum at or before t-k
    for d in range(k - 1, -1, -1):                  # provisional: bar t-d, smallest d wins
        r = x.rolling(d + k + 1, min_periods=1)
        win = (r.max() if is_max else r.min()).to_numpy()
        cand = np.r_[np.full(d, np.nan), xv[: n - d]]
        with np.errstate(invalid="ignore"):
            ok = cmp(cand, win)
        out = np.where(ok, ar - d, out)
    return out


def _extremes(df, order):
    key = (order, len(df), df.index[0], df.index[-1], float(df["High"].sum()), float(df["Low"].sum()))
    if key not in _EXT_CACHE:
        if len(_EXT_CACHE) > 32:
            _EXT_CACHE.clear()
        _EXT_CACHE[key] = (_last_extreme_idx(df["High"], order, True),
                           _last_extreme_idx(df["Low"], order, False))
    return _EXT_CACHE[key]


def sig_frl(df, order, f1, f2):
    """Fibonacci retracement strategy of FRLBacktester. Note: the third level (fibonacci3) is
    computed in the original but never used in any position rule, so it is not a parameter here."""
    c, H, L = df["Close"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy()
    hi, lo = _extremes(df, order)
    valid = (hi >= 0) & (lo >= 0)
    hh = np.where(hi >= 0, H[np.maximum(hi, 0)], np.nan)
    ll = np.where(lo >= 0, L[np.maximum(lo, 0)], np.nan)
    up = hi > lo                                     # latest high more recent than latest low
    down = ~up

    def level(f):
        return np.where(up, hh - (hh - ll) * f, hh - (hh - ll) * (1 - f))

    def sh(a):
        return np.r_[np.nan, a[:-1]]

    R1, R2, p, p0 = level(f1), level(f2), c, sh(c)
    pos = np.where(~((hh == sh(hh)) & (ll == sh(ll))), 0.0, np.nan)   # flat on new high/low
    with np.errstate(invalid="ignore"):
        pos = np.where(down & (p0 < sh(R1)) & (p > R1), 1.0, pos)     # long: breaks f1 upwards
        pos = np.where(down & (p0 < sh(R2)) & (p >= R2), 0.0, pos)    # take profit at f2
        pos = np.where(down & (p0 > sh(ll)) & (p <= ll), 0.0, pos)    # stop at the low
        pos = np.where(up & (p0 > sh(R1)) & (p < R1), -1.0, pos)      # short: breaks f1 downwards
        pos = np.where(up & (p0 > sh(R2)) & (p <= R2), 0.0, pos)      # take profit at f2
        pos = np.where(up & (p0 < sh(hh)) & (p >= hh), 0.0, pos)      # stop at the high
    pos = pd.Series(pos, index=df.index).ffill().fillna(0.0)
    return pos.where(valid, 0.0)


@dataclass
class Strategy:
    name: str
    fn: Callable
    grid: dict
    valid: Callable = lambda p: True


STRATEGIES = [
    Strategy("SMA", sig_sma, {"s": list(range(5, 45, 5)), "l": list(range(30, 150, 10))},
             lambda p: p["s"] < p["l"]),
    Strategy("SMAEMA", sig_smaema, {"sma": list(range(20, 150, 10)), "ema": list(range(5, 45, 5))}),
    Strategy("MACD", sig_macd, {"ema_s": list(range(6, 22, 2)), "ema_l": list(range(20, 55, 5)),
                                "signal_mw": list(range(5, 15, 2))},
             lambda p: p["ema_s"] < p["ema_l"]),
    # `lower` is a dummy axis here (see sig_rsi): every value gives the same signal
    Strategy("RSI", sig_rsi, {"periods": [5, 10, 14, 21, 28, 40], "lower": [30]}),
    Strategy("BB", sig_bb, {"window": list(range(10, 70, 10)), "dev": [1, 2, 3]}),  # dev is cast to int in your optimiser
    Strategy("STOCH", sig_stoch, {"periods": [5, 10, 14, 20, 30], "D_mw": [3, 5, 9]}),
    Strategy("FRL", sig_frl, {"order": [3, 5, 8, 12, 20], "f1": [0.146, 0.236, 0.382],
                              "f2": [0.382, 0.5, 0.618, 0.786]},
             lambda p: p["f1"] < p["f2"]),
]

# Drop-in replacement for the RSI entry if you want the corrected indicator (python ... --fix-rsi)
RSI_FIXED = Strategy("RSI", sig_rsi_fixed, {"periods": [7, 14, 21, 28], "lower": [20, 25, 30, 35, 40],
                                            "upper": [60, 65, 70, 75, 80]},
                     lambda p: p["lower"] < p["upper"])


# ----------------------------------------------------------------------------------------
# Returns, optimisation
# ----------------------------------------------------------------------------------------
def shadow_returns(pos: np.ndarray, r: np.ndarray, cost: float) -> np.ndarray:
    """Net return per bar: position held during bar t was decided at close t-1.
    Proportional cost on every change of position (a flip -1 -> +1 costs 2*cost)."""
    p = np.r_[0.0, pos[:-1]]
    turnover = np.abs(np.diff(p, prepend=0.0))
    return p * r - cost * turnover


@dataclass
class Store:
    combos: list
    rets: np.ndarray   # (n_combos, n_bars) net returns, float32
    ok: np.ndarray     # valid parameter combination?
    shape: tuple       # grid shape, for neighbourhood smoothing


def precompute(df, strat: Strategy, cost: float, r: np.ndarray) -> Store:
    """Indicators are causal, so computing them once on the full history gives exactly the
    values a live bot would have had at each bar. Slicing later is therefore look-ahead free."""
    keys = list(strat.grid)
    combos = list(product(*strat.grid.values()))
    shape = tuple(len(v) for v in strat.grid.values())
    rets = np.zeros((len(combos), len(df)), dtype=np.float32)
    ok = np.zeros(len(combos), dtype=bool)
    for i, c in enumerate(combos):
        p = dict(zip(keys, c))
        if not strat.valid(p):
            continue
        rets[i] = shadow_returns(strat.fn(df, **p).to_numpy(), r, cost)
        ok[i] = True
    return Store(combos, rets, ok, shape)


def select(store: Store, a: int, b: int, smooth: bool = True):
    """Best parameter set on bars [a, b): per-bar Sharpe of net returns. With smooth=True the
    score surface is averaged over neighbouring grid points, which favours plateaus."""
    sub = store.rets[:, a:b].astype(np.float64)
    mu, sd = sub.mean(1), sub.std(1)
    good = store.ok & (sd > 1e-12)
    if not good.any():
        return None
    score = np.where(good, mu / np.where(good, sd, 1.0), np.nan)
    g = score.reshape(store.shape)
    bad = ~np.isfinite(g)
    filled = np.where(bad, np.nanmin(g), g)
    if smooth:
        filled = uniform_filter(filled, size=3, mode="nearest")
    filled[bad] = -np.inf
    return int(np.argmax(filled))


def walk_forward(df, strategies, cost, train_bars, step_bars, smooth=True):
    r = np.log(df["Close"]).diff().fillna(0.0).to_numpy()
    names = [s.name for s in strategies]
    stores = {s.name: precompute(df, s, cost, r) for s in strategies}
    n = len(df)
    pos = pd.DataFrame(0.0, index=df.index, columns=names)
    ret = pd.DataFrame(0.0, index=df.index, columns=names)
    chosen = []
    for b in range(train_bars, n, step_bars):
        e = min(b + step_bars, n)
        for s in strategies:
            st = stores[s.name]
            j = select(st, b - train_bars, b, smooth)
            if j is None:
                continue
            params = dict(zip(s.grid, st.combos[j]))
            chosen.append({"bar": b, "date": df.index[b], "strategy": s.name, **params})
            # recompute on data truncated at e: nothing beyond the fold is ever seen
            pos.iloc[b:e, names.index(s.name)] = s.fn(df.iloc[:e], **params).to_numpy()[b:e]
            ret.iloc[b:e, names.index(s.name)] = st.rets[j, b:e]
    return (pos.iloc[train_bars:], ret.iloc[train_bars:], r[train_bars:],
            pd.DataFrame(chosen))


# ----------------------------------------------------------------------------------------
# Adaptive weights and ensemble
# ----------------------------------------------------------------------------------------
def _bound(w, lo, hi):
    for _ in range(25):
        w = np.clip(w, lo, hi)
        w = w / w.sum()
    return w


def adaptive_weights(shadow: pd.DataFrame, bpy: float, halflife: int = 2000, tau: float = 1.0,
                     update_every: int = 48, lam: float = 0.3, w_min: float = 0.03,
                     w_max: float = 0.40) -> pd.DataFrame:
    """Softmax of EWMA annualised Sharpe of shadow returns. Scores are lagged one bar (causal)."""
    m = shadow.shape[1]
    mu = shadow.ewm(halflife=halflife, min_periods=halflife).mean()
    sd = shadow.ewm(halflife=halflife, min_periods=halflife).std().replace(0, np.nan)
    score = (mu / sd * np.sqrt(bpy)).shift(1).to_numpy()
    W = np.full(shadow.shape, 1.0 / m)
    w = np.full(m, 1.0 / m)
    for t in range(len(shadow)):
        if t % update_every == 0:
            sc = np.nan_to_num(score[t])
            z = np.exp((sc - sc.max()) / tau)
            w = _bound((1 - lam) * w + lam * z / z.sum(), w_min, w_max)
        W[t] = w
    return pd.DataFrame(W, index=shadow.index, columns=shadow.columns)


def ensemble_returns(pos: pd.DataFrame, W: pd.DataFrame, r: np.ndarray, cost: float,
                     threshold: float = 0.2):
    S = (W.to_numpy() * pos.to_numpy()).sum(1)
    p = np.where(np.abs(S) > threshold, np.sign(S), 0.0)
    return pd.Series(shadow_returns(p, r, cost), index=pos.index), pd.Series(p, index=pos.index)


def stats(ret: pd.Series, bpy: float) -> dict:
    mu, sd = ret.mean(), ret.std()
    cum = ret.cumsum()
    return {"ann_return": mu * bpy, "ann_vol": sd * np.sqrt(bpy),
            "sharpe": mu / sd * np.sqrt(bpy) if sd > 0 else np.nan,
            "max_drawdown": (cum - cum.cummax()).min()}


def run(df, timeframe="30min", cost=0.000035, train_bars=4000, step_bars=1000, smooth=True,
        strategies=None, **weight_kw):
    bpy = bars_per_year(timeframe)
    pos, shadow, r, chosen = walk_forward(df, strategies or STRATEGIES, cost, train_bars, step_bars, smooth)
    W = adaptive_weights(shadow, bpy, **weight_kw)
    adaptive, p_ad = ensemble_returns(pos, W, r, cost)
    W_eq = W.copy()
    W_eq[:] = 1.0 / W.shape[1]
    equal, _ = ensemble_returns(pos, W_eq, r, cost)
    table = pd.DataFrame({**{c: stats(shadow[c], bpy) for c in shadow},
                          "EQUAL-WEIGHT": stats(equal, bpy),
                          "ADAPTIVE": stats(adaptive, bpy)}).T
    return {"table": table, "weights": W, "shadow": shadow, "positions": pos,
            "ensemble_position": p_ad, "adaptive": adaptive, "equal": equal, "chosen": chosen}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="glob for histdata M1 csv files")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tf", default="30min")
    ap.add_argument("--cost", type=float, default=0.000035)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--fix-rsi", action="store_true", help="use the corrected RSI instead of the repo version")
    a = ap.parse_args()
    data = synthetic_data(timeframe=a.tf) if a.synthetic else load_histdata(a.data, a.tf)
    strats = [RSI_FIXED if x.name == "RSI" else x for x in STRATEGIES] if a.fix_rsi else STRATEGIES
    out = run(data, a.tf, a.cost, a.train, a.step, strategies=strats)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(out["table"])
    print("\nAverage weights:\n", out["weights"].mean().round(3).to_string())
    out["chosen"].to_csv("chosen_parameters.csv", index=False)
    out["weights"].to_csv("weights_history.csv")