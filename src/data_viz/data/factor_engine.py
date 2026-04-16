"""
FactorEngine — monthly long-short and long-only decile factor portfolios.

Factors
-------
momentum  : 12-month trailing return excl. last month (from ret_pivot)
low_vol   : −1 × rolling 12-month realised annualised vol (low vol = good)
value     : book-to-market ratio (bm)
quality   : composite z-score: roe + npm + cfm×fcf_ocf − de_ratio
growth    : year-on-year change in roa

Portfolio construction
----------------------
* Universe    : all stocks with valid data at the monthly rebalancing date.
* Rebalancing : monthly (last trading day of each month).
* Deciles     : cross-sectional rank → D10 (top 10 %) and D1 (bottom 10 %).
* Z-scores    : winsorised at 2nd / 98th cross-sectional percentile before ranking.
* LO (Long-Only) : equal-weighted D10 portfolio.
* L/S (Long-Short): equal-weighted D10 − D1 (net exposure = 0).
* TC (Transaction costs): 10 bps on actual turnover.
  Formula: tc = (entries + exits) / N × 0.001
  At inception: 100 % of positions are entered → tc = 10 bps.

Benchmarks
----------
* EW    : equal-weighted return of the full universe (for LO comparison).
* CW    : cap-weighted return of the full universe (for LO comparison).
* Tbill : monthly 3-month T-bill return (for L/S Sharpe & excess-return metrics).
          Falls back to 0 % if data not provided.

Performance metrics (per strategy)
-----------------------------------
* Ann. Return  : (1 + total_ret)^(12/n) − 1
* Ann. Vol     : monthly std × √12
* Sharpe       : (ann_ret − ann_rf) / ann_vol          [rf = T-bill]
* Beta         : regression of strategy on EW market   [shows market neutrality for L/S]
* Alpha        : Jensen's CAPM alpha vs EW market + rf
* IR           : active_mean / active_std × √12        [active = strategy − benchmark]
                 For LO: benchmark = EW or CW.
                 For L/S: benchmark = T-bill (active = excess return over rf).
* Max Drawdown : worst peak-to-trough decline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
_TD_YEAR:  int = 252   # approximate trading days in a year
_TD_MONTH: int = 21    # approximate trading days in a month

_QUALITY_COMPONENTS: list[str] = ["roe", "npm", "de_ratio"]
_QUALITY_FCF_COLS:   list[str] = ["cfm", "fcf_ocf"]

# Display metadata per factor: (raw column name, display unit label)
FACTOR_META: dict[str, tuple[str, str]] = {
    "momentum": ("momentum_raw", "12M ret"),
    "low_vol":  ("low_vol_raw",  "Ann. vol"),
    "value":    ("value_raw",    "B/M"),
    "quality":  ("quality_raw",  "comp. z"),
    "growth":   ("growth_raw",   "ΔROA pp"),
    "multi":    ("multi_raw",    "EW avg z"),
}

_PERF_METRIC_LABELS: dict[str, str] = {
    "ann_ret": "Ann. Return",
    "ann_vol": "Ann. Vol",
    "sharpe":  "Sharpe",
    "beta":    "Beta (EW mkt)",
    "alpha":   "Alpha",
    "ir":      "Info Ratio",
    "max_dd":  "Max Drawdown",
}


# ---------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------

def _winsorize(s: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """Clip series at given quantiles (cross-sectional winsorisation)."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; returns all-NaN if std ≈ 0."""
    std = s.std()
    if pd.isna(std) or std < 1e-10:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def _zscore_win(s: pd.Series) -> pd.Series:
    """Z-score then winsorise at 2–98 %."""
    return _winsorize(_zscore(s))


def _annualise_ret(total_ret: float, n_months: int) -> float:
    if n_months <= 0:
        return float("nan")
    return float((1.0 + total_ret) ** (12.0 / n_months) - 1.0)


def _annualise_vol(monthly_std: float) -> float:
    return float(monthly_std * np.sqrt(12))


def _max_drawdown(monthly_rets: pd.Series) -> float:
    """Worst peak-to-trough drawdown from cumulative return series."""
    cum = (1.0 + monthly_rets.fillna(0)).cumprod()
    rolling_peak = cum.cummax()
    dd = (cum - rolling_peak) / rolling_peak
    return float(dd.min()) if not dd.empty else float("nan")


def compute_performance_metrics(
    monthly_rets:  pd.Series,
    ew_market_rets: pd.Series,
    rf_rets:        pd.Series,
    ir_benchmark_rets: pd.Series,
) -> dict[str, float]:
    """
    Compute a standard suite of performance metrics.

    Parameters
    ----------
    monthly_rets       : strategy monthly returns
    ew_market_rets     : equal-weighted market returns (used for Beta & Alpha)
    rf_rets            : monthly risk-free returns, e.g. T-bill (used for Sharpe)
    ir_benchmark_rets  : benchmark for IR computation
                         (EW/CW for LO strategies; T-bill for L/S strategies)
    """
    nan = float("nan")
    empty = dict(ann_ret=nan, ann_vol=nan, sharpe=nan,
                 beta=nan, alpha=nan, ir=nan, max_dd=nan)

    # Align on common index
    df = pd.concat(
        [monthly_rets.rename("s"),
         ew_market_rets.rename("mkt"),
         rf_rets.rename("rf"),
         ir_benchmark_rets.rename("ir_bm")],
        axis=1,
    ).dropna()
    if len(df) < 3:
        return empty

    s, mkt, rf, ir_bm = df["s"], df["mkt"], df["rf"], df["ir_bm"]
    n = len(s)

    total_ret  = float((1 + s).prod() - 1)
    ann_ret    = _annualise_ret(total_ret, n)
    ann_vol    = _annualise_vol(float(s.std()))

    # Sharpe vs T-bill
    ann_rf   = _annualise_ret(float((1 + rf).prod() - 1), n)
    sharpe   = (ann_ret - ann_rf) / ann_vol if ann_vol > 1e-10 else nan

    # Beta & Alpha vs EW market (CAPM)
    cov_mat  = np.cov(s.values, mkt.values)
    beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 1e-10 else nan
    ann_mkt  = _annualise_ret(float((1 + mkt).prod() - 1), n)
    alpha    = ann_ret - (ann_rf + (beta if not np.isnan(beta) else 0.0) * (ann_mkt - ann_rf))

    # IR vs ir_benchmark
    active     = s - ir_bm
    active_std = float(active.std())
    ir = float(active.mean()) / active_std * np.sqrt(12) if active_std > 1e-10 else nan

    max_dd = _max_drawdown(s)

    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        beta=beta, alpha=alpha, ir=ir, max_dd=max_dd,
    )


# ---------------------------------------------------------------
# FactorEngine
# ---------------------------------------------------------------

class FactorEngine:
    """
    Builds and caches monthly and daily factor portfolio returns (LO and L/S).

    Attributes populated after build()
    ------------------------------------
    monthly_lo_returns    : DataFrame (months × factors)  — LO net of TC
    monthly_ls_returns    : DataFrame (months × factors)  — L/S net of TC
    monthly_benchmarks    : DataFrame (months × [ew, cw, tbill])
    daily_lo_returns      : DataFrame (trading days × factors)  — LO net of TC
    daily_ls_returns      : DataFrame (trading days × factors)  — L/S net of TC
    daily_benchmarks      : DataFrame (trading days × [ew, cw, tbill])
    factor_scores_by_month: dict{date → DataFrame(permno × scores+raw+meta)}
    """

    FACTORS: list[str] = ["momentum", "low_vol", "value", "quality", "growth", "multi"]
    _BASE_FACTORS: list[str] = ["momentum", "low_vol", "value", "quality", "growth"]

    def __init__(self, dm) -> None:       # dm: DataManager (avoid circular import)
        self.dm = dm
        self.monthly_lo_returns:     pd.DataFrame | None = None
        self.monthly_ls_returns:     pd.DataFrame | None = None
        self.monthly_benchmarks:     pd.DataFrame | None = None
        self.daily_lo_returns:       pd.DataFrame | None = None
        self.daily_ls_returns:       pd.DataFrame | None = None
        self.daily_benchmarks:       pd.DataFrame | None = None
        self.factor_scores_by_month: dict[pd.Timestamp, pd.DataFrame] = {}

    # ----------------------------------------------------------
    # Public entry point
    # ----------------------------------------------------------

    def build(self) -> None:
        """Pre-compute all factor portfolios. Heavy — call once at startup."""
        from data_viz.dashboard.market_overview import _get_dates_for_granularity

        monthly_dates = _get_dates_for_granularity("monthly", self.dm.dates)
        tbill_monthly = self.dm.tbill_monthly  # pd.Series or None

        logger.info("FactorEngine: building %d monthly periods…", len(monthly_dates) - 1)

        funda_snaps = self._precompute_funda_snaps(monthly_dates)

        lo_rows:  list[dict] = []
        ls_rows:  list[dict] = []
        bm_rows:  list[dict] = []

        # Track previous portfolio sets for transaction-cost computation
        prev_lo:  dict[str, set[int]] = {f: set() for f in self.FACTORS}
        prev_d10: dict[str, set[int]] = {f: set() for f in self.FACTORS}
        prev_d1:  dict[str, set[int]] = {f: set() for f in self.FACTORS}

        for i in range(len(monthly_dates) - 1):
            t0 = monthly_dates[i]       # rebalancing date
            t1 = monthly_dates[i + 1]   # end of holding period

            scores = self._compute_scores(t0, funda_snaps.get(t0))
            if scores.empty:
                continue
            self.factor_scores_by_month[t0] = scores

            stock_rets = self._stock_returns_between(t0, t1)
            if stock_rets.empty:
                continue

            lo_row: dict = {"date": t1}
            ls_row: dict = {"date": t1}

            for factor in self.FACTORS:
                if factor not in scores.columns:
                    lo_row[factor] = ls_row[factor] = np.nan
                    continue

                f_scores = scores[factor].dropna()
                if len(f_scores) < 20:
                    lo_row[factor] = ls_row[factor] = np.nan
                    continue

                n10 = max(1, len(f_scores) // 10)
                ranked = f_scores.sort_values()
                d1_set  = set(ranked.head(n10).index.astype(int))
                d10_set = set(ranked.tail(n10).index.astype(int))

                # --- LO: D10 mean return minus TC ---
                d10_rets = stock_rets.reindex(list(d10_set)).dropna()
                lo_gross = float(d10_rets.mean()) if not d10_rets.empty else np.nan
                tc_lo    = self._tc(prev_lo[factor], d10_set)
                lo_row[factor] = lo_gross - tc_lo if not np.isnan(lo_gross) else np.nan

                # --- L/S: (D10 − D1) mean return minus TC ---
                d1_rets  = stock_rets.reindex(list(d1_set)).dropna()
                if d10_rets.empty or d1_rets.empty:
                    ls_row[factor] = np.nan
                else:
                    ls_gross = float(d10_rets.mean()) - float(d1_rets.mean())
                    tc_d10   = self._tc(prev_d10[factor], d10_set)
                    tc_d1    = self._tc(prev_d1[factor],  d1_set)
                    tc_ls    = (tc_d10 + tc_d1) / 2.0
                    ls_row[factor] = ls_gross - tc_ls

                # Update state
                prev_lo[factor]  = d10_set
                prev_d10[factor] = d10_set
                prev_d1[factor]  = d1_set

            lo_rows.append(lo_row)
            ls_rows.append(ls_row)

            # --- Benchmarks ---
            univ_rets = stock_rets.reindex(scores.index).dropna()
            ew = float(univ_rets.mean()) if not univ_rets.empty else np.nan
            cw = np.nan
            if not univ_rets.empty and "market_cap" in scores.columns:
                mcap = scores["market_cap"].reindex(univ_rets.index).fillna(0.0)
                total_mcap = float(mcap.sum())
                if total_mcap > 0:
                    cw = float((univ_rets * mcap).sum() / total_mcap)

            # T-bill: most recent monthly rate available at t1
            tbill = 0.0
            if tbill_monthly is not None:
                avail = tbill_monthly[tbill_monthly.index <= t1]
                if not avail.empty:
                    tbill = float(avail.iloc[-1])

            bm_rows.append({"date": t1, "ew": ew, "cw": cw, "tbill": tbill})

        self.monthly_lo_returns = (
            pd.DataFrame(lo_rows).set_index("date") if lo_rows else pd.DataFrame()
        )
        self.monthly_ls_returns = (
            pd.DataFrame(ls_rows).set_index("date") if ls_rows else pd.DataFrame()
        )
        self.monthly_benchmarks = (
            pd.DataFrame(bm_rows).set_index("date") if bm_rows else pd.DataFrame()
        )
        logger.info(
            "FactorEngine build complete. LO shape: %s, L/S shape: %s",
            self.monthly_lo_returns.shape,
            self.monthly_ls_returns.shape,
        )
        self._build_daily_returns(monthly_dates, tbill_monthly)

    # ----------------------------------------------------------
    # Daily return builder
    # ----------------------------------------------------------

    def _build_daily_returns(
        self,
        monthly_dates: list[pd.Timestamp],
        tbill_monthly: "pd.Series | None",
    ) -> None:
        """
        Compute daily portfolio returns for every factor strategy.

        Within each holding period (t0, t1] the portfolio weights are fixed
        (monthly rebalancing). Daily EW returns of D10 (LO) and D10−D1 (L/S)
        are computed from ret_pivot. Transaction costs are charged as a one-time
        deduction on the first trading day of each new period.

        Populates:
            self.daily_lo_returns   — DataFrame(trading days × factors)
            self.daily_ls_returns   — DataFrame(trading days × factors)
            self.daily_benchmarks   — DataFrame(trading days × [ew, cw, tbill])
        """
        rp = self.dm.ret_pivot
        if rp is None or rp.empty or not self.factor_scores_by_month:
            logger.warning("FactorEngine: cannot build daily returns — ret_pivot or scores missing.")
            return

        logger.info("FactorEngine: building daily returns…")

        lo_daily:  dict[str, list[float]] = {f: [] for f in self.FACTORS}
        ls_daily:  dict[str, list[float]] = {f: [] for f in self.FACTORS}
        ew_daily:  list[float] = []
        cw_daily:  list[float] = []
        tb_daily:  list[float] = []
        dates_out: list[pd.Timestamp] = []

        # Re-derive D10/D1 and TC tracking across periods
        prev_lo:  dict[str, set[int]] = {f: set() for f in self.FACTORS}
        prev_d10: dict[str, set[int]] = {f: set() for f in self.FACTORS}
        prev_d1:  dict[str, set[int]] = {f: set() for f in self.FACTORS}

        rebal_dates = sorted(self.factor_scores_by_month.keys())

        for i, t0 in enumerate(rebal_dates):
            # Determine end of holding period = next date in monthly_dates after t0
            later = [d for d in monthly_dates if d > t0]
            if not later:
                continue
            t1 = later[0]

            scores = self.factor_scores_by_month[t0]
            period_mask = (rp.index > t0) & (rp.index <= t1)
            period_dates = rp.index[period_mask]
            if len(period_dates) == 0:
                continue

            period_ret = rp.loc[period_dates]      # shape: n_days × all permnos

            # ── Benchmarks ─────────────────────────────────────────────────────
            univ_permnos = scores.index.tolist()
            univ_daily = period_ret.reindex(columns=univ_permnos)

            ew_series = univ_daily.mean(axis=1)

            if "market_cap" in scores.columns:
                mcap = scores["market_cap"].reindex(univ_permnos).fillna(0.0).clip(lower=0.0)
                total_mcap = float(mcap.sum())
                if total_mcap > 0:
                    w = mcap / total_mcap
                    cw_series = univ_daily.fillna(0.0).mul(w, axis=1).sum(axis=1)
                else:
                    cw_series = ew_series
            else:
                cw_series = ew_series

            # T-bill: convert monthly to per-day rate  (1+r)^(1/n)−1
            n_days = len(period_dates)
            if tbill_monthly is not None:
                avail = tbill_monthly[tbill_monthly.index <= t1]
                monthly_rate = float(avail.iloc[-1]) if not avail.empty else 0.0
            else:
                monthly_rate = 0.0
            daily_rf = (1.0 + monthly_rate) ** (1.0 / n_days) - 1.0 if n_days > 0 else 0.0

            # ── Factor portfolios ───────────────────────────────────────────────
            factor_lo: dict[str, pd.Series] = {}
            factor_ls: dict[str, pd.Series] = {}

            for factor in self.FACTORS:
                if factor not in scores.columns:
                    nan_s = pd.Series(np.nan, index=period_dates)
                    factor_lo[factor] = nan_s
                    factor_ls[factor] = nan_s
                    continue

                f_scores = scores[factor].dropna()
                if len(f_scores) < 20:
                    nan_s = pd.Series(np.nan, index=period_dates)
                    factor_lo[factor] = nan_s
                    factor_ls[factor] = nan_s
                    continue

                n10 = max(1, len(f_scores) // 10)
                ranked = f_scores.sort_values()
                d1_set  = set(ranked.head(n10).index.astype(int))
                d10_set = set(ranked.tail(n10).index.astype(int))

                d10_daily_ret = period_ret.reindex(columns=list(d10_set)).mean(axis=1)
                d1_daily_ret  = period_ret.reindex(columns=list(d1_set)).mean(axis=1)

                # LO
                tc_lo = self._tc(prev_lo[factor], d10_set)
                lo_s  = d10_daily_ret.copy().astype(float)
                lo_s.iloc[0] = float(lo_s.iloc[0]) - tc_lo
                factor_lo[factor] = lo_s

                # L/S
                tc_d10 = self._tc(prev_d10[factor], d10_set)
                tc_d1  = self._tc(prev_d1[factor],  d1_set)
                tc_ls  = (tc_d10 + tc_d1) / 2.0
                ls_s   = (d10_daily_ret - d1_daily_ret).copy().astype(float)
                ls_s.iloc[0] = float(ls_s.iloc[0]) - tc_ls
                factor_ls[factor] = ls_s

                prev_lo[factor]  = d10_set
                prev_d10[factor] = d10_set
                prev_d1[factor]  = d1_set

            # ── Append rows ────────────────────────────────────────────────────
            for day_i, date in enumerate(period_dates):
                dates_out.append(date)
                ew_daily.append(float(ew_series.iloc[day_i]))
                cw_daily.append(float(cw_series.iloc[day_i]))
                tb_daily.append(daily_rf)
                for f in self.FACTORS:
                    lo_daily[f].append(float(factor_lo[f].iloc[day_i])
                                       if day_i < len(factor_lo[f]) else np.nan)
                    ls_daily[f].append(float(factor_ls[f].iloc[day_i])
                                       if day_i < len(factor_ls[f]) else np.nan)

        if not dates_out:
            logger.warning("FactorEngine: no daily return rows produced.")
            return

        self.daily_lo_returns = pd.DataFrame(lo_daily, index=dates_out)
        self.daily_ls_returns = pd.DataFrame(ls_daily, index=dates_out)
        self.daily_benchmarks = pd.DataFrame(
            {"ew": ew_daily, "cw": cw_daily, "tbill": tb_daily},
            index=dates_out,
        )
        logger.info(
            "FactorEngine daily returns ready — %d trading days.",
            len(dates_out),
        )

    # ----------------------------------------------------------
    # Score computation
    # ----------------------------------------------------------

    def _compute_scores(
        self,
        date: pd.Timestamp,
        funda_snap: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Return DataFrame indexed by permno with columns:
        - FACTORS (winsorised z-scores)
        - FACTOR_raw (raw values before z-scoring)
        - ticker, gics_sector, market_cap
        """
        universe = self.dm.get_universe_snapshot(date)
        if universe.empty:
            return pd.DataFrame()

        permnos = universe["permno"].astype("Int64").values
        rp = self.dm.ret_pivot

        z:   dict[str, pd.Series] = {}
        raw: dict[str, pd.Series] = {}

        # ---- Momentum ----
        window_full = rp.loc[:date].tail(_TD_YEAR)
        if len(window_full) >= _TD_MONTH + 1:
            window_mom = window_full.iloc[:-_TD_MONTH]
            mom_raw = (1 + window_mom.fillna(0)).prod() - 1
            valid = window_mom.notna().sum()
            mom_raw[valid == 0] = np.nan
        else:
            mom_raw = pd.Series(np.nan, index=rp.columns)
        raw["momentum_raw"] = mom_raw.reindex(permnos)
        z["momentum"]       = _zscore_win(raw["momentum_raw"])

        # ---- Low Vol (low vol = good → negate before scoring) ----
        vol_raw = window_full.std() * np.sqrt(_TD_YEAR)
        raw["low_vol_raw"] = vol_raw.reindex(permnos)      # positive: actual vol
        z["low_vol"]       = _zscore_win(-raw["low_vol_raw"])

        # ---- Fundamental factors ----
        if funda_snap is not None and not funda_snap.empty:
            snap = funda_snap.reindex(permnos)

            # Value: book-to-market
            bm_raw = snap["bm"] if "bm" in snap.columns else pd.Series(np.nan, index=permnos)
            raw["value_raw"] = bm_raw
            z["value"]       = _zscore_win(bm_raw)

            # Quality: composite z-score of components
            q_parts: list[pd.Series] = []
            for col in _QUALITY_COMPONENTS:
                if col in snap.columns:
                    sign = -1.0 if col == "de_ratio" else 1.0
                    q_parts.append(_zscore(sign * snap[col]))
            if all(c in snap.columns for c in _QUALITY_FCF_COLS):
                q_parts.append(_zscore(snap["cfm"] * snap["fcf_ocf"]))
            if q_parts:
                quality_composite = pd.concat(q_parts, axis=1).mean(axis=1)
            else:
                quality_composite = pd.Series(np.nan, index=permnos)
            raw["quality_raw"] = quality_composite              # composite (pre-winsorised)
            z["quality"]       = _winsorize(quality_composite)  # already z-scaled, just clip

            # Growth: YoY change in ROA (percentage points)
            if "roa" in snap.columns and "roa_lag" in snap.columns:
                growth_raw = snap["roa"] - snap["roa_lag"]
            else:
                growth_raw = pd.Series(np.nan, index=permnos)
            raw["growth_raw"] = growth_raw
            z["growth"]       = _zscore_win(growth_raw)
        else:
            for f in ["value", "quality", "growth"]:
                raw[f"{f}_raw"] = pd.Series(np.nan, index=permnos)
                z[f]            = pd.Series(np.nan, index=permnos)

        # Multi-factor: equal-weighted average of the 5 base factor z-scores
        base_zscores = [z[f] for f in self._BASE_FACTORS if f in z]
        if base_zscores:
            multi_composite = pd.concat(base_zscores, axis=1).mean(axis=1)
        else:
            multi_composite = pd.Series(np.nan, index=permnos)
        raw["multi_raw"] = multi_composite
        z["multi"]       = _winsorize(multi_composite)   # already averaged z-scores, just clip

        scores = pd.DataFrame({**z, **raw}, index=permnos)
        scores.index.name = "permno"

        # Attach meta columns
        meta = universe.set_index("permno")[["ticker", "gics_sector", "market_cap"]]
        return scores.join(meta, how="left")

    # ----------------------------------------------------------
    # Fundamental snapshot helpers
    # ----------------------------------------------------------

    def _precompute_funda_snaps(
        self,
        monthly_dates: list[pd.Timestamp],
    ) -> dict[pd.Timestamp, pd.DataFrame]:
        """
        For each monthly date, retrieve the latest available fundamental
        row per stock (merge-asof style), including a 'roa_lag' column
        for the Growth factor.
        """
        ff = self.dm.funda_factors
        if ff.empty:
            return {}

        logger.info(
            "FactorEngine: pre-computing %d fundamental snapshots…", len(monthly_dates)
        )
        result: dict[pd.Timestamp, pd.DataFrame] = {}

        for date in monthly_dates:
            avail = ff[ff["public_date"] <= date]
            if avail.empty:
                result[date] = pd.DataFrame()
                continue
            latest = (
                avail.sort_values("public_date")
                .groupby("permno", sort=False)
                .last()
            )
            # ROA from ~1 year ago for the Growth factor
            one_yr_ago = date - pd.DateOffset(years=1)
            avail_lag  = ff[(ff["public_date"] <= one_yr_ago)]
            if not avail_lag.empty and "roa" in avail_lag.columns:
                lag = (
                    avail_lag.sort_values("public_date")
                    .groupby("permno", sort=False)["roa"]
                    .last()
                    .rename("roa_lag")
                )
                latest = latest.join(lag, how="left")
            result[date] = latest

        return result

    # ----------------------------------------------------------
    # Return helpers
    # ----------------------------------------------------------

    def _stock_returns_between(
        self, t0: pd.Timestamp, t1: pd.Timestamp
    ) -> pd.Series:
        """Cumulative return for each stock in (t0, t1] (exclusive t0)."""
        rp = self.dm.ret_pivot
        window = rp.loc[(rp.index > t0) & (rp.index <= t1)]
        if window.empty:
            return pd.Series(dtype=float)
        cum = (1 + window.fillna(0)).prod() - 1
        cum[window.notna().sum() == 0] = np.nan
        return cum

    @staticmethod
    def _tc(prev: set[int], curr: set[int], bps: float = 10.0) -> float:
        """
        One-way transaction cost fraction.
        tc = (entries + exits) / N × bps / 10 000
        At inception (prev empty): tc = 10 bps (pay to buy all positions).
        """
        n = max(len(curr), 1)
        if not prev:
            return bps / 10_000   # 10 bps at inception
        exits   = len(prev - curr)
        entries = len(curr - prev)
        return (exits + entries) / n * bps / 10_000

    # ----------------------------------------------------------
    # Public accessors for the dashboard
    # ----------------------------------------------------------

    def _get_combined(self, mode: str) -> pd.DataFrame:
        """Returns (returns, ew_col, ir_bm_col) DataFrame for the given mode."""
        rets = self.monthly_lo_returns if mode == "lo" else self.monthly_ls_returns
        bm   = self.monthly_benchmarks
        if rets is None or rets.empty or bm is None:
            return pd.DataFrame()
        bm_cols = ["ew", "cw"] if mode == "lo" else ["tbill"]
        extra   = bm[[c for c in bm_cols if c in bm.columns]]
        return pd.concat([rets, extra], axis=1)

    def get_cumulative_returns(
        self,
        mode: str,
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Cumulative returns rebased to 0 at start_date.
        Returns DataFrame indexed by date (includes a zero row at start_date).
        """
        combined = self._get_combined(mode)
        if combined.empty:
            return combined
        if start_date is not None:
            combined = combined.loc[combined.index > start_date]
        if combined.empty:
            return combined

        cumret = (1 + combined.fillna(0)).cumprod() - 1
        # Prepend a zero row at start_date (or one period before first available)
        zero_date = start_date if start_date is not None else combined.index[0] - pd.offsets.MonthEnd(1)
        zero_row  = pd.DataFrame(0.0, index=[zero_date], columns=cumret.columns)
        return pd.concat([zero_row, cumret])

    def get_daily_cumulative_returns(
        self,
        mode: str,
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Cumulative returns at daily granularity, rebased to 0 at start_date.

        Within each holding period the portfolio weights (D10 for LO, D10−D1
        for L/S) are fixed at the monthly rebalancing date. Daily EW returns of
        these baskets are computed from ret_pivot. TC is charged on day 1 of each
        new period.

        Falls back to monthly data if daily data is not available.
        """
        rets = self.daily_lo_returns if mode == "lo" else self.daily_ls_returns
        bm   = self.daily_benchmarks
        if rets is None or rets.empty or bm is None or bm.empty:
            logger.warning("FactorEngine: daily returns not available, falling back to monthly.")
            return self.get_cumulative_returns(mode, start_date)

        bm_cols = ["ew", "cw"] if mode == "lo" else ["tbill"]
        combined = pd.concat([rets, bm[[c for c in bm_cols if c in bm.columns]]], axis=1)

        if start_date is not None:
            combined = combined.loc[combined.index > start_date]
        if combined.empty:
            return combined

        cumret    = (1 + combined.fillna(0)).cumprod() - 1
        zero_date = start_date if start_date is not None else combined.index[0] - pd.offsets.BDay(1)
        zero_row  = pd.DataFrame(0.0, index=[zero_date], columns=cumret.columns)
        return pd.concat([zero_row, cumret])

    def get_rolling_metric(
        self,
        mode: str,
        window_months: int,
        metric: str = "return",
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Trailing rolling metric over a window of `window_months` months.

        Parameters
        ----------
        metric : "return" | "vol" | "sharpe"
            * return : annualised compounded return
            * vol    : annualised realised volatility
            * sharpe : annualised excess return / annualised vol  (rf = T-bill)
        """
        combined = self._get_combined(mode)
        bm = self.monthly_benchmarks
        if combined.empty:
            return combined
        if start_date is not None:
            combined = combined.loc[combined.index > start_date]

        if metric == "vol":
            result = combined.rolling(window_months).std() * np.sqrt(12)

        elif metric == "sharpe":
            rf = (
                bm["tbill"].reindex(combined.index).fillna(0)
                if bm is not None and "tbill" in bm.columns
                else pd.Series(0.0, index=combined.index)
            )
            excess   = combined.sub(rf, axis=0)
            roll_ret = excess.rolling(window_months).mean() * 12
            roll_vol = combined.rolling(window_months).std() * np.sqrt(12)
            result   = roll_ret / roll_vol.replace(0.0, np.nan)

        else:  # "return"
            def _roll_ann(col: pd.Series) -> pd.Series:
                return col.rolling(window_months).apply(
                    lambda x: float(
                        (1 + np.nan_to_num(x, nan=0.0)).prod() ** (12.0 / window_months) - 1
                    ),
                    raw=True,
                )
            result = combined.apply(_roll_ann)

        return result.dropna(how="all")

    def get_performance_table(
        self,
        mode: str,
        start_date: pd.Timestamp | None = None,
        ir_benchmark: str = "ew",
    ) -> pd.DataFrame:
        """
        Performance metrics for all factors + benchmarks.

        Parameters
        ----------
        mode          : "lo" or "ls"
        start_date    : restrict to returns after this date
        ir_benchmark  : "ew" or "cw" — benchmark used for IR in LO mode.
                        Ignored for L/S mode (IR computed vs T-bill).
        """
        rets = self.monthly_lo_returns if mode == "lo" else self.monthly_ls_returns
        bm   = self.monthly_benchmarks
        if rets is None or rets.empty or bm is None:
            return pd.DataFrame()

        if start_date is not None:
            rets = rets.loc[rets.index > start_date]
            bm   = bm.loc[bm.index > start_date]

        ew_mkt = bm.get("ew", pd.Series(dtype=float))
        rf     = bm.get("tbill", pd.Series(0.0, index=bm.index))

        if mode == "lo":
            ir_bm_series = bm.get(ir_benchmark, ew_mkt)
        else:
            ir_bm_series = rf  # L/S: active return = excess return over T-bill

        rows = []
        for factor in self.FACTORS:
            if factor not in rets.columns:
                continue
            metrics = compute_performance_metrics(rets[factor], ew_mkt, rf, ir_bm_series)
            rows.append({"Strategy": factor, **metrics})

        # Benchmark rows
        bm_to_show = ["ew", "cw"] if mode == "lo" else ["tbill"]
        for col in bm_to_show:
            if col not in bm.columns:
                continue
            metrics = compute_performance_metrics(bm[col], ew_mkt, rf, ir_bm_series)
            rows.append({"Strategy": f"{col.upper()} benchmark", **metrics})

        return pd.DataFrame(rows).set_index("Strategy")

    def get_factor_heatmap_data(
        self,
        mode: str,
        granularity: str = "monthly",
        metric: str = "return",
        window: int = 12,
    ) -> pd.DataFrame:
        """
        Returns DataFrame suitable for a heatmap: rows = periods, columns = factors.

        Parameters
        ----------
        mode        : "lo" or "ls"
        granularity : "monthly" | "yearly"
        metric      : "return"      — raw monthly/yearly return (no rolling)
                      "roll_return" — trailing N-month annualised return
                      "vol"         — trailing N-month annualised vol
                      "sharpe"      — trailing N-month annualised Sharpe
        window      : rolling window in months (used for roll_return / vol / sharpe)
        """
        rets = self.monthly_lo_returns if mode == "lo" else self.monthly_ls_returns
        bm   = self.monthly_benchmarks
        if rets is None or rets.empty:
            return pd.DataFrame()

        rf_s = (
            bm["tbill"].reindex(rets.index).fillna(0)
            if bm is not None and "tbill" in bm.columns
            else pd.Series(0.0, index=rets.index)
        )

        if metric == "vol":
            df = rets.rolling(window).std() * np.sqrt(12)

        elif metric == "sharpe":
            excess    = rets.sub(rf_s, axis=0)
            roll_mean = excess.rolling(window).mean() * 12
            roll_vol  = rets.rolling(window).std() * np.sqrt(12)
            df = roll_mean / roll_vol.replace(0.0, np.nan)

        elif metric == "roll_return":
            def _roll_ann(col: pd.Series) -> pd.Series:
                return col.rolling(window).apply(
                    lambda x: float(
                        (1 + np.nan_to_num(x, nan=0.0)).prod() ** (12.0 / window) - 1
                    ),
                    raw=True,
                )
            df = rets.apply(_roll_ann)

        elif metric == "vol_raw":
            # Single-period annualised vol proxy: |r_t| × √12
            df = rets.abs() * np.sqrt(12)

        elif metric == "sharpe_raw":
            # Non-rolling Sharpe: (r_t − rf_t) × 12 / trailing_12M_vol
            excess = rets.sub(rf_s, axis=0)
            trailing_vol = rets.rolling(12).std() * np.sqrt(12)
            df = (excess * 12) / trailing_vol.replace(0.0, np.nan)

        else:  # "return" — raw monthly, no rolling
            df = rets.copy()

        df.index = pd.DatetimeIndex(df.index)

        if granularity == "yearly":
            period = df.index.to_period("Y")
            if metric == "return":
                # Compound monthly returns within each calendar year
                grouped = df.groupby(period).apply(
                    lambda grp: (1 + grp.fillna(0)).prod() - 1
                )
            elif metric == "vol_raw":
                # Realized vol within each calendar year: std(monthly rets) × √12
                grouped = rets.groupby(rets.index.to_period("Y")).std() * np.sqrt(12)
            elif metric == "sharpe_raw":
                # Annual Sharpe: (compounded ret − compounded rf) / realized vol
                p_r  = rets.index.to_period("Y")
                p_rf = rf_s.index.to_period("Y")
                ret_ann = rets.groupby(p_r).apply(lambda grp: (1 + grp.fillna(0)).prod() - 1)
                rf_ann  = rf_s.groupby(p_rf).apply(lambda grp: (1 + grp.fillna(0)).prod() - 1)
                vol_ann = rets.groupby(p_r).std() * np.sqrt(12)
                grouped = ret_ann.sub(rf_ann, axis=0) / vol_ann.replace(0.0, np.nan)
            else:
                # Average the rolling/computed values within each year
                grouped = df.groupby(period).mean()
            grouped.index = grouped.index.to_timestamp(how="end")
            return grouped.dropna(how="all") if not grouped.empty else df

        return df.dropna(how="all")

    def get_annual_metrics(
        self,
        mode: str,
        metric: str = "ann_ret",
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Calendar-year performance metrics for all strategies + benchmarks.

        Parameters
        ----------
        mode       : "lo" or "ls"
        metric     : "ann_ret" | "ann_vol" | "sharpe"
        start_date : restrict to returns after this date
        """
        rets = self.monthly_lo_returns if mode == "lo" else self.monthly_ls_returns
        bm   = self.monthly_benchmarks
        if rets is None or rets.empty:
            return pd.DataFrame()

        if start_date is not None:
            rets = rets.loc[rets.index > start_date]
            bm   = bm.loc[bm.index > start_date] if bm is not None else None

        rf_s = (
            bm["tbill"].reindex(rets.index).fillna(0)
            if bm is not None and "tbill" in bm.columns
            else pd.Series(0.0, index=rets.index)
        )

        # Append benchmark columns
        all_df = rets.copy()
        bm_cols_show = ["ew", "cw"] if mode == "lo" else ["tbill"]
        if bm is not None:
            for col in bm_cols_show:
                if col in bm.columns:
                    all_df[col] = bm[col].reindex(rets.index).fillna(0)

        period = all_df.index.to_period("Y")

        if metric == "ann_ret":
            result = all_df.groupby(period).apply(
                lambda grp: (1 + grp.fillna(0)).prod() - 1
            )
        elif metric == "ann_vol":
            result = all_df.groupby(period).std() * np.sqrt(12)
        elif metric == "sharpe":
            ret_ann = all_df.groupby(period).apply(
                lambda grp: (1 + grp.fillna(0)).prod() - 1
            )
            rf_ann  = rf_s.groupby(rf_s.index.to_period("Y")).apply(
                lambda grp: (1 + grp.fillna(0)).prod() - 1
            )
            vol_ann = all_df.groupby(period).std() * np.sqrt(12)
            result  = ret_ann.sub(rf_ann, axis=0) / vol_ann.replace(0.0, np.nan)
        else:
            return pd.DataFrame()

        result.index = result.index.to_timestamp(how="end")
        return result.dropna(how="all")

    def get_sector_scores(
        self,
        date: pd.Timestamp | None = None,
        sort_sectors: str = "alpha",
        sort_factors: str = "alpha",
    ) -> pd.DataFrame:
        """
        Average factor z-score per GICS sector at the nearest available month.
        Returns DataFrame: rows = sectors, columns = factors.
        """
        if not self.factor_scores_by_month:
            return pd.DataFrame()

        # Snap to nearest available monthly date
        available = sorted(self.factor_scores_by_month.keys())
        if date is None:
            snap_date = available[-1]
        else:
            snap_date = min(available, key=lambda d: abs((d - date).days))

        scores = self.factor_scores_by_month[snap_date]
        if scores.empty or "gics_sector" not in scores.columns:
            return pd.DataFrame()

        result = (
            scores[self.FACTORS + ["gics_sector"]]
            .groupby("gics_sector")[self.FACTORS]
            .mean()
        )

        # Sort sectors
        if sort_sectors == "avg_score":
            result = result.loc[result.mean(axis=1).sort_values(ascending=False).index]

        # Sort factors
        if sort_factors == "avg_score":
            result = result[result.mean(axis=0).sort_values(ascending=False).index.tolist()]

        return result

    def get_ranked_stocks(
        self,
        factor: str,
        date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        All stocks ranked by the selected factor at the nearest available month.
        Returns DataFrame with z-score, raw value, and meta columns.
        """
        if not self.factor_scores_by_month:
            return pd.DataFrame()

        available = sorted(self.factor_scores_by_month.keys())
        if date is None:
            snap_date = available[-1]
        else:
            snap_date = min(available, key=lambda d: abs((d - date).days))

        scores = self.factor_scores_by_month[snap_date].copy()
        if scores.empty or factor not in scores.columns:
            return pd.DataFrame()

        n10 = max(1, scores[factor].dropna().__len__() // 10)

        scores = scores.reset_index().sort_values(factor, ascending=False, na_position="last")
        scores["rank"] = range(1, len(scores) + 1)

        # Label D10 / D1 / Mid
        n_valid = scores[factor].notna().sum()
        scores["decile_group"] = "Mid"
        scores.loc[scores["rank"] <= n10, "decile_group"] = "D10 (Long)"
        scores.loc[scores["rank"] > n_valid - n10, "decile_group"] = "D1 (Short)"

        return scores