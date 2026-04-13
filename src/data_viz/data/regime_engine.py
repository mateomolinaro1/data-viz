"""
Regime Engine — market regime classification.

Signals (CW daily index)
------------------------
- Trailing 252-day cumulative return   (≈ 12M annualised)
- Trailing 252-day annualised volatility

Classification
--------------
25th / 75th percentile thresholds → 3 states per axis → 9 regimes.
Factor analysis uses only the 4 *extreme* regimes (both axes non-mid).

Cross-sectional bands
---------------------
For each trading day the trailing 252-day return *and* vol are computed
at the individual-stock level; the 10th / 90th cross-sectional percentiles
(unweighted) form bands showing universe-wide dispersion.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from data_viz.data.data import DataManager
    from data_viz.data.factor_engine import FactorEngine

logger = logging.getLogger(__name__)

# ── Regime metadata ───────────────────────────────────────────────────────────

REGIME_NAMES: dict[tuple[str, str], str] = {
    ("High", "High"): "High Mom–High Vol",
    ("High", "Mid"):  "High Mom–Mid Vol",
    ("High", "Low"):  "High Mom–Low Vol",
    ("Mid",  "High"): "Mid Mom–High Vol",
    ("Mid",  "Mid"):  "Mid Mom–Mid Vol",
    ("Mid",  "Low"):  "Mid Mom–Low Vol",
    ("Low",  "High"): "Low Mom–High Vol",
    ("Low",  "Mid"):  "Low Mom–Mid Vol",
    ("Low",  "Low"):  "Low Mom–Low Vol",
}

# 4 pure extreme regimes — used in all factor analysis
EXTREME_REGIMES: list[str] = [
    "High Mom–High Vol",
    "High Mom–Low Vol",
    "Low Mom–High Vol",
    "Low Mom–Low Vol",
]

REGIME_COLORS: dict[str, str] = {
    "High Mom–High Vol": "#4C72B0",   # blue   — recovery / melt-up
    "High Mom–Low Vol":  "#55A868",   # green  — goldilocks
    "Low Mom–High Vol":  "#C44E52",   # red    — crisis
    "Low Mom–Low Vol":   "#DD8452",   # orange — quiet bear
    # Mid regimes — muted
    "High Mom–Mid Vol":  "#b8cce0",
    "Mid Mom–High Vol":  "#e0b8b8",
    "Mid Mom–Low Vol":   "#c4dfc4",
    "Low Mom–Mid Vol":   "#e8ccb0",
    "Mid Mom–Mid Vol":   "#d0d0d0",
}

_ALL_REGIMES: list[str] = list(REGIME_NAMES.values())

_PERF_METRIC_LABELS = {
    "ann_ret": "Ann. Return",
    "ann_vol": "Ann. Vol",
    "sharpe":  "Sharpe",
}


class RegimeEngine:
    """
    Computes daily market regime labels and derived analytics.

    Usage
    -----
    engine = RegimeEngine(data_manager)
    engine.build()
    """

    def __init__(self, data_manager: "DataManager") -> None:
        self.dm = data_manager

        # Built by build()
        self.cw_daily_ret:  pd.Series    | None = None
        self.trailing_ret:  pd.Series    | None = None   # 12M trailing cum. return
        self.trailing_vol:  pd.Series    | None = None   # 12M trailing ann. vol
        self.regime_series: pd.Series    | None = None   # daily regime labels (str)
        self.cs_bands:      pd.DataFrame | None = None   # ret/vol 10-90 pct bands

        # Thresholds (float)
        self.ret_q25: float | None = None
        self.ret_q75: float | None = None
        self.vol_q25: float | None = None
        self.vol_q75: float | None = None

    # ── Public ───────────────────────────────────────────────────────────────

    def build(self) -> None:
        logger.info("RegimeEngine: starting build…")
        self._compute_cw_daily_returns()
        self._compute_trailing_signals()
        self._compute_thresholds()
        self._classify_regimes()
        self._compute_cross_sectional_bands()
        logger.info("RegimeEngine: build complete.")

    # ── Regime series helpers ─────────────────────────────────────────────────

    def get_regime_at_frequency(self, freq: str) -> pd.Series:
        """
        Resample daily regime to the requested display frequency.

        freq : "daily" | "weekly" | "monthly" | "yearly"
        Aggregation: mode (most common label) within each period.
        """
        if self.regime_series is None:
            return pd.Series(dtype=object)
        _map = {"daily": "D", "weekly": "W", "monthly": "M", "yearly": "Y"}
        pd_freq = _map.get(freq, "M")
        if pd_freq == "D":
            return self.regime_series.dropna()
        return (
            self.regime_series.dropna()
            .resample(pd_freq)
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
            .dropna()
        )

    def get_monthly_regime(self) -> pd.Series:
        """Month-end regime label (mode within month). Aligns with monthly factor returns."""
        return self.get_regime_at_frequency("monthly")

    # ── Transition matrix ─────────────────────────────────────────────────────

    def get_transition_matrix(
        self,
        end_date: pd.Timestamp | None = None,
        window_months: int | None = None,
        extreme_only: bool = True,
    ) -> pd.DataFrame:
        """
        Empirical regime transition probability matrix.

        Parameters
        ----------
        end_date      : last date to include (defaults to full series)
        window_months : trailing window length; None = full history up to end_date
        extreme_only  : True → 4×4 (extreme regimes only)
        """
        if self.regime_series is None:
            return pd.DataFrame()

        regimes = self.regime_series.dropna()

        if end_date is not None:
            regimes = regimes.loc[regimes.index <= end_date]
        if window_months is not None and len(regimes):
            cutoff = regimes.index[-1] - pd.DateOffset(months=window_months)
            regimes = regimes.loc[regimes.index >= cutoff]

        states = EXTREME_REGIMES if extreme_only else _ALL_REGIMES
        if extreme_only:
            regimes = regimes[regimes.isin(EXTREME_REGIMES)]

        if len(regimes) < 2:
            return pd.DataFrame(0.0, index=states, columns=states)

        frm = regimes.iloc[:-1].values
        to  = regimes.iloc[1:].values

        counts = pd.DataFrame(0.0, index=states, columns=states)
        for f, t in zip(frm, to):
            if f in counts.index and t in counts.columns:
                counts.loc[f, t] += 1

        row_sums = counts.sum(axis=1).replace(0.0, np.nan)
        return counts.div(row_sums, axis=0).fillna(0.0)

    @property
    def data_start(self) -> pd.Timestamp | None:
        """First date with valid regime signal (after warm-up period)."""
        if self.trailing_ret is None:
            return None
        return self.trailing_ret.first_valid_index()

    def get_regime_stats(
        self,
        end_date: pd.Timestamp | None = None,
        window_months: int | None = None,
    ) -> pd.DataFrame:
        """
        Summary stats per regime for the selected window (all 9 regimes present in data).

        Returns DataFrame: index = regime names, columns = [freq_pct, avg_duration_days]
        """
        if self.regime_series is None:
            return pd.DataFrame()

        regimes = self.regime_series.dropna()
        if end_date is not None:
            regimes = regimes.loc[regimes.index <= end_date]
        if window_months is not None and len(regimes):
            cutoff = regimes.index[-1] - pd.DateOffset(months=window_months)
            regimes = regimes.loc[regimes.index >= cutoff]

        total = len(regimes)

        rows = []
        for r in _ALL_REGIMES:
            cnt = (regimes == r).sum()
            if cnt == 0:
                continue
            freq = cnt / total if total else np.nan

            # Average run length (days)
            in_regime = (regimes == r).astype(int)
            runs      = (in_regime.diff() != 0).cumsum()
            run_lens  = in_regime.groupby(runs).sum()
            avg_dur   = float(run_lens[run_lens > 0].mean()) if (run_lens > 0).any() else np.nan

            rows.append({"Regime": r, "% of time": freq, "Avg duration (days)": avg_dur})

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("Regime")

    def get_rolling_transition_timeseries(
        self,
        window_months: int = 24,
    ) -> pd.DataFrame:
        """
        Rolling 4×4 transition probability time series.

        Returns DataFrame: index = month-end dates,
        columns = MultiIndex(from_regime, to_regime) — up to 16 columns.
        """
        if self.regime_series is None:
            return pd.DataFrame()

        monthly_idx = self.regime_series.resample("M").last().index
        results: dict = {}
        for date in monthly_idx:
            tm = self.get_transition_matrix(
                end_date=date,
                window_months=window_months,
                extreme_only=True,
            )
            if not tm.empty:
                row_data: dict[tuple, float] = {}
                for f in EXTREME_REGIMES:
                    for t in EXTREME_REGIMES:
                        if f in tm.index and t in tm.columns:
                            row_data[(f, t)] = float(tm.loc[f, t])
                if row_data:
                    results[date] = row_data

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).T
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["from", "to"])
        return df.dropna(how="all")

    def get_rolling_persistence(self, window_months: int = 24) -> pd.DataFrame:
        """
        Rolling persistence = P(regime_t+1 = r | regime_t = r) for each
        extreme regime, estimated over a trailing `window_months` window.
        One row per month-end.

        Returns DataFrame: index = month-end dates, columns = extreme regimes.
        """
        if self.regime_series is None:
            return pd.DataFrame()

        monthly_idx = (
            self.regime_series.resample("M")
            .last()
            .index
        )
        results: dict[pd.Timestamp, dict] = {}
        for date in monthly_idx:
            tm = self.get_transition_matrix(
                end_date=date,
                window_months=window_months,
                extreme_only=True,
            )
            if not tm.empty:
                results[date] = {r: tm.loc[r, r] for r in EXTREME_REGIMES if r in tm.index}

        return pd.DataFrame(results).T.dropna(how="all")

    # ── Regime-conditional factor metrics ─────────────────────────────────────

    def get_regime_factor_metrics(
        self,
        factor_engine: "FactorEngine",
        mode: str = "ls",
        metric: str = "ann_ret",
        start_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Factor performance conditional on extreme regime.

        Returns
        -------
        pd.DataFrame — index = factors (+ benchmarks), columns = extreme regimes
        """
        rets = (
            factor_engine.monthly_lo_returns
            if mode == "lo"
            else factor_engine.monthly_ls_returns
        )
        bm = factor_engine.monthly_benchmarks
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

        # Append benchmark columns to the return frame
        all_df = rets.copy()
        bm_cols = ["ew", "cw"] if mode == "lo" else ["tbill"]
        if bm is not None:
            for c in bm_cols:
                if c in bm.columns:
                    all_df[c] = bm[c].reindex(rets.index).fillna(0)
        strategy_cols = [c for c in list(factor_engine.FACTORS) + bm_cols if c in all_df.columns]

        # Align monthly regime with factor returns index
        monthly_regime = self.get_monthly_regime().reindex(all_df.index, method="ffill")

        results: dict[str, pd.Series] = {}
        for regime in EXTREME_REGIMES:
            mask   = monthly_regime == regime
            sub    = all_df.loc[mask, strategy_cols]
            rf_sub = rf_s.loc[mask]
            n      = len(sub)
            if n == 0:
                results[regime] = pd.Series(np.nan, index=strategy_cols)
                continue

            if metric == "ann_ret":
                results[regime] = (1 + sub.fillna(0)).prod() ** (12 / n) - 1
            elif metric == "ann_vol":
                results[regime] = sub.std() * np.sqrt(12)
            elif metric == "sharpe":
                excess = sub.sub(rf_sub, axis=0)
                vol_a  = sub.std() * np.sqrt(12)
                results[regime] = (excess.mean() * 12) / vol_a.replace(0, np.nan)
            else:
                results[regime] = pd.Series(np.nan, index=strategy_cols)

        # Add a "Normal (mid)" column: all months NOT in any extreme regime
        mid_mask   = ~monthly_regime.isin(EXTREME_REGIMES)
        sub_mid    = all_df.loc[mid_mask, strategy_cols]
        rf_mid     = rf_s.loc[mid_mask]
        n_mid      = len(sub_mid)
        if n_mid == 0:
            results["Normal (mid)"] = pd.Series(np.nan, index=strategy_cols)
        elif metric == "ann_ret":
            results["Normal (mid)"] = (1 + sub_mid.fillna(0)).prod() ** (12 / n_mid) - 1
        elif metric == "ann_vol":
            results["Normal (mid)"] = sub_mid.std() * np.sqrt(12)
        elif metric == "sharpe":
            excess_mid = sub_mid.sub(rf_mid, axis=0)
            vol_mid    = sub_mid.std() * np.sqrt(12)
            results["Normal (mid)"] = (excess_mid.mean() * 12) / vol_mid.replace(0, np.nan)
        else:
            results["Normal (mid)"] = pd.Series(np.nan, index=strategy_cols)

        # Column order: 4 extreme regimes then normal baseline
        col_order = EXTREME_REGIMES + ["Normal (mid)"]
        return pd.DataFrame(results).loc[strategy_cols, col_order]

    # ── Private: build steps ──────────────────────────────────────────────────

    def _compute_cw_daily_returns(self) -> None:
        ret = self.dm.ret_pivot
        nd  = self.dm.network_data

        if ret is None or ret.empty:
            logger.warning("RegimeEngine: ret_pivot empty — cannot compute CW index.")
            return

        if nd is None or nd.empty or "market_cap" not in nd.columns:
            logger.warning("RegimeEngine: market_cap unavailable — falling back to EW index.")
            self.cw_daily_ret = ret.mean(axis=1).rename("cw_ret")
            return

        logger.info("RegimeEngine: building market-cap pivot…")
        mcap = (
            nd[["date", "permno", "market_cap"]]
            .dropna(subset=["market_cap"])
            .pivot_table(index="date", columns="permno",
                         values="market_cap", aggfunc="last")
        )
        # Align to ret_pivot dates and forward-fill gaps
        mcap = mcap.reindex(ret.index).ffill()

        common  = ret.columns.intersection(mcap.columns)
        ret_c   = ret[common]
        mcap_c  = mcap[common].fillna(0.0).clip(lower=0.0)

        # Renormalise weights row-by-row; if a stock has NaN return that day,
        # exclude it from the index (adjust denominator accordingly)
        valid   = ret_c.notna()
        w_raw   = mcap_c * valid
        row_sum = w_raw.sum(axis=1).replace(0.0, np.nan)
        weights = w_raw.div(row_sum, axis=0)

        self.cw_daily_ret = (ret_c.fillna(0.0) * weights).sum(axis=1).rename("cw_ret")
        logger.info(
            "RegimeEngine: CW daily return series ready — %d trading days.",
            len(self.cw_daily_ret),
        )

    def _compute_trailing_signals(self) -> None:
        if self.cw_daily_ret is None or self.cw_daily_ret.empty:
            return
        r = self.cw_daily_ret

        # Trailing 252-day cumulative return (log-sum for numerical stability)
        log_r = np.log1p(r.fillna(0.0))
        self.trailing_ret = np.expm1(log_r.rolling(252, min_periods=200).sum())

        # Trailing 252-day annualised volatility
        self.trailing_vol = r.rolling(252, min_periods=200).std() * np.sqrt(252)
        logger.info("RegimeEngine: trailing 12M signals computed.")

    def _compute_thresholds(self) -> None:
        if self.trailing_ret is None:
            return
        self.ret_q25 = float(self.trailing_ret.quantile(0.25))
        self.ret_q75 = float(self.trailing_ret.quantile(0.75))
        self.vol_q25 = float(self.trailing_vol.quantile(0.25))
        self.vol_q75 = float(self.trailing_vol.quantile(0.75))
        logger.info(
            "RegimeEngine: thresholds  ret=[%.3f, %.3f]  vol=[%.3f, %.3f]",
            self.ret_q25, self.ret_q75, self.vol_q25, self.vol_q75,
        )

    def _classify_regimes(self) -> None:
        if self.trailing_ret is None:
            return

        q25r, q75r = self.ret_q25, self.ret_q75
        q25v, q75v = self.vol_q25, self.vol_q75

        def _mom(x: float) -> str | None:
            if np.isnan(x): return None
            if x > q75r:    return "High"
            if x < q25r:    return "Low"
            return "Mid"

        def _vol(x: float) -> str | None:
            if np.isnan(x): return None
            if x > q75v:    return "High"
            if x < q25v:    return "Low"
            return "Mid"

        mom_s = self.trailing_ret.map(_mom)
        vol_s = self.trailing_vol.map(_vol)

        self.regime_series = pd.Series(
            [REGIME_NAMES.get((m, v)) for m, v in zip(mom_s, vol_s)],
            index=self.trailing_ret.index,
            dtype=object,
        )
        logger.info(
            "RegimeEngine: regime distribution:\n%s",
            self.regime_series.value_counts().to_string(),
        )

    def _compute_cross_sectional_bands(self) -> None:
        ret = self.dm.ret_pivot
        nd  = self.dm.network_data
        if ret is None or ret.empty:
            return

        # ── Sector-level EW bands (CW market vol will lie within these) ──────
        if nd is not None and not nd.empty and "gics_sector" in nd.columns:
            sector_map = (
                nd[["permno", "gics_sector"]]
                .dropna(subset=["gics_sector"])
                .drop_duplicates("permno")
                .set_index("permno")["gics_sector"]
            )
            common = ret.columns.intersection(sector_map.index)
            if len(common) > 0:
                ret_c    = ret[common]
                sec_s    = sector_map.reindex(common)
                sec_rets = {
                    str(sec): ret_c[grp.index].mean(axis=1)
                    for sec, grp in sec_s.groupby(sec_s)
                }
                if sec_rets:
                    sector_df   = pd.DataFrame(sec_rets)
                    log_sr      = np.log1p(sector_df.fillna(0.0))
                    roll_ret_12 = np.expm1(log_sr.rolling(252, min_periods=200).sum())
                    roll_vol_12 = sector_df.rolling(252, min_periods=200).std() * np.sqrt(252)
                    self.cs_bands = pd.DataFrame({
                        "ret_p10": roll_ret_12.min(axis=1),
                        "ret_p90": roll_ret_12.max(axis=1),
                        "vol_p10": roll_vol_12.min(axis=1),
                        "vol_p90": roll_vol_12.max(axis=1),
                    })
                    logger.info(
                        "RegimeEngine: sector-level CS bands ready (%d sectors).",
                        len(sec_rets),
                    )
                    return

        # ── Fallback: individual-stock 10/90 bands ────────────────────────────
        logger.info(
            "RegimeEngine: falling back to individual-stock CS bands "
            "(%d stocks × %d days)…", ret.shape[1], ret.shape[0],
        )
        log_ret     = np.log1p(ret.fillna(0.0))
        roll_ret_12 = np.expm1(log_ret.rolling(252, min_periods=200).sum())
        roll_vol_12 = ret.rolling(252, min_periods=200).std() * np.sqrt(252)
        self.cs_bands = pd.DataFrame({
            "ret_p10": roll_ret_12.min(axis=1),
            "ret_p90": roll_ret_12.max(axis=1),
            "vol_p10": roll_vol_12.min(axis=1),
            "vol_p90": roll_vol_12.max(axis=1),
        })
        logger.info("RegimeEngine: individual-stock CS bands ready.")