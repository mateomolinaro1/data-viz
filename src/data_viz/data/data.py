"""
A data manager class to handle the data for the data visualization project.
"""
from __future__ import annotations
import numpy as np
from data_viz.utils.config import Config
import pandas as pd
from better_aws import AWS
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """
    A class to manage the data for the data visualization project.
    """

    def __init__(self, config:Config) -> None:
        """
        Initialize the DataManager with the given data.
        :param config: a Config object containing the configuration for the data manager.
        :arg: dates: list of unique dates in the data, sorted in ascending order.
        :arg: universe: a dictionary mapping each date to a list of PERMNOs present on that date.
        """

        self.config = config

        # Raw data
        self.funda_data: pd.DataFrame | None = None
        self.mkt_data: pd.DataFrame | None = None

        # Prepared / merged data
        self.network_data: pd.DataFrame | None = None

        # Time structures
        self.dates: list[pd.Timestamp] | None = None
        self.universe: dict[pd.Timestamp, list[int]] | None = None

        # Network objects
        self.ret_pivot: pd.DataFrame | None = None
        self.node_features: pd.DataFrame | None = None
        self.reference_corr_matrix: pd.DataFrame | None = None
        self.reference_layout: pd.DataFrame | None = None

        # Cache
        self._graph_snapshot_cache: dict[tuple[pd.Timestamp, float, int | None], tuple[pd.DataFrame, pd.DataFrame]] = {}
        self._graph_summary_cache: dict[tuple[pd.Timestamp, float, int | None], dict[str, float | int]] = {}
        self._node_ranking_cache: dict[tuple[pd.Timestamp, float, int | None], pd.DataFrame] = {}

        # Pre-computed time series (populated by build_graph_summary_timeseries)
        self.graph_summary_timeseries: pd.DataFrame | None = None

        # Slim fundamental data kept for FactorEngine (populated by _free_memory)
        self.funda_factors: pd.DataFrame = pd.DataFrame()

        # Monthly T-bill returns (annualized % → monthly rate, from DGS3MO)
        # None if path not configured; FactorEngine falls back to 0 % rf.
        self.tbill_monthly: pd.Series | None = None

    def load_data(self) -> None:
        """
        Load the data from the source and process it to fill the attributes.
        """
        self._connect_to_s3()
        self._fetch_from_s3()

        self._prepare_market_data()
        self._prepare_fundamental_data()
        self._merge_fundamentals_into_market_data()

        self._build_universe()
        self._build_return_pivot()
        self._build_node_features()
        self._load_tbill_data()

        self._free_memory()

    def _connect_to_s3(self)->None:
        """
        Connect to the S3 bucket and load the data.
        :return:
        """
        self.aws = AWS(region=self.config.aws_default_region, verbose=True)
        # Optional sanity check
        self.aws.identity(print_info=True)
        # 2) Configure S3 defaults
        self.aws.s3.config(
            bucket=self.config.aws_bucket_name,
            output_type="pandas",  # tabular loads -> pandas (or "polars")
            file_type="parquet",  # default tabular format for dataframe uploads without extension
            overwrite=True,
        )

    def _fetch_from_s3(self) -> None:
        """
        Fetch raw data from S3.

        :return: Dictionary containing raw dataframes loaded from S3.
        """
        self.funda_data = self.aws.s3.load(key=self.config.funda_path)
        self.mkt_data = self.aws.s3.load(key=self.config.mkt_path)

    def _prepare_market_data(self) -> None:
        """
        Clean and standardize market data.
        """
        if self.mkt_data is None:
            raise ValueError("mkt_data is not loaded.")

        required_cols = ["date", "permno", "ticker", "ret", "market_cap"]
        missing_cols = [col for col in required_cols if col not in self.mkt_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in mkt_data: {missing_cols}")

        df = self.mkt_data.copy()

        df["date"] = pd.to_datetime(df["date"])
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df["ticker"] = df["ticker"].astype("string")
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

        df = df.dropna(subset=["date", "permno", "ticker", "ret", "market_cap"])
        df = df.sort_values(["permno", "date"]).reset_index(drop=True)

        # Keep one row per (date, permno)
        df = df.drop_duplicates(subset=["date", "permno"], keep="last").reset_index(drop=True)

        self.mkt_data = df

    def _prepare_fundamental_data(self) -> None:
        """
        Clean and standardize fundamental data used in the asof merge.
        """
        if self.funda_data is None:
            raise ValueError("funda_data is not loaded.")

        required_cols = ["public_date", "permno", "gicdesc"]
        missing_cols = [col for col in required_cols if col not in self.funda_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in funda_data: {missing_cols}")

        df = self.funda_data.copy()

        df["public_date"] = pd.to_datetime(df["public_date"])
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df["gicdesc"] = df["gicdesc"].astype("string")

        df = df.dropna(subset=["public_date", "permno", "gicdesc"])
        df = df.sort_values(["permno", "public_date"]).reset_index(drop=True)

        # Keep one row per (public_date, permno)
        df = df.drop_duplicates(subset=["public_date", "permno"], keep="last").reset_index(drop=True)

        self.funda_data = df

    def _merge_fundamentals_into_market_data(self) -> None:
        """
        Merge the latest available fundamental sector information into market data
        using an asof merge by permno.
        """
        if self.mkt_data is None:
            raise ValueError("mkt_data is not prepared.")
        if self.funda_data is None:
            raise ValueError("funda_data is not prepared.")

        mkt = self.mkt_data.copy()
        funda = self.funda_data.copy()

        # merge_asof requires sorting by the merge key first
        mkt = mkt.sort_values(["date", "permno"]).reset_index(drop=True)
        funda = funda[["permno", "public_date", "gicdesc"]] \
            .sort_values(["public_date", "permno"]) \
            .reset_index(drop=True)

        merged = pd.merge_asof(
            mkt,
            funda,
            left_on="date",
            right_on="public_date",
            by="permno",
            direction="backward",
            allow_exact_matches=True,
        )

        merged = merged.rename(columns={"gicdesc": "gics_sector"})
        merged = merged.drop(columns=["public_date"], errors="ignore")

        self.network_data = merged.sort_values(["date", "permno"]).reset_index(drop=True)

    def _build_universe(self) -> None:
        """
        Build the universe attribute from the prepared market data.
        """
        if self.mkt_data is None:
            raise ValueError("mkt_data is not prepared.")

        self.dates = sorted(self.mkt_data["date"].unique().tolist())

        self.universe = (
            self.mkt_data
            .sort_values(["date", "permno"])
            .groupby("date")["permno"]
            .apply(list)
            .to_dict()
        )

    def _build_return_pivot(self) -> None:
        """
        Build and store the return pivot table used later for on-demand
        rolling correlation computation.
        """
        logger.info("Building return pivot table for correlation computations...")
        if self.mkt_data is None:
            raise ValueError("mkt_data is not prepared.")

        self.ret_pivot = (
            self.mkt_data
            .pivot(index="date", columns="permno", values="ret")
            .sort_index()
            .sort_index(axis=1)
        )
        logger.info("Return pivot table built with shape: %s", self.ret_pivot.shape)

    def _load_tbill_data(self) -> None:
        """
        Load and convert DGS3MO (3-month T-bill) data to a monthly return series.
        Expected format: parquet/CSV with columns 'date'+'rate' (or FRED 'DATE'+'DGS3MO').
        Rate should be in annualised % (e.g. 5.25 for 5.25 %).
        Missing values represented as NaN or '.' are forward-filled.
        """
        if not self.config.tbill_path:
            logger.info("T-bill path not configured; using 0 %% risk-free rate.")
            return
        try:
            df = self.aws.s3.load(key=self.config.tbill_path)
            # Normalise FRED column names
            df = df.rename(columns={"DATE": "date", "DGS3MO": "rate"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
            # Resample to month-end, forward-fill gaps (e.g. weekends/holidays)
            monthly = df["rate"].resample("M").last().ffill()
            # Convert annualised % → monthly decimal return
            self.tbill_monthly = (1 + monthly / 100) ** (1 / 12) - 1
            logger.info("T-bill data loaded: %d monthly observations.", len(self.tbill_monthly))
        except Exception as exc:
            logger.warning("T-bill data load failed (%s); using 0 %% rf.", exc)

    def _free_memory(self) -> None:
        """
        Free memory by dropping large intermediate dataframes that are no longer needed.
        A slim copy of funda_data is kept for the FactorEngine.
        """
        _FACTOR_KEEP_COLS = [
            "permno", "public_date",
            "bm", "roe", "npm", "cfm", "fcf_ocf", "de_ratio", "roa", "gpm",
        ]
        if self.funda_data is not None:
            keep = [c for c in _FACTOR_KEEP_COLS if c in self.funda_data.columns]
            self.funda_factors: pd.DataFrame = (
                self.funda_data[keep]
                .dropna(subset=["permno", "public_date"])
                .sort_values(["permno", "public_date"])
                .reset_index(drop=True)
            )
        else:
            self.funda_factors = pd.DataFrame()
        self.mkt_data = None
        self.funda_data = None

    def _build_node_features(self) -> None:
        logger.info("Building node features for the network visualization...")
        """
        Build date-wise node features for the network:
        - x-month momentum excluding the most recent month
        - cross-sectional momentum percentile
        - node opacity
        - cross-sectional market-cap percentile
        - node size
        """
        if self.network_data is None:
            raise ValueError("network_data is not prepared.")

        df = self.network_data.copy()

        required_cols = ["date", "permno", "ticker", "ret", "market_cap", "gics_sector"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in network_data: {missing_cols}")

        df = df.sort_values(["permno", "date"]).reset_index(drop=True)

        momentum_months = self.config.momentum_months
        lookback_days = 21 * momentum_months
        skip_days = 21

        # ------------------------------------------------------------------
        # 1) Compute x-month momentum excluding the most recent month
        #    Momentum at t uses returns from t-lookback_days to t-skip_days
        # ------------------------------------------------------------------
        # Invalid for log1p if ret <= -1
        df["ret_for_mom"] = df["ret"].where(df["ret"] > -1, np.nan)
        df["logret"] = np.log1p(df["ret_for_mom"])

        def compute_group_momentum(logret_series: pd.Series) -> pd.Series:
            # Shift first to exclude the most recent month, then roll
            shifted = logret_series.shift(skip_days)
            rolling_sum = shifted.rolling(
                window=lookback_days,
                min_periods=lookback_days
            ).sum()
            return np.exp(rolling_sum) - 1

        df["momentum_xm"] = (
            df.groupby("permno", group_keys=False)["logret"]
            .apply(compute_group_momentum)
        )

        # ------------------------------------------------------------------
        # 2) Cross-sectional ranks by date
        # ------------------------------------------------------------------
        df["mcap_pct"] = (
            df.groupby("date")["market_cap"]
            .rank(method="average", pct=True)
        )

        df["momentum_pct"] = (
            df.groupby("date")["momentum_xm"]
            .rank(method="average", pct=True)
        )

        # ------------------------------------------------------------------
        # 3) Map to visual features
        # ------------------------------------------------------------------
        min_size, max_size = 8.0, 30.0
        df["node_size"] = min_size + (max_size - min_size) * df["mcap_pct"]

        min_opacity, max_opacity = 0.15, 1.0
        df["node_opacity"] = min_opacity + (max_opacity - min_opacity) * df["momentum_pct"]

        # ------------------------------------------------------------------
        # 4) Cleanup
        # ------------------------------------------------------------------
        df["ticker"] = df["ticker"].astype("string")
        df["gics_sector"] = df["gics_sector"].astype("string")

        df = df.drop(columns=["ret_for_mom", "logret"])

        self.node_features = df.sort_values(["date", "permno"]).reset_index(drop=True)
        logger.info("Node features built with shape: %s", self.node_features.shape)

    def get_node_table(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Return the node feature table for a given date.

        Parameters
        ----------
        date : pd.Timestamp
            Selected date.

        Returns
        -------
        pd.DataFrame
            One row per active node at the given date.
        """
        if self.node_features is None:
            raise ValueError("node_features is not built.")

        date = pd.Timestamp(date)

        nodes = self.node_features.loc[self.node_features["date"] == date].copy()

        if nodes.empty:
            return pd.DataFrame(columns=[
                "date", "permno", "ticker", "market_cap", "gics_sector",
                "momentum_xm", "momentum_pct", "node_opacity",
                "mcap_pct", "node_size"
            ])

        # Keep only rows with the minimum required node information
        nodes = nodes.dropna(subset=["permno", "ticker", "market_cap", "gics_sector"])

        # Optional: if you want nodes with missing momentum to be hidden for now,
        # uncomment the next line
        nodes = nodes.dropna(subset=["momentum_xm", "node_opacity"])

        nodes = nodes.sort_values("permno").reset_index(drop=True)
        return nodes

    def get_corr_matrix(
            self,
            date: pd.Timestamp,
            min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the rolling correlation matrix on demand for a given date,
        using only assets active at that date.

        Parameters
        ----------
        date : pd.Timestamp
            Date at which the rolling correlation matrix is computed.
        min_periods : int | None
            Minimum number of non-missing return observations required
            inside the rolling window for an asset to be kept.

        Returns
        -------
        pd.DataFrame
            Correlation matrix across active assets.
        """
        if self.ret_pivot is None:
            raise ValueError("ret_pivot is not built.")
        if self.universe is None:
            raise ValueError("universe is not built.")

        date = pd.Timestamp(date)

        if date not in self.universe:
            raise ValueError(f"Date {date} not found in universe.")
        if date not in self.ret_pivot.index:
            raise ValueError(f"Date {date} not found in return pivot index.")

        window = self.config.window_corr_matrix
        active_permnos = self.universe[date]

        valid_permnos = [p for p in active_permnos if p in self.ret_pivot.columns]
        if not valid_permnos:
            return pd.DataFrame()

        window_df = self.ret_pivot.loc[:date, valid_permnos].tail(window)

        # Drop names with no observations at all in the window
        window_df = window_df.dropna(axis=1, how="all")

        if min_periods is not None:
            counts = window_df.notna().sum(axis=0)
            keep_cols = counts[counts >= min_periods].index
            window_df = window_df.loc[:, keep_cols]

        if window_df.shape[1] == 0:
            return pd.DataFrame()

        corr = window_df.corr()
        return corr

    def get_edge_table(
            self,
            date: pd.Timestamp,
            threshold: float,
            min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Build the edge list for a given date from the rolling correlation matrix.
        """
        corr = self.get_corr_matrix(date=date, min_periods=min_periods)

        if corr.empty:
            return pd.DataFrame(columns=["source", "target", "corr", "abs_corr"])

        # Keep upper triangle only, excluding diagonal
        mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        corr_upper = corr.where(mask_upper)

        # Give distinct names to row/column axes before stacking
        corr_upper = corr_upper.rename_axis(index="source", columns="target")

        edges = (
            corr_upper.stack(dropna=True)
            .rename("corr")
            .reset_index()
        )

        if edges.empty:
            return pd.DataFrame(columns=["source", "target", "corr", "abs_corr"])

        edges["abs_corr"] = edges["corr"].abs()

        # Apply threshold
        edges = edges.loc[edges["abs_corr"] >= threshold].copy()

        if edges.empty:
            return pd.DataFrame(columns=["source", "target", "corr", "abs_corr"])

        edges = edges.sort_values(["source", "target"]).reset_index(drop=True)
        return edges

    def _compute_graph_snapshot(
            self,
            date: pd.Timestamp,
            threshold: float,
            min_periods: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute the synchronized node table and edge table for one graph snapshot.

        Rules
        -----
        - active universe only
        - rolling correlations on selected date
        - drop NaN edges
        - keep edges with abs(corr) >= threshold
        - drop nodes that become isolated
        - compute node importance on the displayed graph:
            * degree
            * weighted degree = sum of abs(corr) over incident edges
        - set node_size from weighted-degree percentile
        """
        nodes = self.get_node_table(date)
        edges = self.get_edge_table(date=date, threshold=threshold, min_periods=min_periods)

        empty_nodes_cols = [
            "date",
            "permno",
            "ticker",
            "market_cap",
            "gics_sector",
            "momentum_xm",
            "momentum_pct",
            "node_opacity",
            "mcap_pct",
            "node_size",
            "degree",
            "weighted_degree",
            "weighted_degree_pct",
        ]

        if nodes.empty or edges.empty:
            return (
                pd.DataFrame(columns=nodes.columns if not nodes.empty else empty_nodes_cols),
                pd.DataFrame(columns=["source", "target", "corr", "abs_corr"]),
            )

        connected_permnos = pd.unique(
            pd.concat([edges["source"], edges["target"]], ignore_index=True)
        )

        nodes = nodes.loc[nodes["permno"].isin(connected_permnos)].copy()

        edges = edges.loc[
            edges["source"].isin(nodes["permno"])
            & edges["target"].isin(nodes["permno"])
            ].copy()

        if nodes.empty or edges.empty:
            return (
                pd.DataFrame(columns=nodes.columns if not nodes.empty else empty_nodes_cols),
                pd.DataFrame(columns=["source", "target", "corr", "abs_corr"]),
            )

        degree_df = pd.concat(
            [
                edges[["source"]].rename(columns={"source": "permno"}),
                edges[["target"]].rename(columns={"target": "permno"}),
            ],
            ignore_index=True,
        )
        degree_df["degree"] = 1
        degree_df = degree_df.groupby("permno", as_index=False)["degree"].sum()

        weighted_degree_df = pd.concat(
            [
                edges[["source", "abs_corr"]].rename(columns={"source": "permno"}),
                edges[["target", "abs_corr"]].rename(columns={"target": "permno"}),
            ],
            ignore_index=True,
        )
        weighted_degree_df = (
            weighted_degree_df
            .groupby("permno", as_index=False)["abs_corr"]
            .sum()
            .rename(columns={"abs_corr": "weighted_degree"})
        )

        nodes = nodes.merge(degree_df, on="permno", how="left")
        nodes = nodes.merge(weighted_degree_df, on="permno", how="left")

        nodes["degree"] = nodes["degree"].fillna(0).astype(int)
        nodes["weighted_degree"] = nodes["weighted_degree"].fillna(0.0)

        nodes["weighted_degree_pct"] = (
            nodes["weighted_degree"]
            .rank(method="average", pct=True)
        )

        min_size, max_size = 8.0, 30.0
        nodes["node_size"] = min_size + (max_size - min_size) * nodes["weighted_degree_pct"]

        nodes = nodes.sort_values("permno").reset_index(drop=True)
        edges = edges.sort_values(["source", "target"]).reset_index(drop=True)

        return nodes, edges

    def get_graph_snapshot(
            self,
            date: pd.Timestamp,
            threshold: float,
            min_periods: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return a cached graph snapshot if available, otherwise compute and cache it.
        """
        key = self._make_snapshot_cache_key(date=date, threshold=threshold, min_periods=min_periods)

        if key in self._graph_snapshot_cache:
            nodes, edges = self._graph_snapshot_cache[key]
            return nodes.copy(), edges.copy()

        nodes, edges = self._compute_graph_snapshot(
            date=date,
            threshold=threshold,
            min_periods=min_periods,
        )

        self._graph_snapshot_cache[key] = (nodes.copy(), edges.copy())
        return nodes, edges

    def _get_layout_dates(self) -> list[pd.Timestamp]:
        """
        Select a subset of dates used to build the reference layout.
        """
        if self.dates is None:
            raise ValueError("dates are not built.")

        step = self.config.layout_sample_step
        return self.dates[::step]

    def _build_reference_corr_matrix(self) -> None:
        """
        Build the reference median absolute correlation matrix from a sampled
        subset of dates.
        """
        logger.info("Building reference correlation matrix from sampled dates...")

        if self.node_features is None:
            raise ValueError("node_features is not built.")

        layout_dates = self._get_layout_dates()
        threshold_ref = self.config.layout_threshold_ref
        min_periods = self.config.layout_min_periods

        pair_values: dict[tuple[int, int], list[float]] = {}

        for i,date in enumerate(layout_dates):
            if i%10 == 0:
                logger.info(f"Processing date {date} ({i}/{len(layout_dates)+1}) for reference correlation matrix...")
            try:
                corr = self.get_corr_matrix(date=date, min_periods=min_periods)
            except Exception:
                continue

            if corr.empty:
                continue

            abs_corr = corr.abs()

            # keep upper triangle only
            mask_upper = np.triu(np.ones(abs_corr.shape, dtype=bool), k=1)
            abs_corr = abs_corr.where(mask_upper)

            edges = (
                abs_corr.rename_axis(index="source", columns="target")
                .stack(dropna=True)
                .rename("abs_corr")
                .reset_index()
            )

            if edges.empty:
                continue

            edges = edges.loc[edges["abs_corr"] >= threshold_ref].copy()
            if edges.empty:
                continue

            for row in edges.itertuples(index=False):
                key = (int(row.source), int(row.target))
                pair_values.setdefault(key, []).append(float(row.abs_corr))

        if not pair_values:
            self.reference_corr_matrix = pd.DataFrame()
            return

        all_permnos = sorted(
            set(p for pair in pair_values.keys() for p in pair)
        )

        ref = pd.DataFrame(
            data=np.nan,
            index=all_permnos,
            columns=all_permnos,
            dtype=float
        )

        np.fill_diagonal(ref.values, 1.0)

        logger.info("Computing median correlations for reference matrix...")
        for (src, tgt), vals in pair_values.items():
            med = float(np.median(vals))
            ref.loc[src, tgt] = med
            ref.loc[tgt, src] = med

        self.reference_corr_matrix = ref.sort_index().sort_index(axis=1)
        logger.info("Reference correlation matrix built with shape: %s", self.reference_corr_matrix.shape)

    def _build_reference_layout(self) -> None:
        """
        Build frozen 2D coordinates from the reference correlation matrix.
        """
        logger.info("Building reference layout from reference correlation matrix...")

        if self.reference_corr_matrix is None:
            raise ValueError("reference_corr_matrix is not built.")

        ref = self.reference_corr_matrix.copy()

        if ref.empty:
            self.reference_layout = pd.DataFrame(columns=["permno", "x", "y"])
            return

        threshold_ref = self.config.layout_threshold_ref

        # Build graph
        G = nx.Graph()

        for permno in ref.index:
            G.add_node(int(permno))
        logger.info("Graph initialized with %d nodes.", G.number_of_nodes())

        mask_upper = np.triu(np.ones(ref.shape, dtype=bool), k=1)
        ref_upper = ref.where(mask_upper)

        edges = (
            ref_upper.rename_axis(index="source", columns="target")
            .stack(dropna=True)
            .rename("weight")
            .reset_index()
        )

        edges = edges.loc[edges["weight"] >= threshold_ref].copy()

        for row in edges.itertuples(index=False):
            G.add_edge(
                int(row.source),
                int(row.target),
                weight=float(row.weight) ** 2
            )
        logger.info("Graph built with %d edges after applying threshold.", G.number_of_edges())

        if G.number_of_nodes() == 0:
            self.reference_layout = pd.DataFrame(columns=["permno", "x", "y"])
            return

        # Spring layout: stronger edges pull nodes together
        pos = nx.spring_layout(
            G,
            weight="weight",
            seed=self.config.layout_seed,
            dim=2,
            k=self.config.node_spacing,  # increase spacing
            iterations=100
        )

        layout_df = pd.DataFrame(
            [
                {"permno": int(permno), "x": float(coords[0]), "y": float(coords[1])}
                for permno, coords in pos.items()
            ]
        )

        # Rescale to a more readable range for Cytoscape
        if not layout_df.empty:
            for col in ["x", "y"]:
                cmin = layout_df[col].min()
                cmax = layout_df[col].max()
                if pd.notna(cmin) and pd.notna(cmax) and cmax > cmin:
                    layout_df[col] = self.config.coordinate_scale * (layout_df[col] - cmin) / (cmax - cmin)
                else:
                    layout_df[col] = self.config.coordinate_scale

        self.reference_layout = layout_df.sort_values("permno").reset_index(drop=True)
        logger.info("Reference layout built with %d nodes.", self.reference_layout.shape[0])

    def build_reference_network_layout(self) -> None:
        """
        Build the reference correlation structure and the frozen node coordinates.
        """
        self._build_reference_corr_matrix()
        self._build_reference_layout()

    def get_positioned_node_table(self, date: pd.Timestamp, threshold: float,
                                  min_periods: int | None = None) -> pd.DataFrame:
        """
        Return the node table for one graph snapshot, enriched with frozen layout coordinates.
        """
        if self.reference_layout is None:
            raise ValueError("reference_layout is not built.")

        nodes, _ = self.get_graph_snapshot(date=date, threshold=threshold, min_periods=min_periods)

        if nodes.empty:
            return nodes

        nodes = nodes.merge(self.reference_layout, on="permno", how="inner")
        nodes = nodes.sort_values("permno").reset_index(drop=True)
        return nodes

    def get_node_ranking_table(
            self,
            date: pd.Timestamp,
            threshold: float,
            min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Return a node ranking table for one graph snapshot, sorted by weighted degree,
        using cache.
        """
        key = self._make_snapshot_cache_key(date=date, threshold=threshold, min_periods=min_periods)

        if key in self._node_ranking_cache:
            return self._node_ranking_cache[key].copy()

        nodes, _ = self.get_graph_snapshot(
            date=date,
            threshold=threshold,
            min_periods=min_periods,
        )

        if nodes.empty:
            ranking = pd.DataFrame(
                columns=[
                    "rank",
                    "permno",
                    "ticker",
                    "gics_sector",
                    "degree",
                    "weighted_degree",
                    "momentum_xm",
                    "market_cap",
                ]
            )
            self._node_ranking_cache[key] = ranking
            return ranking.copy()

        ranking = nodes[
            [
                "permno",
                "ticker",
                "gics_sector",
                "degree",
                "weighted_degree",
                "momentum_xm",
                "market_cap",
            ]
        ].copy()

        ranking = ranking.sort_values(
            ["weighted_degree", "degree", "ticker"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        ranking.insert(0, "rank", range(1, len(ranking) + 1))

        self._node_ranking_cache[key] = ranking.copy()
        return ranking

    def get_graph_summary_stats(
            self,
            date: pd.Timestamp,
            threshold: float,
            min_periods: int | None = None,
    ) -> dict[str, float | int]:
        """
        Return graph-level summary statistics for one snapshot, using cache.
        """
        key = self._make_snapshot_cache_key(date=date, threshold=threshold, min_periods=min_periods)

        if key in self._graph_summary_cache:
            return dict(self._graph_summary_cache[key])

        nodes, edges = self.get_graph_snapshot(
            date=date,
            threshold=threshold,
            min_periods=min_periods,
        )

        if nodes.empty or edges.empty:
            stats = {
                "n_nodes": 0,
                "n_edges": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "avg_weighted_degree": 0.0,
                "total_edge_weight": 0.0,
                "n_components": 0,
                "largest_component_size": 0,
            }
            self._graph_summary_cache[key] = stats
            return dict(stats)

        n_nodes = int(len(nodes))
        n_edges = int(len(edges))

        density = 0.0
        if n_nodes > 1:
            density = 2.0 * n_edges / (n_nodes * (n_nodes - 1))

        avg_degree = float(nodes["degree"].mean()) if "degree" in nodes.columns else 0.0
        avg_weighted_degree = (
            float(nodes["weighted_degree"].mean())
            if "weighted_degree" in nodes.columns
            else 0.0
        )

        total_edge_weight = float(edges["abs_corr"].sum())

        G = nx.Graph()
        G.add_nodes_from(nodes["permno"].tolist())
        G.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))

        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            n_components = len(components)
            largest_component_size = max(len(c) for c in components) if components else 0
        else:
            n_components = 0
            largest_component_size = 0

        stats = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": avg_degree,
            "avg_weighted_degree": avg_weighted_degree,
            "total_edge_weight": total_edge_weight,
            "n_components": n_components,
            "largest_component_size": largest_component_size,
        }

        self._graph_summary_cache[key] = stats
        return dict(stats)

    def get_node_neighbors_table(
            self,
            date: pd.Timestamp,
            permno: int,
            threshold: float,
            min_periods: int | None = None,
    ) -> pd.DataFrame:
        """
        Return the neighbors of one node in the displayed graph snapshot.

        Columns returned
        ----------------
        - neighbor_permno
        - neighbor_ticker
        - neighbor_sector
        - corr
        - abs_corr
        """
        nodes, edges = self.get_graph_snapshot(
            date=date,
            threshold=threshold,
            min_periods=min_periods,
        )

        if nodes.empty or edges.empty:
            return pd.DataFrame(
                columns=[
                    "neighbor_permno",
                    "neighbor_ticker",
                    "neighbor_sector",
                    "corr",
                    "abs_corr",
                ]
            )

        permno = int(permno)

        incident_as_source = edges.loc[edges["source"] == permno, ["target", "corr", "abs_corr"]].copy()
        incident_as_source = incident_as_source.rename(columns={"target": "neighbor_permno"})

        incident_as_target = edges.loc[edges["target"] == permno, ["source", "corr", "abs_corr"]].copy()
        incident_as_target = incident_as_target.rename(columns={"source": "neighbor_permno"})

        neighbors = pd.concat([incident_as_source, incident_as_target], ignore_index=True)

        if neighbors.empty:
            return pd.DataFrame(
                columns=[
                    "neighbor_permno",
                    "neighbor_ticker",
                    "neighbor_sector",
                    "corr",
                    "abs_corr",
                ]
            )

        node_lookup = nodes[["permno", "ticker", "gics_sector"]].drop_duplicates().copy()

        neighbors = neighbors.merge(
            node_lookup,
            left_on="neighbor_permno",
            right_on="permno",
            how="left",
        )

        neighbors = neighbors.rename(
            columns={
                "ticker": "neighbor_ticker",
                "gics_sector": "neighbor_sector",
            }
        ).drop(columns=["permno"], errors="ignore")

        neighbors = neighbors.sort_values(
            ["abs_corr", "neighbor_ticker"],
            ascending=[False, True],
        ).reset_index(drop=True)

        return neighbors

    ### Cache ###
    @staticmethod
    def _make_snapshot_cache_key(
            date: pd.Timestamp,
            threshold: float,
            min_periods: int | None,
    ) -> tuple[pd.Timestamp, float, int | None]:
        """
        Build a stable cache key for graph snapshots and derived statistics.
        """
        date = pd.Timestamp(date)
        threshold = round(float(threshold), 6)
        return date, threshold, min_periods

    def get_period_returns(
        self,
        date: pd.Timestamp,
        n_days: int,
    ) -> pd.Series:
        """
        Compute the cumulative return for each stock over the past n_days
        trading days ending at date (inclusive).

        Uses ret_pivot, which stores daily returns indexed by date with
        permno as columns.  NaN returns within the window are skipped
        (equivalent to 0-return days for that stock).

        Parameters
        ----------
        date : pd.Timestamp
            End date of the period.  Snapped to the nearest available date
            in ret_pivot if not found exactly.
        n_days : int
            Number of trading days in the look-back window (e.g. 1, 5, 21, 252).

        Returns
        -------
        pd.Series
            Index: permno (Int64).  Values: cumulative return as a decimal
            (e.g. 0.05 = +5 %).  NaN for stocks with no data in window.
        """
        if self.ret_pivot is None:
            raise ValueError("ret_pivot is not built.")

        date = pd.Timestamp(date)

        # Snap to nearest available date
        if date not in self.ret_pivot.index:
            pos = self.ret_pivot.index.searchsorted(date)
            pos = min(pos, len(self.ret_pivot.index) - 1)
            date = self.ret_pivot.index[pos]

        window = self.ret_pivot.loc[:date].tail(n_days)
        if window.empty:
            return pd.Series(dtype=float)

        # (1 + r1)(1 + r2)…(1 + rN) − 1, skipping NaN days
        cum_ret = (1 + window.fillna(0)).prod() - 1

        # Set to NaN stocks that had NO valid data at all in the window
        valid_counts = window.notna().sum()
        cum_ret[valid_counts == 0] = np.nan

        return cum_ret

    def get_universe_snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Return all stocks present in the universe at a given date.

        Uses network_data (no momentum filter) so all stocks are included
        regardless of how much return history they have.

        Returns
        -------
        pd.DataFrame
            Columns: permno, ticker, gics_sector, market_cap
            Sorted by market_cap descending.
        """
        if self.network_data is None:
            raise ValueError("network_data is not prepared.")

        date = pd.Timestamp(date)
        df = (
            self.network_data
            .loc[self.network_data["date"] == date,
                 ["permno", "ticker", "gics_sector", "market_cap"]]
            .dropna(subset=["gics_sector", "market_cap"])
            .sort_values("market_cap", ascending=False)
            .reset_index(drop=True)
        )
        return df

    def get_sector_snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Return sector-level aggregated data at a given date.

        Uses network_data (all stocks, no momentum filter) so early dates
        with insufficient momentum history are still fully represented.

        Returns
        -------
        pd.DataFrame
            Columns: sector, total_mcap, n_stocks, avg_mcap
            Sorted by total_mcap descending.
        """
        if self.network_data is None:
            raise ValueError("network_data is not prepared.")

        date = pd.Timestamp(date)
        df = self.network_data.loc[self.network_data["date"] == date].copy()

        if df.empty:
            return pd.DataFrame(columns=["sector", "total_mcap", "n_stocks", "avg_mcap"])

        snapshot = (
            df.dropna(subset=["gics_sector", "market_cap"])
            .groupby("gics_sector", as_index=False)
            .agg(
                total_mcap=("market_cap", "sum"),
                n_stocks=("permno", "count"),
                avg_mcap=("market_cap", "mean"),
            )
            .rename(columns={"gics_sector": "sector"})
            .sort_values("total_mcap", ascending=False)
            .reset_index(drop=True)
        )
        return snapshot

    def build_graph_summary_timeseries(
        self,
        monthly_dates: list[pd.Timestamp],
        threshold: float,
        min_periods: int | None = None,
    ) -> None:
        """
        Pre-compute graph summary statistics for each supplied monthly date.

        Results are stored in self.graph_summary_timeseries as a DataFrame
        indexed by date with one column per statistic:
            n_nodes, n_edges, density, avg_degree, avg_weighted_degree,
            total_edge_weight, n_components, largest_component_size

        Parameters
        ----------
        monthly_dates : list[pd.Timestamp]
            Dates at which to evaluate the graph (typically one per month).
        threshold : float
            Correlation threshold used to build the graph.
        min_periods : int | None
            Minimum observations required inside the rolling window.
        """
        logger.info(
            "Building graph summary timeseries for %d dates (threshold=%.2f)…",
            len(monthly_dates), threshold,
        )
        records: list[dict] = []

        for i, date in enumerate(monthly_dates):
            if i % 20 == 0:
                logger.info("  …date %d / %d  (%s)", i, len(monthly_dates), date)
            try:
                stats = self.get_graph_summary_stats(
                    date=date,
                    threshold=threshold,
                    min_periods=min_periods,
                )
                records.append({"date": date, **stats})
            except Exception as exc:
                logger.warning("Skipping %s: %s", date, exc)

        if records:
            self.graph_summary_timeseries = (
                pd.DataFrame(records)
                .set_index("date")
                .sort_index()
            )
        else:
            self.graph_summary_timeseries = pd.DataFrame()

        logger.info(
            "Graph summary timeseries ready: shape %s",
            self.graph_summary_timeseries.shape,
        )

    def get_sector_vol_heatmap_data(self) -> pd.DataFrame:
        """
        Compute monthly realized volatility (annualized) per GICS sector,
        using an equal-weighted sector portfolio.

        Daily returns are first averaged across all stocks in each sector
        (equal-weighted portfolio), then monthly realized vol is computed
        as the standard deviation of daily returns within each calendar month,
        annualized by √252.

        Returns
        -------
        pd.DataFrame
            Index  : month-end timestamps (DatetimeIndex, freq="ME")
            Columns: GICS sector names
            Values : annualized realized vol (float, e.g. 0.25 = 25 %)
        """
        if self.network_data is None:
            raise ValueError("network_data is not prepared.")

        df = self.network_data[["date", "ret", "gics_sector"]].dropna(
            subset=["date", "ret", "gics_sector"]
        )

        # Equal-weighted daily sector return: mean across all stocks in sector
        sector_daily = (
            df.groupby(["date", "gics_sector"])["ret"]
            .mean()
            .unstack("gics_sector")
            .sort_index()
        )
        sector_daily.index = pd.DatetimeIndex(sector_daily.index)
        sector_daily.columns.name = None

        # Monthly std of daily returns, annualized by √252
        monthly_vol = sector_daily.resample("M").std() * np.sqrt(252)

        return monthly_vol

    def clear_graph_caches(self) -> None:
        """
        Clear all cached graph snapshots and derived objects.
        """
        self._graph_snapshot_cache.clear()
        self._graph_summary_cache.clear()
        self._node_ranking_cache.clear()