from __future__ import annotations

from typing import Any
from data_viz.utils.config import Config
import pandas as pd


class NetworkBuilder:
    """
    Build Dash Cytoscape elements and styles from precomputed network data.

    Responsibilities
    ----------------
    - map sectors to stable colors
    - convert node/edge tables into Cytoscape elements
    - build the base Cytoscape stylesheet
    - keep visual configuration centralized

    This class does NOT:
    - compute correlations
    - compute node features
    - manage Dash callbacks
    - store mutable graph state
    """

    DEFAULT_SECTOR_COLORS = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#aec7e8",  # light blue
    ]

    def __init__(
        self,
        config: Config
    ) -> None:
        """
        config should contain parameters:
        ----------
        label_market_cap_quantile : float
            Show ticker labels only for nodes with mcap_pct >= this threshold.
        min_edge_width : float
            Minimum displayed edge width.
        max_edge_width : float
            Maximum displayed edge width.
        min_node_opacity_fallback : float
            Fallback opacity when node_opacity is missing.
        default_node_color : str
            Fallback color for nodes with missing/unknown sector.
        positive_edge_color : str
            Edge color when corr > 0.
        negative_edge_color : str
            Edge color when corr < 0.
        """
        self.label_market_cap_quantile = config.label_market_cap_quantile
        self.min_edge_width = config.min_edge_width
        self.max_edge_width = config.max_edge_width
        self.min_node_opacity_fallback = config.min_node_opacity_fallback
        self.default_node_color = config.default_node_color
        self.positive_edge_color = config.positive_edge_color
        self.negative_edge_color = config.negative_edge_color

    def build_sector_color_map(self, sectors: list[str]) -> dict[str, str]:
        unique_sectors = sorted({str(s) for s in sectors if pd.notna(s)})
        color_map: dict[str, str] = {}

        for i, sector in enumerate(unique_sectors):
            color_map[sector] = self.DEFAULT_SECTOR_COLORS[i % len(self.DEFAULT_SECTOR_COLORS)]

        return color_map

    def scale_edge_width(
        self,
        abs_corr: float,
        threshold: float = 0.0,
    ) -> float:
        """
        Map abs(corr) to edge width, rescaled relative to the current threshold.
        """
        if pd.isna(abs_corr):
            return self.min_edge_width

        abs_corr = max(0.0, min(1.0, float(abs_corr)))
        threshold = max(0.0, min(0.999, float(threshold)))

        if abs_corr <= threshold:
            return self.min_edge_width

        x = (abs_corr - threshold) / (1.0 - threshold)
        x = x ** 0.7

        return self.min_edge_width + (self.max_edge_width - self.min_edge_width) * x

    def build_node_elements(
        self,
        nodes: pd.DataFrame,
        sector_color_map: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if nodes.empty:
            return []

        required_cols = [
            "permno",
            "ticker",
            "market_cap",
            "gics_sector",
            "momentum_xm",
            "node_size",
            "node_opacity",
            "mcap_pct",
            "x",
            "y",
        ]
        missing = [c for c in required_cols if c not in nodes.columns]
        if missing:
            raise ValueError(f"Missing required node columns: {missing}")

        if sector_color_map is None:
            sector_color_map = self.build_sector_color_map(
                nodes["gics_sector"].dropna().astype(str).unique().tolist()
            )

        elements: list[dict[str, Any]] = []

        for row in nodes.itertuples(index=False):  # type: ignore[attr-defined]
            sector_val = getattr(row, "gics_sector", None)
            sector = str(sector_val) if pd.notna(sector_val) else "Unknown"
            color = sector_color_map.get(sector, self.default_node_color)

            mcap_pct_val = getattr(row, "mcap_pct", None)
            show_label = pd.notna(mcap_pct_val) and float(mcap_pct_val) >= self.label_market_cap_quantile

            ticker_val = getattr(row, "ticker", "")
            ticker_str = str(ticker_val)
            label = ticker_str if show_label else ""

            market_cap_val = getattr(row, "market_cap", None)
            momentum_val = getattr(row, "momentum_xm", None)
            node_size_val = getattr(row, "node_size", None)
            node_opacity_val = getattr(row, "node_opacity", None)
            permno_val = getattr(row, "permno")
            x_val = getattr(row, "x")
            y_val = getattr(row, "y")

            opacity = (
                float(node_opacity_val)
                if pd.notna(node_opacity_val)
                else self.min_node_opacity_fallback
            )

            elements.append(
                {
                    "data": {
                        "id": str(int(permno_val)),
                        "label": label,
                        "ticker": ticker_str,
                        "ticker_upper": ticker_str.upper(),
                        "permno": int(permno_val),
                        "market_cap": None if pd.isna(market_cap_val) else float(market_cap_val),
                        "sector": sector,
                        "momentum_xm": None if pd.isna(momentum_val) else float(momentum_val),
                        "node_size": None if pd.isna(node_size_val) else float(node_size_val),
                        "node_opacity": opacity,
                        "node_color": color,
                        "mcap_pct": None if pd.isna(mcap_pct_val) else float(mcap_pct_val),
                    },
                    "position": {
                        "x": float(x_val),
                        "y": float(y_val),
                    },
                    "classes": "node",
                }
            )

        return elements

    def build_edge_elements(
        self,
        edges: pd.DataFrame,
        permno_to_ticker: dict[int, str],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        if edges.empty:
            return []

        required_cols = ["source", "target", "corr", "abs_corr"]
        missing = [c for c in required_cols if c not in edges.columns]
        if missing:
            raise ValueError(f"Missing required edge columns: {missing}")

        elements: list[dict[str, Any]] = []

        for row in edges.itertuples(index=False):  # type: ignore[attr-defined]
            source_val = int(getattr(row, "source"))
            target_val = int(getattr(row, "target"))
            corr = float(getattr(row, "corr"))
            abs_corr = float(getattr(row, "abs_corr"))

            color = self.positive_edge_color if corr > 0 else self.negative_edge_color
            width = self.scale_edge_width(abs_corr=abs_corr, threshold=threshold)

            elements.append(
                {
                    "data": {
                        "id": f"{source_val}__{target_val}",
                        "source": str(source_val),
                        "target": str(target_val),
                        "source_permno": source_val,
                        "target_permno": target_val,
                        "source_ticker": permno_to_ticker.get(source_val, str(source_val)),
                        "target_ticker": permno_to_ticker.get(target_val, str(target_val)),
                        "corr": corr,
                        "abs_corr": abs_corr,
                        "edge_color": color,
                        "edge_width": width,
                    },
                    "classes": "edge",
                }
            )

        return elements

    def build_elements(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        threshold: float = 0.0,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        sector_color_map = self.build_sector_color_map(
            nodes["gics_sector"].dropna().astype(str).unique().tolist()
        )

        permno_to_ticker = {
            int(row.permno): str(row.ticker)
            for row in nodes[["permno", "ticker"]].itertuples(index=False)
        }

        node_elements = self.build_node_elements(
            nodes=nodes,
            sector_color_map=sector_color_map,
        )
        edge_elements = self.build_edge_elements(
            edges=edges,
            permno_to_ticker=permno_to_ticker,
            threshold=threshold,
        )

        return node_elements + edge_elements, sector_color_map

    @staticmethod
    def get_base_stylesheet() -> list[dict[str, Any]]:
        return [
            {
                "selector": "node",
                "style": {
                    "width": "data(node_size)",
                    "height": "data(node_size)",
                    "background-color": "data(node_color)",
                    "opacity": "data(node_opacity)",
                    "label": "data(label)",
                    "font-size": 8,
                    "text-valign": "center",
                    "text-halign": "center",
                    "color": "#111111",
                    "text-outline-width": 0,
                    "border-width": 0,
                },
            },
            {
                "selector": "edge",
                "style": {
                    "line-color": "data(edge_color)",
                    "width": "data(edge_width)",
                    "opacity": 0.22,
                    "curve-style": "straight",
                },
            },
            {
                "selector": ".faded",
                "style": {
                    "opacity": 0.08,
                },
            },
            {
                "selector": ".highlighted",
                "style": {
                    "opacity": 1.0,
                    "z-index": 9999,
                },
            },
        ]