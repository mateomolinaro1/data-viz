from __future__ import annotations

import logging
import sys

import pandas as pd
import dash_cytoscape as cyto
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
from dotenv import load_dotenv

from data_viz.data.data import DataManager
from data_viz.network.network import NetworkBuilder
from data_viz.utils.config import Config


# ============================================================
# UI HELPERS
# ============================================================

def build_node_info_panel() -> html.Div:
    return html.Div(
        id="selected-node-panel",
        children=[
            html.H4("Selected node", style={"marginBottom": "10px", "marginTop": "0px"}),
            html.Div("Click a node to see its neighbors and correlations."),
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "width": "260px",
            "marginTop": "20px",
        },
    )


def build_neighbors_table(neighbors_df: pd.DataFrame) -> html.Div:
    display_df = neighbors_df.copy()

    if not display_df.empty:
        if "corr" in display_df.columns:
            display_df["corr"] = display_df["corr"].round(4)
        if "abs_corr" in display_df.columns:
            display_df["abs_corr"] = display_df["abs_corr"].round(4)

    return html.Div(
        [
            html.H5("Neighbors", style={"marginBottom": "8px", "marginTop": "12px"}),
            dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[
                    {"name": "Ticker", "id": "neighbor_ticker"},
                    {"name": "Sector", "id": "neighbor_sector"},
                    {"name": "Corr", "id": "corr"},
                    {"name": "|Corr|", "id": "abs_corr"},
                ],
                sort_action="native",
                page_action="none",
                style_table={
                    "height": "260px",
                    "overflowY": "auto",
                    "overflowX": "auto",
                    "border": "1px solid #ddd",
                },
                style_cell={
                    "textAlign": "left",
                    "padding": "6px",
                    "fontSize": "12px",
                    "whiteSpace": "nowrap",
                },
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": "#f2f2f2",
                },
            ),
        ]
    )


def build_sector_legend(sector_color_map: dict[str, str]) -> html.Div:
    items = []

    for sector, color in sorted(sector_color_map.items()):
        items.append(
            html.Div(
                [
                    html.Div(
                        style={
                            "width": "14px",
                            "height": "14px",
                            "backgroundColor": color,
                            "display": "inline-block",
                            "marginRight": "8px",
                            "border": "1px solid #333",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span(sector, style={"verticalAlign": "middle"}),
                ],
                style={"marginBottom": "6px"},
            )
        )

    return html.Div(
        [
            html.H4("Sector legend", style={"marginBottom": "10px", "marginTop": "0px"}),
            *items,
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "width": "260px",
        },
    )


def build_graph_summary_panel(stats: dict[str, float | int]) -> html.Div:
    rows = [
        ("Nodes", f"{stats['n_nodes']:,}"),
        ("Edges", f"{stats['n_edges']:,}"),
        ("Density", f"{stats['density']:.4f}"),
        ("Avg degree", f"{stats['avg_degree']:.2f}"),
        ("Avg weighted degree", f"{stats['avg_weighted_degree']:.2f}"),
        ("Total edge weight", f"{stats['total_edge_weight']:.2f}"),
        ("Connected components", f"{stats['n_components']:,}"),
        ("Largest component", f"{stats['largest_component_size']:,}"),
    ]

    return html.Div(
        [
            html.H4("Graph summary", style={"marginBottom": "10px", "marginTop": "0px"}),
            html.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(label, style={"fontWeight": "bold", "padding": "4px 10px 4px 0"}),
                                    html.Td(value, style={"padding": "4px 0"}),
                                ]
                            )
                            for label, value in rows
                        ]
                    )
                ],
                style={"width": "100%"},
            ),
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "width": "260px",
            "marginTop": "20px",
        },
    )


def build_ranking_table(ranking_df: pd.DataFrame) -> html.Div:
    display_df = ranking_df.copy()

    if not display_df.empty:
        if "weighted_degree" in display_df.columns:
            display_df["weighted_degree"] = display_df["weighted_degree"].round(3)
        if "momentum_xm" in display_df.columns:
            display_df["momentum_xm"] = display_df["momentum_xm"].round(4)
        if "market_cap" in display_df.columns:
            display_df["market_cap"] = display_df["market_cap"].round(0)

    return html.Div(
        [
            html.H4("Ticker ranking by weighted degree", style={"marginBottom": "10px"}),
            dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[
                    {"name": "Rank", "id": "rank"},
                    {"name": "Ticker", "id": "ticker"},
                    {"name": "Sector", "id": "gics_sector"},
                    {"name": "Degree", "id": "degree"},
                    {"name": "Weighted degree", "id": "weighted_degree"},
                    {"name": "Momentum", "id": "momentum_xm"},
                    {"name": "Market cap", "id": "market_cap"},
                ],
                sort_action="native",
                filter_action="native",
                page_action="none",
                style_table={
                    "height": "420px",
                    "overflowY": "auto",
                    "overflowX": "auto",
                    "border": "1px solid #ccc",
                },
                style_cell={
                    "textAlign": "left",
                    "padding": "6px",
                    "fontSize": "12px",
                    "whiteSpace": "nowrap",
                },
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": "#f2f2f2",
                },
            ),
        ],
        style={"marginTop": "5px"},
    )


def build_edge_info_panel() -> html.Div:
    return html.Div(
        id="selected-edge-panel",
        children=[
            html.H4("Selected edge", style={"marginBottom": "10px", "marginTop": "0px"}),
            html.Div("Click an edge to see its details."),
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "width": "260px",
            "marginTop": "20px",
        },
    )


def build_search_panel() -> html.Div:
    return html.Div(
        [
            html.H4("Ticker search", style={"marginBottom": "10px", "marginTop": "0px"}),
            dcc.Input(
                id="ticker-search-input",
                type="text",
                placeholder="Search ticker (e.g. MSFT)",
                debounce=True,
                style={
                    "width": "100%",
                    "padding": "8px",
                    "fontSize": "13px",
                    "boxSizing": "border-box",
                },
            ),
            html.Div(
                "Type a ticker to highlight it in the graph.",
                style={"marginTop": "8px", "fontSize": "12px", "color": "#555"},
            ),
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "width": "260px",
            "marginTop": "20px",
        },
    )


def build_reset_panel() -> html.Div:
    return html.Div(
        [
            html.Button(
                "Reset highlight",
                id="reset-highlight-btn",
                n_clicks=0,
                style={
                    "width": "100%",
                    "padding": "10px",
                    "fontSize": "13px",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                },
            ),
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "backgroundColor": "#fafafa",
            "width": "260px",
            "marginTop": "20px",
        },
    )


# ============================================================
# STYLESHEET HELPERS
# ============================================================

def build_highlight_stylesheet(base_stylesheet, searched_ticker: str | None):
    stylesheet = list(base_stylesheet)

    if not searched_ticker:
        return stylesheet

    searched_ticker = searched_ticker.strip().upper()
    if not searched_ticker:
        return stylesheet

    stylesheet += [
        {"selector": "node", "style": {"opacity": 0.12}},
        {"selector": "edge", "style": {"opacity": 0.04}},
        {
            "selector": f'node[ticker_upper = "{searched_ticker}"]',
            "style": {
                "opacity": 1.0,
                "border-width": 4,
                "border-color": "#000000",
                "label": "data(ticker)",
                "font-size": 12,
                "z-index": 9999,
            },
        },
    ]
    return stylesheet


def build_node_neighborhood_stylesheet(base_stylesheet, clicked_node_data, elements):
    stylesheet = list(base_stylesheet)

    if not clicked_node_data:
        return stylesheet

    clicked_id = str(clicked_node_data.get("id"))
    if not clicked_id:
        return stylesheet

    neighbor_ids = set()
    edge_ids = set()

    for el in elements or []:
        data = el.get("data", {})
        if "source" in data and "target" in data:
            source = str(data.get("source"))
            target = str(data.get("target"))

            if source == clicked_id:
                neighbor_ids.add(target)
                edge_ids.add(str(data.get("id")))
            elif target == clicked_id:
                neighbor_ids.add(source)
                edge_ids.add(str(data.get("id")))

    stylesheet += [
        {"selector": "node", "style": {"opacity": 0.10}},
        {"selector": "edge", "style": {"opacity": 0.03}},
        {
            "selector": f'node[id = "{clicked_id}"]',
            "style": {
                "opacity": 1.0,
                "border-width": 5,
                "border-color": "#000000",
                "label": "data(ticker)",
                "font-size": 13,
                "z-index": 9999,
            },
        },
    ]

    for neighbor_id in neighbor_ids:
        stylesheet.append(
            {
                "selector": f'node[id = "{neighbor_id}"]',
                "style": {
                    "opacity": 0.95,
                    "border-width": 2,
                    "border-color": "#222222",
                    "label": "data(ticker)",
                    "font-size": 10,
                    "z-index": 9998,
                },
            }
        )

    for edge_id in edge_ids:
        stylesheet.append(
            {
                "selector": f'edge[id = "{edge_id}"]',
                "style": {
                    "opacity": 0.9,
                    "z-index": 9997,
                },
            }
        )

    return stylesheet


# ============================================================
# SNAPSHOT HELPERS
# ============================================================

def build_date_slider_marks(dates: list[pd.Timestamp], n_marks: int = 8) -> dict[int, str]:
    if not dates:
        return {}

    if len(dates) <= n_marks:
        return {i: pd.Timestamp(d).strftime("%Y-%m-%d") for i, d in enumerate(dates)}

    idxs = sorted(set(round(i * (len(dates) - 1) / (n_marks - 1)) for i in range(n_marks)))
    return {i: pd.Timestamp(dates[i]).strftime("%Y-%m-%d") for i in idxs}


def find_nearest_date_index(dates: list[pd.Timestamp], target_date) -> int:
    if not dates:
        raise ValueError("dates is empty.")

    target_ts = pd.Timestamp(target_date)
    date_index = pd.DatetimeIndex(dates)

    pos = date_index.searchsorted(target_ts)

    if pos <= 0:
        return 0
    if pos >= len(date_index):
        return len(date_index) - 1

    before = date_index[pos - 1]
    after = date_index[pos]

    if abs(target_ts - before) <= abs(after - target_ts):
        return pos - 1
    return pos


def parse_date_input_to_index(
    date_input_value: str | None,
    dates: list[pd.Timestamp],
    fallback_idx: int,
) -> int:
    if not date_input_value or not str(date_input_value).strip():
        return fallback_idx

    try:
        parsed = pd.Timestamp(str(date_input_value).strip())
    except Exception:
        return fallback_idx

    return find_nearest_date_index(dates, parsed)


def build_snapshot_bundle(
    data_manager: DataManager,
    builder: NetworkBuilder,
    date_idx: int,
    threshold: float,
    min_periods: int | None,
) -> dict:
    if data_manager.dates is None:
        raise ValueError("data_manager.dates is not built.")
    if data_manager.reference_layout is None:
        raise ValueError("data_manager.reference_layout is not built.")

    date_idx = int(date_idx)
    threshold = float(threshold)
    selected_date = pd.Timestamp(data_manager.dates[date_idx])

    nodes, edges = data_manager.get_graph_snapshot(
        date=selected_date,
        threshold=threshold,
        min_periods=min_periods,
    )

    positioned_nodes = nodes.merge(data_manager.reference_layout, on="permno", how="inner")

    valid_permnos = set(positioned_nodes["permno"].tolist())
    edges = edges.loc[
        edges["source"].isin(valid_permnos) & edges["target"].isin(valid_permnos)
    ].copy()

    elements, sector_color_map = builder.build_elements(
        nodes=positioned_nodes,
        edges=edges,
        threshold=threshold,
    )

    summary_stats = data_manager.get_graph_summary_stats(
        date=selected_date,
        threshold=threshold,
        min_periods=min_periods,
    )

    ranking_df = data_manager.get_node_ranking_table(
        date=selected_date,
        threshold=threshold,
        min_periods=min_periods,
    )

    summary_text = (
        f"Date: {selected_date:%Y-%m-%d} | "
        f"Nodes: {len(positioned_nodes):,} | "
        f"Edges: {len(edges):,} | "
        f"Threshold: {threshold:.2f}"
    )

    return {
        "date": selected_date,
        "nodes": positioned_nodes,
        "edges": edges,
        "elements": elements,
        "sector_color_map": sector_color_map,
        "summary_stats": summary_stats,
        "ranking_df": ranking_df,
        "summary_text": summary_text,
    }


def default_node_panel():
    return [
        html.H4("Selected node", style={"marginBottom": "10px", "marginTop": "0px"}),
        html.Div("Click a node to see its neighbors and correlations."),
    ]


def default_edge_panel():
    return [
        html.H4("Selected edge", style={"marginBottom": "10px", "marginTop": "0px"}),
        html.Div("Click an edge to see its details."),
    ]


# ============================================================
# APP
# ============================================================

def build_dynamic_network_app(data_manager: DataManager, configu: Config) -> Dash:
    appli = Dash(__name__)
    builder = NetworkBuilder(config=configu)

    if data_manager.dates is None or len(data_manager.dates) == 0:
        raise ValueError("data_manager.dates is empty or not built.")
    if data_manager.reference_layout is None:
        raise ValueError("reference_layout is not built.")

    default_date_idx = len(data_manager.dates) - 1
    default_threshold = float(configu.layout_threshold_ref)
    min_periods = configu.layout_min_periods

    initial = build_snapshot_bundle(
        data_manager=data_manager,
        builder=builder,
        date_idx=default_date_idx,
        threshold=default_threshold,
        min_periods=min_periods,
    )

    base_stylesheet = builder.get_base_stylesheet()
    date_marks = build_date_slider_marks(data_manager.dates, n_marks=8)

    appli.layout = html.Div(
        [
            # --- stores ---
            dcc.Store(id="graph-highlight-state", data={"mode": "none", "value": None}),

            # --- interval drives the animation clock ---
            dcc.Interval(
                id="animation-interval",
                interval=800,   # ms between steps — increase if snapshots are slow
                n_intervals=0,
                disabled=True,  # starts paused
            ),

            html.H3("Correlation Network"),
            html.Div(
                id="current-date-label",
                children=f"Selected date: {initial['date']:%Y-%m-%d}",
                style={"marginBottom": "8px"},
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Date"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="date-input",
                                        type="text",
                                        value=initial["date"].strftime("%Y-%m-%d"),
                                        placeholder="YYYY-MM-DD",
                                        debounce=True,
                                        style={
                                            "width": "160px",
                                            "padding": "8px",
                                            "fontSize": "13px",
                                            "marginRight": "12px",
                                            "boxSizing": "border-box",
                                        },
                                    ),
                                    html.Button(
                                        "Play",
                                        id="play-pause-btn",
                                        n_clicks=0,
                                        style={
                                            "padding": "8px 14px",
                                            "fontSize": "13px",
                                            "fontWeight": "bold",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "marginBottom": "10px",
                                    "gap": "8px",
                                },
                            ),
                            dcc.Slider(
                                id="date-slider",
                                min=0,
                                max=len(data_manager.dates) - 1,
                                step=1,
                                value=default_date_idx,
                                marks=date_marks,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={"marginBottom": "18px"},
                    ),
                    html.Div(
                        [
                            html.Label("Threshold"),
                            dcc.Slider(
                                id="threshold-slider",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=default_threshold,
                                marks={0.0: "0.00", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.00"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),
                ]
            ),

            html.Div(
                id="graph-summary-line",
                children=initial["summary_text"],
                style={"marginBottom": "16px"},
            ),

            html.Div(
                [
                    html.Div(
                        [
                            cyto.Cytoscape(
                                id="network-graph",
                                elements=initial["elements"],
                                stylesheet=base_stylesheet,
                                layout={"name": "preset"},
                                style={"width": "100%", "height": "1400px"},
                                minZoom=0.1,
                                maxZoom=5.0,
                                wheelSensitivity=0.2,
                            )
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Div(id="legend-container", children=build_sector_legend(initial["sector_color_map"])),
                            html.Div(id="summary-panel-container", children=build_graph_summary_panel(initial["summary_stats"])),
                            build_search_panel(),
                            build_reset_panel(),
                            build_edge_info_panel(),
                            build_node_info_panel(),
                        ],
                        style={
                            "width": "280px",
                            "marginLeft": "20px",
                            "flexShrink": "0",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start"},
            ),

            html.Div(
                id="ranking-table-container",
                children=build_ranking_table(initial["ranking_df"]),
            ),
        ],
        style={"padding": "10px"},
    )

    # --------------------------------------------------------
    # Play / pause — just flips the interval on or off
    # --------------------------------------------------------
    @appli.callback(
        Output("animation-interval", "disabled"),
        Output("play-pause-btn", "children"),
        Input("play-pause-btn", "n_clicks"),
        State("animation-interval", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_animation(n_clicks, is_disabled):
        # is_disabled=True means we were paused → now play, and vice-versa
        now_playing = bool(is_disabled)
        return not now_playing, "Pause" if now_playing else "Play"

    # --------------------------------------------------------
    # Date input → slider sync
    # --------------------------------------------------------
    @appli.callback(
        Output("date-slider", "value", allow_duplicate=True),
        Input("date-input", "value"),
        State("date-slider", "value"),
        prevent_initial_call=True,
    )
    def sync_date_input_to_slider(date_input_value, current_slider_idx):
        fallback_idx = int(current_slider_idx) if current_slider_idx is not None else default_date_idx
        return parse_date_input_to_index(
            date_input_value=date_input_value,
            dates=data_manager.dates,
            fallback_idx=fallback_idx,
        )

    # --------------------------------------------------------
    # Main snapshot callback
    # Triggered by: slider drag, threshold change, or interval tick
    # --------------------------------------------------------
    @appli.callback(
        Output("network-graph", "elements"),
        Output("graph-summary-line", "children"),
        Output("current-date-label", "children"),
        Output("date-input", "value"),           # keeps text box in sync
        Output("date-slider", "value"),          # keeps slider in sync when interval fires
        Output("legend-container", "children"),
        Output("summary-panel-container", "children"),
        Output("ranking-table-container", "children"),
        Output("graph-highlight-state", "data"),
        Output("selected-edge-panel", "children"),
        Output("selected-node-panel", "children"),
        Input("date-slider", "value"),
        Input("threshold-slider", "value"),
        Input("animation-interval", "n_intervals"),
        State("animation-interval", "disabled"),
        prevent_initial_call=True,
    )
    def update_snapshot(date_idx, threshold, n_intervals, interval_disabled):
        from dash import callback_context

        triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        # when the interval fires and animation is running, advance by one step
        if triggered_id == "animation-interval" and not interval_disabled:
            date_idx = (int(date_idx) + 1) % len(data_manager.dates)
        else:
            date_idx = int(date_idx)

        snap = build_snapshot_bundle(
            data_manager=data_manager,
            builder=builder,
            date_idx=date_idx,
            threshold=float(threshold),
            min_periods=min_periods,
        )

        date_str = snap["date"].strftime("%Y-%m-%d")

        return (
            snap["elements"],
            snap["summary_text"],
            f"Selected date: {date_str}",
            date_str,
            date_idx,
            build_sector_legend(snap["sector_color_map"]),
            build_graph_summary_panel(snap["summary_stats"]),
            build_ranking_table(snap["ranking_df"]),
            {"mode": "none", "value": None},
            default_edge_panel(),
            default_node_panel(),
        )

    # --------------------------------------------------------
    # Highlight state callback
    # --------------------------------------------------------
    @appli.callback(
        Output("graph-highlight-state", "data", allow_duplicate=True),
        Input("network-graph", "tapNodeData"),
        Input("ticker-search-input", "value"),
        Input("reset-highlight-btn", "n_clicks"),
        State("graph-highlight-state", "data"),
        prevent_initial_call=True,
    )
    def update_highlight_state(clicked_node_data, searched_ticker, reset_clicks, current_state):
        from dash import callback_context

        triggered = callback_context.triggered
        if not triggered:
            return current_state

        trigger_id = triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "reset-highlight-btn" and reset_clicks:
            return {"mode": "none", "value": None}

        if trigger_id == "network-graph" and clicked_node_data:
            return {"mode": "node", "value": clicked_node_data}

        if trigger_id == "ticker-search-input":
            if searched_ticker and searched_ticker.strip():
                return {"mode": "ticker", "value": searched_ticker.strip()}
            return {"mode": "none", "value": None}

        return current_state

    # --------------------------------------------------------
    # Stylesheet callback
    # --------------------------------------------------------
    @appli.callback(
        Output("network-graph", "stylesheet"),
        Input("graph-highlight-state", "data"),
        State("network-graph", "elements"),
    )
    def update_graph_stylesheet(highlight_state, current_elements):
        if not highlight_state:
            return base_stylesheet

        mode = highlight_state.get("mode")
        value = highlight_state.get("value")

        if mode == "node" and value:
            return build_node_neighborhood_stylesheet(
                base_stylesheet=base_stylesheet,
                clicked_node_data=value,
                elements=current_elements,
            )

        if mode == "ticker" and value:
            return build_highlight_stylesheet(
                base_stylesheet=base_stylesheet,
                searched_ticker=value,
            )

        return base_stylesheet

    # --------------------------------------------------------
    # Edge panel callback
    # --------------------------------------------------------
    @appli.callback(
        Output("selected-edge-panel", "children", allow_duplicate=True),
        Input("network-graph", "tapEdgeData"),
        prevent_initial_call=True,
    )
    def update_selected_edge_panel(edge_data):
        if not edge_data:
            return default_edge_panel()

        return [
            html.H4("Selected edge", style={"marginBottom": "10px", "marginTop": "0px"}),
            html.Table(
                [
                    html.Tbody(
                        [
                            html.Tr([html.Td("Source", style={"fontWeight": "bold"}), html.Td(edge_data.get("source_ticker", "NA"))]),
                            html.Tr([html.Td("Target", style={"fontWeight": "bold"}), html.Td(edge_data.get("target_ticker", "NA"))]),
                            html.Tr([html.Td("Correlation", style={"fontWeight": "bold"}), html.Td(f"{edge_data.get('corr', float('nan')):.4f}")]),
                            html.Tr([html.Td("|Correlation|", style={"fontWeight": "bold"}), html.Td(f"{edge_data.get('abs_corr', float('nan')):.4f}")]),
                        ]
                    )
                ],
                style={"width": "100%"},
            ),
        ]

    # --------------------------------------------------------
    # Node panel callback
    # --------------------------------------------------------
    @appli.callback(
        Output("selected-node-panel", "children", allow_duplicate=True),
        Input("network-graph", "tapNodeData"),
        State("date-slider", "value"),
        State("threshold-slider", "value"),
        prevent_initial_call=True,
    )
    def update_selected_node_panel(node_data, date_idx, threshold):
        if not node_data:
            return default_node_panel()

        snap = build_snapshot_bundle(
            data_manager=data_manager,
            builder=builder,
            date_idx=int(date_idx),
            threshold=float(threshold),
            min_periods=min_periods,
        )

        nodes = snap["nodes"]
        selected_date = snap["date"]

        permno = int(node_data.get("permno"))
        ticker = node_data.get("ticker", "NA")
        sector = node_data.get("sector", "NA")

        selected_node_row = nodes.loc[nodes["permno"] == permno].copy()

        if selected_node_row.empty:
            degree = "NA"
            weighted_degree = "NA"
            momentum = "NA"
        else:
            degree_val = selected_node_row["degree"].iloc[0] if "degree" in selected_node_row.columns else None
            weighted_degree_val = selected_node_row["weighted_degree"].iloc[0] if "weighted_degree" in selected_node_row.columns else None
            momentum_val = selected_node_row["momentum_xm"].iloc[0] if "momentum_xm" in selected_node_row.columns else None

            degree = "NA" if pd.isna(degree_val) else f"{int(degree_val)}"
            weighted_degree = "NA" if pd.isna(weighted_degree_val) else f"{float(weighted_degree_val):.3f}"
            momentum = "NA" if pd.isna(momentum_val) else f"{float(momentum_val):.4f}"

        neighbors_df = data_manager.get_node_neighbors_table(
            date=selected_date,
            permno=permno,
            threshold=float(threshold),
            min_periods=min_periods,
        )

        return [
            html.H4("Selected node", style={"marginBottom": "10px", "marginTop": "0px"}),
            html.Table(
                [
                    html.Tbody(
                        [
                            html.Tr([html.Td("Ticker", style={"fontWeight": "bold"}), html.Td(ticker)]),
                            html.Tr([html.Td("Sector", style={"fontWeight": "bold"}), html.Td(sector)]),
                            html.Tr([html.Td("Degree", style={"fontWeight": "bold"}), html.Td(degree)]),
                            html.Tr([html.Td("Weighted degree", style={"fontWeight": "bold"}), html.Td(weighted_degree)]),
                            html.Tr([html.Td("Momentum", style={"fontWeight": "bold"}), html.Td(momentum)]),
                        ]
                    )
                ],
                style={"width": "100%"},
            ),
            build_neighbors_table(neighbors_df),
        ]

    return appli


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = Config()

    dm = DataManager(config=config)
    dm.load_data()
    dm.build_reference_network_layout()

    app = build_dynamic_network_app(dm, configu=config)
    app.run(debug=True, use_reloader=False)