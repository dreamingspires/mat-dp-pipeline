import os.path
import typing
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from mat_dp_pipeline.pipeline import PipelineOutput
from mat_dp_pipeline.presentation.plotting import (
    indicator_by_resource_agg,
    indicator_by_resource_over_years,
    indicator_by_tech_agg,
    required_resources_agg,
    required_resources_by_tech_agg,
    required_resources_over_years,
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "26rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "28rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


class App:
    dash_app: Dash
    outputs: PipelineOutput

    def __init__(self, outputs: PipelineOutput, path_leaf_split: list[str]):
        self.dash_app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.outputs = outputs
        self.indicators = sorted(self.outputs.indicators)
        self.paths = typing.cast(list[Path], sorted(self.outputs.keys(Path)))

        self.path_leaf_split = path_leaf_split

        path_inputs = [Input("country", "value")] + [
            Input(f"dropdown_{i}", "value") for i in range(len(self.path_leaf_split))
        ]
        self.register_callback(
            self.render_tab_content,
            Output("tab-content", "children"),
            Input("tabs", "active_tab"),
            path_inputs,
        )

    def _country_dropdown(self) -> dcc.Dropdown:
        """
        * Trim the lowest common ancestor from all the paths' labels (not values!)
        * Trim the lowest levels' bits (as defined by self.path_leaf_split) - these will
          form other drop downs
        """
        lowest_common_ancestor = os.path.commonpath(self.paths)
        levels_to_trim = len(self.path_leaf_split)
        if levels_to_trim:
            paths = sorted({Path(*p.parts[:-levels_to_trim]) for p in self.paths})
        else:
            paths = sorted(self.paths)
        return dcc.Dropdown(
            options={str(p): str(p.relative_to(lowest_common_ancestor)) for p in paths},
            id="country",
        )

    def _leaf_dropdowns(self) -> list[tuple[str, dcc.Dropdown]]:
        levels_to_trim = len(self.path_leaf_split)
        dropdowns = []
        for i, category in enumerate(self.path_leaf_split):
            # Take only a set of parts of the paths on the lower levels,
            # starting from the highest one, moving towards to the leaf
            paths = sorted(
                {p.parts[len(p.parts) - (levels_to_trim - i)] for p in self.paths}
            )
            dropdowns.append((category, dcc.Dropdown(paths, id=f"dropdown_{i}")))
        return dropdowns

    def layout(self):
        leaf_dropdowns = [
            html.Div([html.Label(label), dd]) for label, dd in self._leaf_dropdowns()
        ]
        sidebar = html.Div(
            [
                html.H2("Menu", className="display-4"),
                html.Hr(),
                html.P("Choose appropriate options", className="lead"),
                html.Hr(),
                html.Div([html.Label("Country"), self._country_dropdown()]),
                *leaf_dropdowns,
                # # Hidden input with full path. Filled and used by the callbacks
                # dcc.Input(type="hidden", id="path", value=""),
            ],
            style=SIDEBAR_STYLE,
        )

        tabs = [dbc.Tab(label="Materials", tab_id="materials")]
        for i, ind in enumerate(self.indicators):
            tabs.append(dbc.Tab(label=ind, tab_id=f"ind_{i}"))

        content = html.Div(
            [
                dcc.Store(id="store"),
                html.H1("MAT-DP Pipeline Results"),
                html.Hr(),
                dbc.Tabs(tabs, id="tabs", active_tab="materials"),
                html.Div(id="tab-content", className="p-4"),
            ],
            style=CONTENT_STYLE,
        )

        return html.Div([sidebar, content])

    def render_tab_content(self, active_tab: str, *path_parts):
        # Path has missing levels - do not update
        if any(not p for p in path_parts):
            raise PreventUpdate

        path = Path(*path_parts)
        is_indicator_tab = active_tab.startswith("ind_")
        if is_indicator_tab:
            ind_idx = int(active_tab.split("_")[1])
            indicator = self.indicators[ind_idx]
            plots = self.generate_indicator_graphs(path, indicator)
        else:  # materials tab
            plots = self.generate_materials_graphs(path)

        return [dcc.Graph(figure=fig, style={"height": "25vh"}) for fig in plots]

    def generate_materials_graphs(self, path: Path) -> list[go.Figure]:
        return [
            required_resources_over_years(self.outputs, path),
            required_resources_by_tech_agg(self.outputs, path),
            required_resources_agg(self.outputs, path),
        ]

    def generate_indicator_graphs(self, path: Path, indicator: str) -> list[go.Figure]:
        return [
            indicator_by_resource_over_years(self.outputs, path, indicator),
            indicator_by_resource_agg(self.outputs, path, indicator),
            indicator_by_tech_agg(self.outputs, path, indicator),
        ]

    def register_callback(self, fn, *spec):
        self.dash_app.callback(*spec)(fn)

    def serve(self):
        self.dash_app.layout = self.layout()
        self.dash_app.run_server(debug=False)
