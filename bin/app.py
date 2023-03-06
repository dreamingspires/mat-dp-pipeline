from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from mat_dp_pipeline import pipeline
from mat_dp_pipeline.data_sources.mat_dp_db import MatDpDB
from mat_dp_pipeline.pipeline import LabelledOutput, PipelineOutput
from mat_dp_pipeline.plotting import (
    indicator_by_resource_agg,
    indicator_by_resource_over_years,
    indicator_by_tech_agg,
    required_resources_agg,
    required_resources_by_tech_agg,
    required_resources_over_years,
)
from mat_dp_pipeline.standard_data_format import Year

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


class App:
    dash_app: Dash
    outputs: PipelineOutput

    def __init__(self, outputs: PipelineOutput):
        self.dash_app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.outputs = outputs
        self.indicators = sorted(self.outputs.indicators)

        self.register_callback(
            self.render_tab_content,
            Output("tab-content", "children"),
            Input("tabs", "active_tab"),
            Input("country", "value"),
        )

    def layout(self):
        countries = sorted(str(p) for p in set(self.outputs.keys(Path)))

        country_input = html.Div(
            [
                html.Label("Country"),
                dcc.Dropdown(countries, id="country"),
            ]
        )

        sidebar = html.Div(
            [
                html.H2("Menu", className="display-4"),
                html.Hr(),
                html.P("Choose appropriate options", className="lead"),
                html.Hr(),
                country_input,
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

    def render_tab_content(self, active_tab: str, country: str | None):
        if not country:
            raise PreventUpdate

        is_indicator_tab = active_tab.startswith("ind_")
        if is_indicator_tab:
            ind_idx = int(active_tab.split("_")[1])
            indicator = self.indicators[ind_idx]
            plots = self.generate_indicator_graphs(country, indicator)
        else:  # materials tab
            plots = self.generate_materials_graphs(country)

        return [dcc.Graph(figure=fig, style={"height": "25vh"}) for fig in plots]

    def generate_materials_graphs(self, country: str) -> list[go.Figure]:
        return [
            required_resources_over_years(self.outputs, country),
            required_resources_by_tech_agg(self.outputs, country),
            required_resources_agg(self.outputs, country),
        ]

    def generate_indicator_graphs(
        self, country: str, indicator: str
    ) -> list[go.Figure]:
        return [
            indicator_by_resource_agg(self.outputs, country, indicator),
            indicator_by_resource_over_years(self.outputs, country, indicator),
            indicator_by_tech_agg(self.outputs, country, indicator),
        ]

    def register_callback(self, fn, *spec):
        self.dash_app.callback(*spec)(fn)

    def serve(self):
        self.dash_app.layout = self.layout()
        self.dash_app.run_server(debug=True)


def main():
    path = Path(__file__).parent.parent / "test_data/World"
    # path = Path("/tmp/sdf/General")
    # MATERIALS_EXCEL = "scratch/Material_intensities_database.xlsx"
    # TARGETS_CSV = "scratch/results_1.5deg.csv"
    # TARGETS_PARAMETER = "Power Generation (Aggregate)"

    # ds = MatDpDB(Path(MATERIALS_EXCEL), Path(TARGETS_CSV), TARGETS_PARAMETER)
    output = pipeline(path)
    print("Processed.")
    App(output).serve()


if __name__ == "__main__":
    main()
