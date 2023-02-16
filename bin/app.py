from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from mat_dp_pipeline import pipeline
from mat_dp_pipeline.pipeline import LabelledOutput, PipelineOutput
from mat_dp_pipeline.plotting import (
    EmissionsPlotter,
    emissions_by_material,
    emissions_by_resources,
    emissions_by_tech,
)
from mat_dp_pipeline.standard_data_format import Year


class App:
    dash_app: Dash
    outputs: PipelineOutput
    selected_outputs: dict[Year, LabelledOutput] | None

    def __init__(self, outputs: PipelineOutput):
        self.dash_app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.register_callback(
            self.render_graph_tab,
            Output("tab", "children"),
            Input("tabs", "active_tab"),
            Input("country", "value"),
            Input("indicator", "value"),
        )

        self.outputs = outputs
        self.tabs = [
            ("Emissions by material", emissions_by_material),
            ("Emissions by tech", emissions_by_tech),
            (
                "Emissions by resources",
                emissions_by_resources,
            ),
        ]
        self.selected_outputs = None

    def _tab(self, tab_id: str) -> tuple[str, EmissionsPlotter]:
        # tab_id's are strings "0", "1", ...
        return self.tabs[int(tab_id)]

    def controls(self):
        countries = sorted(str(p) for p in set(self.outputs.keys(Path)))
        indicators = sorted(self.outputs.indicators)
        body = dbc.CardBody(
            [
                html.Div(
                    [
                        html.Label("Country"),
                        dcc.Dropdown(countries, id="country"),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Indicator"),
                        dcc.Dropdown(indicators, id="indicator"),
                    ]
                ),
            ]
        )

        return dbc.Card([dbc.CardHeader("Data selection"), body], body=True)

    def plots(self):
        return [
            dbc.Tabs(
                [
                    # Encode tab_id's as "0", "1", etc., so that each tab is easy to access from a list
                    dbc.Tab(label=title, tab_id=str(i))
                    for i, (title, _) in enumerate(self.tabs)
                ],
                id="tabs",
            ),
            html.Div(id="tab"),
        ]

    def layout(self):
        return dbc.Container(
            [
                html.H1("Mat DP"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(self.controls(), md=4, align="start"),
                        dbc.Col(self.plots(), md=8),
                    ],
                    align="center",
                ),
            ],
            fluid=True,
        )

    def render_graph_tab(self, tab_id: str, country: str, indicator: str):
        if not indicator:  # TODO: or country not in outputs
            raise PreventUpdate

        title, plotter = self._tab(tab_id)
        fig = plotter(self.outputs, country, indicator)
        return [html.H2(title), dcc.Graph(figure=fig, style={"height": "75vh"})]

    def register_callback(self, fn, *spec):
        self.dash_app.callback(*spec)(fn)

    def serve(self):
        self.dash_app.layout = self.layout()
        self.dash_app.run_server()


if __name__ == "__main__":
    # path = Path(__file__).parent.parent / "test_data/World"
    path = Path("/tmp/sdf/General")
    output = pipeline(path)
    print("Processed.")
    App(output).serve()
