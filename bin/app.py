from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from mat_dp_pipeline import Pipeline
from mat_dp_pipeline.pipeline import LabelledOutput, PipelineOutput
from mat_dp_pipeline.standard_data_format import Year


def required_resources_fig(data: LabelledOutput) -> go.Figure:
    resources = data.required_resources.reset_index()
    resources["Tech"] = resources["Category"] + "/" + resources["Specific"]
    resources = resources.drop(columns=["Category", "Specific"])
    resources.set_index("Tech", inplace=True)
    return px.bar(
        resources, x=resources.columns, y=resources.index, labels={"value": "Quantity"}
    )


def emissions_by_tech_fig(data: LabelledOutput) -> go.Figure:
    # TODO:
    return required_resources_fig(data)


def emissions_by_material_fig(data: LabelledOutput) -> go.Figure:
    emissions = data.emissions.dropna().reset_index()
    emissions["CO3"] = emissions["CO2"] * 2
    print(emissions)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            emissions,
            x=["CO2", "CO3"],
            y="Resource",
        )
    )
    fig.update_layout(scattermode="group")
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig


class App:
    dash_app: Dash
    outputs: dict[tuple[str, Year], LabelledOutput]
    selected_output: LabelledOutput | None

    def __init__(self, outputs: PipelineOutput):
        self.dash_app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.register_callback(
            self.render_graph_tab,
            Output("tab", "children"),
            Input("tabs", "active_tab"),
            Input("country", "value"),
            Input("year", "value"),
        )

        self.outputs = {(str(o.path), o.year): o for o in outputs.data}
        assert len(self.outputs) == len(outputs.data), "Duplicated keys found!"
        self.selected_output = None

    def controls(self):
        countries, years = zip(*self.outputs.keys()) if self.outputs else ([], [])
        # Unique only
        countries = sorted(set(countries))
        years = sorted(set(years))
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
                        html.Label("Year"),
                        dcc.Dropdown(years, id="year"),
                    ]
                ),
            ]
        )

        return dbc.Card([dbc.CardHeader("Data selection"), body], body=True)

    def plots(self):
        return [
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Emissions by material",
                        tab_id="emissions_by_material",
                    ),
                    dbc.Tab(
                        label="Emissions by tech",
                        tab_id="emissions_by_tech",
                    ),
                    dbc.Tab(
                        label="Required resources",
                        tab_id="required_resources",
                    ),
                ],
                id="tabs",
                active_tab="emissions_by_material",
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

    def tab_layout(self, title: str, fig: go.Figure):
        return [html.H2(title), dcc.Graph(figure=fig)]

    def render_graph_tab(self, tab, country, year):
        self.selected_output = self.outputs.get((country, year))
        if not self.selected_output:
            raise PreventUpdate

        if tab == "emissions_by_material":
            fig = emissions_by_material_fig(self.selected_output)
            return self.tab_layout("Emissions from material production", fig)
        elif tab == "emissions_by_tech":
            fig = emissions_by_tech_fig(self.selected_output)
            return self.tab_layout("Emissions by technolgy", fig)
        elif tab == "required_resources":
            fig = required_resources_fig(self.selected_output)
            return self.tab_layout("Required resources", fig)

    def register_callback(self, fn, *spec):
        self.dash_app.callback(*spec)(fn)

    def serve(self):
        self.dash_app.layout = self.layout()
        self.dash_app.run_server(debug=True)


if __name__ == "__main__":
    path = Path("tests/data/World")
    outputs = Pipeline(path).process()
    App(outputs).serve()
