from abc import ABC, abstractmethod
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from mat_dp_pipeline import pipeline
from mat_dp_pipeline.pipeline import LabelledOutput, PipelineOutput
from mat_dp_pipeline.standard_data_format import Year


class EmissionsPlotter(ABC):
    _data: PipelineOutput

    def __init__(self, data: PipelineOutput):
        self._data = data

    @property
    def data(self):
        return self._data

    @abstractmethod
    def __call__(self, country: str, indicator: str):
        ...


class EmissionsByMaterialPlotter(EmissionsPlotter):
    def __call__(self, country: str, indicator: str):
        emissions = (
            self.data.emissions(country, indicator)
            .drop(columns=["Category", "Specific"])
            .groupby("Year")
            .sum()
            .rename_axis("Resource", axis=1)
        )
        return px.area(emissions, labels={"value": indicator})


class EmissionsByTechPlotter(EmissionsPlotter):
    def __call__(self, country: str, indicator: str):
        emissions = self.data.emissions(country, indicator)
        emissions["Tech"] = emissions["Category"] + "/" + emissions["Specific"]
        # Emissions will be a data frame with index of Techs and columns Resources
        # The values are individual emissions per given tech/resource
        emissions = (
            emissions.drop(columns=["Category", "Specific"])
            .set_index("Tech")
            .groupby("Tech")
            .sum()
            .drop(columns="Year")
            .rename_axis("Resource", axis=1)
        )
        return px.bar(
            emissions,
            x=emissions.columns,
            y=emissions.index,
            labels={"value": indicator},
        )


class EmissionsByResourcesPlotter(EmissionsPlotter):
    def __call__(self, country: str, indicator: str):
        emissions = (
            self.data.emissions(country, indicator)
            .drop(columns=["Year", "Category", "Specific"])
            .sum()
            .to_frame(indicator)
            .rename_axis("Resource", axis=0)
            .reset_index()
        )
        return px.bar(emissions, x=indicator, y="Resource", color="Resource")


def emissions_by_resources_fig(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    return EmissionsByResourcesPlotter(data)(country, indicator)


def emissions_by_tech_fig(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    return EmissionsByTechPlotter(data)(country, indicator)


def emissions_by_material_fig(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    return EmissionsByMaterialPlotter(data)(country, indicator)


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
        # TODO: self.tabs with list of tuple[title, Plotter]
        self.tabs = {
            "emissions_by_material": (
                "Emissions by material",
                EmissionsByMaterialPlotter(outputs),
            ),
            "emissions_by_tech": ("Emissions by tech", EmissionsByTechPlotter(outputs)),
            "emissions_by_resources": (
                "Emissions by resources",
                EmissionsByResourcesPlotter(outputs),
            ),
        }
        self.selected_outputs = None

    def controls(self):
        countries = sorted(str(p) for p in set(self.outputs.keys(Path)))
        indicators = ["CO2"]  # TODO:
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
                    dbc.Tab(label=title, tab_id=tab_id)
                    for tab_id, (title, _) in self.tabs.items()
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

    def tab_layout(self, title: str, fig: go.Figure):
        return [html.H2(title), dcc.Graph(figure=fig)]

    def render_graph_tab(self, tab: str, country: str, indicator: str):
        if not indicator:  # TODO: or country not in outputs
            raise PreventUpdate

        title, plotter = self.tabs[tab]
        return self.tab_layout(title, fig=plotter(country, indicator))

    def register_callback(self, fn, *spec):
        self.dash_app.callback(*spec)(fn)

    def serve(self):
        self.dash_app.layout = self.layout()
        self.dash_app.run_server()


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "test_data/World"
    output = pipeline(path)
    App(output).serve()
