from typing import Callable

import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from mat_dp_pipeline.pipeline import PipelineOutput

EmissionsPlotter = Callable[[PipelineOutput, str, str], go.Figure]


def x_log_switch():
    # Add log/linear x-axis switch
    return dict(
        type="buttons",
        direction="right",
        x=0.1,
        y=1.1,
        buttons=[
            dict(
                args=[{"xaxis.type": "linear"}],
                label="Linear Scale",
                method="relayout",
            ),
            dict(
                args=[{"xaxis.type": "log"}],
                label="Log Scale",
                method="relayout",
            ),
        ],
    )


def emissions_by_material(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    emissions = (
        data.emissions(country, indicator)
        .groupby("Year")
        .sum()
        .rename_axis("Resource", axis=1)
    )
    # Drop resources with only 0s and sorr columns, so that the resources
    # generating the most emissions are first.
    emissions = emissions.loc[:, (emissions != 0).any(axis=0)].sort_index(
        axis=1, ascending=False, key=lambda c: emissions[c].max()
    )
    fig = px.area(
        emissions,
        labels={"value": indicator},
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    fig.update_traces(hovertemplate="%{x}: %{y}")
    return fig


def emissions_by_tech(data: PipelineOutput, country: str, indicator: str) -> go.Figure:
    emissions = data.emissions(country, indicator).dropna(how="all").reset_index()
    emissions["Tech"] = emissions["Category"] + "/" + emissions["Specific"]
    # Emissions will be a data frame with index of Techs and columns Resources
    # The values are individual emissions per given tech/resource
    emissions = (
        emissions.drop(columns=["Category", "Specific"])
        .groupby("Tech")
        .sum()
        .drop(columns="Year")
    )
    fig = px.bar(
        emissions,
        x=emissions.columns,
        y=emissions.index,
        labels={"value": indicator},
    )
    fig.update_layout(
        updatemenus=[x_log_switch()], yaxis={"categoryorder": "total ascending"}
    )
    return fig


def emissions_by_resources(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    emissions = (
        data.emissions(country, indicator)
        .reset_index(drop=True)
        .sum()
        .replace(0, np.nan)
        .dropna()
        .sort_values(ascending=False)
        .to_frame(indicator)
        .reset_index()
    )

    fig = px.bar(emissions, x=indicator, y="Resource", color="Resource")
    fig.update_layout(updatemenus=[x_log_switch()])
    return fig
