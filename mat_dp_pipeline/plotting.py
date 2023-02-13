from typing import Callable

import plotly.express as px
import plotly.graph_objects as go

from mat_dp_pipeline.pipeline import PipelineOutput

EmissionsPlotter = Callable[[PipelineOutput, str, str], go.Figure]


def emissions_by_material(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    emissions = (
        data.emissions(country, indicator)
        .drop(columns=["Category", "Specific"])
        .groupby("Year")
        .sum()
        .rename_axis("Resource", axis=1)
    )
    return px.area(emissions, labels={"value": indicator})


def emissions_by_tech(data: PipelineOutput, country: str, indicator: str) -> go.Figure:
    emissions = data.emissions(country, indicator)
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


def emissions_by_resources(
    data: PipelineOutput, country: str, indicator: str
) -> go.Figure:
    emissions = (
        data.emissions(country, indicator)
        .drop(columns=["Year", "Category", "Specific"])
        .sum()
        .to_frame(indicator)
        .rename_axis("Resource", axis=0)
        .reset_index()
    )
    return px.bar(emissions, x=indicator, y="Resource", color="Resource")
