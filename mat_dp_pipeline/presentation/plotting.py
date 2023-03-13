from pathlib import Path
from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from mat_dp_pipeline.pipeline import PipelineOutput
from mat_dp_pipeline.sdf import Year

IndicatorPlotter = Callable[[PipelineOutput, str, str], go.Figure]
RequiredResourcesPlotter = Callable[[PipelineOutput, str], go.Figure]


def x_log_switch():
    # Add log/linear x-axis switch
    return dict(
        type="buttons",
        direction="right",
        x=0.8,
        y=1.2,
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


def indicator_by_resource_over_years(
    data: PipelineOutput, path: Path, indicator: str
) -> go.Figure:
    emissions = (
        data.emissions(path, indicator)
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
    fig.update_layout(
        title="Emissions from material production",
        title_font_size=24,
    )
    return fig


def indicator_by_tech_agg(
    data: PipelineOutput, path: Path, indicator: str
) -> go.Figure:
    emissions = data.emissions(path, indicator).dropna(how="all").reset_index()
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
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    years = sorted(data.keys(Year))
    fig.update_layout(
        title=f"Emissions by technology ({years[0]}-{years[-1]})",
        title_font_size=24,
        updatemenus=[x_log_switch()],
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def indicator_by_resource_agg(
    data: PipelineOutput, path: Path, indicator: str
) -> go.Figure:
    emissions = (
        data.emissions(path, indicator)
        .reset_index(drop=True)
        .sum()
        .replace(0, np.nan)
        .dropna()
        .sort_values(ascending=False)
        .to_frame(indicator)
        .reset_index()
    )

    fig = px.bar(
        emissions,
        x=indicator,
        y="Resource",
        color="Resource",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
    )
    years = sorted(data.keys(Year))
    fig.update_layout(
        title=f"Emissions by technology ({years[0]}-{years[-1]})",
        title_font_size=24,
        updatemenus=[x_log_switch()],
    )
    return fig


def required_resources_over_years(data: PipelineOutput, path: Path) -> go.Figure:
    materials = (
        data.resources(path).groupby("Year").sum().reset_index().set_index("Year")
    )
    materials = materials.loc[:, (materials != 0).any(axis=0)]
    fig = px.area(materials, labels={"value": "Kg"})  # TODO: !!!! UNITS!!!
    fig.update_traces(hovertemplate="%{x}: %{y}")
    fig.update_layout(title="Required resources", title_font_size=24)
    return fig


def required_resources_by_tech_agg(data: PipelineOutput, path: Path) -> go.Figure:
    materials = data.resources(path).reset_index()
    materials["Tech"] = materials["Category"] + "/" + materials["Specific"]
    materials = materials.drop(columns=["Category", "Specific"])
    materials = materials.set_index("Year").groupby("Tech").sum()
    materials = materials.loc[:, (materials != 0).any(axis=0)]
    materials = materials.loc[~(materials == 0).all(axis=1)]

    fig = px.bar(
        materials,
        x=materials.columns,
        y=materials.index,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        labels={"value": "Kg"},  # TODO: !!!! UNITS!!!
    )
    years = sorted(data.keys(Year))
    fig.update_layout(
        title=f"Materials production by technology ({years[0]}-{years[-1]})",
        title_font_size=24,
        updatemenus=[x_log_switch()],
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def required_resources_agg(data: PipelineOutput, path: Path):
    materials = data.resources(path).sum()
    materials = materials[materials > 0]
    years = sorted(data.keys(Year))
    fig = px.bar(
        materials,
        x=materials,
        y=materials.index,
        color=materials.index,
        labels={"value": "Kg"},  # TODO: units!!!
    )
    fig.update_layout(
        title=f"Materials production ({years[0]}-{years[-1]})",
        title_font_size=24,
        updatemenus=[x_log_switch()],
        yaxis={"categoryorder": "total ascending"},
    )
    return fig
