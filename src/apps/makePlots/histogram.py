"""
Plot the kernel density estimation and histogram of the input list of values.

Parameters
----------
values : list of float
    The list of values to plot the kernel density estimation and histogram of.

Returns
-------
plotly.graph_objects.Figure
    The plotly figure object containing the kernel density estimation and histogram of the input list of values.
"""

import plotly.graph_objects as go
import numpy as np


def plot(values):
    """
    Plot the kernel density estimation and histogram of the input list of values.

    Parameters
    ----------
    values : list of float
        The list of values to plot the kernel density estimation and histogram of.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure object containing the kernel density estimation and histogram of the input list of values.
    """
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=10, histnorm='density', opacity=0.6, marker_color='green')])
    fig.update_layout(
        width=500,
        height=700,
        autosize=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            title="Values",
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            title="Density",
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        showlegend=False
    )
    return fig


