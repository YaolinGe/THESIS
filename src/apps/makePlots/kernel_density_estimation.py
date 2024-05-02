"""
Plot the kernel density estimation of the input list of values.

Parameters
----------
values : list of float
    The list of values to plot the kernel density estimation of.

Returns
-------
plotly.graph_objects.Figure
    The plotly figure object containing the kernel density estimation of the input list of values.
"""

import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde

def plot(values):
    """
    Plot the kernel density estimation of the input list of values.

    Parameters
    ----------
    values : list of float
        The list of values to plot the kernel density estimation of.

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure object containing the kernel density estimation of the input list of values.
    """
    kde = gaussian_kde(values)
    x = np.linspace(min(values), max(values), 1000)
    y = kde(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='red')))
    mean = np.mean(values)
    std = np.std(values)
    fig.add_shape(
        type="line",
        x0=mean,
        x1=mean,
        y0=0,
        y1=max(y),
        line=dict(color="white", width=1, dash="dash")
    )
    fig.add_annotation(
        x=mean,
        y=max(y),
        text=f"\u03BC: {mean:.2f}\n\n\n\u03C3: {std:.2f}",
        showarrow=False,
        font=dict(size=24),
        xanchor="center",
        yanchor="bottom",
        xshift=0,
        yshift=-10
    )
    fig.update_layout(
        width=500,
        height=550,
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