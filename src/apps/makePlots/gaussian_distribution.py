"""
This script is used to produce a Gaussian distribution probability density function plot given the mean and variance and a threshold value.

Parameters:
-----------
mu: float
    Mean of the Gaussian distribution.
sigma: float
    Standard deviation of the Gaussian distribution.
threshold: float
    Threshold value for calculating the likelihood of the excursion set.

Returns:
--------
plotly.graph_objects.Figure
    A plotly figure object.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

def plot(mu: float, sigma: float, threshold: float) -> go.Figure:
    x = np.linspace(mu - 4.5 * sigma, mu + 4.5 * sigma, 200)
    y = norm.pdf(x, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.add_shape(type='line', x0=threshold, x1=threshold, y0=0, y1=norm.pdf(threshold, mu, sigma), line=dict(color='red', width=2))
    area_x = np.linspace(np.amin(x), threshold, 200)
    area_y = norm.pdf(area_x, mu, sigma)
    fig.add_trace(go.Scatter(x=area_x, y=area_y, mode='lines', fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)'))
    fig.add_annotation(
        x=mu, y=norm.pdf(mu, mu, sigma),
        showarrow=False,
        text=f"EP: {norm.cdf(threshold, mu, sigma):.2f}",
        font=dict(
            size=14,
            color="white"
        ),
        xanchor="left",
        yanchor="top",
        xshift=10,
        yshift=20
    )
    fig.update_layout(
        title='Gaussian Distribution PDF',
        xaxis=dict(
            range=[-1.5, 1.5],
            title='x'
        ),
        yaxis=dict(
            title='Density'
        ),
        showlegend=False,
        width=700,
        height=500
    )
    return fig


if __name__ == '__main__':
    fig = plot(0, 1, .1)
    fig.show()

