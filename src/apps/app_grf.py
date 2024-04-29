"""
Gaussian Random Field module handles the data assimilation and EIBV calculation associated with locations.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

Objectives:
    1. Construct the Gaussian Random Field (GRF) kernel.
    2. Update the prior mean and covariance matrix.
    3. Assimilate in-situ data.

Methodology:
    1. Construct the GRF kernel.
        1.1. Construct the distance matrix using
            .. math::
                d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + \ksi^2 (z_i - z_j)^2}
        1.2. Construct the covariance matrix.
            .. math::
                \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})
    2. Update the prior mean and covariance matrix.
        2.1. Update the prior mean.
        2.2. Update the prior covariance matrix.
    3. Calculate the EIBV for given locations.
        3.1. Compute the EIBV for given locations.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from GRF import GRF


# polygon_border_wgs = pd.read_csv("polygon_border.csv").to_numpy()
# polygon_obstacle_wgs = pd.read_csv("polygon_obstacle.csv").to_numpy()
polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
polygon_obstacle = np.empty((0, 2))
grid_size = .05
lateral_range = .6
sigma = .2
nugget = .01
threshold = .5

st.title('Demo-Gaussian Random Field (GRF)')
with st.sidebar.expander("Parameters"):
    grid_size = st.slider('Grid size', 0.015, .5, grid_size)
    lateral_range = st.slider('Lateral Range', 0.01, 1.0, lateral_range)
    sigma = st.slider('$\sigma$', 0.01, 1.0, sigma)
    nugget = st.slider('$\epsilon$', 0.01, 1.0, nugget)
    threshold = st.slider('$\zeta$', 0.01, 1.0, threshold)

with st.sidebar.expander("Control"):
    isShowGrid = st.toggle('Show Grid')
    isShowCovariance = st.toggle('Show Covariance')
    isShowPriorMean = st.toggle('Show Prior Mean', True)
    isShowGroundTruth = st.toggle('Show Ground Truth')
    isShowExcursionSet = st.toggle('Show Excursion Set')
    isShowExcursionProbability = st.toggle('Show Excursion Probability')
    isShowEIBV = st.toggle('Show EIBV')
    


grf = GRF(polygon_border, polygon_obstacle, grid_size, lateral_range, sigma, nugget, threshold)

col1, col2 = st.columns(2, gap='small')
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color='white')))
    fig.update_layout(width=300, height=300, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color=grf.get_mu().flatten(), colorscale='BrBG', cmin=0, cmax=1)))
    fig.update_layout(width=300, height=300, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

col1, col2 = st.columns(2, gap='small')
with col1:
    pass
    # fig = go.Figure()
    # # fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color=np.diag(grf.get_covariance_matrix()), colorscale='RdBu', cmin=0, cmax=.01)))
    # fig.add_trace(go.Heatmap(z=grf.get_covariance_matrix(), colorscale='RdBu', zmin=0, zmax=0.01, showscale=False, x0=0, y0=0, dx=1, dy=1))
    # fig.update_layout(width=300, height=300, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=0, b=0))
    # st.plotly_chart(fig)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color=grf.get_random_realization().flatten(), colorscale='BrBG', cmin=0, cmax=1)))
    fig.update_layout(width=300, height=300, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

col1, col2 = st.columns(2, gap='small')
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color=grf.get_excursion_set().flatten(), colorscale='RdBu', cmin=-.01, cmax=1.1)))
    fig.update_layout(width=300, height=300, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(coloraxis_colorbar=dict(
        title='Value',
        len=0.5,
        thickness=10,
        xanchor='left',
        yanchor='middle',
        y=0.5,
        ticks='outside',
        ticklen=5,
        tickcolor='white'
    ))
    st.plotly_chart(fig)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color=grf.get_excursion_probability().flatten(), colorscale='GnBu', cmin=0, cmax=1)))
    fig.update_layout(width=300, height=300, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

# from matplotlib.cm import get_cmap
# fig = plt.figure()
# # plt.scatter(grf.grid[:, 0], grf.grid[:, 1], c=grf.get_covariance_matrix(), cmap=get_cmap("RdBu", 10), )
# plt.imshow(grf.get_covariance_matrix(), cmap='RdBu', extent=[0, 1, 0, 1]  )
# plt.colorbar()
# st.pyplot(fig)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=grf.grid[:, 0], y=grf.grid[:, 1], mode='markers', marker=dict(size=5, color='white')))
# fig.update_layout(width=400, height=500, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
# st.plotly_chart(fig)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=grf.polygon_border[:, 0], y=grf.polygon_border[:, 1], mode='lines', line=dict(color='red')))
# fig.update_layout(width=400, height=500, xaxis_title='X', yaxis_title='Y', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
# st.plotly_chart(fig)


# fig = plt.figure(figsize=(3.5, 3.5))
# plt.scatter(grf.grid[:, 0], grf.grid[:, 1], s=5)
# plt.plot(grf.polygon_border[:, 0], grf.polygon_border[:, 1])
# plt.xlabel('X')
# plt.ylabel('Y')
# st.pyplot(fig, clear_figure=True)




