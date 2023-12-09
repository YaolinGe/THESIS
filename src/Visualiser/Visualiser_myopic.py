"""
Visualiser object handles the planning visualisation part.
"""

# from Agent import Agent
import os
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from usr_func.interpolate_3d import interpolate_3d
from usr_func.checkfolder import checkfolder
from usr_func.vectorize import vectorize


class Visualiser:

    agent = None

    def __init__(self, agent: 'Agent', figpath: 'str') -> None:
        self.agent = agent
        checkfolder(figpath + "mu/")
        checkfolder(figpath + "mvar/")
        self.figpath = figpath
        self.myopic = self.agent.myopic
        self.kernel = self.myopic.kernel
        self.grid = self.myopic.kernel.get_grid()

        self.ind_remove_top_layer = np.where(self.grid[:, 2] > 0)[0]
        self.xgrid = self.grid[self.ind_remove_top_layer, 0]
        self.ygrid = self.grid[self.ind_remove_top_layer, 1]
        self.rotated_angle = self.kernel.get_rotated_angle()
        self.xrotated = self.xgrid * np.cos(self.rotated_angle) - self.ygrid * np.sin(self.rotated_angle)
        self.yrotated = self.xgrid * np.sin(self.rotated_angle) + self.ygrid * np.cos(self.rotated_angle)
        self.xplot = self.yrotated
        self.yplot = self.xrotated
        self.RR = np.array([[np.cos(self.rotated_angle), -np.sin(self.rotated_angle), 0],
                            [np.sin(self.rotated_angle), np.cos(self.rotated_angle), 0],
                            [0, 0, 1]])

    def plot_agent(self):
        mu = self.kernel.get_mu()
        mvar = self.kernel.get_mvar()
        self.cnt = self.agent.get_counter()
        mu[mu < 0] = 0
        ind_selected_to_plot = np.where(mu[self.ind_remove_top_layer] >= 0)[0]
        self.xplot = self.xplot[ind_selected_to_plot]
        self.yplot = self.yplot[ind_selected_to_plot]
        self.zplot = -self.grid[self.ind_remove_top_layer, 2][ind_selected_to_plot]

        """ plot mean """
        value = mu[self.ind_remove_top_layer][ind_selected_to_plot]
        vmin = 0
        vmax = 28
        filename = self.figpath + "mu/P_{:03d}.html".format(self.cnt)
        self.plot_figure(value, vmin=vmin, vmax=vmax, filename=filename, title="mean", cmap="YlGnBu")

        """ plot mvar """
        filename = self.figpath + "mvar/P_{:03d}.html".format(self.cnt)
        value = mvar[self.ind_remove_top_layer][ind_selected_to_plot]
        vmin = np.amin(value)
        vmax = np.amax(value)
        self.plot_figure(value, vmin=vmin, vmax=vmax, filename=filename, title="marginal variance", cmap="RdBu")

    def plot_figure(self, value, vmin=0, vmax=30, filename=None, title=None, cmap=None):
        points_grid, values_grid = interpolate_3d(self.xplot, self.yplot, self.zplot, value)
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(go.Volume(
            x=points_grid[:, 0],
            y=points_grid[:, 1],
            z=points_grid[:, 2],
            value=values_grid,
            isomin=vmin,
            isomax=vmax,
            opacity=.3,
            surface_count=10,
            colorscale=cmap,
            # coloraxis="coloraxis",
            colorbar=dict(x=0.75, y=0.5, len=.5),
            # reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        id = self.myopic.get_current_index()
        wp = self.myopic.waypoint_graph.get_waypoint_from_ind(id)
        wp = np.dot(self.RR, wp)
        fig.add_trace(go.Scatter3d(
            name="Current waypoint",
            x=[wp[1]],
            y=[wp[0]],
            z=[-wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="red",
                showscale=False,
            ),
            showlegend=True,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_next_index()
        wp = self.myopic.waypoint_graph.get_waypoint_from_ind(id)
        wp = np.dot(self.RR, wp)
        fig.add_trace(go.Scatter3d(
            name="Next waypoint",
            x=[wp[1]],
            y=[wp[0]],
            z=[-wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="blue",
                showscale=False,
            ),
            showlegend=True,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_pioneer_index()
        wp = self.myopic.waypoint_graph.get_waypoint_from_ind(id)
        wp = np.dot(self.RR, wp)
        fig.add_trace(go.Scatter3d(
            name="Pioneer waypoint",
            x=[wp[1]],
            y=[wp[0]],
            z=[-wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="green",
                showscale=False,
            ),
            showlegend=True,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_trajectory_indices()
        if len(id) > 0:
            wp = self.myopic.waypoint_graph.get_waypoint_from_ind(id)
            wp = (self.RR @ wp.T).T
            fig.add_trace(go.Scatter3d(
                name="Trajectory",
                x=wp[:, 1],
                y=wp[:, 0],
                z=-wp[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color="black",
                    showscale=False,
                ),
                line=dict(
                    color="yellow",
                    width=3,
                    showscale=False,
                ),
                showlegend=True,
            ),
                row='all', col='all'
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Conditional " + title + " field",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-5.5, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )
        plotly.offline.plot(fig, filename=filename, auto_open=False)
