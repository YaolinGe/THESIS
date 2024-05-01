import plotly.graph_objects as go
from usr_func.interpolate_2d import interpolate_2d


def plot(fig, grid, truth, number_of_nodes: int=50, x: float=0.5, y: float=0.5): 
    grid_x, grid_y, truth_interpolated = interpolate_2d(grid[:, 0], grid[:, 1], number_of_nodes, number_of_nodes, truth)
    fig.add_trace(
        go.Scatter(
            x=grid_x.flatten(),
            y=grid_y.flatten(),
            mode='markers',
            marker=dict(
                size=10,
                color=truth_interpolated.flatten(),
                colorscale='BrBG',
                showscale=True,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    thickness=20,
                )
            )
        )
    )
    fig.add_vline(x=x, line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=y, line_width=2, line_dash="dash", line_color="red")
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(color="red", symbol="circle", size=10)))
    fig.update_layout(
        # title_text="Random realization",
        width=500,
        height=700,
        autosize=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(
            title="X",
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            title="Y",
            showgrid=False,
            zeroline=False,
            showticklabels=True
        ), 
        showlegend=False
    )
    return fig