from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import redis
from GRF import GRF

redis_host = 'localhost'
redis_port = 6379
redis_db = 0
redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
polygon_obstacle = np.empty((0, 2))

grid_size = float(redis_client.get('grid_size'))
lateral_range = float(redis_client.get('lateral_range'))
sigma = float(redis_client.get('sigma'))
nugget = float(redis_client.get('nugget'))
threshold = float(redis_client.get('threshold'))


app = Dash(__name__)

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='GRF'), 
], 
style={'width': '100%', 'height': '100%',}
)

@app.callback(
    Output('GRF', 'children'),
    
)


if __name__ == '__main__':
    app.run_server(debug=True)
