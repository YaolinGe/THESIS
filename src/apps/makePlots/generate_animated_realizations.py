import streamlit as st
import numpy as np
import plotly.graph_objects as go
from makePlots import random_realization

def generate_animated_realizations(grf, number_of_nodes, x, y, number_of_realizations: int=10):
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()

    for i in range(number_of_realizations):
        st.session_state["truth"] = grf.get_random_realization()
        ind = grf.get_ind_from_location(np.array([x, y]))
        mean = st.session_state["truth"][ind, 0][0]
        st.session_state["mean"].append(mean)
        progress_bar.progress((i + 1) / number_of_realizations)
        fig = go.Figure()
        random_realization.plot(fig, grf.grid, st.session_state["truth"], number_of_nodes, x, y)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    progress_bar.empty()
