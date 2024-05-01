import streamlit as st
import plotly.graph_objects as go
import numpy as np
from GRF import GRF
from makePlots import random_realization
from makePlots import histogram
from makePlots import kernel_density_estimation
from makeFormulas import makeFormulaForGRF
import time

if "truth" not in st.session_state:
    st.session_state["truth"] = None

if "x" not in st.session_state:
    st.session_state["x"] = 0.5

if "y" not in st.session_state:
    st.session_state["y"] = 0.5

if "mean" not in st.session_state:
    st.session_state["mean"] = []

if "sigma" not in st.session_state:
    st.session_state["sigma"] = 0.2

if "nugget" not in st.session_state:
    st.session_state["nugget"] = 0.01

if "threshold" not in st.session_state:
    st.session_state["threshold"] = 0.5

if "lateral_range" not in st.session_state:
    st.session_state["lateral_range"] = 0.6

if "grid_size" not in st.session_state:
    st.session_state["grid_size"] = 0.05

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

grf_instance = None
def initialize_grf():
    global grf_instance
    polygon_border = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    polygon_obstacle = np.empty((0, 2))
    grid_size = st.session_state["grid_size"]
    lateral_range = st.session_state["lateral_range"]
    sigma = st.session_state["sigma"]
    nugget = st.session_state["nugget"]
    threshold = st.session_state["threshold"]
    grf_instance = GRF(polygon_border, polygon_obstacle, grid_size, lateral_range, sigma, nugget, threshold)






""" Render the Sense page """
def renderSensePage():
    global grf_instance
    if grf_instance is None:
        initialize_grf()

    grf = grf_instance

    st.sidebar.title("Demo GRF")
    with st.sidebar.expander("What is a GRF?"):
        isRegenerateGRF = st.button("Generate 1 realization")
        isShowMoreButton = st.toggle("More", False)
        if isShowMoreButton:
            isGenerate10GRF = st.button("Generate 10 realizations")
            isGenerate100GRF = st.button("Generate 100 realizations")
            isGenerate1000GRF = st.button("Generate 1000 realizations")
        if isRegenerateGRF:
            st.session_state["truth"] = grf.get_random_realization()
        number_of_nodes = st.slider("Number of nodes for visualization", 10, 100, 50)
        x = st.slider("x", 0.1, .9, 0.5)
        y = st.slider("y", 0.1, .9, 0.5)
        isShowKDE = st.toggle("Show KDE", False)
        isShowPrior = st.toggle("Show prior", False)
        if st.session_state["x"] != x or st.session_state["y"] != y:
            st.session_state["mean"] = []
            st.session_state["x"] = x
            st.session_state["y"] = y

    with st.sidebar.expander("Why is it useful?"):
        isDataAssimilation = st.checkbox("Data assimilation", value=False)
        isShowFormula = st.toggle("Show formula", False)
        st.session_state["sigma"] = st.slider("σ", 0.01, 1.0, 0.2)
        st.session_state["nugget"] = st.slider(r"$\tau$", 0.01, 1.0, 0.01)
        st.session_state["lateral_range"] = st.slider(r"$\frac{4.5}{\phi}$", 0.1, 1.0, 0.6)



    """ Main content """
    if not isDataAssimilation: 
        st.markdown("""
            ### A Gaussian random field (GRF) is a random field that is Gaussian distributed.
                    """)
        st.latex(r"""
            \boldsymbol{\xi} = (\xi_{\boldsymbol{u}_1}, \dots, \xi_{\boldsymbol{u}_n})^T, \hspace{5mm}  \boldsymbol{\xi} \sim N(\boldsymbol{\mu, \Sigma})
            """)
        col1, col2 = st.columns(2)
        with col1:
            if isShowPrior:
                st.header("Prior mean of a GRF")
                fig = go.Figure()
                random_realization.plot(fig, grf.grid, grf.get_mu_prior(), number_of_nodes, x, y)
                st.plotly_chart(fig, use_container_width=True)
            else: 
                st.header("Realization of a GRF")
                if st.session_state["truth"] is not None:
                    if not isShowMoreButton:
                        fig = go.Figure()
                        random_realization.plot(fig, grf.grid, st.session_state["truth"], number_of_nodes, x, y)
                        st.plotly_chart(fig, use_container_width=True)
                        ind = grf.get_ind_from_location(np.array([x, y]))
                        mean = st.session_state["truth"][ind, 0][0]
                        st.session_state["mean"].append(mean)
                    else: 
                        if isRegenerateGRF: 
                            generate_animated_realizations(grf, number_of_nodes, x, y, 1)
                        if isGenerate10GRF:
                            generate_animated_realizations(grf, number_of_nodes, x, y, 10)
                        elif isGenerate100GRF:
                            generate_animated_realizations(grf, number_of_nodes, x, y, 100)
                        elif isGenerate1000GRF:
                            generate_animated_realizations(grf, number_of_nodes, x, y, 1000)
                else:
                    st.write("You need to generate the GRF first.")

        with col2:
            if isShowKDE: 
                st.header(f"KDE at ({x:.2f}, {y:.2f})")
            else: 
                st.header(f"Histogram at ({x:.2f}, {y:.2f})")
            if st.session_state["truth"] is not None:
                if len(st.session_state["mean"]) > 2:
                    if isShowKDE:
                        fig = kernel_density_estimation.plot(st.session_state["mean"])
                    else:
                        fig = histogram.plot(st.session_state["mean"])
                    st.plotly_chart(fig, use_container_width=True)
            else: 
                st.write("You need to generate the GRF first.")

    else:
        if isShowFormula:
            makeFormulaForGRF.render()
        
        
        

