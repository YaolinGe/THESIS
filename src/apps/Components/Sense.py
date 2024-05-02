import streamlit as st
import plotly.graph_objects as go
import numpy as np
from GRF import GRF
from makePlots import random_realization
from makePlots import histogram
from makePlots import kernel_density_estimation
from makePlots.generate_animated_realizations import generate_animated_realizations
from manageStates.initializeSenseState import initializeSenseState
from makeFormulas import makeFormulaForGRF
from usr_func.interpolate_2d import interpolate_2d
from usr_func.generate_a_path import generate_a_path


grf_instance = None
def initialize_grf():
    global grf_instance
    grf_instance = GRF(polygon_border=st.session_state["polygon_border"],
                    polygon_obstacle=st.session_state["polygon_obstacle"],
                    grid_size=st.session_state["grid_size"],
                    lateral_range=st.session_state["lateral_range"],
                    sigma=st.session_state["sigma"],
                    nugget=st.session_state["nugget"],
                    threshold=st.session_state["threshold"])






""" Render the Sense page """
def renderSensePage():
    
    initializeSenseState()

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
        isHideFormula = st.toggle("Hide formula", False)
        st.session_state['isGridInterpolated'] = st.toggle("Interpolate for visualization", False)
        isShowPath = st.toggle("Show path", False)
        isRandomPath = st.button("Generate random path")
        path = np.array([[0.5, 0.01], [0.5, .99]])
        middle_points = np.linspace(path[0], path[1], num=10, endpoint=False)[1:-1]
        path = np.concatenate((path, middle_points), axis=0)
        st.session_state["path"] = path
        if isRandomPath:
            path = generate_a_path((np.amin(grf.grid[:, 0]), np.amax(grf.grid[:, 0])), 
                                   (np.amin(grf.grid[:, 1]), np.amax(grf.grid[:, 1])), 0.1, np.random.randint(5, 16))
            st.session_state["path"] = path
        st.session_state["grid_size"] = st.slider("Grid size", 0.02, 0.5, 0.1)
        st.session_state["sigma"] = st.slider("σ", 0.01, 1.0, 0.2)
        st.session_state["nugget"] = st.slider(r"$\tau$", 0.01, 1.0, 0.01)
        st.session_state["lateral_range"] = st.slider(r"$\frac{4.5}{\phi}$", 0.1, 1.0, 0.6)
        grf.update_kernel(polygon_border=st.session_state["polygon_border"],
                          polygon_obstacle=st.session_state["polygon_obstacle"],
                          grid_size=st.session_state["grid_size"],
                          lateral_range=st.session_state["lateral_range"],
                          sigma=st.session_state["sigma"],
                          nugget=st.session_state["nugget"],
                          threshold=st.session_state["threshold"])





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
        if not isHideFormula:
            makeFormulaForGRF.render()
        
        def make_subplot(x, y, value, title, xaxis_title, yaxis_title, colorscale='BrBG', cmin=0, cmax=1, path=None, title_x=0.35):
            fig = go.Figure()
            if st.session_state['isGridInterpolated']: 
                try: 
                    grid_x, grid_y, value = interpolate_2d(x, y, 50, 50, value)
                    x = grid_x.flatten()
                    y = grid_y.flatten()
                except:
                    pass
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10, color=value.flatten(), colorscale=colorscale, showscale=True, cmin=cmin, cmax=cmax, colorbar=dict(thickness=20)))
            )
            if path is not None:
                if isShowPath: 
                    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(color='red', width=4)))
            fig.update_layout(width=350, height=400,
                            xaxis_title=xaxis_title,
                            yaxis_title=yaxis_title, 
                            title=title, 
                            title_x=title_x,
                            showlegend=False
                            )
            return fig
        
        col1, col2, col3 = st.columns(3)
        with col1: 
            fig = go.Figure(data=go.Heatmap(z=grf.get_covariance_matrix(), 
                                    colorscale='RdBu',
                                    x0=0,
                                    y0=1,
                                    dx=1,
                                    dy=-1))
            fig.update_layout(width=350, height=400,
                            xaxis_title="X",
                            yaxis_title="Y", 
                            title="Covariance matrix", 
                            title_x=0.35)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2: 
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_mu().flatten(), "Prior mean", "X", "Y")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3: 
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], np.sqrt(np.diag(grf.get_covariance_matrix())), "Prior uncertainty field", "X", "Y", colorscale='GnBu', cmin=0, cmax=1, title_x=.25)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        truth1 = grf.get_random_realization()
        truth2 = grf.get_random_realization()
        truth3 = grf.get_random_realization()
        with col1: 
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], truth1.flatten(), "Ground truth I", "X", "Y", path=st.session_state["path"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], truth2.flatten(), "Ground truth II", "X", "Y", path=st.session_state["path"])
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], truth3.flatten(), "Ground truth III", "X", "Y", path=st.session_state["path"])
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        ind_sampled = grf.get_ind_from_location(st.session_state["path"])
        x_path = st.session_state["path"][:, 0]
        y_path = st.session_state["path"][:, 1]
        with col1: 
            data = np.column_stack((x_path, y_path, truth1[ind_sampled, 0]))
            grf.assimilate_data(data)
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_mu().flatten(), "Posterior mean I", "X", "Y", path=st.session_state["path"])
            st.plotly_chart(fig, use_container_width=True)
    
        with col2:
            data = np.column_stack((x_path, y_path, truth2[ind_sampled, 0]))
            grf.assimilate_data(data)
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_mu().flatten(), "Posterior mean II", "X", "Y", path=st.session_state["path"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            data = np.column_stack((x_path, y_path, truth3[ind_sampled, 0]))
            grf.assimilate_data(data)
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_mu().flatten(), "Posterior mean III", "X", "Y", path=st.session_state["path"])
            st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1: 
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], np.sqrt(np.diag(grf.get_covariance_matrix())), "Posterior uncertainty field I", "X", "Y", colorscale='GnBu', cmin=0, cmax=st.session_state["sigma"], path=st.session_state["path"], title_x=.2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], np.sqrt(np.diag(grf.get_covariance_matrix())), "Posterior uncertainty field II", "X", "Y", colorscale='GnBu', cmin=0, cmax=st.session_state["sigma"], path=st.session_state["path"], title_x=.2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], np.sqrt(np.diag(grf.get_covariance_matrix())), "Posterior uncertainty field III", "X", "Y", colorscale='GnBu', cmin=0, cmax=st.session_state["sigma"], path=st.session_state["path"], title_x=.2)
            st.plotly_chart(fig, use_container_width=True)
