import streamlit as st
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
from matplotlib.patches import Ellipse
import numpy as np
from GRF import GRF
from pathPlanningAlgorithms.PRM import PRM
from pathPlanningAlgorithms.RRTStar import RRTStar
from pathPlanningAlgorithms.AStar import AStar
from manageStates.initializePlanState import initializePlanState
from makePlots import gaussian_distribution
from usr_func.interpolate_2d import interpolate_2d

def make_subplot(x, y, value, title, xaxis_title, yaxis_title, colorscale='BrBG', cmin=0, cmax=1, binary_colorscale=False, 
                 showXYline=False, width=350, height=400):
    fig = go.Figure()
    if st.session_state['isGridInterpolated']: 
        try: 
            if width <= 400:
                grid_x, grid_y, value = interpolate_2d(x, y, 50, 50, value)
            else:
                grid_x, grid_y, value = interpolate_2d(x, y, 100, 100, value)
            x = grid_x.flatten()
            y = grid_y.flatten()
        except:
            pass
    if binary_colorscale:
        colorbar=dict(thickness=20, tickvals=[0, 1], ticktext=['False', 'True'])
    else:
        colorbar=dict(thickness=20)
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10, color=value.flatten(), colorscale=colorscale, showscale=True, cmin=cmin, cmax=cmax, colorbar=colorbar))
    )
    if showXYline: 
        fig.add_vline(x=st.session_state['x'], line_width=2, line_dash="dash", line_color="red")
        fig.add_hline(y=st.session_state['y'], line_width=2, line_dash="dash", line_color="red")
        fig.add_trace(go.Scatter(x=[st.session_state['x']], y=[st.session_state['y']], mode="markers", marker=dict(color="red", symbol="circle", size=10)))
    
    fig.update_layout(width=width, height=height,
                    xaxis_title=xaxis_title,
                    yaxis_title=yaxis_title, 
                    title=title, 
                    title_x=0.35,
                    showlegend=False
                    )
    return fig

grf_instance = None
prm_instance = None
rrtstar_instance = None
astar_instance = None
def initialize_global_instances():
    global grf_instance
    global prm_instance
    global rrtstar_instance
    global astar_instance
    grf_instance = GRF(polygon_border=st.session_state["polygon_border"],
                    polygon_obstacle=st.session_state["polygon_obstacle"],
                    grid_size=st.session_state["grid_size"],
                    lateral_range=st.session_state["lateral_range"],
                    sigma=st.session_state["sigma"],
                    nugget=st.session_state["nugget"],
                    threshold=st.session_state["threshold"])
    prm_instance = PRM(st.session_state['pathplanning_border'], 
                    st.session_state['pathplanning_obstacles'])
    rrtstar_instance = RRTStar(st.session_state['pathplanning_border'], 
                    st.session_state['pathplanning_obstacles'])
    astar_instance = AStar(st.session_state['pathplanning_border'],
                    st.session_state['pathplanning_obstacles'])






""" Render the Plan page """
def renderPlanPage():
    
    initializePlanState()

    global grf_instance, prm_instance, rrtstar_instance, astar_instance
    if grf_instance is None:
        initialize_global_instances()

    grf = grf_instance
    prm = prm_instance
    rrtstar = rrtstar_instance
    astar = astar_instance


    st.sidebar.title("Demo Plan")

    st.session_state['isGridInterpolated'] = st.sidebar.toggle("Interpolate", value=False)
    st.session_state['grid_size'] = st.sidebar.slider("Grid size", 0.03, 0.5, 0.15)
    st.session_state['threshold'] = st.sidebar.slider("Threshold", 0.0, 1.0, .5)
    with st.sidebar.expander("How to define a boundary?"):
        isShowBoundary = st.checkbox("Show boundary", value=False)
        isShowEPDemo = st.toggle("Show EP demo", value=False)
        if isShowEPDemo:
            st.session_state['x'] = st.slider("x", .01, .99, 0.5)
            st.session_state['y'] = st.slider("y", .01, .99, 0.5)

        grf.update_kernel(polygon_border=st.session_state["polygon_border"],
                        polygon_obstacle=st.session_state["polygon_obstacle"],
                        grid_size=st.session_state["grid_size"],
                        lateral_range=st.session_state["lateral_range"],
                        sigma=st.session_state["sigma"],
                        nugget=st.session_state["nugget"],
                        threshold=st.session_state["threshold"])

    with st.sidebar.expander("What is a good planning metric?"):
        isShowSamplingStrategy = st.checkbox("Show planning metrics")
        isHideFormulas = st.toggle("Hide formulas", False)
        isShowOperationalConstraints = st.toggle("Show operational constraints", value=False)
        if isShowOperationalConstraints:
            isObstacleExist = st.toggle("Obstacle", value=False)
            isBudget = st.toggle("Budget", value=False)
            if isBudget: 
                st.session_state['budget'] = st.slider("Remaining", 1.25, 2.5, 2.5)
                st.session_state['current_x'] = st.slider("Current x", 0.0, 1.0, .1)
                st.session_state['current_y'] = st.slider("Current y", 0.0, 1.0, .1)
                st.session_state['destination_x'] = st.slider("Destination x", 0.0, 1.0, 0.9)
                st.session_state['destination_y'] = st.slider("Destination y", 0.0, 1.0, 0.9)
        
    with st.sidebar.expander("How do we find a path then? "):
        isPathPlanning = st.checkbox("Path planning", value=False)
        if isPathPlanning:
            isNonMyopicPathPlanning = st.toggle("Non-myopic", value=False)
        isSimulation = st.checkbox("Simulation", value=False)
        if isSimulation:
            path_planning_algorithm = st.selectbox("Path Planning Algorithm", ["A*", "PRM", "RRT*", ])
            isAnimated = st.checkbox("Animated", value=False)
            isRunPathPlanningSimulation = st.button("Run")

            st.sidebar.markdown("### Location start")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                x = st.slider("x", 0.01, .99, 0.01)
            with col2: 
                y = st.slider("y", 0.01, .99, 0.01)
            st.session_state['loc_start'] = np.array([x, y])

            st.sidebar.markdown("### Location end")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                x = st.slider("x", 0.01, .99, 0.99)
            with col2:
                y = st.slider("y", 0.01, .99, 0.99)
            st.session_state['loc_end'] = np.array([x, y])


            



    """ Main content """
    if isShowBoundary:
        st.markdown("### Excursion set")
        st.latex(r"""
                \begin{equation}
                    \text{{ES}} = \{\boldsymbol{u} \in \mathcal{M} \quad : \quad \xi_{\boldsymbol{u}} \leq t\}.
                \end{equation}
                """)

        st.markdown("### Excursion probability")
        st.latex(r"""
                \begin{equation}
                    \text{{EP}} = \mathbb{P}(\xi(\boldsymbol{u}) \leq t).
                \end{equation}
                    """)
        
        if isShowEPDemo:
            loc = np.array([st.session_state['x'], st.session_state['y']])
            ind = grf.get_ind_from_location(loc)
            mean = grf.get_mu()[ind, 0][0]
            std = np.sqrt(grf.get_covariance_matrix()[ind, ind])[0]
            fig = gaussian_distribution.plot(mean, std, st.session_state['threshold'])
            st.plotly_chart(fig, use_container_width=True)

        
        col1, col2, col3 = st.columns(3)
        with col1: 
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_mu_prior().flatten(), 'Prior mean', 'x', 'y', showXYline=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_excursion_set().flatten(), 'Excursion set', 'x', 'y', colorscale='YlOrRd', cmin=0, cmax=1, binary_colorscale=True, showXYline=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], grf.get_excursion_probability().flatten(), 'Excursion probability', 'x', 'y', colorscale='YlGnBu', cmin=0, cmax=1, showXYline=True)
            st.plotly_chart(fig, use_container_width=True)
    
    elif isShowSamplingStrategy: 
        if not isHideFormulas:
            st.markdown("### Bernoulli variance")
            st.latex(r"""
                    \begin{equation}
                        \text{BV} = \text{EP}_{\boldsymbol{u}}(1 - \text{EP}_{\boldsymbol{u}})
                    \end{equation}
                    """)

            st.markdown("### Intergrated Bernoulli variance")
            st.latex(r"""
                    \begin{equation}
                        \text{IBV} = \int \text{{EP}}_{\boldsymbol{u}}(1 - \text{{EP}}_{\boldsymbol{u}})d\boldsymbol{u}
                    \end{equation}
                    """)
            # st.markdown("### Expected integrated Bernoulli variance]")
            st.markdown("<h3 style='color: yellow;'>Expected integrated Bernoulli variance</h3>", unsafe_allow_html=True)
            st.latex(r"""
                    \begin{equation}
                        \text{EIBV}(\boldsymbol{D}_j) = \int E_{\boldsymbol{y}_j|\mathcal{Y}_{j-1};\boldsymbol{D}_j} \left[B_{\boldsymbol{u}}(\boldsymbol{y}_j) \right] d\boldsymbol{u}
                    \end{equation}
                    """)
            st.latex(r"""
                    \begin{equation}   
                    B_{\boldsymbol{u}}(\boldsymbol{y}_j) = \text{EP}_{\boldsymbol{u}}(\boldsymbol{y}_j,\boldsymbol{D}_j,\mathcal{Y}_{j-1})(1 - \text{EP}_{\boldsymbol{u}}(\boldsymbol{y}_j,\boldsymbol{D}_j,\mathcal{Y}_{j-1})
                    \end{equation}
                    """)
            # st.markdown("### Integrated variance reduction")
            st.markdown("<h3 style='color: yellow;'>Integrated variance reduction</h3>", unsafe_allow_html=True)
            st.latex(r"""
                    \begin{equation}
                        \text{IVR}(\boldsymbol{D}_j) = \text{trace}(\boldsymbol{S}_{j-1} \boldsymbol{F}_{\boldsymbol{D}_j}^T (\boldsymbol{F}_{\boldsymbol{D}_j} \boldsymbol{S}_{j-1} \boldsymbol{F}_{\boldsymbol{D}_j}^T + \boldsymbol{R}_j)^{-1} \boldsymbol{F}_{\boldsymbol{D}_j} \boldsymbol{S}_{j-1})
                    \end{equation}
                    """)
            # st.markdown("### Operational constraints")
            st.markdown("<h3 style='color: yellow;'>Operational constraints</h3>", unsafe_allow_html=True)
            st.markdown("(time left, distance traveled, collision risk etc.)")
            st.latex(r"""
                    \begin{equation}
                        \text{OC} = \sum \text{Constraint}(\boldsymbol{D}_j)
                    \end{equation}
                    """)
            
        if isShowOperationalConstraints:
            oc = np.zeros(grf.grid.shape[0])
            if isObstacleExist: 
                polygon_obstacle = np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]])
                polygon_obstacle_shapely = Polygon(polygon_obstacle)
                for i in range(grf.grid.shape[0]):
                    point = Point(grf.grid[i, :])
                    if polygon_obstacle_shapely.contains(point):
                        oc[i] = 100
            if isBudget:
                middle_x = (st.session_state['current_x'] + st.session_state['destination_x']) / 2
                middle_y = (st.session_state['current_y'] + st.session_state['destination_y']) / 2
                dx = st.session_state['destination_x'] - st.session_state['current_x']
                dy = st.session_state['destination_y'] - st.session_state['current_y']
                angle = np.arctan2(dy, dx)
                ellipse_a = st.session_state['budget'] / 2
                ellipse_c = np.sqrt(dx**2 + dy**2) / 2
                ellipse_b = np.sqrt(ellipse_a**2 - ellipse_c**2)
                ellipse = Ellipse(xy=(middle_x, middle_y), width=2 * ellipse_a, height=2 * ellipse_b, angle=np.degrees(angle))
                vertices = ellipse.get_verts()
                polygon_ellipse = np.array(vertices)
                polygon_ellipse_shapely = Polygon(polygon_ellipse)
                xg = grf.grid[:, 0] - middle_x
                yg = grf.grid[:, 1] - middle_y
                xr = xg * np.cos(angle) + yg * np.sin(angle)
                yr = -xg * np.sin(angle) + yg * np.cos(angle)
                u = (xr / ellipse_a)**2 + (yr / ellipse_b)**2
                for i in range(grf.grid.shape[0]):
                    point = Point(grf.grid[i, :])
                    if polygon_ellipse_shapely.contains(point):
                        oc[i] += 0
                    else:
                        oc[i] += u[i]

            fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], oc, 'Operational constraint', 'x', 'y', colorscale='YlGnBu', cmin=0, cmax=5, width=650, height=800)
            if isObstacleExist: 
                fig.add_trace(go.Scatter(x=polygon_obstacle[:, 0], y=polygon_obstacle[:, 1], mode='lines', line=dict(color='red')))
            if isBudget:
                fig.add_trace(go.Scatter(x=[st.session_state['current_x']], y=[st.session_state['current_y']], mode='markers', marker=dict(color='red', size=20)))
                fig.add_trace(go.Scatter(x=[st.session_state['destination_x']], y=[st.session_state['destination_y']], mode='markers', marker=dict(color='green', size=20)))
                fig.add_trace(go.Scatter(x=polygon_ellipse[:, 0], y=polygon_ellipse[:, 1], mode='lines', line=dict(color='yellow')))
                fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            eibv, ivr = grf.get_ei_field_at_locations(grf.grid)
            with col1: 
                fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], eibv, 'EIBV heuristic', 'x', 'y', colorscale='RdYlBu', cmin=np.amin(eibv), cmax=np.amax(eibv), width=500, height=550)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = make_subplot(grf.grid[:, 0], grf.grid[:, 1], ivr, 'IVR heuristic', 'x', 'y', colorscale='YlOrBr', cmin=np.amin(ivr), cmax=np.amax(ivr), width=500, height=550)
                st.plotly_chart(fig, use_container_width=True)

    elif isPathPlanning:
        if not isSimulation:
            if not isNonMyopicPathPlanning:
                st.markdown("### Myopic Path Planning")
                st.markdown("This is an illustration of myopic path planning.")
                st.image("figs/myopic.png")
            else:
                st.markdown("### Non-Myopic Path Planning")
                st.markdown("This is an illustration of non-myopic path planning.")
                st.image("figs/nonmyopic.png")
        else:
            st.title(f"{path_planning_algorithm} Simulation")

            if path_planning_algorithm == "A*":
                st.session_state['astar_max_iter'] = st.sidebar.slider("Max iteration", 200, 2000, 1000)
                st.session_state['astar_stepsize'] = st.sidebar.slider("Stepsize", 0.05, 0.5, 0.1)
                st.session_state['astar_distance_tolerance_target'] = st.sidebar.slider("Distance tolerance target", 0.05, 0.2, 0.075)
                st.session_state['astar_distance_tolerance'] = st.sidebar.slider("Distance tolerance", 0.05, 0.2, 0.075)
                path = astar.search_path(st.session_state['loc_start'], st.session_state['loc_end'], 
                                         max_iter=st.session_state['astar_max_iter'], stepsize=st.session_state['astar_stepsize'],
                                         distance_tolerance_target=st.session_state['astar_distance_tolerance_target'],
                                         distance_tolerance=st.session_state['astar_distance_tolerance'], animated=isAnimated)
                if not isAnimated:
                    if path is None:
                        st.error("No path found.")
                    else:
                        fig = go.Figure()
                        fig.update_layout(
                            width=500, 
                            height=700,
                            showlegend=False,
                            )
                        fig.add_trace(go.Scatter(x=st.session_state['pathplanning_border'][:, 0], y=st.session_state['pathplanning_border'][:, 1], mode='lines', line=dict(color='red', dash='dash')))
                        for obs in st.session_state['pathplanning_obstacles']:
                            fig.add_trace(go.Scatter(x=obs[:, 0], y=obs[:, 1], mode='lines', line=dict(color='red')))
                        path = np.array(path)
                        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(color='yellow', width=5)))
                        fig.add_trace(go.Scatter(x=[st.session_state['loc_start'][0]], y=[st.session_state['loc_start'][1]], mode='markers', marker=dict(size=20, color='green')))
                        fig.add_trace(go.Scatter(x=[st.session_state['loc_end'][0]], y=[st.session_state['loc_end'][1]], mode='markers', marker=dict(size=20, color='blue')))
                        st.plotly_chart(fig, use_container_width=True)
                

            elif path_planning_algorithm == "RRT*":
                st.session_state['rrtstar_max_expansion_iteration'] = st.sidebar.slider("Max expansion iteration", 100, 2000, 1500)
                st.session_state['rrtstar_stepsize'] = st.sidebar.slider("Stepsize", 0.01, 0.5, 0.1)
                st.session_state['rrtstar_goal_sampling_rate'] = st.sidebar.slider("Goal sampling rate", 0.01, 0.5, 0.01)
                st.session_state['rrtstar_home_radius'] = st.sidebar.slider("Home radius", 0.01, 0.5, 0.09)
                st.session_state['rrtstar_neighbour_radius'] = st.sidebar.slider("Neighbour radius", 0.01, 0.5, 0.12)
                # if isRunPathPlanningSimulation:
                path = rrtstar.get_path(loc_start=st.session_state['loc_start'], loc_target=st.session_state['loc_end'], 
                                        max_expansion_iteration=st.session_state['rrtstar_max_expansion_iteration'],
                                        stepsize=st.session_state['rrtstar_stepsize'], goal_sampling_rate=st.session_state['rrtstar_goal_sampling_rate'], 
                                        home_radius=st.session_state['rrtstar_home_radius'], rrtstar_neighbour_radius=st.session_state['rrtstar_neighbour_radius'],
                                        animated=isAnimated)
                if not isAnimated:
                    fig = go.Figure()
                    fig.update_layout(
                        width=500, 
                        height=700,
                        showlegend=False,
                        )
                    fig.add_trace(go.Scatter(x=st.session_state['pathplanning_border'][:, 0], y=st.session_state['pathplanning_border'][:, 1], mode='lines', line=dict(color='red', dash='dash')))
                    for obs in st.session_state['pathplanning_obstacles']:
                        fig.add_trace(go.Scatter(x=obs[:, 0], y=obs[:, 1], mode='lines', line=dict(color='red')))
                    for node in rrtstar.nodes:
                        if node.get_parent() is not None:
                            loc = node.get_location()
                            loc_p = node.get_parent().get_location()
                            fig.add_trace(go.Scatter(x=[loc[0], loc_p[0]], y=[loc[1], loc_p[1]], mode='lines', line=dict(color='white', width=0.5)))
                    path = np.array(path)
                    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(color='yellow', width=5)))
                    fig.add_trace(go.Scatter(x=[st.session_state['loc_start'][0]], y=[st.session_state['loc_start'][1]], mode='markers', marker=dict(size=20, color='green')))
                    fig.add_trace(go.Scatter(x=[st.session_state['loc_end'][0]], y=[st.session_state['loc_end'][1]], mode='markers', marker=dict(size=20, color='blue')))
                    st.plotly_chart(fig, use_container_width=True)


            elif path_planning_algorithm == "PRM":
                st.session_state['prm_num_nodes'] = st.sidebar.slider("Number of nodes", 10, 1000, 100)
                st.session_state['prm_num_neighbours'] = st.sidebar.slider("Number of neighbours", 2, 20, 10)

                # if isRunPathPlanningSimulation:
                path, fig = prm.get_path(st.session_state['loc_start'], st.session_state['loc_end'], st.session_state['prm_num_nodes'], st.session_state['num_neighbours'], animated=isAnimated)
                if not isAnimated:
                    fig = go.Figure()
                    fig.update_layout(
                        width=500, 
                        height=700,
                        showlegend=False,
                        )
                    fig.add_trace(go.Scatter(x=st.session_state['pathplanning_border'][:, 0], y=st.session_state['pathplanning_border'][:, 1], mode='lines', line=dict(color='red', dash='dash')))
                    for obs in st.session_state['pathplanning_obstacles']:
                        fig.add_trace(go.Scatter(x=obs[:, 0], y=obs[:, 1], mode='lines', line=dict(color='red')))
                    loc_nodes = np.array([[node.x, node.y] for node in prm.nodes]).reshape(-1, 2)
                    fig.add_trace(go.Scatter(x=loc_nodes[:, 0], y=loc_nodes[:, 1], mode='markers', marker=dict(size=7, color='white')))
                    fig.add_trace(go.Scatter(x=[st.session_state['loc_start'][0]], y=[st.session_state['loc_start'][1]], mode='markers', marker=dict(size=20, color='green')))
                    fig.add_trace(go.Scatter(x=[st.session_state['loc_end'][0]], y=[st.session_state['loc_end'][1]], mode='markers', marker=dict(size=20, color='blue')))

                    for node in prm.nodes:
                        if node.neighbours is not None:
                            for i in range(len(node.neighbours)):
                                fig.add_trace(go.Scatter(x=[node.x, node.neighbours[i].x],
                                                        y=[node.y, node.neighbours[i].y],
                                                        mode='lines',
                                                        line=dict(color='white', width=0.5), opacity=.5))
                    path = np.array(path)
                    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(color='yellow', width=5)))
                    st.plotly_chart(fig, use_container_width=True)
            

