import streamlit as st
import numpy as np


def initializePlanState(): 
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
    if "polygon_border" not in st.session_state:
        st.session_state["polygon_border"] = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    if "polygon_obstacle" not in st.session_state:
        st.session_state["polygon_obstacle"] = np.empty((0, 2))
    
    if "loc_start" not in st.session_state:
        st.session_state["loc_start"] = np.array([.01, .01])
    if "loc_end" not in st.session_state:
        st.session_state["loc_end"] = np.array([0.9, 0.9])
    if "pathplanning_border" not in st.session_state:
        st.session_state["pathplanning_border"] = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    if "pathplanning_obstacles" not in st.session_state:
        st.session_state["pathplanning_obstacles"] = [np.array([[.1, .1], [.2, .1], [.2, .2], [.1, .2], [.1, .1]]),
                                                      np.array([[.7, .7], [.8, .7], [.8, .8], [.7, .8], [.7, .7]]),
                                                      np.array([[.3, .3], [.5, .3], [.5, .5], [.3, .5], [.3, .3]]),
                                                      np.array([[.6, .25], [.85, .25], [.85, .65], [.6, .65], [.6, .25]])]
    if "prm_num_nodes" not in st.session_state:
        st.session_state["num_nodes"] = 100
    if "prm_num_neighbours" not in st.session_state:
        st.session_state["num_neighbours"] = 10
