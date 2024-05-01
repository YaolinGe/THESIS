import streamlit as st
import numpy as np

def initialize(): 

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
