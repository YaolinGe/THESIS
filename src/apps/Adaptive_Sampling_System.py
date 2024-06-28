"""
This app is used to demonstrate the GRF algorithm. The user can adjust the parameters of the algorithm and see the results in real-time.
"""
import streamlit as st
from Components.Sense import renderSensePage
from Components.Plan import renderPlanPage
from Components.Act import renderActPage
import os
import requests

st.set_page_config(
    page_title="Demo GRF",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://yaolinge.github.io/',
        'Report a bug': "https://yaolinge.github.io/",
        'About': "This is a demo GRF app. Have fun!"
    }
)

page_options = ['🧪 Sense', '🎲 Plan', '🤖 Act']

selected_page = st.sidebar.selectbox('Adaptive Sampling System', page_options)

# Download the data
url = "https://drive.google.com/file/d/1mIGUevAlptjYIOdOjpSiVss2VyW-1qsS/view?usp=sharing"
if not os.path.exists("interpolator_medium.joblib"):
    response = requests.get(url)
    with open("interpolator_medium.joblib", "wb") as file:
        file.write(response.content)

        if selected_page == '🧪 Sense':
            renderSensePage()
        elif selected_page == '🎲 Plan':
            renderPlanPage()
        elif selected_page == '🤖 Act':
            renderActPage()




