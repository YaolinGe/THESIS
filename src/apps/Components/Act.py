import os
import streamlit as st
from PIL import Image

def renderActPage():
    st.title("Act")
    
    figpath = os.path.join(os.getcwd(), "src", "apps", "figs")
    image1 = Image.open(os.path.join(figpath, "act.webp"))
    image2 = Image.open(os.path.join(figpath, "ros_imc_bridge.png"))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, use_column_width=True)
    with col2:
        st.image(image2, use_column_width=True)

if __name__ == "__main__":
    renderActPage()
