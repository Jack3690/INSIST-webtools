import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PISTA",
)

def _max_width_():
    max_width_str = f"max-width: 2800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()
st.title("PISTA")
st.header("Python Image Simulation and Testing Application")
