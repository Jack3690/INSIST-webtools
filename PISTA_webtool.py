import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PISTA",
)

def _max_width_():
    max_width_str = f"max-width: 1800px;"
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
c30, c31, c32 = st.columns([10, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("Python Image Simulation and Testing Application")
    st.header("")
