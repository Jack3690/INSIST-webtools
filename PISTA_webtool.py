import streamlit as st
import pandas as pd
import numpy as np
from pista.analysis import Analyzer

st.set_page_config(
    page_title="PISTA",
    layout="wide"
)


st.title("PISTA")
st.header("Python Image Simulation and Testing Application")

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   PISTA simulates individual stars and adds different noises. 
    The input parameter space is designed to inculcate observational parameters,
    telescope parameters and detector characteristics.
	    """
    )

    st.markdown("")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    

    submit_button = st.form_submit_button(label="✨ Get me the data!")
