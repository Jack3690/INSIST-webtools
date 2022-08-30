import streamlit as st
import pandas as pd
import numpy as np

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
