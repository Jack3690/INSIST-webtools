import streamlit as st
import pandas as pd
import numpy as np
import pista as pis
from pathlib import Path

st.set_page_config(
    page_title="PISTA",
    layout="wide"
)

data_path = Path(pis.__file__).parent.joinpath()

st.title("INSIST-PISTA")
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
df = pd.read_csv(f'{data_path}/data/example.csv')
sim = pis.Analyzer(df)
sim()
fig,ax = sim.show_image()
with st.form(key="my_form"):
	c1, c2, c3 = st.columns([ 1, 2,1])
	with c1:
		df_up = st.file_uploader(DataFrame, type=['fits','csv'])
		submit_button = st.form_submit_button(label="✨ Get me the data!")
	with c2:
		img = st.pyplot(fig=fig,figsize = (2,2))
