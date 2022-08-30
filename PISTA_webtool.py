import streamlit as st
import pandas as pd
import numpy as np
import pista as pis
from pathlib import Path
from astropy.table import Table

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

with st.form(key="my_form"):
	c1, c2, c3 = st.columns([ 1, 2,1])
	with c1:
		df_upload = st.file_uploader('DataFrame', type=['fits','csv'])
		
		exp_time = st.number_input(
			    "Exposure Time",
			    min_value=1,
			    max_value=10000)
		submit_button = st.form_submit_button(label="✨ Generate Image")


if df_upload is not None:
	if 'csv' in df_upload.type:
		df = pd.read_csv(df_upload)
	if 'fit' in df_upload.type or 'fits' in df_upload.type :
		df = Table.read(df_upload).to_pandas()

st.write(df_upload)
if submit_button:
	sim = pis.Analyzer(df=df, exp_time = exp_time)
	sim()
	fig,ax = sim.show_image(cmap = 'gray')
	with c2:
		img = st.pyplot(fig=fig)


