import streamlit as st
import pandas as pd
import numpy as np
import pista as pis
from pathlib import Path
from astropy.table import Table
from astropy.io import fits

st.set_page_config(
    page_title="PISTA",
    layout="wide"
)

data_path = Path(pis.__file__).parent.joinpath()

st.title("INSIST-PISTA")
st.header("Python Image Simulation and Testing Application (PISTA)")

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
   	This webtool is based on PISTA, a python-based resolved-stellar population simulation package. 
	This interface is designed to take minimal input from user to simulate fields with INSIST specifications
	    """
    )

    st.markdown("")

with st.form(key="my_form"):
	c1, c2, c3 = st.columns([ 1, 2,0.8])
	with c1:
		df_upload = st.file_uploader('DataFrame', type=['fits','csv'])
		
		exp_time = st.number_input(
			    "Exposure Time",
			    min_value=1,
			    max_value=10000)
		
		n_x = st.number_input(
			    "n_pix x axis",
			    value =1000,
			    min_value=10,
			    max_value=8000)
		
		n_y = st.number_input(
			    "n_pix y axis",
			    value =1000,
			    min_value=10,
			    max_value=8000)
		submit_button = st.form_submit_button(label="✨ Generate Image")

Valid_df = False
df = None
if df_upload is not None:
	if 'csv' in df_upload.type:
		df = pd.read_csv(df_upload)
	if 'fit' in df_upload.type or 'fits' in df_upload.type or 'octet' in df_upload.type:
		hdu = fits.open(df_upload)
		st.write(hdu)
		df = tab.to_pandas()
	Valid_df = True
if df is not None:
	for i in ['ra','dec','mag']:
		if i not in df.keys():
			Valid_df = False	
if not Valid_df:
	st.write('Default DataFrame selected')
	ra = [0]
	dec = [0]
	mag = [10]
	df = pd.DataFrame(zip(ra,dec,mag), columns = ['ra','dec','mag'])
if submit_button:
	sim = pis.Analyzer(df=df, exp_time = exp_time, n_x = n_x, n_y = n_y)
	sim()
	fig,ax = sim.show_image(cmap = 'gray')
	with c2:
		img = st.pyplot(fig=fig)
	with c3:
		fig, ax = sim.show_field()
		img1 = st.pyplot(fig=fig)
		
		fig, ax = sim.show_image('Source')
		img2 = st.pyplot(fig=fig)
		
		fig, ax = sim.show_image('DC')
		img3 = st.pyplot(fig=fig)


