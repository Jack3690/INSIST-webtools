import streamlit as st
import pandas as pd
import numpy as np
import pista as pis
from pathlib import Path
from astropy.table import Table
from astropy.io import fits
from matplotlib import colors as col
import matplotlib.pyplot as plt

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
		hdul = fits.open(df_upload)
		for hdu in hdul:
			if 'XTENSION' in hdu.header.keys():
				if hdu.header['XTENSION']=='BINTABLE':
					tab = Table(hdu.data)
					df = tab.to_pandas()
	Valid_df = True
if df is not None:
	for i in ['ra','dec','mag']:
		if i not in df.keys():
			Valid_df = False	
if not Valid_df:
	st.write('Default DataFrame selected')
	ra = [10]
	dec = [10]
	mag = [10]
	df = pd.DataFrame(zip(ra,dec,mag), columns = ['ra','dec','mag'])
else : 
	st.write(f'{df_upload.name} dataframe selected')
if submit_button:
	sim = pis.Analyzer(df=df, exp_time = exp_time, n_x = n_x, n_y = n_y)
	sim.cuda = False
	sim.fftconv = False
	sim()
	norm = col.LogNorm()
	fig = plt.figure()
	ax = fig.add_subplot(projection = sim.wcs)
	img = ax.imshow(sim.digital,cmap='gray' , norm = norm)
	plt.colorbar(img,ax = ax, location = 'bottom', anchor = (0.5,1.8), shrink = 0.75)
	ax.set_title(f'Digital \nRequested center : {sim.name}')
	ax.grid(False)
	with c2:
		img = st.pyplot(fig=fig)
	with c3:
		fig, ax = sim.show_field()
		img1 = st.pyplot(fig=fig)

		fig = plt.figure()
		ax = fig.add_subplot()
		img = ax.imshow(sim.light_array,cmap='jet' , norm = norm)
		plt.colorbar(img,ax = ax, location = 'bottom', anchor = (0.5,1.8), shrink = 0.75)
		ax.set_title(f'Source \nRequested center : {sim.name}')
		ax.grid(False)
		img2 = st.pyplot(fig=fig)
		
		fig = plt.figure()
		ax = fig.add_subplot()
		img = ax.imshow(sim.DC_array,cmap='seismic' , norm = norm)
		plt.colorbar(img,ax = ax, location = 'bottom', anchor = (0.5,1.8), shrink = 0.75)
		ax.set_title(f'DC \nRequested center : {sim.name}')
		ax.grid(False)
		img3 = st.pyplot(fig=fig)
