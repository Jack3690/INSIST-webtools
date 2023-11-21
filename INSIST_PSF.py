import streamlit as st
import pandas as pd
import numpy as np
import poppy as poy

from pathlib import Path
from astropy.table import Table
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
import seaborn as sb
from astropy.modeling import fitting, models
from scipy.integrate import quadrature,trapz
import matplotlib

sb.set_style('white')
data_path = pt.data_dir
matplotlib.rcParams['font.size']=20
matplotlib.rcParams['figure.figsize']=(10,10)


st.set_page_config(
    page_title="INSIST PSF Simulator",
    layout="wide"
)
st.title("INSIST-PSF")
st.subheader("A basic PSF Simulator using POPPY")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
   	   This webtool is for generic PSF Simulation
	    """
    )

    st.markdown("")

with st.form(key="my_form"):
	c1, c2, c3 = st.columns([ 1, 2,0.8])
	with c1:
		on_off = st.selectbox('Type',
			  ('Off Axis', 'On Axis'))
		
		pri = st.number_input(
			    "Primary Mirror Aperture",
			    min_value=10.,
			    value=100.,
			    max_value=3000.
	                    help = "Diameter of primary mirror in cms")
		
		sec = st.number_input(
			    "Secondary Obstruction Diameter",
			    value =20,
			    min_value=1,
			    max_value=3000,
	                   help="Diameter of the secondary obstruction in cms")

		sec_width = st.number_input(
				"Spider width",
				value =20,
				min_value=1,
				max_value=3000,
				help="Width of strcuture supporting the secondary in cms")
	
		focal_length = st.number_input(
				"Total Focal Length",
				value =20,
				min_value=1,
				max_value=3000,
				help="Width of strcuture supporting the secondary in cms")
		st.subheader('Wavelength')
		wav_min = st.number_input(
				r"$\lambda_1$",
				value =100,
				min_value=1,
				max_value=30000,
				help="Starting wavelength in Angstrom")
		wav_max = st.number_input(
			r"$\lambda_2$",
			value =100,
			min_value=1,
			max_value=30000,
			help="Ending wavelength in Angstrom")
		
		wav_step = st.number_input(
			r"$\\delta lambda$",
			value =100,
			min_value=1,
			max_value=30000,
			help="Wavelength step in Angstrom")
		
submit_button = st.form_submit_button(label="✨ Calculate")

if submit_button:
	if on_off == 'Off Axis':
		osys = poy.OpticalSystem(oversample = 5, npix = 2000)
	
		# On axis Aperture
		osys.add_pupil(poy.CircularAperture(radius=pri/2*u.cm))
		osys.add_pupil(poy.SecondaryObscuration(secondary_radius = sec*u.cm,
		                                        support_width = sec_width*u.cm,
		               support_angle_offset = 0))	
		
		# Detector
		osys.add_detector(pixelscale=0.1, fov_arcsec=20.1)
	
	psfs = 0
	for wav in np.linspace(150,300,100):
	  psf = osys.calc_psf(wav*1e-9)
	  psfs += psf[0].data
	psf[0].data = psfs/psfs.max()
	
	with c2:
		fig, ax = plt.subplots()
		ax, cb = poy.display_psf(psf, title = 'Broadband PSF', ax=ax,return_ax=True)
		ax.grid(False)
		st.pyplot(fig)
	with c3:	
		st.pyplot(osys.display())
		
