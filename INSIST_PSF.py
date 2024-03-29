import streamlit as st
import pandas as pd
import numpy as np
import poppy as poy

from pathlib import Path
from astropy.table import Table
from astropy.io import fits
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
import seaborn as sb
from astropy.modeling import fitting, models
from scipy.integrate import quadrature,trapz
import matplotlib
import io

sb.set_style('white')

matplotlib.rcParams['font.size']=20
matplotlib.rcParams['figure.figsize']=(10,10)


st.set_page_config(
    page_title="INSIST-PSF",
    layout="wide"
)
st.title("INSIST PSF Simulator")
st.subheader("A basic PSF Simulator using POPPY")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
   	   This webtool is for generic PSF Simulation. The PSF is 5 times oversampled and is generated for a FoV of 20.1 arcseconds. 
       	   The PSF can be downloaded as a Numpy .npy file.
	    """
    )

    st.markdown("")

#with st.form(key="my_form"):
c1, c2, c3 = st.columns([ 1, 2,1.2])
with c1:
	st.subheader('Optics')
	on_off = st.selectbox('Type',
		  ('Off Axis', 'On Axis'))
	
	pri = st.number_input(
		    "Primary Mirror Aperture",
		    min_value=10.,
		    value=100.,
		    max_value=3000.,
		    help = "Diameter of primary mirror in cms")
	
	sec = st.number_input(
		    "Secondary Obstruction Diameter",
		    value =20,
		    min_value=1,
		    max_value=3000,
		    help="Diameter of the secondary obstruction in cms")

	sec_width = st.number_input(
			"Spider width",
			value = 2.5,
			min_value=1.,
			max_value=3000.,
			help="Width of structure supporting the secondary in cms")

	st.subheader('Wavelength')
	wav_min = st.number_input(
			r"$\lambda_1$",
			value =1500.,
			min_value=1.,
			max_value=30000.,
			help="Starting wavelength in Angstrom")
	
	wav_max = st.number_input(
		r"$\lambda_2$",
		value =3000.,
		min_value=1.,
		max_value=30000.,
		help="Ending wavelength in Angstrom")
	
	wav_step = st.number_input(
		r"$\delta \lambda$",
		value =50.,
		min_value=0.1,
		max_value=100.,
		help="Wavelength step in Angstrom")
	
	ps = st.number_input(
		"Pixel scale",
		value =0.1,
		min_value=0.,
		max_value=5.,
		help="Pixel scale in arcsec/pixel")
	
	submit_button = st.button(label="✨ Simulate")

if submit_button:
	if on_off == 'Off Axis':
		osys = poy.OpticalSystem(oversample = 5, npix = 2000)
	
		# Off axis Aperture
		osys.add_pupil(poy.CircularAperture(radius=(pri/2)*u.cm))
		
		# Detector
		osys.add_detector(pixelscale=0.1, fov_arcsec=20.1)
	if on_off == 'On Axis':
		osys = poy.OpticalSystem(oversample = 5, npix = 2000)
	
		# On axis Aperture
		osys.add_pupil(poy.CircularAperture(radius=(pri/2)*u.cm))
		osys.add_pupil(poy.SecondaryObscuration(secondary_radius = (sec/2)*u.cm,
		                                        support_width = sec_width*u.cm,
		               support_angle_offset = 0))	
		
		# Detector
		osys.add_detector(pixelscale=ps, fov_arcsec=20.1)
	
	psfs = 0
	for wav in np.arange(wav_min, wav_max, wav_step):
	  psf = osys.calc_psf(wav*1e-10)
	  psfs += psf[0].data
		
	psf[0].data = psfs/psfs.max()
	
	with c2:
		fig, ax = plt.subplots()
		ax, cb = poy.display_psf(psf, title = 'Broadband PSF', ax=ax,return_ax=True)
		ax.grid(False)
		st.pyplot(fig)
		with io.BytesIO() as buffer:
		    # Write array to buffer
		    np.save(buffer, psf[0].data)
		    st.download_button("Download PSF", buffer, 'psf.npy')
		
	with c3:	
		if on_off == 'Off Axis':
			fig, ax = plt.subplots(1,1, figsize=(7,7))
			osys.planes[0].display(ax=ax)

		elif on_off == 'On Axis':
			fig, ax = plt.subplots(2,1, figsize=(9,18))
			osys.planes[0].display(ax=ax[0])
			osys.planes[1].display(ax=ax[1])
		st.pyplot(fig)

