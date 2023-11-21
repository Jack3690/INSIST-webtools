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
		osys = poy.OpticalSystem(oversample = 10, npix = 4000)
	
		# On axis Aperture
		osys.add_pupil(poy.CircularAperture(radius=pri/2*u.cm))
		osys.add_pupil(poy.SecondaryObscuration(secondary_radius = sec*u.cm,
		                                        support_width = sec_width*u.cm,
		               support_angle_offset = 0))
		
		# MOS
		ap1 = poy.SquareAperture(size = 1*u.m)
		ap2 = poy.SecondaryObscuration(secondary_radius = 1*u.cm,
		                                        support_width = 2.5*u.cm,
		               support_angle_offset = 0)
		
		atlast = poy.CompoundAnalyticOptic( opticslist=[ap1, ap2 ], name='MOS Mirrorlets')
		#osys.add_pupil(atlast)
		
		
		# Detector
		osys.add_detector(pixelscale=0.1, fov_arcsec=40.1)
	
	psfs = 0
	for wav in np.linspace(150,300,100):
	  psf = osys.calc_psf(wav*1e-9)
	  psfs += psf[0].data
	with c2:
		wav = np.arange(1000, 8000, 1)
		flux = 3631/(3.34e4*wav**2)   # AB flux
		fig, ax = plt.subplots(figsize=(15,5))
		fig, ax, _, params_ = bandpass(wav, flux, sim.response_funcs,fig=fig, ax=ax,
		plot=True)

		st.text(f'Exposure time required for {mag} magnitude star with SNR = {SNR}: {np.round(exp_time,3)} seconds')
		
		lambda_phot, int_flux, int_flux_Jy, W_eff, flux_ratio = params_
		
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		ax.tick_params(which='both', width=2,direction="in", top = True,right = True,
		bottom = True, left = True)
		ax.tick_params(which='major', length=7,direction="in")
		ax.tick_params(which='minor', length=4, color='black',direction="in")
		
		st.pyplot(fig)

		text = f"**Central Wavelength** : {np.round(params['wavelength'],2)} " + r"$\AA$"
		text += f" \|| **Bandwidth** : {np.round(params['bandwidth'],2)} " + r"$\AA$"
		text += f" \|| **Effective area** : {np.round(params['effective_area'],2)} " + r"$cm^2$"
		text += f" \|| **Sky magnitude** : {np.round(params['sky_brightness'],2)} "
			
		st.caption(text)
	with c3:	
		fig, ax = sim.show_image(show_wcs=False)
		ax.set_title(None)
		fig.suptitle("2D SNR Output [ADUs]",fontsize=40)
		st.pyplot(fig)
		
