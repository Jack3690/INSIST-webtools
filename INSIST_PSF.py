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
		primary = st.number_input(
			    "Primary Mirror Aperture",
			    min_value=10.,
          		    value=100.,
			    max_value=3000.
          help = "Diameter of primary mirror in cms")
		
		sec_obs = st.number_input(
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

    
		
		submit_button = st.form_submit_button(label="✨ Calculate")

if submit_button:
	if filter == 'g' :
		tel_params ={
		    'aperture'       : 100,
		    'pixel_scale'    : 0.1,
		    'psf_file'       : f'{data_path}/PSF/INSIST/off_axis_poppy.npy',
		    'response_funcs' :  [ f'{data_path}/INSIST/G/M1.dat,5,100', 
					  f'{data_path}/INSIST/G/Dichroic.dat,1,100',
					  f'{data_path}/INSIST/G/Filter.dat,1,100',      # 6 mirrors
					  f'{data_path}/INSIST/G/QE.dat,1,100',
					],        
		     'coeffs'       : 1,
		     'theta'        : 0                  
		            }
	elif filter =='u' :
		tel_params ={
		    'aperture'       : 100,
		    'pixel_scale'    : 0.1,
		    'psf_file'       : f'{data_path}/PSF/INSIST/off_axis_poppy.npy',
		    'response_funcs' :  [ f'{data_path}/INSIST/U/M1.dat,5,100', 
					  f'{data_path}/INSIST/U/Dichroic.dat,2,100',
					  f'{data_path}/INSIST/U/Filter.dat,1,100',      # 6 mirrors
					  f'{data_path}/INSIST/U/QE.dat,1,100',
					],        
		     'coeffs'       : 1,
		     'theta'        : 0                  
		    }
	elif filter=='UV':
		tel_params = {
		    'aperture'       : 100,
		    'pixel_scale'    : 0.1,
		    'psf_file'       : f'{data_path}/PSF/INSIST/off_axis_poppy.npy',
		    'response_funcs' :  [f'{data_path}/INSIST/UV/Coating.dat,5,100', 
					 f'{data_path}/INSIST/UV/Dichroic.dat,2,100',
			    		 f'{data_path}/INSIST/UV/Filter.dat,1,100', 
					 f'{data_path}/INSIST/UV/QE.dat,1,100'
					 
					],        
		     'coeffs'       : 1,
		     'theta'        : 0                  
		    }	
	df = pd.DataFrame()
	df['ra']=[0,0]
	df['dec']=[0,0]
	df['mag']= [mag,100]
	
	sim = pt.Imager(df, tel_params=tel_params, n_x=51, n_y=51, exp_time=600)
	det_params = {'shot_noise' :  'Poisson',
              'qe_response': [],
              'qe_mean'    :  1,
              'G1'         :  1,
              'bias'       :  50,
              'PRNU_frac'  :  0.25/100,
              'RN'         :  3,
              'T'          :  218,
              'DN'         :  0.01/100
              }
	sim(det_params=det_params, photometry = None)
	params = {}
	
	params['wavelength'] = sim.lambda_phot
	params['bandwidth'] = sim.W_eff
	params['effective_area'] = np.pi*(100/2)**2*sim.flux_ratio
	params['sky_brightness'] = sim.det_params['M_sky']
	params['plate_scale'] = sim.pixel_scale
	params['aperture'] = 0.6
	params['dark_current'] = np.mean(sim.DR)
	params['read_noise'] = sim.det_params['RN']

	exp_time = float(exposure_time(params,mag,SNR))
	sim = pt.Imager(df, tel_params=tel_params, n_x=51, n_y=51, exp_time=exp_time)
	sim.QE = False
	sim(det_params=det_params, photometry = None, fwhm=1.5)
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
		
