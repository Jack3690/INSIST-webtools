import streamlit as st
import pandas as pd
import numpy as np
import pista as pt
from pista.utils import bandpass

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
    page_title="INSIST ETC",
    layout="wide"
)
st.title("INSIST-Exposure Time Calculator")
st.header("A basic exposure time calculator for the INdian Spectroscopic and Imaging Space Telescope")


def coeff_calc(x0,xn,x=None,y=None,mode = None):
  if mode == 'Gaussian':
    model = models.Gaussian1D(mean = (x0+xn)*0.5, stddev = xn-x0)
    int_y,err = quadrature(model,x0,xn)
    return int_y/(xn-x0)
  else :
    return trapz(y,x)/(xn-x0)

def exposure_time(det_params,M,SNR):
  wavelength     = det_params['wavelength']
  bandwidth      = det_params['bandwidth']
  effective_area = det_params['effective_area']
  M_sky          = det_params['sky_brightness']
  plate_scale    = det_params['plate_scale']
  aperture       = det_params['aperture']
  dark_current   = det_params['dark_current']
  read_noise     = det_params['read_noise']

  F_0_p   = 1.51e3*(bandwidth/wavelength)*3631*effective_area
  F_m_p   = F_0_p*pow(10,-0.4*M)
  M_sky_p = M_sky - 2.5*np.log10(plate_scale**2)
  F_sky_p = F_0_p*pow(10,-0.4*M_sky_p)

  n_pix   = np.pi*((0.5*aperture)/plate_scale)**2

  A =  (F_m_p/SNR)**2
  B = -(F_m_p + F_sky_p*n_pix + dark_current*n_pix)
  C = -n_pix*(read_noise)**2

  t1 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
  t2 = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
  t = np.where(t1>0,t1,t2)
  
  return t

with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
   	This webtool is based on preliminary design for INSIST.
	    """
    )

    st.markdown("")

with st.form(key="my_form"):
	c1, c2, c3 = st.columns([ 1, 2,0.8])
	with c1:
		filter = st.selectbox('Filter',
                          ('g', 'u', 'UV'))
		SNR = st.number_input(
			    "SNR",
			    min_value=1.,
          		    value=5.,
			    max_value=10000.)
		
		mag = st.number_input(
			    "mag",
			    value =20,
			    min_value=0,
			    max_value=35,
          help="Magnitude in AB system")
		
		submit_button = st.form_submit_button(label="✨ Generate Image")

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

		st.text(f'Exposure time required for {mag} magnitude star with SNR = {SNR}: {exp_time}')
		
		lambda_phot, int_flux, int_flux_Jy, W_eff, flux_ratio = params_
		
		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		
		ax.tick_params(which='both', width=2,direction="in", top = True,right = True,
		bottom = True, left = True)
		ax.tick_params(which='major', length=7,direction="in")
		ax.tick_params(which='minor', length=4, color='black',direction="in")
		
		st.pyplot(fig)

		text = ""
		for i, value in params.items():
			if i=='plate_scale':
				break
			if i == 'wavelength':
				text+= "Central"
			text += f"**{i.capitalize()}** : {np.round(value,2)} "
		st.caption(text)
	with c3:	
		fig, ax = sim.show_image(show_wcs=False)
		ax.set_title(None)
		fig.suptitle("2D SNR Output [ADUs]",fontsize=40)
		st.pyplot(fig)
		
