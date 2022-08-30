import streamlit as st
import pandas as pd
import numpy as np
from pista.analysis import Analyzer

st.set_page_config(
    page_title="PISTA",
    layout="wide"
)


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
ra = [0]
dec = [0]
mag = [10]
df = pd.DataFrame(zip(ra,dec,mag), columns= ['ra','dec','mag'])
sim = Analyzer(df)
fig, ax = sim.show_field()
with st.form(key="my_form"):
	c1, c2, c3 = st.columns([ 1, 2,1])
	with c1:
		submit_button = st.form_submit_button(label="✨ Get me the data!")
	with c2:
		img = st.pyplot(fig=fig,figsize = (2,2))
