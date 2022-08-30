import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="PISTA",
    layout="wide"
)


st.title("PISTA")
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


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)", "Flair"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Default (DistilBERT)":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT(model=roberta)

            kw_model = load_model()

        else:
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.

