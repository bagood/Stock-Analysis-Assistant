import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from tsa_predictions import TSAPredictions
from technical_analysis_indicators import TechnicalAnalysisIndicators
from website_functions import WebsiteFunctions

wf = WebsiteFunctions()
tai = TechnicalAnalysisIndicators()

st.set_page_config(
    page_title="Stock Analysis Assistant",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "This Web App is made by Data Apes Team for TSDN 2022"
    }
)
with st.sidebar:
    select = wf.sidebar()

emiten = wf.select_main()

if select == "Stock Analysis":
    tab1, tab2 = st.tabs(['Technical', 'Fundamental'])
    with tab1:
        wf.select_technical(emiten)