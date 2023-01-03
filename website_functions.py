import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

from tsa_predictions import TSAPredictions
from technical_analysis_indicators import TechnicalAnalysisIndicators

class WebsiteFunctions:
    def __init__(self):
        self.tsa = TSAPredictions()
        self.tai = TechnicalAnalysisIndicators()
    
    def sidebar(self):
        st.header("Menu")
        select = option_menu(
            menu_title=None,
            options=["Stock Analysis", "Notes"],
            icons=["cash-stack", "filter-left"],
            styles={"nav-link":{"font-size":"13px"}}
        )
    
        return select
    
    def select_main(self):
        st.markdown("<h1 style='text-align: center; '>Stock Analysis Assistant</h1>",
                    unsafe_allow_html=True)
        st.write('')
        emiten = st.selectbox("Choose an Emiten",self.tai.show_emitens_code())
        st.subheader(f'{self.tai.translate_code_to_emiten(emiten)}')
        
        return emiten
    
    def select_technical(self, emiten):
        dates = st.date_input("Start Date:", datetime.now() - timedelta(days=7))
        stock_data = self.tai.scrape_stock_price(emiten)
        indicators = st.multiselect('Select indicators to show', ['Volume', 'Forecast', 'MA20', 'MA50', 'MA100'])
        fig = self.tai.visualize(emiten, stock_data, dates, indicators)
        st.pyplot(fig)

        return