from warnings import simplefilter

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

class TechnicalAnalysisIndicators:
    def __init__(self):
        plt.style.use("dark_background")
        # plt.style.use("seaborn-whitegrid")
        plt.rc("figure", autolayout=True, figsize=(12, 10))
        plt.rc(
            "axes",
            labelweight="bold",
            labelsize="large",
            titleweight="bold",
            titlesize=16,
            titlepad=10,
        )
        simplefilter("ignore")
        self.data = pd.read_csv('emiten_code_list.csv')    

    def show_emitens_code(self):
        codes = self.data['Kode'].values

        return codes
    
    def translate_code_to_emiten(self, emiten):
        name = self.data.loc[self.data['Kode'] == emiten, 'Nama Perusahaan'].values[0]

        return name

    def scrape_stock_price(self, emiten, start=datetime(2022, 1, 1), end=datetime.now()):
        """Collects an emiten's stocks price data from yahoo finance
        :param start: a datetime object that tells the first date where the datas are being collected
        :param end: a datetime object that tells the last date where the datas are being collected
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :return stock_data: a pandas dataframe that contain the emiten's stocks data
        """
        stock_data = yf.download(emiten.upper() + '.JK', start, end)
        return stock_data
    
    def visualize(self, stock_data, indicators=[]):
        fig, ax = plt.subplots()
        
        stock_data['Close'].plot(ax=ax, c='r')
        if 'Volume' in indicators:
            ax_ = ax.twinx()
            stock_data['Volume'].plot(ax=ax_, c='y')
        
        return fig