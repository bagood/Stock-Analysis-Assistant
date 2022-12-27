from warnings import simplefilter

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from tsa_predictions import TSAPredictions

class TechnicalAnalysisIndicators:
    def __init__(self):
        plt.style.use("dark_background")
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
        self.tsa = TSAPredictions()
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
    
    def _add_volume(self, stock_data, ax):
        ax_ = ax.twinx()
        stock_data['Volume'].plot(ax=ax_, c='yellow', alpha=0.50)

        return 
    
    def _add_tsa_predictions(self, emiten, stock_data, ax):
        target_fore, perc, rmse = self.tsa.execute_time_series_analysis(emiten)
        data_fore = pd.DataFrame({'Date':[stock_data.index[-1], 
                                            stock_data.index[-1] + timedelta(days=1)],
                                    'Close':[stock_data['Close'].values[-1],
                                                target_fore[0]]}).set_index('Date')
        data_fore['Close'].plot(ax=ax, c='lime', alpha=0.50)

        return
    
    def _add_ma(self, stock_data, window, ax):
        ma_series = stock_data['Close'].rolling(window, min_periods=1).mean()
        ma_series.plot(ax=ax, alpha=0.50)

        return

    def visualize(self, emiten, stock_data, indicators=[]):
        fig, ax = plt.subplots()
        stock_data['Close'].plot(ax=ax, c='red')
        if 'Volume' in indicators:
            self._add_volume(stock_data, ax)
        check_data_fore = True
        if 'Forecast' in indicators and check_data_fore:
            self._add_tsa_predictions(emiten, stock_data, ax)
            check_data_fore = False
        if 'MA20' in indicators:
            self._add_ma(stock_data, 20, ax)
        if 'MA50' in indicators:
            self._add_ma(stock_data, 50, ax)
        if 'MA100' in indicators:
            self._add_ma(stock_data, 100, ax)

        return (fig, ax)