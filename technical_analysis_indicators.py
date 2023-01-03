from warnings import simplefilter

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
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
        """Show the list of all emitens that can be analyzed
        :return codes: a list containing all the emiten's code
        """
        codes = self.data['Kode'].values

        return codes
    
    def translate_code_to_emiten(self, emiten):
        """Match a code to it's emiten
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :return name: a string that represents the name of the company related to its emiten's code
        """
        name = self.data.loc[self.data['Kode'] == emiten, 'Nama Perusahaan'].values[0]

        return name

    def scrape_stock_price(self, emiten):
        """Collects an emiten's stocks price data from yahoo finance
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :return stock_data: a pandas dataframe that contain the emiten's stocks data
        """
        start = datetime(2021, 1, 1)
        end = datetime.now()
        stock_data = yf.download(emiten.upper() + '.JK', start, end)

        return stock_data

    def subset_from_date(self, stock_data, start=date.today() - timedelta(days=7)):
        """Subsets the dataframe starting from the date that the user wants to visualize
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :param start: a datetime object that tells the first date where the datas will be collected
        :return subset_data: a pandas dataframe that contain the subsetted emiten's stocks data
        """
        try:
            start = start.date()
        except:
            pass
        stock_data_ = stock_data.reset_index()
        stock_data_['Date'] = stock_data_['Date'].apply(lambda row: row.date())
        avail_dates = stock_data_['Date'].values
        while start not in avail_dates:
            start = start - timedelta(days=1)
        index_start = list(avail_dates).index(start)
        subset_dates = avail_dates[index_start:]
        subset_data = stock_data_.loc[stock_data_['Date'].isin(subset_dates), :].set_index('Date')

        return subset_data
    
    def _add_volume(self, stock_data, ax):
        """Adds the volume indicator into the plot
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :param ax: an integer that repersentes the axes for which the plots will be visualized        
        """
        ax_ = ax.twinx()
        stock_data[['Volume']].plot(ax=ax_, c='yellow', alpha=0.50)

        return 
    
    def _add_tsa_predictions(self, emiten, stock_data, ax):
        """Adds the time series analysis predictions into the plot
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :param ax: an integer that repersentes the axes for which the plots will be visualized          
        """
        target_fore, perc, rmse = self.tsa.execute_time_series_analysis(emiten)
        data_fore = pd.DataFrame({'Date':[stock_data.index[-1], 
                                            stock_data.index[-1] + timedelta(days=1)],
                                    'Forecast':[stock_data['Close'].values[-1],
                                                target_fore[0]]}).set_index('Date')
        data_fore[['Forecast']].plot(ax=ax, c='lime', alpha=0.50)

        return
    
    def _add_ma(self, stock_data, window, ax, start):
        """Adds the moving average indicators into the plot
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :param window: an integer the represents the number of window for the generated moving average
        :param ax: an integer that repersentes the axes for which the plots will be visualized     
        :param start: a datetime object that tells the first date where the datas will be collected   
        """
        ma = stock_data['Close'].rolling(window, min_periods=1).mean()
        ma_series = self.subset_from_date(ma, start).rename(columns={'Close':f'MA{window}'})
        ma_series.plot(ax=ax, alpha=0.50)

        return

    def visualize(self, emiten, stock_data, start, indicators=[]):
        """Visualize the indicators of an emiten's stock data
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :param start: a datetime object that tells the first date where the datas will be collected
        :param indicators: a list containing the indicators that will be shown in the plot
        :return fig: a figure object that contained the visualizations of the stock's data
        """
        fig, ax = plt.subplots()
        stock_data_vis = self.subset_from_date(stock_data, start)
        stock_data_vis[['Close']].plot(ax=ax, c='r')
        if 'Volume' in indicators:
            self._add_volume(stock_data_vis, ax)
        check_data_fore = True
        if 'Forecast' in indicators and check_data_fore:
            self._add_tsa_predictions(emiten, stock_data_vis, ax)
            check_data_fore = False
        if 'MA20' in indicators:
            self._add_ma(stock_data, 20, ax, start)
        if 'MA50' in indicators:
            self._add_ma(stock_data, 50, ax, start)
        if 'MA100' in indicators:
            self._add_ma(stock_data, 100, ax, start)

        return fig