from warnings import simplefilter

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

class TSAPredictions:
    def __init__(self):
        simplefilter("ignore")
    
    def scrape_stock_price(self, emiten, start=datetime(2022, 1, 1), end=datetime.now()):
        """Collects an emiten's stocks price data from yahoo finance
        :param start: a datetime object that tells the first date where the datas will be collected
        :param end: a datetime object that tells the last date where the datas are being collected
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :return stock_data: a pandas dataframe that contain the emiten's stocks data
        """
        stock_data = yf.download(emiten.upper() + '.JK', start, end).to_period('D')
        return stock_data
    
    def time_series_analysis_model(self, fourier_pairs, stock_data):
        """Creates the time series analysis model that fits the stock's price over a set period of time
        :param fourier_pairs: an integer that represents the number of fourier pairs used to approximate the stock price
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :return target_pred: a numpy arrays consisting of predictions made for the stock's price
        :return target_fore: a numpy arrays that consist 1 day forecast of the stock's price
        :return perc: a float that represents the stock's price percentage increase on the next day 
        :return rmse: a float that represents the root mean squared of the predictions made
        """
        target = stock_data['Close']
        fourier = CalendarFourier(freq="A", order=fourier_pairs) 
        dp = DeterministicProcess(
            index=stock_data.index,
            constant=True,              
            order=1,                   
            seasonal=True,               
            additional_terms=[fourier],
            drop=True,                   
        )
        X = dp.in_sample() 
        linreg_model = LinearRegression(fit_intercept=False)
        linreg_model.fit(X, target)
        target_pred = pd.Series(linreg_model.predict(X), index=target.index)
        features_fore = dp.out_of_sample(steps=1)
        target_fore = pd.Series(linreg_model.predict(features_fore), index=features_fore.index)
        rmse = mean_squared_error(target, target_pred, squared=False)**0.5
        perc = (target_fore[0] * 100 / target[-1]) - 100      

        return (target_pred, target_fore, perc, rmse)        

    def auto_rejection_stock_boundaries(self, stock_data, target_fore, perc):
        """Tunes the stock's price increase when it is outside the auto rejection boundaries
        :param stock_data: a pandas dataframe that contain the emiten's stocks data
        :param target_fore: a numpy arrays that consist 1 day forecast of the stock's price
        :param perc: a float that represents the stock's price percentage increase on the next day 
        :return target_fore: tunned 1 day forecast of the stock's price
        :return perc: tunned stock's price percentage increase on the next day 
        """
        target = stock_data['Close']
        if target[-1] > 5000:
            cap = 20
        elif target[-1] > 200:
            cap = 25
        else:
            cap = 35      
        if np.abs(perc) > cap and perc > 0:
            perc = cap
            target_fore = target[-1] * (100 + cap)
        elif np.abs(perc) > cap and perc > 0:
            perc = cap * -1
            target_fore = target[-1] * (100 + cap) 

        return (target_fore, perc)
    
    # def create_plot(self, stock_data, target_pred, target_fore, ax, emiten): 
    #     """Creates the visualisation for the stock's price, stock price predictions, and stock price forecast
    #     :param stock_data: a pandas dataframe that contain the emiten's stocks data
    #     :param target_pred: a numpy arrays consisting of predictions made for the stock's price
    #     :param target_fore: a numpy arrays that consist 1 day forecast of the stock's price
    #     :param ax: an integer that repersentes the axes for which the plots will be visualized
    #     :param emiten: a string that represents the emiten's code that is listed in the stock market
    #     """
    #     stock_data['Close'].plot(ax=ax)
    #     target_pred.plot(ax=ax)
    #     target_fore.plot(ax=ax)
    #     ax.set_title(f'Saham {emiten.upper()}')
    #     ax.set_ylabel('Close Price')
    #     ax.legend(['Price', 'Model Forecasts'])
    #     plt.show()
        
    #     return
    
    def execute_time_series_analysis(self, emiten):
        """Generate the 1 day forecast of the stock price
        :param emiten: a string that represents the emiten's code that is listed in the stock market
        :return target_fore: 1 day forecast of the stock's price
        :return perc: stock's price percentage increase on the next day
        :return rmse: a float that represents the root mean squared of the predictions made
        """
        try:
            stock_data = self.scrape_stock_price(emiten)
            target_pred, target_fore, perc, rmse = self.time_series_analysis_model(12, stock_data)
            target_fore, perc = self.auto_rejection_stock_boundaries(stock_data, target_fore, perc)
            # self.create_plot(stock_data, target_pred, target_fore, ax, emiten)
        except:
            return
        
        return (target_fore, perc, rmse)
