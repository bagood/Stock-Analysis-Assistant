from warnings import simplefilter

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

class TimeSeriesPredictions:
    def __init__(self):
        simplefilter("ignore")
    
    def scrape_stock_price(self, start, end, emiten):
        stock_data = yf.download(emiten.upper() + '.JK', start, end).to_period('D')
        return stock_data
    
    def time_series_analysis_model(self, fourier_pairs, stock_data):
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
    
    
    def create_plot(self, stock_data, target_pred, target_fore, ax, emiten):
        stock_data['Close'].plot(ax=ax)
        target_pred.plot(ax=ax)
        target_fore.plot(ax=ax)
        ax.set_title(f'Saham {emiten.upper()}')
        ax.set_ylabel('Close Price')
        ax.legend(['Price', 'Model Forecasts'])
        plt.show()
        
        return
    
    def execute_time_series_analysis(self, emiten):
        end = datetime.now()
        start = datetime(2022, 1, 1)
        _, ax = plt.subplots()
        try:
            stock_data = self.scrape_stock_price(start, end, emiten)
            target_pred, target_fore, perc, rmse = self.time_series_analysis_model(12, stock_data)
            target_fore, perc = self.auto_rejection_stock_boundaries(stock_data, target_fore, perc)
            self.create_plot(stock_data, target_pred, target_fore, ax, emiten)
        except:
            return
        
        return (target_fore, perc, rmse)




