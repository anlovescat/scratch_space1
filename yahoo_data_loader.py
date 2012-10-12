from matplotlib.finance import quotes_historical_yahoo, quotes_historical_yahoo, quotes_historical_yahoo
import matplotlib.finance as mfinance
import numpy as np
import datetime

def getYahooData(ticker, date1, date2):
    data = quotes_historical_yahoo(ticker, date1, date2, asobject=True, adjusted=True)
    data.date = [datetime.date.fromordinal(int(dt)) for dt in data.date]
    data = zip(data.date, data.open, data.close, data.high, data.low, data.volume)
    data = np.array(data, dtype = [('Date', datetime.date), 
                                   ('Open', float), ('Close', float), ('High', float),
                                   ('Low', float), ('Volume', float)])
    idx = np.where(data['Volume'] > 0)
    data = data[idx]
    return data




    
