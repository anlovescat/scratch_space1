import csv
from matplotlib.mlab import csv2rec
import numpy as np

SHANGHAI_FILE = 'shanghai_list.csv'
SHENZHEN_FILE = 'shenzhen_list.csv'

HEADER = ['last_price', 'volume', 'ticker', 'amount', 'vol_per_trade',
          'exchange_percent', 'pe_ratio', 'liquid_asset', 'total_asset', 'pb_ratio']

def ticker_converter(tick_num, exchange):
    tick_num += 1000000
    ticker = str(tick_num) + '.' + exchange
    return ticker[1:]

def load_file(filename):
    if filename == SHANGHAI_FILE:
        exchange = 'SS'
    elif filename == SHENZHEN_FILE:
        exchange = 'SZ'
    else :
        raise "unknown filename"
    tmp = csv2rec(filename)
    ticker = [ticker_converter(tick_num, exchange) for tick_num in tmp['ticker']]
    data = zip(ticker, tmp['last_price'], tmp['volume'], tmp['amount'],
               tmp['pe_ratio'], tmp['liquid_asset'], tmp['total_asset'], tmp['pb_ratio'])
    data = np.array(data, dtype = [('ticker', 'S9'), ('last_price', float), ('volume', int),
                                   ('amount', float), ('pe_ratio', float), ('liquid_share', float),
                                   ('total_share', float), ('pb_ratio', float)])
    return data

def load_ticker_info():
    ssdata = load_file(SHANGHAI_FILE)
    szdata = load_file(SHENZHEN_FILE)
    data = np.append(ssdata, szdata)
    return data



