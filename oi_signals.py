import requests
import pandas as pd
from datetime import datetime
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import AutocompleteInput,ColumnDataSource, Span, SingleIntervalTicker, LinearAxis, PreText, Div, Button, LabelSet, Label, DaysTicker, Range1d
from bokeh.models.tools import HoverTool
from bokeh.layouts import row, column
from bokeh.transform import dodge
import time as _time
from datetime import date
from bokeh.models.formatters import DatetimeTickFormatter
from nsepy import get_history
import os
import json
import sys
from pytz import timezone
from numpy import arange

# Dataframe settings
pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 1500)
symbol = "NIFTY"
import time
start_time = time.time()

buyZone = {}
sellZone = {}

ctime = datetime.now(timezone('Asia/Kolkata'))
hour = int(ctime.strftime("%H"))
minute = int(ctime.strftime("%M"))
filter1 = (hour * 3600) + (minute * 60)

def get_df(symbol):
    global headers, pcr, pcr_change, pe_oi_change,ce_oi_change, max_oi_change, max_oi, minValue, maxValue, minRange, maxRange, mp, lastPrice,df, width, min_oi_change, min_oi
    # URL and Headers for getting option data
    if symbol == "NIFTY" or symbol == "BANKNIFTY":

        url = "https://www.nseindia.com/api/option-chain-indices?symbol=" + symbol
        expiry = "13-Aug-2020"
    else:
        url = "https://www.nseindia.com/api/option-chain-equities?symbol="+ symbol
        expiry = "27-Aug-2020"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36',
        'accept-encoding': 'gzip, deflate, br', 'accept-language': 'en-US,en;q=0.9'}

    try:
        r = requests.get(url, headers=headers).json()
        ce_values = [data['CE'] for data in r['records']['data'] if
                     "CE" in data and str(data['expiryDate']).lower() == str(expiry).lower()]
        pe_values = [data['PE'] for data in r['records']['data'] if
                     "PE" in data and str(data['expiryDate']).lower() == str(expiry).lower()]
    except:
        oi_filename = os.path.join("data", "{0}.json".format(symbol))
        r = json.loads(open(oi_filename).read())
        ce_values = [data['CE'] for data in r['filtered']['data'] if "CE" in data]
        pe_values = [data['PE'] for data in r['filtered']['data'] if "PE" in data]

    # Method to get option data

    ce_data = pd.DataFrame(ce_values)
    pe_data = pd.DataFrame(pe_values)
    ce_data = ce_data.sort_values(['strikePrice'])
    pe_data = pe_data.sort_values(['strikePrice'])

    # PCR
    pcr = (pe_data['openInterest'].sum() / ce_data['openInterest'].sum()).round(decimals=2)
    pe_oi_change = pe_data['changeinOpenInterest'].sum().round(decimals=2)
    ce_oi_change = ce_data['changeinOpenInterest'].sum().round(decimals=2)
    pcr_change = ( pe_oi_change/ ce_oi_change).round(decimals=2)

    # Dataframe with combined OI data
    df = pd.DataFrame()
    df['CE'] = ce_data['lastPrice']
    df['CE_OI'] = ce_data['openInterest']
    df['CE_OI_Change'] = ce_data['changeinOpenInterest']
    df['strikePrice'] = ce_data['strikePrice']
    df['PE'] = pe_data['lastPrice']
    df['PE_OI'] = pe_data['openInterest']
    df['PE_OI_Change'] = pe_data['changeinOpenInterest']
    df['Nifty'] = pe_data['underlyingValue']

    maxCEOI = df['CE_OI'].max()
    mce = df['strikePrice'].loc[df['CE_OI'] == maxCEOI].iloc[0]
    mcePlus = int((mce * 0.005) + mce)
    mceMinus = int(mce - (mce * 0.005))

    maxPEOI = df['PE_OI'].max()
    mpe = df['strikePrice'].loc[df['PE_OI'] == maxPEOI].iloc[0]
    mpePlus = int((mpe * 0.005) + mpe)
    mpeMinus =int(mpe - (mpe * 0.005))

    #print(mceMinus, mce,mcePlus)
    #print(mpeMinus, mpe, mpePlus)

    if int(df['Nifty'].iloc[0]) in range(mceMinus, mcePlus):
        print("{} is in Range of Sell Zone".format(symbol))
        sellZone[symbol] = [df['Nifty'].iloc[0], 'Sell Zone']
        #print(sellZone)
    elif int(df['Nifty'].iloc[0]) in range(mpeMinus, mpePlus):
        print("{} is in Range of Buy Zone".format(symbol))
        buyZone[symbol] = [df['Nifty'].iloc[0], 'Buy Zone']
        #print(buyZone)
data = pd.read_csv("fno.csv")

while True:
    if(filter1 > 33600 and filter1 < 57600):
        for symbol in data['Symbol']:
            get_df(symbol)

        df2 = pd.DataFrame.from_dict(sellZone, orient='index', columns=[ 'Close', 'Signal' ])
        df3 = pd.DataFrame.from_dict(buyZone, orient='index', columns=['Close', 'Signal'])
        df2 = df2[df2.Close > 200]
        df3 = df3[df3.Close > 200]
        df2.index.name = "Symbol"
        df3.index.name = "Symbol"
        df2.to_csv("data/Sellers.csv")
        df3.to_csv("data/Buyers.csv")
        print(df2)
        print(df3)
        print("--- %s seconds ---" % (time.time() - start_time))
        ctime = datetime.now(timezone('Asia/Kolkata'))
        hour = int(ctime.strftime("%H"))
        minute = int(ctime.strftime("%M"))
        filter1 = (hour * 3600) + (minute * 60)

        time.sleep(60)
    else:
        ctime = datetime.now(timezone('Asia/Kolkata'))
        hour = int(ctime.strftime("%H"))
        minute = int(ctime.strftime("%M"))
        filter1 = (hour * 3600) + (minute * 60)
        print("Markets Closed {}:{}".format(hour,minute))
        time.sleep(60)
