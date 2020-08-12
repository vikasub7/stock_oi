import requests
import pandas as pd
from datetime import datetime
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import AutocompleteInput, ColumnDataSource, Span, SingleIntervalTicker, LinearAxis, PreText, Div, \
    Button, LabelSet, Label, DaysTicker, Range1d, DataTable, TableColumn
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
oi_filename = os.path.join("data", "{0}.json".format(symbol))


class YahooFinance:
    def __init__(self, ticker, result_range='1mo', start=None, end=None, interval='15m', dropna=True):
        """
        Return the stock data of the specified range and interval
        Note - Either Use result_range parameter or use start and end
        Note - Intraday data cannot extend last 60 days
        :param ticker:  Trading Symbol of the stock should correspond with yahoo website
        :param result_range: Valid Ranges "1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"
        :param start: Starting Date
        :param end: End Date
        :param interval:Valid Intervals - Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
        :return:
        """
        if result_range is None:
            start = int(_time.mktime(_time.strptime(start, '%d-%m-%Y')))
            end = int(_time.mktime(_time.strptime(end, '%d-%m-%Y')))
            # defining a params dict for the parameters to be sent to the API
            params = {'period1': start, 'period2': end, 'interval': interval}

        else:
            params = {'range': result_range, 'interval': interval}

        # sending get request and saving the response as response object
        url = "https://query1.finance.yahoo.com/v7/finance/chart/{}".format(ticker)
        r = requests.get(url=url, params=params)
        data = r.json()
        # Getting data from json
        error = data['chart']['error']
        if error:
            raise ValueError(error['description'])
        self._result = self._parsing_json(data)
        if dropna:
            self._result.dropna(inplace=True)

    @property
    def result(self):
        return self._result

    def _parsing_json(self, data):
        timestamps = data['chart']['result'][0]['timestamp']
        # Formatting date from epoch to local time
        timestamps = [_time.strftime('%a, %d %b %Y %H:%M:%S', _time.localtime(x)) for x in timestamps]
        volumes = data['chart']['result'][0]['indicators']['quote'][0]['volume']
        opens = data['chart']['result'][0]['indicators']['quote'][0]['open']
        opens = self._round_of_list(opens)
        closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
        closes = self._round_of_list(closes)
        lows = data['chart']['result'][0]['indicators']['quote'][0]['low']
        lows = self._round_of_list(lows)
        highs = data['chart']['result'][0]['indicators']['quote'][0]['high']
        highs = self._round_of_list(highs)
        df_dict = {'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes}
        df = pd.DataFrame(df_dict, index=timestamps)
        df.index = pd.to_datetime(df.index)
        return df

    def _round_of_list(self, xlist):
        temp_list = []
        for x in xlist:
            if isinstance(x, float):
                temp_list.append(round(x, 2))
            else:
                temp_list.append(pd.np.nan)
        return temp_list

    def to_csv(self, file_name):
        self.result.to_csv(file_name)


# CPR Calculation
def get_pivot(symbol):
    global sentiment, PTC, PBC, TC, BC, cpr_width
    if symbol == "NIFTY":
        df7 = YahooFinance("^NSEI", result_range='3d', interval='1d', dropna='True').result

    elif (symbol == "BANKNIFTY"):
        df7 = YahooFinance("^NSEBANK", result_range='3d', interval='1d', dropna='True').result

    else:
        df7 = YahooFinance(symbol + ".NS", result_range='3d', interval='1d', dropna='True').result

    pivot = (df7['Low'].iloc[1] + df7['High'].iloc[1] + df7['Close'].iloc[1]) / 3
    ppivot = (df7['Low'].iloc[0] + df7['High'].iloc[0] + df7['Close'].iloc[0]) / 3

    change = (df7['Close'].iloc[1] - df7['Close'].iloc[0]) / df7['Close'].iloc[0] * 100

    pbc = (df7['Low'].iloc[0] + df7['High'].iloc[0]) / 2
    ptc = (ppivot - pbc) + ppivot
    PTC = max(pbc, ptc)
    PBC = min(pbc, ptc)

    bc = (df7['Low'].iloc[1] + df7['High'].iloc[1]) / 2
    tc = (pivot - bc) + pivot
    TC = max(bc, tc).__round__(2)
    BC = min(bc, tc).__round__(2)

    s1 = 2 * pivot - df7['High'].iloc[1]
    r1 = 2 * pivot - df7['Low'].iloc[1]
    s2 = pivot - (df7['High'].iloc[1] - df7['Low'].iloc[1])
    r2 = pivot + (df7['High'].iloc[1] - df7['Low'].iloc[1])

    phigh = df7['High'].iloc[1]
    plow = df7['Low'].iloc[1]
    cpr_width = (((TC - BC) / pivot) * 100).__round__(2)
    sentiment = 'Neutral'


def get_df(symbol):
    global headers, pcr, pcr_change, pe_oi_change, ce_oi_change, max_oi_change, max_oi, minValue, maxValue, minRange, maxRange, mp, lastPrice, df, width, min_oi_change, min_oi
    # URL and Headers for getting option data
    if symbol == "NIFTY" or symbol == "BANKNIFTY":

        url = "https://www.nseindia.com/api/option-chain-indices?symbol=" + symbol
        expiry = "13-Aug-2020"
    else:
        url = "https://www.nseindia.com/api/option-chain-equities?symbol=" + symbol
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
    pcr_change = (pe_oi_change / ce_oi_change).round(decimals=2)

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

    df = df.dropna()

    # MaxPain Calculation
    df['CE_SUM'] = df['CE_OI'].cumsum()
    df['CE_SUM'] = df['CE_SUM'] * df['strikePrice']
    df['CE_SP'] = df['strikePrice'] * df['CE_OI']
    df['CE_SP'] = df['CE_SP'].cumsum()
    df['CE_Value'] = df['CE_SUM'] - df['CE_SP']
    df['PE_SP'] = df['strikePrice'] * df['PE_OI']

    import numpy as np
    array = df['PE_OI'].to_numpy()
    array = np.flip(array)  # to flip the order
    array = np.cumsum(array)
    array = np.flip(array)  # to flip back to original order
    df['PE_SUM'] = array
    df['PE_SUM'] = df['PE_SUM'] * df['strikePrice']

    array = df['PE_SP'].to_numpy()
    array = np.flip(array)  # to flip the order
    array = np.cumsum(array)
    array = np.flip(array)
    df['PE_SP'] = array  # to flip back to original order
    df['PE_Value'] = df['PE_SP'] - df['PE_SUM']
    df['Total'] = df['CE_Value'] + df['PE_Value']
    max_oi_change = max(df['CE_OI_Change'].max(), df['PE_OI_Change'].max())
    min_oi_change = min(df['CE_OI_Change'].max(), df['PE_OI_Change'].min())
    max_oi = max(df['CE_OI'].max(), df['PE_OI'].max())
    min_oi = min(df['CE_OI'].max(), df['PE_OI'].min())
    minRange = (df['Nifty'].iloc[1] - (df['Nifty'].iloc[1] * 0.08)).round(decimals=0)

    maxRange = ((df['Nifty'].iloc[1] * 0.08) + df['Nifty'].iloc[1]).round(decimals=0)
    # print(minRange, maxRange)
    maxpain = df['Total'].min()
    mp = df['strikePrice'].loc[df['Total'] == maxpain].iloc[0]
    df['Total'] = df['Total'] / 1000000

    minValue = df['Total'].min()
    maxValue = df['Total'].max()
    lastPrice = df['Nifty'].iloc[1]
    width = ((df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0]) / 3)
    # print(width)
    if symbol == "NIFTY" or symbol == "BANKNIFTY":
        width = ((df['strikePrice'].iloc[1] - df['strikePrice'].iloc[0]) / 2)
        minRange = (df['Nifty'].iloc[1] - (df['Nifty'].iloc[1] * 0.05)).round(decimals=0)
        maxRange = ((df['Nifty'].iloc[1] * 0.05) + df['Nifty'].iloc[1]).round(decimals=0)
    return df


def get_oi_data(symbol):
    end = date.today()
    global nifty_fut
    if symbol == "NIFTY" or symbol == "BANKNIFTY":
        nifty_fut = get_history(symbol=symbol,
                                start=date(2020, 8, 1),
                                end=date(end.year, end.month, end.day),
                                index=True,
                                futures=True,
                                expiry_date=date(2020, 8, 27))
        nifty_fut['Open Interest'] = nifty_fut['Open Interest'] / 1000
        nifty_fut['tips'] = nifty_fut['Open Interest'] / 1000
        nifty_fut['tips2'] = nifty_fut['Settle Price']
        nifty_fut['Date2'] = nifty_fut.index
        nifty_fut['Axis'] = nifty_fut.index.to_list()
        nifty_fut['Axis'] = pd.to_datetime(nifty_fut['Axis'])
        nifty_fut['Axis'] = nifty_fut['Axis'].dt.strftime('%b %d')
        # print(nifty_fut.ioc[-1])
    else:
        nifty_fut = get_history(symbol=symbol,
                                start=date(2020, 8, 1),
                                end=date(end.year, end.month, end.day),
                                index=False,
                                futures=True,
                                expiry_date=date(2020, 8, 27))
        nifty_fut['Open Interest'] = nifty_fut['Open Interest'] / 1000
        nifty_fut['tips'] = nifty_fut['Open Interest'] / 1000
        nifty_fut['tips2'] = nifty_fut['Settle Price']
        nifty_fut['Date2'] = nifty_fut.index
        nifty_fut['Axis'] = nifty_fut.index.to_list()
        nifty_fut['Axis'] = pd.to_datetime(nifty_fut['Axis'])
        nifty_fut['Axis'] = nifty_fut['Axis'].dt.strftime('%b %d')
        # print(nifty_fut.iloc[-1])
    return nifty_fut


def plot_des():
    global p, p2, p3, p4, source, source2, pv1, pv2, p2v1, p2v2, ltp, ltp2, my_label, my_label2, ltp4, my_label3, pcrDiv, pcrCDiv, p4v1, p3l1, p3a1
    source = ColumnDataSource(df)
    source2 = ColumnDataSource(nifty_fut)
    # Change in OI Plot
    p = figure(name='coi', x_range=(minRange, maxRange), plot_width=650, plot_height=350, x_minor_ticks=2,
               tools="xpan,hover,reset", tooltips="@strikePrice CE: @CE_OI_Change <br> @strikePrice PE: @PE_OI_Change",
               toolbar_location="above")
    pv1 = p.vbar(x=dodge('strikePrice', -width / 2, range=p.x_range), top='CE_OI_Change', source=source,
                 color="#E74C3C", width=width,
                 fill_alpha=0.5)
    pv2 = p.vbar(x=dodge('strikePrice', width / 2, range=p.x_range), top='PE_OI_Change', source=source, color="#1E90FF",
                 width=width,
                 fill_alpha=0.5)
    p.background_fill_color = "#0b0c11"
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.visible = False
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.axis_line_color = "#C6C6CF"
    p.xaxis.axis_label_text_color = "#C6C6CF"
    p.xaxis.major_label_text_color = "#C6C6CF"
    p.xaxis.major_tick_line_color = "#C6C6CF"
    p.xaxis.minor_tick_line_color = "#C6C6CF"
    # p.yaxis.major_label_text_color = None
    nifty = df['Nifty'].iloc[1]
    ltp = Span(location=nifty, dimension='height', line_color='green', line_dash='dashed', line_width=3)
    p.add_layout(ltp)
    my_label = Label(x=nifty, y=max_oi_change / 2, text=str(nifty), border_line_color='green', border_line_alpha=1.0,
                     text_color='white', text_font_size='10pt',
                     background_fill_color='green', background_fill_alpha=0.5)
    p.add_layout(my_label)
    p.toolbar.autohide = True
    labels = LabelSet(x='strikePrice', y='PE_OI_Change', text='PE_OI_Change', level='glyph',
                      x_offset=5, y_offset=-5, source=source, render_mode='canvas')
    p.sizing_mode = "stretch_width"
    p.border_fill_color = '#0b0c11'
    p.outline_line_color = '#0b0c11'

    # Total OI Plot
    p2 = figure(name='oi', x_range=(minRange, maxRange), plot_width=650, plot_height=350, x_minor_ticks=2,
                tools="xpan,hover,reset", tooltips="@strikePrice CE: @CE_OI <br> @strikePrice PE: @PE_OI",
                toolbar_location="above")
    p2v1 = p2.vbar(x=dodge('strikePrice', -width / 2, range=p2.x_range), top='CE_OI', source=source, color="#E74C3C",
                   width=width,
                   fill_alpha=0.5)
    p2v2 = p2.vbar(x=dodge('strikePrice', width / 2, range=p2.x_range), top='PE_OI', source=source, color="#1E90FF",
                   width=width,
                   fill_alpha=0.5)
    p2.background_fill_color = "#0b0c11"
    p.border_fill_color = '#0b0c11'
    p2.xgrid.grid_line_color = None
    p2.ygrid.grid_line_color = None
    nifty = df['Nifty'].iloc[1]
    p2.yaxis.visible = False
    p2.yaxis.major_tick_line_color = None
    p2.yaxis.minor_tick_line_color = None
    ltp2 = Span(location=nifty, dimension='height', line_color='green', line_dash='dashed', line_width=3)
    p2.add_layout(ltp)
    my_label2 = Label(x=nifty, y=max_oi / 2, text=str(nifty), border_line_color='green', border_line_alpha=1.0,
                      text_color='white', text_font_size='10pt',
                      background_fill_color='green', background_fill_alpha=0.5)
    p2.add_layout(my_label2)
    p2.toolbar.autohide = True
    p2.sizing_mode = "stretch_width"
    p2.xaxis.axis_line_color = "#C6C6CF"
    p2.xaxis.axis_label_text_color = "#C6C6CF"
    p2.xaxis.major_label_text_color = "#C6C6CF"
    p2.xaxis.major_tick_line_color = "#C6C6CF"
    p2.xaxis.minor_tick_line_color = "#C6C6CF"
    p2.border_fill_color = '#0b0c11'
    p2.outline_line_color = '#0b0c11'

    p3 = figure(name='nb', y_range=(0, int(nifty_fut['Open Interest'].max())), plot_width=650, plot_height=350,
                y_minor_ticks=2, x_range=list(nifty_fut['Axis']),
                tools="hover,reset", tooltips=" @Symbol: @Close{1.11} <br> Open Interest: @tips M ",
                toolbar_location="above")

    p3.vbar(x='Axis', top='Open Interest', bottom=0, source=source2, color="#0749F3", width=0.5, fill_alpha=0.5)
    p3.background_fill_color = "#0b0c11"
    p3.xgrid.grid_line_color = None
    p3.ygrid.grid_line_color = None
    p3.yaxis.visible = False
    p3.yaxis.major_tick_line_color = None
    p3.yaxis.minor_tick_line_color = None
    p3.extra_y_ranges = {"y2": Range1d(start=int(nifty_fut['Close'].min() - (nifty_fut['Close'].min() * 0.02)),
                                       end=int(nifty_fut['Close'].max() + (nifty_fut['Close'].max() * 0.02)))}
    p3.add_layout(LinearAxis(y_range_name="y2"), 'right')
    p3l1 = p3.line(x='Axis', y='Close', source=source2, color="#EBECF1", width=2, y_range_name="y2")
    p3a1 = p3.annulus(x='Axis', y='Close', source=source2, color="#1E90FF", inner_radius=0.02, outer_radius=0.03,
                      alpha=0.6, y_range_name="y2")
    # p3.xaxis.ticker = DaysTicker(days=list(range(1, 32)))
    # p3.xaxis.formatter = DatetimeTickFormatter(days="%d-%b")
    p3.sizing_mode = "stretch_width"
    p3.toolbar.autohide = True
    p3.xaxis.axis_line_color = "#C6C6CF"
    p3.xaxis.axis_label_text_color = "#C6C6CF"
    p3.xaxis.major_label_text_color = "#C6C6CF"
    p3.xaxis.major_tick_line_color = "#C6C6CF"
    p3.xaxis.minor_tick_line_color = "#C6C6CF"
    p3.border_fill_color = '#0b0c11'
    p3.outline_line_color = '#0b0c11'

    p4 = figure(name='mpc', x_range=(df['strikePrice'].iloc[0], df['strikePrice'].iloc[-1]),
                y_range=(minValue, maxValue), plot_width=650, plot_height=350, x_minor_ticks=2, tools="reset",
                toolbar_location="above")
    p4v1 = p4.vbar(x=dodge('strikePrice', 0, range=p4.x_range), top='Total', source=source, color="#1E90FF",
                   width=width,
                   fill_alpha=0.5)
    p4.background_fill_color = "#0b0c11"
    p4.xgrid.grid_line_color = None
    p4.ygrid.grid_line_color = None
    nifty = df['Nifty'].iloc[1]
    p4.yaxis.visible = False
    p4.yaxis.major_tick_line_color = None
    p4.yaxis.minor_tick_line_color = None
    ltp4 = Span(location=mp, dimension='height', line_color='#FFA726', line_dash='dashed', line_width=3)
    p4.add_layout(ltp4)
    my_label3 = Label(x=nifty, y=(maxValue + minValue) / 2, text=str(mp), border_line_color='#2C3E50',
                      border_line_alpha=1.0, y_offset=3,
                      text_color='white', text_font_size='12pt',
                      background_fill_color='#FFA726', background_fill_alpha=0.5)
    p4.add_layout(my_label3)
    p4.toolbar.autohide = True
    p4.sizing_mode = "stretch_width"
    p4.xaxis.axis_line_color = "#C6C6CF"
    p4.xaxis.axis_label_text_color = "#C6C6CF"
    p4.xaxis.major_label_text_color = "#C6C6CF"
    p4.xaxis.major_tick_line_color = "#C6C6CF"
    p4.xaxis.minor_tick_line_color = "#C6C6CF"
    p4.border_fill_color = '#0b0c11'
    p4.outline_line_color = '#0b0c11'

    Buyers = pd.read_csv("stock_oi/data/Buyers.csv")
    Sellers = pd.read_csv("stock_oi/data/Sellers.csv")

    # print(Buyers)
    # print(Sellers)
    global sourceTable, data_table
    sourceTable = ColumnDataSource(Buyers)
    columns = [
        TableColumn(field="Symbol", title="Symbol"),
        TableColumn(field="Close", title="Close"),
        TableColumn(field="Signal", title="Signal"),
    ]
    data_table = DataTable(name='signals', source=sourceTable, columns=columns, width=600, height=280,
                           background="#0b0c11")
    data_table.sizing_mode = "stretch_width"
    data_table.sizing_mode = "stretch_height"

    sourceTable.selected.on_change('indices', update_2)

    global sourceTable2, data_table2
    sourceTable2 = ColumnDataSource(Sellers)
    data_table2 = DataTable(name='signals2', source=sourceTable2, columns=columns, width=600, height=280,
                            background="#0b0c11")
    data_table2.sizing_mode = "stretch_width"
    data_table2.sizing_mode = "stretch_height"

    sourceTable2.selected.on_change('indices', update_3)

    # Buttons
    try:
        ctime = datetime.now(timezone('Asia/Kolkata'))
        hour = (ctime.strftime("%H"))
        minute = (ctime.strftime("%M"))
        sec = (ctime.strftime("%S"))
    except:
        hour = "0"
        minute = "0"
        sec = "0"
    webTime = 'Last Updated: ' + hour + ':' + minute + ':' + sec
    global pre
    pre = Div(name='time', text="""<font size="1"  face ="Nunito">{a}</font>""".format(a=webTime))

    global pcrDiv
    pcrDiv = Div(name='pcr',
                 text="""<font size="3" color="#0DB96F" face ="Nunito" >{a}</font>""".format(a=pcr))  # E74C3C
    if pcr > 1:
        pcrDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr)
    else:
        pcrDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr)

    pcrCDiv = Div(name='pcrChange',
                  text="""<font size="3" color="#0DB96F" face ="Nunito">{a}</font>""".format(a=pcr_change))  # E74C3C
    if pe_oi_change > ce_oi_change or ce_oi_change < 0:
        pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
    else:
        pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

    global sentimentDiv
    sentimentDiv = Div(name='sentiment', text="""<font size="3" color="#0DB96F" face ="Nunito">{a}</font>""".format(
        a=sentiment))  # E74C3C
    if TC < PBC:
        sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Very Bearish</font>"""  # 037645
    elif BC > PTC:
        sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Very Bullish</font>"""
    elif TC < PTC and PBC < BC:
        sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Inside Day</font>"""
    elif TC > PTC and PBC > BC:
        sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Outside Day</font>"""
    elif TC > PTC and PBC < BC:
        sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Bullish</font>"""
    elif TC < PTC and PBC > BC:
        sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Bearish</font>"""

    close1 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 1]
    close2 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 2]
    op1 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 1]
    op2 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 2]

    global dayDiv
    dayDiv = Div(name='dayType',
                 text="""<font size="3" color="#0DB96F" face ="Nunito">{a}</font>""".format(a=sentiment))  # E74C3C
    if (close1 > close2) and (op1 > op2):
        dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Long Buildup</font>"""  # 037645
    elif (close1 < close2) and (op1 > op2):
        dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Short Buildup</font>"""
    elif (close1 > close2) and (op1 < op2):
        dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Short Covering</font>"""
    elif (close1 < close2) and (op1 < op2):
        dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Profit Booking</font>"""

    data = pd.read_csv("stock_oi/data/fno.csv")
    menu_item = data['Symbol'].to_list()

    global input_box
    input_box = AutocompleteInput(name='search')
    input_box.value = "NIFTY"
    input_box.max_width = 200
    input_box.sizing_mode = "stretch_width"
    input_box.completions = menu_item
    input_box.case_sensitive = False
    input_box.on_change('value', lambda attr, old, new: update())


def update_2(attr, old, new):
    global symbol
    print("Selected")
    try:
        selected_index = sourceTable.selected.indices[0]
        symbol = str(sourceTable.data["Symbol"][selected_index])
        input_box.value = symbol
        get_pivot(symbol)
        source.data = get_df(symbol)
        source2.data = get_oi_data(symbol)

        p.x_range.start = minRange
        p.x_range.end = maxRange
        p.y_range.start = min_oi_change
        p.y_range.end = max_oi_change
        pv1.glyph.width = width
        pv1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
        pv2.glyph.width = width
        pv2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
        ltp.location = df['Nifty'].iloc[1]
        my_label.x = df['Nifty'].iloc[1]
        my_label.y = max_oi_change / 2
        my_label.text = str(df['Nifty'].iloc[1])

        p2.x_range.start = minRange
        p2.x_range.end = maxRange
        p2.y_range.start = min_oi
        p2.y_range.end = max_oi
        p2v1.glyph.width = width
        p2v1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
        p2v2.glyph.width = width
        p2v2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
        ltp2.location = df['Nifty'].iloc[1]
        my_label2.x = df['Nifty'].iloc[1]
        my_label2.y = max_oi / 2
        my_label2.text = str(df['Nifty'].iloc[1])

        p3.y_range.start = 0
        p3.y_range.end = int(nifty_fut['Open Interest'].max())
        p3.extra_y_ranges['y2'].start = int(nifty_fut['Close'].min() - (nifty_fut['Close'].min() * 0.02))
        p3.extra_y_ranges['y2'].end = int(nifty_fut['Close'].max() + (nifty_fut['Close'].max() * 0.02))
        p3l1.y_range_name = "y2"
        p3a1.y_range_name = "y2"

        p4.x_range.start = df['strikePrice'].iloc[0]
        p4.x_range.end = df['strikePrice'].iloc[-1]
        p4.y_range.start = minValue
        p4.y_range.end = maxValue
        p4v1.glyph.width = width
        ltp4.location = mp
        my_label3.x = mp
        my_label3.y = (maxValue + minValue) / 2
        my_label3.text = str(mp)

        # pcr div
        if pcr > 1:
            pcrDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr)
        else:
            pcrDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr)

        # pcrCDiv
        if pe_oi_change > ce_oi_change or ce_oi_change < 0:
            pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
        else:
            pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

        if pe_oi_change > ce_oi_change or ce_oi_change < 0:
            pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
        else:
            pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

        # Sentiment Div
        if TC < PBC:
            sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Very Bearish</font>"""  # 037645
        elif BC > PTC:
            sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Very Bullish</font>"""
        elif TC < PTC and PBC < BC:
            sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Inside Day</font>"""
        elif TC > PTC and PBC > BC:
            sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Outside Day</font>"""
        elif TC > PTC and PBC < BC:
            sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Bullish</font>"""
        elif TC < PTC and PBC > BC:
            sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Bearish</font>"""

        close1 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 1]
        close2 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 2]
        op1 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 1]
        op2 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 2]
        if (close1 > close2) and (op1 > op2):
            dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Long Buildup</font>"""  # 037645
        elif (close1 < close2) and (op1 > op2):
            dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Short Buildup</font>"""
        elif (close1 > close2) and (op1 < op2):
            dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Short Covering</font>"""
        elif (close1 < close2) and (op1 < op2):
            dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Profit Booking</font>"""
    except Exception as e:
        print(e)
        pass


def update_3(attr, old, new):
    global symbol
    print("Selected")
    try:
        selected_index = sourceTable2.selected.indices[0]
        symbol = str(sourceTable2.data["Symbol"][selected_index])
        input_box.value = symbol
        get_pivot(symbol)
        source.data = get_df(symbol)
        source2.data = get_oi_data(symbol)

        p.x_range.start = minRange
        p.x_range.end = maxRange
        p.y_range.start = min_oi_change
        p.y_range.end = max_oi_change
        pv1.glyph.width = width
        pv1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
        pv2.glyph.width = width
        pv2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
        ltp.location = df['Nifty'].iloc[1]
        my_label.x = df['Nifty'].iloc[1]
        my_label.y = max_oi_change / 2
        my_label.text = str(df['Nifty'].iloc[1])

        p2.x_range.start = minRange
        p2.x_range.end = maxRange
        p2.y_range.start = min_oi
        p2.y_range.end = max_oi
        p2v1.glyph.width = width
        p2v1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
        p2v2.glyph.width = width
        p2v2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
        ltp2.location = df['Nifty'].iloc[1]
        my_label2.x = df['Nifty'].iloc[1]
        my_label2.y = max_oi / 2
        my_label2.text = str(df['Nifty'].iloc[1])

        p3.y_range.start = 0
        p3.y_range.end = int(nifty_fut['Open Interest'].max())
        p3.extra_y_ranges['y2'].start = int(nifty_fut['Close'].min() - (nifty_fut['Close'].min() * 0.02))
        p3.extra_y_ranges['y2'].end = int(nifty_fut['Close'].max() + (nifty_fut['Close'].max() * 0.02))
        p3l1.y_range_name = "y2"
        p3a1.y_range_name = "y2"

        p4.x_range.start = df['strikePrice'].iloc[0]
        p4.x_range.end = df['strikePrice'].iloc[-1]
        p4.y_range.start = minValue
        p4.y_range.end = maxValue
        p4v1.glyph.width = width
        ltp4.location = mp
        my_label3.x = mp
        my_label3.y = (maxValue + minValue) / 2
        my_label3.text = str(mp)

        # pcr div
        if pcr > 1:
            pcrDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr)
        else:
            pcrDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr)

        # pcrCDiv
        if pe_oi_change > ce_oi_change or ce_oi_change < 0:
            pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
        else:
            pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

        if pe_oi_change > ce_oi_change or ce_oi_change < 0:
            pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
        else:
            pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

        # Sentiment Div
        if TC < PBC:
            sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Very Bearish</font>"""  # 037645
        elif BC > PTC:
            sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Very Bullish</font>"""
        elif TC < PTC and PBC < BC:
            sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Inside Day</font>"""
        elif TC > PTC and PBC > BC:
            sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Outside Day</font>"""
        elif TC > PTC and PBC < BC:
            sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Bullish</font>"""
        elif TC < PTC and PBC > BC:
            sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Bearish</font>"""

        close1 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 1]
        close2 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 2]
        op1 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 1]
        op2 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 2]
        if (close1 > close2) and (op1 > op2):
            dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Long Buildup</font>"""  # 037645
        elif (close1 < close2) and (op1 > op2):
            dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Short Buildup</font>"""
        elif (close1 > close2) and (op1 < op2):
            dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Short Covering</font>"""
        elif (close1 < close2) and (op1 < op2):
            dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Profit Booking</font>"""
    except Exception as e:
        print(e)
        pass


def update():
    global symbol
    # print(input_box.value)
    symbol = input_box.value
    get_pivot(symbol)
    source.data = get_df(symbol)
    source2.data = get_oi_data(symbol)

    p.x_range.start = minRange
    p.x_range.end = maxRange
    p.y_range.start = min_oi_change
    p.y_range.end = max_oi_change
    pv1.glyph.width = width
    pv1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
    pv2.glyph.width = width
    pv2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
    ltp.location = df['Nifty'].iloc[1]
    my_label.x = df['Nifty'].iloc[1]
    my_label.y = max_oi_change / 2
    my_label.text = str(df['Nifty'].iloc[1])

    p2.x_range.start = minRange
    p2.x_range.end = maxRange
    p2.y_range.start = min_oi
    p2.y_range.end = max_oi
    p2v1.glyph.width = width
    p2v1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
    p2v2.glyph.width = width
    p2v2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
    ltp2.location = df['Nifty'].iloc[1]
    my_label2.x = df['Nifty'].iloc[1]
    my_label2.y = max_oi / 2
    my_label2.text = str(df['Nifty'].iloc[1])

    p3.y_range.start = 0
    p3.y_range.end = int(nifty_fut['Open Interest'].max())
    p3.extra_y_ranges['y2'].start = int(nifty_fut['Close'].min() - (nifty_fut['Close'].min() * 0.02))
    p3.extra_y_ranges['y2'].end = int(nifty_fut['Close'].max() + (nifty_fut['Close'].max() * 0.02))
    p3l1.y_range_name = "y2"
    p3a1.y_range_name = "y2"

    p4.x_range.start = df['strikePrice'].iloc[0]
    p4.x_range.end = df['strikePrice'].iloc[-1]
    p4.y_range.start = minValue
    p4.y_range.end = maxValue
    p4v1.glyph.width = width
    ltp4.location = mp
    my_label3.x = mp
    my_label3.y = (maxValue + minValue) / 2
    my_label3.text = str(mp)

    # pcr div
    if pcr > 1:
        pcrDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr)
    else:
        pcrDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr)

    # pcrCDiv
    if pe_oi_change > ce_oi_change or ce_oi_change < 0:
        pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
    else:
        pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

    if pe_oi_change > ce_oi_change or ce_oi_change < 0:
        pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
    else:
        pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

    # Sentiment Div
    if TC < PBC:
        sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Very Bearish</font>"""  # 037645
    elif BC > PTC:
        sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Very Bullish</font>"""
    elif TC < PTC and PBC < BC:
        sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Inside Day</font>"""
    elif TC > PTC and PBC > BC:
        sentimentDiv.text = """<font size="3" color="#E1BC15" face ="Nunito">Outside Day</font>"""
    elif TC > PTC and PBC < BC:
        sentimentDiv.text = """<font size="3" color="#037645" face ="Nunito">Bullish</font>"""
    elif TC < PTC and PBC > BC:
        sentimentDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Bearish</font>"""

    close1 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 1]
    close2 = nifty_fut['Close'].iloc[len(nifty_fut['Close']) - 2]
    op1 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 1]
    op2 = nifty_fut['Open Interest'].iloc[len(nifty_fut['Open Interest']) - 2]
    if (close1 > close2) and (op1 > op2):
        dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Long Buildup</font>"""  # 037645
    elif (close1 < close2) and (op1 > op2):
        dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Short Buildup</font>"""
    elif (close1 > close2) and (op1 < op2):
        dayDiv.text = """<font size="3" color="#037645" face ="Nunito">Short Covering</font>"""
    elif (close1 < close2) and (op1 < op2):
        dayDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">Profit Booking</font>"""


def callback():
    try:
        ctime = datetime.now(timezone('Asia/Kolkata'))
        hour = (ctime.strftime("%H"))
        minute = (ctime.strftime("%M"))
        sec = (ctime.strftime("%S"))
        filter1 = (hour * 3600) + (minute * 60)
        webTime = 'Last Updated:' + hour + ':' + minute + ':' + sec
        pre.text = """<font size="1"  face ="Nunito">{a}</font>""".format(a=webTime)

        Buyers = pd.read_csv("stock_oi/data/Buyers.csv")
        Sellers = pd.read_csv("stock_oi/data/Sellers.csv")

        source.data = get_df(symbol)
        source2.data = get_oi_data(symbol)
        sourceTable.data = Buyers
        sourceTable2.data = Sellers

        if pcr > 1:
            pcrDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr)
        else:
            pcrDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr)

        if pe_oi_change > ce_oi_change or ce_oi_change < 0:
            pcrCDiv.text = """<font size="3" color="#037645" face ="Nunito">{a}</font>""".format(a=pcr_change)
        else:
            pcrCDiv.text = """<font size="3" color="#E74C3C" face ="Nunito">{a}</font>""".format(a=pcr_change)

        # p.x_range.start = minRange
        # p.x_range.end = maxRange
        # p.y_range.start = min_oi_change
        # p.y_range.end = max_oi_change
        pv1.glyph.width = width
        pv1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
        pv2.glyph.width = width
        pv2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
        ltp.location = df['Nifty'].iloc[1]
        my_label.x = df['Nifty'].iloc[1]
        my_label.y = max_oi_change / 2
        my_label.text = str(df['Nifty'].iloc[1])

        # p2.x_range.start = minRange
        # p2.x_range.end = maxRange
        # p2.y_range.start = min_oi
        # p2.y_range.end = max_oi
        p2v1.glyph.width = width
        p2v1.glyph.x = dodge('strikePrice', -width / 2, range=p.x_range)
        p2v2.glyph.width = width
        p2v2.glyph.x = dodge('strikePrice', width / 2, range=p.x_range)
        ltp2.location = df['Nifty'].iloc[1]
        my_label2.x = df['Nifty'].iloc[1]
        my_label2.y = max_oi / 2
        my_label2.text = str(df['Nifty'].iloc[1])

        # p3.y_range.start = 0
        # p3.y_range.end = int(nifty_fut['Open Interest'].max())
        p3.extra_y_ranges['y2'].start = int(nifty_fut['Close'].min() - (nifty_fut['Close'].min() * 0.02))
        p3.extra_y_ranges['y2'].end = int(nifty_fut['Close'].max() + (nifty_fut['Close'].max() * 0.02))
        p3l1.y_range_name = "y2"
        p3a1.y_range_name = "y2"

        # p4.x_range.start = df['strikePrice'].iloc[0]
        # p4.x_range.end = df['strikePrice'].iloc[-1]
        # p4.y_range.start = minValue
        # p4.y_range.end = maxValue
        ltp4.location = mp
        my_label3.x = mp
        my_label3.y = (maxValue + minValue) / 2
        my_label3.text = str(mp)

    except:
        pass

get_pivot(symbol)
get_df(symbol)
get_oi_data(symbol)
plot_des()

curdoc().add_root(data_table)
curdoc().add_root(data_table2)
curdoc().add_root(input_box)
curdoc().add_root(p)
curdoc().add_root(p2)
curdoc().add_root(p3)
curdoc().add_root(p4)
curdoc().add_root(pcrDiv)
curdoc().add_root(pcrCDiv)
curdoc().add_root(sentimentDiv)
curdoc().add_root(dayDiv)
curdoc().add_root(pre)
curdoc().title = "VA Trade"
curdoc().add_periodic_callback(callback, 10000)
