import time  # to simulate a real time data, time loop


import streamlit as st  # 🎈 data web app development
import os
from screener import plot_coint



import datetime

import pandas as pd
import requests
import json

from binance.um_futures import UMFutures



st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

@st.experimental_memo
def exchange_info():
    client = UMFutures()
    exchange_info = client.exchange_info()
#     exchange_info = requests.get('https://api.binance.com/fapi/v1/exchangeInfo').json()
    symbols = exchange_info["symbols"]

    return symbols

symbols = exchange_info()
root_url = 'https://api.binance.com/api/v1/klines'

p = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

price = []
for i in p:
    price.append(float(requests.get('https://api.binance.com/api/v3/ticker/price?symbol=' + i).json()['price']))

st.session_state.prev_price = price


# if there is no specification of limit in the url, then default the recent 500 entries
# read csv from a URL

def get_bars(symbol, interval='1m'):
    url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&limit='  + str(1000)
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [datetime.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return df

# @st.cache(allow_output_mutation=True)
def combine(interval, rank = 10):
    perps = []
    for symbol in symbols:
        if symbol['contractType'] == 'PERPETUAL' and symbol['quoteAsset'] == 'USDT' and symbol["status"] == "TRADING":
            perps.append(symbol['symbol'])

    data = {}
    volume = {}
    for perp in perps:
        try:
            data[perp] = get_bars(perp, interval)
            volume[perp] = float(data[perp].iloc[-1, :].volume)
            # print(volume[perp])

        except:
            pass

    selected = sorted(volume, key=volume.get, reverse=True)[:rank]

    close = {}
    for p in selected:
        close[p] = data[p].close.astype(float)

    close_df = pd.DataFrame(close)

    return close_df




# dashboard title
st.title("Real-Time / Live Crypto Dashboard")

# top-level filters
_rank = st.selectbox("Select the Volume Rank", range(10, 143))

# creating a single-element container
placeholder = st.empty()
place = st.empty()



def price():
    p = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

    price = []
    for i in p:
        price.append(float(requests.get('https://api.binance.com/api/v3/ticker/price?symbol=' + i).json()['price']))


        # create three columns
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="BTCUSDT",
            value=price[0],
            delta=str(round((price[0] -
                             st.session_state.prev_price[0])*100/
                            st.session_state.prev_price[0], 2)) + '%',
        )

        kpi2.metric(
            label="ETHUSDT",
            value=price[1],
            delta=str(round((price[1] -
                             st.session_state.prev_price[1])*100/
                            st.session_state.prev_price[1], 2)) + '%',
        )

        kpi3.metric(
            label="ADAUSDT",
            value=price[2],
            delta=str(round((price[2] -
                             st.session_state.prev_price[2])*100/
                            st.session_state.prev_price[2], 2)) + '%',
        )

        st.session_state.prev_price = price

        st.markdown('Updated {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def refresh(rank):
    df1 = combine('1m', rank)
    df5 = combine('5m', rank)
    with place.container():
        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            st.markdown("### 1 Minute")
            fig = plot_coint(df1, '1m')
            st.markdown('Updated {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            st.write(fig)

        with fig_col2:
            st.markdown("### 5 Minute")
            fig2 = plot_coint(df5, '5m')
            st.markdown('Updated {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            # st.markdown("###### Updated {}".format(datetime.datetime.now()))
            st.write(fig2)

        # st.markdown("### Detailed Data View")
        # st.dataframe(df)
        # time.sleep(1)




while True:
    price()
    refresh(int(_rank))
    # st.markdown("###### Updated {}".format(datetime.datetime.now()))
