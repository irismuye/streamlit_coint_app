import time  # to simulate a real time data, time loop
import streamlit as st  # 🎈 data web app development
import os
from screener import plot_coint
import datetime

import pandas as pd
import requests
import json
import webbrowser

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

    perps = []
    for symbol in symbols:
        if symbol['contractType'] == 'PERPETUAL' and symbol['quoteAsset'] == 'USDT' and symbol["status"] == "TRADING":
            perps.append(symbol['symbol'])

    return perps


perps = exchange_info()
global df1, df5

st.session_state.load = True
root_url = 'https://api.binance.com/api/v1/klines'
time_url = "https://api.binance.com/api/v3/time"

url = 'https://api.binance.com/api/v3/ticker?symbols=%5B%22BTCUSDT%22,%22ETHUSDT%22,%22ADAUSDT%22%5D&windowSize=1m'


# if there is no specification of limit in the url, then default the recent 500 entries
# read csv from a URL

def get_bars(symbol, interval='1m'):
    url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&limit=' + str(1000)
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [datetime.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return df


# @st.cache(allow_output_mutation=True)
def combine(interval, rank=10):

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

    return close_df, selected


# dashboard title
st.title("Real-Time / Live Crypto Dashboard")
# top-level filters



# creating a single-element container
placeholder = st.empty()
place = st.empty()


st.session_state.run = 0

def price():
    price = requests.get(url).json()

    # create three columns
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="BTCUSDT",
            value=price[0]['lastPrice'][:-6],
            delta=str(price[0]['priceChangePercent']) + '%',
        )

        kpi2.metric(
            label="ETHUSDT",
            value=price[1]['lastPrice'][:-6],
            delta=str(price[1]['priceChangePercent']) + '%',
        )

        kpi3.metric(
            label="ADAUSDT",
            value=price[2]['lastPrice'][:-4],
            delta=str(price[2]['priceChangePercent']) + '%',
        )

        now_time = datetime.datetime.fromtimestamp(price[0]['closeTime'] / 1000.0)

        st.markdown('Price at Server Time {}'.format(now_time.strftime("%Y-%m-%d %H:%M:%S")))

st.session_state.fig = [0, 0]

def refresh(rank):
    # webbrowser.open("http://www.github.com")
    global df1, df5
    df1 = combine('1m', rank)
    df5 = combine('5m', rank)

    # st.balloons()
    st.session_state.load = True
    st.session_state.data = [df1, df5]

    st.session_state.fig[0] = plot_coint(df1[0], '1m')
    st.session_state.fig[1] = plot_coint(df1[0], '5m')

    if st.session_state.load:
        with place.container():
            fig_col1, fig_col2 = st.columns(2)

            with fig_col1:
                st.markdown("### 1 Minute")
                st.markdown('Close Price at Server Time {}'.format(df1[0].index[-1].strftime("%Y-%m-%d %H:%M:%S")))
                if st.session_state.fig[0] != 0:
                    st.write(st.session_state.fig[0])

            with fig_col2:

                if st.session_state.load:
                    st.markdown("### 5 Minute")

                    st.markdown('Close Price at Server Time {}'.format(df5[0].index[-1].strftime("%Y-%m-%d %H:%M:%S")))
                # st.markdown("###### Updated {}".format(datetime.datetime.now()))
                    if st.session_state.fig[1] != 0:
                        st.write(st.session_state.fig[1])


            # csv = convert_df(df5[0][[f1, f2]])

def freeze():
    if len(f1) >= 2:
        st.session_state.data[0][0][f1].to_csv('one_min.csv')
    if len(f5) >= 2:
        st.session_state.data[1][0][f5].to_csv('five_min.csv')
    st.write('HEY!!!!!')

st.sidebar.title("Crypto Cointegration Analysis")
with st.sidebar:

    if st.session_state.load:
        _rank = st.selectbox("Select the Volume Rank", range(10, 143))
        c1, c2 = st.columns(2)
        with c1:
            run_button = st.button("RUN")

        if st.session_state.run >= 1:
            with placeholder.container():
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown("### 1 Minute")
                    st.markdown('Close Price at Server Time {}'.format(
                        st.session_state.data[0][0].index[-1].strftime("%Y-%m-%d %H:%M:%S")))
                    if st.session_state.fig[0] != 0:
                        st.write(st.session_state.fig[0])

                with fig_col2:

                    if st.session_state.load:
                        st.markdown("### 5 Minute")

                        st.markdown('Close Price at Server Time {}'.format(
                            st.session_state.data[1][0].index[-1].strftime("%Y-%m-%d %H:%M:%S")))
                        # st.markdown("###### Updated {}".format(datetime.datetime.now()))
                        if st.session_state.fig[1] != 0:
                            st.write(st.session_state.fig[1])

        if run_button:
            with st.spinner('Running...'):
                refresh(int(_rank))
                st.session_state.run += 1

        with c2:
            st.button("FREEZE", on_click=freeze)


        with st.form(key = "pairs"):

            f1 = st.multiselect('Select Two Futures for 1-min Data:', perps)
            f5 = st.multiselect('Select Two Futures for 5-min Data:', perps)

            submit = st.form_submit_button(label='FREEZE', on_click=freeze)


        link = '[GitHub](http://www.github.com)'
        st.markdown(link, unsafe_allow_html=True)




# while True:
#     price()
