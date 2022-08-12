import time  # to simulate a real time data, time loop
import streamlit as st  # 🎈 data web app development
import os
from screener import plot_coint
import datetime

import pandas as pd
import requests
import json
import webbrowser
from backend import CointegrationAnalysis

from binance.um_futures import UMFutures
import base64

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)
# dashboard title
st.title("Real-time Crypto Dashboard")



global df1, df5, fig1, fig2
test = CointegrationAnalysis()

st.session_state.load = True
root_url = 'https://api.binance.com/api/v1/klines'
time_url = "https://api.binance.com/api/v3/time"


bars = ["3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"]

def format_func(bar):

    if bar[-1] == 'm':
        return bar[:-1] + ' Minutes'
    elif bar[-1] == 'h':
        return bar[:-1] + ' Hours'
    elif bar[-1] == 'd':
        return bar[:-1] + ' Days'
    elif bar[-1] == 'w':
        return bar[:-1] + ' Week'
    else:
        return '1 Month'


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


futures = exchange_info()


# if there is no specification of limit in the url, then default the recent 500 entries
# read csv from a URL

def get_bars(symbol, interval='1m', limit=1000):
    root_url = 'https://api.binance.com/api/v1/klines'
    url = root_url + '?symbol=' + symbol + '&interval=' + interval + '&limit=' + str(limit)
    data = json.loads(requests.get(url).text)
    # df = pd.DataFrame(data)
    keys = ['open_time',
            'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades',
            'taker_base_vol', 'taker_quote_vol', 'ignore']

    res = []
    for d in data:
        dict1 = dict(zip(keys, d))
        dict1['symbol'] = symbol
        res.append(dict1)
        # print(dict1)
    # df.index = [datetime.datetime.fromtimestamp(x / 1000.0).strftime("%Y/%m/%d %H:%M:%S") for x in df.close_time]
    return pd.DataFrame(res)


def choose_volume(rank):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = json.loads(requests.get(url).text)

    volume = pd.DataFrame(data).set_index('symbol')[['volume']].astype(float).sort_values(ascending=False, by='volume')
    volume = volume.loc[[x for x in volume.index if 'USDT' in x], :]
    volume = volume.loc[[x for x in volume.index if x in futures], :]
    selected = volume.index[:rank]

    # data = data1 + data2 + data3 + data4
    return selected.tolist()



# @st.cache(allow_output_mutation=True)
def combine(interval, rank=10, limit=1000):
    selected = choose_volume(rank)

    data = pd.DataFrame()
    for perp in selected:
        try:
            data = pd.concat([data, get_bars(perp, interval, limit)], axis=0, join='outer')
            # volume[perp] = float(data[perp].iloc[-1, :].volume)
            # print(volume[perp])

        except:
            pass

    # selected = sorted(volume, key=volume.get, reverse=True)[:rank]

    # close_df = pd.DataFrame(data).set_index('close_time')[['close']]
    temp = data.set_index('symbol')[['close', 'close_time']].reset_index()
    close = temp.pivot(index='close_time', columns='symbol', values='close')
    close.index = [datetime.datetime.fromtimestamp(x / 1000.0).strftime("%Y/%m/%d %H:%M:%S") for x in close.index]

    return close.astype(float), selected

# top-level filters



# creating a single-element container
placeholder = st.empty()
place = st.empty()


st.session_state.run = 0
st.session_state.load = True
st.session_state.submit = False



def refresh(rank, interval1, interval2, limit1, limit2):
    # webbrowser.open("http://www.github.com")
    global df1, df5, fig1, fig2
    df1 = combine(interval1, rank, limit1)
    df5 = combine(interval2, rank, limit2)


    st.session_state.data = [df1, df5]
    # st.balloons()
    st.session_state.load = True

    fig1 = plot_coint(df1[0], interval1)
    fig2 = plot_coint(df5[0], interval2)
    st.session_state.fig = [fig1, fig2]

    if st.session_state.load:
        with place.container():
            fig_col1, fig_col2 = st.columns(2)

            with fig_col1:
                st.markdown(f"### {format_func(interval1)}")
                st.markdown('Last Close Time at Server Time(UTC) {}'.format(df1[0].index[-1]))

                st.write(fig1)


            with fig_col2:

                if st.session_state.load:
                    st.markdown(f"### {format_func(interval2)}")

                    st.markdown('Last Close Time at Server Time(UTC) {}'.format(df5[0].index[-1]))
                # st.markdown("###### Updated {}".format(datetime.datetime.now()))

                    st.write(fig2)

    # return df1, df5, fig1, fig2

            # csv = convert_df(df5[0][[f1, f2]])


def csv_downloader(data, filename, title):
    csvfile = data.to_csv()
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "{}_{}_.csv".format(filename, data.index[-1])
    st.markdown(f"###### Download {title} ######")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)

def freeze(f1, f5, df1, df5, fig1, fig2, interval1, interval2):
    one_min_close = df1[0][f1]
    fit = test.fit_eg(one_min_close, f1[0], f1[1],
                        startDate=one_min_close.index[0],
                        endDate=one_min_close.index[-1], plot = False)

    fitdf = fit.fitdf
    spreaddf = fit.spreaddf

    five_min_close = df5[0][f5]
    fit_ = test.fit_eg(five_min_close, f5[0], f5[1],
                       startDate=five_min_close.index[0],
                       endDate=five_min_close.index[-1], plot=False)

    fitdf_ = fit_.fitdf
    spreaddf_ = fit_.spreaddf


    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"##### Cointegration Parameters for {format_func(interval1)}")
        st.dataframe(fitdf)

        csv_downloader(one_min_close, f'{format_func(interval1).lower()}_close', f'{format_func(interval1)} Close Price')

        csv_downloader(spreaddf, f'spread_{format_func(interval1).lower()}', f'{format_func(interval1)} Spread')

        st.markdown(f"##### Cointegration Parameters for {format_func(interval2)}")

        st.dataframe(fitdf_)

        csv_downloader(five_min_close, f'{format_func(interval2).lower()}_close', f'{format_func(interval2)}Close Price')

        csv_downloader(spreaddf_, f'spread_{format_func(interval2).lower()}', f'{format_func(interval2).lower()} Spread')


    with col2:
        st.markdown(f"### {format_func(interval1)}")
        st.markdown('Last Close Time at Server Time(UTC) {}'.format(
            one_min_close.index[-1]))
        st.write(fig1)


        st.markdown(f"### {format_func(interval2)}")
        st.markdown('Last Close Time at Server Time(UTC) {}'.format(
            five_min_close.index[-1]))
        # st.markdown("###### Updated {}".format(datetime.datetime.now()))
        st.write(fig2)


    # rolling = st.button("Rolling Window")
    # rolling_ = st.button("Rolling Window")


st.sidebar.title("Crypto Cointegration Analysis")
with st.sidebar:

    if st.session_state.load:
        f = st.form(key='params')
        _rank = f.selectbox("Select the Volume Rank", range(10, 143))
        perps = choose_volume(int(_rank))
        interval1 = f.selectbox("Select the 1st Bar Length", bars, format_func=format_func, key = 'interval1')
        limit1 = f.slider('Slide 1st Data Length', min_value=30, max_value=1000, step=10, key = 'limit1')
        interval2 = f.selectbox("Select the 2nd Bar Length", bars, format_func=format_func, key = 'interval2')
        limit2 = f.slider('Slide 2nd Data Length', min_value=30, max_value=1000, step=10, key = 'limit2')


        run_button = f.form_submit_button(label = "RUN")
        if run_button:
            with st.spinner('Running...'):
                refresh(int(_rank),
                        st.session_state.interval1,
                        st.session_state.interval2,
                        st.session_state.limit1,
                        st.session_state.limit2)



        # p = st.empty()

        form = st.form(key="pairs")


        f1 = form.multiselect(f'Select Two Futures for {format_func(interval1)} Data:', perps, key = 'p1')

        # st.write(f1)
        f5 = form.multiselect(f'Select Two Futures for {format_func(interval2)} Data:', perps, key = 'p5')
                # st.write(f5)

        # st.write(window)
        submit = form.form_submit_button(label='FREEZE',
                                         on_click=lambda: freeze(st.session_state.p1,
                                                                 st.session_state.p5,
                                                                 st.session_state.data[0], st.session_state.data[1],
                                                                 st.session_state.fig[0], st.session_state.fig[1],
                                                                 st.session_state.interval1, st.session_state.interval2))

        link = '[GitHub](https://github.com/irismguo/streamlit_coint_app)'
        st.markdown(link, unsafe_allow_html=True)

