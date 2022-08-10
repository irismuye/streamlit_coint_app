import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

from cmath import nan

from tqdm import tqdm

import altair as alt

from backend import CointegrationAnalysis, StationaryTest

from statsmodels.tsa.stattools import adfuller, coint


def _filter_volume(path, rank=10):
    file = os.listdir(path)[6]

    file_path = path + '/' + file

    vol = _extract_volume(file_path).iloc[-1, :]

    selected = vol.sort_values(ascending=False)[:rank].index

    close = _extract_close(file_path)[selected]

    return close


def plot_coint(close, interval):
    coins = close.columns
    price = np.log(close).dropna()
    # print(price.columns)

    plot_df = pd.DataFrame(index=coins)
    # for i in tqdm(range(len(coins)), desc='Calculating...', ncols=75, leave=True, position=0):
    for i in range(len(coins)):
        plist = [nan] * len(coins)
        for j in range(i + 1, len(coins)):
            test = StationaryTest()
            if test.is_OneOrder(price.iloc[:, [i, j]], dim='multiple'):
                p = coint(price.iloc[:, i], price.iloc[:, j], trend='c', autolag='AIC')[1]
                plist[j] = p

        # print(plist)
        single = pd.DataFrame(index=coins, columns=[coins[i]], data=plist)
        # print(single)
        plot_df = pd.concat([plot_df, single], axis=1)

    for coin in coins:
        plot_df.loc[coin, :] = plot_df.loc[:, coin]

    # print(plot_df)

    lmt = plot_df.reset_index().melt(id_vars='index', var_name='coin', value_name='p')
    # print(lmt)
    chart = alt.Chart(lmt).mark_rect().encode(x='index:O', y='coin:O', tooltip=['index:O', 'coin:O', 'p:Q'],
                                              color=alt.Color("p:Q", scale=alt.Scale(scheme="magma"))
                                              ).properties(
        width=500,
        height=500
    )

    chart.interactive()
    chart.save('screener_{}.html'.format(interval))

    return chart


def screener(rank=10, interval='1m'):
    if interval == '1m':
        close_1m = _filter_volume('Data/1m', rank)
        plot_coint(close_1m, '1m')

    if interval == '5m':
        close_5m = _filter_volume('Data/5m', rank)
        plot_coint(close_5m, '5m')

    if interval == '1d':
        close_1d = _filter_volume('Data/1d', rank)
        plot_coint(close_1d, '1d')

# print(_filter_volume('Data/5m'))
# print(_filter_volume('Data/1m'))
#
# print(_filter_volume('Data/1d'))


