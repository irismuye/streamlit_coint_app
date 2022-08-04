# -*-coding:utf-8-*-
"""The Cointegration Matrix Program Prototype API"""
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.vector_ar.vecm import CointRankResults, select_order, select_coint_rank
from statsmodels.tools.validation import *
from statsmodels.tsa.tsatools import *
from statsmodels.tsa.adfvalues import *
from statsmodels.regression.linear_model import *

import matplotlib.colors as mcolors


class StationaryTest:

    def __init__(self, method="ADF", regression="c", threshold=0.05):
        self.method = method
        self.regression = regression
        self.threshold = threshold

        self.testres_ = None

    def _check_param(self, df):
        '''
        Function: check if params valid

        '''
        # if df.isnull().values.any():
        #     nan_rows = df[df.isnull().any(axis = 1)]
        #     print(f"Detected NaN Values in Rows: {list(nan_rows.index.format())}. Dropped.")

        return df.dropna(how='any')

    def ADFtest_pvalues(self, df):

        '''
        Function:
        *************
        Calculate the ADF test p-value of a 1x1 dataframe

        Input:
        *************
        df: dataframe -> a 1x1 dataframe or a 1d serie


        Return:
        *************
        float -> p-value of ADF test

        '''
        # print("Results of Augmented Dickey-Fuller Test:")
        # regression = {“c”,”ct”,”ctt”,”n”}
        if self.method == "ADF":
            dftest = adfuller(df, autolag='AIC', regression=self.regression)

        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistics",
                "p-value",
                "# Lags Used",
                "Number of Observations Used"
            ]
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
            # print(dfoutput)

        return dftest[1]  # return only p-values

    def is_OneOrder(self, df, dim='single'):
        '''
        Function:
        *************

        Check if a given 1d series is I(1) order

        Input:
        *************

        df: dataframe -> a 1x1 dataframe, or a 1d serie
        dim: string -> default as 'single'. It chooses from {'single', 'multiple'}.
                    'single': if only checks one future data; 'multiple': if checks multiple futures data

        Return:
        *************
        bool -> whether it's all I(1) or not

        '''
        # test the I(1) or I(0) variables for the given columns in the dataframe
        if dim == 'single':
            pvalueLevel = self.ADFtest_pvalues(df)
            pvalueDiff = self.ADFtest_pvalues((df - df.shift(1)).dropna())
            return ((pvalueLevel > self.threshold) & (pvalueDiff < self.threshold))

        if dim == 'multiple':
            res = []

            for col in df.columns:
                p = self.ADFtest_pvalues(df.loc[:, col])
                pdiff = self.ADFtest_pvalues((df.loc[:, col] - df.loc[:, col].shift(1)).dropna())
                res.append((p > self.threshold) & (pdiff < self.threshold))

            return all(res)

    def rolling_zero_order_count(self, full_df,
                                 startDate='2010/01/01', endDate='2022/03/31',
                                 window=126, plot=True):

        '''
        Function:
        *************
        Count the number of stationary(0 order) futures during every rolling period of window size,
        and plot the number against rolling end time

        Input:
        *************

        full_df: dataframe -> historical data
        startDate: string -> the start date of the desired time frame, e.g. '2010-01-01', '2010/01/01', 01/01/2010'
        endDate: string -> the end date of the desired time frame, e.g. '2010-01-01', 2010/01/01', 01/01/2010'
        window: integer -> default as 126. It specifies the size of the rolling window.
                           This parameter has the same frequency as full_df, thus needs to be scaled by users before plugging in
        plot: bool -> if plot, then True

        Return:
        *************
        resultDF: dataframe -> results of rolling count, with columns of rolling startDate and endDate, and stationary count


        '''

        df = full_df.loc[startDate:endDate]
        futures = df.columns

        resultDF = pd.DataFrame(columns=['StartDate', 'EndDate', 'ZeroOrderCount'])
        for ilc in range(window - 1, len(df)):
            startdate = df.index[(ilc + 1) - window]
            enddate = df.index[ilc]

            sel_df = self._check_param(df.loc[startdate:enddate])
            count = 0
            for f in futures:
                if self.ADFtest_pvalues(sel_df.loc[:, f]) <= .05:
                    count += 1

            resultDF.loc[df.index[ilc], 'StartDate'] = startdate
            resultDF.loc[df.index[ilc], 'EndDate'] = enddate
            resultDF.loc[df.index[ilc], 'ZeroOrderCount'] = count

        if plot:
            tdf = resultDF.reset_index().rename(columns={'index': 'period_end'})
            sns.set(style="whitegrid", rc={'figure.figsize': (50, 50)})
            sns.relplot(x="period_end", y="ZeroOrderCount",
                        data=tdf,
                        dashes=False, markers=True, kind="line", aspect=5000 / 300)
            sns.scatterplot(x="period_end", y="ZeroOrderCount", data=tdf)

            plt.ylabel("Number of Stationary Series")
            plt.xlabel(f"End Date of Rolling Window of {window}")
            plt.ylim(0, 6)
            plt.xticks(resultDF.index[range(0, len(tdf), 30)])
            plt.xticks(rotation=45)

        return resultDF

    def rolling_zero_order_test(self, full_df,
                                startDate='2010/01/01', endDate='2022/03/31',
                                window=126, plot=True):

        '''
        Function:
        *************
        This function calculates the adfuller p-value of each future data included in the input dataframe, and output the heatmap
        of both p-value and stationary test boolean results

        Input:
        *************
        full_df: dataframe: historical data
        startDate: string-> the start date of the desired time frame, e.g.'2010-01-01', '01/01/2010', '2010/01/01'
        endDate: string -> the end date of the desired time frame, e.g.'2010-01-01', '01/01/2010', '2010/01/01'
        window: integer -> default as 126. It specifies the size of the rolling window.
                           This parameter has the same frequency as the input dataframe full_df, thus needs to be scaled by user
                           before plugging in
        plot: bool -> default as True. If plot, then True


        Return:
        **************
        resdf: dataframe -> dataframe of the rolling stationary results, with columns of rolling window start date and end date, p-value
                            and stationary test result, and index of each future columns names

        '''

        df = full_df.loc[startDate:endDate]
        futures = df.columns

        res_list = []
        for ilc in range(window - 1, len(df)):
            resultDF = pd.DataFrame(columns=['StartDate', 'EndDate', 'p-value', 'Stationary'])
            startdate = df.index[(ilc + 1) - window]
            enddate = df.index[ilc]

            sel_df = self._check_param(df.loc[startdate:enddate])

            for f in futures:
                fp = self.ADFtest_pvalues(sel_df.loc[:, f])
                resultDF.loc[f, 'p-value'] = fp
                resultDF.loc[f, 'StartDate'] = startdate
                resultDF.loc[f, 'EndDate'] = enddate
                resultDF.loc[f, 'Stationary'] = (fp <= .05)

            res_list.append(resultDF)

        resdf = pd.concat(res_list, join='inner')

        if plot:
            tdf = resdf.reset_index().rename(columns={'index': 'futures'})[
                ['futures', 'EndDate', 'p-value', 'Stationary']].copy()
            # print(tdf)
            tdf0 = tdf.pivot(columns='futures', index='EndDate', values='p-value').astype(float)
            tdf_1 = tdf.pivot(columns='futures', index='EndDate', values='Stationary').astype(float)
            tdf0.index = tdf0.index.format()
            tdf_1.index = tdf_1.index.format()
            # tdf.to_excel('mid_res_zero_order.xlsx')
            # print(tdf)

            fig, (ax1, ax2) = plt.subplots(ncols=2)
            fig.subplots_adjust(wspace=0.01)

            sns.set(rc={'figure.figsize': (50, 50)})
            cmap = "YlGnBu_r"
            sns.heatmap(tdf0, annot=True, ax=ax1, cmap=cmap)
            sns.heatmap(tdf_1, annot=True, ax=ax2, cmap=cmap, fmt='g')
            plt.title(f"Stationary Test under Window {window}")
            plt.ylabel(f"End Dates of Rolling Time Window of {window}")
            plt.xlabel("Futures")
            plt.show()

        return resdf



class CointegrationAnalysis:
    '''
    Example:

    >>> test = CointegrationAnalysis()
    >>> test.fit(newLDF, 'ES', 'NQ', startDate = '04/30/2000', endDate = '05/30/2020')
    >>> test.test(newLDF, 'ES', 'NQ')


    '''

    def __init__(self,
                 method = "EG", thresh = .05):


        '''
        Input:
        df: dataframe ->
        method: string -> coint test method
        trend: string -> trend type
        frequency: string -> historical data frequency
        window: integer -> rolling window
        '''
        self.method = method
        self.thresh = thresh


        self.coef = None
        self.const = None
        self.mean = None
        self.std = None

        self.fitdf = None

        self.testdf = None
        self.testBool = None


    def _check_param(self, df):
        '''
        Function: check if params valid

        '''
        if df.isnull().values.any():
            nan_rows = df[df.isnull().any(axis = 1)]
            print(f"Detected NaN Values in Rows: {list(nan_rows.index.format())}. Dropped.")


        return df.dropna(how = 'any')

    def _tune_index(self, df):

        df.index = pd.to_datetime(df.index)
        return df.copy()





    def fit_eg(self, full_df, endoCol, exogCol,
               startDate = '1999/06/25', endDate = '2022/03/31',
               trend = 'c', plot = True):
        '''
        Function:
        *************

        Calculate cointegration vector, const(alpha), spread mean and spread std, and store them in a dataframe,
        and plot the spread. Access the result dataframe through self.fitdf, the spread dataframe
        through self.spreaddf.

        This function can be used either for pair or multiple futures calculation, through altering the 'exogCol' parameter
        to be a single string of future name, or a list of future names.

        If the user wants access the coefficients, const, mean and std directly, use self.coef, self.const, self.mean,
        self.std

        Input:
        *************
        df: dataframe -> full historical data
        endoCol: string -> Column name of endo, must be 1d
        exogCol: string or list -> Column names of exog, 1d when pair calculation, nd when multiple calculation
        startDate: string -> start date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        endDate: string -> end date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        trend: string -> default as 'c'. It chooses from {'c', 'ct'}, used to specify the trend add to the exog in the regression
        plot: bool -> default as True. If plot(for spread)

        Return:
        *************
        self

        Call Results:
        *************

        >>> self.coef -> cointegration vector/matrix
        >>> self.const -> const of the regression
        >>> self.mean -> spread mean
        >>> self.std  -> spread std
        >>> self.fitdf -> dataframe of fitting result, including coefs, const, mean, and std
        >>> self.spreaddf -> dataframe of spread series, with columns of spread values, and index of datetime

        '''


        full_df = self._tune_index(full_df)

        #Check input validation

        if pd.to_datetime(startDate) < full_df.index[0]:
            raise ValueError("startDate out of bounary")

        if pd.to_datetime(endDate) > full_df.index[-1]:
            raise ValueError("enddate out of boundary")

        if trend not in {'c', 'ct'}:
            raise ValueError("trend Invalid")



        df = full_df.loc[startDate:endDate]
        # print(df.iloc[0, :])
        df = self._check_param(df).copy()
        endo = array_like(df[endoCol], "endo")
        exog = array_like(df[exogCol], "exog", ndim=2)
        trend = rename_trend(trend)
        trend = string_like(trend, "trend", options=("c", "n", "ct", "ctt"))
        xx = add_trend(exog, trend=trend, prepend=False)


        res_ols = OLS(endo, xx).fit()
        if isinstance(exogCol, str):
            exogCol = [exogCol]
        beta = res_ols.params[0:len(exogCol)]
        beta = np.concatenate((np.array([1]), -beta))
        self.coef = beta
        coef_df = pd.DataFrame(index = ['values'])
        for i in range(len(beta)):
            coef_df.loc['values', str(([endoCol]+exogCol)[i])] = beta[i]
        # print(coef_df)

        mean = res_ols.params[1]
        self.const = mean
        miu = mean + np.mean(res_ols.resid)
        self.mean = miu

        sigma = np.std(res_ols.resid)
        self.std = sigma

        spread = endo - res_ols.predict(xx) + mean
        # spread = self.coef@df.loc[:, [endoCol, exogCol]].T

        fitdf = pd.DataFrame(columns = ['const', 'mean', 'std'], index = ['values'])
        fitdf = pd.concat([coef_df, fitdf], axis = 1, join = 'outer')
        fitdf.loc['values', 'const'] = mean
        fitdf.loc['values', 'mean'] = miu
        fitdf.loc['values', 'std'] = sigma
        self.fitdf = fitdf
        self.coefdf = coef_df
        self.spreaddf = pd.DataFrame(data = spread, columns = ['spread']).copy()
        # print(df.index)
        # print(self.spread_df)
        self.spreaddf.loc[:, 'dates'] = df.index


        if plot:
            f, ax = plt.subplots()
            sns.set(rc = {'figure.figsize':(100, 50)})
            sns.relplot(x = "dates", y = "spread",
                        data = self.spreaddf,
                        dashes = False, markers = True, kind = "line", aspect = 100/50)
            sns.scatterplot(x = "dates", y = "spread", data = self.spreaddf)

            plt.ylabel("portfolio spread")
            plt.xlabel(f"dates")
            # plt.xticks(resultDF.index[range(0, len(tdf), 30)])
            plt.xticks(rotation = 45)
            plt.savefig('fit_plot.png')
            plt.close()

        return self





    def test_eg(self, full_df, endoCol, exogCol,
                trend = 'c', dim = 'pair',
                startDate = '06/25/1999', endDate = '05/31/2022'):
        '''
        Function:
        *************

        This is basic test function for paired or multiple futures for the functions below, it does:
           a. Calculate t-statistics, p-values and criterion values for certain timeframe
           b. Check if cointegrate

        Returns result dataframes

        Input:
        *************

        df: dataframe -> historical data
        endoCol: string -> Column name of endo, must be 1d
        exogCol: string or list -> Column names of exog, 1d when paired calculation, nd when multiple calculation
        trend: string -> default as 'c'. It chooses from {'c', 'ct'}, which is an input for adfuller
        dim: string -> default as 'pair'. It chooses from {'pair', 'multiple'}, which specifies the test portfolio size
        startDate: string -> start date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        endDate: string -> end date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'


        Return:
        *************

        self

        Call Results:
        *************

        >>> self.testdf -> dataframe of test statistics: t-statistics, p-values and critical values
        >>> self.testBool -> bool value of whether the series are cointegrated


        '''

        full_df = self._tune_index(full_df)

        if pd.to_datetime(startDate) < full_df.index[0]:
            raise ValueError("startDate out of boundary")

        if pd.to_datetime(endDate) > full_df.index[-1]:
            raise ValueError("enddate out of boundary")

        if trend not in {'c', 'ct'}:
            raise ValueError("trend Invalid")




        df = full_df.loc[startDate:endDate].copy()
        df = self._check_param(df).copy()
        # print(df.loc[:, endoCol])
        endo_1 = StationaryTest().is_OneOrder(df.loc[:, endoCol])

        if dim == 'pair':
            exog_1 = StationaryTest().is_OneOrder(df.loc[:, exogCol])
        elif dim == 'multiple':
            exog_1_l = []
            for e in exogCol:
                exog_1_l.append(StationaryTest().is_OneOrder(df.loc[:, e]))

            exog_1 = all(exog_1_l)

        if endo_1 & exog_1:
            result = coint(df.loc[:, endoCol], df.loc[:, exogCol], trend = trend, autolag='AIC')

            resseries = pd.Series(result[0:2], index=['t-statistic', 'p-value'])
            critic = ['1%', '5%', '10%']
            for i in range(0, len(critic)):
                resseries[critic[i]] = result[2][i]

            testdf = pd.DataFrame(resseries, columns=[endDate]).T
            testdf.loc[endDate, 'res'] = (result[1] < self.thresh)
            self.testdf = testdf.copy()


            self.testBool = (result[1] <= self.thresh) # the bool result of pair test
        else:

            #if not all I(1), use an empty dataframe
            self.testdf = pd.DataFrame(index = [endDate],
                                       columns = ['t-statistic', 'p-value', '1%', '5%', '10%', 'res'])
            self.testdf.loc[endDate, 'res'] = np.nan

            self.testBool = 'Not All I(1) order'
        # print(result)



        return self



    def johansen_test(self, df, columns, det_order, trend = 'c', method = 'trace', select_order = -1):

        '''
        this function conducts the johansen test on the given columns of the given dataframe and calculates the eigenvalue and the cointegraton vector/matirx


        Input:
        df: dataframe -> input future data
        columns: list of strings -> column names of the futures (only accepts 2 futures for now)
        det_order: integer -> {-1, 0, 1}. -1: no deterministic term; 0: constant term; 1: linear trend
        trend: string -> default as 'c'. It specifies the trend in VAR model
        method: string -> {'trace', 'maxeig'}. It specifies the method used in select_coint_rank
        select_order: integer -> -1: code-selected order(based on VAR model); >0: select_order user-selected order

        Return:

        JohansenResults() -> stores rank, eigenvalues, eigenvectors and test result, among which only the cointegrated eigenvalues and eigenvectors are kept

        If there are no cointegration relationships exist, NO value is returned

        Access rank, eigenvalues, eigenvectors, test results and future names through .rank, .eig, .evec, .res, .futures

        '''
        if len(columns) > 2:
            raise ValueError('Only Accepts Two Futures For Now.')


        sel_df = self._check_param(df[columns])

        #select order
        testres = False
        print('$$ Johansen Cointegration Test Results Summary $$: ')
        if StationaryTest().is_OneOrder(sel_df, dim = 'multiple'):
            if select_order == -1:
                model = VAR(sel_df)
                order = model.select_order(trend = trend).selected_orders['aic'] - 1

            else:
                order = select_order



            rank = select_coint_rank(sel_df, det_order, k_ar_diff = order, method=method).rank


            if rank > 0 and rank != len(columns):

                testres = True

                test = coint_johansen(sel_df, det_order, order)


                eig = test.eig[:rank]

                evec = test.evec[:, :rank]

                return JohansenResults(rank, eig, evec, testres, columns)

            elif rank == len(columns):
                res = 'Stationary Series Exist.'
                return testres, res

            else:
                res = 'No Cointegration Relationships Found.'
                return testres, res

        else:
            res = 'ADF Test: Not All I(1) Order'
            return testres, res





    def johansen_validation(self, df, columns, det_order, trend = 'c',
                            method = 'trace',
                            thresh = .05, select_order = -1):


        '''
        This function validates the Johansen test results. Check if the cointegration vectors/matrices really create stationary processes

        Input:

        df: dataframe -> an input dataframe of level data
        columns: list -> a list of strings that specifies the column names of the futures
        det_order: integer -> It chooses from {-1, 0, 1}. -1: no deterministic term; 0: constant term; 1: linear trend
        trend: string -> default as 'c'. It specifies the trend used in johansen_test()
        method: string -> default as 'trace'. It chooses from {'trace', 'maxeig'}, in which the strings specify the method used in select_coint_rank
        thresh: float -> default as .05. It specifies the confidence level of the test result
        select_order: integer -> Choose whether let the computer(code) select lag order(based on VAR model), or let user select a lag order. -1: let the computer(code) select;
                                >0: use select_order as the lag order(user-select)



        Return:
        JohansenValidationresults() -> stores the ranks, eigenvalues, eigenvectors, lists of boolean results and results dataframes, among which only the cointegrated eigenvalues and eigenvectors are stored.

        If there are no cointegration relationships exist, No value will be returned.

        Access the ranks, eigenvalues, eigenvectors, lists of boolean results and results dataframes through .rank, .eig, .evec, .res, .resdf

    '''

        if len(columns) > 2:
            raise ValueError('Only Accepts two Futures For Now')


        sel_df = self._check_param(df[columns])

        test = self.johansen_test(sel_df, columns, det_order, trend = trend, method = method, select_order = select_order)

        if not isinstance(test, tuple):

            evec = test.evec
            eig = test.eig
            rank = test.rank


            check_list = []
            pv = []

            for i in range(rank):
                v = evec[:, i]
                spread = np.dot(sel_df, v)

                p = adfuller(spread)[1]
                pv.append(p)

                check_list.append(p <= thresh)

            resdf = pd.DataFrame(data = check_list, index = np.arange(rank)+1, columns = ['Stationary'])

            resdf.loc[:, 'eig'] = eig
            resdf.loc[:, 'p-values'] = pv

            resdf.loc[:, columns] = evec.T

            return JohansenValidationResults(rank, eig, evec, check_list, resdf)

        else:
            return JohansenValidationResults(None, None, None, None, 'No Need to Check. Test Result False.')





    def rolling_test(self, full_df, endoCol, exogCol, dim = 'pair',
                     startDate = '06/25/1999', endDate = '05/31/2022',
                     window = 30,
                     plot = True):

        '''
        Function:
        *************

        Calculate the rolling window test of paired or multiple futures, and plot the overview of the cointegration.

        This function supports one future, and multiple futures in exogCol. exogCol and dim specifies the types of the regressions

        Tips: Don't input a very long time frame, or the lines and areas will be squeezed together and become useless,
        and it takes very long time to run

        Inputs:
        *************

        full_df: dataframe -> historical data
        endoCol: string -> column name of endo, must be 1d
        exogCol: string or list -> columns names of exog, 1d when paired calculation, nd when multiple calculation
        dim: string -> default as 'pair'. It chooses from {'pair', 'multiple'}, which specifies the size of the test portfolio
        startDate: string -> start date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        endDate: string -> end date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        window: integer -> default as 30. It specifies the rolling window size.
                           This parameter has the same unit as full_df, thus needs to be scaled by user before plugging it in,
                           e.g. if full_df is daily data, then window = 30 means a rolling window of 30 days.
        plot: bool -> default as True. If plot, then True.

        Return:
        *************

        self

        Call Results:
        *************

        >>> self.rolling_testdf -> dataframe that include every results, including the nan values where not all futures are I(1)
        >>> self.rolling_coint ->  dataframe that only stores the cointegrated period




        '''

        full_df = self._tune_index(full_df)

        #check fot invalid input

        if pd.to_datetime(startDate) < full_df.index[0]:
            raise ValueError("startDate out of boundary")
        if pd.to_datetime(endDate) > full_df.index[-1]:
            raise ValueError("endDate out of boundary")
        if dim not in {'pair', 'multiple'}:
            raise ValueError("dim not recognized")
        if not isinstance(window, int):
            raise ValueError("Invalid window value")


        df = full_df.loc[startDate:endDate]
        df = self._check_param(df.copy()).copy()
        resultDF = pd.DataFrame(columns=['StartDate', 'EndDate', 'p-value'])
        for ilc in range(window-1, len(df)):
            test = self.test_eg(df,
                                     endoCol, exogCol,
                                     startDate=df.index[(ilc+1)-window],
                                     endDate = df.index[ilc], dim = dim)
            singleResult  = test.testdf.copy()

            resultDF = pd.concat([resultDF, singleResult], axis=0, join='outer')
            resultDF.loc[df.index[ilc], 'StartDate'] = df.index[(ilc+1)-window]
            resultDF.loc[df.index[ilc], 'EndDate'] = df.index[ilc]

            if test.testBool == True:
                fitres = self.fit_eg(df, endoCol, exogCol, startDate = df.index[(ilc+1)-window],
                                     endDate = df.index[ilc], plot = False).fitdf.copy()
                resultDF.loc[df.index[ilc], endoCol] = fitres.loc['values', endoCol]
                resultDF.loc[df.index[ilc], exogCol] = fitres.loc['values', exogCol]

            # #fit

            # if resultDF.loc[df.index[ilc], 'p-value'] < self.thresh:
            #     resultDF.loc[df.index[ilc],
            #                  ['beta', 'const', 'mean_spread', 'std_spread']] = self.fit(
            #                      df.iloc[(ilc+1)-window:(ilc+1)], endoCol, exogCol)

        self.rolling_testdf = resultDF.copy()
        # print(resultDF)
        coint_df = resultDF[resultDF['p-value'] <= self.thresh].copy()
        self.rolling_coint = coint_df.copy()

        if plot:

            # time series plot
            # sns.set(style="whitegrid", rc = {'figure.figsize':(100, 50)})
            # df.index = df.index.strftime('%Y/%m/%d')
            sns.set_theme(rc = {'figure.figsize':(18.5, 10.5)})
            if isinstance(exogCol, list):
                fs = exogCol
                fs.append(endoCol)
                ax = sns.lineplot(data = df[fs], palette = "tab10", linewidth = 2.5)
            else:
                ax = sns.lineplot(data=df[[endoCol, exogCol]], palette="tab10", linewidth=2.5)


            if len(coint_df) > 0:
                for i in range(len(coint_df)):
                    start = coint_df.loc[coint_df.index[i], 'StartDate']
                    end = coint_df.loc[coint_df.index[i], 'EndDate']
                    if isinstance(exogCol, list):
                        minf = df[exogCol].mean().idxmin()
                        ax.fill_between(df.loc[start:end].index,
                                        df[endoCol].loc[start:end],
                                        df[minf].loc[start:end],
                                        interpolate=True,
                                        color = 'purple',
                                        alpha = 0.25,
                                        label = "Cointegrated Period")
                    else:
                        ax.fill_between(df.loc[start:end].index,
                                        df[endoCol].loc[start:end],
                                        df[exogCol].loc[start:end],
                                        interpolate=True,
                                        color = 'purple',
                                        alpha = 0.25,
                                        label = "Cointegrated Period")

                    ax.axvline(start, color = 'red')
                    ax.axvline(end, color = 'green')
            else:
                print(f'No Cointegrated Period under window {window}')

            plt.title(f"Rolling EG test under window {window}")
            plt.xlabel('Dates')
            plt.ylabel('Log Close Price')
            plt.savefig('rolling_test_plot.png')
            # plt.show()
            plt.close()

        return self



    def window_test(self, full_df,
                    startDate = '2009-02-02', endDate = '2022-03-31',
                    window = 126, dim = "pair", axis = 2, plot = True):


        '''
        Function:
        *************
        In this function, the data is divided into chunks with size of window, and cointegration test was conducted in each chunk.
        The function can be calculated under pairwise or multiple future situations.

        Pairwise or multiple of multiple futures dataframe(using param 'axis' to change endo and exog)

        (pairwise and multiple) when we say 'pairwise' here, we loop through every possible combination of equities/future

        Input:
        *************

        full_df: dataframe -> dataframe with all future's close price, with their names as column names
        startDate: string -> start date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        endDate: string -> end date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        window: integer -> default as 126. It specifies the size of the window that divides the whole data
                           This parameter has the same frequency as the input full_df, thus needs to be scaled
                           by user before plugging in
        dim: string -> default as 'pair'.  It chooses from {'pair', 'multiple'}, which specifies the size of the test portfolio
        axis: integer -> only needs to specify when dim = "pair"
                         0 means endo, exog follows the ascending order of future names,
                         1 means endo, exog follows the descending order of future names
                         2 means both orders are calculated.
        plot: bool -> if plot, then plot the heatmap of p-values of each pair and time frame


        Return:
        *************
        self

        Call Results:
        *************
        self.window_testdf: dataframe ->
                            dates are datetime
                            columns: 'StartDate', 'Enddate', 't-statitic', 'p-value', 'pair', '1%', '5%', '10%', 'res'
                            index: 'period_end' = 'EndDate'


        '''

        full_df = self._tune_index(full_df)
        #future pairs
        future = full_df.columns
        # print(future_pairs)

        #dataframe
        df = full_df.loc[startDate:endDate].copy()

        res_list = []

        if dim == "pair":
            future_pairs = list(combinations(future, 2))
            for pairs in future_pairs:
                sel_df = df[[pairs[0], pairs[1]]]

                if axis == 0 or axis == 2:
                    print(pairs)
                    resultDF = pd.DataFrame(columns=['StartDate', 'EndDate'])

                    for ilc in range(window-1, len(sel_df), window):
                        test = self.test_eg(sel_df, pairs[0], pairs[1],
                                                 startDate=sel_df.index[ilc+1-window],
                                                 endDate=sel_df.index[ilc])
                        singleResult = test.testdf.copy()
                        resultDF = pd.concat([resultDF, singleResult], axis=0, join='outer').copy()
                        resultDF.loc[sel_df.index[ilc], 'StartDate'] = sel_df.index[(ilc+1)-window]
                        resultDF.loc[sel_df.index[ilc], 'EndDate'] = sel_df.index[ilc]
                        resultDF.loc[sel_df.index[ilc], 'pair'] = str((pairs[0], pairs[1]))

                        if test.testBool == True:
                            fitres = self.fit_eg(sel_df, pairs[0], pairs[1],
                                                 startDate = sel_df.index[ilc+1-window],
                                                 endDate = sel_df.index[ilc],
                                                 plot = False).fitdf.copy()
                            resultDF.loc[df.index[ilc], pairs[0]] = fitres.loc['values', pairs[0]]
                            resultDF.loc[df.index[ilc], pairs[1]] = fitres.loc['values', pairs[1]]

                    if axis == 0:
                        res_list.append(resultDF)

                if axis == 1 or axis == 2:
                    print((pairs[1], pairs[0]))
                    resultDF_1 = pd.DataFrame(columns=['StartDate', 'EndDate'])

                    for ilc in range(window-1, len(sel_df), window):
                        test_1 = self.test_eg(sel_df, pairs[1], pairs[0],
                                                   startDate=df.index[ilc+1-window],
                                                   endDate = df.index[ilc])
                        singleResult_1 = test_1.testdf.copy()
                        resultDF_1 = pd.concat([resultDF_1, singleResult_1], axis=0, join='outer').copy()
                        resultDF_1.loc[sel_df.index[ilc], 'StartDate'] = df.index[(ilc+1)-window]
                        resultDF_1.loc[sel_df.index[ilc], 'EndDate'] = df.index[ilc]
                        resultDF_1.loc[sel_df.index[ilc], 'pair'] = str((pairs[1], pairs[0]))

                        if test_1.testBool:
                            fitres_1 = self.fit_eg(sel_df, pairs[1], pairs[0],
                                                 startDate = sel_df.index[ilc+1-window],
                                                 endDate = sel_df.index[ilc],
                                                 plot = False).fitdf.copy()
                            resultDF_1.loc[df.index[ilc], pairs[0]] = fitres_1.loc['values', pairs[0]]
                            resultDF_1.loc[df.index[ilc], pairs[1]] = fitres_1.loc['values', pairs[1]]

                    if axis == 1:
                        res_list.append(resultDF_1)


                if axis == 2:
                    res = pd.concat([resultDF.copy(), resultDF_1.copy()], axis = 0, join = 'outer').copy()
                    res_list.append(res)

            res = pd.concat(res_list, axis=0, join='outer')
            resdf = res.reset_index()
            resdf = resdf.rename(columns={'index': 'period_end'}).copy()
            self.window_testdf = resdf


            if plot:
                # nan for p-values where the futures are not all T(1) order
                tdf = resdf[['pair', 'period_end', 'p-value']]
                tdf = tdf.pivot(columns = "pair", index = "period_end").astype(float)
                tdf.index = tdf.index.strftime("%Y/%m/%d")

                # plt.tight_layout()
                sns.set(rc = {'figure.figsize':(50, 70)})
                cmap = "YlGnBu_r"
                sns.heatmap(tdf, annot = True, cmap = cmap)
                plt.title(f"Cointegration p-values of every {window} days from {startDate} to {endDate} (nan values: not I(1) order series)")
                plt.ylabel(f"End Dates of Each {window} days window")
                plt.savefig('window_test_plot.png')
                plt.close()


        if dim == "multiple":

            for idx in range(len(future)): #Loop: let every future be endo for once

                endo = future[idx]
                exog = future[future != endo]

                print(f"endo: {endo}")

                resultDF = pd.DataFrame(columns=['StartDate', 'EndDate'])

                for ilc in range(window-1, len(df), window):
                    #loop: not rolling but divide the entire timeframe into chunks with window length

                    singleResult = self.test_eg(df, endo, exog,
                                             startDate=df.index[ilc+1-window],
                                             endDate=df.index[ilc],
                                             dim=dim).testdf
                    resultDF = pd.concat([singleResult, resultDF], axis = 0, join = 'outer')
                    resultDF.loc[df.index[ilc], 'StartDate'] = df.index[ilc+1-window]
                    resultDF.loc[df.index[ilc], 'EndDate'] = df.index[ilc]
                    resultDF.loc[df.index[ilc], 'endo'] = endo

                res_list.append(resultDF)

            res = pd.concat(res_list, axis = 0, join = 'outer').copy()
            resdf = res.reset_index()
            resdf = resdf.rename(columns = {'index': 'period_end'}).copy()
            self.window_testdf = resdf


            if plot:
                tdf = resdf[['endo', 'period_end', 'p-value']]
                tdf = tdf.pivot(columns='period_end', index = 'endo', values = 'p-value').astype(float)
                tdf.columns = tdf.columns.strftime("%Y/%m/%d")

                plt.tight_layout()
                sns.set(rc={'figure.figsize': (60, 10)})
                cmap = "YlGnBu_r"
                sns.heatmap(tdf, annot=True, cmap=cmap)
                plt.title(f"Multiple Futures cointegration p-values of every {window} from {startDate} to {endDate} (nan values: not I(1) order series)")
                plt.ylabel(f"End Dates of Each {window} window")
                plt.savefig('window_test_plot.png')
                plt.close()


        return self





    def lookback(self, full_df, endoCol, exogCol, dim = 'pair',
                 startDate = '1999-06-25', endDate = '2022-05-31',
                 minwindow = 26, maxwindow = 52, step = 1, plot = True):

        '''
        Function:
        *************
        This function calculate the lookback result of a pair or multiple futures in certain period with a list
        of user-input range of windows, and plot the heatmap of p-values of the tests

        User can customise the range of lookback windows with minwindow, maxwindow and step.

        Tips: Don't be greedy and plot every time and every window size, otherwise it will take forever to run,
              the plot would be crowded and unclear and REALLY long.
              Small steps everytime.


        Inputs:
        *************

        full_df: dataframe -> historical data
        endoCol: string -> column name of endo, must be 1d
        exogCol: string -> column names of exog, 1d when pair calculation, nd when multiple calculation
        dim: string -> default as 'pair'. It chooses from {'pair', 'multiple'}, which specifies the size of test portfolio
        startDate: string -> start date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        endDate: string -> end date of the desired time frame, e.g.'2010/01/01', '01/01/2010', '2010-01-01'
        minwindow: integer -> specify the minimum size of lookback window range
        maxwindow: integer -> specify the maxmize size of lookback window range
        step: integer -> specify the size of gap between two lookback window size
        plot: bool -> if plot, then True

        Returns:
        *************

        self

        Call Results:
        *************

        >>> self.lookback_testdf -> dataframe of lookback results



        '''

        full_df = self._tune_index(full_df)
        df = full_df.loc[startDate:endDate]

        resultDF = pd.DataFrame(columns = range(minwindow, maxwindow+1, step), index = df.index)

        for num_week in range(minwindow, maxwindow + 1, step):
            for ilc in range(num_week-1, len(df)):
                test = self.test_eg(df, endoCol, exogCol,
                                         startDate = df.index[ilc+1-num_week],
                                         endDate=df.index[ilc], dim = dim)

                singleResult = test.testdf.copy()
                resultDF.loc[df.index[ilc], num_week] = singleResult.loc[df.index[ilc], 'p-value']

                # if test.testBool:
                #
                #     fitres = self.fit_eg(df, endoCol, exogCol,
                #                          startDate = df.index[ilc+1-num_week],
                #                          endDate = df.index[ilc], plot = False).fitdf.copy()
                #     resultDF.loc[df.index[ilc], [endoCol, exogCol]] = fitres.loc['values', [endoCol, exogCol]]



        self.lookback_testdf = resultDF.reset_index()



        if plot:
            resultDF.index = resultDF.index.format()
            resultDF_p = resultDF.astype(float)

            sns.set(rc = {'figure.figsize':(100, 100)})
            cmap = "YlGnBu_r"
            sns.heatmap(resultDF_p, annot = True, cmap = cmap)
            plt.title(f"LookBack p-values")
            plt.ylabel("Dates")
            plt.xlabel("Number of Weeks")
            plt.savefig('lookback_test_plot.png')
            plt.close()

        return self


    def tests_comparison(self, df, columns, det_order, method = 'trace', select_order = -1, trend = 'c'):

        '''
        This function compares the cointegration relationship results and cointegration vector/matrix calculated from Johansen test and EG test.

        Input:

        df: dataframe -> full dataframe of testing data
        columns: list -> list of strings that are column names, which represent the futures we want to test for cointegration
        det_order: integer -> user choose from {-1, 0, 1}. -1: no deterministic terms; 0: constant term; 1: linear trend
        method: string -> default as 'trace'. It chooses from {'trace', 'maxeig'}, and specifies the method that we are using in select_coint_rank
        select_order: integer -> default as -1. -1: let the computer(code) selects the order based on VAR model; if > 0: the value of select_order is the user's chosen order
        trend: string -> default as 'c', which specifies the trend type for EG test and the johansen test


        Return:

        TestComparisonResults() -> stores the following values:

        a. johansen_res: boolean -> johansen test cointegration result
        b. eg_res: boolean -> eg test cointegration result
        c. johansen_vec: dataframe -> johansen test eigenvector that is cointegrated
        d. eg_vec: dataframe -> eg cointegration vector if cointegrated

        Access through .johansen_res, .eg_res, .johansen_vec, .eg_vec


        '''
        if len(columns) > 2:

            raise ValueError('Only Accepts Two Futures For Now.')

        sel_df = self._check_param(df[columns])

        test = self.johansen_test(sel_df, columns, det_order, method = method, select_order = select_order)


        if not isinstance(test, tuple):
            jr = test.res
            rank = test.rank
            evec = pd.DataFrame(test.evec.T, columns = columns, index = ['johansen']*rank)

        else:
            jr = test[0]
            evec = pd.DataFrame(columns = columns, index = ['johansen'])


        egr = self.test_eg(sel_df, columns[0], columns[1], trend = trend, startDate=sel_df.index[0], endDate = sel_df.index[-1]).testBool

        vlist = pd.DataFrame()
        # cointl = []
        if egr:
            for f in columns:
                rest = [fu for fu in columns if fu != f]
                egv = self.fit_eg(sel_df, f, rest,
                                  startDate = sel_df.index[0], endDate = sel_df.index[-1],
                                  plot = False).coefdf
                egv.index = ['EG']
                # cointl.append(eg_coint)

                vlist = pd.concat([vlist, egv], axis = 0, join = 'outer')


            # vlist = np.array(vlist).T

            # egvdf = pd.DataFrame(vlist, index = columns, columns = columns)

            # egc = pd.DataFrame(cointl, index = columns, columns = ['eg_coint'])'

        else:
            vlist = pd.DataFrame(columns = columns, index = ['EG'])

        return TestComparisonResults(jr, egr, evec, vlist)


class JohansenResults:


    def __init__(self, rank, eig, evec, res, columns):
        self.rank = rank
        self.eig = eig
        self.evec = evec
        self.res = res
        self.futures = columns

        resdf = pd.DataFrame(columns = ['rank', 'eigenvalue', 'res'] + self.futures, index = ['values'])
        resdf.loc['values', 'rank'] = rank
        resdf.loc['values', 'eigenvalue'] = eig
        resdf.loc['values', 'res'] = res
        resdf.loc['values', self.futures] = np.array(evec).T

        self.resdf = resdf



        # self.summary()

    def summary(self):


        resstr = '------------------------------------------------\n'

        resstr += f'Cointegrated: {self.res}\n'

        resstr += f'Number of Cointegration Relationships: {self.rank}\n'

        resstr += '------------------------------------------------\n'
        resstr += 'Selected Cointegrated Eigenvalue: \n'

        eigtab = pd.DataFrame(self.eig, columns = ['lambda'], index = range(1, self.rank+1))

        resstr += f'{eigtab} \n'

        resstr += '------------------------------------------------\n'

        resstr += 'Selected Stationary Cointegration Vectors: \n'

        evectab = pd.DataFrame(self.evec, columns = range(1, self.rank+1), index = self.futures)

        resstr += f'{evectab} \n'

        resstr += '------------------------------------------------\n'

        return resstr




class JohansenValidationResults:

    def __init__(self, rank, eig, evec, res, resdf):
        self.rank = rank
        self.eig = eig
        self.evec = evec
        self.res = res
        self.resdf = resdf

        # self.summary()


    def summary(self):

        resstr = '\n $$ Cointegration Validation Results Summary $$: \n'

        resstr += '------------------------------------------------ \n'

        resstr += str(self.resdf) + '\n'

        resstr += '------------------------------------------------ \n'

        return resstr


class TestComparisonResults:
    def __init__(self, jr, egr, jv, egv):
        self.johansen_res = jr
        self.eg_res = egr
        self.johansen_vec = jv
        self.eg_vec = egv

        # self.summary()

    def summary(self):

        resstr = '\n $$ Test Comparison Results Summary $$: \n'

        resstr += '------------------------------------------------ \n'

        resstr += 'Cointegration Test Results: \n'
        resdf = pd.DataFrame([self.johansen_res, self.eg_res], index = ['johansen', 'EG'], columns = ['Stationary'])

        resstr += str(resdf) + '\n'

        resstr += '------------------------------------------------\n'

        resstr += 'Cointegration Vectors: \n'

        resstr += str(self.johansen_vec) + '\n'
        resstr += '\n' + str(self.eg_vec) + '\n'

        return resstr


