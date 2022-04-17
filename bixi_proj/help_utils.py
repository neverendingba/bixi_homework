import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
import warnings
from typing import Tuple, List
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")
from sklearn.preprocessing import RobustScaler, PowerTransformer

"""
Help utils to handle basic timeseries data review and ARIM, SARIMAX operation 
"""
class ForecastMaker:
    def __init__(self):
        self.raw_df = self.get_data()
        self.pool_of_models = []
        self.num_test_tp = 14
        self.num_forecast_tp = 14

    #conncatinates all the data files into single df
    def get_data(self) -> pd.DataFrame:
        df_full = pd.DataFrame()
        for file in os.listdir("data"):
            file_loc = os.path.join("data", file)
            df_full = pd.read_csv(file_loc, parse_dates=True) if df_full.empty else pd.concat([df_full, pd.read_csv(file_loc, parse_dates=True)])
        df_full.shape
        df_full[["start_station_code", "end_station_code"]] = df_full[["start_station_code", "end_station_code"]].astype(int)
        return df_full

    #aggregates data on daily level and return only df for task 1
    def get_data_ex_one(self, special=False) -> pd.DataFrame:
        df = self.raw_df[(self.raw_df["start_station_code"] == 6184) & (self.raw_df["end_station_code"] == 6015)]
        df.index = pd.to_datetime(df["start_date"])
        df["duration_sec"] = df["duration_sec"].astype(float)
        df = df[["duration_sec"]]
        df = df.resample("D").count()
        df.columns = ["n_rides"]
        if special:
            df = df[df.index.month.isin([9, 8])]
        return df
    
    #aggregates data on daily level and return only df for task 2
    def get_data_ex_two(self, special=False) -> pd.DataFrame:
        df = self.raw_df
        df.index = pd.to_datetime(df["start_date"])
        df["duration_sec"] = df["duration_sec"].astype(float)
        df = df[["duration_sec"]]
        df = df.resample("D").count()
        df.columns = ["n_rides"]
        if special:
            df = df[df.index.month.isin([9, 8])]
        return df

    #aggregates data on daily level and return only df for task 3, member or non member
    def get_data_ex_three(self, member=False, special=False) -> pd.DataFrame:
        df = self.raw_df
        if member:
            df = df[df["is_member"] == 1]
        else:
            df = df[df["is_member"] == 0]
        df.index = pd.to_datetime(df["start_date"])
        df["duration_sec"] = df["duration_sec"].astype(float)
        df = df[["duration_sec"]]
        df = df.resample("D").count()
        df.columns = ["n_rides"]
        if special:
            df = df[df.index.month.isin([9, 8])]
        return df

    #checks whether data is stationary, outputs str with p value 
    @staticmethod
    def is_stationary(ts: pd.Series) -> None:
        test_results = adfuller(ts)
        if test_results[1] > 0.05:
            print(f"Data is not stationary, p value is {test_results[1]}")
        elif test_results[1] <= 0.01:
            print(f"Data is stationary, p value is {test_results[1]}")
        else:
            print(f"Data is arguably stationary, p value is {test_results[1]}")

    #evaluates passed model, creates forecast, compares to test data. return MSE value
    def eavluate_model(self, model, train, test, exog) -> float:
        try:
            results = model.fit()
        except: # may return error that except doesn't handle
            print ("BOBO")
            return 1e10

        y_hat = results.forecast(self.num_test_tp) if exog is None else results.forecast(self.num_test_tp, exog=exog)
        mse = mean_squared_error(test, y_hat)
        return mse

    # used by evaluate_and_return_best_forecast method
    # generates combinations for order and seasonal order for ARIMA/SARIMA 
    # accepts lists with all possible parameter options
    # yields set of parameters at the time
    def _get_param_combinations(self, *args) -> tuple:
        if len(args) == 3:
            all_comniations = list(itertools.product(*list(args)))
            for i in all_comniations:
                yield i
        elif len(args) == 7:
            orders = list(itertools.product(*list(args[:3])))
            seasonal_orders = list(itertools.product(*list(args[3:])))
            all_combinations = list(itertools.product(orders, seasonal_orders))
            for i in all_combinations:
                yield i


    # handles logic to evaluate and compare models models
    # internaly creates combinations for passed model, fits data, get MSE returns best model and it predictions
    def evaluate_and_return_best_forecast(self, model_type, df, orders, task, exog=None) -> dict:
        train = df.iloc[: -self.num_test_tp]
        test = df.iloc[-self.num_test_tp :]
        best_mse = 1e10
        print(len(orders))
        print(orders)
        for i in self._get_param_combinations(*orders):
            if len(orders) == 3:
                model = model_type(train["n_rides"], order=i)
                mse = self.eavluate_model(model, train, test, exog=None)
                if mse < best_mse:
                    print(i, mse)
                    best_mse = mse
                    best_model = model_type(df["n_rides"], order=i)
                    result = best_model.fit()
                    start = df.shape[0]
                    end = df.shape[0] + 6
                    np_start = datetime(2017, 9, 1)
                    np_end = datetime(2017, 9, 8)
                    best_forecast = result.predict(start=start, end=end)
                    index_dates = np.arange(np_start, np_end, 1, dtype="datetime64[D]")
                    best_forecast.index = index_dates
            elif len(orders) == 7:
                try:
                    print("i", i)
                    if exog is not None:
                        model = model_type(train["n_rides"], order=i[0], seasonal_order=i[1], exog=exog[: -self.num_test_tp])
                        mse = self.eavluate_model(model, train, test, exog=pd.Series([8] * 14))
                    else:
                        model = model_type(train["n_rides"], order=i[0], seasonal_order=i[1])
                        mse = self.eavluate_model(model, train, test, exog=None)

                    if mse < best_mse:
                        print(i, mse)
                        best_mse = mse
                        best_model = model_type(df["n_rides"], order=i[0], seasonal_order=i[1])
                        result = best_model.fit()
                        #handling start and end as int as breaking when passed as datetimes
                        if task == 2:
                            start = df.shape[0] + 3
                            end = df.shape[0] + 9
                            np_start = datetime(2017, 9, 4)
                            np_end = datetime(2017, 9, 11)
                        else:
                            start = df.shape[0]
                            end = df.shape[0] + 6
                            np_start = datetime(2017, 9, 1)
                            np_end = datetime(2017, 9, 8)
                        if exog is None:
                            best_forecast = result.predict(start=start, end=end)
                        else:
                            best_forecast = result.predict(start=start, end=end, exog=pd.Series([9] * 7))
                        index_dates = np.arange(np_start, np_end, 1, dtype="datetime64[D]")
                        best_forecast.index = index_dates

                except ValueError:
                    print("Error on ", i)
        return {"best_model": best_model, "best_mse": best_mse, "best_forecast": best_forecast}

    #plots forecast
    def plot_forecast(self, df, best_results) -> None:
        title = "Last 30 days of data and Forecast"
        ylabel = " Num of rides"
        xlabel = ""

        ax = df["n_rides"][-30:].plot(legend=True, figsize=(12, 6), title=title)
        best_results["best_forecast"].plot(legend=True)
        ax.autoscale(axis="x", tight=True)
        ax.set(xlabel=xlabel, ylabel=ylabel)