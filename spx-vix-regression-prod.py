# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report

def binarizer(value):
    
    if value > 0: return 1 
    else: return 0

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

trading_dates = calendar.schedule(start_date = "2023-05-01", end_date = (datetime.today() + timedelta(days=3)).strftime("%Y-%m-%d")).index.strftime("%Y-%m-%d").values
full_trading_dates = calendar.schedule(start_date = "2023-05-01", end_date = (datetime.today() + timedelta(days=30)).strftime("%Y-%m-%d")).index.strftime("%Y-%m-%d").values

# =============================================================================
# Dataset Building
# =============================================================================

index_ticker = "I:VIX"

vix_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{index_ticker}/range/1/day/{trading_dates[0]}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
vix_data.index = pd.to_datetime(vix_data.index, unit="ms", utc=True).tz_convert("America/New_York")
vix_data["date"] = vix_data.index.strftime("%Y-%m-%d")

target_ticker = "SPY"

big_target_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{target_ticker}/range/1/day/{trading_dates[0]}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
big_target_data.index = pd.to_datetime(big_target_data.index, unit="ms", utc=True).tz_convert("America/New_York")
big_target_data["date"] = big_target_data.index.strftime("%Y-%m-%d")

full_feature_data_list = []

# date = trading_dates[1:-1][-1]
for date in trading_dates[1:-1]:
    
    try:
    
        prior_day = trading_dates[trading_dates < date][0]
        next_day = trading_dates[trading_dates > date][0]
            
        big_ticker_data = vix_data[vix_data["date"] <= date].copy().tail(100)
        big_ticker_data["1_mo_avg"] = big_ticker_data["c"].rolling(window=20).mean()
        big_ticker_data["3_mo_avg"] = big_ticker_data["c"].rolling(window=60).mean()
        big_ticker_data["regime"] = big_ticker_data.apply(lambda row: 1 if (row['1_mo_avg'] > row['3_mo_avg']) else 0, axis=1)
        
        regime = big_ticker_data["regime"].iloc[-1]
        
        ticker_data = vix_data[(vix_data["date"] >= prior_day) & (vix_data["date"] <= date)].copy()
        ticker_data["pct_change"] = round(ticker_data["c"].pct_change()*100,2)
        
        daily_return = ticker_data["pct_change"].iloc[-1]
        daily_value = ticker_data["c"].iloc[-1]
        implied_move = round((daily_value / np.sqrt(252)), 2)
        
        feature_data = pd.DataFrame([{"regime": regime, "daily_return": daily_return, "daily_value": daily_value,
                                             "daily_implied_move": implied_move}])
        
        oos_data = big_target_data[(big_target_data["date"] >= date) & (big_target_data["date"] <= next_day)].copy()
        oos_data["pct_change"] = round(oos_data["c"].pct_change()*100, 2)
    
        next_day_return = oos_data["pct_change"].iloc[-1]
        
        feature_data["next_day_return"] = next_day_return
        feature_data["date"] = date     
        
        full_feature_data_list.append(feature_data)
    except Exception:
        continue
        
full_feature_data = pd.concat(full_feature_data_list)

# =============================================================================
# Forward prediction for the next trading day
# =============================================================================

trading_date = full_feature_data["date"].iloc[-1]
next_day = full_trading_dates[full_trading_dates > trading_date][0]

historical_data = full_feature_data[full_feature_data["date"] < trading_date].copy().tail(252)

direction_features = ["daily_return", "daily_value"]
    
X_Classification = historical_data[direction_features].values
Y_Classification = historical_data["next_day_return"].apply(binarizer).values

Direction_Model = LogisticRegression().fit(X_Classification, Y_Classification)

out_of_sample_data = full_feature_data[full_feature_data["date"] == trading_date].copy()

X_Test_Classification = out_of_sample_data[direction_features].values

direction_prediction = Direction_Model.predict(X_Test_Classification)
direction_prediction_proba = Direction_Model.predict_proba(X_Test_Classification)
direction_prediction_proba = [direction_prediction_proba[0][direction_prediction[0]]]

prediction_data = pd.DataFrame({"date": trading_date, "predicted_direction": direction_prediction, "proba": direction_prediction_proba})

# =============================================================================
# Model Output
# =============================================================================

if direction_prediction == 1:
    print(f"\nModel expects S&P 500 to go *up* on the next trading day, {next_day}, with a {round(direction_prediction_proba[0]*100, 2)}% probability.")
elif direction_prediction == 0:
    print(f"\nModel expects S&P 500 to go *down* on the next trading day,{next_day}, with a {round((1-direction_prediction_proba[0])*100, 2)}% probability.")