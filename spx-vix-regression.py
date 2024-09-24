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

vix_data = pd.read_csv("VIX_History.csv").rename(columns={"DATE": "date", "CLOSE": "c"})
vix_data["date"] = pd.to_datetime(vix_data["date"]).dt.strftime("%Y-%m-%d")

trading_dates = calendar.schedule(start_date = "2019-01-01", end_date = vix_data["date"].iloc[-1]).index.strftime("%Y-%m-%d").values

# =============================================================================
# Dataset Building
# =============================================================================

target_ticker = "SPY"

big_target_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{target_ticker}/range/1/day/{trading_dates[0]}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
big_target_data.index = pd.to_datetime(big_target_data.index, unit="ms", utc=True).tz_convert("America/New_York")
big_target_data["date"] = big_target_data.index.strftime("%Y-%m-%d")

full_feature_data_list = []

# date = trading_dates[1:-1][-1]
for date in trading_dates[1:-1]:
    
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
    next_day_return_open = round(((oos_data["o"].iloc[-1] - oos_data["c"].iloc[0]) / oos_data["c"].iloc[0])*100, 2)
    
    feature_data["next_day_return"] = next_day_return
    feature_data["next_day_return_open"] = next_day_return_open     
    feature_data["date"] = date     
    
    full_feature_data_list.append(feature_data)
        
full_feature_data = pd.concat(full_feature_data_list)

# =============================================================================
# Model Testing | Walk-Forward Approach
# =============================================================================

backtest_dates = calendar.schedule(start_date = "2024-01-01", end_date = trading_dates[-1]).index.strftime("%Y-%m-%d").values

full_prediction_list = []
times = []

# trading_date = backtest_dates[:-1][0]
for trading_date in backtest_dates[:-1]:
    
    start_time = datetime.now()
    
    historical_data = full_feature_data[full_feature_data["date"] < trading_date].copy().tail(252)
    next_day = backtest_dates[backtest_dates > trading_date][0]
    
    direction_features = ["daily_return", "daily_value"]
        
    X_Classification = historical_data[direction_features].values
    Y_Classification = historical_data["next_day_return"].apply(binarizer).values
    
    Direction_Model = LogisticRegression().fit(X_Classification, Y_Classification)
    
    out_of_sample_data = full_feature_data[full_feature_data["date"] == trading_date].copy()
    
    X_Test_Classification = out_of_sample_data[direction_features].values
    
    direction_prediction = Direction_Model.predict(X_Test_Classification)
    direction_prediction_proba = Direction_Model.predict_proba(X_Test_Classification)[:, 1]
    # direction_prediction_proba = [direction_prediction_proba[0][direction_prediction[0]]]
    
    actual_direction = binarizer(out_of_sample_data["next_day_return"].iloc[0])
    actual_vol = abs(out_of_sample_data["next_day_return"].iloc[0])
    
    prediction_data = pd.DataFrame({"date": trading_date, "predicted_direction": direction_prediction, "1_proba": direction_prediction_proba,
                                    "actual_direction": actual_direction, "actual_vol": actual_vol})
                                    
    full_prediction_list.append(prediction_data)
    
full_prediction_data = pd.concat(full_prediction_list)

# =============================================================================
# Model Analysis Calcs.
# =============================================================================

preds = full_prediction_data["predicted_direction"].values
probas = full_prediction_data["1_proba"].values
actuals = full_prediction_data["actual_direction"].values

print("\nClassification Report:")
print(classification_report(actuals, preds))

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(actuals, probas)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8), dpi = 600)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('VIX Features to SPX Direction –– ROC Curve')
plt.legend(loc="lower right")
plt.show()