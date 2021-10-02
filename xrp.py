#author: dani atalla
#XRP price predictor

#the code works i just wanted to put in some data to play with like if you should buy or not but its giving me a float 64 and i have no time to fix it :(


import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np



SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "coin_XRP"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# def preprocess_df(df):
#     df = df.drop('future', 1)
#     df = df.reset_index()
#
#     for col in df.columns:
#         if col != "target":
#             df[col] = df[col].pct_change()
#             df.dropna(inplace=True)
#             df[col] = preprocessing.scale(df[col].values)
#
#     df.dropna(inplace=True)
#
#     sequential_data = []
#     prev_days = deque(maxLen=SEQ_LEN)

    for i in df.values:
     prev_days.append([n for n in i[:-1]])
    if len(prev_days) ==  SEQ_LEN:
        sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
    return np.array (x), y


    # print(df.head())
    # for c in df.coluns:
    #     print(c)
# df = pd.DataFrame()

ratios = ["coin_XRP"]
for ratio in ratios:
    dataset = f"C:/Users/user/Desktop/predict/crypto_data/{ratio}.csv"

    df = pd.read_csv(dataset, names=["Date", "Low", "High", "Open", "Close", "Volume"])


    df.rename(columns={"Close": f"{ratio}_Close", "Volume": f"{ratio}_Volume"}, inplace=True)
    df.set_index("Date", inplace=True)
    df = df[[f"{ratio}_Close", f"{ratio}_Volume"]]


    # if len(main_df) == 0:
    # main_df = df
    # else
    # main_df = main_df.join(df)
df['future'] = df[f"{RATIO_TO_PREDICT}_Close"].shift(-FUTURE_PERIOD_PREDICT)
df['target'] = list(map(classify, df[f"{RATIO_TO_PREDICT}_Close"], df["future"]))
# print(df[[f"{RATIO_TO_PREDICT}_Close", "future", "target"]].head(20))
times = sorted(df.index.values)
last_Spct = times[-int(0.05*len(times))]

validation_df = df[(df.index >= last_Spct)]
df= df[(df.index < last_Spct)]

# preprocess_df(df)
# train_x, train_y = preprocess_df(df)
# validation_x, validation_y = preprocess_df(validation_df)

# print(f"train_data: {len(train_x)} validation: {len(validation_x)}")
# print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
# print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
print(df.head(20))