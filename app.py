
import streamlit as st
import pandas as pd
import numpy as np
import os

HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
BALANCE_FILE = "sol_balance.txt"
MARTINGALE_FILE = "martingale_balance.txt"
INITIAL_BALANCE = 0.1
BET_AMOUNT = 0.01

@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df['multiplier'].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction','actual','correct'])

def save_results(df):
    df.to_csv(RESULTS_FILE, index=False)

def normalize_input(value):
    if value > 10:
        return value / 100
    return value

def compute_improved_confidence(data):
    # simplified for brevity
    data = np.array(data)
    base = np.mean(data > 2)
    trend = np.mean(data[-10:] > 2) if len(data)>=1 else 0.5
    return base, 1-base

def update_result(prediction, actual):
    df = load_results()
    correct = (prediction=="Above" and actual>2.0) or (prediction=="Under" and actual<=2.0)
    df = df.append({'prediction':prediction,'actual':actual,'correct':correct}, ignore_index=True)
    save_results(df)
    update_balance(prediction, actual)
    update_martingale(prediction, actual)

def get_balance_series():
    df = load_results()
    balance = INITIAL_BALANCE
    series = []
    for _, row in df.iterrows():
        if row['prediction']=="Above":
            balance += BET_AMOUNT if row['correct'] else -BET_AMOUNT
        series.append(balance)
    return series

def get_martingale_series():
    df = load_results()
    balance = INITIAL_BALANCE
    series = []
    streak = 0
    for _, row in df.iterrows():
        bet = BET_AMOUNT*(2**streak)
        if row['prediction']=="Above'":
            if row['correct']:
                balance += bet
                streak = 0
            else:
                balance -= bet
                streak += 1
        series.append(balance)
    return series

def get_balance():
    series = get_balance_series()
    return series[-1] if series else INITIAL_BALANCE

def get_martingale():
    series = get_martingale_series()
    return series[-1] if series else INITIAL_BALANCE

def main():
    st.title("Crash Predictor with SOL Balance Graphs")

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        st.session_state.history = load_csv(uploaded)
        save_history(st.session_state.history)
    new = st.text_input("Enter multiplier")
    if st.button("Add"):
        val = normalize_input(float(new))
        st.session_state.history.append(val)
        save_history(st.session_state.history)
        # make prediction
        above_conf, under_conf = compute_improved_confidence(st.session_state.history)
        pred = "Above" if above_conf>under_conf else "Under"
        update_result(pred, val)

    st.subheader("SOL Balances Over Time")
    bal_series = get_balance_series()
    mart_series = get_martingale_series()
    df = pd.DataFrame({
        'Flat Bet Balance': bal_series,
        'Martingale Balance': mart_series
    })
    st.line_chart(df)

if __name__=="__main__":
    main()
