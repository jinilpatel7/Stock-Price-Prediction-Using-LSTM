import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Loading trained LSTM model
MODEL_PATH = "lstm_stock_price_model.keras"
model = load_model(MODEL_PATH)


TOP_10_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'IBM']
st.title("📈 Stock Price Prediction App")
st.markdown("### Predict stock prices using an LSTM model trained on historical data.")
st.markdown("**Suggested Stocks:**")
st.write(", ".join(TOP_10_STOCKS))


ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

# Button to fetch stock data
if st.button("Get Stock Data"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=1440)).strftime('%Y-%m-%d')

    st.write(f"Fetching data for **{ticker}** from {start_date} to {end_date}...")

    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        st.success(f"Successfully loaded data for {ticker}")

        # Plotting original stock prices
        st.subheader("Original Stock Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index, data['Close'], label=f"{ticker} Closing Prices", color='blue')
        ax.set_title(f"{ticker} Stock Price Chart")
        ax.legend()
        st.pyplot(fig)

        # Storing data for future processing
        st.session_state["stock_data"] = data

    except Exception as e:
        st.error(f"Error loading stock data: {e}")

# Button for forecasting
if "stock_data" in st.session_state:
    if st.button("Predict Future Prices"):
        data = st.session_state["stock_data"]

        # Preprocess data for LSTM
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data[['Close']])

        # Create sequences for LSTM
        sequence_length = 250
        def create_sequences(data, seq_length):
            sequences, targets = [], []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i+seq_length])
                targets.append(data[i+seq_length])
            return np.array(sequences), np.array(targets)

        X_lstm, y_lstm = create_sequences(data_scaled, sequence_length)
        train_size = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
        y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

        # Predictions using LSTM
        st.subheader("LSTM Stock Price Prediction")
        predictions = model.predict(X_test_lstm)
        predictions_rescaled = scaler.inverse_transform(predictions)
        y_test_rescaled = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

        # Plot actual vs predicted prices
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(data.index[train_size+sequence_length:], y_test_rescaled, label='Actual Prices', color='blue')
        ax2.plot(data.index[train_size+sequence_length:], predictions_rescaled, label='LSTM Predictions', color='red')
        ax2.set_title(f"{ticker} - Actual vs LSTM Predictions")
        ax2.legend()
        st.pyplot(fig2)

        # Forecast future prices
        st.subheader("LSTM Future Stock Price Forecast")

        last_sequence = X_lstm[-1].reshape(1, sequence_length, 1)
        future_predictions = []
        for _ in range(20):  # Predicting next 20 days
            next_pred = model.predict(last_sequence)
            future_predictions.append(next_pred[0][0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1, 0] = next_pred[0][0] + np.random.normal(0, 0.05)

        # Convert predictions back to original scale
        future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Generate future dates
        dates_future = [data.index[-1] + timedelta(days=i) for i in range(1, 21)]

        # Plot future predictions
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(data.index[train_size+sequence_length:], y_test_rescaled, label='Actual Prices', color='blue')
        ax3.plot(data.index[train_size+sequence_length:], predictions_rescaled, label='LSTM Predictions', color='red')
        ax3.plot(dates_future, future_predictions_rescaled, label="LSTM Future Forecast", color='orange', linestyle='dashed')

        ax3.set_title(f"{ticker} - LSTM Future Price Forecast")
        ax3.legend()
        st.pyplot(fig3)
        
        # **Display Table of Future Predictions**
        st.subheader("Future Stock Price Predictions Table")
        future_df = pd.DataFrame({"Date": dates_future, "Predicted Price": future_predictions_rescaled.flatten()})
        st.dataframe(future_df.style.format({"Predicted Price": "{:.2f}"}))  # Format to 2 decimal places

        st.success("Stock price forecast complete!")
