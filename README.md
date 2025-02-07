# Stock-Price-Prediction-Using-LSTM
Predict future stock prices using deep learning techniques.

# 📈 Stock Price Prediction Using LSTM & XGBoost

## 🚀 Project Description  

### 🔹 Objective  
Predict future stock prices using machine learning and deep learning techniques, comparing the performance of XGBoost and LSTM models.  

### 🔹 Dataset  
- Collected historical stock price data from **Yahoo Finance** (e.g., Apple - AAPL).  
- Used features like **Open, High, Low, Close, and Volume** for analysis.  

### 🔹 Preprocessing & Feature Engineering  
- Checked for missing values and performed data cleaning.  
- Used **MinMaxScaler** for data normalization.  
- Conducted **stationarity tests** (ADF, KPSS) and seasonal decomposition.  

### 🔹 Modeling Approach  

#### 📌 XGBoost Model  
- Initially implemented **XGBoost**, a powerful gradient boosting algorithm.  
- Faced challenges with accuracy and time series pattern capture.  
- Due to limitations, explored a deep learning approach.  

#### 📌 LSTM (Long Short-Term Memory) Model  
- Implemented **LSTM**, a type of **Recurrent Neural Network (RNN)** designed for time series forecasting.  
- Used **Bidirectional LSTM** for improved pattern recognition.  
- Added **Dropout layers** to prevent overfitting.  
- Optimized using the **RMSprop optimizer** and **Early Stopping**.  

### 🔹 Evaluation Metrics  
📊 **Model Performance was evaluated using:**  
- ✅ **Mean Squared Error (MSE)**  
- ✅ **Mean Absolute Error (MAE)**  
- ✅ **R² Score**  

### 🔹 Findings  
- ❌ **XGBoost** struggled with sequential dependencies, leading to lower performance.  
- ✅ **LSTM** captured time series patterns better and provided more accurate predictions.  

### 🔹 Deployment  
- Integrated the trained **LSTM model** into a **Streamlit web app** for real-time stock price predictions.  
- Users can enter a stock ticker and get predicted future prices.  

---

