import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from flask import Flask, request, jsonify

app = Flask(__name__)

# Function to fetch stock data using yfinance API
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Function to optimize portfolio using Mean-Variance Optimization
def optimize_portfolio(stock_data):
    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(stock_data)
    S = risk_models.sample_cov(stock_data)
    
    # Optimize portfolio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()  # Maximize Sharpe Ratio
    cleaned_weights = ef.clean_weights()
    
    return cleaned_weights, ef.portfolio_performance(verbose=True)

# API route for portfolio optimization
@app.route('/optimize', methods=['POST'])
def optimize():
    request_data = request.json
    tickers = request_data['tickers']
    start_date = request_data['start_date']
    end_date = request_data['end_date']
    
    # Fetch stock data
    stock_data = get_stock_data(tickers, start_date, end_date)
    
    # Perform optimization
    weights, performance = optimize_portfolio(stock_data)
    
    return jsonify({
        "weights": weights,
        "performance": performance
    })

if __name__ == "__main__":
    app.run(debug=True)
