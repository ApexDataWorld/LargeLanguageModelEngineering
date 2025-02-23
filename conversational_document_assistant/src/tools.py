from langchain.tools import Tool
import requests

def fetch_stock_price(symbol):
    """Fetch real-time stock price from Alpha Vantage API."""
    API_KEY = "your_alpha_vantage_api_key"
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("Global Quote", {}).get("05. price", "Price not found.")
    return "Error fetching stock price."

def sentiment_analysis(text):
    """Placeholder for sentiment analysis logic."""
    return "Positive" if "good" in text else "Negative"

stock_price_tool = Tool(
    name="Stock Price Fetcher",
    func=fetch_stock_price,
    description="Fetch real-time stock prices for a given symbol."
)

sentiment_tool = Tool(
    name="Sentiment Analyzer",
    func=sentiment_analysis,
    description="Perform sentiment analysis on input text."
)
