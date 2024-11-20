import pandas as pd
import yfinance as yf

def get_sp500_tickers():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].tolist()

def get_additional_etfs():
    additional_etfs = [
        'SPY', 'XOP', 'XLE', 'USO', 'DBC', 'GLD', 'JETS', 'PEJ',
        'VNQ', 'IYR', 'HYG', 'JNK', 'ANGL', 'DVY', 'VYM', 'SDIV', 'EMB', 'HYEM'
    ]
    return additional_etfs

def fetch_and_save_data():
    tickers = get_sp500_tickers() + get_additional_etfs()
    data = yf.download(tickers, start="2015-01-01", end="2024-12-31", progress=False)['Adj Close']
    # data.to_parquet("sp500_etf.parquet")
    data.to_pickle("sp500_etf.pkl")
    
if __name__ == "__main__":
    fetch_and_save_data()
