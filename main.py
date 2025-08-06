import pandas as pd

# Read CSV, compute returns, and display results
def main():
    df = pd.read_csv('/Users/liqinye/Desktop/demo/synthetic_stock_data.csv', parse_dates=['Date'])
    df['Return'] = df['Close'].pct_change().fillna(0)
    print(df.head())

if __name__ == '__main__':
    main()
