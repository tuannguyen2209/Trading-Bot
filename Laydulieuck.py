import pandas as pd
import os
import yfinance as yf # Import thư viện yfinance và đặt bí danh là yf
vn30_stocks = [
    "ACB", "BID", "CTG", "HDB", "HPG", "MBB", "MSN", "MWG", "PNJ", 
    "POW", "SAB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIC", 
    "VJC", "VNM", "VPB", "FPT", "GAS", "GVR", "KDH", "NVL", "PDR", 
    "PLX", "VIB", "VRE"
]

for stock in vn30_stocks:
    dat = yf.Ticker(f"{stock}.VN")
    df = dat.history(period="5y", interval="1d")
    print(df)
    file_path = os.path.join('F:\\Learning\\Data Analysic\\CKVN', f'{stock}_2020_2025.csv')
    df.to_csv(file_path)
    print(f"\nĐã lưu dữ liệu thành công vào: {file_path}")