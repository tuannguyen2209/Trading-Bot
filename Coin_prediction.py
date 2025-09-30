import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ==============================================================================
# BƯỚC 0: THIẾT LẬP CÁC ĐƯỜNG DẪN
# ==============================================================================

# !!! THAY ĐỔI ĐƯỜNG DẪN NÀY ĐẾN THƯ MỤC CHỨA CÁC FILE CSV CỦA BẠN !!!
DATA_DIRECTORY = r'F:\Learning\Data Analysic\COIN_DATA' 

# Tên file để lưu kết quả cuối cùng
OUTPUT_FILE_PATH = r'F:\learning\coin_prediction_results_10_09_2025.csv'

# Các cột và hằng số
TIME_COLUMN = 'Date'
PRICE_COLUMN = 'Close'

# Danh sách để lưu kết quả của từng file
all_predictions = []

# ==============================================================================
# BẮT ĐẦU VÒNG LẶP QUA CÁC FILE
# ==============================================================================

# Lấy danh sách tất cả các file trong thư mục
try:
    all_files = os.listdir(DATA_DIRECTORY)
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy thư mục '{DATA_DIRECTORY}'. Vui lòng kiểm tra lại đường dẫn.")
    all_files = []

for filename in all_files:
    if filename.endswith(".csv"):
        # Lấy ticker từ tên file (ví dụ: 'BTC_data_2024_2025.csv' -> 'BTC')
        ticker = filename.split('_')[0].upper()
        file_path = os.path.join(DATA_DIRECTORY, filename)
        
        print(f"\n{'='*30}\n>>> BẮT ĐẦU XỬ LÝ: {ticker} ({filename}) <<<\n{'='*30}")

        try:
            # BƯỚC 1: Tải và làm sạch dữ liệu
            print("BƯỚC 1: Tải và làm sạch dữ liệu...")
            df = pd.read_csv(file_path, parse_dates=[TIME_COLUMN])
            df.sort_values(by=TIME_COLUMN, inplace=True)
            df.set_index(TIME_COLUMN, inplace=True)

            # BƯỚC 2: Feature Engineering (Thêm các chỉ báo kỹ thuật)
            print("BƯỚC 2: Tính toán các chỉ báo kỹ thuật...")
            delta = df[PRICE_COLUMN].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['SMA_10'] = df[PRICE_COLUMN].rolling(window=10).mean()
            df['SMA_30'] = df[PRICE_COLUMN].rolling(window=30).mean()

            # BƯỚC 3: Resample dữ liệu (nếu cần)
            # Dữ liệu đã là hàng ngày nên bước này không cần thiết nếu dữ liệu đã sạch.
            # Tuy nhiên, giữ lại để đảm bảo không có ngày nào bị thiếu.
            print("BƯỚC 3: Chuẩn hóa dữ liệu hàng ngày...")
            aggregation_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'RSI': 'last', 'SMA_10': 'last', 'SMA_30': 'last'}
            existing_cols = {col: rule for col, rule in aggregation_rules.items() if col in df.columns}
            df_daily = df.resample('D').agg(existing_cols)
            df_daily.dropna(inplace=True)

            if df_daily.empty:
                print(f"LỖI ({ticker}): Dữ liệu bị rỗng sau khi xử lý. Bỏ qua file này.")
                continue
            
            # Lưu lại dữ liệu của ngày cuối cùng để dùng cho việc dự đoán
            last_day_data_for_prediction = df_daily.iloc[-1:]

            # BƯỚC 4: Tạo biến mục tiêu (Target)
            print("BƯỚC 4: Tạo biến mục tiêu...")
            df_daily["Target"] = (df_daily[PRICE_COLUMN].shift(-1) - df_daily[PRICE_COLUMN]) / df_daily[PRICE_COLUMN] * 100
            df_daily.dropna(inplace=True)

            # BƯỚC 5: Chuẩn bị dữ liệu và Huấn luyện mô hình
            print("BƯỚC 5: Huấn luyện mô hình...")
            x_train = df_daily.drop(columns=["Target"])
            y_train = df_daily["Target"]
            
            if x_train.empty:
                print(f"LỖI ({ticker}): Không đủ dữ liệu để huấn luyện mô hình. Bỏ qua.")
                continue

            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=200, learning_rate=0.1)
            model.fit(x_train, y_train)

            # BƯỚC 6: Dự đoán
            print("BƯỚC 6: Đưa ra dự đoán...")
            prediction = model.predict(last_day_data_for_prediction)

            last_available_close_price = last_day_data_for_prediction[PRICE_COLUMN].values[0]
            predicted_price_change_percent = prediction[0]
            predicted_price_for_next_day = last_available_close_price * (1 + predicted_price_change_percent / 100)
            
            last_available_date = last_day_data_for_prediction.index[0]
            prediction_date = last_available_date + pd.Timedelta(days=1)

            print(f"\n--- KẾT QUẢ DỰ ĐOÁN CHO {ticker} ---")
            print(f"Dữ liệu của ngày {last_available_date.strftime('%Y-%m-%d')} được dùng để dự đoán.")
            # **THAY ĐỔI**: Hiển thị giá với 6 chữ số thập phân
            print(f"Giá đóng cửa cuối cùng ({last_available_date.strftime('%Y-%m-%d')}): {last_available_close_price:,.6f}")
            # **THAY ĐỔI**: Hiển thị % thay đổi với 4 chữ số thập phân
            print(f"Dự đoán % thay đổi cho ngày tiếp theo: {predicted_price_change_percent:.4f}%")
            # **THAY ĐỔI**: Hiển thị giá dự đoán với 6 chữ số thập phân
            print(f"=> Dự đoán giá đóng cửa cho ngày {prediction_date.strftime('%Y-%m-%d')}: {predicted_price_for_next_day:,.6f}")

            # Thêm kết quả vào danh sách tổng
            result = {
                'Ticker': ticker,
                'Last_Data_Date_Used': last_available_date.strftime('%Y-%m-%d'),
                'Prediction_Date': prediction_date.strftime('%Y-%m-%d'),
                'Last_Close_Price': last_available_close_price,
                'Predicted_Change_Percent': predicted_price_change_percent,
                'Predicted_Next_Close': predicted_price_for_next_day
            }
            all_predictions.append(result)

        except Exception as e:
            print(f"!!!!!! ĐÃ XẢY RA LỖI KHI XỬ LÝ FILE {filename}: {e} !!!!!!")
            print("--- Bỏ qua file này và tiếp tục ---")

# ==============================================================================
# BƯỚC 7: LƯU TẤT CẢ KẾT QUẢ RA FILE CSV
# ==============================================================================

if all_predictions:
    print(f"\n{'='*30}\n>>> HOÀN TẤT XỬ LÝ TẤT CẢ CÁC FILE <<<\n{'='*30}")
    results_df = pd.DataFrame(all_predictions)
    
    # Sắp xếp lại các cột để dễ đọc hơn
    results_df = results_df[[
        'Ticker', 
        'Last_Data_Date_Used', 
        'Prediction_Date', 
        'Last_Close_Price', 
        'Predicted_Change_Percent', 
        'Predicted_Next_Close'
    ]]
    
    # **THAY ĐỔI**: Tăng độ chính xác khi làm tròn số
    # Làm tròn giá đến 8 chữ số thập phân
    results_df['Last_Close_Price'] = results_df['Last_Close_Price'].round(8)
    # Làm tròn phần trăm thay đổi đến 6 chữ số thập phân
    results_df['Predicted_Change_Percent'] = results_df['Predicted_Change_Percent'].round(6)
    # Làm tròn giá dự đoán đến 8 chữ số thập phân
    results_df['Predicted_Next_Close'] = results_df['Predicted_Next_Close'].round(8)

    # Lưu file
    results_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
    print(f"Tất cả kết quả đã được lưu vào file: '{OUTPUT_FILE_PATH}'")
    
    # Cấu hình Pandas để hiển thị nhiều số thập phân hơn khi in ra màn hình
    pd.set_option('display.float_format', lambda x: '%.8f' % x)
    print("\nNội dung file kết quả:")
    print(results_df)
else:
    print("\nKhông có file nào được xử lý thành công. Không có kết quả để lưu.")
