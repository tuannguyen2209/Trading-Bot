import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ==============================================================================
# BƯỚC 0: THIẾT LẬP CÁC ĐƯỜNG DẪN VÀ HÀM
# ==============================================================================

# !!! THAY ĐỔI ĐƯỜNG DẪN NÀY ĐẾN THƯ MỤC CHỨA CÁC FILE CSV CỦA BẠN !!!
DATA_DIRECTORY = r'F:\Learning\Data Analysic\CKVN' 

# Tên file để lưu kết quả cuối cùng
OUTPUT_FILE_PATH = r'F:\learning\prediction_results_18_09_2025_adjusted.csv'

# Các cột và hằng số
TIME_COLUMN = 'Date'
PRICE_COLUMN = 'Close'

# ==============================================================================
# HÀM MỚI: LÀM TRÒN GIÁ THEO LUẬT SÀN HOSE
# ==============================================================================
def adjust_price_for_hose(price):
    """
    Hàm này nhận một mức giá và làm tròn nó theo quy tắc bước giá (tick size) của sàn HOSE.
    """
    if price < 10000:
        # Làm tròn đến bội số gần nhất của 10
        return round(price / 10) * 10
    elif 10000 <= price <= 49950:
        # Làm tròn đến bội số gần nhất của 50
        return round(price / 50) * 50
    else: # price > 49950
        # Làm tròn đến bội số gần nhất của 100
        return round(price / 100) * 100

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
        # Lấy ticker từ tên file (ví dụ: 'VRE_data.csv' -> 'VRE')
        ticker = filename.split('_')[0].split('.')[0].upper()
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
            print("BƯỚC 3: Gom dữ liệu sang hàng ngày...")
            aggregation_rules = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'RSI': 'last', 'SMA_10': 'last', 'SMA_30': 'last'}
            existing_cols = {col: rule for col, rule in aggregation_rules.items() if col in df.columns}
            df_daily = df.resample('D').agg(existing_cols)
            df_daily.dropna(inplace=True)

            if df_daily.empty:
                print(f"LỖI ({ticker}): Dữ liệu bị rỗng sau khi xử lý. Bỏ qua file này.")
                continue
            
            # Lưu lại dữ liệu của ngày cuối cùng (sau khi đã xử lý) để dùng cho việc dự đoán
            last_day_data_for_prediction = df_daily.iloc[-1:]

            # BƯỚC 4: Tạo biến mục tiêu (Target)
            print("BƯỚC 4: Tạo biến mục tiêu...")
            df_daily["Target"] = (df_daily[PRICE_COLUMN].shift(-1) - df_daily[PRICE_COLUMN]) / df_daily[PRICE_COLUMN] * 100
            df_daily.dropna(inplace=True)

            # BƯỚC 5: Chuẩn bị dữ liệu và Huấn luyện mô hình
            print("BƯỚC 5: Huấn luyện mô hình...")
            x_train = df_daily.drop(columns=["Target"])
            y_train = df_daily["Target"]

            model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=200, learning_rate=0.1)
            model.fit(x_train, y_train)

            # BƯỚC 6: Dự đoán
            print("BƯỚC 6: Đưa ra dự đoán...")
            # Dùng dữ liệu của ngày cuối cùng có trong file để làm input cho mô hình
            prediction = model.predict(last_day_data_for_prediction)

            last_available_close_price = last_day_data_for_prediction[PRICE_COLUMN].values[0]
            predicted_price_change_percent = prediction[0]
            
            # *** THAY ĐỔI BẮT ĐẦU TỪ ĐÂY ***
            # 1. Tính toán giá dự đoán thô (chưa làm tròn)
            raw_predicted_price = last_available_close_price * (1 + predicted_price_change_percent / 100)
            
            # 2. Áp dụng hàm làm tròn theo luật HOSE
            adjusted_predicted_price = adjust_price_for_hose(raw_predicted_price)
            # *** KẾT THÚC THAY ĐỔI ***

            last_available_date = last_day_data_for_prediction.index[0]
            # NGÀY DỰ ĐOÁN = Ngày cuối cùng có dữ liệu + 1 ngày
            prediction_date = last_available_date + pd.Timedelta(days=1)

            print(f"\n--- KẾT QUẢ DỰ ĐOÁN CHO {ticker} ---")
            print(f"Dữ liệu của ngày {last_available_date.strftime('%Y-%m-%d')} được dùng để dự đoán.")
            print(f"Giá đóng cửa cuối cùng ({last_available_date.strftime('%Y-%m-%d')}): {last_available_close_price:,.0f} VNĐ")
            print(f"Dự đoán % thay đổi cho ngày tiếp theo: {predicted_price_change_percent:.2f}%")
            print(f"Giá dự đoán (chưa làm tròn): {raw_predicted_price:,.2f} VNĐ")
            print(f"=> ✅ Dự đoán giá đóng cửa cho ngày {prediction_date.strftime('%Y-%m-%d')} (đã làm tròn): {adjusted_predicted_price:,.0f} VNĐ")

            # Thêm kết quả vào danh sách tổng (sử dụng giá đã làm tròn)
            result = {
                'Ticker': ticker,
                'Last_Data_Date_Used': last_available_date.strftime('%Y-%m-%d'),
                'Prediction_Date': prediction_date.strftime('%Y-%m-%d'),
                'Last_Close_Price': last_available_close_price,
                'Predicted_Change_Percent': predicted_price_change_percent,
                'Raw_Predicted_Close': raw_predicted_price, # Thêm cột giá thô để so sánh
                'Adjusted_Predicted_Close': adjusted_predicted_price # Cột giá đã làm tròn
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
        'Raw_Predicted_Close',
        'Adjusted_Predicted_Close'
    ]]
    
    # *** THAY ĐỔI ĐỊNH DẠNG ***
    # Định dạng lại cột số cho dễ đọc
    results_df['Predicted_Change_Percent'] = results_df['Predicted_Change_Percent'].round(4)
    results_df['Raw_Predicted_Close'] = results_df['Raw_Predicted_Close'].round(2)
    # Cột giá đã làm tròn không cần .round() vì nó đã là số nguyên
    
    # Lưu file
    results_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
    print(f"Tất cả kết quả đã được lưu vào file: '{OUTPUT_FILE_PATH}'")
    print("\nNội dung file kết quả:")
    print(results_df)
else:
    print("\nKhông có file nào được xử lý thành công. Không có kết quả để lưu.")