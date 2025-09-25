import ccxt
import numpy as np
import json
import pandas as pd # Thư viện mạnh mẽ để xử lý dữ liệu

# --- Cấu hình Bot và Backtest ---
# Giữ nguyên cấu hình bot và thêm cấu hình cho backtest
CONFIG = {
    'symbol': 'ETH/USDT',
    'exchange_name': 'binance',
    'grid': {
        'upper_price': 4200,
        'lower_price': 3700,
        'levels': 50,
    },
    'trade': {
        'fee_percent': 0.1,
        'quantity_per_grid': 0.0001
    },
    'backtest': {
        'timeframe': '1h', # Khung thời gian của nến (1h, 4h, 1d)
        'since': '2025-08-01T00:00:00Z' # Bắt đầu lấy dữ liệu từ ngày này
    }
}

# --- Khởi tạo kết nối sàn ---
exchange = getattr(ccxt, CONFIG['exchange_name'])()

def fetch_historical_data(symbol, timeframe, since):
    """
    Hàm mới: Tải dữ liệu OHLCV lịch sử từ sàn giao dịch.
    """
    try:
        print(f"Đang tải dữ liệu lịch sử cho {symbol} từ {since}...")
        # Lấy dữ liệu nến: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, exchange.parse8601(since), limit=1000)
        if not ohlcv:
            print("Không có dữ liệu trả về.")
            return None
        # Chuyển thành DataFrame của Pandas để dễ xử lý
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"Tải thành công {len(df)} cây nến {timeframe}.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

def setup_grid():
    """Tạo các mức giá trong lưới."""
    return np.linspace(CONFIG['grid']['lower_price'], CONFIG['grid']['upper_price'], CONFIG['grid']['levels'])

def run_backtest():
    """
    Hàm chính được thay đổi: Chạy logic bot trên dữ liệu lịch sử.
    """
    print("🚀 Bắt đầu quá trình Backtest...")
    
    # Bước 1: Lấy dữ liệu
    historical_data = fetch_historical_data(
        CONFIG['symbol'],
        CONFIG['backtest']['timeframe'],
        CONFIG['backtest']['since']
    )
    if historical_data is None:
        return

    # Khởi tạo các thông số cho backtest
    grid_levels = setup_grid()
    state = {'open_buy_orders': {}}
    total_profit = 0
    completed_trades = 0

    print("\n" + "="*50)
    print("BẮT ĐẦU MÔ PHỎNG GIAO DỊCH")
    print("="*50 + "\n")

    # Bước 2: Vòng lặp chính duyệt qua từng cây nến
    for index, row in historical_data.iterrows():
        current_price = row['close'] # Sử dụng giá đóng cửa của nến để mô phỏng
        current_time = row['timestamp']

        # Bước 3: Áp dụng logic bot (tương tự như code cũ)
        # --- Logic Bán ---
        for buy_price_str in list(state['open_buy_orders'].keys()):
            buy_price = float(buy_price_str)
            buy_order_info = state['open_buy_orders'][buy_price_str]
            sell_target_price = grid_levels[buy_order_info['grid_index'] + 1]

            # Trong backtest, ta giả định lệnh khớp nếu giá cao nhất của nến vượt qua mức bán
            if row['high'] >= sell_target_price:
                buy_cost = buy_price * CONFIG['trade']['quantity_per_grid']
                sell_revenue = sell_target_price * CONFIG['trade']['quantity_per_grid']
                buy_fee = buy_cost * (CONFIG['trade']['fee_percent'] / 100)
                sell_fee = sell_revenue * (CONFIG['trade']['fee_percent'] / 100)
                profit = (sell_revenue - buy_cost) - (buy_fee + sell_fee)
                
                total_profit += profit
                completed_trades += 1

                print(f"✅ [{current_time}] CHỐT LỜI: Mua {buy_price:.2f} -> Bán {sell_target_price:.2f} | Lợi nhuận: {profit:.4f} USDT")
                
                del state['open_buy_orders'][buy_price_str]

        # --- Logic Mua ---
        for i, level_price in enumerate(grid_levels):
            if i == len(grid_levels) - 1: continue

            level_price_str = f"{level_price:.2f}"
            
            # Giả định lệnh mua khớp nếu giá thấp nhất của nến chạm hoặc xuống dưới mức mua
            if row['low'] <= level_price and level_price_str not in state['open_buy_orders']:
                print(f"🔵 [{current_time}] ĐẶT LỆNH MUA tại {level_price:.2f} USDT")
                
                state['open_buy_orders'][level_price_str] = {
                    'grid_index': i,
                    'quantity': CONFIG['trade']['quantity_per_grid']
                }

    # Bước 4: Báo cáo kết quả
    print("\n" + "="*50)
    print("📊 BÁO CÁO KẾT QUẢ BACKTEST")
    print("="*50)
    print(f"Khoảng thời gian: {historical_data['timestamp'].iloc[0]} -> {historical_data['timestamp'].iloc[-1]}")
    print(f"Khung thời gian: {CONFIG['backtest']['timeframe']}")
    print("-" * 20)
    print(f"Tổng số giao dịch hoàn thành: {completed_trades}")
    print(f"Tổng lợi nhuận ròng: {total_profit:.4f} USDT")
    
    # Tính toán lợi nhuận nếu giữ nguyên không trade (Buy and Hold)
    initial_price = historical_data['open'].iloc[0]
    final_price = historical_data['close'].iloc[-1]
    buy_and_hold_return = ((final_price - initial_price) / initial_price) * 100
    print(f"Tỷ suất lợi nhuận nếu 'Buy and Hold': {buy_and_hold_return:.2f}%")
    
    print("\nLưu ý: Kết quả này chưa tính đến trượt giá và các yếu tố thị trường phức tạp khác.")


if __name__ == '__main__':
    run_backtest()