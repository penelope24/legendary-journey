"""
批量预测
对于config/stocks.py中hs300_list列表内的每一个股票代码，进行src/predict中的预测，并保存结果。
保存路径: 保存路径为output/hs_300
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import traceback

# 导入自定义模块
from src.predict import StockPredictor, Strategy
from src.data import StockInfoFetcher, Period
from config.stocks import hs300_list

# 创建输出目录
OUTPUT_DIR = 'output/hs_300'
HIGHEST_PEAK_DIR = f'{OUTPUT_DIR}/highest_peak'
LATEST_PEAK_DIR = f'{OUTPUT_DIR}/latest_peak'
IMAGES_DIR_HIGHEST = f'{HIGHEST_PEAK_DIR}/images'
IMAGES_DIR_LATEST = f'{LATEST_PEAK_DIR}/images'
DATA_DIR_HIGHEST = f'{HIGHEST_PEAK_DIR}/data'
DATA_DIR_LATEST = f'{LATEST_PEAK_DIR}/data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HIGHEST_PEAK_DIR, exist_ok=True)
os.makedirs(LATEST_PEAK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR_HIGHEST, exist_ok=True)
os.makedirs(IMAGES_DIR_LATEST, exist_ok=True)
os.makedirs(DATA_DIR_HIGHEST, exist_ok=True)
os.makedirs(DATA_DIR_LATEST, exist_ok=True)

# 记录运行日志
LOG_FILE = f'{OUTPUT_DIR}/prediction_log.txt'

def log_message(message):
    """记录日志消息"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {message}\n')
    print(message)

def save_prediction_result(stock_code, prediction, strategy_name, stock_name=''):
    """保存预测结果到JSON文件"""
    result = {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'strategy': strategy_name,
        'peak_date': prediction.peak_date,
        'peak_price': prediction.peak_price,
        'buy_date': prediction.buy_date,
        'buy_price': prediction.buy_price,
        'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 根据策略选择保存目录
    data_dir = DATA_DIR_HIGHEST if strategy_name == Strategy.HIGHEST_PEAK else DATA_DIR_LATEST
    
    # 保存为JSON文件
    filename = f'{data_dir}/{stock_code}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filename

def save_prediction_image(stock_code, predictor, prediction, strategy_name, stock_name=''):
    """保存预测图片"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形但不立即显示
    fig = plt.figure(figsize=(14, 7))
    
    # 生成可视化但不显示
    predictor.visualize(prediction, show=False)
    
    # 根据策略类型，调整标题加入股票代码
    strategy_text = "最高点" if strategy_name == Strategy.HIGHEST_PEAK else "最近高点"
    title = f'{stock_code} {stock_name} - 股票预测分析 (策略: {strategy_text})'
    plt.title(title)
    
    # 根据策略选择保存目录
    images_dir = IMAGES_DIR_HIGHEST if strategy_name == Strategy.HIGHEST_PEAK else IMAGES_DIR_LATEST
    
    # 保存图像
    filename = f'{images_dir}/{stock_code}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，释放内存
    
    return filename

def predict_stock(stock_code):
    """对单支股票进行预测"""
    try:
        # 创建股票数据获取器
        fetcher = StockInfoFetcher(stock_code)
        
        # 获取股票数据（从2008年到现在）
        start_date = '20080101'
        end_date = '20231231'
        df = fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
        
        # 如果数据为空，记录错误并跳过
        if df.empty or len(df) < 52:  # 至少需要一年的数据
            log_message(f"错误: {stock_code} 数据不足，无法进行预测")
            return False
            
        # 获取股票名称（如果数据中有的话）
        stock_name = ''
        if 'name' in df.columns and not df['name'].empty:
            stock_name = df.iloc[0]['name']
        
        # 创建预测器
        predictor = StockPredictor(df)
        
        # 使用最高点策略预测
        try:
            highest_prediction = predictor.predict(strategy=Strategy.HIGHEST_PEAK)
            # 保存预测结果
            save_prediction_result(stock_code, highest_prediction, Strategy.HIGHEST_PEAK, stock_name)
            # 保存预测图片
            save_prediction_image(stock_code, predictor, highest_prediction, Strategy.HIGHEST_PEAK, stock_name)
        except Exception as e:
            log_message(f"警告: {stock_code} 最高点策略预测失败: {str(e)}")
        
        # 使用最近高点策略预测
        try:
            latest_prediction = predictor.predict(strategy=Strategy.LATEST_PEAK)
            # 保存预测结果
            save_prediction_result(stock_code, latest_prediction, Strategy.LATEST_PEAK, stock_name)
            # 保存预测图片
            save_prediction_image(stock_code, predictor, latest_prediction, Strategy.LATEST_PEAK, stock_name)
        except Exception as e:
            log_message(f"警告: {stock_code} 最近高点策略预测失败: {str(e)}")
        
        return True
    
    except Exception as e:
        log_message(f"错误: {stock_code} 预测过程出错: {str(e)}")
        traceback.print_exc()
        return False

def batch_predict():
    """批量处理沪深300股票预测"""
    log_message(f"开始批量预测沪深300股票，共 {len(hs300_list)} 支股票")
    
    # 创建汇总结果表
    summary = {
        Strategy.HIGHEST_PEAK: [],
        Strategy.LATEST_PEAK: []
    }
    
    # 使用tqdm显示进度条
    for stock_code in tqdm(hs300_list, desc="预测进度"):
        log_message(f"开始预测: {stock_code}")
        success = predict_stock(stock_code)
        if success:
            log_message(f"完成预测: {stock_code}")
        else:
            log_message(f"跳过预测: {stock_code}")
    
    # 读取所有预测结果并汇总
    strategy_dirs = {
        Strategy.HIGHEST_PEAK: DATA_DIR_HIGHEST,
        Strategy.LATEST_PEAK: DATA_DIR_LATEST
    }
    
    for strategy, data_dir in strategy_dirs.items():
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary[strategy].append(data)
    
    # 保存汇总结果
    for strategy, results in summary.items():
        if results:
            # 转换为DataFrame并保存为CSV
            df = pd.DataFrame(results)
            output_dir = HIGHEST_PEAK_DIR if strategy == Strategy.HIGHEST_PEAK else LATEST_PEAK_DIR
            df.to_csv(f'{output_dir}/summary.csv', index=False, encoding='utf-8')
            # 保存为Excel
            df.to_excel(f'{output_dir}/summary.xlsx', index=False)
    
    log_message(f"批量预测完成，结果保存在 {OUTPUT_DIR} 目录")

if __name__ == "__main__":
    batch_predict()