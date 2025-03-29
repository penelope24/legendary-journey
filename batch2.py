"""
批量预测
对于config/stocks.py中hs300_list列表内的每一个股票代码，进行src/filter中的预测，并保存结果。
保存路径: 保存路径为output/hs_300_filter
"""

import os
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt

# 导入自定义模块
from src.filter import StockFilter
from src.data import StockInfoFetcher, Period
from config.stocks import hs300_list

# 创建输出目录
OUTPUT_DIR = 'output/hs_300_filter'
IMAGES_DIR = f'{OUTPUT_DIR}/images'
DATA_DIR = f'{OUTPUT_DIR}/data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 记录运行日志
LOG_FILE = f'{OUTPUT_DIR}/filter_log.txt'

def log_message(message):
    """记录日志消息"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {message}\n')
    print(message)

def save_analysis_result(stock_code, result, stock_name=''):
    """保存分析结果到JSON文件"""
    # 将结果转换为可序列化的格式
    serializable_result = {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stage1_results': [],
        'stage2_results': []
    }
    
    # 处理Stage 1结果
    for stage1 in result['stage1_results']:
        stage1_dict = {
            'historical_peak': {
                'time': stage1.historical_peak[0],
                'price': stage1.historical_peak[1]
            },
            'predicted_low': stage1.predicted_low,
            'time_window_end': stage1.time_window_end.strftime('%Y-%m-%d'),
            'actual_low': {
                'time': stage1.actual_low[0],
                'price': stage1.actual_low[1]
            } if stage1.actual_low else None,
            'is_early': stage1.is_early,
            'status': stage1.status
        }
        serializable_result['stage1_results'].append(stage1_dict)
    
    # 处理Stage 2结果
    for stage2 in result['stage2_results']:
        stage2_dict = {
            'has_rebound': stage2.has_rebound if stage2 else None,
            'rebound_price': stage2.rebound_price if stage2 else None,
            'predicted_second_low': stage2.predicted_second_low if stage2 else None
        }
        serializable_result['stage2_results'].append(stage2_dict)
    
    # 保存为JSON文件
    filename = f'{DATA_DIR}/{stock_code}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    return filename

def save_analysis_image(stock_code, filter, df, result, stock_name=''):
    """保存分析图片"""
    # 生成可视化但不显示
    fig = filter.visualize(df, result, stock_code, show=False)
    
    # 调整标题加入股票代码和名称
    title = f'{stock_code} {stock_name} - 股票迭代分析结果'
    fig.axes[0].set_title(title)
    
    # 保存图像
    filename = f'{IMAGES_DIR}/{stock_code}.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，释放内存
    
    return filename

def analyze_stock(stock_code):
    """对单支股票进行分析"""
    try:
        # 创建股票数据获取器
        fetcher = StockInfoFetcher(stock_code)
        
        # 获取股票数据（从2012年到现在）
        start_date = '20120101'
        end_date = datetime.now().strftime('%Y%m%d')
        df = fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
        
        # 如果数据为空，记录错误并跳过
        if df.empty or len(df) < 52:  # 至少需要一年的数据
            log_message(f"错误: {stock_code} 数据不足，无法进行分析")
            return False
            
        # 获取股票名称（如果数据中有的话）
        stock_name = ''
        if 'name' in df.columns and not df['name'].empty:
            stock_name = df.iloc[0]['name']
        
        # 创建股票筛选器
        filter = StockFilter()
        
        # 分析股票
        result = filter.analyze_stock(df)
        
        # 保存分析结果
        save_analysis_result(stock_code, result, stock_name)
        
        # 保存分析图片
        save_analysis_image(stock_code, filter, df, result, stock_name)
        
        return True
    
    except Exception as e:
        log_message(f"错误: {stock_code} 分析过程出错: {str(e)}")
        traceback.print_exc()
        return False

def batch_analyze():
    """批量处理沪深300股票分析"""
    log_message(f"开始批量分析沪深300股票，共 {len(hs300_list)} 支股票")
    
    # 使用tqdm显示进度条
    for stock_code in tqdm(hs300_list, desc="分析进度"):
        log_message(f"开始分析: {stock_code}")
        success = analyze_stock(stock_code)
        if success:
            log_message(f"完成分析: {stock_code}")
        else:
            log_message(f"跳过分析: {stock_code}")
    
    # 读取所有分析结果并汇总
    summary = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                summary.append(data)
    
    # 保存汇总结果
    if summary:
        # 转换为DataFrame并保存为CSV
        df = pd.DataFrame(summary)
        df.to_csv(f'{OUTPUT_DIR}/summary.csv', index=False, encoding='utf-8')
        # 保存为Excel
        df.to_excel(f'{OUTPUT_DIR}/summary.xlsx', index=False)
    
    log_message(f"批量分析完成，结果保存在 {OUTPUT_DIR} 目录")

if __name__ == "__main__":
    batch_analyze()