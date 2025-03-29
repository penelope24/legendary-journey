"""
测试股票筛选算法，特别关注2015年6-7月的高点识别问题
"""

import pandas as pd
from datetime import datetime
from data import StockInfoFetcher, Period
from filter import StockFilter

def test_filter_logic():
    """测试股票筛选算法中的高点识别逻辑"""
    # 创建股票数据获取器
    stock_code = '000166.SZ'
    fetcher = StockInfoFetcher(stock_code)
    
    # 获取2014-2017年的数据
    start_date = '20140101'
    end_date = '20170101'
    df = fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
    
    # 打印2015年6-7月的数据
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    jun_jul_data = df[(df['trade_date'] >= '2015-06-01') & (df['trade_date'] <= '2015-07-31')]
    print("2015年6-7月数据:")
    print(jun_jul_data[['trade_date', 'close']].sort_values('trade_date'))
    
    # 创建股票筛选器
    filter = StockFilter()
    
    # 分析股票
    result = filter.analyze_stock(df)
    
    # 打印分析结果
    stage1_results = result['stage1_results']
    
    print(f"\n分析完成，共{len(stage1_results)}轮迭代")
    
    for i, stage1 in enumerate(stage1_results):
        print(f"\n--- 第{i+1}轮分析 ---")
        print(f"历史最高点: {stage1.historical_peak[0]} - ¥{stage1.historical_peak[1]:.2f}")
        print(f"预测低点: ¥{stage1.predicted_low:.2f}")
        print(f"时间窗口结束: {stage1.time_window_end.strftime('%Y-%m-%d')}")
        
        if stage1.actual_low:
            print(f"实际低点: {stage1.actual_low[0]} - ¥{stage1.actual_low[1]:.2f}")
            print(f"是否提前出现: {'是' if stage1.is_early else '否'}")
        else:
            print("未找到实际低点")
        
        print(f"分析状态: {stage1.status}")

if __name__ == "__main__":
    test_filter_logic() 