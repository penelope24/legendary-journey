"""
测试股票筛选算法
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.filter import StockFilter
import os

# 载入测试数据
df = pd.read_csv('test_stock.csv')
df['trade_date'] = pd.to_datetime(df['trade_date'])

# 打印测试数据的基本信息
print(f"数据范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
print(f"最高价: {df['close'].max():.2f}, 最低价: {df['close'].min():.2f}")

# 创建股票筛选器
filter = StockFilter()

# 修改开始分析日期为数据的开始日期
filter.analysis_start_date = df['trade_date'].min().strftime('%Y-%m-%d')

# 分析股票
result = filter.analyze_stock(df)

# 打印分析结果
stage1_results = result['stage1_results']
stage2_results = result['stage2_results']

print(f"\n=== 分析完成，共{len(stage1_results)}轮迭代 ===")

for i, (stage1, stage2) in enumerate(zip(stage1_results, stage2_results)):
    print(f"\n--- 第{i+1}轮分析 ---")
    
    # 打印Stage 1结果
    print(f"历史最高点: {stage1.historical_peak[0]} - ¥{stage1.historical_peak[1]:.2f}")
    print(f"预测低点: ¥{stage1.predicted_low:.2f}")
    print(f"时间窗口结束: {stage1.time_window_end.strftime('%Y-%m-%d')}")
    
    if stage1.actual_low:
        print(f"实际低点: {stage1.actual_low[0]} - ¥{stage1.actual_low[1]:.2f}")
        print(f"是否提前出现: {'是' if stage1.is_early else '否'}")
    else:
        print("未找到实际低点")
    
    print(f"分析状态: {stage1.status}")
    
    # 打印Stage 2结果
    if stage2:
        print(f"是否出现回调: {'是' if stage2.has_rebound else '否'}")
        
        if stage2.has_rebound:
            print(f"回调价格: ¥{stage2.rebound_price:.2f}")
            print(f"预测第二低点: ¥{stage2.predicted_second_low:.2f}")
    else:
        print("跳过Stage 2分析")

# 可视化分析结果并保存图表
fig = filter.visualize(df, result, "测试股票", show=True)

# 保存图表
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
fig.savefig(os.path.join(output_dir, 'test_stock_analysis.png'), dpi=200, bbox_inches='tight')
print(f"图表已保存到 {os.path.join(output_dir, 'test_stock_analysis.png')}") 