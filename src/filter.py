"""
股票筛选算法实现

基于历史高点和低点检测的股票筛选算法，包含两个阶段：
1. 历史高点和低点检测
2. 低点的回调与新低点预测
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.peak import SignificantPeakAnalyzer, SignificantPeak
from src.zigzag import ZigZag
from src.data import StockInfoFetcher, Period

@dataclass
class Stage1Result:
    """Stage 1 分析结果
    
    包含第一阶段（历史高点和低点检测）的所有分析结果。
    
    Attributes:
        historical_peak (Tuple[str, float]): 历史最高点信息
            - str: 时间，格式为'YYYYMMDD'
            - float: 最高点价格
        original_peak_date (str): 原始历史最高点时间
            格式为'YYYYMMDD'，用于确定高点时间窗口位置
        predicted_low (float): 预测低点价格
            计算方法：历史最高点价格 * 0.3
        time_window_end (datetime): 预测时间窗口的结束日期
            计算方法：历史最高点时间 + 3.5年
        actual_low (Optional[Tuple[str, float]]): 实际低点信息
            - str: 时间，格式为'YYYYMMDD'
            - float: 实际低点价格
            如果未找到实际低点，则为None
        is_early (bool): 实际低点是否提前出现
            - True: 实际低点在时间窗口结束前出现
            - False: 实际低点在时间窗口结束后出现或未出现
        status (str): 分析状态
            - "提前出现": 实际低点在时间窗口结束前出现
            - "符合标准": 实际低点在时间窗口结束后出现
            - "不满足条件": 未找到实际低点
    """
    historical_peak: Tuple[str, float]  # (时间, 价格)
    original_peak_date: str  # 原始高点日期，用于窗口位置
    predicted_low: float
    time_window_end: datetime
    actual_low: Optional[Tuple[str, float]]  # (时间, 价格)
    is_early: bool
    status: str

@dataclass
class Stage2Result:
    """Stage 2 分析结果
    
    包含第二阶段（低点的回调与新低点预测）的所有分析结果。
    
    Attributes:
        has_rebound (bool): 是否出现价格回调
            - True: 价格涨幅超过70%
            - False: 价格涨幅未达到70%
        rebound_price (Optional[float]): 回调价格
            - 第一次超过70%涨幅时的价格
            - 如果未出现回调，则为None
        predicted_second_low (Optional[float]): 预测的第二低点价格
            - 计算方法：回调价格 * 0.3
            - 如果未出现回调，则为None
    """
    has_rebound: bool
    rebound_price: Optional[float]
    predicted_second_low: Optional[float]

class StockFilter:
    """股票筛选器"""
    
    def __init__(self):
        # Stage 1 参数
        self.analysis_start_date = '2012-01-01'
        self.price_threshold = 0.85  # 85%
        self.time_window_years = 1.0
        self.prediction_threshold = 0.3  # 30%
        self.prediction_window_years = 3.5
        
        # Stage 2 参数
        self.rebound_threshold = 0.7  # 70%
        
    def analyze_stock(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析单只股票
        
        Args:
            df: 股票数据DataFrame，必须包含'trade_date'和'close'列
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # Stage 1: 历史高点和低点检测（迭代版本）
        stage1_results = self._analyze_stage1(df)
        
        # 如果没有找到任何结果，返回空结果
        if not stage1_results:
            return {
                'stage1_results': [],
                'stage2_results': []
            }
        
        # 对每个Stage 1结果执行Stage 2分析
        stage2_results = []
        for stage1_result in stage1_results:
            # 如果Stage 1没有找到实际低点，跳过Stage 2
            if not stage1_result.actual_low:
                stage2_results.append(None)
                continue
            
            # Stage 2: 低点的回调与新低点预测
            stage2_result = self._analyze_stage2(df, stage1_result)
            stage2_results.append(stage2_result)
        
        return {
            'stage1_results': stage1_results,
            'stage2_results': stage2_results
        }
    
    def _analyze_stage1(self, df: pd.DataFrame) -> List[Stage1Result]:
        """
        Stage 1: 历史高点和低点检测
        
        修改后的方法将迭代地查找历史高点和预测低点，当一个预测低点出现后，
        继续寻找新的显著高点并重复分析过程。
        
        Returns:
            List[Stage1Result]: 所有迭代分析的结果列表，按时间顺序排序
        """
        # 创建结果列表
        results = []
        
        # 创建一个临时DataFrame副本用于分析
        temp_df = df.copy()
        
        # 确保trade_date列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(temp_df['trade_date']):
            temp_df['trade_date'] = pd.to_datetime(temp_df['trade_date'])
        
        # 获取当前日期作为分析截止日期
        current_date = pd.to_datetime(datetime.now())
        
        # 当前分析的起始日期
        analysis_start_date = pd.to_datetime(self.analysis_start_date)
        
        # 首先对整个时间范围进行一次显著峰值分析，避免在每次迭代中重复计算
        print("计算整个时间范围内的显著峰值...")
        full_analyzer = SignificantPeakAnalyzer(temp_df)
        full_peak_result = full_analyzer.analyze(
            threshold_percentage=self.price_threshold,
            threshold=0.06  # ZigZag阈值
        )
        
        # 获取所有显著高点
        all_significant_peaks = full_peak_result.get('significant_peaks', [])
        
        # 记录已经分析过的高点，避免重复分析
        analyzed_peak_times = set()
        
        # 开始迭代分析
        iteration = 1
        continue_analysis = True
        
        while continue_analysis:
            print(f"\n开始第{iteration}次迭代分析")
            print(f"分析起始日期: {analysis_start_date}")
            
            # 筛选出当前分析范围内的数据
            current_df = temp_df[(temp_df['trade_date'] >= analysis_start_date) & 
                                (temp_df['trade_date'] <= current_date)]
            
            # 如果没有足够的数据，就结束迭代
            if len(current_df) < 10:  # 假设需要至少10个数据点
                print(f"数据点不足，结束迭代")
                break
            
            # 从所有显著高点中筛选出当前分析区间内的高点
            current_significant_peaks = []
            for peak in all_significant_peaks:
                peak_date = pd.to_datetime(peak.time)
                if peak_date >= analysis_start_date and peak_date <= current_date:
                    # 确保这个高点之前没有被分析过
                    if peak.time not in analyzed_peak_times:
                        current_significant_peaks.append(peak)
            
            # 如果当前区间内没有显著高点，结束迭代
            if not current_significant_peaks:
                print(f"当前区间内没有找到显著高点，结束迭代")
                break
            
            # 按时间排序当前区间内的高点
            current_significant_peaks.sort(key=lambda p: pd.to_datetime(p.time))
            
            # 找出当前区间内的最高点作为原始历史最高点
            original_peak = max(current_significant_peaks, key=lambda p: p.price)
            original_peak_time = original_peak.time
            original_peak_price = original_peak.price
            original_peak_date = pd.to_datetime(original_peak_time)
            
            # 标记这个高点已经被分析
            analyzed_peak_times.add(original_peak_time)
            
            # 定义在历史最高点前后一年的时间窗口
            window_start = original_peak_date - timedelta(days=365)
            window_end = original_peak_date + timedelta(days=365)
            
            print(f"原始历史最高点: {original_peak_time} - ¥{original_peak_price:.2f}")
            print(f"高点时间窗口: {window_start.strftime('%Y-%m-%d')} 至 {window_end.strftime('%Y-%m-%d')}")
            
            # 在时间窗口内查找更合适的时间点
            window_peaks = []
            for peak in current_significant_peaks:
                peak_date = pd.to_datetime(peak.time)
                # 原来的代码排除了原始历史最高点本身，现在需要修改为包含原始高点
                if window_start <= peak_date <= window_end:
                    window_peaks.append(peak)
                    if peak.time != original_peak_time:  # 只打印窗口内的其他显著高点
                        print(f"  窗口内显著高点: {peak.time} - ¥{peak.price:.2f}")
            
            # 选择最终的历史最高点时间
            peak_time = original_peak_time
            if window_peaks:
                # 计算每个高点与今天的时间差，选择距离今天最近的高点
                current_date_for_comparison = current_date  # 使用当前日期作为参考
                closest_peak = max(window_peaks, 
                                  key=lambda p: pd.to_datetime(p.time))  # 选择日期最靠近今天的高点
                peak_time = closest_peak.time
                print(f"选择时间上最近的高点作为时间参考: {peak_time}")
            else:
                print(f"窗口内无显著高点，使用原始历史最高点时间: {peak_time}")
            
            # 保持原始历史最高点的价格
            peak_price = original_peak_price
            
            # 计算预测低点和时间窗口
            predicted_low = peak_price * self.prediction_threshold
            peak_date = pd.to_datetime(peak_time)
            time_window_end = peak_date + timedelta(days=self.prediction_window_years * 365)
            
            print(f"最终选定的历史最高点: 时间={peak_time}，价格=¥{peak_price:.2f}")
            print(f"预测低点: ¥{predicted_low:.2f}")
            print(f"预测时间窗口结束: {time_window_end.strftime('%Y-%m-%d')}")
            
            # 在历史最高点之后寻找实际低点
            actual_low = self._find_actual_low(current_df, peak_date, predicted_low)
            
            if actual_low is None:
                # 如果没有找到实际低点，创建结果并结束迭代
                result = Stage1Result(
                    historical_peak=(peak_time, peak_price),
                    original_peak_date=original_peak_time,
                    predicted_low=predicted_low,
                    time_window_end=time_window_end,
                    actual_low=None,
                    is_early=False,
                    status="不满足条件"
                )
                results.append(result)
                break
            
            # 判断是否提前出现
            actual_low_date = pd.to_datetime(actual_low[0])
            is_early = actual_low_date < time_window_end
            
            # 创建结果
            result = Stage1Result(
                historical_peak=(peak_time, peak_price),
                original_peak_date=original_peak_time,
                predicted_low=predicted_low,
                time_window_end=time_window_end,
                actual_low=actual_low,
                is_early=is_early,
                status="提前出现" if is_early else "符合标准"
            )
            
            # 将结果添加到列表
            results.append(result)
            
            # 检查是否继续迭代
            # 如果已经到达或接近当前日期，就结束迭代
            if actual_low_date >= current_date - timedelta(days=30):
                print(f"实际低点接近当前日期，结束迭代")
                break
            
            # 更新分析起始日期为实际低点日期
            analysis_start_date = actual_low_date
            
            # 递增迭代计数
            iteration += 1
        
        return results
    
    def _analyze_stage2(self, df: pd.DataFrame, stage1_result: Stage1Result) -> Stage2Result:
        """
        Stage 2: 低点的回调与新低点预测
        """
        if not stage1_result.actual_low:
            return None
        
        # 获取第一低点信息
        first_low_time, first_low_price = stage1_result.actual_low
        first_low_date = pd.to_datetime(first_low_time)
        
        # 确保trade_date列是datetime类型
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 在第一低点之后的数据中寻找回调
        mask = df['trade_date'] > first_low_date
        post_low_data = df[mask]
        
        if post_low_data.empty:
            return Stage2Result(
                has_rebound=False,
                rebound_price=None,
                predicted_second_low=None
            )
        
        # 计算相对于第一低点的涨幅
        price_increases = (post_low_data['close'] - first_low_price) / first_low_price
        
        # 检查是否有超过70%的回调
        has_rebound = (price_increases > self.rebound_threshold).any()
        
        if not has_rebound:
            return Stage2Result(
                has_rebound=False,
                rebound_price=None,
                predicted_second_low=None
            )
        
        # 获取回调价格（取第一次超过70%时的价格）
        rebound_idx = price_increases[price_increases > self.rebound_threshold].index[0]
        rebound_price = post_low_data.loc[rebound_idx, 'close']
        
        # 预测第二低点
        predicted_second_low = rebound_price * self.prediction_threshold
        
        return Stage2Result(
            has_rebound=True,
            rebound_price=rebound_price,
            predicted_second_low=predicted_second_low
        )
    
    def _find_actual_low(self, df: pd.DataFrame, 
                        start_date: datetime, 
                        threshold: float) -> Optional[Tuple[str, float]]:
        """
        在指定日期start_date之后，寻找第一次价格达到预测低点threshold的点
        
        Args:
            df: 股票数据DataFrame
            start_date: 开始日期
            threshold: 预测低点价格
            
        Returns:
            Optional[Tuple[str, float]]: (时间, 预测低点价格)，如果未找到则返回None
            - 时间是第一次价格达到预测低点的日期
            - 价格统一使用预测低点价格，而不是实际交易价格
        """
        # 确保trade_date列是datetime类型
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 筛选开始日期之后的数据
        mask = df['trade_date'] > start_date
        post_data = df[mask].copy()
        
        if post_data.empty:
            print("未找到开始日期之后的数据")
            return None
        
        # 确保价格是浮点型
        post_data = post_data.copy()
        post_data['close'] = post_data['close'].astype(float)
        
        print(f"\n调试信息 - 查找达到预测低点的时间点")
        print(f"预测低点价格: {threshold}")
        
        # 按时间排序
        post_data = post_data.sort_values('trade_date')
        
        # 计算每个点与预测低点的差距
        post_data['diff'] = post_data['close'] - threshold
        
        # 保存原始日期格式，以便返回
        post_data['date_str'] = post_data['trade_date'].dt.strftime('%Y%m%d')
        
        # 策略：找第一个满足以下条件之一的点
        # 1. 价格刚好等于预测低点（diff = 0）
        # 2. 价格刚好小于预测低点（diff < 0），且前一个点的价格大于预测低点（前一个点的diff > 0）
        
        # 先找到差距变号的点（从大于预测低点变为小于预测低点）
        potential_points = []
        
        for i in range(1, len(post_data)):
            prev_diff = post_data.iloc[i-1]['diff']
            curr_diff = post_data.iloc[i]['diff']
            
            # 如果前一个点高于预测低点，当前点低于或等于预测低点
            if prev_diff > 0 and curr_diff <= 0:
                row = post_data.iloc[i]
                date_str = row['date_str']
                # 使用预测低点价格而不是实际价格
                potential_points.append((date_str, threshold))
                print(f"找到交叉点: {row['trade_date']} - 实际价格: {row['close']}, 使用预测价格: {threshold}")
                # 找到第一个符合条件的点后就跳出循环
                break
        
        if potential_points:
            # 返回第一个交叉点，使用预测低点价格
            date_str, price = potential_points[0]
            print(f"选择第一个交叉点: {date_str} - 使用预测价格: {price}")
            return (date_str, price)
        
        # 如果没有找到交叉点，查找是否有点刚好等于预测低点
        equal_points = post_data[post_data['diff'].abs() < 0.001]  # 使用非常小的容差找等于点
        if not equal_points.empty:
            row = equal_points.iloc[0]
            date_str = row['date_str']
            # 使用预测低点价格
            print(f"找到刚好等于预测低点的点: {row['trade_date']} - 实际价格: {row['close']}, 使用预测价格: {threshold}")
            return (date_str, threshold)
        
        # 如果没有等于预测低点的点，尝试找最接近的
        # 我们要找与预测低点相差不超过5%的点
        tolerance = threshold * 0.05  # 容差范围
        
        # 标记差距小于5%的点
        close_points = post_data[post_data['diff'].abs() <= tolerance]
        
        if not close_points.empty:
            row = close_points.iloc[0]  # 按时间顺序取第一个
            date_str = row['date_str']
            # 使用预测低点价格
            print(f"找到接近预测低点的点(容差{tolerance}): {row['trade_date']} - 实际价格: {row['close']}, 使用预测价格: {threshold}")
            return (date_str, threshold)
        
        print("未找到符合条件的点")
        return None

    def visualize(self, df: pd.DataFrame, result: Dict[str, Any], stock_code: str, show: bool = True):
        """
        可视化分析结果
        
        Args:
            df: 股票数据DataFrame
            result: 分析结果字典
            stock_code: 股票代码
            show: 是否显示图表，默认为True
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
        from dateutil.relativedelta import relativedelta
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        import numpy as np
        import matplotlib.ticker as mticker
        
        # 设置风格和字体
        plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['axes.edgecolor'] = '#cccccc'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#cccccc'
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        
        # 创建一个更大的图表
        fig = plt.figure(figsize=(16, 10))
        
        # 创建主价格图，不再分割为两部分
        ax_price = fig.add_subplot(111)
        
        # 确保日期列是datetime类型用于绘图
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 获取所有Stage 1结果和Stage 2结果
        stage1_results = result['stage1_results']
        stage2_results = result['stage2_results']
        
        # 统一参考peak.py.bak配色方案
        price_color = 'gray'
        zigzag_color = 'blue'
        threshold_color = 'orange'
        historical_high_color = 'purple'
        peak_color = 'red'
        bottom_color = 'green'
        rebound_color = '#9932CC'  # 深紫色，用于回调点
        
        # 统一标记样式
        peak_marker = '*'  # 使用星号标记高点，与peak.py.bak一致
        low_marker = 'v'   # 使用下三角标记低点
        rebound_marker = '^'  # 使用上三角标记回调点
        
        # 绘制原始价格
        dates = df['trade_date']
        ax_price.plot(dates, df['close'], color=price_color, alpha=0.5, linewidth=1.2, label='周线价格')
        
        # 设置更好的价格格式化
        def price_formatter(x, pos):
            return f"¥{x:.0f}" if x >= 100 else f"¥{x:.2f}"
        ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(price_formatter))
        
        # 在分析阶段指示图中创建区域
        phase_colors = ['#f0f8ff', '#e2f0f9', '#e6f5eb', '#f5e8f7']  # 不同阶段的颜色，更柔和
        phase_labels = []  # 存储每个阶段的标签，用于图例
        
        # 计算Y轴的最小最大值，用于后续绘图
        min_price = df['close'].min() * 0.9  # 留出10%的空间
        max_price = df['close'].max() * 1.1
        
        # 高亮显示分析区间（2012年至今）- 遵循设计文档
        analysis_start = pd.to_datetime('2012-01-01')  # 强制使用固定日期而不是self.analysis_start_date
        analysis_end = dates.max()
        # 确保只显示2012年至今的分析区间，使用更明显的颜色
        ax_price.axvspan(analysis_start, analysis_end, color='#cce5ff', alpha=0.4, zorder=0, label='分析区间(2012至今)')
        
        # 绘制整个图表的背景交替带，突显年份区域
        years = pd.DatetimeIndex(dates).year.unique()
        for i, year in enumerate(years):
            year_start = pd.Timestamp(f"{year}-01-01")
            year_end = pd.Timestamp(f"{year}-12-31") if year < years[-1] else dates.max()
            if i % 2 == 0:  # 偶数年添加浅色背景
                ax_price.axvspan(year_start, year_end, color='#f8f9fa', alpha=0.3, zorder=0)
                
        # 遍历所有Stage 1结果进行绘制
        for i, stage1 in enumerate(stage1_results):
            # 使用统一颜色
            phase_color = phase_colors[i % len(phase_colors)]
            
            # 获取阶段起止时间
            peak_date = pd.to_datetime(stage1.historical_peak[0], format='%Y%m%d')
            
            # 为该分析阶段创建背景区域
            if i < len(stage1_results) - 1:
                next_peak_date = pd.to_datetime(stage1_results[i+1].historical_peak[0], format='%Y%m%d')
                # 在主图添加淡色背景表示分析周期
                ax_price.axvspan(peak_date, next_peak_date, 
                               color=phase_color, alpha=0.05, zorder=0)
                # 添加文本标记周期
                ax_price.text(peak_date + (next_peak_date - peak_date)/2, max_price*0.93, f"周期 {i+1}", 
                            ha='center', va='top', fontsize=9, alpha=0.7,
                            bbox=dict(boxstyle="round,pad=0.1", fc=phase_color, ec='none', alpha=0.3))
            else:
                # 最后一个阶段延伸到图表结束
                ax_price.axvspan(peak_date, dates.max(), 
                               color=phase_color, alpha=0.05, zorder=0)
                # 添加文本标记周期
                ax_price.text(peak_date + (dates.max() - peak_date)/2, max_price*0.93, f"周期 {i+1}", 
                            ha='center', va='top', fontsize=9, alpha=0.7,
                            bbox=dict(boxstyle="round,pad=0.1", fc=phase_color, ec='none', alpha=0.3))
            
            # 绘制历史高点的85%阈值线，贯穿整个图表
            threshold_value = stage1.historical_peak[1] * self.price_threshold
            ax_price.axhline(y=threshold_value, 
                         color=threshold_color, linestyle='--', alpha=0.7, linewidth=1.5,
                         label=f'阈值线 (85%): ¥{threshold_value:.2f}' if i == 0 else "")
            
            # 绘制历史高点区域（前后1年的时间窗口）
            # 获取分析时使用的确切窗口时间（使用原始高点日期确定窗口位置）
            original_peak_date = pd.to_datetime(stage1.original_peak_date, format='%Y%m%d')
            window_start = original_peak_date - timedelta(days=365)
            window_end = original_peak_date + timedelta(days=365)

            # 添加窗口日期信息标签
            ax_price.text(window_start, max_price*0.85, 
                        f'窗口起始: {window_start.strftime("%Y/%m/%d")}', 
                        ha='left', va='top', fontsize=8, color=historical_high_color,
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=historical_high_color, alpha=0.5))
            ax_price.text(window_end, max_price*0.80, 
                        f'窗口结束: {window_end.strftime("%Y/%m/%d")}', 
                        ha='right', va='top', fontsize=8, color=historical_high_color,
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=historical_high_color, alpha=0.5))

            # 使用确切的窗口时间绘制高点区域
            ax_price.axvspan(window_start, window_end, 
                           alpha=0.1, color=historical_high_color, edgecolor=None, zorder=1)
            
            # 绘制历史最高点
            ax_price.scatter([peak_date], [stage1.historical_peak[1]], 
                           c=historical_high_color, s=150, marker=peak_marker, edgecolors='white', linewidth=1.5, zorder=10,
                           label=f'历史高点: ¥{stage1.historical_peak[1]:.2f}' if i == 0 else "")
            
            # 添加高点编号标签，简化注释
            ax_price.annotate(f"#{i+1}", 
                            xy=(peak_date, stage1.historical_peak[1]),
                            xytext=(10, 10),  # 改变偏移量避免重叠
                            textcoords='offset points',
                            ha='left',  # 更改为左对齐
                            fontsize=10,
                            color=historical_high_color,
                            weight='bold')
            
            # 绘制预测时间窗口
            prediction_window_end = peak_date + relativedelta(years=int(self.prediction_window_years), 
                                                            months=int((self.prediction_window_years % 1) * 12))
            ax_price.axvspan(peak_date, prediction_window_end, 
                           alpha=0.05, color=historical_high_color, edgecolor=historical_high_color, linestyle='--',
                           zorder=1, label=f'预测窗口' if i == 0 else "")
            
            # 添加预测窗口结束标记
            prediction_window_text = f'预测窗口结束: {prediction_window_end.strftime("%Y/%m")}'
            ax_price.text(prediction_window_end, max_price*0.89, 
                        prediction_window_text,
                        ha='right', va='top', fontsize=8, color=historical_high_color,
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=historical_high_color, alpha=0.5),
                        rotation=0)
            
            # 绘制预测低点线（使用虚线），贯穿整个图表
            ax_price.axhline(y=stage1.predicted_low, 
                         color=bottom_color, linestyle='--', alpha=0.7, linewidth=1.5,
                         label=f'预测低点: ¥{stage1.predicted_low:.2f}' if i == 0 else "")
            
            # 为预测低点添加简洁标签，调整位置
            ax_price.text(peak_date + relativedelta(months=6), 
                        stage1.predicted_low * 0.97,  # 轻微调整位置避免重叠
                        f'¥{stage1.predicted_low:.2f}',
                        ha='left', va='top', fontsize=9, color=bottom_color,
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=bottom_color, alpha=0.85))
            
            # 绘制时间窗口结束线
            ax_price.axvline(x=stage1.time_window_end, color=historical_high_color, linestyle=':', linewidth=1.2,
                           alpha=0.6, label=f'可以开始考虑' if i == 0 else "")
            
            # 添加简化的窗口结束注释
            window_end_text = f'{stage1.time_window_end.strftime("%Y/%m")}'
            ax_price.text(stage1.time_window_end, min_price*1.05, window_end_text,
                        rotation=90, ha='center', va='bottom', fontsize=8, color=historical_high_color)
            
            # 绘制实际低点（如果存在）
            if stage1.actual_low:
                actual_low_date = pd.to_datetime(stage1.actual_low[0], format='%Y%m%d')
                actual_low_price = stage1.actual_low[1]
                
                # 绘制实际低点
                ax_price.scatter([actual_low_date], [actual_low_price], 
                               c=bottom_color, s=150, marker=low_marker, edgecolors='white', linewidth=1.5, zorder=10,
                               label=f'实际低点' if i == 0 else "")
                
                # 添加简化的实际低点编号注释
                drop_pct = ((stage1.historical_peak[1] - actual_low_price) / stage1.historical_peak[1]) * 100
                
                # 计算标签位置，避免重叠
                xytext_y = -25 if i % 2 == 0 else -40
                
                # 添加低点标签
                ax_price.annotate(f"↓{drop_pct:.1f}%", 
                                xy=(actual_low_date, actual_low_price),
                                xytext=(0, xytext_y),
                                textcoords='offset points',
                                ha='center',
                                fontsize=10,
                                color=bottom_color,
                                weight='bold')
                
                # 绘制从高点到低点的连线
                ax_price.plot([peak_date, actual_low_date], 
                             [stage1.historical_peak[1], actual_low_price], 
                             color=peak_color, linestyle='--', alpha=0.6, zorder=2)
                
                # 在折线下方显示天数
                midpoint_date = peak_date + (actual_low_date - peak_date) / 2
                midpoint_price = (stage1.historical_peak[1] + actual_low_price) / 2 - (stage1.historical_peak[1] - actual_low_price) * 0.1
                ax_price.text(midpoint_date, midpoint_price, f"{(actual_low_date - peak_date).days}天", 
                             ha='center', va='top', color=peak_color, fontsize=8, 
                             bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=peak_color, alpha=0.7))
                
                # 绘制Stage 2结果（如果存在）
                if i < len(stage2_results) and stage2_results[i] and stage2_results[i].has_rebound:
                    stage2 = stage2_results[i]
                    
                    # 找到回调价格对应的日期点
                    mask = df['close'] >= stage2.rebound_price
                    if mask.any():
                        # 获取第一次超过回调价格的日期
                        rebound_mask = (df['trade_date'] > actual_low_date) & mask
                        if rebound_mask.any():
                            rebound_date = df.loc[rebound_mask, 'trade_date'].iloc[0]
                            
                            # 绘制回调点
                            ax_price.scatter([rebound_date], [stage2.rebound_price], 
                                           c=rebound_color, s=150, marker=rebound_marker, edgecolors='white', linewidth=1.5, zorder=10,
                                           label=f'回调点' if i == 0 else "")
                            
                            # 添加回调阈值标注
                            rebound_threshold_value = actual_low_price * (1 + self.rebound_threshold)
                            ax_price.plot([actual_low_date, rebound_date + relativedelta(months=3)], 
                                         [rebound_threshold_value, rebound_threshold_value], 
                                         color=rebound_color, linestyle=':', alpha=0.5, linewidth=1,
                                         label=f'回调阈值: +{self.rebound_threshold*100:.0f}%' if i == 0 else "")
                            
                            # 绘制预测第二低点线
                            ax_price.plot([rebound_date, rebound_date + relativedelta(years=3)], 
                                         [stage2.predicted_second_low, stage2.predicted_second_low], 
                                         color=bottom_color, linestyle='-.', alpha=0.6, linewidth=1.5,
                                         label=f'预测第二低点: ¥{stage2.predicted_second_low:.2f}' if i == 0 else "")
                            
                            # 为预测第二低点添加简洁标签
                            ax_price.text(rebound_date + relativedelta(months=6), 
                                        stage2.predicted_second_low*0.97, 
                                        f'¥{stage2.predicted_second_low:.2f}',
                                        ha='left', va='top', fontsize=8, color=bottom_color,
                                        bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=bottom_color, alpha=0.7))
                            
                            # 添加回调点注释
                            rebound_pct = ((stage2.rebound_price - actual_low_price) / actual_low_price) * 100
                            
                            # 计算标签位置，避免重叠
                            xytext_y = 20 if i % 2 == 0 else 35
                            
                            # 添加回调点标签
                            ax_price.annotate(f"↑{rebound_pct:.1f}%", 
                                            xy=(rebound_date, stage2.rebound_price),
                                            xytext=(0, xytext_y),
                                            textcoords='offset points',
                                            ha='center',
                                            fontsize=10,
                                            color=rebound_color,
                                            weight='bold')
                            
                            # 绘制从低点到回调点的连线
                            ax_price.plot([actual_low_date, rebound_date], 
                                         [actual_low_price, stage2.rebound_price], 
                                         color=rebound_color, linestyle='--', alpha=0.6, zorder=2)
                            
                            # 在折线旁显示天数
                            midpoint_date = actual_low_date + (rebound_date - actual_low_date) / 2
                            midpoint_price = (actual_low_price + stage2.rebound_price) / 2 - (stage2.rebound_price - actual_low_price) * 0.1
                            ax_price.text(midpoint_date, midpoint_price, f"{(rebound_date - actual_low_date).days}天", 
                                        ha='center', va='top', color=rebound_color, fontsize=8, 
                                        bbox=dict(boxstyle="round,pad=0.1", fc='white', ec=rebound_color, alpha=0.7))
        
        # 设置主价格图的标题和标签
        title = f'{stock_code} 股票迭代分析结果'
        if len(stage1_results) > 0:
            title += f' - 共{len(stage1_results)}轮分析 (2012-{pd.Timestamp.now().year})'
        ax_price.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax_price.set_ylabel('价格 (元)', fontsize=12, fontweight='bold')
        ax_price.yaxis.set_label_coords(-0.01, 0.5)  # 调整y轴标签位置
        
        # 设置X轴标签
        ax_price.set_xlabel('日期', fontsize=12, fontweight='bold')
        
        # 设置合适的Y轴范围，增加额外空间以便文字显示
        ax_price.set_ylim(min_price*0.9, max_price*1.1)
        
        # 设置更好的网格线
        ax_price.grid(True, which='major', axis='both', linestyle='-', alpha=0.3)
        ax_price.grid(True, which='minor', axis='both', linestyle=':', alpha=0.1)
        
        # 设置时间轴，大刻度为年，小刻度为季度
        ax_price.xaxis.set_major_locator(YearLocator())
        ax_price.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax_price.xaxis.set_minor_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
        
        # 添加图例，不再包含周期标签
        handles, labels = ax_price.get_legend_handles_labels()
        legend = ax_price.legend(handles, labels, 
                         loc='upper left', fontsize=9, 
                         framealpha=0.95, fancybox=True, shadow=True)
        legend.get_frame().set_edgecolor('#cccccc')
        
        # 直接调整子图位置，避免使用tight_layout
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
        
        # 添加参考日期线 - 当前日期
        current_date = pd.to_datetime(datetime.now())
        if current_date > dates.min() and current_date < dates.max():
            ax_price.axvline(current_date, color='#6c757d', linestyle='-', linewidth=1, alpha=0.7)
            ax_price.text(current_date, max_price*0.99, '当前', 
                         ha='right', va='top', color='#6c757d', fontsize=8, 
                         bbox=dict(boxstyle="round,pad=0.1", fc='white', ec='#6c757d', alpha=0.8))
        
        # 添加图表脚注
        plt.figtext(0.5, 0.02, 
                   f"分析日期: {datetime.now().strftime('%Y-%m-%d')} | 数据范围: {df['trade_date'].min().strftime('%Y-%m-%d')} - {df['trade_date'].max().strftime('%Y-%m-%d')}", 
                   ha="center", fontsize=9, style='italic')
        
        plt.figtext(0.99, 0.02, 
                   "基于历史高点和低点的迭代分析", 
                   ha="right", fontsize=9, style='italic')
        
        # 显示图表
        if show:
            # 不使用tight_layout()，因为已经手动调整了边距
            plt.show()
            
        return fig  # 返回图表对象，以便可能的进一步处理或保存

if __name__ == "__main__":
    # 示例：分析茅台股票数据
    from src.data import StockInfoFetcher, Period
    import os
    
    # 创建股票数据获取器
    stock_code = '000596.SZ'  # 茅台股票代码
    fetcher = StockInfoFetcher(stock_code)
    
    # 获取股票数据（从2010年到现在）
    start_date = '20100101'
    end_date = datetime.now().strftime('%Y%m%d')  # 动态获取当前日期
    df = fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
    
    # 创建股票筛选器
    filter = StockFilter()
    
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
    
    # 可视化分析结果
    fig = filter.visualize(df, result, stock_code)
    
    # 保存图表
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f'{stock_code}_analysis.png')
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n图表已保存到 {output_file}")
    