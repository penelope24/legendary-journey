"""
预测逻辑：
1. 首先设定分析区间：2012年至今。这并不代表输入的dataframe需要从2012年开始，而是指分析的区间。例如输入的数据从2008年开始，则股票的原始价格曲线从2008年开始，分析的区间从2012年开始。
2. 在分析区间内，找到所有的显著高点。
3. 这一步需要实现两种不同的策略：
    - 第一种策略：找到所有显著高点中的最高点，返回预测(time, price)
    - 第二种策略：找到所有显著高点中距今最近的一个，返回预测(time, price)
    其中time为最高点出现的时间 + 3.5年，代表最高点出现后3.5年可以开始考虑买入。price位最高点价格 * 0.3，代表建议的买入价格。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from matplotlib.dates import YearLocator, DateFormatter
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 修改导入语句
from src.peak import SignificantPeakAnalyzer, SignificantPeak
from src.data import StockInfoFetcher, Period

class Strategy:
    """预测策略枚举"""
    HIGHEST_PEAK = "highest_peak"  # 使用所有显著高点中的最高点
    LATEST_PEAK = "latest_peak"    # 使用所有显著高点中距今最近的一个


class PredictionResult:
    """预测结果类"""
    def __init__(self, 
                 peak_date: str, 
                 peak_price: float, 
                 buy_date: str, 
                 buy_price: float,
                 strategy: str):
        self.peak_date = peak_date        # 高点日期 YYYYMMDD
        self.peak_price = peak_price      # 高点价格
        self.buy_date = buy_date          # 建议买入日期 YYYYMMDD
        self.buy_price = buy_price        # 建议买入价格
        self.strategy = strategy          # 使用的策略
    
    def __str__(self) -> str:
        strategy_name = "最高点" if self.strategy == Strategy.HIGHEST_PEAK else "最近高点"
        return (f"预测结果 (策略: {strategy_name}):\n"
                f"  高点: {self.peak_date}, 价格: ¥{self.peak_price:.2f}\n"
                f"  建议买入: {self.buy_date}, 价格: ¥{self.buy_price:.2f}")


class StockPredictor:
    """股票预测类"""
    
    # 预测参数
    WAIT_YEARS = 3.5        # 高点后等待的年数
    PRICE_RATIO = 0.3       # 买入价格与高点价格的比例
    START_YEAR = 2012       # 分析开始年份
    
    def __init__(self, df: pd.DataFrame, price_col: str = 'close'):
        """
        初始化股票预测器
        
        Args:
            df: 包含价格数据的DataFrame
            price_col: 价格列名，默认为'close'
        """
        self.df = df.copy()
        self.price_col = price_col
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
            self.df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 筛选分析区间（从START_YEAR年开始）
        start_date = datetime(self.START_YEAR, 1, 1)
        self.analysis_df = self.df[self.df['trade_date'] >= start_date].reset_index(drop=True)
        
        # 创建峰值分析器
        self.peak_analyzer = SignificantPeakAnalyzer(self.analysis_df, price_col=price_col)
    
    def predict(self, 
               strategy: str = Strategy.HIGHEST_PEAK, 
               threshold_percentage: float = 0.85,
               threshold: float = 0.06,
               reference_date: Optional[str] = None) -> PredictionResult:
        """
        执行预测
        
        Args:
            strategy: 预测策略，可选值：HIGHEST_PEAK（使用最高点）, LATEST_PEAK（使用最近的高点）
            threshold_percentage: 标准线占历史最高点价格的百分比，默认为0.85（85%）
            threshold: ZigZag分析的阈值，默认为0.06（6%）
            reference_date: 预测参考日期（格式：YYYYMMDD），如果提供，则只使用该日期之前的数据进行预测
            
        Returns:
            PredictionResult: 预测结果对象
        """
        # 如果提供了参考日期，则筛选数据
        if reference_date:
            # 将字符串转换为datetime对象
            ref_date = datetime.strptime(reference_date, "%Y%m%d")
            
            # 筛选参考日期之前的数据
            filtered_df = self.analysis_df[self.analysis_df['trade_date'] <= ref_date].reset_index(drop=True)
            
            if filtered_df.empty:
                raise ValueError(f"参考日期 {reference_date} 之前没有可用数据")
                
            # 创建临时峰值分析器
            temp_analyzer = SignificantPeakAnalyzer(filtered_df, price_col=self.price_col)
            
            # 执行峰值分析
            result = temp_analyzer.analyze(
                threshold_percentage=threshold_percentage, 
                threshold=threshold
            )
        else:
            # 使用原始数据的峰值分析器
            result = self.peak_analyzer.analyze(
                threshold_percentage=threshold_percentage, 
                threshold=threshold
            )
        
        # 获取显著高点列表
        significant_peaks = result.get('significant_peaks', [])
        
        if not significant_peaks:
            raise ValueError("未找到显著高点，无法进行预测")
        
        # 根据策略选择高点
        if strategy == Strategy.HIGHEST_PEAK:
            # 第一种策略：使用最高点
            # 由于significant_peaks已经按价格降序排序，第一个点就是最高点
            selected_peak = significant_peaks[0]
        elif strategy == Strategy.LATEST_PEAK:
            # 第二种策略：使用最近的高点
            # 按日期排序并选择最新的高点
            sorted_by_date = sorted(significant_peaks, 
                                    key=lambda p: datetime.strptime(p.time, "%Y%m%d"), 
                                    reverse=True)
            selected_peak = sorted_by_date[0]
        else:
            raise ValueError(f"不支持的策略: {strategy}")
        
        # 计算建议买入日期（高点日期 + 3.5年）
        peak_date = datetime.strptime(selected_peak.time, "%Y%m%d")
        # 转换3.5年为3年和6个月
        years = int(self.WAIT_YEARS)
        months = int((self.WAIT_YEARS - years) * 12)
        buy_date = peak_date + relativedelta(years=years, months=months)
        
        # 计算建议买入价格（高点价格 * 0.3）
        buy_price = selected_peak.price * self.PRICE_RATIO
        
        # 创建预测结果
        prediction_result = PredictionResult(
            peak_date=selected_peak.time,
            peak_price=selected_peak.price,
            buy_date=buy_date.strftime("%Y%m%d"),
            buy_price=buy_price,
            strategy=strategy
        )
        
        # 保存分析结果以便后续可视化使用
        self.last_analysis_result = result
        self.last_selected_peak = selected_peak
        self.reference_date = reference_date
        
        return prediction_result
    
    def evaluate_prediction(self, prediction_result: PredictionResult) -> Dict[str, Any]:
        """
        评估预测结果与实际走势的差距
        
        Args:
            prediction_result: 预测结果对象
            
        Returns:
            Dict: 包含评估指标的字典
        """
        # 转换预测的买入日期为datetime
        buy_date = datetime.strptime(prediction_result.buy_date, "%Y%m%d")
        
        # 获取实际价格走势
        actual_prices = self.df[self.df['trade_date'] >= buy_date]
        
        if actual_prices.empty:
            return {
                "status": "future",
                "message": "预测买入日期尚未到来，无法评估"
            }
        
        # 获取实际数据中最低价格及其对应日期
        min_idx = actual_prices[self.price_col].idxmin()
        actual_min_price = actual_prices.loc[min_idx, self.price_col]
        actual_min_date = actual_prices.loc[min_idx, 'trade_date']
        
        # 判断实际最低价格是否低于预测买入价格
        target_reached = actual_min_price <= prediction_result.buy_price
        
        # 计算预测价格与实际最低价格的差距百分比
        price_gap_pct = ((actual_min_price / prediction_result.buy_price) - 1) * 100
        
        # 计算实际买入日期与预测买入日期的差距（天数）
        if target_reached:
            # 找到第一个低于等于预测价格的日期
            target_dates = actual_prices[actual_prices[self.price_col] <= prediction_result.buy_price]
            if not target_dates.empty:
                actual_buy_date = target_dates.iloc[0]['trade_date']
                date_gap_days = (actual_buy_date - buy_date).days
            else:
                date_gap_days = None
                actual_buy_date = None
        else:
            date_gap_days = None
            actual_buy_date = None
            
        # 返回评估结果
        return {
            "status": "completed" if actual_prices.shape[0] > 0 else "future",
            "target_reached": target_reached,
            "predicted_buy_date": buy_date,
            "predicted_buy_price": prediction_result.buy_price,
            "actual_min_date": actual_min_date,
            "actual_min_price": actual_min_price,
            "price_gap_percentage": price_gap_pct,
            "actual_buy_date": actual_buy_date,
            "date_gap_days": date_gap_days
        }
    
    def visualize(self, prediction_result: PredictionResult, show: bool = True, include_future: bool = True):
        """
        可视化预测结果
        
        Args:
            prediction_result: 预测结果对象
            show: 是否显示图表，默认为True。在批量处理时可设为False只保存不显示
            include_future: 是否包含参考日期之后的数据，默认为True
        """
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        plt.figure(figsize=(14, 7))
        
        # 确定数据范围
        if hasattr(self, 'reference_date') and self.reference_date:
            ref_date = datetime.strptime(self.reference_date, "%Y%m%d")
            if include_future:
                # 使用全部数据，但标记参考日期
                plot_df = self.df
                plot_analysis_df = self.analysis_df
            else:
                # 只使用参考日期之前的数据
                plot_df = self.df[self.df['trade_date'] <= ref_date]
                plot_analysis_df = self.analysis_df[self.analysis_df['trade_date'] <= ref_date]
        else:
            # 没有参考日期，使用全部数据
            plot_df = self.df
            plot_analysis_df = self.analysis_df
        
        # 绘制完整价格序列
        dates = plot_df['trade_date']
        prices = plot_df[self.price_col]
        plt.plot(dates, prices, 'gray', alpha=0.6, label='价格')
        
        # 绘制分析区间
        analysis_dates = plot_analysis_df['trade_date']
        analysis_prices = plot_analysis_df[self.price_col]
        plt.plot(analysis_dates, analysis_prices, 'blue', alpha=0.8, label=f'分析区间 ({self.START_YEAR}-至今)')
        
        # 获取所有显著高点和标准线信息
        if hasattr(self, 'last_analysis_result'):
            # 绘制标准线
            threshold_line = self.last_analysis_result.get('threshold_line')
            historical_high = self.last_analysis_result.get('historical_high')
            threshold_percentage = self.last_analysis_result.get('threshold_percentage')
            
            if threshold_line and historical_high:
                # 绘制标准线
                plt.axhline(y=threshold_line, color='purple', linestyle='-.',
                           label=f'阈值线 ({threshold_percentage*100:.0f}%): ¥{threshold_line:.2f}')
                
                # 绘制历史最高点
                highest_date = datetime.strptime(historical_high['time'], "%Y%m%d")
                plt.scatter([highest_date], [historical_high['price']], 
                           c='purple', s=150, marker='*',
                           label=f"历史最高点: ¥{historical_high['price']:.2f}")
            
            # 绘制所有显著高点
            all_peaks = self.last_analysis_result.get('significant_peaks', [])
            
            # 先添加一个非选中的高点到图例
            if len(all_peaks) > 1:
                plt.scatter([], [], c='darkorange', s=100, marker='^', 
                           edgecolors='black', alpha=0.9,
                           label=f'其他显著高点 ({len(all_peaks)-1}个)')
            
            for peak in all_peaks:
                peak_date = datetime.strptime(peak.time, "%Y%m%d")
                
                # 判断是否为选中的高点
                is_selected = (peak.time == prediction_result.peak_date and 
                              peak.price == prediction_result.peak_price)
                
                # 设置标记颜色和大小
                color = 'red' if is_selected else 'darkorange'
                size = 120 if is_selected else 100
                marker = '^'
                alpha = 1.0 if is_selected else 0.9
                zorder = 10 if is_selected else 9
                
                # 所有高点都绘制，但只有选中的高点加入图例
                if is_selected:
                    plt.scatter([peak_date], [peak.price], c=color, s=size, marker=marker,
                               alpha=alpha, zorder=zorder, edgecolors='black',
                               label=f'选中的高点: {peak.time} (¥{peak.price:.2f})')
                else:
                    plt.scatter([peak_date], [peak.price], c=color, s=size, marker=marker,
                               alpha=alpha, zorder=zorder, edgecolors='black')
        
        # 转换日期
        peak_date = datetime.strptime(prediction_result.peak_date, "%Y%m%d")
        buy_date = datetime.strptime(prediction_result.buy_date, "%Y%m%d")
        
        # 高亮显示高点周围两年区间（前后各一年）
        peak_highlight_start = peak_date - relativedelta(years=1)
        peak_highlight_end = peak_date + relativedelta(years=1)
        plt.axvspan(peak_highlight_start, peak_highlight_end, 
                   alpha=0.2, color='yellow', 
                   label=f'高点区域 (±1年)')
        
        # 绘制等待期的垂直线，标记3.5年后可以考虑买入的时间点
        plt.axvline(x=buy_date, color='green', linestyle='--', linewidth=1.5,
                   label=f'最早买入时间: {prediction_result.buy_date}')
        
        # 绘制建议买入价格的水平线
        plt.axhline(y=prediction_result.buy_price, color='green', linestyle='-', linewidth=1.5,
                   label=f'目标买入价格: ¥{prediction_result.buy_price:.2f}')
        
        # 在买入时间点添加特殊标记和注释
        plt.scatter([buy_date], [prediction_result.buy_price], 
                   c='green', s=120, marker='o', 
                   label=f'目标买入点')
        
        # 添加买入点注释
        plt.annotate(f'此日期后可考虑买入\n目标价格: ¥{prediction_result.buy_price:.2f} (峰值的{self.PRICE_RATIO*100:.0f}%)', 
                    xy=(buy_date, prediction_result.buy_price),
                    xytext=(30, 30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
        
        # 如果有参考日期，添加参考日期线
        if hasattr(self, 'reference_date') and self.reference_date:
            ref_date = datetime.strptime(self.reference_date, "%Y%m%d")
            plt.axvline(x=ref_date, color='red', linestyle='--', linewidth=1.5,
                       label=f'参考日期: {self.reference_date}')
            
            # 添加评估信息
            evaluation = self.evaluate_prediction(prediction_result)
            
            if evaluation['status'] == 'completed':
                # 如果预测日期已经过去，添加实际最低价格点
                plt.scatter([evaluation['actual_min_date']], [evaluation['actual_min_price']], 
                           c='blue', s=120, marker='o', 
                           label=f'实际最低点: ¥{evaluation["actual_min_price"]:.2f}')
                
                # 添加实际买入点（如果价格达到了预测价格）
                if evaluation['target_reached'] and evaluation['actual_buy_date']:
                    plt.scatter([evaluation['actual_buy_date']], [prediction_result.buy_price], 
                               c='cyan', s=120, marker='o', 
                               label=f'实际买入点: {evaluation["actual_buy_date"].strftime("%Y%m%d")}')
                    
                    # 添加评估信息注释
                    plt.annotate(
                        f'预测结果评估:\n'
                        f'预测价格: ¥{prediction_result.buy_price:.2f}\n'
                        f'实际最低价: ¥{evaluation["actual_min_price"]:.2f} ({evaluation["price_gap_percentage"]:.1f}%)\n'
                        f'实际达到目标: {"是" if evaluation["target_reached"] else "否"}\n'
                        f'实际买入时间与预测差距: {evaluation["date_gap_days"] if evaluation["date_gap_days"] is not None else "未达到"}天',
                        xy=(buy_date, prediction_result.buy_price * 0.9),
                        xytext=(-180, -80),
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
                else:
                    # 添加未达到目标的评估信息
                    plt.annotate(
                        f'预测结果评估:\n'
                        f'预测价格: ¥{prediction_result.buy_price:.2f}\n'
                        f'实际最低价: ¥{evaluation["actual_min_price"]:.2f} ({evaluation["price_gap_percentage"]:.1f}%)\n'
                        f'实际达到目标: {"是" if evaluation["target_reached"] else "否"}',
                        xy=(buy_date, prediction_result.buy_price * 0.9),
                        xytext=(-180, -60),
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        
        # 设置图表
        strategy_name = "最高点" if prediction_result.strategy == Strategy.HIGHEST_PEAK else "最近高点"
        title = f'股票预测分析 (策略: {strategy_name})'
        if hasattr(self, 'reference_date') and self.reference_date:
            title += f' - 参考日期: {self.reference_date}'
        plt.title(title)
        plt.xlabel('日期')
        plt.ylabel('价格 (元)')
        plt.grid(True, alpha=0.3)
        
        # 设置时间轴，显示每年刻度
        ax = plt.gca()
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        
        plt.legend(loc='upper left')
        
        # 显示图表
        plt.tight_layout()
        if show:
            plt.show()
        # 如果不显示，则在调用方负责关闭图形以释放内存


if __name__ == "__main__":
    # 创建茅台股票数据获取器
    stock_code = '600519.SH'  # 茅台股票代码
    fetcher = StockInfoFetcher(stock_code)
    
    # 获取茅台股票数据（从2008年到现在）
    start_date = '20080101'
    end_date = '20231231'
    df = fetcher.get_kline_data(Period.WEEKLY, start_date, end_date)
    
    # 创建预测器
    predictor = StockPredictor(df)
    
    # 使用最高点策略预测
    highest_prediction = predictor.predict(strategy=Strategy.HIGHEST_PEAK)
    print(highest_prediction)
    
    # 可视化最高点策略结果
    predictor.visualize(highest_prediction)
    
    # 使用最近高点策略预测
    latest_prediction = predictor.predict(strategy=Strategy.LATEST_PEAK)
    print(latest_prediction)
    
    # 可视化最近高点策略结果
    predictor.visualize(latest_prediction)
    
    # ========= 回测示例 =========
    # 选取一个历史时间点进行回测（例如2021年10月1日）
    reference_date = '20181001'
    
    print(f"\n回测示例 - 参考日期: {reference_date}")
    
    # 使用最高点策略进行回测预测
    historical_prediction = predictor.predict(
        strategy=Strategy.HIGHEST_PEAK, 
        reference_date=reference_date
    )
    print(historical_prediction)
    
    # 评估预测结果
    evaluation = predictor.evaluate_prediction(historical_prediction)
    
    # 打印评估结果
    if evaluation['status'] == 'completed':
        print(f"预测评估结果:")
        print(f"  预测买入日期: {historical_prediction.buy_date}, 价格: ¥{historical_prediction.buy_price:.2f}")
        print(f"  实际最低价格: ¥{evaluation['actual_min_price']:.2f}, 日期: {evaluation['actual_min_date'].strftime('%Y%m%d')}")
        print(f"  价格差距百分比: {evaluation['price_gap_percentage']:.2f}%")
        print(f"  是否达到目标价格: {'是' if evaluation['target_reached'] else '否'}")
        
        if evaluation['target_reached'] and evaluation['actual_buy_date']:
            print(f"  实际可买入日期: {evaluation['actual_buy_date'].strftime('%Y%m%d')}")
            print(f"  日期差距: {evaluation['date_gap_days']}天")
    else:
        print(f"  {evaluation['message']}")
    
    # 可视化回测结果（包含实际走势）
    predictor.visualize(historical_prediction, include_future=True)
    
    # 仅可视化回测当时可见的数据（不包含未来数据）
    # predictor.visualize(historical_prediction, include_future=False)